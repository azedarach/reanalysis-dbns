"""
Provides routines for fitting stepwise Bayes regression model.
"""

# License: MIT

from __future__ import absolute_import, division

import warnings

import arviz as az
import numpy as np
import patsy
import scipy.linalg as sl
import scipy.special as sp

import reanalysis_dbns.utils as rdu

from sklearn.utils import check_random_state

from .sampler_helpers import write_stepwise_mc3_samples
from .stepwise_mc3_sampler import sample_stepwise_mc3


def _check_gamma_hyperparameters(a_tau, b_tau):
    """Check given hyperparameters of gamma distribution are valid."""

    if not rdu.is_scalar(a_tau) or a_tau <= 0.0:
        raise ValueError(
            'Hyperparameter a_tau must be a positive scalar '
            '(got a_tau=%r)' % a_tau)

    if not rdu.is_scalar(b_tau) or b_tau <= 0.0:
        raise ValueError(
            'Hyperparameter b_tau must be a positive scalar '
            '(got b_tau=%r)' % b_tau)

    return a_tau, b_tau


def _check_snr_parameter(nu_sq):
    """Check given SNR parameter is valid."""

    if not rdu.is_scalar(nu_sq) or nu_sq <= 0.0:
        raise ValueError(
            'SNR parameter nu_sq must be a positive scalar '
            '(got nu_sq=%r)' % nu_sq)

    return nu_sq


def _check_design_matrices(y, X):
    """Check design matrices have required attributes."""

    if not hasattr(y.design_info, 'terms') or y.design_info.terms is None:
        raise ValueError(
            'Outcomes matrix must have terms attribute')

    if not hasattr(X.design_info, 'terms') or X.design_info.terms is None:
        raise ValueError(
            'Design matrix must have terms attribute')

    return y, X


def _get_optional_terms(design_info, force_intercept=True):
    """Get list of optional terms."""

    optional_terms = []
    optional_term_names = []
    for i, t in enumerate(design_info.terms):
        if force_intercept and t == patsy.INTERCEPT:
            continue

        optional_terms.append(t)
        optional_term_names.append(design_info.term_names[i])

    return optional_terms, optional_term_names


def log_uniform_indicator_set_prior(k, max_nonzero=None):
    """Evaluate uniform prior on indicator set."""

    n_indicators = k.shape[0]

    if max_nonzero is None:
        max_nonzero = n_indicators
    else:
        if not rdu.is_integer(max_nonzero) or max_nonzero < 0:
            raise ValueError(
                'Maximum size of indicator set must be a '
                'non-negative integer')

    n_nonzero = np.sum(k > 0)

    if n_nonzero > max_nonzero:
        return -np.inf

    # Otherwise, assign uniform prior mass to all possible indicator sets
    # consistent with the maximum predictor set size constraint.
    n_indicator_sets = np.array([sp.comb(n_indicators, i)
                                 for i in range(max_nonzero + 1)])

    return -np.log(np.sum(n_indicator_sets))


def _bayes_regression_normal_gamma_covariance_matrices(X, nu_sq):
    """Evaluate posterior covariance matrices."""

    n_samples, n_features = X.shape

    c = np.eye(n_features) + nu_sq * np.dot(X.T, X)
    c_chol = sl.cholesky(c, lower=True)
    c_inv_chol = sl.solve_triangular(c_chol, np.eye(n_features),
                                     lower=True).T
    c_inv = np.dot(c_inv_chol, c_inv_chol.T)

    sigma_inv = np.eye(n_samples) - nu_sq * np.dot(X, np.dot(c_inv, X.T))

    return c_inv, sigma_inv


def bayes_regression_normal_gamma_log_marginal_likelihood(y, X, nu_sq,
                                                          a_tau, b_tau,
                                                          sigma_inv=None):
    """Evaluate log marginal likelihood for Bayes regression model."""

    n_samples = X.shape[0]

    if sigma_inv is None:
        _, sigma_inv = _bayes_regression_normal_gamma_covariance_matrices(
            X, nu_sq)

    # Construct covariance matrix.
    cov_term = 1.0 + 0.5 * b_tau * np.dot(y.T, np.dot(sigma_inv, y))

    log_marginal_likelihood = (
        sp.gammaln(0.5 * (n_samples + 2.0 * a_tau)) -
        sp.gammaln(a_tau) - 0.5 * n_samples * np.log(2 * np.pi) +
        0.5 * n_samples * np.log(b_tau) +
        0.5 * np.linalg.slogdet(sigma_inv)[1] -
        0.5 * (n_samples + 2.0 * a_tau) * np.log(cov_term))

    return log_marginal_likelihood


def _default_indicator_name(p):
    """Return name of indicator variable for inclusion of term."""
    return 'i_{}'.format(p)


def _add_named_indicator_variables_to_fit(fit, term_names):
    """Add individual named indicator variables.

    Parameters
    ----------
    fit : dict
        Dictionary containing the results of fitting the model.

    term:names : list
        List of term names for each term corresponding to an
        entry in the indicator vector k.

    Returns
    -------
    fit : dict
        Fit with named indicator variables added.
    """

    n_chains = len(fit['samples'])

    for i in range(n_chains):

        k = fit['samples'][i]['chains']['k']
        n_samples = k.shape[0]

        for t in term_names:
            ind = _default_indicator_name(t)
            fit['samples'][i]['chains'][ind] = np.empty(
                (n_samples,), dtype=int)

        for j in range(n_samples):
            for m, km in enumerate(k[j]):
                ind = _default_indicator_name(term_names[m])
                fit['samples'][i]['chains'][ind][j] = km

    return fit


class StepwiseBayesRegression():
    """Bayes regression model with conjugate priors and fixed SNR.

    Parameters
    ----------
    a_tau : float, default: 1.0
        Shape parameter of the gamma prior on the outcome precision.

    b_tau : float, default: 10.0
        Scale parameter of the gamma prior on the outcome precision.

    nu_sq : float, default: 1.0
        Signal-to-noise parameter used for normal prior on
        the regression coefficients.
    """
    def __init__(self, a_tau=1.0, b_tau=10.0, nu_sq=1.0,
                 force_intercept=True):

        self.a_tau, self.b_tau = _check_gamma_hyperparameters(a_tau, b_tau)
        self.nu_sq = _check_snr_parameter(nu_sq)

        self.allow_exchanges = True
        self.calculate_diagnostics = True
        self.force_intercept = force_intercept

    def _initialize_mc3_random(self, n_terms, n_chains=1, max_terms=None,
                               random_state=None):
        """Get initial state for MC3 sampler."""

        rng = check_random_state(random_state)

        initial_k = np.zeros((n_chains, n_terms), dtype=int)

        if max_terms is None:
            max_terms = n_terms

        invalid_max_terms = (not rdu.is_integer(max_terms) or
                             max_terms < 0 or max_terms > n_terms)
        if invalid_max_terms:
            raise ValueError(
                'Maximum number of terms must be an integer between '
                '0 and %d '
                '(got max_terms=%d)' % (n_terms, max_terms))

        # Draw initial set of terms uniformly from possible sets of
        # terms.
        n_possible_sets = np.sum([sp.comb(n_terms, k)
                                  for k in range(max_terms + 1)])
        weights = np.array([sp.comb(n_terms, k) / n_possible_sets
                            for k in range(max_terms + 1)])

        for i in range(n_chains):

            terms_set_size = rng.choice(max_terms + 1, p=weights)
            term_indices = rng.choice(n_terms, size=terms_set_size,
                                      replace=False)

            initial_k[i, term_indices] = 1

        return initial_k

    def _initialize_mc3(self, n_terms, method='random', **kwargs):
        """Draw initial values for indicator set."""

        if method == 'random':
            return self._initialize_mc3_random(n_terms, **kwargs)

        raise ValueError(
            "Unrecognized initialization method '%r'" % method)

    def _log_model_posterior(self, k, data=None, max_terms=None,
                             lhs_terms=None, optional_terms=None):
        """Evaluate log model posterior probability."""

        lp = log_uniform_indicator_set_prior(k, max_nonzero=max_terms)

        rhs_terms = []
        if self.force_intercept:
            rhs_terms.append(patsy.INTERCEPT)

        for m, km in enumerate(k):
            if km == 1:
                rhs_terms.append(optional_terms[m])

        model_desc = patsy.ModelDesc(lhs_terms, rhs_terms)

        y, X = patsy.dmatrices(model_desc, data=data)

        lp += bayes_regression_normal_gamma_log_marginal_likelihood(
            y, X, a_tau=self.a_tau, b_tau=self.b_tau,
            nu_sq=self.nu_sq)

        return lp

    def sample_structures_mc3(self, formula_like, data=None, n_chains=4,
                              n_iter=1000, warmup=None, thin=1, verbose=False,
                              n_jobs=-1, max_terms=None, sample_file=None,
                              restart_file=None, init='random',
                              intercept_name='Intercept',
                              allow_exchanges=True, random_state=None):
        """Sample models using MC3 algorithm."""

        rng = check_random_state(random_state)

        y, X = patsy.dmatrices(formula_like, data=data)
        y, X = _check_design_matrices(y, X)

        lhs_terms = y.design_info.terms
        optional_terms, optional_term_names = _get_optional_terms(
            X.design_info, force_intercept=self.force_intercept)

        n_terms = len(optional_terms)
        n_chains = rdu.check_number_of_chains(n_chains)

        if restart_file is None:
            initial_k = self._initialize_mc3(n_terms, n_chains=n_chains,
                                             max_terms=max_terms, method=init,
                                             random_state=rng)

        else:
            restart_fit = az.from_netcdf(restart_file)

            if n_chains != restart_fit.posterior.sizes['chain']:
                warnings.warn(
                    'Number of saved chains does not match number '
                    'of requested chains '
                    '(got n_chains=%d but saved n_chains=%d)' %
                    (n_chains, restart_fit.posterior.sizes['chain']),
                    UserWarning)

                n_chains = restart_fit.posterior.sizes['chain']

            initial_k = restart_fit.posterior['k'].isel(draw=-1).data.copy()

        def logp(k, data):
            return self._log_model_posterior(
                k, data=data, max_terms=max_terms,
                lhs_terms=lhs_terms,
                optional_terms=optional_terms)

        fit = sample_stepwise_mc3(
            initial_k, logp, data=data,
            n_chains=n_chains, n_iter=n_iter, thin=thin,
            warmup=warmup, verbose=verbose, n_jobs=n_jobs,
            max_nonzero=max_terms, allow_exchanges=self.allow_exchanges,
            random_state=rng)

#        if n_chains > 1 and self.calculate_diagnostics:
#            _check_stepwise_mc3_convergence_diagnostics(
#                fit, split=True)

        # Generate named indicator variables for convenience.
        fit = _add_named_indicator_variables_to_fit(
            fit, term_names=optional_term_names)

        if sample_file is not None:
            write_stepwise_mc3_samples(
                fit, sample_file, data=data)

        return fit
