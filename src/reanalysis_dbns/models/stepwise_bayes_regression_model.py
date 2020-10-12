"""
Provides routines for fitting stepwise Bayes regression model.
"""

# License: MIT

from __future__ import absolute_import, division

import collections
from copy import deepcopy
import itertools
import warnings

import arviz as az
import numpy as np
import pandas as pd
import patsy
import scipy.linalg as sl
import scipy.special as sp

import reanalysis_dbns.utils as rdu

from sklearn.utils import check_random_state

from .sampler_diagnostics import estimate_stationary_distribution
from .sampler_helpers import (get_sampled_parameter_dimensions,
                              write_stepwise_mc3_samples)
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


def _percentile_string(q):
    """Format percentile."""
    if 0 <= q <= 1.0:
        q = 100.0 * q
    return '{:3.1f}%'.format(q)


def _get_sample_summary_statistics(var_values, probs=None):
    """Get sample summary statistics."""

    n_vars = len(var_values)

    if probs is None:
        probs = [0.025, 0.25, 0.5, 0.75, 0.975]

    col_names = ['par_name', 'mean', 'sd']
    for q in probs:
        col_names += [_percentile_string(q)]

    sample_stats = {'par_name': []}
    for n in col_names[1:]:
        sample_stats[n] = np.empty((n_vars,), dtype=float)

    for i, p in enumerate(var_values):
        sample_stats['par_name'].append(p)
        sample_stats['mean'][i] = np.mean(var_values[p])
        sample_stats['sd'][i] = np.std(var_values[p])

        for q in probs:
            sample_stats[_percentile_string(q)][i] = np.quantile(
                var_values[p], q)

    data_vars = collections.OrderedDict(
        {c: sample_stats[c] for c in col_names})

    return pd.DataFrame(data_vars)


def _add_named_indicator_variables_to_fit(fit, term_names):
    """Add individual named indicator variables.

    Parameters
    ----------
    fit : dict
        Dictionary containing the results of fitting the model.

    term_names : list
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


class StepwiseBayesRegressionMC3Results():
    """Wrapper class for results of MC3 sampling.

    Parameters
    ----------
    fit : dict
        Fit result returned from calling sample_stepwise_mc3.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    Attributes
    ----------
    samples : list
        Samples produced by MC3 sampling.

    n_chains : int
        Number of chains used for sampling.

    n_iter : int
        Number of samples per chain.

    warmup : int
        Number of warmup samples.

    thin : int
        Stride used for thinning samples.

    n_save : list
        Number of saved samples in each chain.

    warmup2 : int
        Number of warmup samples after thinning in each chain.

    permutation : list
        List of permutations for individual chains.

    random_seeds : list
        Random seeds used for sampling.
    """
    def __init__(self, fit, random_state=None):

        self.samples = deepcopy(fit['samples'])
        self.n_chains = fit['n_chains']
        self.n_iter = fit['n_iter']
        self.warmup = fit['warmup']
        self.thin = fit['thin']
        self.n_save = deepcopy(fit['n_save'])
        self.warmup2 = deepcopy(fit['warmup2'])
        self.permutation = deepcopy(fit['permutation'])
        self.random_seeds = deepcopy(fit['random_seeds'])
        self.max_nonzero = fit['max_nonzero']
        self.random_state = random_state

        # Options for calculating diagnostics.
        self.sparse = True
        self.min_epsilon = 1e-10
        self.tolerance = 1e-6
        self.fit_kwargs = None

        # Name of the indicator variable in sampling output.
        self._ind_var = 'k'

    def _unique_models(self, inc_warmup=False):
        """Get list of unique models."""

        if self.samples[0]['chains'][self._ind_var].ndim == 1:
            n_indicators = 1
        else:
            n_indicators = self.samples[0]['chains'][self._ind_var].shape[1]

        n_kept = []
        for i in range(self.n_chains):
            if inc_warmup:
                n_kept.append(self.n_save[i])
            else:
                n_kept.append(self.n_save[i] - self.warmup2[i])

        n_total = sum(n_kept)

        k = np.empty((n_total, n_indicators))
        pos = 0
        for i in range(self.n_chains):
            chain_k = self.samples[i]['chains'][self._ind_var]
            if chain_k.ndim == 1:
                chain_k = chain_k[:, np.newaxis]
            k[pos:pos + n_kept[i]] = chain_k[-n_kept[i]:]
            pos += n_kept[i]

        return np.unique(k, axis=0)

    def _count_possible_models(self):
        """Count the number of possible models."""

        if self.samples[0]['chains'][self._ind_var].ndim == 1:
            n_indicators = 1
        else:
            n_indicators = self.samples[0]['chains'][self._ind_var].shape[1]

        if self.max_nonzero is None:
            max_nonzero = n_indicators

        return sum([sp.comb(n_indicators, i)
                    for i in range(max_nonzero + 1)])

    def _get_model_lookup(self, inc_warmup=False):
        """Get lookup table for model labels."""

        unique_k = self._unique_models(inc_warmup=inc_warmup)

        return {tuple(ki): '{:d}'.format(i)
                for i, ki in enumerate(unique_k)}

    def _get_model_indicators(self, inc_warmup=False,
                              only_sampled_models=True):
        """Get indicator variables for individual models."""

        model_lookup = self._get_model_lookup(inc_warmup=inc_warmup)

        if inc_warmup:
            n_samples = self.n_save[0]
        else:
            n_samples = self.n_save[0] - self.warmup2[0]

        z = np.empty((self.n_chains, n_samples), dtype=str)

        for i in range(self.n_chains):

            if inc_warmup:
                chain_k = self.samples[i]['chains'][self._ind_var]
            else:
                n_kept = self.n_save[i] - self.warmup2[i]
                chain_k = self.samples[i]['chains'][self._ind_var][-n_kept:]

            for t in range(n_samples):
                z[i, t] = model_lookup[tuple(chain_k[t])]

        model_indicators = [model_lookup[i] for i in model_lookup]
        if not only_sampled_models:
            n_observed_models = len(model_indicators)
            n_possible_models = self._count_possible_models()
            model_indicators += ['{:d}'.format(i)
                                 for i in range(n_observed_models,
                                                n_possible_models)]

        return z, model_indicators

    def diagnostics(self, n_samples=100, epsilon=None, inc_warmup=False,
                    only_sampled_models=True):

        z, model_indicators = self._get_model_indicators(
            inc_warmup=inc_warmup, only_sampled_models=only_sampled_models)

        return estimate_stationary_distribution(
            z, model_indicators=model_indicators, sparse=self.sparse,
            epsilon=epsilon, n_samples=n_samples,
            min_epsilon=self.min_epsilon, tolerance=self.tolerance,
            fit_kwargs=self.fit_kwargs, random_state=self.random_state)

    def posterior_mode(self, inc_warmup=False):
        """Get posterior mode for sampled structures."""

        post_mode_lp = -np.inf
        post_mode = None
        for i in range(self.n_chains):

            chain = self.samples[i]['chains']

            if inc_warmup:
                chain_log_post = chain['lp__']
                chain_mode_ind = np.argmax(chain_log_post)
            else:
                n_kept = self.n_save[i] - self.warmup2[i]
                chain_log_post = chain['lp__'][-n_kept:]
                chain_mode_ind = n_kept + np.argmax(chain_log_post)

            chain_mode = {p: chain[p][chain_mode_ind] for p in chain}
            chain_mode_lp = chain['lp__'][chain_mode_ind]

            if chain_mode_lp > post_mode_lp:
                post_mode_lp = chain_mode_lp
                post_mode = chain_mode

        return post_mode, post_mode_lp

    def _get_nonindicator_parameters(self):
        """Get names of all parameters other than indicators vector."""
        par_names = []
        for i in range(self.n_chains):

            chain = self.samples[i]['chains']
            chain_pars = [p for p in chain if p != self._ind_var]

            for p in chain_pars:
                if p not in par_names:
                    par_names.append(p)

        return par_names

    def indicators_summary(self, probs=None, inc_warmup=False):
        """Return summary of sampled indicator variables."""

        # Determine variables to show in output and number of
        # samples for each.
        par_names = self._get_nonindicator_parameters()

        n_kept = []
        for i in range(self.n_chains):
            if inc_warmup:
                n_kept.append(self.n_save[i])
            else:
                n_kept.append(self.n_save[i] - self.warmup2[i])

        n_samples = sum(n_kept)

        par_samples = {}
        for p in par_names:
            par_dims = get_sampled_parameter_dimensions(
                self.samples[0]['chains'], p)

            par_data = np.empty((n_samples,) + par_dims)

            pos = 0
            for i in range(self.n_chains):
                par_data[pos:pos + n_kept[i]] = \
                    self.samples[i]['chains'][p][-n_kept[i]:]
                pos += n_kept[i]

            if par_dims:
                par_data = np.moveaxis(par_data, 0, -1)
                # Flatten samples.
                indices = itertools.product(*[range(d) for d in par_dims])
                for idx in indices:
                    idx_str = ''.join(['{:d}'.format(d) for d in idx])
                    flat_par_name = '{}_{}'.format(p, idx_str)
                    par_samples[flat_par_name] = par_data[idx]
            else:
                par_samples[p] = par_data

        # Evaluate summary statistics for sample.
        return _get_sample_summary_statistics(par_samples, probs=probs)

    def _get_model_indicator_arrays(self, inc_warmup=False):
        """Get values of indicator variables in each model."""

        if self.samples[0]['chains'][self._ind_var].ndim == 1:
            n_indicators = 1
        else:
            n_indicators = self.samples[0]['chains'][self._ind_var].shape[1]

        model_lookup = self._get_model_lookup(inc_warmup=inc_warmup)
        indicator_lookup = {model_lookup[k]: k for k in model_lookup}

        model_names = np.array(list(indicator_lookup.keys()))
        n_observed_models = len(model_names)

        indicator_vals = {}
        for i in range(n_indicators):
            ki_vals = np.zeros((n_observed_models,), dtype=int)
            for j, m in enumerate(model_names):
                ki_vals[j] = indicator_lookup[m][i]
            indicator_vals['k{:d}'.format(i)] = ki_vals

        return indicator_vals

    def _calculate_model_posterior_probabilities(self, inc_warmup=False):
        """Calculate posterior probability for each model."""

        model_lookup = self._get_model_lookup(inc_warmup=inc_warmup)
        indicator_lookup = {model_lookup[k]: k for k in model_lookup}

        model_names = np.array(list(indicator_lookup.keys()))
        n_observed_models = len(model_names)

        model_counts = np.zeros((n_observed_models,), dtype=int)

        n_samples = 0
        for j in range(self.n_chains):

            if inc_warmup:
                chain_k = self.samples[j]['chains'][self._ind_var]
            else:
                n_kept = self.n_save[j] - self.warmup2[j]
                chain_k = self.samples[j]['chains'][self._ind_var][-n_kept:]

            chain_k = [tuple(k) for k in chain_k]
            n_samples += len(chain_k)

            for i, m in enumerate(model_names):
                km = indicator_lookup[m]
                model_counts[i] += sum([k == km for k in chain_k])

        return model_counts / n_samples

    def _calculate_posterior_probability_ci(self, inc_warmup=False,
                                            alpha=0.05, **kwargs):
        """Calculate credible intervals for posterior probabilities."""

        diagnostics = self.diagnostics(inc_warmup=inc_warmup, **kwargs)

        ci_lower = np.quantile(diagnostics['pi'], 0.05 * alpha, axis=0)
        ci_upper = np.quantile(diagnostics['pi'], 1 - 0.05 * alpha, axis=0)

        return ci_lower, ci_upper

    def model_summary(self, inc_warmup=False, show_indicators=True,
                      sort_by_probs=True, include_ci=False, alpha=0.05,
                      **diagnostics_kwargs):
        """Return summary of sampled models."""

        model_lookup = self._get_model_lookup(inc_warmup=inc_warmup)
        indicator_lookup = {model_lookup[k]: k for k in model_lookup}

        model_names = np.array(list(indicator_lookup.keys()))

        if show_indicators:
            indicator_vals = self._get_model_indicator_arrays(
                inc_warmup=inc_warmup)
        else:
            indicator_vals = None

        model_probs = self._calculate_model_posterior_probabilities(
            inc_warmup=inc_warmup)

        if include_ci:
            model_probs_lower, model_probs_upper = \
                self._calculate_posterior_probability_ci(
                    inc_warmup=inc_warmup, alpha=alpha,
                    **diagnostics_kwargs)
        else:
            model_probs_lower = None
            model_probs_upper = None

        if sort_by_probs:
            model_order = np.argsort(-model_probs)

            model_names = model_names[model_order]
            model_probs = model_probs[model_order]

            if show_indicators:
                for ki in indicator_vals:
                    indicator_vals[ki] = indicator_vals[ki][model_order]

            if include_ci:
                model_probs_lower = model_probs_lower[model_order]
                model_probs_upper = model_probs_upper[model_order]

        data_vars = collections.OrderedDict({'model': model_names})
        if show_indicators:
            for ki in indicator_vals:
                data_vars[ki] = indicator_vals[ki]

        data_vars['p'] = model_probs

        if include_ci:
            data_vars[_percentile_string(0.5 * alpha)] = model_probs_lower
            data_vars[_percentile_string(1 - 0.5 * alpha)] = \
                model_probs_upper

        return pd.DataFrame(data_vars)


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

        # Generate named indicator variables for convenience.
        fit = _add_named_indicator_variables_to_fit(
            fit, term_names=optional_term_names)

        if sample_file is not None:
            write_stepwise_mc3_samples(
                fit, sample_file, data=data)

        return StepwiseBayesRegressionMC3Results(fit)
