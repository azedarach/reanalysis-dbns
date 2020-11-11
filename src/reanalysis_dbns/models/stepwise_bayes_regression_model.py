"""
Provides routines for fitting stepwise Bayes regression model.
"""

# License: MIT

from __future__ import absolute_import, division

import collections
import warnings

import arviz as az
import numpy as np
import pandas as pd
import patsy
import scipy.linalg as sl
import scipy.special as sp
import scipy.stats as ss
import xarray as xr

import reanalysis_dbns.utils as rdu

from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from .sampler_diagnostics import (estimate_convergence_rate,
                                  estimate_stationary_distribution,
                                  rjmcmc_batch_chisq_convergence,
                                  rjmcmc_batch_kstest_convergence,
                                  rjmcmc_chisq_convergence,
                                  rjmcmc_kstest_convergence)
from .sampler_helpers import (check_max_nonzero_indicators,
                              convert_samples_dict_to_inference_data)
from .stepwise_mc3_sampler import (initialize_stepwise_mc3,
                                   sample_stepwise_mc3)


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


def _get_optional_terms(term_list, term_names=None, force_intercept=True):
    """Get list of optional terms."""

    optional_terms = []
    if term_names is not None:
        optional_term_names = []
    else:
        optional_term_names = None

    for i, t in enumerate(term_list):
        if force_intercept and t == patsy.INTERCEPT:
            continue

        optional_terms.append(t)
        if term_names is not None:
            optional_term_names.append(term_names[i])

    return optional_terms, optional_term_names


def _get_outcome_and_optional_terms(formula_like, data=None,
                                    force_intercept=True):
    """Get terms from model description."""

    try:
        # Try to infer the possible terms from the general case
        # where pre-formed design matrices or a formula string
        # may be given.
        y, X = patsy.dmatrices(formula_like, data=data)
        y, X = _check_design_matrices(y, X)

        lhs_terms = y.design_info.terms
        outcome_names = y.design_info.column_names
        optional_terms, optional_term_names = _get_optional_terms(
            X.design_info.terms, term_names=X.design_info.term_names,
            force_intercept=force_intercept)

    except patsy.PatsyError:
        # If this fails, fall back on trying to construct the
        # model from a formula string.
        full_model = patsy.ModelDesc.from_formula(formula_like)

        lhs_terms = full_model.lhs_termlist
        outcome_names = [t.name() for t in full_model.lhs_termlist]
        optional_terms, _ = _get_optional_terms(
            full_model.rhs_termlist, force_intercept=force_intercept)
        optional_term_names = [t.name() for t in optional_terms]

    return lhs_terms, outcome_names, optional_terms, optional_term_names


def _get_model_description(k, lhs_terms=None, optional_terms=None,
                           force_intercept=True):
    """Get model description from indicators vector."""

    rhs_terms = []
    if force_intercept:
        rhs_terms.append(patsy.INTERCEPT)

    for m, km in enumerate(k):
        if km == 1:
            rhs_terms.append(optional_terms[m])

    model_desc = patsy.ModelDesc(lhs_terms, rhs_terms)

    return model_desc


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
            ind = rdu.get_default_indicator_name(t)
            fit['samples'][i]['chains'][ind] = np.empty(
                (n_samples,), dtype=int)

        for j in range(n_samples):
            for m, km in enumerate(k[j]):
                ind = rdu.get_default_indicator_name(term_names[m])
                fit['samples'][i]['chains'][ind][j] = km

    return fit


def log_uniform_indicator_set_prior(k, max_nonzero=None):
    """Evaluate uniform prior on indicator set."""

    n_indicators = k.shape[0]

    max_nonzero = check_max_nonzero_indicators(
        max_nonzero, n_indicators=n_indicators)

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


def _bayes_regression_normal_gamma_posterior_parameters(y, X, nu_sq, a_tau,
                                                        b_tau, c_inv=None,
                                                        sigma_inv=None):
    """Calculate parameters for posterior distribution."""

    n_samples = X.shape[0]

    if c_inv is None or sigma_inv is None:
        c_inv, sigma_inv = \
            _bayes_regression_normal_gamma_covariance_matrices(X, nu_sq)

    cov_term = (1.0 + 0.5 * b_tau * np.dot(y.T, np.dot(sigma_inv, y)))

    mean_beta_post = np.ravel(nu_sq * np.dot(c_inv, np.dot(X.T, y)))
    # NB, posterior covariance = sigma_beta_post / precision
    sigma_beta_post = nu_sq * c_inv

    a_tau_post = a_tau + 0.5 * n_samples
    b_tau_post = b_tau / cov_term

    return a_tau_post, b_tau_post, mean_beta_post, sigma_beta_post


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


def bayes_regression_log_model_posterior(k, data=None, max_terms=None,
                                         a_tau=1.0, b_tau=1.0, nu_sq=1.0,
                                         formula_like=None,
                                         lhs_terms=None, optional_terms=None,
                                         force_intercept=True):
    """Evaluate (non-normalized) log model posterior probability."""

    lp = log_uniform_indicator_set_prior(k, max_nonzero=max_terms)

    has_terms = lhs_terms is not None and optional_terms is not None
    if formula_like is None and not has_terms:
        raise ValueError(
            'Either a model formula or term lists must be provided')

    if formula_like is not None and not has_terms:
        y, X = patsy.dmatrices(formula_like, data=data)
        y, X = _check_design_matrices(y, X)

        lhs_terms = y.design_info.terms
        optional_terms, _ = _get_optional_terms(
            X.design_info.terms, term_names=X.design_info.term_names,
            force_intercept=force_intercept)

    model_desc = _get_model_description(
        k, lhs_terms=lhs_terms, optional_terms=optional_terms,
        force_intercept=force_intercept)

    y, X = patsy.dmatrices(model_desc, data=data)

    lp += bayes_regression_normal_gamma_log_marginal_likelihood(
        y, X, a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    return lp


def _sample_uniform_structures_prior(n_terms, max_terms=None,
                                     size=1, random_state=None):
    """Draw a single sample from a uniform prior on structures."""

    rng = check_random_state(random_state)

    max_terms = check_max_nonzero_indicators(
        max_terms, n_indicators=n_terms)

    k = np.zeros((size, n_terms), dtype=int)
    lp = np.zeros((size,))

    for i in range(size):

        # Choose number of terms with weights given by the
        # fraction of the possible sets of terms with that
        # size.
        n_possible_sets = sum([sp.comb(n_terms, m)
                               for m in range(max_terms + 1)])
        weights = np.array([sp.comb(n_terms, m) / n_possible_sets
                            for m in range(max_terms + 1)])

        term_set_size = rng.choice(max_terms + 1, p=weights)

        # Choose uniformly from the set of terms with the
        # chosen size.
        term_indices = rng.choice(n_terms, size=term_set_size,
                                  replace=False)

        k[i, term_indices] = 1
        lp[i] = log_uniform_indicator_set_prior(
            k[i], max_nonzero=max_terms)

    if size == 1:
        k = k[0]
        lp = lp[0]

    return k, lp


def _sample_parameters_conjugate_priors(n_coefs, a_tau=1.0, b_tau=1.0,
                                        nu_sq=1.0, size=1, random_state=None):
    """Sample parameters from conjugate normal-gamma priors."""

    rng = check_random_state(random_state)

    tau_sq = ss.gamma.rvs(a_tau, scale=b_tau, size=size, random_state=rng)

    beta = np.empty(size)
    beta = np.broadcast_to(beta, (n_coefs,) + beta.shape)
    beta = ss.norm.rvs(loc=0.0, scale=np.sqrt(nu_sq / tau_sq),
                       size=beta.shape, random_state=rng)

    lp = (ss.gamma.logpdf(tau_sq, a_tau, scale=b_tau) +
          np.sum(ss.norm.logpdf(beta, loc=0.0,
                                scale=np.sqrt(nu_sq / tau_sq)),
                 axis=0))

    beta = np.moveaxis(beta, 0, -1)

    return beta, tau_sq, lp


def _sample_parameters_posterior(y, X, a_tau=1.0, b_tau=1.0, nu_sq=1.0,
                                 c_inv=None, sigma_inv=None, size=1,
                                 random_state=None):
    """Sample parameters from posterior distribution."""

    rng = check_random_state(random_state)

    a_tau_post, b_tau_post, mean_beta_post, sigma_beta_post = \
        _bayes_regression_normal_gamma_posterior_parameters(
            y, X, a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq,
            c_inv=c_inv, sigma_inv=sigma_inv)

    tau_sq = ss.gamma.rvs(a_tau_post, scale=b_tau_post,
                          size=size, random_state=rng)

    flat_tau_sq = np.ravel(np.atleast_1d(tau_sq))

    n_samples = flat_tau_sq.shape[0]
    n_coef = X.shape[1]
    beta = np.empty((n_samples, n_coef))
    lp = np.ravel(ss.gamma.logpdf(flat_tau_sq, a_tau_post,
                                  scale=b_tau_post))

    for i in range(n_samples):
        beta[i] = ss.multivariate_normal.rvs(
            mean=mean_beta_post,
            cov=(sigma_beta_post / flat_tau_sq[i]),
            random_state=rng)
        lp[i] += ss.multivariate_normal.logpdf(
            beta[i], mean=mean_beta_post,
            cov=(sigma_beta_post / flat_tau_sq[i]))

    if np.ndim(tau_sq) == 0:
        beta = beta[0]
        lp = lp[0]
    else:
        beta = np.reshape(beta, tau_sq.shape + (n_coef,))
        lp = np.reshape(lp, tau_sq.shape)

    return beta, tau_sq, lp


def _sample_outcomes(X, beta, tau_sq, random_state=None):
    """Sample outcomes given parameter values.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Fixed design matrix used for prediction.

    beta : array-like, shape (..., n_features)
        Array of sampled coefficient values.

    tau_sq : array-like
        Array of sampled precision values.

    Returns
    -------
    sampled_outcomes : array, shape (..., n_samples, n_outcomes)
        Array of sampled outcome values.

    log_likelihood : array
        Array containing the values of the log-likelihood
        for each function.
    """

    rng = check_random_state(random_state)

    # Size of sample to generate.
    size = np.shape(tau_sq)

    n_features = X.shape[1]

    if np.shape(beta) != (size + (n_features,)):
        raise ValueError(
            'Sizes of sampled coefficients and precision values '
            ' are inconsistent '
            '(got np.shape(tau_sq)=%r but np.shape(beta)=%r)' %
            (size, np.shape(beta)))

    # Evaluate predicted mean values for each sample,
    # with shape (..., n_samples, n_outcomes)
    cond_means = np.matmul(X, beta[..., np.newaxis])
    cond_scales = np.sqrt(1.0 / tau_sq)
    cond_scales = np.broadcast_to(cond_scales[..., np.newaxis, np.newaxis],
                                  cond_means.shape)

    sampled_outcomes = ss.norm.rvs(
        loc=cond_means, scale=cond_scales,
        random_state=rng)

    log_likelihood = np.sum(
        ss.norm.logpdf(
            sampled_outcomes, loc=cond_means,
            scale=cond_scales),
        axis=(-2, -1))

    return sampled_outcomes, log_likelihood


def _sample_prior_fixed_model(formula_like, data=None,
                              a_tau=1.0, b_tau=1.0, nu_sq=1.0,
                              n_iter=2000,
                              generate_prior_predictive=False,
                              random_state=None):
    """Sample from prior for a fixed model."""

    rng = check_random_state(random_state)

    y, X = patsy.dmatrices(formula_like, data=data)
    y, X = _check_design_matrices(y, X)

    outcome_names = y.design_info.column_names
    coef_names = [rdu.get_default_coefficient_name(n)
                  for n in X.design_info.column_names]
    n_coefs = len(coef_names)

    beta, tau_sq, lp = _sample_parameters_conjugate_priors(
        n_coefs, a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq,
        size=n_iter, random_state=rng)

    chains = collections.OrderedDict({'tau_sq': tau_sq})
    for j, t in enumerate(coef_names):
        chains[t] = beta[:, j]
    chains['lp__'] = lp

    outcome_chains = None
    if generate_prior_predictive:
        sampled_outcomes, _ = _sample_outcomes(
            X, beta, tau_sq, random_state=rng)

        outcome_chains = collections.OrderedDict(
            {n: sampled_outcomes[..., i]
             for i, n in enumerate(outcome_names)})

    args = {'random_state': random_state, 'n_iter': n_iter}

    results = {'chains': chains,
               'args': args,
               'acceptance': 1.0,
               'accept_stat': np.ones((n_iter,), dtype=float),
               'mean_lp__': np.mean(chains['lp__'])}

    prior_predictive = None
    if generate_prior_predictive:
        prior_predictive = {
            'chains': outcome_chains,
            'args': args,
            'acceptance': 1.0,
            'accept_stat': np.ones((n_iter,), dtype=float)
        }

    return results, prior_predictive


def _sample_posterior_fixed_model(formula_like, data=None,
                                  a_tau=1.0, b_tau=1.0, nu_sq=1.0,
                                  n_iter=2000, thin=1,
                                  generate_posterior_predictive=False,
                                  random_state=None):
    """Sample from posterior for a fixed model."""

    rng = check_random_state(random_state)

    y, X = patsy.dmatrices(formula_like, data=data)
    y, X = _check_design_matrices(y, X)

    outcome_names = y.design_info.column_names
    coef_names = [rdu.get_default_coefficient_name(n)
                  for n in X.design_info.column_names]

    beta, tau_sq, lp = _sample_parameters_posterior(
        y, X, a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq,
        size=n_iter, random_state=rng)

    chains = collections.OrderedDict({'tau_sq': tau_sq[::thin]})
    for j, t in enumerate(coef_names):
        chains[t] = beta[::thin, j]
    chains['lp__'] = lp[::thin]

    outcome_chains = None
    if generate_posterior_predictive:
        sampled_outcomes, _ = _sample_outcomes(
            X, beta, tau_sq, random_state=rng)

        outcome_chains = collections.OrderedDict(
            {n: sampled_outcomes[::thin, ..., i]
             for i, n in enumerate(outcome_names)})

    args = {'random_state': random_state, 'n_iter': n_iter,
            'thin': thin}

    results = {'chains': chains,
               'args': args,
               'acceptance': 1.0,
               'accept_stat': np.ones((n_iter,), dtype=float),
               'mean_lp__': np.mean(chains['lp__'])}

    posterior_predictive = None
    if generate_posterior_predictive:
        posterior_predictive = {
            'chains': outcome_chains,
            'args': args,
            'acceptance': 1.0,
            'accept_stat': np.ones((n_iter,), dtype=float)
        }

    return results, posterior_predictive


def _get_structure_samples_posterior_predictive(fit, formula_like, data=None,
                                                a_tau=1.0, b_tau=1.0,
                                                nu_sq=1.0,
                                                force_intercept=True,
                                                random_state=None):
    """Get posterior predictive samples corresponding to structure sample."""

    rng = check_random_state(random_state)

    y, X = patsy.dmatrices(formula_like, data=data)
    y, X = _check_design_matrices(y, X)

    lhs_terms = y.design_info.terms
    outcome_names = y.design_info.column_names
    optional_terms, optional_term_names = _get_optional_terms(
        X.design_info.terms, term_names=X.design_info.term_names,
        force_intercept=force_intercept)

    n_outcomes = len(outcome_names)

    max_nonzero = fit['max_nonzero']
    allow_exchanges = fit['samples'][0]['args']['allow_exchanges']
    n_chains = len(fit['samples'])
    n_save = fit['n_save']
    thin = fit['thin']

    random_seeds = rng.choice(1000000 * n_chains,
                              size=n_chains, replace=False)

    samples = []
    for i in range(n_chains):

        chain_rng = check_random_state(random_seeds[i])
        chain_k = fit['samples'][i]['chains']['k']

        sampled_outcomes = None
        for j in range(n_save[i]):

            model_desc = _get_model_description(
                chain_k[j], lhs_terms=lhs_terms,
                optional_terms=optional_terms,
                force_intercept=force_intercept)

            _, model_sample = _sample_posterior_fixed_model(
                model_desc, data=data, a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq,
                n_iter=1,
                generate_posterior_predictive=True,
                random_state=chain_rng)

            if sampled_outcomes is None:
                n_samples = model_sample['chains'][outcome_names[0]].shape[0]
                sampled_outcomes = np.zeros((n_save[i], n_samples, n_outcomes))

            for m, n in enumerate(outcome_names):
                sampled_outcomes[j, ..., m] = model_sample['chains'][n]

        chains = collections.OrderedDict(
            {n: sampled_outcomes[..., m]
             for m, n in enumerate(outcome_names)})

        args = {'random_state': random_seeds[i], 'n_iter': fit['n_iter'],
                'thin': thin,
                'allow_exchanges': allow_exchanges,
                'max_nonzero': max_nonzero}

        samples.append({'chains': chains,
                        'args': args,
                        'acceptance': fit['samples'][i]['acceptance'],
                        'accept_stat': fit['samples'][i]['accept_stat']})

    posterior_predictive = {
        'samples': samples,
        'n_chains': n_chains,
        'n_iter': fit['n_iter'],
        'warmup': fit['warmup'],
        'thin': fit['thin'],
        'n_save': fit['n_save'],
        'warmup2': fit['warmup2'],
        'max_nonzero': fit['max_nonzero'],
        'permutation': fit['permutation'],
        'random_seeds': random_seeds
    }

    return posterior_predictive


def _sample_prior_full(formula_like, data=None,
                       a_tau=1.0, b_tau=1.0, nu_sq=1.0,
                       max_terms=None, n_iter=2000,
                       force_intercept=True,
                       generate_prior_predictive=False,
                       random_state=None):
    """Sample from prior over all models."""

    rng = check_random_state(random_state)

    lhs_terms, outcome_names, optional_terms, optional_term_names = \
        _get_outcome_and_optional_terms(
            formula_like, data=data, force_intercept=force_intercept)

    n_outcomes = len(outcome_names)
    n_terms = len(optional_terms)
    max_terms = check_max_nonzero_indicators(
        max_terms, n_indicators=n_terms)

    k = np.zeros((n_iter, n_terms), dtype=int)
    named_indicators = {
        rdu.get_default_indicator_name(t): np.zeros((n_iter,), dtype=int)
        for t in optional_term_names}
    lp = np.empty((n_iter,))
    sampled_outcomes = None

    for i in range(n_iter):

        k[i], lp[i] = _sample_uniform_structures_prior(
            n_terms, max_terms=max_terms, random_state=rng)

        for m in range(n_terms):
            if k[i, m] == 1:
                ind = rdu.get_default_indicator_name(
                    optional_term_names[m])
                named_indicators[ind][i] = 1

        if generate_prior_predictive:
            model_desc = _get_model_description(
                k[i], lhs_terms=lhs_terms, optional_terms=optional_terms,
                force_intercept=force_intercept)

            _, model_sample = _sample_prior_fixed_model(
                model_desc, data=data, a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq,
                n_iter=1,
                generate_prior_predictive=generate_prior_predictive,
                random_state=rng)

            if sampled_outcomes is None:
                n_samples = model_sample['chains'][outcome_names[0]].shape[0]
                sampled_outcomes = np.zeros((n_iter, n_samples, n_outcomes))

            for j, n in enumerate(outcome_names):
                sampled_outcomes[i, ..., j] = model_sample['chains'][n]

    chains = collections.OrderedDict({'k': k})
    for ind in named_indicators:
        chains[ind] = named_indicators[ind]
    chains['lp__'] = lp

    args = {'random_state': random_state, 'n_iter': n_iter,
            'max_nonzero': max_terms}

    results = {'chains': chains,
               'args': args,
               'acceptance': 1.0,
               'accept_stat': np.ones((n_iter,), dtype=float),
               'mean_lp__': np.mean(chains['lp__'])}

    prior_predictive = None
    if generate_prior_predictive:
        outcome_chains = collections.OrderedDict(
            {n: sampled_outcomes[..., i]
             for i, n in enumerate(outcome_names)})

        prior_predictive = {
            'chains': outcome_chains,
            'args': args,
            'acceptance': 1.0,
            'accept_stat': np.ones((n_iter,), dtype=float)
        }

    return results, prior_predictive


def _sample_posterior_full(formula_like, data=None,
                           a_tau=1.0, b_tau=1.0, nu_sq=1.0,
                           n_chains=4, n_iter=1000, warmup=None, thin=1,
                           verbose=False,
                           n_jobs=-1, max_terms=None,
                           restart_file=None, init='random',
                           allow_exchanges=True,
                           generate_posterior_predictive=False,
                           force_intercept=True,
                           random_state=None):
    """Sample from model posterior using MC3 algorithm."""

    rng = check_random_state(random_state)

    y, X = patsy.dmatrices(formula_like, data=data)
    y, X = _check_design_matrices(y, X)

    outcome_names = y.design_info.column_names
    optional_terms, optional_term_names = _get_optional_terms(
        X.design_info.terms, term_names=X.design_info.term_names,
        force_intercept=force_intercept)

    constant_data_names = None
    if data is not None:
        constant_data_names = [n for n in data if n not in outcome_names]

    n_terms = len(optional_terms)
    n_chains = rdu.check_number_of_chains(n_chains)

    if restart_file is None:
        initial_k = initialize_stepwise_mc3(
            n_terms, n_chains=n_chains, max_nonzero=max_terms, method=init,
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
        return bayes_regression_log_model_posterior(
            k, data=data, max_terms=max_terms,
            a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq,
            formula_like=formula_like,
            force_intercept=force_intercept)

    fit = sample_stepwise_mc3(
        initial_k, logp, data=data,
        n_chains=n_chains, n_iter=n_iter, thin=thin,
        warmup=warmup, verbose=verbose, n_jobs=n_jobs,
        max_nonzero=max_terms, allow_exchanges=allow_exchanges,
        random_state=rng)

    # Generate named indicator variables for convenience.
    fit = _add_named_indicator_variables_to_fit(
        fit, term_names=optional_term_names)
    coords = {'term': optional_term_names}
    dims = {'k': ['term']}

    posterior_predictive = None
    if generate_posterior_predictive:
        posterior_predictive = _get_structure_samples_posterior_predictive(
            fit, formula_like, data=data,
            a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq,
            force_intercept=force_intercept, random_state=rng)

    return convert_samples_dict_to_inference_data(
        posterior=fit, posterior_predictive=posterior_predictive,
        observed_data=data, constant_data=data, save_warmup=True,
        observed_data_names=outcome_names,
        constant_data_names=constant_data_names,
        coords=coords, dims=dims)


def _unique_models(sample_ds, indicator_var='k'):
    """Get list of unique models."""

    n_total = sample_ds.sizes['chain'] * sample_ds.sizes['draw']

    indicator_shape = sample_ds[indicator_var].shape
    if len(indicator_shape) > 2:
        n_indicators = indicator_shape[-1]
        flat_indicators = np.reshape(
            sample_ds[indicator_var].data, (n_total, n_indicators))
    else:
        flat_indicators = np.reshape(
            sample_ds[indicator_var].data, (n_total, 1))

    return np.unique(flat_indicators, axis=0)


def _count_possible_models(sample_ds, max_nonzero=None, indicator_var='k'):
    """Count the number of possible models."""

    indicator_shape = sample_ds[indicator_var].shape
    if len(indicator_shape) > 2:
        n_indicators = indicator_shape[-1]
    else:
        n_indicators = 1

    if max_nonzero is None:
        max_nonzero = n_indicators

    return int(sum([sp.comb(n_indicators, i)
                    for i in range(max_nonzero + 1)]))


def _get_model_lookup(sample_ds, indicator_var='k'):
    """Get lookup table for model labels."""

    unique_k = _unique_models(sample_ds, indicator_var=indicator_var)

    return {tuple(ki): '{:d}'.format(i)
            for i, ki in enumerate(unique_k)}


def _get_model_indicators(sample_ds, only_sampled_models=True,
                          max_nonzero=None, indicator_var='k'):
    """Get indicator variables for individual models."""

    model_lookup = _get_model_lookup(sample_ds, indicator_var=indicator_var)

    n_chains = sample_ds.sizes['chain']
    n_draws = sample_ds.sizes['draw']

    z = np.empty((n_chains, n_draws), dtype=str)

    for i in range(n_chains):

        chain_k = sample_ds[indicator_var].isel(chain=i).data

        for t in range(n_draws):
            z[i, t] = model_lookup[tuple(chain_k[t])]

    model_indicators = [model_lookup[i] for i in model_lookup]
    if not only_sampled_models:
        n_observed_models = len(model_indicators)
        n_possible_models = _count_possible_models(
            sample_ds, max_nonzero=max_nonzero, indicator_var=indicator_var)
        model_indicators += ['{:d}'.format(i)
                             for i in range(n_observed_models,
                                            n_possible_models)]

    return z, model_indicators


def structure_sample_convergence_rate(sample_ds, max_nonzero=None,
                                      indicator_var='k', sparse=True,
                                      combine_chains=False):
    """Estimate convergence rate for structure chains."""

    z, model_indicators = _get_model_indicators(
        sample_ds, max_nonzero=max_nonzero,
        only_sampled_models=True,
        indicator_var=indicator_var)

    return estimate_convergence_rate(z, sparse=sparse,
                                     combine_chains=combine_chains)


def structure_sample_chi2_convergence_diagnostics(fit, max_nonzero=None,
                                                  indicator_var='k',
                                                  batch=True, **kwargs):
    """Calculate chi squared convergence diagnostics."""

    if batch:
        samples = xr.concat(
            [fit.warmup_posterior, fit.posterior], dim='draw')
    else:
        samples = fit.posterior

    z, _ = _get_model_indicators(
        samples, max_nonzero=max_nonzero,
        only_sampled_models=True,
        indicator_var=indicator_var)

    if batch:
        return rjmcmc_batch_chisq_convergence(
            z, **kwargs)

    return rjmcmc_chisq_convergence(
        z, **kwargs)


def structure_sample_ks_convergence_diagnostics(fit, max_nonzero=None,
                                                indicator_var='k',
                                                batch=True, **kwargs):
    """Calculate chi squared convergence diagnostics."""

    if batch:
        samples = xr.concat(
            [fit.warmup_posterior, fit.posterior], dim='draw')
    else:
        samples = fit.posterior

    z, _ = _get_model_indicators(
        samples, max_nonzero=max_nonzero,
        only_sampled_models=True,
        indicator_var=indicator_var)

    if batch:
        return rjmcmc_batch_kstest_convergence(
            z, **kwargs)

    return rjmcmc_kstest_convergence(
        z, **kwargs)


def structure_sample_diagnostics(sample_ds, max_nonzero=None,
                                 n_samples=100, only_sampled_models=True,
                                 epsilon=None, sparse=True,
                                 min_epsilon=1e-6, tolerance=1e-4,
                                 fit_kwargs=None, indicator_var='k',
                                 random_state=None):

    z, model_indicators = _get_model_indicators(
        sample_ds, max_nonzero=max_nonzero,
        only_sampled_models=only_sampled_models,
        indicator_var=indicator_var)

    return estimate_stationary_distribution(
        z, model_indicators=model_indicators, sparse=sparse,
        epsilon=epsilon, n_samples=n_samples,
        min_epsilon=min_epsilon, tolerance=tolerance,
        fit_kwargs=fit_kwargs, random_state=random_state)


def _get_model_indicator_arrays(sample_ds, indicator_var='k'):
    """Get values of indicator variables in each model."""

    indicator_shape = sample_ds[indicator_var].shape
    if len(indicator_shape) > 2:
        n_indicators = indicator_shape[-1]
    else:
        n_indicators = 1

    term_names = sample_ds['term'].values

    model_lookup = _get_model_lookup(sample_ds, indicator_var=indicator_var)
    indicator_lookup = {model_lookup[k]: k for k in model_lookup}

    model_names = np.array(list(indicator_lookup.keys()))
    n_observed_models = len(model_names)

    indicator_vals = {}
    for i in range(n_indicators):
        ki_vals = np.zeros((n_observed_models,), dtype=int)
        for j, m in enumerate(model_names):
            ki_vals[j] = indicator_lookup[m][i]

        term_name = term_names[i]
        indicator_vals[term_name] = ki_vals

    return indicator_vals


def _calculate_model_logp(sample_ds, sample_stats_ds, indicator_var='k'):
    """Calculate logp for each model."""

    n_chains = sample_ds.sizes['chain']

    model_lookup = _get_model_lookup(sample_ds, indicator_var=indicator_var)
    indicator_lookup = {model_lookup[k]: k for k in model_lookup}

    model_names = np.array(list(indicator_lookup.keys()))
    n_observed_models = len(model_names)

    model_lp = np.zeros((n_observed_models,))
    model_counts = np.zeros((n_observed_models,), dtype=int)
    for j in range(n_chains):

        chain_k = sample_ds[indicator_var].isel(chain=j).data
        chain_lp = sample_stats_ds['lp'].isel(chain=j).data

        for i, m in enumerate(indicator_lookup):
            model_k = indicator_lookup[m]
            mask = np.all(chain_k == model_k, axis=1)
            model_lp[i] += np.sum(chain_lp[mask])
            model_counts[i] += np.sum(mask)

    model_lp = model_lp / model_counts

    return model_lp


def _calculate_model_posterior_probabilities(sample_ds, indicator_var='k'):
    """Calculate posterior probability for each model."""

    n_chains = sample_ds.sizes['chain']

    model_lookup = _get_model_lookup(sample_ds, indicator_var=indicator_var)
    indicator_lookup = {model_lookup[k]: k for k in model_lookup}

    model_names = np.array(list(indicator_lookup.keys()))
    n_observed_models = len(model_names)

    model_counts = np.zeros((n_observed_models,), dtype=int)

    n_samples = 0
    for j in range(n_chains):

        chain_k = sample_ds[indicator_var].isel(chain=j).data

        chain_k = [tuple(k) for k in chain_k]
        n_samples += len(chain_k)

        for i, m in enumerate(model_names):
            km = indicator_lookup[m]
            model_counts[i] += sum([k == km for k in chain_k])

    return model_counts / n_samples


def stepwise_model_samples_summary(inference_data, show_indicators=True,
                                   posterior=True, sort_by_probs=True,
                                   indicator_var='k'):
    """Return summary of sampled models."""

    if posterior:
        sample_ds = inference_data.posterior
    else:
        sample_ds = inference_data.prior

    model_lookup = _get_model_lookup(sample_ds, indicator_var=indicator_var)
    indicator_lookup = {model_lookup[k]: k for k in model_lookup}

    model_names = np.array(list(indicator_lookup.keys()))

    if show_indicators:
        indicator_vals = _get_model_indicator_arrays(
            sample_ds, indicator_var=indicator_var)
    else:
        indicator_vals = None

    model_probs = _calculate_model_posterior_probabilities(
        sample_ds, indicator_var=indicator_var)
    model_lp = _calculate_model_logp(
        sample_ds, inference_data.sample_stats,
        indicator_var=indicator_var)

    if sort_by_probs:
        model_order = np.argsort(-model_probs)

        model_names = model_names[model_order]
        model_probs = model_probs[model_order]
        model_lp = model_lp[model_order]

        if show_indicators:
            for ki in indicator_vals:
                indicator_vals[ki] = indicator_vals[ki][model_order]

    data_vars = collections.OrderedDict({'model': model_names})
    if show_indicators:
        for ki in indicator_vals:
            data_vars[ki] = indicator_vals[ki]

    data_vars['lp__'] = model_lp
    data_vars['p'] = model_probs

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

    def sample_structures_prior(self, formula_like, data=None,
                                max_terms=None, n_chains=4, n_iter=1000,
                                n_jobs=-1, generate_prior_predictive=False,
                                random_state=None):
        """Sample structures for model from uniform priors.

        Parameters
        ----------
        formula_like : object
            Formula-like object describing the largest possible model
            to be allowed, e.g., a string of the form
            "y ~ x1 + x2 + ... + xN", where y is the outcome variable
            to be modelled and x1, x2, ..., xN are all of the possible
            predictors considered for inclusion in the model.

        data : dict-like
            Data to be used in constructing the model.

        n_chains : int, default: 4
            Number of chains.

        n_iter : int, default: 1000
            Number of samples per chain.

        max_terms : int
            Maximum number of terms allowed in model.

        random_state : integer, RandomState or None
            If an integer, random_state is the seed used by the
            random number generator. If a RandomState instance,
            random_state is the random number generator. If None,
            the random number generator is the RandomState instance
            used by `np.random`.

        Returns
        -------
        draws : dict
            Dictionary containing the results of sampling.
        """

        rng = check_random_state(random_state)

        _, outcome_names, _, _ = \
            _get_outcome_and_optional_terms(
                formula_like, data=data,
                force_intercept=self.force_intercept)

        constant_data_names = None
        if data is not None:
            constant_data_names = [n for n in data
                                   if n not in outcome_names]

        n_chains = rdu.check_number_of_chains(n_chains)
        n_iter = rdu.check_number_of_iterations(n_iter)

        if n_jobs is None:
            n_jobs = -1
        elif n_chains == 1:
            n_jobs = 1

        random_seeds = rng.choice(1000000 * n_chains,
                                  size=n_chains, replace=False)

        def _sample(seed):
            return _sample_prior_full(
                formula_like, data=data,
                a_tau=self.a_tau, b_tau=self.b_tau, nu_sq=self.nu_sq,
                max_terms=max_terms, n_iter=n_iter,
                force_intercept=self.force_intercept,
                generate_prior_predictive=generate_prior_predictive,
                random_state=seed)

        samples = Parallel(n_jobs=n_jobs)(
            delayed(_sample)(seed) for seed in random_seeds)

        structure_samples = [sample[0] for sample in samples]

        prior = {'samples': structure_samples,
                 'n_chains': len(samples),
                 'n_iter': n_iter,
                 'n_save': [n_iter] * n_chains,
                 'random_seeds': random_seeds}

        prior_predictive = None
        if generate_prior_predictive:
            generated_samples = [sample[1] for sample in samples]
            prior_predictive = {
                'samples': generated_samples,
                'n_chains': len(samples),
                'n_iter': n_iter,
                'n_save': [n_iter] * n_chains,
                'random_seeds': random_seeds}

        return convert_samples_dict_to_inference_data(
            prior=prior,
            prior_predictive=prior_predictive,
            observed_data=data, constant_data=data, save_warmup=True,
            observed_data_names=outcome_names,
            constant_data_names=constant_data_names)

    def sample_parameters_prior(self, formula_like, data=None, n_chains=4,
                                n_iter=1000, n_jobs=-1,  random_state=None,
                                generate_prior_predictive=False):
        """Sample parameters for model from conditional priors.

        Parameters
        ----------
        formula_like : object
            Formula-like object describing the terms in the given model
            (i.e., containing only those terms present in the model, not
            the full set of possible predictors).

        data : dict-like
            Data to be used in constructing the model.

        n_chains : int, default: 4
            Number of chains.

        n_iter : int, default: 1000
            Number of samples per chain.

        random_state : integer, RandomState or None
            If an integer, random_state is the seed used by the
            random number generator. If a RandomState instance,
            random_state is the random number generator. If None,
            the random number generator is the RandomState instance
            used by `np.random`.

        Returns
        -------
        draws : dict
            Dictionary containing the results of sampling.
        """

        rng = check_random_state(random_state)

        n_chains = rdu.check_number_of_chains(n_chains)
        n_iter = rdu.check_number_of_iterations(n_iter)

        if n_jobs is None:
            n_jobs = -1
        elif n_chains == 1:
            n_jobs = 1

        y, X = patsy.dmatrices(formula_like, data=data)
        y, X = _check_design_matrices(y, X)

        outcome_names = y.design_info.column_names

        constant_data_names = None
        if data is not None:
            constant_data_names = [n for n in data if n not in outcome_names]

        random_seeds = rng.choice(1000000 * n_chains,
                                  size=n_chains, replace=False)

        def _sample(seed):
            return _sample_prior_fixed_model(
                formula_like, data=data, a_tau=self.a_tau, b_tau=self.b_tau,
                nu_sq=self.nu_sq, n_iter=n_iter,
                generate_prior_predictive=generate_prior_predictive,
                random_state=seed)

        samples = Parallel(n_jobs=n_jobs)(
            delayed(_sample)(seed) for seed in random_seeds)

        parameter_samples = [sample[0] for sample in samples]

        prior = {'samples': parameter_samples,
                 'n_chains': len(samples),
                 'n_iter': n_iter,
                 'n_save': [n_iter] * n_chains,
                 'random_seeds': random_seeds}

        prior_predictive = None
        if generate_prior_predictive:
            generated_samples = [sample[1] for sample in samples]
            prior_predictive = {
                'samples': generated_samples,
                'n_chains': len(samples),
                'n_iter': n_iter,
                'n_save': [n_iter] * n_chains,
                'random_seeds': random_seeds}

        return convert_samples_dict_to_inference_data(
            prior=prior,
            prior_predictive=prior_predictive,
            observed_data=data, constant_data=data, save_warmup=True,
            observed_data_names=outcome_names,
            constant_data_names=constant_data_names)

    def sample_structures_posterior(self, formula_like, data=None,
                                    max_terms=None,
                                    n_chains=4, n_iter=1000, thin=1,
                                    warmup=None, n_jobs=-1,
                                    verbose=False, restart_file=None,
                                    init='random', random_state=None,
                                    generate_posterior_predictive=False):
        """Sample parameters for model from conditional posteriors.

        Parameters
        ----------
        formula_like : object
            Formula-like object describing the terms in the given model
            (i.e., containing only those terms present in the model, not
            the full set of possible predictors).

        data : dict-like
            Data to be used in constructing the model.

        n_chains : int, default: 4
            Number of chains.

        n_iter : int, default: 1000
            Number of samples per chain.

        thin : int, default: 1
            Interval used for thinning samples.

        warmup : int, optional
            Number of warm-up samples.

        random_state : integer, RandomState or None
            If an integer, random_state is the seed used by the
            random number generator. If a RandomState instance,
            random_state is the random number generator. If None,
            the random number generator is the RandomState instance
            used by `np.random`.

        Returns
        -------
        draws : dict
            Dictionary containing the results of sampling.
        """

        return _sample_posterior_full(
            formula_like, data=data,
            a_tau=self.a_tau, b_tau=self.b_tau, nu_sq=self.nu_sq,
            n_chains=n_chains, n_iter=n_iter, warmup=warmup, thin=thin,
            verbose=verbose, n_jobs=n_jobs, max_terms=max_terms,
            restart_file=restart_file, init=init,
            allow_exchanges=self.allow_exchanges,
            force_intercept=self.force_intercept,
            generate_posterior_predictive=generate_posterior_predictive,
            random_state=random_state)

    def sample_parameters_posterior(self, formula_like, data=None,
                                    n_chains=4, n_iter=1000, thin=1,
                                    warmup=None, n_jobs=-1,
                                    random_state=None,
                                    generate_posterior_predictive=False):
        """Sample parameters for model from conditional posteriors.

        Parameters
        ----------
        formula_like : object
            Formula-like object describing the terms in the given model
            (i.e., containing only those terms present in the model, not
            the full set of possible predictors).

        data : dict-like
            Data to be used in constructing the model.

        n_chains : int, default: 4
            Number of chains.

        n_iter : int, default: 1000
            Number of samples per chain.

        thin : int, default: 1
            Interval used for thinning samples.

        warmup : int, optional
            Number of warm-up samples.

        random_state : integer, RandomState or None
            If an integer, random_state is the seed used by the
            random number generator. If a RandomState instance,
            random_state is the random number generator. If None,
            the random number generator is the RandomState instance
            used by `np.random`.

        Returns
        -------
        draws : dict
            Dictionary containing the results of sampling.
        """

        rng = check_random_state(random_state)

        n_chains = rdu.check_number_of_chains(n_chains)
        n_iter = rdu.check_number_of_iterations(n_iter)

        if warmup is None:
            warmup = n_iter // 2
        else:
            warmup = rdu.check_warmup(warmup, n_iter)

        if n_jobs is None:
            n_jobs = -1
        elif n_chains == 1:
            n_jobs = 1

        y, X = patsy.dmatrices(formula_like, data=data)
        y, X = _check_design_matrices(y, X)

        outcome_names = y.design_info.column_names

        constant_data_names = None
        if data is not None:
            constant_data_names = [n for n in data if n not in outcome_names]

        random_seeds = rng.choice(1000000 * n_chains,
                                  size=n_chains, replace=False)

        def _sample(seed):
            return _sample_posterior_fixed_model(
                formula_like, data=data, a_tau=self.a_tau, b_tau=self.b_tau,
                nu_sq=self.nu_sq, n_iter=n_iter, thin=thin,
                generate_posterior_predictive=generate_posterior_predictive,
                random_state=seed)

        samples = Parallel(n_jobs=n_jobs)(
            delayed(_sample)(seed) for seed in random_seeds)

        parameter_samples = [sample[0] for sample in samples]

        warmup2 = warmup // thin
        n_save = 1 + (n_iter - 1) // thin
        n_kept = n_save - warmup2
        perm_lst = [rng.permutation(int(n_kept)) for _ in range(n_chains)]

        posterior = {'samples': parameter_samples,
                     'n_chains': len(samples),
                     'n_iter': n_iter,
                     'warmup': warmup,
                     'thin': thin,
                     'n_save': [n_save] * n_chains,
                     'warmup2': [warmup2] * n_chains,
                     'permutation': perm_lst,
                     'random_seeds': random_seeds}

        posterior_predictive = None
        if generate_posterior_predictive:
            generated_samples = [sample[1] for sample in samples]
            posterior_predictive = {
                'samples': generated_samples,
                'n_chains': len(samples),
                'n_iter': n_iter,
                'warmup': warmup,
                'thin': thin,
                'n_save': [n_save] * n_chains,
                'warmup2': [warmup2] * n_chains,
                'permutation': perm_lst,
                'random_seeds': random_seeds}

        return convert_samples_dict_to_inference_data(
            posterior=posterior,
            posterior_predictive=posterior_predictive,
            observed_data=data, constant_data=data, save_warmup=True,
            observed_data_names=outcome_names,
            constant_data_names=constant_data_names)
