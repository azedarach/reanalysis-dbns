"""
Provides helper routines for working with sampler output.
"""

# License: MIT

from __future__ import absolute_import, division


import arviz as az
import numpy as np

import reanalysis_dbns.utils as rdu


def get_indicator_set_neighborhood(k, allow_exchanges=True,
                                   max_nonzero=None,
                                   include_current_set=False):
    """Get all indicator variable sets in neighborhood of current set.

    Parameters
    ----------
    k : array-like, shape (n_indicators,)
        Array containing indicator variables.

    allow_exchanges : bool, default: True
        If True, allow moves that exchange one non-zero indicator
        for a currently zero indicator. If False, only moves
        that change the state of a single indicator are considered.

    max_nonzero : int
        If given, the maximum number of non-zero indicators to
        allow.

    include_current_set : bool, default: False
        If True, include the current set of indicators in the
        definition of the neighborhood.

    Returns
    -------
    nhd : list
        List containing the neighborhood of the current indicator set.
        Each element of the list is a dict with keys 'increment' and
        'decrement', with the corresponding values being lists
        containing the indices of the indicator variables to increment
        or decrement, respectively.
    """

    nonzero_indices = np.nonzero(k)[0]
    zero_indices = np.array([i for i in np.arange(np.size(k))
                             if i not in nonzero_indices])

    if include_current_set:
        nhd = [{'increment': [], 'decrement': []}]
    else:
        nhd = []

    # Find all indicator sets differing by zeroing of one indicator.
    for i in nonzero_indices:
        nhd.append({'increment': [], 'decrement': [i]})

    current_nonzero_set_size = np.size(nonzero_indices)
    has_max_nonzero = np.size(zero_indices) == 0

    if allow_exchanges and not has_max_nonzero:
        # Find all indicator sets differing by exchange of a pair of
        # indicators.
        for i in nonzero_indices:
            for j in zero_indices:
                nhd.append({'increment': [j],
                            'decrement': [i]})

    if max_nonzero is not None:

        if current_nonzero_set_size > max_nonzero:
            raise RuntimeError(
                'Current number of non-zero indicators exceeds '
                'maximum allowed '
                '(got np.size(nonzero_indices)=%d but max_nonzero=%d)' %
                (current_nonzero_set_size, max_nonzero))

        has_max_nonzero = (has_max_nonzero or
                           current_nonzero_set_size >= max_nonzero)

    # Find all indicator sets differing by addition of one indicator.
    if not has_max_nonzero:
        for i in zero_indices:
            nhd.append({'increment': [i], 'decrement': []})

    return nhd


def check_common_sampler_settings(n_chains, n_iter, warmup=None, n_jobs=None):
    """Check common sampler settings."""

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

    return n_chains, n_iter, warmup, n_jobs


def get_sampled_parameter_dimensions(chain, par):
    """Get dimensions of parameter in fit output."""

    if par not in chain:
        raise ValueError(
            "Could not find parameter '%r' in fit output" % par)

    if chain[par].ndim == 1:
        # Output has shape (n_iter,), i.e., parameter is a scalar.
        return ()

    return chain[par].shape[1:]


def get_log_likelihood_data(fit):
    """Get log-likelihood group from fit for writing to file."""

    n_chains = len(fit['samples'])
    n_iter = fit['n_iter']
    thin = fit['thin']
    n_draws = n_iter // thin

    if 'log_likelihood' in fit['samples'][0]['chains']:
        log_likelihood = {'log_likelihood': np.empty((n_chains, n_draws))}
        for i in range(n_chains):
            log_likelihood['log_likelihood'][i] = \
                fit['samples'][i]['chains']['log_likelihood'].copy()
    else:
        log_likelihood = None

    return log_likelihood


def get_sample_stats_data(fit):
    """Get sample statistics group from fit for writing to file."""

    n_chains = len(fit['samples'])
    n_iter = fit['n_iter']
    thin = fit['thin']
    n_draws = n_iter // thin

    accept_stat = np.empty((n_chains, n_draws))
    lp = np.empty((n_chains, n_draws))
    for i in range(n_chains):
        accept_stat[i] = fit['samples'][i]['accept_stat'].copy()
        lp[i] = fit['samples'][i]['chains']['lp__'].copy()

    sample_stats = {'accept_stat': accept_stat, 'lp': lp}
    if 'log_marginal_likelihood' in fit['samples'][0]['chains']:
        sample_stats['log_marginal_likelihood'] = np.empty(
            (n_chains, n_draws))
        for i in range(n_chains):
            sample_stats['log_marginal_likelihood'][i] = \
                fit['samples'][i]['chains'][
                    'log_marginal_likelihood'].copy()

    return sample_stats


def get_mc3_posterior_data(fit):
    """Get posterior group data from fit."""

    n_chains = len(fit['samples'])
    n_iter = fit['n_iter']
    thin = fit['thin']
    n_draws = n_iter // thin

    n_indicators = fit['samples'][0]['chains']['k'].shape[1]

    k = np.empty((n_chains, n_draws, n_indicators), dtype='i8')
    extra_pars = {}
    for p in fit['samples'][0]['chains']:
        if p == 'lp__':
            continue

        if fit['samples'][0]['chains'][p].ndim > 1:
            p_shape = fit['samples'][0]['chains'][p][0].shape
        else:
            p_shape = []

        extra_pars[p] = np.empty([n_chains, n_draws] + p_shape)

    for i in range(n_chains):
        k[i] = fit['samples'][i]['chains']['k'].copy()
        for p in extra_pars:
            extra_pars[p][i] = fit['samples'][i]['chains'][p].copy()

    posterior = {'k': k}
    coords = {'indicator': np.arange(n_indicators)}
    dims = {'k': ['indicator']}

    for p in extra_pars:
        posterior[p] = extra_pars[p]

        if extra_pars[p].ndim > 2:
            extra_dims = extra_pars[p].shape[2:]
            n_extra_dims = len(extra_dims)
            for i, d in enumerate(extra_dims):
                coords['{}_dim_{:d}'.format(p, i)] = np.arange(d)
            dims[p] = ['{}_dim_{:d}'.format(p, i) for i in range(n_extra_dims)]

    return posterior, coords, dims


def write_stepwise_mc3_samples(fit, sample_file, data=None):
    """Write MC3 samples to file."""

    posterior, coords, dims = get_mc3_posterior_data(fit)
    sample_stats = get_sample_stats_data(fit)
    log_likelihood = get_log_likelihood_data(fit)

    samples = az.from_dict(
        posterior,
        log_likelihood=log_likelihood,
        sample_stats=sample_stats,
        coords=coords, dims=dims)

    samples.to_netcdf(sample_file)
