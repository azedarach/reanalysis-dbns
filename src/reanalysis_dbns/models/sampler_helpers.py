"""
Provides helper routines for working with sampler output.
"""

# License: MIT

from __future__ import absolute_import, division

import collections

import arviz as az
import numpy as np
import pandas as pd

import reanalysis_dbns.utils as rdu


def check_max_nonzero_indicators(max_nonzero, n_indicators):
    """Check maximum number of non-zero indicators is valid."""

    if max_nonzero is None:
        max_nonzero = n_indicators

    invalid_max_nonzero = (not rdu.is_integer(max_nonzero) or
                           max_nonzero < 0 or
                           max_nonzero > n_indicators)
    if invalid_max_nonzero:
        raise ValueError(
            'Maximum number of non-zero indicators must be an integer '
            'between 0 and %d '
            '(got max_nonzero=%r)' % (n_indicators, max_nonzero))

    return max_nonzero


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


def format_percentile(q):
    """Format percentile as a string."""
    if 0 <= q <= 1.0:
        q = 100.0 * q
    return '{:3.1f}%'.format(q)


def get_sample_summary_statistics(var_values, probs=None):
    """Get sample summary statistics."""

    n_vars = len(var_values)

    if probs is None:
        probs = [0.025, 0.25, 0.5, 0.75, 0.975]

    col_names = ['par_name', 'mean', 'sd']
    for q in probs:
        col_names += [format_percentile(q)]

    sample_stats = {'par_name': []}
    for n in col_names[1:]:
        sample_stats[n] = np.empty((n_vars,), dtype=float)

    for i, p in enumerate(var_values):
        sample_stats['par_name'].append(p)
        sample_stats['mean'][i] = np.mean(var_values[p])
        sample_stats['sd'][i] = np.std(var_values[p])

        for q in probs:
            sample_stats[format_percentile(q)][i] = np.quantile(
                var_values[p], q)

    data_vars = collections.OrderedDict(
        {c: sample_stats[c] for c in col_names})

    return pd.DataFrame(data_vars)


def _get_number_of_warmup_and_kept_draws_from_fit(fit):
    """Get number of warmup draws and kept draws."""

    if 'warmup2' not in fit:
        n_warmup = 0
        n_draws = fit['n_save'][0]
    else:
        n_warmup = fit['warmup2'][0]
        n_draws = fit['n_save'][0] - fit['warmup2'][0]

    return n_warmup, n_draws


def _get_log_likelihood_values(fit, n_draws, is_warmup=False):
    """Get values of log-likelihood."""

    n_chains = len(fit['samples'])

    if 'log_likelihood' in fit['samples'][0]['chains']:
        log_likelihood = {'log_likelihood': np.empty((n_chains, n_draws))}
        for i in range(n_chains):
            if is_warmup:
                log_likelihood['log_likelihood'][i] = \
                    fit['samples'][i]['chains']['log_likelihood'][:n_draws]
            else:
                log_likelihood['log_likelihood'][i] = \
                    fit['samples'][i]['chains']['log_likelihood'][-n_draws:]
    else:
        log_likelihood = None

    return log_likelihood


def get_log_likelihood_data(fit):
    """Get log-likelihood group from fit for writing to file."""

    if fit is None:
        return None

    n_warmup, n_draws = _get_number_of_warmup_and_kept_draws_from_fit(fit)

    if n_warmup > 0:
        warmup_log_likelihood = _get_log_likelihood_values(
            fit, n_warmup, is_warmup=True)
    else:
        warmup_log_likelihood = None

    log_likelihood = _get_log_likelihood_values(
        fit, n_draws, is_warmup=False)

    return log_likelihood, warmup_log_likelihood


def _get_sample_stats_values(fit, n_draws, is_warmup=False):
    """Get values of sample statistics."""

    n_chains = len(fit['samples'])

    accept_stat = np.empty((n_chains, n_draws))
    lp = np.empty((n_chains, n_draws))
    for i in range(n_chains):
        if is_warmup:
            accept_stat[i] = fit['samples'][i]['accept_stat'][:n_draws]
            lp[i] = fit['samples'][i]['chains']['lp__'][:n_draws]
        else:
            accept_stat[i] = fit['samples'][i]['accept_stat'][-n_draws:]
            lp[i] = fit['samples'][i]['chains']['lp__'][-n_draws:]

    sample_stats = {'accept_stat': accept_stat, 'lp': lp}

    if 'log_marginal_likelihood' in fit['samples'][0]['chains']:
        sample_stats['log_marginal_likelihood'] = np.empty(
            (n_chains, n_draws))
        for i in range(n_chains):
            if is_warmup:
                sample_stats['log_marginal_likelihood'][i] = \
                    fit['samples'][i]['chains'][
                        'log_marginal_likelihood'][:n_draws]
            else:
                sample_stats['log_marginal_likelihood'][i] = \
                    fit['samples'][i]['chains'][
                        'log_marginal_likelihood'][-n_draws:]

    return sample_stats


def get_sample_stats_data(fit):
    """Get sample statistics group from fit for writing to file."""

    if fit is None:
        return None

    n_warmup, n_draws = _get_number_of_warmup_and_kept_draws_from_fit(fit)

    if n_warmup > 0:
        warmup_sample_stats = _get_sample_stats_values(
            fit, n_warmup, is_warmup=True)
    else:
        warmup_sample_stats = None

    sample_stats = _get_sample_stats_values(fit, n_draws, is_warmup=False)

    return sample_stats, warmup_sample_stats


def _get_sampled_parameter_values(fit, n_draws, is_warmup=False):
    """Get sampled parameter values."""

    excluded_vars = ('lp__', 'log_marginal_likelihood')

    n_chains = len(fit['samples'])

    draws = {}
    for p in fit['samples'][0]['chains']:
        if p in excluded_vars:
            continue

        if fit['samples'][0]['chains'][p].ndim > 1:
            p_shape = fit['samples'][0]['chains'][p][0].shape
        else:
            p_shape = ()

        draws[p] = np.empty((n_chains, n_draws) + p_shape)

    for i in range(n_chains):
        for p in draws:
            if is_warmup:
                draws[p][i] = fit['samples'][i]['chains'][p][:n_draws]
            else:
                draws[p][i] = fit['samples'][i]['chains'][p][-n_draws:]

    return draws


def get_sampled_parameters_data(fit, coords=None, dims=None):
    """Get posterior group data from fit."""

    if fit is None:
        return None

    n_warmup, n_draws = _get_number_of_warmup_and_kept_draws_from_fit(fit)

    if n_warmup > 0:
        warmup_draws = _get_sampled_parameter_values(
            fit, n_warmup, is_warmup=True)
    else:
        warmup_draws = None

    draws = _get_sampled_parameter_values(fit, n_draws, is_warmup=False)

    if coords is None:
        coords = {}

    if dims is None:
        dims = {}

    for p in draws:

        if draws[p].ndim > 2:
            extra_dims = draws[p].shape[2:]
            n_extra_dims = len(extra_dims)

            if p not in coords:
                for i, d in enumerate(extra_dims):
                    coords['{}_dim_{:d}'.format(p, i)] = np.arange(d)

            if p not in dims:
                dims[p] = ['{}_dim_{:d}'.format(p, i)
                           for i in range(n_extra_dims)]

    return draws, warmup_draws, coords, dims


def get_observed_data(data, names=None, dims=None, coords=None):
    """Get observed data group."""

    if data is None:
        return None

    observed_data = {}

    if names is None:
        names = [n for n in data]

    if dims is None:
        dims = {}

    if coords is None:
        coords = {}

    for n in names:
        observed_data[n] = data[n]

        if np.ndim(data[n]) > 0:

            data_dims = np.shape(data[n])
            n_dims = len(data_dims)

            if n not in coords:
                for i, d in enumerate(data_dims):
                    coords['{}_dim_{:d}'.format(n, i)] = np.arange(d)

            if n not in dims:
                dims[n] = ['{}_dim_{:d}'.format(n, i)
                           for i in range(n_dims)]

    return observed_data, coords, dims


def convert_samples_dict_to_inference_data(posterior=None, prior=None,
                                           posterior_predictive=None,
                                           prior_predictive=None,
                                           observed_data=None,
                                           constant_data=None,
                                           coords=None,
                                           dims=None,
                                           observed_data_names=None,
                                           constant_data_names=None,
                                           save_warmup=False):
    """Convert a dictionary containing sampling results to InferenceData."""

    warmup_posterior = None
    sample_stats = None
    warmup_sample_stats = None
    log_likelihood = None
    warmup_log_likelihood = None

    if posterior is not None:
        sample_stats, warmup_sample_stats = get_sample_stats_data(posterior)
        log_likelihood, warmup_log_likelihood = get_log_likelihood_data(
            posterior)
        posterior, warmup_posterior, coords, dims = \
            get_sampled_parameters_data(
                posterior, coords=coords, dims=dims)

    warmup_posterior_predictive = None
    if posterior_predictive is not None:
        posterior_predictive, warmup_posterior_predictive, coords, dims = \
            get_sampled_parameters_data(
                posterior_predictive, coords=coords, dims=dims)

    if prior is not None:
        prior, _, coords, dims = get_sampled_parameters_data(
            prior, coords=coords, dims=dims)

    if prior_predictive is not None:
        prior_predictive, _, coords, dims = \
            get_sampled_parameters_data(
                prior_predictive, coords=coords, dims=dims)

    if observed_data is not None:
        observed_data, coords, dims = get_observed_data(
            observed_data, names=observed_data_names,
            coords=coords, dims=dims)

    if constant_data is not None:
        constant_data, coords, dims = get_observed_data(
            constant_data, names=constant_data_names,
            coords=coords, dims=dims)

    samples = az.from_dict(
        posterior=posterior,
        warmup_posterior=warmup_posterior,
        warmup_posterior_predictive=warmup_posterior_predictive,
        posterior_predictive=posterior_predictive,
        prior=prior,
        prior_predictive=prior_predictive,
        log_likelihood=log_likelihood,
        warmup_log_likelihood=warmup_log_likelihood,
        sample_stats=sample_stats,
        observed_data=observed_data,
        constant_data=constant_data,
        coords=coords, dims=dims, save_warmup=save_warmup)

    return samples


def write_stepwise_mc3_samples(sample_file, **kwargs):
    """Write MC3 samples to file."""

    samples = convert_samples_dict_to_inference_data(
        **kwargs)

    samples.to_netcdf(sample_file)
