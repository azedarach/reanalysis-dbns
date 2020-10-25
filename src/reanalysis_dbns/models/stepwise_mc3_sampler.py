"""
Provides stepwise MC3 sampler.
"""

# License: MIT

from __future__ import absolute_import, division, print_function

import collections
import time

import numpy as np
import scipy.special as sp

from joblib import Parallel, delayed
from sklearn.utils import check_random_state

import reanalysis_dbns.utils as rdu

from .sampler_helpers import (check_max_nonzero_indicators,
                              get_indicator_set_neighborhood)


def _initialize_stepwise_mc3_random(n_indicators, n_chains=1,
                                    max_nonzero=None, random_state=None):
    """Get initial state for MC3 sampler."""

    rng = check_random_state(random_state)

    initial_k = np.zeros((n_chains, n_indicators), dtype=int)

    max_nonzero = check_max_nonzero_indicators(
        max_nonzero, n_indicators=n_indicators)

    # Draw initial set of terms uniformly from possible sets of
    # terms.
    n_possible_sets = np.sum([sp.comb(n_indicators, k)
                              for k in range(max_nonzero + 1)])
    weights = np.array([sp.comb(n_indicators, k) / n_possible_sets
                        for k in range(max_nonzero + 1)])

    for i in range(n_chains):

        indicator_set_size = rng.choice(max_nonzero + 1, p=weights)
        indicator_indices = rng.choice(
            n_indicators, size=indicator_set_size, replace=False)

        initial_k[i, indicator_indices] = 1

    return initial_k


def initialize_stepwise_mc3(n_indicators, method='random', **kwargs):
    """Draw initial values for indicator set in MC3 sampling."""

    if method == 'random':
        return _initialize_stepwise_mc3_random(n_indicators, **kwargs)

    raise ValueError(
        "Unrecognized initialization method '%r'" % method)


def stepwise_mc3_propose_model(k, max_nonzero=None, allow_exchanges=True,
                               random_state=None):
    """Propose new predictor set from neighborhood of current set.

    Parameters
    ----------
    k : array-like, shape (n_indicators,)
        Array of indicator variables defining the current model.

    max_nonzero : int, optional
        If given, the maximum number of non-zero indicators allowed.

    allow_exchanges : bool, default: True
        If True, include an exchange move in the set of possible moves.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    Returns
    -------
    next_k : array, shape (n_indicators,)
        Array of indicator variables defining the proposed model.

    log_proposal_ratio : float
        The value of the log of the ratio of the proposal densities.
    """

    rng = check_random_state(random_state)

    if max_nonzero is not None:
        if not rdu.is_integer(max_nonzero) or max_nonzero < 0:
            raise ValueError(
                'Maximum number of non-zero indicators must be a '
                'non-negative integer')

    nhd = get_indicator_set_neighborhood(
        k, allow_exchanges=allow_exchanges,
        max_nonzero=max_nonzero, include_current_set=True)
    current_nhd_size = len(nhd)

    move = rng.choice(nhd)

    next_k = k.copy()

    for i in move['decrement']:
        next_k[i] = 0

    for i in move['increment']:
        next_k[i] = 1

    next_nhd = get_indicator_set_neighborhood(
        next_k, allow_exchanges=allow_exchanges,
        max_nonzero=max_nonzero, include_current_set=True)
    next_nhd_size = len(next_nhd)

    log_proposal_ratio = np.log(current_nhd_size) - np.log(next_nhd_size)

    return next_k, log_proposal_ratio


def _stepwise_mc3_step(k, lp, logp, *logp_args, data=None,
                       max_nonzero=None, allow_exchanges=True,
                       random_state=None, logp_kwargs=None):
    """Perform single step of stepwise MC3 sampler.

    Parameters
    ----------
    k : array-like, shape (n_indicators,)
        Array of indicator variables defining the current model.

    lp : float
        The value of the (possibly non-normalized) log-posterior
        density for the current model.

    logp : callable
        A function callable with the signature
        logp(k, *logp_args, data=data, **logp_kwargs)
        evaluating the value of the log-posterior density for the model.

    *logp_args :
        Additional positional parameters for the log-posterior density
        function.

    data : dict-like
        Dictionary containing data used for sampling.

    max_nonzero : int, optional
        If given, the maximum number of non-zero indicators allowed.

    allow_exchanges : bool, default: True
        If True, include an exchange move in the set of possible moves.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    logp_kwargs : dict
        Additional keyword arguments for log-posterior density function.

    Returns
    -------
    next_k : array, shape (n_indicators,)
        Array of indicator variables defining the next model.

    next_lp : float
        The value of the log-posterior density for the next model.

    accepted : bool
        Flag indicating whether the proposed move was accepted or not.
    """

    rng = check_random_state(random_state)

    next_k, log_proposal_ratio = stepwise_mc3_propose_model(
        k, max_nonzero=max_nonzero, allow_exchanges=allow_exchanges,
        random_state=rng)

    if logp_kwargs is None:
        logp_kwargs = {}

    next_lp = logp(next_k, *logp_args, data=data, **logp_kwargs)

    log_acceptance = min(0.0, next_lp - lp + log_proposal_ratio)

    u = rng.uniform()

    if np.log(u) <= log_acceptance:
        accepted = True
    else:
        accepted = False

        next_k = k.copy()
        next_lp = lp

    return next_k, next_lp, accepted


def _stepwise_mc3(initial_k, logp, *logp_args, data=None, n_iter=1000,
                  thin=1, verbose=False, max_nonzero=None,
                  allow_exchanges=True, random_state=None, logp_kwargs=None):
    """Run a single chain of the stepwise MC3 sampler.

    Parameters
    ----------
    initial_k : array-like, shape (n_indicators,)
        Array of indicator variables defining the initial model.

    logp : callable
        A function callable with the signature
        logp(k, *logp_args, data=data, **logp_kwargs)
        evaluating the value of the log-posterior density for the model.

    *logp_args :
        Additional positional parameters for the log-posterior density
        function.

    data : dict-like
        Dictionary containing data used for sampling.

    n_iter : int, default: 1000
        Number of iterations.

    thin : int, default: 1
        Interval used for thinning samples.

    verbose : bool, default: False
        If True, produce verbose output.

    max_nonzero : int, optional
        If given, the maximum number of non-zero indicators allowed.

    allow_exchanges : bool, default: True
        If True, include an exchange move in the set of possible moves.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    logp_kwargs : dict
        Additional keyword arguments for log-posterior density function.

    Returns
    -------
    results : dict
        Dictionary containing the sampler results.
    """

    rng = check_random_state(random_state)

    n_iter = rdu.check_number_of_iterations(n_iter)

    if max_nonzero is not None:
        if not rdu.is_integer(max_nonzero) or max_nonzero < 0:
            raise ValueError(
                'Maximum number of non-zero indicators must be a '
                'non-negative integer')

    if logp_kwargs is None:
        logp_kwargs = {}

    k = np.empty((n_iter, initial_k.shape[0]))
    lp = np.empty((n_iter,))
    accept_stat = np.empty((n_iter,))

    k[0] = initial_k.copy()
    lp[0] = logp(initial_k, *logp_args, data=data, **logp_kwargs)
    accept_stat[0] = 1.0

    if verbose:
        header = '{:<12s} | {:<13s} | {:<12s}'.format(
            'Iteration', 'Acceptance', 'Time')
        sep = len(header) * '-'

        print(header)
        print(sep)

    n_accepted = 1
    for i in range(1, n_iter):

        start_time = time.perf_counter()

        k[i], lp[i], accepted = \
            _stepwise_mc3_step(
                k[i - 1], lp[i - 1],
                logp, *logp_args, data=data,
                max_nonzero=max_nonzero,
                allow_exchanges=allow_exchanges,
                random_state=rng, logp_kwargs=logp_kwargs)

        if accepted:
            n_accepted += 1

        accept_stat[i] = n_accepted / (i + 1)

        if verbose:
            print('{:12d} | {: 12.6e} | {: 12.6e}'.format(
                i + 1, n_accepted / (i + 1), time.perf_counter() - start_time))

    chains = collections.OrderedDict(
        {'k': k[::thin], 'lp__': lp[::thin]})

    args = {'random_state': random_state, 'allow_exchanges': allow_exchanges,
            'n_iter': n_iter, 'thin': thin, 'max_nonzero': max_nonzero}

    results = {'chains': chains,
               'args': args,
               'acceptance': n_accepted / n_iter,
               'accept_stat': accept_stat,
               'mean_lp__': np.mean(chains['lp__'])}

    return results


def optimize_stepwise_mc3(initial_k, logp, *logp_args, data=None,
                          max_nonzero=None, allow_exchanges=True,
                          tol=1e-4, n_iter=2000,
                          random_state=None, logp_kwargs=None):
    """Find local maximum of log-posterior density via greedy search.

    Parameters
    ----------
    initial_k : array-like, shape (n_indicators,)
        Array of indicator variables defining the initial model.

    logp : callable
        A function callable with the signature
        logp(k, *logp_args, data=data, **logp_kwargs)
        evaluating the value of the log-posterior density for the model.

    *logp_args :
        Additional positional parameters for the log-posterior density
        function.

    data : dict-like
        Dictionary containing data used for sampling.

    max_nonzero : int, optional
        If given, the maximum number of non-zero indicators allowed.

    allow_exchanges : bool, default: True
        If True, include an exchange move in the set of possible moves.

    tol : float, default: 1e-4
        Tolerance used for detecting convergence.

    n_iter : int, default: 1000
        Maximum number of iterations.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    logp_kwargs : dict
        Additional keyword arguments for log-posterior density function.

    Returns
    -------
    k_max : array, shape (n_indicators,)
        Array of indicator variables corresponding to the found local
        maximum.

    lp_max : float
        Value of the log-posterior density at the local maximum.
    """

    rng = check_random_state(random_state)

    if max_nonzero is not None:
        if not rdu.is_integer(max_nonzero) or max_nonzero < 0:
            raise ValueError(
                'Maximum number of non-zero indicators must be a '
                'non-negative integer')

    n_iter = rdu.check_number_of_iterations(n_iter)
    tol = rdu.check_tolerance(tol)

    if logp_kwargs is None:
        logp_kwargs = {}

    max_k = initial_k.copy()
    max_lp = logp(initial_k, *logp_args, data=data, **logp_kwargs)

    cached_k = {tuple(max_k): max_lp}
    for i in range(n_iter):

        next_k, _ = stepwise_mc3_propose_model(
            max_k, max_nonzero=max_nonzero, allow_exchanges=allow_exchanges,
            random_state=rng)

        if tuple(next_k) in cached_k:
            next_lp = cached_k[tuple(next_k)]
        else:
            next_lp = logp(next_k, *logp_args, data=data, **logp_kwargs)
            cached_k[tuple(next_k)] = next_lp

        if next_lp > max_lp:
            has_converged = np.abs((next_lp - max_lp) / max_lp) < tol

            max_k = next_k.copy()
            max_lp = next_lp

            if has_converged:
                break

    return max_k, max_lp


def sample_stepwise_mc3(initial_k, logp, *logp_args, data=None,
                        n_chains=4, n_iter=1000, thin=1,
                        warmup=None, verbose=False, n_jobs=-1,
                        max_nonzero=None, allow_exchanges=True,
                        random_state=None, logp_kwargs=None):
    """Sample models using stepwise MC3.

    Parameters
    ----------
    initial_k : array-like, shape (n_indicators,)
        Array of indicator variables defining the initial model.

    logp : callable
        A function callable with the signature
        logp(k, *logp_args, data=data, **logp_kwargs)
        evaluating the value of the log-posterior density for the model.

    *logp_args :
        Additional positional parameters for the log-posterior density
        function.

    data : dict-like
        Dictionary containing data used for sampling.

    n_chains : int, default: 4
        Number of separate chains to run.

    n_iter : int, default: 1000
        Number of iterations.

    thin : int, default: 1
        Interval used for thinning samples.

    warmup : int, optional
        Number of warm-up samples.

    verbose : bool, default: False
        If True, produce verbose output.

    n_jobs : int, default: -1
        Number of parallel jobs to run.

    max_nonzero : int, optional
        If given, the maximum number of non-zero indicators allowed.

    allow_exchanges : bool, default: True
        If True, include an exchange move in the set of possible moves.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    logp_kwargs : dict
        Additional keyword arguments for log-posterior density function.

    Returns
    -------
    results : dict
        Dictionary containing the sampler results.
    """

    rng = check_random_state(random_state)

    if max_nonzero is not None:
        if not rdu.is_integer(max_nonzero) or max_nonzero < 0:
            raise ValueError(
                'Maximum number of non-zero indicators must be a '
                'non-negative integer')

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

    if initial_k.ndim == 1:
        # If initial set of indicator variables is one-dimensional,
        # pass the same initial state to all chains.
        initial_k = np.tile(initial_k, (n_chains, 1))

    if initial_k.shape[0] != n_chains:
        raise ValueError(
            'Number of initial predictor sets does not match number of chains')

    random_seeds = rng.choice(1000000 * n_chains,
                              size=n_chains, replace=False)

    def _sample(i, seed):
        return _stepwise_mc3(
            initial_k[i], logp, *logp_args,
            data=data,
            n_iter=n_iter, thin=thin, verbose=verbose,
            max_nonzero=max_nonzero,
            allow_exchanges=allow_exchanges, random_state=seed,
            logp_kwargs=logp_kwargs)

    samples = Parallel(n_jobs=n_jobs)(
        delayed(_sample)(i, seed) for i, seed in enumerate(random_seeds))

    warmup2 = warmup // thin
    n_save = 1 + (n_iter - 1) // thin
    n_kept = n_save - warmup2
    perm_lst = [rng.permutation(int(n_kept)) for _ in range(n_chains)]

    fit = {'samples': samples,
           'n_chains': len(samples),
           'n_iter': n_iter,
           'warmup': warmup,
           'thin': thin,
           'n_save': [n_save] * n_chains,
           'warmup2': [warmup2] * n_chains,
           'max_nonzero': max_nonzero,
           'permutation': perm_lst,
           'random_seeds': random_seeds}

    return fit
