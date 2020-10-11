"""
Provides routines for calculating sampler diagnostics.
"""

# License: MIT

from __future__ import absolute_import, division

import warnings

import arviz as az
import numpy as np
import scipy.linalg as sl
import scipy.sparse as sa
import scipy.special as sp
import scipy.stats as ss

from sklearn.utils import check_array, check_random_state


def _check_model_indicator_values(k, model_indicators=None):
    """Check model indicator values are valid."""

    k = check_array(k, dtype=None)

    unique_indicators = np.unique(k)

    if model_indicators is not None:

        for ki in unique_indicators:

            if ki not in model_indicators:
                raise ValueError(
                    "Invalid indicator value '%r'" % ki)

    else:

        model_indicators = unique_indicators

    return k, model_indicators


def _check_model_indicators_input(k, split=False, model_indicators=None):
    """Check model indicators input is valid."""

    k, model_indicators = _check_model_indicator_values(
        k, model_indicators=model_indicators)

    n_iter = k.shape[1]

    if split:
        kept_indicators = k
    else:
        if n_iter % 2 == 0:
            kept_indicators = k[:, (n_iter // 2):]
        else:
            kept_indicators = k[:, ((n_iter - 1) // 2):]

    for ki in model_indicators:

        if ki not in kept_indicators:
            warnings.warn(
                "Model %r does not occur in sampled models "
                "and will be ignored" % ki, UserWarning)

    return k, model_indicators


def _check_parameters_input(theta, n_chains, n_iter):
    """Check parameters input is valid."""

    theta = check_array(theta, ensure_2d=False, allow_nd=True)

    if theta.shape[0] != n_chains:
        raise ValueError(
            'Number of parameter chains does not match number of chains '
            '(got theta.shape[0]=%d but n_chains=%d)' %
            (theta.shape[0], n_chains))

    if theta.shape[1] != n_iter:
        raise ValueError(
            'Number of parameter sweeps does not match number of sweeps '
            '(got theta.shape[1]=%d but n_iter=%d)' %
            (theta.shape[1], n_iter))

    if theta.ndim == 2:
        theta = np.expand_dims(theta, -1)

    return theta


def _standard_rhat(theta, split=True):
    """Calculate ordinary PSRF."""

    if split:
        method = 'split'
    else:
        method = 'rank'

    ordinary_rhat = az.rhat(
        az.convert_to_dataset(theta), method=method)['x'].data

    return {'PSRF1': ordinary_rhat}


def _get_samples_to_keep(k, theta, split=True):
    """Get samples to keep in PSRF calculation."""

    n_chains, n_iter = k.shape
    n_parameters = theta.shape[2]

    if split:
        # If calculating split diagnostics, treat the first half of
        # each chain as a separate chain.
        if n_iter % 2 == 0:
            k = np.reshape(k, (2 * n_chains, n_iter // 2))
            theta = np.reshape(
                theta, (2 * n_chains, n_iter // 2, n_parameters))
        else:
            k = np.reshape(k[:, 1:], (2 * n_chains, (n_iter - 1) // 2))
            theta = np.reshape(
                theta[:, 1:, ...],
                (2 * n_chains, (n_iter - 1) // 2, n_parameters))

    else:
        # Otherwise, discard the first half of each chain.
        n_warmup = int(n_iter // 2)

        k = k[:, n_warmup:]
        theta = theta[:, n_warmup:, ...]

    return k, theta


def _calculate_variation_within_models(k, theta, model_indicators):
    """Calculate variation in parameters within models."""

    n_chains, n_iter = k.shape
    n_parameters = theta.shape[0]
    n_models = len(model_indicators)

    Wm = np.zeros((n_models, n_parameters, n_parameters))
    for i, m in enumerate(model_indicators):

        mask = (k == m).ravel()

        if not np.any(mask):
            warnings.warn("Model %r does not occur in samples" % m,
                          UserWarning)
            continue

        theta_m = theta.reshape((n_parameters, n_chains * n_iter))[:, mask]
        theta_bar_m = np.mean(theta_m, keepdims=True, axis=1)

        resid = theta_m - theta_bar_m
        Wm[i] = np.sum(np.einsum('ik,jk->kij', resid, resid), axis=0)

    Wm = np.sum(Wm, axis=0) / (n_chains * n_iter - n_models)

    return Wm


def _calculate_variation_within_chains_and_models(k, theta, model_indicators):
    """Calculate variation within chains and models."""

    n_chains, n_iter = k.shape
    n_parameters = theta.shape[0]
    n_models = len(model_indicators)

    WmWc = np.zeros((n_models, n_chains, n_parameters, n_parameters))
    for i, m in enumerate(model_indicators):

        mask = k == m

        if not np.any(mask):
            warnings.warn("Model '%r' does not occur in samples" % m,
                          UserWarning)
            continue

        for c in range(n_chains):

            theta_cm = theta[:, c, mask[c]]
            theta_bar_cm = np.mean(theta_cm, axis=1, keepdims=True)

            resid = theta_cm - theta_bar_cm
            WmWc[i, c] = np.sum(np.einsum('ik,jk->kij', resid, resid), axis=0)

    WmWc = (np.sum(np.sum(WmWc, axis=1), axis=0) /
            (n_chains * (n_iter - n_models)))

    return WmWc


def rjmcmc_rhat(k, theta, model_indicators=None, split=True):
    """Calculate PSRF diagnostics for RJMCMC output.

    Parameters
    ----------
    k : array-like, shape (n_chains, n_iter)
        Array containing model indicator variables.

    theta : array-like, shape (n_chains, n_iter, n_parameters)
        Array containing shared parameters across models.

    model_indicators : array-like, shape (n_models,)
        Array containing the list of all possible model indicators.

    split : bool, default: True
        If True, retain the first half of each chain and treat it as an
        additional chain. Otherwise, the first half of each chain is
        discarded.

    Returns
    -------
    results : dict
        Dictionary containing the univariate and, if applicable, multivariate
        convergence diagnostics.

    References
    ----------
    Castelloe, J. M., and Zimmerman, D. L., "Convergence Assessment
    for Reversible Jump MCMC Samplers", Department of Statistics and
    Actuarial Science, University of Iowa, Technical Report 313 (2002)
    """

    k, model_indicators = _check_model_indicators_input(
        k, split=split, model_indicators=model_indicators)

    n_chains, n_iter = k.shape

    theta = _check_parameters_input(theta, n_chains=n_chains, n_iter=n_iter)

    n_parameters = theta.shape[2]

    n_models = len(model_indicators)
    if n_models == 1:
        return _standard_rhat(theta, split=split)

    # Get retained samples by either splitting chains or dropping
    # first half of each chain.
    k, theta = _get_samples_to_keep(k, theta, split=split)

    # Update values for number of chains and number of sweeps for use in
    # averages.
    n_chains, n_iter = k.shape

    # Check if any models initially present are no longer visited in
    # the restricted sample.
    kept_models = []
    present_models = np.unique(k)
    for m in model_indicators:
        if m not in present_models:
            warnings.warn(
                "Model %r does not occur in sampled models "
                "and will be ignored" % m, UserWarning)
        else:
            kept_models.append(m)

    model_indicators = np.array(kept_models, dtype=k.dtype)
    n_models = len(model_indicators)

    # Reshape theta so that parameter index is leading axis.
    theta = theta.transpose((2, 0, 1))

    # Compute total variation across chains and models.
    Vhat = np.atleast_2d(
        np.cov(theta.reshape((n_parameters, n_chains * n_iter)), ddof=1))

    # Compute variation within chains.
    Wc = np.zeros((n_chains, n_parameters, n_parameters))
    for c in range(n_chains):
        Wc[c] = np.atleast_2d(np.cov(theta[:, c, :], ddof=1))
    Wc = np.mean(Wc, axis=0)

    # Compute variation within models.
    Wm = _calculate_variation_within_models(k, theta, model_indicators)

    # Compute variation within chains and models.
    WmWc = _calculate_variation_within_chains_and_models(
        k, theta, model_indicators)

    results = {'Vhat': Vhat, 'Wc': Wc, 'Wm': Wm, 'WmWc': WmWc}

    # Calculate univariate PSRF diagnostics
    results['PSRF1'] = (Vhat.flat[::n_parameters + 1] /
                        Wc.flat[::n_parameters + 1])
    results['PSRF2'] = (Wm.flat[::n_parameters + 1] /
                        WmWc.flat[::n_parameters + 1])

    if n_parameters > 1:
        # Calculate multivariate PSRF diagnostics
        try:
            results['MPSRF1'] = np.max(sl.eigvalsh(Vhat, Wc))
            results['MPSRF2'] = np.max(sl.eigvalsh(Wm, WmWc))
        except sl.LinAlgError:
            warnings.warn(
                'Calculation of MPSRF failed. Try increasing the number '
                'of samples.',
                UserWarning)
            results['MPSRF1'] = np.NaN
            results['MPSRF2'] = np.NaN
    else:
        results['Vhat'] = results['Vhat'][0]
        results['Wc'] = results['Wc'][0]
        results['Wm'] = results['Wm'][0]
        results['WmWc'] = results['WmWc'][0]

    return results


def rjmcmc_batch_rhat(k, theta, batch_size=None, model_indicators=None,
                      split=True):
    """Calculate batched PSRF diagnostics for RJMCMC output.

    Note, warmup samples are assumed to have already
    been discarded.

    Parameters
    ----------
    k : array-like, shape (n_chains, n_iter)
        Array containing model indicator variables.

    theta : array-like, shape (n_chains, n_iter, n_parameters)
        Array containing shared parameters across models.

    batch_size : integer, optional
        Batch size used for computing convergence diagnostics.

    Returns
    -------
    results : dict
        Dictionary containing the values of the convergence diagnostics
        for each batch of samples.

    References
    ----------
    Castelloe, J. M., and Zimmerman, D. L., "Convergence Assessment
    for Reversible Jump MCMC Samplers", Department of Statistics and
    Actuarial Science, University of Iowa, Technical Report 313 (2002)
    """

    k, model_indicators = _check_model_indicators_input(
        k, model_indicators=model_indicators)

    n_chains, n_iter = k.shape

    theta = _check_parameters_input(theta, n_chains=n_chains, n_iter=n_iter)

    n_parameters = theta.shape[2]

    if batch_size is None:
        batch_size = int((n_iter // 2) // 20)

    n_batches = int(np.ceil((n_iter // 2) / batch_size))

    results = {'batch_size': batch_size,
               'PSRF1': np.empty((n_batches, n_parameters)),
               'PSRF2': np.empty((n_batches, n_parameters)),
               'Vhat': np.empty((n_batches, n_parameters, n_parameters)),
               'Wc': np.empty((n_batches, n_parameters, n_parameters)),
               'Wm': np.empty((n_batches, n_parameters, n_parameters)),
               'WmWc': np.empty((n_batches, n_parameters, n_parameters))}

    if n_parameters > 1:
        results['MPSRF1'] = np.empty((n_batches,))
        results['MPSRF2'] = np.empty((n_batches,))

    for q in range(1, n_batches + 1):

        batch_stop = min(n_iter, 2 * q * batch_size)

        k_batch = k[:, :batch_stop]
        theta_batch = theta[:, :batch_stop, ...]

        batch_rhat = rjmcmc_rhat(
            k_batch, theta_batch, model_indicators=model_indicators,
            split=split)

        for d in batch_rhat:
            results[d][q - 1] = batch_rhat[d]

    return results


def _invert_digamma(y, max_iterations=1000, tolerance=1e-10):
    """Compute inverse of digamma function.

    Parameters
    ----------
    y : float
        Value of digamma function, y = Psi(x)

    max_iterations : int, default: 100
        Maximum number of Newton iterations.

    tolerance : float, default: 1e-12
        Tolerance on the absolute value of the difference between iterates
        required for convergence.

    Returns
    -------
    x : float
        Approximate solution to Psi(x) = y on the interval x > 0.

    References
    ----------
    Minka, T., "Estimating a Dirichlet Distribution", Technical Report, MIT
    (2000).
    """

    threshold = -2.22

    scalar_input = np.isscalar(y)

    y = np.atleast_1d(y)

    x = np.exp(y) + 0.5
    x[y < threshold] = -1.0 / (y[y < threshold] - sp.digamma(1))

    delta = np.full(x.shape, np.inf)
    converged = False
    for n_iter in range(max_iterations):

        psi_x = sp.digamma(x)
        psip_x = sp.polygamma(1, x)

        incr = (y - psi_x) / psip_x

        non_converged = delta >= tolerance
        x[non_converged] = x[non_converged] + incr[non_converged]

        delta[non_converged] = np.abs(incr[non_converged])

        if np.all(delta < tolerance):
            converged = True
            break

    if not converged:
        warnings.warn(
            'Inverting digamma function failed (achieved tolerance: %.4e, '
            'requested tolerance: %.4e)' % (np.max(delta), tolerance),
            RuntimeWarning)

    if scalar_input:
        x = x[0]

    return x


def _estimate_dirichlet_shape_parameters(p, tolerance=1e-6,
                                         max_iterations=5000,
                                         min_alpha=0.01, min_p=1e-4,
                                         omit_nan=True):
    """Estimate the shape parameters of a Dirichlet distribution.

    Parameters
    ----------
    p : array-like, shape (n_samples, k)
        Array containing samples from a Dirichlet distribution of dimension k.

    tolerance : float, default: 1e-4
        Stopping tolerance.

    max_iterations : int, default: 1000
        Maximum number of iterations.

    Returns
    -------
    alpha : array, shape (k,)
        Maximum likelihood estimate for the shape parameters.

    References
    ----------
    Minka, T., "Estimating a Dirichlet Distribution", Technical Report, MIT
    (2000).
    """

    p = check_array(p, force_all_finite=(not omit_nan))

    if omit_nan:
        valid_samples = np.all(np.isfinite(p), axis=1)
        p = p[valid_samples, :]

    # guard against exactly zero elements
    p_safety = np.zeros_like(p)
    p_safety[p == 0] = min_p
    p = p + p_safety

    p_mean = np.mean(p, axis=0)
    p2_mean = np.mean(p**2, axis=0)
    log_pbar = np.mean(np.log(p), axis=0)

    alpha_0 = p_mean * (p_mean - p2_mean) / (p2_mean - p_mean**2)

    alpha_old = np.fmax(min_alpha, alpha_0)
    alpha = np.zeros_like(alpha_old)
    converged = False
    for n_iter in range(max_iterations):

        psi_alpha = sp.digamma(np.sum(alpha_old))
        alpha = _invert_digamma(psi_alpha + log_pbar)

        delta = np.abs(alpha - alpha_old)
        if np.all(delta < tolerance):
            converged = True
            break

        alpha_old[:] = alpha

    if not converged:
        warnings.warn(
            'Solving for Dirichlet shape parameter did not converge '
            '(achieved tolerance: %.4e, requested tolerance: %.4e)' %
            (np.max(delta), tolerance), RuntimeWarning)

    return alpha


def _count_model_transitions(k, model_indicators=None, sparse=False):
    """Count transitions between models in sampled chains.

    Parameters
    ----------
    k : array-like, shape (n_chains, n_iter)
        Array containing model indicator variables.

    model_indicators : array-like, shape (n_models,), optional
        Array containing indicators for all possible models. If not given,
        the set of models observed in the collection of samples is assumed
        to contain all possible models.

    sparse : bool, default: False
        Return counts in sparse matrix format.

    Returns
    -------
    n : array-like, shape (n_chains, n_models, n_models)
        Array containing the transition matrices estimated from each chain.
    """

    k, model_indicators = _check_model_indicator_values(
        k, model_indicators=model_indicators)

    if model_indicators is None:
        model_indicators = np.unique(k)
    else:
        model_indicators = np.asarray(model_indicators)

    n_models = len(model_indicators)
    n_chains, n_iter = k.shape

    if sparse:
        n = sa.dok_matrix((n_models, n_models), dtype=np.uint64)
    else:
        n = np.zeros((n_models, n_models), dtype=np.uint64)

    for t in range(1, n_iter):
        for i in range(n_chains):

            k_from = k[i, t - 1]
            k_to = k[i, t]

            row_index = np.where(model_indicators == k_from)[0][0]
            col_index = np.where(model_indicators == k_to)[0][0]

            n[row_index, col_index] += 1

    if sparse:
        n = n.tocsr()

    return n


def _sample_stationary_distributions(k, model_indicators=None, sparse=False,
                                     epsilon=None, min_epsilon=1e-10,
                                     n_samples=100,
                                     tolerance=1e-2, random_state=None):
    """Draw a single Markov transition matrix for the given sample."""

    rng = check_random_state(random_state)

    k, model_indicators = _check_model_indicator_values(
        k, model_indicators=model_indicators)

    n_chains, n_iter = k.shape

    n = _count_model_transitions(
        k, model_indicators=model_indicators, sparse=sparse)

    n_models = n.shape[0]
    n_observed_models = np.size(np.unique(k))

    if epsilon is None:
        mask = np.sum(n, axis=1) == 0
        epsilon = np.full((n_models,), 1.0 / n_observed_models)
        epsilon[mask] = min_epsilon
    elif np.isscalar(epsilon):
        epsilon = np.full((n_models,), epsilon)

    P = np.zeros((n_samples, n_models, n_models))
    for i in range(n_models):
        alpha = n[i, :] + epsilon[i]
        P[:, i, :] = ss.gamma.rvs(alpha, size=(n_samples, n_models),
                                  random_state=rng)
        row_sums = np.sum(P[:, i, :], axis=1)
        mask = row_sums > 0.0
        P[mask, i, :] = P[mask, i, :] / row_sums[mask, np.newaxis]

    if not np.all(np.isfinite(P)):
        raise RuntimeError(
            'Non-finite values in sampled transition matrices. '
            'Try increasing the value of epsilon.')

    pi = np.zeros((n_samples, n_models))
    for r in range(n_samples):
        evals, evecs = sl.eig(P[r], left=True, right=False)
        index = np.argmax(np.real(evals))
        if np.abs(np.real(evals[index]) - 1.0) > tolerance:
            pi[r] = np.full((n_models,), np.NaN)
        else:
            pi[r] = (np.real(evecs[:, index]) /
                     np.sum(np.real(evecs[:, index])))

    return pi, epsilon


def estimate_stationary_distribution(k, model_indicators=None, sparse=False,
                                     epsilon=None, n_samples=100,
                                     min_epsilon=1e-10,
                                     tolerance=1e-6, fit_kwargs=None,
                                     random_state=None):
    """Calculate summary of estimated stationary distribution for models."""

    rng = check_random_state(random_state)

    pi_samples, epsilon = _sample_stationary_distributions(
        k, model_indicators=model_indicators, sparse=sparse,
        epsilon=epsilon, min_epsilon=min_epsilon, n_samples=n_samples,
        tolerance=tolerance, random_state=rng)

    n_observed_models = np.size(np.unique(k))
    n_models = pi_samples.shape[1]

    if fit_kwargs is None:
        fit_kwargs = {}

    alpha_hat = _estimate_dirichlet_shape_parameters(
        pi_samples, **fit_kwargs)

    n_eff = np.sum(alpha_hat) - n_observed_models * np.sum(epsilon)

    return {'pi': pi_samples, 'ess': n_eff, 'n_models': n_models,
            'n_observed_models': n_observed_models, 'epsilon': epsilon}
