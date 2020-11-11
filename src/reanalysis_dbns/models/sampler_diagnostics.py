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

from sklearn.preprocessing import normalize
from sklearn.utils import check_array, check_random_state


def _check_model_indicator_values(k, model_indicators=None,
                                  single_chain=False):
    """Check model indicator values are valid."""

    k = check_array(k, dtype=None, ensure_2d=(not single_chain))

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
                'Model %r does not occur in sampled models' % ki,
                UserWarning)

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


def _convert_to_sparse_format_like(x, target):
    """Convert x to the same sparse format as target."""

    if sa.isspmatrix_csc(target):
        return x.tocsc()

    if sa.isspmatrix_csr(target):
        return x.tocsr()

    if sa.isspmatrix_bsr(target):
        return x.tobsr()

    if sa.isspmatrix_lil(target):
        return x.tolil()

    if sa.isspmatrix_dok(target):
        return x.todok()

    if sa.isspmatrix_coo(target):
        return x.tocoo()

    if sa.isspmatrix_dia(target):
        return x.todia()

    raise ValueError(
        "Unsupported target type '%r'" % type(target))


def _move_all_zero_dimensions_sparse(x, axis=-1):
    """Move dimensions containing only zeros to end of axis."""

    if not sa.issparse(x):
        raise ValueError(
            'Input matrix must be sparse (got type(x)=%r)' % type(x))

    if x.ndim != 2:
        raise NotImplementedError(
            'Axis shuffling only implemented for two-dimensional '
            'sparse matrices (got x.ndim=%d)' % x.ndim)

    axis = axis % 2

    n_rows, n_cols = x.shape

    x_permuted = x.copy().tocoo()

    x_permuted.eliminate_zeros()
    if axis == 0:
        # Find all columns with at least one non-zero element.
        old_inds = x_permuted.col
    else:
        # Find all rows with at least one non-zero element.
        old_inds = x_permuted.row

    # Remove duplicates.
    nonzero_entries = np.unique(old_inds)
    new_inds = np.arange(np.size(nonzero_entries))

    p = np.zeros((old_inds.shape[0], new_inds.shape[0]), dtype=int)
    for i, ind in enumerate(nonzero_entries):
        mask = old_inds == ind
        p[mask, i] = 1

    # Move rows/columns with non-zero elements to front.
    if axis == 0:
        x_permuted.col = np.dot(p, old_inds)
    else:
        x_permuted.row = np.dot(p, old_inds)

    # Restore original storage format.
    return _convert_to_sparse_format_like(x_permuted, x)


def _relabel_unobserved_markov_chain_states_sparse(p):
    """Relabel states in a transition matrix that are not observed."""

    if not sa.issparse(p):
        raise ValueError(
            'Input matrix must be sparse (got type(x)=%r)' % type(p))

    if p.ndim != 2:
        raise ValueError(
            'Input matrix must be two-dimensional '
            '(got x.ndim=%d)' % p.ndim)

    n_states = p.shape[0]

    p_permuted = p.copy().tocoo()
    p_permuted.eliminate_zeros()

    nonzero_transitions = np.unique(p_permuted.row)

    perm = np.argsort(
        np.isin(np.arange(n_states), nonzero_transitions, invert=True))

    def _permute(label):
        return np.nonzero(perm == label)[0][0]

    p_permuted.row = np.array([_permute(i) for i in p_permuted.row])
    p_permuted.col = np.array([_permute(i) for i in p_permuted.col])

    # Restore original storage format.
    return _convert_to_sparse_format_like(p_permuted, p)


def _relabel_unobserved_markov_chain_states(p, tol=1e-10):
    """Relabel states in a transition matrix that are not observed."""

    if sa.issparse(p):
        return _relabel_unobserved_markov_chain_states_sparse(p)

    p = check_array(p)

    mask = np.abs(np.sum(p, axis=1)) < tol

    if not np.any(mask):
        return p

    permutation = np.argsort(mask)

    return p[permutation][:, permutation]


def _standard_rhat(theta, split=True):
    """Calculate ordinary PSRF."""

    if split:
        method = 'split'
    else:
        method = 'rank'

    ordinary_rhat = az.rhat(
        az.convert_to_dataset(theta), method=method)['x'].data

    return {'PSRF1': ordinary_rhat}


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

    Note, warmup samples are assumed to have already
    been discarded.

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

        # Discard first half of each chain in the batch.
        n_warmup = int(batch_stop // 2)

        k_batch = k[:, n_warmup:]
        theta_batch = theta_batch[:, n_warmup:, ...]

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
                                         min_alpha=0.01, min_p=1e-10,
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
    p = np.fmax(p, min_p)

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


def _calculate_model_transition_count_matrix(k, model_indicators=None,
                                             sparse=False):
    """Calculate counts of model transitions for a single chain.

    Parameters
    ----------
    k : array-like, shape (n_iter,)
        Array containing model indicator variables.

    model_indicators : array-like, shape (n_models,), optional
        Array containing indicators for all possible models. If not given,
        the set of models observed in the collection of samples is assumed
        to contain all possible models.

    sparse : bool, default: False
        Return counts in sparse matrix format.

    Returns
    -------
    n : array-like, shape (n_models, n_models)
        Array containing the counts of each possible transition.
    """

    k, model_indicators = _check_model_indicator_values(
        k, model_indicators=model_indicators, single_chain=True)

    if model_indicators is None:
        model_indicators = np.unique(k)
    else:
        model_indicators = np.asarray(model_indicators)

    n_models = len(model_indicators)
    n_iter = k.shape[0]

    if sparse:
        n = sa.dok_matrix((n_models, n_models), dtype=np.uint64)
    else:
        n = np.zeros((n_models, n_models), dtype=np.uint64)

    for t in range(1, n_iter):

        k_from = k[t - 1]
        k_to = k[t]

        row_index = np.where(model_indicators == k_from)[0][0]
        col_index = np.where(model_indicators == k_to)[0][0]

        n[row_index, col_index] += 1

    if sparse:
        n = n.tocsr()

    return n


def _count_model_transitions(k, model_indicators=None, sparse=False,
                             combine_chains=True):
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

    combine_chains : bool, default: True
        If True, sum counts of each transition across all chains.

    Returns
    -------
    n : array, shape (n_chains, n_models, n_models) or (n_models, n_models)
        Array containing the transition matrices estimated from each chain.
    """

    k, model_indicators = _check_model_indicator_values(
        k, model_indicators=model_indicators)

    if model_indicators is None:
        model_indicators = np.unique(k)
    else:
        model_indicators = np.asarray(model_indicators)

    n_chains = k.shape[0]

    transition_matrices = [_calculate_model_transition_count_matrix(
        k[i], model_indicators=model_indicators, sparse=sparse)
        for i in range(n_chains)]

    if combine_chains:
        n = transition_matrices[0]
        for i in range(1, n_chains):
            n += transition_matrices[i]

    else:
        if sparse:
            n = transition_matrices
        else:
            n_models = transition_matrices[0].shape[0]
            n = np.zeros((n_chains, n_models, n_models),
                         dtype=np.uint64)
            for i in range(n_chains):
                n[i, :, :] = transition_matrices[i]

    return n


def _calculate_model_transition_matrix(k, model_indicators=None,
                                       sparse=False,
                                       combine_chains=False):
    """Calculate transition matrices in sampled chains.

    Parameters
    ----------
    k : array-like, shape (n_chains, n_iter)
        Array containing model indicator variables.

    model_indicators : array-like, shape (n_models,), optional
        Array containing indicators for all possible models. If not given,
        the set of models observed in the collection of samples is assumed
        to contain all possible models.

    sparse : bool, default: False
        Return transition matrix in sparse matrix format.

    combine_chains : bool, default: True
        If True, sum counts of each transition across all chains
        before estimating transition probabilities.

    Returns
    -------
    p : array, shape (n_chains, n_models, n_models) or (n_models, n_models)
        Array containing the transition matrices estimated from each chain.
    """

    n = _count_model_transitions(
        k, model_indicators=model_indicators, sparse=sparse,
        combine_chains=combine_chains)

    if combine_chains:
        return normalize(n, norm='l1', axis=1)

    if sparse:
        p = []
        for ni in n:
            p.append(normalize(ni, norm='l1', axis=1))
    else:
        p = np.zeros_like(n)
        n_chains = len(n) if sparse else n.shape[0]
        for i in range(n_chains):
            p[i] = normalize(n[i], norm='l1', axis=1)

    return p


def estimate_convergence_rate(k, model_indicators=None, sparse=False,
                              combine_chains=False):
    """Estimate the convergence rate for transdimensional sampler.

    Parameters
    ----------
    k : array-like, shape (n_chains, n_iter)
        Array containing the values of the model indicator at each
        draw of the MCMC output.

    model_indicators : array-like, shape (n_models,)
        If given, an array or list of the possible model indicators
        (potentially including models that were not visited by the
        chain).

    sparse : bool, default: False
        If True, use sparse matrices for the model and transition counts.

    combine_chains : bool, default: False
        If True, calculate the transition matrix based on samples
        from all chains combined.

    Returns
    -------
    rho : float or array, shape (n_chains,)
        Estimated convergence rate, computed as the second eigenvalue
        of the observed transition matrix.

    References
    ----------
    Brooks, S. P., Giudici, P., and Phillipe, A., "Nonparametric
    Convergence Assessment for MCMC Model Selection", Journal of
    Computational and Graphical Statistics 12:1, 1 - 22 (2003),
    doi:10.1198/1061860031347
    """

    n = _count_model_transitions(
        k, model_indicators=model_indicators, sparse=sparse,
        combine_chains=combine_chains)

    transition_matrices = []
    if combine_chains:
        n = _relabel_unobserved_markov_chain_states(n)
        transition_matrices.append(normalize(n, norm='l1', axis=1))
    else:
        if sparse:
            for ni in n:
                pi = _relabel_unobserved_markov_chain_states(ni)
                transition_matrices.append(normalize(pi, norm='l1', axis=1))
        else:
            n_chains = n.shape[0]
            for i in range(n_chains):
                pi = _relabel_unobserved_markov_chain_states(n[i])
                transition_matrices.append(normalize(pi, norm='l1', axis=1))

    convergence_rates = []
    for p in transition_matrices:
        if sparse:
            evals = sa.linalg.eigs(p, which='LR',
                                   return_eigenvectors=False)
        else:
            evals = sl.eigvals(p)

        if evals.shape[0] < 2:
            raise ValueError(
                'At least two states required for convergence rate '
                'estimation')

        evals_order = np.argsort(np.abs(np.real(evals)))[::-1]
        sorted_evals = evals[evals_order]

        convergence_rates.append(np.real(sorted_evals[1]))

    if len(convergence_rates) == 1:
        convergence_rates = convergence_rates[0]

    return convergence_rates


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
        if sparse:
            mask = np.ravel(mask)
        epsilon = np.full((n_models,), 1.0 / n_observed_models)
        epsilon[mask] = min_epsilon
    elif np.isscalar(epsilon):
        epsilon = np.full((n_models,), epsilon)

    P = np.zeros((n_samples, n_models, n_models))
    for i in range(n_models):
        if sparse:
            alpha = n[i, :].toarray().ravel() + epsilon[i]
        else:
            alpha = n[i, :] + epsilon[i]
        P[:, i, :] = ss.gamma.rvs(alpha, size=(n_samples, n_models),
                                  random_state=rng)
        row_sums = np.sum(P[:, i, :], axis=1)
        mask = row_sums > 0.0
        P[mask, i, :] = P[mask, i, :] / row_sums[mask, np.newaxis]

    assert np.all(np.sum(P, axis=2) - 1.0 < 0.001)

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
    """Calculate summary of estimated stationary distribution for models.

    Parameters
    ----------
    k : array-like, shape (n_chains, n_iter)
        Array containing the values of the model indicator at each
        draw of the MCMC output.

    model_indicators : array-like, shape (n_models,)
        If given, an array or list of the possible model indicators
        (potentially including models that were not visited by the
        chain).

    sparse : bool, default: False
        If True, use sparse matrices for the model and transition counts.

    epsilon : float, optional
        If given, the value of the prior sample size parameter used in
        the Dirichlet prior for the transition matrix.

    n_samples : int, default: 100
        Number of draws from the estimated posterior distribution.

    min_epsilon : float, default: 1e-10
        Minimum value of the Dirichlet prior parameter.

    tolerance : float, default: 1e-6
        Tolerance used for determining if an eigenvalue has unit
        magnitude.

    fit_kwargs : dict
        Keyword arguments to pass when fitting the Dirichlet distribution.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    Returns
    -------
    diagnostics : dict
        Dict containing the results of fitting a Markov model
        to the sequence of transitions between models, with entries:

        - 'pi': the sampled stationary distributions of the
           fitted Markov chain

        - 'ess': the effective sample size

        - 'n_models': the number of possible models

        - 'n_observed_models': the number of models visited by the chain

        - 'epsilon': the value of the parameter used for the Dirichlet prior

    References
    ----------
    Heck, D. W., Overstall, A. M., Gronau, Q. F., and Wagenmakers, E.,
    "Quantifying uncertainty in transdimensional Markov chain Monte Carlo
    using discrete Markov models", Statistics and Computing 29,
    631 - 643 (2019), doi:10.1007/s11222-018-9828-0
    """

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


def _reorder_table_columns(x, new_order):
    """Reorder columns in matrix."""

    if len(new_order) != x.shape[1]:
        raise ValueError(
            'Length of column permutation does not match number '
            'of columns (got len(new_order)=%d but x.shape[1]=%d)' %
            (len(new_order), x.shape[1]))

    if sa.issparse(x):
        new_x = x.tocoo()

        def _permute(i):
            return np.nonzero(new_order == i)[0][0]

        new_x.col = np.array([_permute(i) for i in new_x.col])

        return _convert_to_sparse_format_like(new_x, x)

    return x[:, new_order]


def _merge_table_columns(x, cols_to_merge, dest):
    """Merge given columns and place merged column at given index."""

    if len(cols_to_merge) == 0:
        return x

    n_rows, n_init_cols = x.shape
    n_cols_to_merge = len(cols_to_merge)
    n_final_cols = n_init_cols - n_cols_to_merge + 1

    if dest >= n_final_cols:
        raise IndexError(
            'Target column (%d) out of range' % dest)

    unmerged_cols = np.array([d for d in range(n_init_cols)
                              if d not in cols_to_merge])
    unmerged_col_inds = np.array([d for d in range(n_final_cols)
                                  if d != dest])

    if len(unmerged_cols) == 0:
        merged = x.sum(axis=1)
        if merged.ndim == 1:
            merged = np.reshape(merged, (n_rows, 1))
        return merged

    if sa.issparse(x):

        merged = sa.dok_matrix((n_rows, n_final_cols), dtype=x.dtype)

        nnz = x.nnz
        nonzero_rows, nonzero_cols, nonzero_vals = sa.find(x)
        for i in range(nnz):
            if nonzero_cols[i] in cols_to_merge:
                merged[nonzero_rows[i], dest] += nonzero_vals[i]
            else:
                new_col = unmerged_col_inds[
                    np.nonzero(unmerged_cols == nonzero_cols[i])[0][0]]
                merged[nonzero_rows[i], new_col] = nonzero_vals[i]

        return _convert_to_sparse_format_like(merged, x)

    merged = np.zeros((n_rows, n_final_cols), dtype=x.dtype)

    merged[:, unmerged_col_inds] = x[:, unmerged_cols]
    merged[:, dest] = np.sum(x[:, cols_to_merge], axis=1)

    return merged


def _merge_low_expected_frequency_cells(observed, min_expected_count=5):
    """Merge table cells with expected counts below threshold."""

    # Calculate initial set of expected column counts.
    n_rows = observed.shape[0]
    expected_counts = np.ravel(observed.sum(axis=0)) / n_rows

    if not np.any(expected_counts < min_expected_count):
        return observed

    # Permute initial column ordering so that columns
    # with smallest expected counts are left-most.
    col_order = np.argsort(expected_counts)
    merged = _reorder_table_columns(observed, col_order)
    expected_counts = np.sort(expected_counts)

    still_merging = True
    n_cols = expected_counts.shape[0]
    while still_merging and n_cols > 1:

        # Combine all left-most columns until expected count is at least
        # equal to minimum expected count.
        cumulative_counts = np.cumsum(expected_counts)

        cols_above_expected_count = np.where(
            cumulative_counts >= min_expected_count)[0]

        if np.size(cols_above_expected_count) == 0:
            raise ValueError(
                'Expected counts too small after merging all columns')

        cols_to_merge = np.arange(cols_above_expected_count[0] + 1)

        merged = _merge_table_columns(merged, cols_to_merge, 0)

        # Find expected counts for each column and number of merged
        # columns.
        expected_counts = np.ravel(merged.sum(axis=0)) / n_rows
        n_cols = expected_counts.shape[0]

        # Permute columns so that columns with smallest expected
        # counts are still left-most.
        col_order = np.argsort(expected_counts)
        merged = _reorder_table_columns(merged, col_order)
        expected_counts = np.sort(expected_counts)

        # Check if all columns have expected counts exceeding the
        # minimum required.
        still_merging = np.any(expected_counts < min_expected_count)

    return merged


def rjmcmc_chisq_convergence(k, thin=1, sparse=False,
                             split=True, merge_cells=True,
                             correction=True,
                             use_likelihood_ratio=False,
                             min_expected_count=5):
    """Calculate chi squared convergence diagnostic.

    Note, warmup samples are assumed to have already been
    discarded.

    Parameters
    ----------
    k : array-like, shape (n_chains, n_iter)
        Array containing the values of the model indicator at each
        draw of the MCMC output.

    model_indicators : array-like, shape (n_models,)
        If given, an array or list of the possible model indicators
        (potentially including models that were not visited by the
        chain).

    sparse : bool, default: False
        If True, use sparse matrices for the model and transition counts.

    merge_cells : bool, default: True
        If True, merge cells corresponding to chain x model combinations
        until all cells have an expected frequency above 5.

    split : bool, default: True
        If True, retain the first half of each chain and treat it as an
        additional chain. Otherwise, the first half of each chain is
        discarded.

    Returns
    -------
    test_statistic : float
        Value of the test statistic.

    pval : float
        Value of the p-value for the test statistic under the
        asymptotic sampling distribution.

    References
    ----------
    Brooks, S. P., Giudici, P., and Phillipe, A., "Nonparametric
    Convergence Assessment for MCMC Model Selection", Journal of
    Computational and Graphical Statistics 12:1, 1 - 22 (2003),
    doi:10.1198/1061860031347
    """

    k = check_array(k, dtype=None)

    n_chains, n_iter = k.shape

    if split:
        # If calculating split diagnostics, treat the first half of
        # each chain as a separate chain.
        if n_iter % 2 == 0:
            k = np.reshape(k, (2 * n_chains, n_iter // 2))
        else:
            k = np.reshape(k[:, 1:], (2 * n_chains, (n_iter - 1) // 2))

    # Get number of models visited across all chains.
    visited_models = np.unique(k)
    n_models = len(visited_models)

    # Update values for number of chains and number of sweeps for use in
    # averages.
    n_chains, n_iter = k.shape

    # Construct contingency table of model occurrences within each
    # chain.
    if sparse:
        observed_counts = sa.dok_matrix((n_chains, n_models), dtype=int)
    else:
        observed_counts = np.zeros((n_chains, n_models), dtype=int)

    for i in range(n_chains):
        for j, z in enumerate(visited_models):
            observed_counts[i, j] = np.sum(k[i] == z)

    if merge_cells:
        # If required, merge cells with too small expected frequencies.
        observed_counts = _merge_low_expected_frequency_cells(
            observed_counts, min_expected_count=min_expected_count)

    if sparse:
        observed_counts = observed_counts.toarray()

    # Calculate test statistic and p-value.
    if use_likelihood_ratio:
        test = 'log-likelihood'
    else:
        test = 'pearson'

    test_statistic, pval, _, _ = ss.chi2_contingency(
        observed_counts, correction=correction, lambda_=test)

    return test_statistic, pval


def rjmcmc_batch_chisq_convergence(k, batch_size=None,
                                   sparse=False, thin=1,
                                   split=False, merge_cells=True,
                                   correction=True,
                                   use_likelihood_ratio=False,
                                   min_expected_count=5):
    """Calculate batched convergence diagnostics for RJMCMC output.

    Parameters
    ----------
    k : array-like, shape (n_chains, n_iter)
        Array containing model indicator variables.

    batch_size : integer, optional
        Batch size used for computing convergence diagnostics.

    Returns
    -------
    test_statistic : array, shape (n_batches,)
        Array containing the test statistic for each batch.

    pvals : array, shape (n_batches,)
        Array containing the p-values for each batch.

    samples : array, shape (n_batches,)
        The number of iterations used in each batch.
    """

    k = check_array(k, dtype=None)

    # Perform additional thinning of samples if necessary.
    if thin > 1:
        k = k[:, ::thin]

    n_chains, n_iter = k.shape

    if batch_size is None:
        batch_size = int((n_iter // 2) // 20)

    n_batches = int(np.ceil((n_iter // 2) / batch_size))

    test_statistics = np.zeros((n_batches,))
    pvals = np.zeros((n_batches,))
    samples = np.zeros((n_batches,), dtype=int)

    for q in range(1, n_batches + 1):

        batch_stop = min(n_iter, 2 * q * batch_size)

        k_batch = k[:, :batch_stop]

        # Discard first half of each chain in the batch.
        n_warmup = int(batch_stop // 2)

        k_batch = k_batch[:, n_warmup:]

        samples[q - 1] = batch_stop
        test_statistics[q - 1], pvals[q - 1] = rjmcmc_chisq_convergence(
            k_batch, thin=thin, sparse=sparse, split=split,
            merge_cells=merge_cells, correction=correction,
            use_likelihood_ratio=use_likelihood_ratio,
            min_expected_count=min_expected_count)

    return test_statistics, pvals, samples


def rjmcmc_kstest_convergence(k, split=True, thin=1, mode='auto'):
    """Calculate pairwise KS test based convergence diagnostic.

    Note, warmup samples are assumed to already have been discarded.

    Parameters
    ----------
    k : array-like, shape (n_chains, n_iter)
        Array containing the values of the model indicator at each
        draw of the MCMC output.

    model_indicators : array-like, shape (n_models,)
        If given, an array or list of the possible model indicators
        (potentially including models that were not visited by the
        chain).

    split : bool, default: True
        If True, retain the first half of each chain and treat it as an
        additional chain. Otherwise, the first half of each chain is
        discarded.

    mode : 'auto' | 'exact' | 'asymp'
        Method used for calculating the p-value.

    Returns
    -------
    test_statistics : array, shape (n_chains * (n_chains - 1) / 2,)
        Values of the test statistics for each pairwise comparison
        between chains.

    pval : array, shape (n_chains * (n_chains - 1) / 2,)
        Values of the p-values for the test statistic under the
        asymptotic sampling distribution.

    References
    ----------
    Brooks, S. P., Giudici, P., and Phillipe, A., "Nonparametric
    Convergence Assessment for MCMC Model Selection", Journal of
    Computational and Graphical Statistics 12:1, 1 - 22 (2003),
    doi:10.1198/1061860031347
    """

    k = check_array(k, dtype=None)

    n_chains, n_iter = k.shape

    if split:
        # If calculating split diagnostics, treat the first half of
        # each chain as a separate chain.
        if n_iter % 2 == 0:
            k = np.reshape(k, (2 * n_chains, n_iter // 2))
        else:
            k = np.reshape(k[:, 1:], (2 * n_chains, (n_iter - 1) // 2))

    # Update values for number of chains and number of sweeps for use in
    # averages.
    n_chains, n_iter = k.shape

    # Perform pairwise comparisons of CDFs between chains,
    # taking the worst-case result.
    n_comparisons = n_chains * (n_chains - 1) // 2
    test_statistics = np.zeros((n_comparisons,))
    pvals = np.zeros((n_comparisons,))
    idx = 0
    for i in range(n_chains):
        for j in range(i + 1, n_chains):
            test_statistics[idx], pvals[idx] = ss.ks_2samp(
                k[i], k[j], mode=mode)
            idx += 1

    return test_statistics, pvals


def rjmcmc_batch_kstest_convergence(k, batch_size=None, thin=1,
                                    split=False, mode='auto'):
    """Calculate batched convergence diagnostics for RJMCMC output.

    Parameters
    ----------
    k : array-like, shape (n_chains, n_iter)
        Array containing model indicator variables.

    batch_size : integer, optional
        Batch size used for computing convergence diagnostics.

    Returns
    -------
    test_statistic : array, shape (n_batches, n_chains * (n_chains - 1) / 2)
        Array containing the test statistics for each batch.

    pvals : array, shape (n_batches, n_chains * (n_chains - 1) / 2)
        Array containing the p-values for each batch.

    samples : array, shape (n_batches,)
        The number of iterations used in each batch.
    """

    k = check_array(k, dtype=None)

    # Perform additional thinning of samples if necessary.
    if thin > 1:
        k = k[:, ::thin]

    n_chains, n_iter = k.shape

    if batch_size is None:
        batch_size = int((n_iter // 2) // 20)

    n_batches = int(np.ceil((n_iter // 2) / batch_size))
    if split:
        n_comparisons = n_chains * (2 * n_chains - 1)
    else:
        n_comparisons = n_chains * (n_chains - 1) // 2

    test_statistics = np.zeros((n_batches, n_comparisons))
    pvals = np.zeros((n_batches, n_comparisons))
    samples = np.zeros((n_batches,), dtype=int)

    for q in range(1, n_batches + 1):

        batch_stop = min(n_iter, 2 * q * batch_size)

        k_batch = k[:, :batch_stop]

        # Discard first half of each chain in the batch.
        n_warmup = int(batch_stop // 2)

        k_batch = k_batch[:, n_warmup:]

        samples[q - 1] = batch_stop
        test_statistics[q - 1], pvals[q - 1] = rjmcmc_kstest_convergence(
            k_batch, thin=thin, split=split, mode=mode)

    return test_statistics, pvals, samples
