"""
Provides implementation of bridge sampler for estimating marginal likelihood.
"""

# License: MIT

from __future__ import absolute_import, division

import warnings

import arviz as az
import numpy as np
import scipy.linalg as sl
import scipy.stats as ss
from statsmodels.tsa.ar_model import AR

try:
    import pymc3 as pm
    HAS_PYMC3 = True
except ImportError:
    HAS_PYMC3 = False


def _iterate_optimal_rhat(ln_q11, ln_q12, ln_q21, ln_q22, n_eff,
                          l_star=None, r0=0.5, criterion='scaled_rhat',
                          tol=1e-8, max_iter=1000):
    """Iteratively compute the optimal bridge function.

    Iteratively calculates the normalizing constant ratio estimator
    given in Eq. (4.1) of Meng and Wong 1996.

    See https://github.com/quentingronau/bridgesampling/blob/master/R/bridge_sampler_internals.R .  # noqa: E501

    Parameters
    ----------
    ln_q11 : array-like, shape (n1,)
        Array containing the values of the logarithm of the unnormalized
        density q_1 evaluated at the n1 draws from the density p_1,
        ln[ q_1(w_{1j})].

    ln_q12 : array-like, shape (n2,)
        Array containing the values of the logarithm of the unnormalized
        density q_1 evaluated at the n2 draws from the density p_2,
        ln[ q_1(w_{2j})].

    ln_q21 : array-like, shape (n1,)
        Array containing the values of the logarithm of the unnormalized
        density q_2 evaluated at the n1 draws from the density p_2,
        ln[ q_2(w_{1j})].

    ln_q22 : array-like, shape (n2,)
        Array containing the values of the logarithm of the unnormalized
        density q_2 evaluated at the n2 draws from the density p_2,
        ln[ q_2(w_{2j})].

    n_eff : integer
        The effective number of draws.

    l_star : float, default: np.median(ln_q11 - ln_q21)
        If given, the constant subtracted from the ratios l1 and l2 for
        numerical stability.

    r0 : float, default: 0.5
        Initial guess for the scaled value of the ratio, rhat * exp(-l_star) .

    criterion : 'scaled_rhat' | 'log_rhat'
        Quantity to monitor for convergence, either the scaled rhat values
        or the logarithm of the unscaled values.

    tol : float, default: 1e-8
        Tolerance used to determine when convergence is reached.

    max_iter : integer, default: 1000
        Maximum number of iterations to perform.

    Returns
    -------
    result : dict
        A dict with the following keys and values:

        - 'log_rhat': the value of the logarithm of the ratio estimate
          on the last iteration.
        - 'scaled_rhat': the value of the scaled ratio estimate on the last
          iteration.
        - 'n_iter': the number of iterations completed.
        - 'scaled_rhat_values': a list containing the values of the scaled
          ratio estimate at each iteration.
        - 'log_rhat_values': a list containing the values of the logarithm of
          the ratio estimate at each iteration.
        - 'converged': a boolean value that is True if the convergence criterion
          was satisfied in fewer than the maximum number of iterations, False
          otherwise.

    References
    ----------
    X. L. Meng and W. H. Wong, "Simulating ratios of normalizing constants via
    a simple identity: a theoretical exploration.", Statistica Sinica 6, 4
    (1996), 831 - 860.
    """

    if criterion not in ('scaled_rhat', 'log_rhat'):
        raise ValueError("Unrecognized convergence criterion '%r'" % criterion)

    # Compute the quantities l1 and l2, note that inputs are logs of the
    # sampled values.
    ln_l1 = ln_q11 - ln_q21
    ln_l2 = ln_q12 - ln_q22

    # Subtract constant for stability.
    if l_star is None:
        l_star = np.median(ln_l1)

    n1 = ln_l1.shape[0]
    n2 = ln_l2.shape[0]

    # Compute sample size factors s1 and s2, using the given effective sample
    # size.
    s1 = n_eff / (n_eff + n2)
    s2 = n2 / (n_eff + n2)

    # Maintain list of scaled ratio values, and the logarithm of the
    # unscaled ratio.
    scaled_rhat = r0
    log_rhat = np.log(r0) + l_star

    scaled_rhat_values = [scaled_rhat]
    log_rhat_values = [log_rhat]

    # Iterate until convergence.
    delta = tol + 1.0
    converged = False
    for n_iter in range(max_iter):

        old_scaled_rhat = scaled_rhat
        old_log_rhat = log_rhat

        numerator_terms = (np.exp(ln_l2 - l_star) /
                           (s1 * np.exp(ln_l2 - l_star) +
                            s2 * old_scaled_rhat))
        denominator_terms = 1.0 / (s1 * np.exp(ln_l1 - l_star) +
                                   s2 * old_scaled_rhat)

        if (np.any(~np.isfinite(numerator_terms)) or
                np.any(~np.isfinite(denominator_terms))):
            warnings.warn(
                'Infinite value encountered at iteration %d. '
                'Try rerunning with additional samples.' % (n_iter + 1),
                UserWarning)

            scaled_rhat = np.NaN
            log_rhat = np.NaN

            scaled_rhat_values.append(scaled_rhat)
            log_rhat_values.append(log_rhat)

            break

        scaled_rhat = (n1 / n2) * (np.sum(numerator_terms) /
                                   np.sum(denominator_terms))
        log_rhat = np.log(scaled_rhat) + l_star

        scaled_rhat_values.append(scaled_rhat)
        log_rhat_values.append(log_rhat)

        if criterion == 'scaled_rhat':
            delta = np.abs((scaled_rhat - old_scaled_rhat) / old_scaled_rhat)
        else:
            delta = np.abs((log_rhat - old_log_rhat) / log_rhat)

        if delta < tol:
            converged = True
            break

    if n_iter == max_iter and tol > 0:
        warnings.warn('Maximum number of iterations %d reached.' %
                      max_iter, UserWarning)

    return dict(log_rhat=log_rhat, scaled_rhat=scaled_rhat, n_iter=n_iter,
                scaled_rhat_values=scaled_rhat_values,
                log_rhat_values=log_rhat_values,
                converged=converged)


def _get_overall_n_eff(n_effs):
    """Get overall effective sample size."""

    combined_n_effs = []

    for v in n_effs:
        var_da = n_effs[v]
        combined_n_effs += [var_da.data.flatten()]

    return np.median(np.concatenate(combined_n_effs))


def _bridge_sampler_normal_proposal(samples_for_fit, samples_for_iter,
                                    log_posterior,
                                    n_eff=None, N2=None, n_repetitions=1,
                                    r0=0.5, tol=1e-8, tol_fallback=1e-4,
                                    max_iter=1000, return_samples=True):
    """Estimate marginal likelihood using bridge sampling with normal proposal.

    The input posterior samples are assumed to have already been transformed
    to take values on the real line. The log_posterior function should be
    callable with a 1D array containing the values of the transformed
    variables, returning the value of the unnormalized log posterior density
    and any required Jacobian factor.

    The notation used is from https://arxiv.org/abs/1703.05984 . The
    implementation follows that in the R package bridgesampling
    (see https://github.com/quentingronau/bridgesampling/blob/master/R/bridge_sampler_normal.R)  # noqa: E501

    Parameters
    ----------
    samples_for_fit : array-like, shape (n_fit_samples, n_variables)
        Array containing the values of the transformed posterior samples
        to use for fitting the proposal density.

    samples_for_iter : array-like, shape (n_iter_samples, n_variables)
        Array containing the values of the transformed posterior samples
        to use for calculating the bridge sampling estimate.

    log_posterior : callable
        A function callable with the signature log_posterior(theta),
        where theta is a 1D array containing the values of the transformed
        variables. The function should include any Jacobian factor associated
        with the change of variables.

    n_eff : integer, optional
        If given, the effective MCMC sample size to use. If absent, the
        number of posterior samples used for the bridge estimate is taken to be
        the effective sample size.

    N2 : integer, optional
        If given, the number of draws from the proposal density. If absent,
        the number of draws from the proposal density is chosen to be equal
        to the number of posterior samples used in the iteration.

    n_repetitions : integer, default: 1
        The number of times to repeat the marginal likelihood calculation with
        different draws from the proposal density.

    r0 : float, default: 0.5
        Initial guess for the marginal likelihood.

    tol : float, default: 1e-8
        The tolerance for the stopping criterion.

    tol_fallback: float, default: 1e-4
        The stopping tolerance used if an initial iteration fails to converge.

    max_iter: integer, default: 1000
        The maximum allowed number of iterations.

    Returns
    -------
    log_marginal_likelihoods: float or array, shape (n_repetitions,)
        The estimated log marginal likelihoods from each repetition.

    n_iters : integer or array, shape (n_repetitions,)
        The number of iterations used to calculate the log marginal likelihoods
        on each repetition.

    References
    ----------
    Q. F. Gronau et al, "A tutorial on bridge sampling", Journal of
    Mathematical Psychology 81 (2017), 80 - 97, doi:10.1016/j.jmp.2017.09.005 .
    """

    n_variables = samples_for_fit.shape[1]

    if n_eff is None:
        n_eff = samples_for_iter.shape[0]

    if N2 is None:
        N2 = samples_for_iter.shape[0]

    # Fit normal proposal density to posterior samples.
    sample_mean = np.mean(samples_for_fit, axis=0)
    sample_cov = np.cov(samples_for_fit, rowvar=False)
    sample_chol = sl.cholesky(sample_cov, lower=True)

    # Generate samples from proposal density.
    proposal_samples = (sample_mean +
                        np.dot(ss.norm.rvs(0, 1,
                                           size=(n_repetitions, N2,
                                                 n_variables)),
                               sample_chol.T))

    # Evaluate the fitted proposal density at the posterior samples
    # and at the proposal samples.
    ln_q21 = ss.multivariate_normal.logpdf(
        samples_for_iter, mean=sample_mean, cov=sample_cov)
    ln_q22 = ss.multivariate_normal.logpdf(
        proposal_samples, mean=sample_mean, cov=sample_cov)

    if n_repetitions == 1 and ln_q22.shape[0] != n_repetitions:
        ln_q22 = np.reshape(ln_q22, (n_repetitions,) + ln_q22.shape)

    # Evaluate the unnormalized posterior (together with any Jacobian,
    # keeping in mind that the variables are those transformed
    # to the real line) at the posterior samples and the proposal samples.
    ln_q11 = np.array([log_posterior(sample) for sample in samples_for_iter])
    ln_q12 = np.empty((n_repetitions, samples_for_iter.shape[0]))
    for rep in range(n_repetitions):
        ln_q12[rep] = np.array(
            [log_posterior(sample) for sample in proposal_samples[rep]])

    # Iteratively compute the estimate for the marginal likelihood.
    log_marginal_likelihoods = np.empty((n_repetitions,))
    n_iters = np.empty((n_repetitions,))

    for rep in range(n_repetitions):
        iter_result = _iterate_optimal_rhat(
            ln_q11, ln_q12[rep], ln_q21, ln_q22[rep], n_eff,
            r0=r0, criterion='scaled_rhat',
            tol=tol, max_iter=max_iter)

        n_iter = iter_result['n_iter']

        if not iter_result['converged']:
            warnings.warn(
                'Estimation of marginal likelihood failed, restarting with '
                'adjusted starting value',
                UserWarning)

            r0_fallback = np.sqrt(iter_result['scaled_rhat_values'][-1] *
                                  iter_result['scaled_rhat_values'][-2])

            iter_result = _iterate_optimal_rhat(
                ln_q11, ln_q12[rep], ln_q21, ln_q22[rep], n_eff,
                r0=r0_fallback, criterion='log_rhat',
                tol=tol_fallback, max_iter=max_iter)

            n_iter = max_iter + iter_result['n_iter']

        log_marginal_likelihoods[rep] = iter_result['log_rhat']
        n_iters[rep] = n_iter

    if n_repetitions == 1:
        log_marginal_likelihoods = log_marginal_likelihoods[0]
        n_iters = n_iters[0]

    if return_samples:
        return dict(log_marginal_likelihoods=log_marginal_likelihoods,
                    n_iters=n_iters,
                    post_sample_log_posterior=ln_q11,
                    prop_sample_log_posterior=ln_q12,
                    post_sample_log_proposal=ln_q21,
                    prop_sample_log_proposal=ln_q22)

    return log_marginal_likelihoods, n_iters


def _spectral_density_at_zero_frequency(y):
    """Estimate spectral density at zero frequency."""

    if y.ndim == 1:
        y = y[:, np.newaxis]

    n_samples, n_features = y.shape

    x = np.arange(1, n_samples + 1)
    x = x[:, np.newaxis] ** [0, 1]

    spec = np.empty((n_features,))
    order = np.empty((n_features,))

    for i in range(n_features):
        coefs, _, _, _ = sl.lstsq(x, y[:, i])
        residuals = y[:, i] - np.dot(x, coefs)

        if np.std(residuals) == 0:
            spec[i] = 0
            order[i] = 0
        else:
            ar_fit = AR(y).fit(ic='aic', trend='c')
            order[i] = ar_fit.k_ar
            spec[i] = (np.var(ar_fit.resid) /
                       (1.0 - np.sum(ar_fit.params[1:])) ** 2)

    if n_features == 1:
        spec = spec[0]
        order = order[0]

    return spec, order


def _bridge_sampler_normal_proposal_stan(fit, log_posterior=None,
                                         use_n_eff=True, **kwargs):
    """Estimate marginal likelihood with normal proposal.

    NB, the Stan model must be written such that all additive constants are
    retained in the log density.
    """

    if log_posterior is None:
        def _logp(theta):
            return fit.log_prob(theta, adjust_transform=True)

        log_posterior = _logp

    posterior_samples = fit.extract(
        pars=fit.model_pars, permuted=False, inc_warmup=False)

    # Determine the number of posterior samples to be used for fitting
    # the proposal density and for computing the bridge estimate.
    n_post_samples_per_chain = fit.sim['n_save'][0] - fit.sim['warmup2'][0]
    n_chains = fit.sim['chains']

    # The first half of the posterior samples (per chain) are used for fitting
    # the proposal density.
    n_fit_samples_per_chain = n_post_samples_per_chain // 2
    n_fit_samples = n_fit_samples_per_chain * n_chains

    # The remaining N1 posterior samples are used for computing the bridge
    # estimate.
    N1_per_chain = n_post_samples_per_chain - n_fit_samples_per_chain
    N1 = N1_per_chain * n_chains

    # Get posterior samples for the unconstrained parameters on the real line.
    unconstrained_posterior_samples = None
    for i in range(n_post_samples_per_chain):
        for j in range(n_chains):

            unconstrained_pars = fit.unconstrain_pars(
                {p: posterior_samples[p][i, j] for p in posterior_samples})

            if unconstrained_posterior_samples is None:
                unconstrained_posterior_samples = np.empty(
                    (n_post_samples_per_chain, n_chains,
                     unconstrained_pars.shape[0]))

            unconstrained_posterior_samples[i, j] = unconstrained_pars

    # Construct arrays containing samples for fitting the proposal density
    # and for computing the bridge estimate. Note that in the case of
    # multiple chains, the first half of each chain is used for fitting the
    # proposal density.
    samples_for_fit = np.reshape(
        unconstrained_posterior_samples[:n_fit_samples_per_chain],
        (n_fit_samples,) + unconstrained_posterior_samples.shape[2:])
    samples_for_iter = np.reshape(
        unconstrained_posterior_samples[n_fit_samples_per_chain:],
        (N1,) + unconstrained_posterior_samples.shape[2:])

    var_n_effs = az.ess(
        {v: unconstrained_posterior_samples[
            n_fit_samples_per_chain:, :, i].swapaxes(0, 1)
         for i, v in enumerate(fit.unconstrained_param_names())})

    if use_n_eff:
        n_eff = _get_overall_n_eff(var_n_effs)
    else:
        n_eff = None

    # Calculate the bridge sampling estimate for the log marginal likelihood.
    return _bridge_sampler_normal_proposal(
        samples_for_fit, samples_for_iter, log_posterior,
        n_eff=n_eff, **kwargs)


def _bridge_sampler_normal_proposal_pymc3(mcmc_trace, model=None,
                                          log_posterior=None, use_n_eff=True,
                                          **kwargs):
    """Estimate marginal likelihood using bridge sampling with normal proposal."""  # noqa: E501

    model = pm.modelcontext(model)

    if log_posterior is None:
        log_posterior = model.logp_array

    # Determine the number of posterior samples to be used for
    # fitting the proposal density and for computing the bridge estimate.
    n_post_samples_per_chain = len(mcmc_trace)
    n_chains = mcmc_trace.nchains

    # The first half of the posterior samples (per chain) are used for fitting
    # the proposal density.
    n_fit_samples_per_chain = n_post_samples_per_chain // 2
    n_fit_samples = n_fit_samples_per_chain * n_chains

    # The remaining N1 posterior samples are used for computing the bridge
    # estimate.
    N1_per_chain = n_post_samples_per_chain - n_fit_samples_per_chain
    N1 = N1_per_chain * n_chains

    # Construct arrays containing samples for fitting the proposal density
    # and for computing the bridge estimate. Note that in the case of
    # multiple chains, the first half of each chain is used for fitting the
    # proposal density.
    n_variables = model.bijection.ordering.size
    samples_for_fit = np.empty((n_fit_samples, n_variables))
    samples_for_iter = np.empty((N1, n_variables))

    random_vars = model.free_RVs
    var_n_effs = dict()
    for v in random_vars:
        var_map = model.bijection.ordering.by_name[v.name]

        var_samples_for_fit = mcmc_trace[:n_fit_samples_per_chain][v.name]
        if var_samples_for_fit.ndim > 1:
            samples_for_fit[:, var_map.slc] = var_samples_for_fit.reshape(
                (var_samples_for_fit.shape[0],
                 np.prod(var_samples_for_fit.shape[1:], dtype='i8')))
        else:
            samples_for_fit[:, var_map.slc] = var_samples_for_fit.reshape(
                (var_samples_for_fit.shape[0], 1))

        var_samples_for_iter = mcmc_trace[n_fit_samples_per_chain:][v.name]
        if var_samples_for_iter.ndim > 1:
            samples_for_iter[:, var_map.slc] = var_samples_for_iter.reshape(
                (var_samples_for_iter.shape[0],
                 np.prod(var_samples_for_iter.shape[1:], dtype='i8')))
        else:
            samples_for_iter[:, var_map.slc] = var_samples_for_iter.reshape(
                (var_samples_for_iter.shape[0], 1))

        if pm.util.is_transformed_name(v.name):
            key = pm.util.get_untransformed_name(v.name)
        else:
            key = v.name

        var_n_effs.update(
            pm.ess(mcmc_trace[n_fit_samples_per_chain:], var_names=[key]))

    if use_n_eff:
        n_eff = _get_overall_n_eff(var_n_effs)
    else:
        n_eff = None

    # Calculate the bridge sampling estimate for the log marginal likelihood.
    return _bridge_sampler_normal_proposal(
        samples_for_fit, samples_for_iter, log_posterior,
        n_eff=n_eff, **kwargs)


def bridge_sampler(samples, method='normal', **kwargs):
    """Compute log marginal likelihood using bridge sampling."""

    if method not in ('normal',):
        raise ValueError("Unrecognized method '%r'" % method)

    if HAS_PYMC3 and isinstance(samples, pm.backends.base.MultiTrace):
        return _bridge_sampler_normal_proposal_pymc3(samples, **kwargs)

    if hasattr(samples, 'stansummary'):
        return _bridge_sampler_normal_proposal_stan(samples, **kwargs)

    raise NotImplementedError(
        'Bridge sampling not implemented for given sample format')


def bridge_sampler_relative_mse_estimate(log_marginal_likelihood,
                                         post_sample_log_posterior,
                                         prop_sample_log_posterior,
                                         post_sample_log_proposal,
                                         prop_sample_log_proposal):
    """Compute relative mean squared error estimate from bridge sampler output."""  # noqa: E501

    g_post = np.exp(post_sample_log_proposal)
    p_post = np.exp(post_sample_log_posterior - log_marginal_likelihood)

    g_prop = np.exp(prop_sample_log_proposal)
    p_prop = np.exp(prop_sample_log_posterior - log_marginal_likelihood)

    N1 = len(p_post)
    N2 = len(g_prop)

    s1 = N1 / (N1 + N2)
    s2 = N2 / (N1 + N2)

    f1 = p_prop / (s1 * p_prop + s2 * g_prop)
    f2 = g_post / (s1 * p_post + s2 * g_post)

    rho_f2, _ = _spectral_density_at_zero_frequency(f2)

    term_one = np.var(f1) / (N2 * np.mean(f1) ** 2)
    term_two = rho_f2 * np.var(f2) / (N1 * np.mean(f2) ** 2)

    relative_mse = term_one + term_two
    coef_var = np.sqrt(relative_mse)

    return relative_mse, coef_var
