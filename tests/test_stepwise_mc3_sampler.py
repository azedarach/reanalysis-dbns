"""
Provides unit tests for stepwise MC3 sampler.
"""

# License: MIT

from __future__ import absolute_import, division


import numpy as np
import scipy.special as sp

import reanalysis_dbns.models as rdm


def beta_binomial_log_marginal_likelihood(n, k, alpha, beta):
    """Calculate log marginal likelihood under beta-binomial model."""

    return (np.log(sp.comb(n, k)) + sp.betaln(k + alpha, n - k + beta) -
            sp.betaln(alpha, beta))


def test_stepwise_mc3_propose_model():
    """Test stepwise model proposal."""

    k = np.array([1, 0, 0])

    nhd = [
        np.array([1, 0, 0]),
        np.array([1, 1, 0]),
        np.array([1, 0, 1]),
        np.array([0, 0, 0])
    ]

    next_k, log_proposal_ratio = rdm.stepwise_mc3_propose_model(
        k, max_nonzero=None, allow_exchanges=False)

    in_expected_nhd = False
    for n in nhd:
        if np.allclose(next_k, n):
            in_expected_nhd = True

    assert in_expected_nhd
    assert log_proposal_ratio == 0.0


def test_stepwise_mc3_beta_binomial():
    """Test converges to correct posterior distribution."""

    N_obs = 10
    m_obs = 2

    alpha_one = 1
    beta_one = 1
    alpha_two = 0.5
    beta_two = 1

    def logp(k, data=None):
        N = data['N']
        m = data['m']

        if k[0] == 0:
            return beta_binomial_log_marginal_likelihood(
                N, m, alpha_one, beta_one)

        return beta_binomial_log_marginal_likelihood(
            N, m, alpha_two, beta_two)

    data = {'N': N_obs,  'm': m_obs}
    initial_k = np.array([[0], [0], [1], [1]])

    n_chains = 4
    n_iter = 10000
    n_jobs = 1

    fit = rdm.sample_stepwise_mc3(
        initial_k, logp, data=data, n_chains=n_chains,
        n_iter=n_iter, n_jobs=n_jobs)

    log_post = np.array([
        beta_binomial_log_marginal_likelihood(
            N_obs, m_obs, alpha_one, beta_one) + np.log(0.5),
        beta_binomial_log_marginal_likelihood(
            N_obs, m_obs, alpha_two, beta_two) + np.log(0.5)])

    log_normalization = sp.logsumexp(log_post)

    expected_p1 = np.exp(log_post[0] - log_normalization)
    expected_p2 = np.exp(log_post[1] - log_normalization)

    model_1_count = 0
    model_2_count = 0
    for i in range(n_chains):
        model_1_count += np.sum(fit['samples'][i]['chains']['k'] == 0)
        model_2_count += np.sum(fit['samples'][i]['chains']['k'] == 1)

    sampled_p1 = model_1_count / (n_chains * n_iter)
    sampled_p2 = model_2_count / (n_chains * n_iter)

    assert np.abs(expected_p1 - sampled_p1) < 0.005
    assert np.abs(expected_p2 - sampled_p2) < 0.005


def test_optimize_mc3_beta_binomial():
    """Test converges to most probable model."""

    N_obs = 10
    m_obs = 2

    alpha_one = 1
    beta_one = 1
    alpha_two = 0.5
    beta_two = 1

    def logp(k, data=None):
        N = data['N']
        m = data['m']

        if k[0] == 0:
            return beta_binomial_log_marginal_likelihood(
                N, m, alpha_one, beta_one)

        return beta_binomial_log_marginal_likelihood(
            N, m, alpha_two, beta_two)

    data = {'N': N_obs,  'm': m_obs}
    initial_k = np.array([0])

    n_iter = 1000

    k_max, lp_max = rdm.optimize_stepwise_mc3(
        initial_k, logp, data=data, n_iter=n_iter)

    assert np.allclose(k_max, np.array([1]))
