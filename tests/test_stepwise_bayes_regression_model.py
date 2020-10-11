"""
Provides unit tests for stepwise Bayes regression model.
"""

# License: MIT

from __future__ import absolute_import, division

import numpy as np
import pystan as ps
import scipy.special as sp

import reanalysis_dbns.models as rdm

from sklearn.utils import check_random_state


def test_uniform_indicator_set_priors():
    """Test evaluation of uniform indicator set priors."""

    k = np.zeros(3)

    p = rdm.log_uniform_indicator_set_prior(k)
    expected = -np.log(2**3)

    assert np.abs(p - expected) < 1e-12

    p = rdm.log_uniform_indicator_set_prior(k, max_nonzero=2)
    expected = -np.log(7)

    assert np.abs(p - expected) < 1e-12


def test_bayes_regression_log_marginal_likelihood():
    """Test calculation of log marginal likelihood."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 500
    a_tau = 1.0
    b_tau = 10.0
    nu_sq = 0.9

    X = random_state.normal(loc=1.0, scale=3.0, size=(n_samples, 3))
    y = random_state.normal(loc=0.0, scale=0.5, size=n_samples)
    data = {'T': n_samples, 'X': X, 'y': y,
            'a_tau': a_tau, 'b_tau': b_tau, 'nu_sq': nu_sq}

    model_code = """
data {
    int<lower=0> T;
    real X[T, 3];
    real y[T];
    real<lower=0> a_tau;
    real<lower=0> b_tau;
    real<lower=0> nu_sq;
}
transformed data {
    real b_tau_inv = 1.0 / b_tau;
}
parameters {
    real<lower=0> tau_sq;
    real beta_0;
    real beta_1;
    real beta_2;
}
transformed parameters {
    real sigma = 1.0 / sqrt(tau_sq);
}
model {
    target += gamma_lpdf(tau_sq | a_tau, b_tau_inv);
    target += normal_lpdf(beta_0 | 0, sqrt(nu_sq / tau_sq));
    target += normal_lpdf(beta_1 | 0, sqrt(nu_sq / tau_sq));
    target += normal_lpdf(beta_2 | 0, sqrt(nu_sq / tau_sq));
    for (t in 1:T) {
        real mu_x = beta_0 * X[t, 1] + beta_1 * X[t, 2] + beta_2 * X[t, 3];
        target += normal_lpdf(y[t] | mu_x, sigma);
    }
}
    """

    n_chains = 4
    n_iter = 10000
    sm = ps.StanModel(model_code=model_code)
    fit = sm.sampling(data=data, iter=n_iter, chains=n_chains, n_jobs=1)

    bridge_result = rdm.bridge_sampler(fit, return_samples=True)

    analytic_result = \
        rdm.bayes_regression_normal_gamma_log_marginal_likelihood(
            y, X, nu_sq=nu_sq, a_tau=a_tau, b_tau=b_tau)

    assert np.abs(
        analytic_result - bridge_result['log_marginal_likelihoods']) < 0.1


def test_stepwise_bayes_regression_stepwise_mc3_single_predictor():
    """Test stepwise MC3 with single linear predictor model."""

    a_tau = 1.0
    b_tau = 1.0
    nu_sq = 1.0

    model = rdm.StepwiseBayesRegression(
        a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    random_seed = 0
    random_state = check_random_state(random_seed)

    beta_0 = 2.2
    beta_1 = 0
    sigma = 0.5
    n_samples = 500

    x = np.linspace(0.0, 3.0, n_samples)
    y = beta_0 + beta_1 * x + random_state.normal(
        loc=0.0, scale=sigma, size=(n_samples,))

    data = {'y': y, 'x': x}

    n_chains = 4
    n_iter = 5000
    n_jobs = 1
    thin = 10

    fit = model.sample_structures_mc3(
        'y ~ x', data=data, n_chains=n_chains, n_iter=n_iter,
        thin=thin, n_jobs=n_jobs, random_state=random_state)

    n_warmup = fit['warmup2'][0]
    n_kept = fit['n_save'][0] - n_warmup

    indicator_samples = np.empty((n_chains * n_kept,))

    for i in range(n_chains):
        indicator_samples[i * n_kept:(i + 1) * n_kept] = \
            fit['samples'][i]['chains']['i_x'][-n_kept:]

    model_1_log_evidence = \
        rdm.bayes_regression_normal_gamma_log_marginal_likelihood(
            y, np.ones((n_samples, 1)), nu_sq=nu_sq,
            a_tau=a_tau, b_tau=b_tau)
    model_2_log_evidence = \
        rdm.bayes_regression_normal_gamma_log_marginal_likelihood(
            y, np.vstack([np.ones((n_samples,)), x]).T,
            nu_sq=nu_sq, a_tau=a_tau, b_tau=b_tau)

    log_bayes_factor = model_2_log_evidence - model_1_log_evidence

    p_model_2 = np.exp(log_bayes_factor) / (1.0 + np.exp(log_bayes_factor))

    assert np.abs(p_model_2 - np.mean(indicator_samples)) < 0.01


def test_stepwise_bayes_regression_stepwise_mc3_two_predictors():
    """Test stepwise MC3 with linear model."""

    a_tau = 1.5
    b_tau = 20.0
    nu_sq = 3.0

    model = rdm.StepwiseBayesRegression(
        a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    random_seed = 0
    random_state = check_random_state(random_seed)

    beta_0 = 2.2
    beta_1 = 0
    sigma = 0.5
    n_samples = 500

    x1 = np.linspace(0.0, 3.0, n_samples)
    x2 = np.linspace(-1.0, 1.0, n_samples)**2
    y = beta_0 + beta_1 * x2 + random_state.normal(
        loc=0.0, scale=sigma, size=(n_samples,))

    data = {'y': y, 'x1': x1, 'x2': x2}

    n_chains = 4
    n_iter = 5000
    n_jobs = 1
    thin = 10

    fit = model.sample_structures_mc3(
        'y ~ x1 + x2', data=data, n_chains=n_chains, n_iter=n_iter,
        thin=thin, n_jobs=n_jobs, random_state=random_state,
        max_terms=1)

    n_warmup = fit['warmup2'][0]
    n_kept = fit['n_save'][0] - n_warmup

    indicator_one_samples = np.empty((n_chains * n_kept,))
    indicator_two_samples = np.empty((n_chains * n_kept,))

    for i in range(n_chains):
        indicator_one_samples[i * n_kept:(i + 1) * n_kept] = \
            fit['samples'][i]['chains']['i_x1'][-n_kept:]
        indicator_two_samples[i * n_kept:(i + 1) * n_kept] = \
            fit['samples'][i]['chains']['i_x2'][-n_kept:]

    model_1_log_evidence = \
        rdm.bayes_regression_normal_gamma_log_marginal_likelihood(
            y, np.ones((n_samples, 1)), nu_sq=nu_sq,
            a_tau=a_tau, b_tau=b_tau)
    model_2_log_evidence = \
        rdm.bayes_regression_normal_gamma_log_marginal_likelihood(
            y, np.vstack([np.ones((n_samples,)), x1]).T,
            nu_sq=nu_sq, a_tau=a_tau, b_tau=b_tau)
    model_3_log_evidence = \
        rdm.bayes_regression_normal_gamma_log_marginal_likelihood(
            y, np.vstack([np.ones((n_samples,)), x2]).T,
            nu_sq=nu_sq, a_tau=a_tau, b_tau=b_tau)

    log_evidence = np.array([model_1_log_evidence,
                             model_2_log_evidence,
                             model_3_log_evidence])
    log_normalization = sp.logsumexp(log_evidence - np.log(3.0))

    model_2_log_post = (model_2_log_evidence - np.log(3.0) -
                        log_normalization)
    model_3_log_post = (model_3_log_evidence - np.log(3.0) -
                        log_normalization)

    assert np.abs(np.exp(model_2_log_post) -
                  np.mean(indicator_one_samples)) < 0.01
    assert np.abs(np.exp(model_3_log_post) -
                  np.mean(indicator_two_samples)) < 0.01
