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


def test_sample_structures_prior():
    """Test sampling from uniform priors on structures."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    a_tau = 0.5
    b_tau = 2.3
    nu_sq = 0.9
    model = rdm.StepwiseBayesRegression(a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    n_chains = 4
    n_iter = 2000
    n_jobs = 1
    prior_samples = model.sample_structures_prior(
        'y ~ x1 + x2', n_chains=n_chains, n_iter=n_iter,
        n_jobs=n_jobs, random_state=random_state)

    assert prior_samples.prior.sizes['chain'] == n_chains
    assert prior_samples.prior.sizes['draw'] == n_iter

    possible_models = [np.array([0, 0]), np.array([1, 0]),
                       np.array([0, 1]), np.array([1, 1])]
    n_possible_models = len(possible_models)
    for i in range(n_chains):
        chain = prior_samples.prior.isel(chain=i)

        sampled_models = np.unique(chain['k'].data, axis=0)
        assert len(sampled_models) == n_possible_models

        for k in sampled_models:
            assert any([np.all(k == ki) for ki in possible_models])
            k_full = np.tile(k, (n_iter, 1))
            model_count = np.sum(np.all(chain['k'].data == k_full, axis=1))
            model_prob = model_count / n_iter
            assert np.abs(model_prob - 1.0 / n_possible_models) < 0.05

    n_samples = 200
    data = {'y': random_state.uniform(size=(n_samples,)),
            'x1': random_state.uniform(size=(n_samples,)),
            'x2': random_state.choice(2, size=(n_samples,))}

    prior_samples = model.sample_structures_prior(
        'y ~ x1 + x2', data=data, n_chains=n_chains, n_iter=n_iter,
        n_jobs=n_jobs, random_state=random_state)

    assert prior_samples.prior.sizes['chain'] == n_chains
    assert prior_samples.prior.sizes['draw'] == n_iter

    n_possible_models = 4
    for i in range(n_chains):
        chain = prior_samples.prior.isel(chain=i)

        sampled_models = np.unique(chain['k'].data, axis=0)
        assert len(sampled_models) == n_possible_models

        for k in sampled_models:
            assert any([np.all(k == ki) for ki in possible_models])
            k_full = np.tile(k, (n_iter, 1))
            model_count = np.sum(np.all(chain['k'].data == k_full, axis=1))
            model_prob = model_count / n_iter
            assert np.abs(model_prob - 1.0 / n_possible_models) < 0.05


def test_sample_parameters_prior():
    """Test sampling from parameter priors."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 200
    y = random_state.uniform(size=(n_samples,))
    x = random_state.uniform(size=(n_samples,))

    data = {'y': y, 'x': x}

    a_tau = 1.5
    b_tau = 5.0
    nu_sq = 2.3
    model = rdm.StepwiseBayesRegression(a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    n_chains = 4
    n_iter = 20000
    n_jobs = 1
    prior_samples = model.sample_parameters_prior(
        'y ~ x', data=data, n_chains=n_chains, n_iter=n_iter,
        n_jobs=n_jobs, random_state=random_state)

    assert prior_samples.prior.sizes['chain'] == n_chains
    assert prior_samples.prior.sizes['draw'] == n_iter

    expected_mean_tau_sq = a_tau * b_tau
    expected_mean_beta = 0.0
    expected_scale_beta = np.sqrt(nu_sq / (a_tau * b_tau))
    expected_df_beta = 2.0 * a_tau
    expected_std_beta = expected_scale_beta * np.sqrt(
        expected_df_beta / (expected_df_beta - 2.0))

    for i in range(n_chains):
        chain = prior_samples.prior.isel(chain=i)
        assert np.abs(np.mean(chain['tau_sq']) - expected_mean_tau_sq) < 0.05
        assert (np.abs(
            np.mean(chain['beta_Intercept']) - expected_mean_beta) < 0.05)
        assert (np.abs(
            np.std(chain['beta_Intercept']) - expected_std_beta) < 0.05)
        assert np.abs(np.mean(chain['beta_x']) - expected_mean_beta) < 0.05
        assert np.abs(np.std(chain['beta_x']) - expected_std_beta) < 0.05


def test_sample_parameters_posterior():
    """Test sampling from posterior distributions."""

    a_tau = 1.0
    b_tau = 1.2
    nu_sq = 2.5
    model = rdm.StepwiseBayesRegression(
        a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    random_seed = 0
    random_state = check_random_state(random_seed)

    beta_0 = 2.2
    beta_1 = 2.3
    beta_2 = -0.1
    sigma = 0.5
    n_samples = 500

    x = np.linspace(0.0, 3.0, n_samples)
    y = (beta_0 + beta_1 * x + beta_2 * x**2 +
         random_state.normal(loc=0.0, scale=sigma, size=(n_samples,)))

    data = {'y': y, 'x': x}

    n_chains = 4
    n_iter = 20000
    thin = 10
    n_jobs = 1

    fit = model.sample_parameters_posterior(
        'y ~ x', data=data, n_chains=n_chains, n_iter=n_iter,
        thin=thin, n_jobs=n_jobs, random_state=random_state)

    n_warmup = fit.warmup_posterior.sizes['draw']
    n_kept = fit.posterior.sizes['draw']

    assert n_warmup == 1000
    assert n_kept == 1000

    intercept_samples = np.empty((n_chains * n_kept,))
    predictor_samples = np.empty((n_chains * n_kept,))
    precision_samples = np.empty((n_chains * n_kept,))

    for i in range(n_chains):
        chain = fit.posterior.isel(chain=i)
        intercept_samples[i * n_kept:(i + 1) * n_kept] = \
            chain['beta_Intercept'].data
        predictor_samples[i * n_kept:(i + 1) * n_kept] = \
            chain['beta_x'].data
        precision_samples[i * n_kept:(i + 1) * n_kept] = \
            chain['tau_sq'].data

    min_intercept = np.quantile(intercept_samples, q=0.05)
    max_intercept = np.quantile(intercept_samples, q=0.95)
    mean_intercept = np.mean(intercept_samples)

    min_predictor = np.quantile(predictor_samples, q=0.05)
    max_predictor = np.quantile(predictor_samples, q=0.95)
    mean_predictor = np.mean(predictor_samples)

    min_precision = np.quantile(precision_samples, q=0.05)
    max_precision = np.quantile(precision_samples, q=0.95)
    mean_precision = np.mean(precision_samples)

    model_code = """
data {
    int<lower=0> T;
    real y[T];
    real X[T, 2];
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
}
transformed parameters {
    real sigma = 1.0 / sqrt(tau_sq);
}
model {
    target += gamma_lpdf(tau_sq | a_tau, b_tau_inv);
    target += normal_lpdf(beta_0 | 0, sqrt(nu_sq / tau_sq));
    target += normal_lpdf(beta_1 | 0, sqrt(nu_sq / tau_sq));
    for (t in 1:T) {
        target += normal_lpdf(y[t] | beta_0 + beta_1 * X[t, 1], sigma);
    }
}
    """

    data = {'T': n_samples, 'y': y, 'X': np.vstack([x, x**2]).T,
            'a_tau': a_tau, 'b_tau': b_tau, 'nu_sq': nu_sq}

    n_chains = 4
    n_iter = 10000
    sm = ps.StanModel(model_code=model_code)
    hmc_fit = sm.sampling(data=data, iter=n_iter, chains=n_chains, n_jobs=1)

    n_warmup = hmc_fit.sim['warmup2'][0]
    n_kept = hmc_fit.sim['n_save'][0] - n_warmup

    hmc_intercept_samples = np.empty((n_chains * n_kept,))
    hmc_predictor_samples = np.empty((n_chains * n_kept,))
    hmc_precision_samples = np.empty((n_chains * n_kept,))

    for i in range(n_chains):
        hmc_intercept_samples[i * n_kept:(i + 1) * n_kept] = \
            hmc_fit.sim['samples'][i]['chains']['beta_0'][-n_kept:]
        hmc_predictor_samples[i * n_kept:(i + 1) * n_kept] = \
            hmc_fit.sim['samples'][i]['chains']['beta_1'][-n_kept:]
        hmc_precision_samples[i * n_kept:(i + 1) * n_kept] = \
            hmc_fit.sim['samples'][i]['chains']['tau_sq'][-n_kept:]

    hmc_min_intercept = np.quantile(hmc_intercept_samples, q=0.05)
    hmc_max_intercept = np.quantile(hmc_intercept_samples, q=0.95)
    hmc_mean_intercept = np.mean(hmc_intercept_samples)

    hmc_min_predictor = np.quantile(hmc_predictor_samples, q=0.05)
    hmc_max_predictor = np.quantile(hmc_predictor_samples, q=0.95)
    hmc_mean_predictor = np.mean(hmc_predictor_samples)

    hmc_min_precision = np.quantile(hmc_precision_samples, q=0.05)
    hmc_max_precision = np.quantile(hmc_precision_samples, q=0.95)
    hmc_mean_precision = np.mean(hmc_precision_samples)

    assert np.abs(min_intercept - hmc_min_intercept) < 0.01
    assert np.abs(max_intercept - hmc_max_intercept) < 0.01
    assert np.abs(mean_intercept - hmc_mean_intercept) < 0.01

    assert np.abs(min_predictor - hmc_min_predictor) < 0.01
    assert np.abs(max_predictor - hmc_max_predictor) < 0.01
    assert np.abs(mean_predictor - hmc_mean_predictor) < 0.01

    assert np.abs(min_precision - hmc_min_precision) < 0.01
    assert np.abs(max_precision - hmc_max_precision) < 0.02
    assert np.abs(mean_precision - hmc_mean_precision) < 0.01


def test_prior_predictive_checks_fixed_model():
    """Test prior predictive sampling for a single model."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 30
    x1 = random_state.uniform(size=n_samples)
    x2 = random_state.uniform(size=n_samples)**2
    intcpt = 1.0
    coef = -2.3
    scale = 2.3
    y = random_state.normal(loc=(intcpt + coef * x1),
                            scale=scale, size=n_samples)

    data = {'y': y, 'x1': x1, 'x2': x2}

    a_tau = 1.5
    b_tau = 2.1
    nu_sq = 2.3
    model = rdm.StepwiseBayesRegression(a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    df = 2 * a_tau

    expected_mean_tau_sq = a_tau * b_tau
    expected_mean_beta = 0.0
    expected_scale_beta = np.sqrt(nu_sq / (a_tau * b_tau))
    expected_df_beta = 2.0 * a_tau
    expected_std_beta = expected_scale_beta * np.sqrt(
        expected_df_beta / (expected_df_beta - 2.0))

    n_chains = 4
    n_iter = 200000
    n_jobs = 1

    prior_samples = model.sample_parameters_prior(
        'y ~ 1', data=data,
        n_chains=n_chains, n_iter=n_iter, n_jobs=n_jobs,
        generate_prior_predictive=True,
        random_state=random_state)

    assert prior_samples.prior.sizes['chain'] == n_chains
    assert prior_samples.prior_predictive.sizes['chain'] == n_chains

    X = np.ones((n_samples, 1))
    for i in range(n_chains):

        prior_chain = prior_samples.prior.isel(chain=i)
        prior_pred_chain = prior_samples.prior_predictive.isel(chain=i)

        assert 'y' in prior_pred_chain
        assert 'tau_sq' in prior_chain

        assert prior_pred_chain['y'].shape == (n_iter, n_samples)
        assert prior_chain['tau_sq'].shape == (n_iter,)

        assert (np.abs(
            np.mean(prior_chain['tau_sq']) - expected_mean_tau_sq) < 0.1)
        assert (np.abs(
            np.mean(
                prior_chain['beta_Intercept']) - expected_mean_beta) < 0.1)
        assert (np.abs(
            np.std(prior_chain['beta_Intercept']) - expected_std_beta) < 0.1)

        assert np.all(np.abs(np.mean(prior_pred_chain['y'], axis=0)) < 0.1)

        yc = prior_pred_chain['y'].data - np.mean(prior_pred_chain['y'].data,
                                                  axis=0, keepdims=True)
        sample_cov = np.dot(yc.T, yc) / (n_iter - 1)
        expected_cov = ((np.eye(n_samples) + nu_sq * np.dot(X, X.T)) /
                        (a_tau * b_tau))
        expected_cov = df * expected_cov / (df - 2)

        assert np.all(
            np.abs((sample_cov - expected_cov)) < 0.5)

    prior_samples = model.sample_parameters_prior(
        'y ~ x1', data=data,
        n_chains=n_chains, n_iter=n_iter, n_jobs=n_jobs,
        generate_prior_predictive=True,
        random_state=random_state)

    assert prior_samples.prior.sizes['chain'] == n_chains
    assert prior_samples.prior_predictive.sizes['chain'] == n_chains

    X = np.vstack([np.ones(n_samples), x1]).T
    for i in range(n_chains):

        prior_chain = prior_samples.prior.isel(chain=i)
        prior_pred_chain = prior_samples.prior_predictive.isel(chain=i)

        assert 'y' in prior_pred_chain
        assert 'tau_sq' in prior_chain
        assert 'beta_x1' in prior_chain

        assert prior_pred_chain['y'].shape == (n_iter, n_samples)
        assert prior_chain['tau_sq'].shape == (n_iter,)
        assert prior_chain['beta_x1'].shape == (n_iter,)

        assert (np.abs(
            np.mean(prior_chain['tau_sq']) - expected_mean_tau_sq) < 0.1)
        assert (np.abs(
            np.mean(
                prior_chain['beta_Intercept']) - expected_mean_beta) < 0.1)
        assert (np.abs(
            np.std(prior_chain['beta_Intercept']) - expected_std_beta) < 0.1)
        assert (np.abs(
            np.mean(prior_chain['beta_x1']) - expected_mean_beta) < 0.1)
        assert (np.abs(
            np.std(prior_chain['beta_x1']) - expected_std_beta) < 0.1)

        assert np.all(np.abs(np.mean(prior_pred_chain['y'], axis=0)) < 0.1)

        yc = prior_pred_chain['y'].data - np.mean(prior_pred_chain['y'].data,
                                                  axis=0, keepdims=True)
        sample_cov = np.dot(yc.T, yc) / (n_iter - 1)
        expected_cov = ((np.eye(n_samples) + nu_sq * np.dot(X, X.T)) /
                        (a_tau * b_tau))
        expected_cov = df * expected_cov / (df - 2)

        assert np.all(
            np.abs((sample_cov - expected_cov)) < 0.5)


def test_prior_predictive_checks():
    """Test prior predictive sampling for all models."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 30
    x1 = random_state.uniform(size=n_samples)
    x2 = random_state.uniform(size=n_samples)**2
    intcpt = 1.0
    coef = -2.3
    scale = 2.3
    y = random_state.normal(loc=(intcpt + coef * x1),
                            scale=scale, size=n_samples)

    data = {'y': y, 'x1': x1, 'x2': x2}

    a_tau = 1.5
    b_tau = 2.1
    nu_sq = 2.3
    model = rdm.StepwiseBayesRegression(a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    n_chains = 4
    n_iter = 2000
    n_jobs = 1
    prior_samples = model.sample_structures_prior(
        'y ~ x1 + x2', data=data,
        n_chains=n_chains, n_iter=n_iter, n_jobs=n_jobs,
        generate_prior_predictive=True,
        random_state=random_state)

    assert prior_samples.prior.sizes['chain'] == n_chains
    assert prior_samples.prior_predictive.sizes['chain'] == n_chains

    possible_models = [np.array([0, 0]), np.array([1, 0]),
                       np.array([0, 1]), np.array([1, 1])]
    n_possible_models = len(possible_models)
    for i in range(n_chains):

        prior_chain = prior_samples.prior.isel(chain=i)
        prior_pred_chain = prior_samples.prior_predictive.isel(chain=i)

        assert 'y' in prior_pred_chain
        assert 'k' in prior_chain
        assert 'i_x1' in prior_chain
        assert 'i_x2' in prior_chain

        assert prior_pred_chain['y'].shape == (n_iter, n_samples)
        assert prior_chain['k'].shape == (n_iter, 2)
        assert prior_chain['i_x1'].shape == (n_iter,)
        assert prior_chain['i_x2'].shape == (n_iter,)

        sampled_models = np.unique(prior_chain['k'].data, axis=0)

        assert len(sampled_models) == n_possible_models

        for k in sampled_models:
            assert any([np.all(k == ki) for ki in possible_models])
            k_full = np.tile(k, (n_iter, 1))
            model_count = np.sum(
                np.all(prior_chain['k'].data == k_full, axis=1))
            model_prob = model_count / n_iter
            assert np.abs(model_prob - 1.0 / n_possible_models) < 0.05


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

    fit = model.sample_structures_posterior(
        'y ~ x', data=data, n_chains=n_chains, n_iter=n_iter,
        thin=thin, n_jobs=n_jobs, random_state=random_state)

    assert fit.posterior.sizes['chain'] == n_chains
    assert fit.posterior.sizes['draw'] == 250

    n_kept = fit.posterior.sizes['draw']

    indicator_samples = np.empty((n_chains * n_kept,))

    for i in range(n_chains):
        chain = fit.posterior.isel(chain=i)
        indicator_samples[i * n_kept:(i + 1) * n_kept] = \
            chain['i_x'][-n_kept:]

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

    model_summary = rdm.stepwise_model_samples_summary(fit)

    assert np.abs(model_summary['p'][1] - p_model_2) < 0.01
    assert np.abs(np.sum(model_summary['p']) - 1.0) < 0.001


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

    fit = model.sample_structures_posterior(
        'y ~ x1 + x2', data=data, n_chains=n_chains, n_iter=n_iter,
        thin=thin, n_jobs=n_jobs, random_state=random_state,
        max_terms=1)

    assert fit.posterior.sizes['chain'] == n_chains
    assert fit.posterior.sizes['draw'] == 250

    n_kept = fit.posterior.sizes['draw']

    indicator_one_samples = np.empty((n_chains * n_kept,))
    indicator_two_samples = np.empty((n_chains * n_kept,))

    for i in range(n_chains):
        chain = fit.posterior.isel(chain=i)

        indicator_one_samples[i * n_kept:(i + 1) * n_kept] = \
            chain['i_x1'][-n_kept:]
        indicator_two_samples[i * n_kept:(i + 1) * n_kept] = \
            chain['i_x2'][-n_kept:]

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

    model_summary = rdm.stepwise_model_samples_summary(fit)

    assert np.abs(model_summary['p'][1] - np.exp(model_3_log_post)) < 0.01
    assert np.abs(model_summary['p'][2] - np.exp(model_2_log_post)) < 0.01
    assert np.abs(np.sum(model_summary['p']) - 1.0) < 0.001


def test_posterior_predictive_checks_fixed_model_intercept_only():
    """Test posterior predictive sampling for a single model."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 30
    x1 = random_state.uniform(size=n_samples)
    x2 = random_state.uniform(size=n_samples)**2
    intcpt = 1.0
    coef = -2.3
    scale = 2.3
    y = random_state.normal(loc=(intcpt + coef * x1),
                            scale=scale, size=n_samples)

    data = {'y': y, 'x1': x1, 'x2': x2}

    a_tau = 1.5
    b_tau = 2.1
    nu_sq = 2.3
    model = rdm.StepwiseBayesRegression(a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    n_chains = 4
    n_iter = 200000
    warmup = n_iter // 2
    thin = 1
    n_jobs = 1

    posterior_samples = model.sample_parameters_posterior(
        'y ~ 1', data=data,
        n_chains=n_chains, n_iter=n_iter, n_jobs=n_jobs, thin=thin,
        warmup=warmup,
        generate_posterior_predictive=True,
        random_state=random_state)

    assert posterior_samples.posterior.sizes['chain'] == n_chains
    assert posterior_samples.posterior_predictive.sizes['chain'] == n_chains

    post_y_mean = np.zeros((n_chains, n_samples))
    post_y_cov = np.zeros((n_chains, n_samples, n_samples))
    for i in range(n_chains):

        post_pred_chain = posterior_samples.posterior_predictive.isel(chain=i)

        assert 'y' in post_pred_chain

        assert post_pred_chain['y'].shape == (n_iter - warmup, n_samples)

        post_y_mean[i] = np.mean(post_pred_chain['y'], axis=0)

        yc = post_pred_chain['y'].data - np.mean(post_pred_chain['y'].data,
                                                 axis=0, keepdims=True)
        post_y_cov[i] = np.dot(yc.T, yc) / (n_iter - warmup - 1)

    model_code = """
data {
    int<lower=0> T;
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
}
transformed parameters {
    real sigma = 1.0 / sqrt(tau_sq);
}
model {
    target += gamma_lpdf(tau_sq | a_tau, b_tau_inv);
    target += normal_lpdf(beta_0 | 0, sqrt(nu_sq / tau_sq));
    for (t in 1:T) {
        target += normal_lpdf(y[t] | beta_0, sigma);
    }
}
generated quantities {
    real y_tilde[T] = normal_rng(rep_array(beta_0, T), sigma);
}
    """

    data = {'T': n_samples, 'y': y,
            'a_tau': a_tau, 'b_tau': b_tau, 'nu_sq': nu_sq}

    n_chains = 4
    n_iter = 20000
    sm = ps.StanModel(model_code=model_code)
    hmc_fit = sm.sampling(data=data, iter=n_iter, chains=n_chains, n_jobs=1)

    hmc_post_y_mean = np.zeros((n_chains, n_samples))
    hmc_post_y_cov = np.zeros((n_chains, n_samples, n_samples))
    for i in range(n_chains):
        hmc_chain = hmc_fit.sim['samples'][i]['chains']
        for t in range(n_samples):
            var_name = 'y_tilde[{:d}]'.format(t + 1)
            hmc_post_y_mean[i, t] = np.mean(hmc_chain[var_name])

        for t1 in range(n_samples):
            for t2 in range(n_samples):
                var_name_1 = 'y_tilde[{:d}]'.format(t1 + 1)
                var_name_2 = 'y_tilde[{:d}]'.format(t2 + 1)

                y1 = hmc_chain[var_name_1]
                y2 = hmc_chain[var_name_2]

                hmc_post_y_cov[i, t1, t2] = np.cov(np.vstack([y1, y2]))[0, 1]

    assert np.all(
        np.abs((post_y_mean - hmc_post_y_mean)) < 0.1)
    assert np.all(
        np.abs((post_y_cov - hmc_post_y_cov)) < 0.5)


def test_posterior_predictive_checks_fixed_model_single_predictor():
    """Test posterior predictive sampling for a single model."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 30
    x1 = random_state.uniform(size=n_samples)
    x2 = random_state.uniform(size=n_samples)**2
    intcpt = 1.0
    coef = -2.3
    scale = 2.3
    y = random_state.normal(loc=(intcpt + coef * x1),
                            scale=scale, size=n_samples)

    data = {'y': y, 'x1': x1, 'x2': x2}

    a_tau = 1.5
    b_tau = 2.1
    nu_sq = 2.3
    model = rdm.StepwiseBayesRegression(a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    n_chains = 4
    n_iter = 200000
    warmup = n_iter // 2
    thin = 1
    n_jobs = 1

    posterior_samples = model.sample_parameters_posterior(
        'y ~ x2', data=data,
        n_chains=n_chains, n_iter=n_iter, n_jobs=n_jobs, thin=thin,
        warmup=warmup,
        generate_posterior_predictive=True,
        random_state=random_state)

    assert posterior_samples.posterior.sizes['chain'] == n_chains
    assert posterior_samples.posterior_predictive.sizes['chain'] == n_chains

    post_y_mean = np.zeros((n_chains, n_samples))
    post_y_cov = np.zeros((n_chains, n_samples, n_samples))
    for i in range(n_chains):

        post_pred_chain = posterior_samples.posterior_predictive.isel(chain=i)

        assert 'y' in post_pred_chain

        assert post_pred_chain['y'].shape == (n_iter - warmup, n_samples)

        post_y_mean[i] = np.mean(post_pred_chain['y'], axis=0)

        yc = post_pred_chain['y'].data - np.mean(post_pred_chain['y'].data,
                                                 axis=0, keepdims=True)
        post_y_cov[i] = np.dot(yc.T, yc) / (n_iter - warmup - 1)

    model_code = """
data {
    int<lower=0> T;
    real y[T];
    real x2[T];
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
}
transformed parameters {
    real sigma = 1.0 / sqrt(tau_sq);
}
model {
    target += gamma_lpdf(tau_sq | a_tau, b_tau_inv);
    target += normal_lpdf(beta_0 | 0, sqrt(nu_sq / tau_sq));
    target += normal_lpdf(beta_1 | 0, sqrt(nu_sq / tau_sq));
    for (t in 1:T) {
        target += normal_lpdf(y[t] | beta_0 + beta_1 * x2[t], sigma);
    }
}
generated quantities {
    real y_tilde[T];
    for (t in 1:T) {
        y_tilde[t] = normal_rng(beta_0 + beta_1 * x2[t], sigma);
    }
}
    """

    data = {'T': n_samples, 'y': y, 'x2': x2,
            'a_tau': a_tau, 'b_tau': b_tau, 'nu_sq': nu_sq}

    n_chains = 4
    n_iter = 20000
    sm = ps.StanModel(model_code=model_code)
    hmc_fit = sm.sampling(data=data, iter=n_iter, chains=n_chains, n_jobs=1)

    hmc_post_y_mean = np.zeros((n_chains, n_samples))
    hmc_post_y_cov = np.zeros((n_chains, n_samples, n_samples))
    for i in range(n_chains):
        hmc_chain = hmc_fit.sim['samples'][i]['chains']
        for t in range(n_samples):
            var_name = 'y_tilde[{:d}]'.format(t + 1)
            hmc_post_y_mean[i, t] = np.mean(hmc_chain[var_name])

        for t1 in range(n_samples):
            for t2 in range(n_samples):
                var_name_1 = 'y_tilde[{:d}]'.format(t1 + 1)
                var_name_2 = 'y_tilde[{:d}]'.format(t2 + 1)

                y1 = hmc_chain[var_name_1]
                y2 = hmc_chain[var_name_2]

                hmc_post_y_cov[i, t1, t2] = np.cov(np.vstack([y1, y2]))[0, 1]

    assert np.all(
        np.abs((post_y_mean - hmc_post_y_mean)) < 0.05)
    assert np.all(
        np.abs((post_y_cov - hmc_post_y_cov)) < 0.5)


def test_posterior_predictive_checks():
    """Test posterior predictive sampling for all models."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 30
    x1 = random_state.uniform(size=n_samples)
    x2 = random_state.uniform(size=n_samples)**2
    intcpt = 1.0
    coef = -2.3
    scale = 2.3
    y = random_state.normal(loc=(intcpt + coef * x1),
                            scale=scale, size=n_samples)

    data = {'y': y, 'x1': x1, 'x2': x2}

    a_tau = 1.5
    b_tau = 2.1
    nu_sq = 2.3
    model = rdm.StepwiseBayesRegression(a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    n_chains = 4
    n_iter = 2000
    thin = 1
    warmup = 1000
    n_jobs = 1
    posterior_samples = model.sample_structures_posterior(
        'y ~ x1 + x2', data=data,
        n_chains=n_chains, n_iter=n_iter, n_jobs=n_jobs,
        thin=thin, warmup=warmup,
        generate_posterior_predictive=True,
        random_state=random_state)

    assert posterior_samples.posterior.sizes['chain'] == n_chains
    assert posterior_samples.posterior_predictive.sizes['chain'] == n_chains

    possible_models = [np.array([0, 0]), np.array([1, 0]),
                       np.array([0, 1]), np.array([1, 1])]
    n_possible_models = len(possible_models)
    n_kept = n_iter - warmup
    for i in range(n_chains):

        posterior_chain = posterior_samples.posterior.isel(chain=i)
        posterior_pred_chain = \
            posterior_samples.posterior_predictive.isel(chain=i)

        assert 'y' in posterior_pred_chain
        assert 'k' in posterior_chain
        assert 'i_x1' in posterior_chain
        assert 'i_x2' in posterior_chain

        assert posterior_pred_chain['y'].shape == (n_kept, n_samples)
        assert posterior_chain['k'].shape == (n_kept, 2)
        assert posterior_chain['i_x1'].shape == (n_kept,)
        assert posterior_chain['i_x2'].shape == (n_kept,)

        sampled_models = np.unique(posterior_chain['k'].data, axis=0)

        assert len(sampled_models) == n_possible_models

        assert np.abs(np.mean(posterior_pred_chain['y'].data)) > 0.1
        assert np.std(posterior_pred_chain['y'].data) > 0.0


def test_structure_sample_chi2_convergence_diagnostics():
    """Test calculation of convergence diagnostics."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 30
    x1 = random_state.uniform(size=n_samples)
    intcpt = -2.0
    coef = 4.3
    scale = 1.3
    y = random_state.normal(loc=(intcpt + coef * x1),
                            scale=scale, size=n_samples)

    data = {'y': y, 'x1': x1}

    a_tau = 1.5
    b_tau = 2.1
    nu_sq = 1.0
    model = rdm.StepwiseBayesRegression(a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    n_chains = 4
    n_iter = 2000
    warmup = n_iter // 2
    thin = 1
    n_jobs = 1

    posterior_samples = model.sample_structures_posterior(
        'y ~ x1', data=data,
        n_chains=n_chains, n_iter=n_iter, n_jobs=n_jobs, thin=thin,
        warmup=warmup,
        generate_posterior_predictive=False,
        random_state=random_state)

    test_statistic, pval = \
        rdm.structure_sample_chi2_convergence_diagnostics(
            posterior_samples.posterior, batch=False)

    assert test_statistic >= 0.0
    assert 0.0 <= pval <= 1.0

    test_statistics, pvals, batch_sizes = \
        rdm.structure_sample_chi2_convergence_diagnostics(
            posterior_samples.posterior, batch=True)

    n_batches = batch_sizes.shape[0]

    assert test_statistics.shape == (n_batches,)
    assert pvals.shape == (n_batches,)

    assert np.all(test_statistics >= 0.0)
    assert (np.all(pvals >= 0.0) and np.all(pvals <= 1.0))


def test_structure_sample_ks_convergence_diagnostics():
    """Test calculation of convergence diagnostics."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 30
    x1 = random_state.uniform(size=n_samples)
    intcpt = -2.0
    coef = 4.3
    scale = 1.3
    y = random_state.normal(loc=(intcpt + coef * x1),
                            scale=scale, size=n_samples)

    data = {'y': y, 'x1': x1}

    a_tau = 1.5
    b_tau = 2.1
    nu_sq = 1.0
    model = rdm.StepwiseBayesRegression(a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    n_chains = 4
    n_iter = 2000
    warmup = n_iter // 2
    thin = 1
    n_jobs = 1

    posterior_samples = model.sample_structures_posterior(
        'y ~ x1', data=data,
        n_chains=n_chains, n_iter=n_iter, n_jobs=n_jobs, thin=thin,
        warmup=warmup,
        generate_posterior_predictive=False,
        random_state=random_state)

    test_statistic, pval = \
        rdm.structure_sample_ks_convergence_diagnostics(
            posterior_samples.posterior, batch=False)

    assert np.all(test_statistic >= 0.0)
    assert (np.all(pval >= 0.0) and np.all(pval <= 1.0))

    test_statistics, pvals, batch_sizes = \
        rdm.structure_sample_ks_convergence_diagnostics(
            posterior_samples.posterior, batch=True)

    n_batches = batch_sizes.shape[0]
    n_comparisons = n_chains * (n_chains - 1) // 2

    assert test_statistics.shape == (n_batches, n_comparisons)
    assert pvals.shape == (n_batches, n_comparisons)

    assert np.all(test_statistics >= 0.0)
    assert (np.all(pvals >= 0.0) and np.all(pvals <= 1.0))
