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
    prior_samples = model.sample_structures_prior(
        'y ~ x1 + x2', n_chains=n_chains, n_iter=n_iter,
        random_state=random_state)

    assert len(prior_samples['samples']) == n_chains

    possible_models = [np.array([0, 0]), np.array([1, 0]),
                       np.array([0, 1]), np.array([1, 1])]
    n_possible_models = len(possible_models)
    for i in range(n_chains):
        chains = prior_samples['samples'][i]['chains']

        sampled_models = np.unique(chains['k'], axis=0)
        assert len(sampled_models) == n_possible_models

        for k in sampled_models:
            assert any([np.all(k == ki) for ki in possible_models])
            k_full = np.tile(k, (n_iter, 1))
            model_count = np.sum(np.all(chains['k'] == k_full, axis=1))
            model_prob = model_count / n_iter
            assert np.abs(model_prob - 1.0 / n_possible_models) < 0.05

    n_samples = 200
    data = {'y': random_state.uniform(size=(n_samples,)),
            'x1': random_state.uniform(size=(n_samples,)),
            'x2': random_state.choice(2, size=(n_samples,))}

    prior_samples = model.sample_structures_prior(
        'y ~ x1 + x2', data=data, n_chains=n_chains, n_iter=n_iter,
        random_state=random_state)

    assert len(prior_samples['samples']) == n_chains

    n_possible_models = 4
    for i in range(n_chains):
        chains = prior_samples['samples'][i]['chains']

        sampled_models = np.unique(chains['k'], axis=0)
        assert len(sampled_models) == n_possible_models

        for k in sampled_models:
            assert any([np.all(k == ki) for ki in possible_models])
            k_full = np.tile(k, (n_iter, 1))
            model_count = np.sum(np.all(chains['k'] == k_full, axis=1))
            model_prob = model_count / n_iter
            assert np.abs(model_prob - 1.0 / n_possible_models) < 0.05


def test_sample_parameter_priors():
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
    prior_samples = model.sample_parameter_priors(
        'y ~ x', data=data, n_chains=n_chains, n_iter=n_iter,
        random_state=random_state)

    assert len(prior_samples['samples']) == n_chains

    expected_mean_tau_sq = a_tau * b_tau
    expected_mean_beta = 0.0
    expected_scale_beta = np.sqrt(nu_sq / (a_tau * b_tau))
    expected_df_beta = 2.0 * a_tau
    expected_std_beta = expected_scale_beta * np.sqrt(
        expected_df_beta / (expected_df_beta - 2.0))

    for i in range(n_chains):
        chains = prior_samples['samples'][i]['chains']
        assert np.abs(np.mean(chains['tau_sq']) - expected_mean_tau_sq) < 0.05
        assert np.abs(np.mean(chains['Intercept']) - expected_mean_beta) < 0.05
        assert np.abs(np.std(chains['Intercept']) - expected_std_beta) < 0.05
        assert np.abs(np.mean(chains['x']) - expected_mean_beta) < 0.05
        assert np.abs(np.std(chains['x']) - expected_std_beta) < 0.05


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

    prior_samples = model.sample_prior_predictive(
        'y ~ 1', data=data, fixed_model=True,
        n_chains=n_chains, n_iter=n_iter, n_jobs=n_jobs,
        random_state=random_state)

    assert len(prior_samples['samples']) == n_chains

    X = np.ones((n_samples, 1))
    for i in range(n_chains):

        chains = prior_samples['samples'][i]['chains']

        assert 'y' in chains
        assert 'tau_sq' in chains
        assert 'lp__' in chains

        assert chains['y'].shape == (n_iter, n_samples)
        assert chains['tau_sq'].shape == (n_iter,)
        assert chains['lp__'].shape == (n_iter,)

        assert np.abs(np.mean(chains['tau_sq']) - expected_mean_tau_sq) < 0.1
        assert np.abs(np.mean(chains['Intercept']) - expected_mean_beta) < 0.1
        assert np.abs(np.std(chains['Intercept']) - expected_std_beta) < 0.1

        assert np.all(np.abs(np.mean(chains['y'], axis=0)) < 0.1)

        yc = chains['y'] - np.mean(chains['y'], axis=0, keepdims=True)
        sample_cov = np.dot(yc.T, yc) / (n_iter - 1)
        expected_cov = ((np.eye(n_samples) + nu_sq * np.dot(X, X.T)) /
                        (a_tau * b_tau))
        expected_cov = df * expected_cov / (df - 2)

        assert np.all(
            np.abs((sample_cov - expected_cov)) < 0.5)

    prior_samples = model.sample_prior_predictive(
        'y ~ x1', data=data, fixed_model=True,
        n_chains=n_chains, n_iter=n_iter, n_jobs=n_jobs,
        random_state=random_state)

    assert len(prior_samples['samples']) == n_chains

    X = np.vstack([np.ones(n_samples), x1]).T
    for i in range(n_chains):

        chains = prior_samples['samples'][i]['chains']

        assert 'y' in chains
        assert 'tau_sq' in chains
        assert 'x1' in chains
        assert 'lp__' in chains

        assert chains['y'].shape == (n_iter, n_samples)
        assert chains['tau_sq'].shape == (n_iter,)
        assert chains['x1'].shape == (n_iter,)
        assert chains['lp__'].shape == (n_iter,)

        assert np.abs(np.mean(chains['tau_sq']) - expected_mean_tau_sq) < 0.1
        assert np.abs(np.mean(chains['Intercept']) - expected_mean_beta) < 0.1
        assert np.abs(np.std(chains['Intercept']) - expected_std_beta) < 0.1
        assert np.abs(np.mean(chains['x1']) - expected_mean_beta) < 0.1
        assert np.abs(np.std(chains['x1']) - expected_std_beta) < 0.1

        assert np.all(np.abs(np.mean(chains['y'], axis=0)) < 0.1)

        yc = chains['y'] - np.mean(chains['y'], axis=0, keepdims=True)
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
    prior_samples = model.sample_prior_predictive(
        'y ~ x1 + x2', data=data, fixed_model=False,
        n_chains=n_chains, n_iter=n_iter, n_jobs=n_jobs,
        random_state=random_state)

    assert len(prior_samples['samples']) == n_chains

    possible_models = [np.array([0, 0]), np.array([1, 0]),
                       np.array([0, 1]), np.array([1, 1])]
    n_possible_models = len(possible_models)
    for i in range(n_chains):

        chains = prior_samples['samples'][i]['chains']

        assert 'y' in chains
        assert 'k' in chains
        assert 'i_x1' in chains
        assert 'i_x2' in chains
        assert 'lp__' in chains

        assert chains['y'].shape == (n_iter, n_samples)
        assert chains['k'].shape == (n_iter, 2)
        assert chains['i_x1'].shape == (n_iter,)
        assert chains['i_x2'].shape == (n_iter,)
        assert chains['lp__'].shape == (n_iter,)

        sampled_models = np.unique(chains['k'], axis=0)

        assert len(sampled_models) == n_possible_models

        for k in sampled_models:
            assert any([np.all(k == ki) for ki in possible_models])
            k_full = np.tile(k, (n_iter, 1))
            model_count = np.sum(np.all(chains['k'] == k_full, axis=1))
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

    fit = model.sample_structures_mc3(
        'y ~ x', data=data, n_chains=n_chains, n_iter=n_iter,
        thin=thin, n_jobs=n_jobs, random_state=random_state)

    n_warmup = fit.warmup2[0]
    n_kept = fit.n_save[0] - n_warmup

    indicator_samples = np.empty((n_chains * n_kept,))

    for i in range(n_chains):
        indicator_samples[i * n_kept:(i + 1) * n_kept] = \
            fit.samples[i]['chains']['i_x'][-n_kept:]

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

    model_summary = fit.model_summary(inc_warmup=False)

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

    fit = model.sample_structures_mc3(
        'y ~ x1 + x2', data=data, n_chains=n_chains, n_iter=n_iter,
        thin=thin, n_jobs=n_jobs, random_state=random_state,
        max_terms=1)

    n_warmup = fit.warmup2[0]
    n_kept = fit.n_save[0] - n_warmup

    indicator_one_samples = np.empty((n_chains * n_kept,))
    indicator_two_samples = np.empty((n_chains * n_kept,))

    for i in range(n_chains):
        indicator_one_samples[i * n_kept:(i + 1) * n_kept] = \
            fit.samples[i]['chains']['i_x1'][-n_kept:]
        indicator_two_samples[i * n_kept:(i + 1) * n_kept] = \
            fit.samples[i]['chains']['i_x2'][-n_kept:]

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

    model_summary = fit.model_summary(inc_warmup=False, include_ci=True)

    assert np.abs(model_summary['p'][1] - np.exp(model_3_log_post)) < 0.01
    assert np.abs(model_summary['p'][2] - np.exp(model_2_log_post)) < 0.01
    assert np.abs(np.sum(model_summary['p']) - 1.0) < 0.001

    indicators_summary = fit.indicators_summary()

    assert (indicators_summary['mean'].where(
        indicators_summary['par_name'] == 'i_x1').dropna().item() ==
        np.mean(indicator_one_samples))

    assert (indicators_summary['mean'].where(
        indicators_summary['par_name'] == 'i_x2').dropna().item() ==
        np.mean(indicator_two_samples))
