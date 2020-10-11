"""
Provides tests for bridge sampler.
"""

# License: MIT

from __future__ import absolute_import

import numpy as np
import pystan as ps

from reanalysis_dbns.models import (bridge_sampler,
                                    bridge_sampler_relative_mse_estimate)


def test_pystan_method_converges_to_beta_binomial_analytic_result():
    """Test sampler gives correct result for beta-binomial model."""

    tolerance = 1e-2

    observed_k = 23
    n_trials = 100

    exact_log_marginal_likelihood = -np.log(n_trials + 1.0)

    n_samples = 30000
    n_chains = 4

    model_code = """
data {
   int<lower=0> n;
   int<lower=0> k;
   real a;
   real b;
}

parameters {
   real<lower=0, upper=1> p;
}

model {
   target += beta_lpdf(p | a, b);
   target += binomial_lpmf(k | n, p);
}
    """

    data = {'n': n_trials, 'k': observed_k, 'a': 1, 'b': 1}

    sm = ps.StanModel(model_code=model_code)
    fit = sm.sampling(data=data, iter=n_samples, chains=n_chains,
                      n_jobs=1)

    bridge_result = bridge_sampler(fit, return_samples=True)

    relative_mse = bridge_sampler_relative_mse_estimate(
        bridge_result['log_marginal_likelihoods'],
        bridge_result['post_sample_log_posterior'],
        bridge_result['prop_sample_log_posterior'][0],
        bridge_result['post_sample_log_proposal'],
        bridge_result['prop_sample_log_proposal'][0])[0]

    assert (np.abs(
        np.exp(exact_log_marginal_likelihood) -
        np.exp(bridge_result['log_marginal_likelihoods'])) < tolerance)

    actual_relative_mse = ((np.exp(bridge_result['log_marginal_likelihoods']) -
                            np.exp(exact_log_marginal_likelihood)) ** 2 /
                           np.exp(exact_log_marginal_likelihood) ** 2)

    assert actual_relative_mse <= 10 * relative_mse
