"""
Provides routines for fitting DBNs.
"""

# License: MIT


from __future__ import absolute_import

from .bridge_sampler import (bridge_sampler,
                             bridge_sampler_relative_mse_estimate)
from .sampler_diagnostics import (estimate_stationary_distribution,
                                  rjmcmc_batch_rhat, rjmcmc_rhat)
from .sampler_helpers import get_indicator_set_neighborhood
from .stepwise_bayes_regression_model import (
    bayes_regression_normal_gamma_log_marginal_likelihood,
    log_uniform_indicator_set_prior,
    StepwiseBayesRegression)
from .stepwise_mc3_sampler import (optimize_stepwise_mc3,
                                   sample_stepwise_mc3,
                                   stepwise_mc3_propose_model,)

__all__ = [
    'bayes_regression_normal_gamma_log_marginal_likelihood',
    'bridge_sampler',
    'bridge_sampler_relative_mse_estimate',
    'estimate_stationary_distribution',
    'get_indicator_set_neighborhood',
    'log_uniform_indicator_set_prior',
    'optimize_stepwise_mc3',
    'rjmcmc_batch_rhat',
    'rjmcmc_rhat',
    'sample_stepwise_mc3',
    'stepwise_mc3_propose_model',
    'StepwiseBayesRegression'
]
