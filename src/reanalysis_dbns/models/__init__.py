"""
Provides routines for fitting DBNs.
"""

# License: MIT


from __future__ import absolute_import

from .bridge_sampler import (bridge_sampler,
                             bridge_sampler_relative_mse_estimate)
from .sampler_diagnostics import (estimate_convergence_rate,
                                  estimate_stationary_distribution,
                                  rjmcmc_batch_chisq_convergence,
                                  rjmcmc_batch_kstest_convergence,
                                  rjmcmc_batch_rhat,
                                  rjmcmc_chisq_convergence,
                                  rjmcmc_kstest_convergence,
                                  rjmcmc_rhat)
from .sampler_helpers import (check_max_nonzero_indicators,
                              get_indicator_set_neighborhood,
                              get_sampled_parameter_dimensions,
                              write_stepwise_mc3_samples)
from .stepwise_bayes_regression_model import (
    bayes_regression_normal_gamma_log_marginal_likelihood,
    log_uniform_indicator_set_prior,
    stepwise_model_samples_summary,
    StepwiseBayesRegression,
    structure_sample_diagnostics,
    structure_sample_chi2_convergence_diagnostics,
    structure_sample_convergence_rate,
    structure_sample_ks_convergence_diagnostics)
from .stepwise_mc3_sampler import (initialize_stepwise_mc3,
                                   optimize_stepwise_mc3,
                                   sample_stepwise_mc3,
                                   stepwise_mc3_propose_model,)

__all__ = [
    'bayes_regression_normal_gamma_log_marginal_likelihood',
    'bridge_sampler',
    'bridge_sampler_relative_mse_estimate',
    'check_max_nonzero_indicators',
    'estimate_convergence_rate',
    'estimate_stationary_distribution',
    'get_indicator_set_neighborhood',
    'get_sampled_parameter_dimensions',
    'log_uniform_indicator_set_prior',
    'initialize_stepwise_mc3',
    'optimize_stepwise_mc3',
    'rjmcmc_batch_chisq_convergence',
    'rjmcmc_batch_kstest_convergence',
    'rjmcmc_batch_rhat',
    'rjmcmc_chisq_convergence',
    'rjmcmc_kstest_convergence',
    'rjmcmc_rhat',
    'sample_stepwise_mc3',
    'stepwise_mc3_propose_model',
    'stepwise_model_samples_summary',
    'StepwiseBayesRegression',
    'structure_sample_diagnostics',
    'structure_sample_chi2_convergence_diagnostics',
    'structure_sample_convergence_rate',
    'structure_sample_ks_convergence_diagnostics',
    'write_stepwise_mc3_samples'
]
