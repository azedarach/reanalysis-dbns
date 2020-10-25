"""
Fit driver-response system.
"""

# License: MIT

from __future__ import absolute_import, division


import argparse
import os

import numpy as np
import scipy.stats as ss

import reanalysis_dbns.models as rdm
import reanalysis_dbns.utils as rdu


def sample_driver_response_system(alpha, gamma, tau, n_samples=500,
                                  n_realizations=1, warmup=0.5,
                                  random_state=None):
    """Generate samples from the driver-response AR system.

    Parameters
    ----------
    alpha : float
        Value of the lag-1 autocorrelation of the driver process.

    gamma : float
        Value of the standard deviation of the noise added to the
        response process.

    tau : int
        Value of the lag used to generate the response process from
        the driver process.

    n_samples : int, default: 500
        Length of the generated time-series.

    n_realizations : int, default: 1
        Number of independent time-series realizations to generate.

    warmup : float, default: 0.5
        Length of transient period to discard. The total number of
        samples generated is given by n_samples / warmup, of which
        only the last n_samples values are returned.

    random_state : None, int, or np.random.Generator
        Random state used for generating the time-series, passed
        to np.random.default_rng.

    Returns
    -------
    D : array-like, shape (n_realizations, n_samples)
        Array containing the values of the driver process.

    R : array-like, shape (n_realizations, n_samples)
        Array containing the values of the response process.
    """

    rng = np.random.default_rng(random_state)

    # Check input parameters.
    if not rdu.is_scalar(alpha) or np.abs(alpha) >= 1.0:
        raise ValueError(
            'Lag-1 autocorrelation must be a number between -1 and 1 '
            '(got alpha=%r)' % alpha)

    if not rdu.is_integer(tau) or tau < 0:
        raise ValueError(
            'Lag parameter must be a non-negative integer '
            '(got tau=%r)' % tau)

    if not rdu.is_scalar(warmup) or warmup <= 0 or warmup >= 1:
        raise ValueError(
            'Warm-up fraction must be a number between 0 and 1 '
            '(got warmup=%r)' % warmup)

    n_total_samples = int(np.ceil((1.0 / (1.0 - warmup)) * n_samples))

    presample_length = max(1, tau)

    D = np.empty((n_realizations, presample_length + n_total_samples),
                 dtype=float)
    R = np.empty((n_realizations, presample_length + n_total_samples),
                 dtype=float)

    # Draw initial values for driver from standard normal distribution.
    D[:, :presample_length] = rng.normal(
        loc=0.0, scale=1.0, size=(n_realizations, presample_length))
    R[:, :presample_length] = 0.0

    # Step system forward in time.
    for t in range(presample_length, presample_length + n_total_samples):

        epsD = rng.normal(loc=0.0, scale=1.0, size=n_realizations)
        epsR = rng.normal(loc=0.0, scale=1.0, size=n_realizations)

        D[:, t] = alpha * D[:, t - 1] + np.sqrt(1 - alpha**2) * epsD
        R[:, t] = D[:, t - tau] + gamma * epsR

    # Discard warm-up samples.
    return D[:, -n_samples:], R[:, -n_samples:]


def get_nu_sq_value(a_tau, b_tau, R=4, q=0.99):
    """Calculate value of SNR hyperparameter."""
    df = 2 * a_tau
    t_crit = ss.t.ppf(0.5 * (1 - q), df)
    return a_tau * b_tau * R**2 / t_crit**2


def get_fit_output_file(output_dir, alpha, gamma, tau, max_lag,
                        max_terms, a_tau, b_tau, nu_sq):
    """Get filename for writing fit to."""
    prefix = '.'.join(['driver_response', 'alpha-{:.3f}'.format(alpha),
                       'gamma-{:.3f}'.format(gamma),
                       'tau-{:d}'.format(tau),
                       'max_lag-{:d}'.format(max_lag)])
    suffix = '.'.join(['stepwise_bayes_regression',
                       'max_terms-{:d}'.format(max_terms),
                       'a_tau-{:.3f}'.format(a_tau),
                       'b_tau-{:.3f}'.format(b_tau),
                       'nu_sq-{:.3f}'.format(nu_sq)])
    filename = '.'.join([prefix, suffix, 'nc'])
    return os.path.join(output_dir, filename)


def estimate_thinning_parameter(model_spec, a_tau, b_tau, nu_sq, data=None,
                                n_chains=4, n_iter=5000, warmup=None,
                                thin=1, max_terms=4, n_jobs=-1,
                                reduction_frac=0.01,
                                random_seed=None):
    """Estimate required thinning parameter."""

    model = rdm.StepwiseBayesRegression(a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    samples = model.sample_structures_posterior(
        model_spec, data=data, max_terms=max_terms,
        n_chains=n_chains, n_iter=n_iter,
        warmup=warmup, thin=thin, n_jobs=n_jobs,
        generate_posterior_predictive=False,
        random_state=random_seed)

    convergence_rate = rdm.structure_sample_convergence_rate(
        samples.posterior, max_nonzero=max_terms,
        combine_chains=True)

    return int(np.ceil(np.log(reduction_frac) /
                       np.log(np.real(convergence_rate))))


def fit_stepwise_regression(model_spec, a_tau, b_tau, nu_sq, data=None,
                            n_chains=4, n_iter=2000, max_terms=4,
                            warmup=None, thin=1,
                            n_jobs=-1, random_seed=None):
    """Fit stepwise Bayes regression."""

    model = rdm.StepwiseBayesRegression(a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    samples = model.sample_structures_posterior(
        model_spec, data=data, max_terms=max_terms,
        n_chains=n_chains, n_iter=n_iter, warmup=warmup,
        thin=thin, n_jobs=n_jobs,
        generate_posterior_predictive=True,
        random_state=random_seed)

    return samples


def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Fit driver-response system')

    parser.add_argument('output_dir', help='fit output directory')

    parser.add_argument('--alpha', dest='alpha', type=float,
                        default=0.1, help='lag-1 autocorrelation')
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=1.0, help='noise level')
    parser.add_argument('--tau', dest='tau', type=int,
                        default=1, help='response lag')
    parser.add_argument('--n-samples', dest='n_samples', type=int,
                        default=500, help='number of data samples')
    parser.add_argument('--a-tau', dest='a_tau', type=float,
                        default=1.0, help='precision shape hyperparameter')
    parser.add_argument('--b-tau', dest='b_tau', type=float,
                        default=1.0, help='precision scale hyperparameter')
    parser.add_argument('-R', dest='R', type=float,
                        default=4.0, help='coefficient range')
    parser.add_argument('-q', dest='q', type=float,
                        default=0.99, help='coefficient interval probability')
    parser.add_argument('--max-lag', dest='max_lag', type=int,
                        default=6, help='maximum lag')
    parser.add_argument('--max-terms', dest='max_terms', type=int,
                        default=4, help='maximum number of terms')
    parser.add_argument('--n-chains', dest='n_chains', type=int,
                        default=4, help='number of chains')
    parser.add_argument('--n-iter', dest='n_iter', type=int,
                        default=2000, help='number of iterations')
    parser.add_argument('--n-jobs', dest='n_jobs', type=int,
                        default=-1, help='number of jobs')
    parser.add_argument('--random-seed', dest='random_seed', type=int,
                        default=None, help='random seed')

    return parser.parse_args()


def main():
    """Fit driver-response system."""

    args = parse_cmd_line_args()

    random_seed = args.random_seed

    warmup = 0.1
    n_realizations = 1

    D, R = sample_driver_response_system(
        args.alpha, args.gamma, args.tau,
        n_samples=args.n_samples, n_realizations=1,
        warmup=warmup, random_state=random_seed)

    # Only use a single realization.
    D = D[0]
    R = R[0]

    # Standardize observed data to zero mean and unit variance.
    D_std = (D - np.mean(D)) / np.std(D)
    R_std = (R - np.mean(D)) / np.std(R)

    # Construct lagged values of each time-series for use as predictors.
    max_lag = args.max_lag
    data = {'D': D_std[max_lag:], 'R': R_std[max_lag:]}
    for i in range(1, max_lag + 1):
        data['D_lag_{:d}'.format(i)] = D_std[max_lag - i:-i]
        data['R_lag_{:d}'.format(i)] = R_std[max_lag - i:-i]

    # The complete set of models that we consider include all main
    # effects from lagged variables.
    R_model_spec = 'R ~ {}'.format(
        ' + '.join([p for p in data if p not in ('R', 'D')]))
    D_model_spec = 'D ~ {}'.format(
        ' + '.join([p for p in data if p not in ('R', 'D')]))

    # Get hyperparameter values.
    a_tau = args.a_tau
    b_tau = args.b_tau
    nu_sq = get_nu_sq_value(a_tau, b_tau, R=args.R, q=args.q)

    # First, perform a series of short runs for estimating tuning parameters.
    R_thin = estimate_thinning_parameter(
        R_model_spec, a_tau, b_tau, nu_sq, data=data,
        n_chains=args.n_chains, max_terms=args.max_terms,
        n_jobs=args.n_jobs, random_seed=random_seed)

    D_thin = estimate_thinning_parameter(
        D_model_spec, a_tau, b_tau, nu_sq, data=data,
        n_chains=args.n_chains, max_terms=args.max_terms,
        n_jobs=args.n_jobs, random_seed=random_seed)

    thin = max(R_thin, D_thin)

    # Run full fits.
    R_samples = fit_stepwise_regression(
        R_model_spec, a_tau, b_tau, nu_sq, data=data,
        n_chains=args.n_chains, n_iter=args.n_iter,
        max_terms=args.max_terms, thin=thin,
        n_jobs=args.n_jobs, random_seed=random_seed)

    D_samples = fit_stepwise_regression(
        D_model_spec, a_tau, b_tau, nu_sq, data=data,
        n_chains=args.n_chains, n_iter=args.n_iter,
        max_terms=args.max_terms, thin=thin,
        n_jobs=args.n_jobs, random_seed=random_seed)

    base_filename = get_fit_output_file(
        args.output_dir, args.alpha, args.gamma, args.tau,
        args.max_lag, args.max_terms, a_tau,
        b_tau, nu_sq)

    R_output_file = base_filename.replace('.nc', '.R_samples.nc')
    R_samples.to_netcdf(R_output_file)

    D_output_file = base_filename.replace('.nc', '.D_samples.nc')
    D_samples.to_netcdf(D_output_file)


if __name__ == '__main__':
    main()
