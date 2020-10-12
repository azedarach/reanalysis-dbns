"""
Provides unit tests for sampler diagnostics.
"""

# License: MIT

from __future__ import absolute_import, division

import arviz as az
import numpy as np
import pytest
import scipy.optimize as so
import scipy.special as sp
import scipy.stats as ss

import reanalysis_dbns.models as rdm

from reanalysis_dbns.models.sampler_diagnostics import (
    _count_model_transitions, _estimate_dirichlet_shape_parameters,
    _invert_digamma, _sample_stationary_distributions)


def test_rjmcmc_rhat_single_model():
    """Test calculation of convergence diagnostics for RJMCMC."""

    random_seed = 0
    rng = np.random.default_rng(random_seed)

    n_chains = 4
    n_iter = 10

    k = np.zeros((n_chains, n_iter))
    theta = rng.uniform(size=(n_chains, n_iter))

    rhat_result = rdm.rjmcmc_rhat(k, theta, split=True)
    expected_rhat = az.rhat(
        az.convert_to_dataset(theta[:, :, np.newaxis]),
        method='split')['x'].data

    assert rhat_result['PSRF1'].shape == expected_rhat.shape
    assert np.allclose(rhat_result['PSRF1'], expected_rhat)

    model_indicators = np.array([2])

    with pytest.raises(ValueError):
        rhat_result = rdm.rjmcmc_rhat(
            k, theta, model_indicators=model_indicators, split=True)

    n_parameters = 3
    theta = rng.uniform(size=(n_chains, n_iter, n_parameters))

    rhat_result = rdm.rjmcmc_rhat(k, theta, split=True)
    expected_rhat = az.rhat(
        az.convert_to_dataset(theta), method='split')['x'].data

    assert rhat_result['PSRF1'].shape == expected_rhat.shape
    assert np.allclose(rhat_result['PSRF1'], expected_rhat)


def test_rjmcmc_rhat_single_parameter_no_split():  # noqa: C901
    """Test RJMCMC with a single parameter."""

    random_seed = 0
    rng = np.random.default_rng(random_seed)

    n_chains = 4
    n_iter = 10

    k = np.zeros((n_chains, n_iter))
    k[:, 5:] = 1
    theta = rng.uniform(size=(n_chains, n_iter))

    rhat_result = rdm.rjmcmc_rhat(k, theta, split=False)

    k = k[:, 5:]
    theta = theta[:, 5:]

    present_models = np.unique(k)
    n_models = np.size(present_models)

    theta_bar_cm = np.zeros((n_chains, n_models))
    theta_bar_c = np.zeros((n_chains,))
    theta_bar_m = np.zeros((n_models,))
    theta_bar = np.atleast_1d(np.mean(theta))

    for c in range(n_chains):
        theta_bar_c[c] = np.mean(theta[c])
        for m in range(n_models):
            mask = k[c] == present_models[m]
            theta_bar_m[m] += np.sum(theta[c, mask])
            theta_bar_cm[c, m] = np.mean(theta[c, mask])

    for m in range(n_models):
        Rm = np.sum(k == present_models[m])
        theta_bar_m[m] = theta_bar_m[m] / Rm

    Vhat = np.zeros((1,))
    Wc = np.zeros((1,))
    Wm = np.zeros((1,))
    WmWc = np.zeros((1,))
    for c in range(n_chains):
        for m in range(n_models):
            mask = k[c] == present_models[m]
            Vhat[0] += np.sum((theta[c, mask] - theta_bar)**2)
            Wc[0] += np.sum((theta[c, mask] - theta_bar_c[c])**2)
            Wm[0] += np.sum((theta[c, mask] - theta_bar_m[m])**2)
            WmWc[0] += np.sum((theta[c, mask] - theta_bar_cm[c, m])**2)

    Vhat /= (n_chains * k.shape[1] - 1.0)
    Wc /= (n_chains * (k.shape[1] - 1.0))
    Wm /= (n_chains * k.shape[1] - n_models)
    WmWc /= (n_chains * (k.shape[1] - n_models))

    expected_psrf1 = Vhat / Wc
    expected_psrf2 = Wm / WmWc

    assert np.allclose(rhat_result['Vhat'], Vhat)
    assert np.allclose(rhat_result['Wc'], Wc)
    assert np.allclose(rhat_result['Wm'], Wm)
    assert np.allclose(rhat_result['WmWc'], WmWc)
    assert np.allclose(rhat_result['PSRF1'], expected_psrf1)
    assert np.allclose(rhat_result['PSRF2'], expected_psrf2)

    k = np.zeros((n_chains, n_iter))
    k[:, ::2] = 1
    theta = rng.uniform(size=(n_chains, n_iter))

    rhat_result = rdm.rjmcmc_rhat(k, theta, split=False)

    n_models = np.size(np.unique(k))
    k = k[:, 5:]
    theta = theta[:, 5:]

    theta_bar_cm = np.zeros((n_chains, n_models))
    theta_bar_c = np.zeros((n_chains,))
    theta_bar_m = np.zeros((n_models,))
    theta_bar = np.atleast_1d(np.mean(theta))

    for c in range(n_chains):
        theta_bar_c[c] = np.mean(theta[c])
        for m in range(n_models):
            mask = k[c] == m
            theta_bar_m[m] += np.sum(theta[c, mask])
            theta_bar_cm[c, m] = np.mean(theta[c, mask])

    for m in range(n_models):
        Rm = np.sum(k == m)
        theta_bar_m[m] = theta_bar_m[m] / Rm

    Vhat = np.zeros((1,))
    Wc = np.zeros((1,))
    Wm = np.zeros((1,))
    WmWc = np.zeros((1,))
    for c in range(n_chains):
        for m in range(n_models):
            mask = k[c] == m
            Vhat[0] += np.sum((theta[c, mask] - theta_bar)**2)
            Wc[0] += np.sum((theta[c, mask] - theta_bar_c[c])**2)
            Wm[0] += np.sum((theta[c, mask] - theta_bar_m[m])**2)
            WmWc[0] += np.sum((theta[c, mask] - theta_bar_cm[c, m])**2)

    Vhat /= (n_chains * k.shape[1] - 1.0)
    Wc /= (n_chains * (k.shape[1] - 1.0))
    Wm /= (n_chains * k.shape[1] - n_models)
    WmWc /= (n_chains * (k.shape[1] - n_models))

    expected_psrf1 = Vhat / Wc
    expected_psrf2 = Wm / WmWc

    assert np.allclose(rhat_result['Vhat'], Vhat)
    assert np.allclose(rhat_result['Wc'], Wc)
    assert np.allclose(rhat_result['Wm'], Wm)
    assert np.allclose(rhat_result['WmWc'], WmWc)
    assert np.allclose(rhat_result['PSRF1'], expected_psrf1)
    assert np.allclose(rhat_result['PSRF2'], expected_psrf2)


def test_rjmcmc_rhat_single_parameter_split():  # noqa: C901
    """Test RJMCMC with a single parameter."""

    random_seed = 0
    rng = np.random.default_rng(random_seed)

    n_chains = 4
    n_iter = 20

    initial_k = np.zeros((n_chains, n_iter))
    initial_k[:, ::2] = 1
    initial_k[:, ::3] = 2
    initial_theta = rng.uniform(size=(n_chains, n_iter))

    rhat_result = rdm.rjmcmc_rhat(initial_k, initial_theta, split=True)

    n_models = np.size(np.unique(initial_k))

    k = np.zeros((2 * n_chains, n_iter // 2))
    theta = np.zeros((2 * n_chains, n_iter // 2))
    for i in range(n_chains):
        k[2 * i] = initial_k[i, :n_iter // 2]
        k[2 * i + 1] = initial_k[i, n_iter // 2:]
        theta[2 * i] = initial_theta[i, :n_iter // 2]
        theta[2 * i + 1] = initial_theta[i, n_iter // 2:]

    theta_bar_cm = np.zeros((2 * n_chains, n_models))
    theta_bar_c = np.zeros((2 * n_chains,))
    theta_bar_m = np.zeros((n_models,))
    theta_bar = np.atleast_1d(np.mean(theta))

    for c in range(2 * n_chains):
        theta_bar_c[c] = np.mean(theta[c])
        for m in range(n_models):
            mask = k[c] == m
            theta_bar_m[m] += np.sum(theta[c, mask])
            theta_bar_cm[c, m] = np.mean(theta[c, mask])

    for m in range(n_models):
        Rm = np.sum(k == m)
        theta_bar_m[m] = theta_bar_m[m] / Rm

    Vhat = np.zeros((1,))
    Wc = np.zeros((1,))
    Wm = np.zeros((1,))
    WmWc = np.zeros((1,))
    for c in range(2 * n_chains):
        for m in range(n_models):
            mask = k[c] == m
            Vhat[0] += np.sum((theta[c, mask] - theta_bar)**2)
            Wc[0] += np.sum((theta[c, mask] - theta_bar_c[c])**2)
            Wm[0] += np.sum((theta[c, mask] - theta_bar_m[m])**2)
            WmWc[0] += np.sum((theta[c, mask] - theta_bar_cm[c, m])**2)

    Vhat /= (2 * n_chains * k.shape[1] - 1.0)
    Wc /= (2 * n_chains * (k.shape[1] - 1.0))
    Wm /= (2 * n_chains * k.shape[1] - n_models)
    WmWc /= (2 * n_chains * (k.shape[1] - n_models))

    expected_psrf1 = Vhat / Wc
    expected_psrf2 = Wm / WmWc

    assert np.allclose(rhat_result['Vhat'], Vhat)
    assert np.allclose(rhat_result['Wc'], Wc)
    assert np.allclose(rhat_result['Wm'], Wm)
    assert np.allclose(rhat_result['WmWc'], WmWc)
    assert np.allclose(rhat_result['PSRF1'], expected_psrf1)
    assert np.allclose(rhat_result['PSRF2'], expected_psrf2)

    n_chains = 3
    n_iter = 21

    initial_k = np.zeros((n_chains, n_iter))
    initial_k[:, ::2] = 1
    initial_k[:, ::3] = 2
    initial_theta = rng.uniform(size=(n_chains, n_iter))

    rhat_result = rdm.rjmcmc_rhat(initial_k, initial_theta, split=True)

    n_models = np.size(np.unique(initial_k))

    k = np.zeros((2 * n_chains, (n_iter - 1) // 2))
    theta = np.zeros((2 * n_chains, (n_iter - 1) // 2))
    for i in range(n_chains):
        k[2 * i] = initial_k[i, 1:(n_iter - 1) // 2 + 1]
        k[2 * i + 1] = initial_k[i, 1 + (n_iter - 1) // 2:]
        theta[2 * i] = initial_theta[i, 1:(n_iter - 1) // 2 + 1]
        theta[2 * i + 1] = initial_theta[i, 1 + (n_iter - 1) // 2:]

    theta_bar_cm = np.zeros((2 * n_chains, n_models))
    theta_bar_c = np.zeros((2 * n_chains,))
    theta_bar_m = np.zeros((n_models,))
    theta_bar = np.atleast_1d(np.mean(theta))

    for c in range(2 * n_chains):
        theta_bar_c[c] = np.mean(theta[c])
        for m in range(n_models):
            mask = k[c] == m
            theta_bar_m[m] += np.sum(theta[c, mask])
            theta_bar_cm[c, m] = np.mean(theta[c, mask])

    for m in range(n_models):
        Rm = np.sum(k == m)
        theta_bar_m[m] = theta_bar_m[m] / Rm

    Vhat = np.zeros((1,))
    Wc = np.zeros((1,))
    Wm = np.zeros((1,))
    WmWc = np.zeros((1,))
    for c in range(2 * n_chains):
        for m in range(n_models):
            mask = k[c] == m
            Vhat[0] += np.sum((theta[c, mask] - theta_bar)**2)
            Wc[0] += np.sum((theta[c, mask] - theta_bar_c[c])**2)
            Wm[0] += np.sum((theta[c, mask] - theta_bar_m[m])**2)
            WmWc[0] += np.sum((theta[c, mask] - theta_bar_cm[c, m])**2)

    Vhat /= (2 * n_chains * k.shape[1] - 1.0)
    Wc /= (2 * n_chains * (k.shape[1] - 1.0))
    Wm /= (2 * n_chains * k.shape[1] - n_models)
    WmWc /= (2 * n_chains * (k.shape[1] - n_models))

    expected_psrf1 = Vhat / Wc
    expected_psrf2 = Wm / WmWc

    assert np.allclose(rhat_result['Vhat'], Vhat)
    assert np.allclose(rhat_result['Wc'], Wc)
    assert np.allclose(rhat_result['Wm'], Wm)
    assert np.allclose(rhat_result['WmWc'], WmWc)
    assert np.allclose(rhat_result['PSRF1'], expected_psrf1)
    assert np.allclose(rhat_result['PSRF2'], expected_psrf2)


def test_rjmcmc_rhat_multiple_parameter_no_split():  # noqa: C901
    """Test RJMCMC with multiple parameters."""

    random_seed = 0
    rng = np.random.default_rng(random_seed)

    n_chains = 7
    n_iter = 21
    n_parameters = 4

    initial_k = np.zeros((n_chains, n_iter))
    initial_k[:, ::2] = 1
    initial_theta = rng.uniform(size=(n_chains, n_iter, n_parameters))

    rhat_result = rdm.rjmcmc_rhat(initial_k, initial_theta, split=False)

    n_models = np.size(np.unique(initial_k))

    k = initial_k[:, n_iter // 2:]
    theta = initial_theta[:, n_iter // 2:, :]

    theta_bar_cm = np.zeros((n_chains, n_models, n_parameters))
    theta_bar_c = np.zeros((n_chains, n_parameters))
    theta_bar_m = np.zeros((n_models, n_parameters))
    theta_bar = np.zeros((n_parameters,))

    for c in range(n_chains):
        for m in range(n_models):
            mask = k[c] == m
            for i in range(n_parameters):
                theta_bar_c[c, i] = np.mean(theta[c, :, i])
                theta_bar[i] += np.sum(theta[c, mask, i])
                theta_bar_m[m, i] += np.sum(theta[c, mask, i])
                theta_bar_cm[c, m, i] = np.mean(theta[c, mask, i])

    theta_bar /= (theta.shape[0] * theta.shape[1])
    for m in range(n_models):
        Rm = np.sum(k == m)
        theta_bar_m[m] = theta_bar_m[m] / Rm

    Vhat = np.zeros((n_parameters,))
    Wc = np.zeros((n_parameters,))
    Wm = np.zeros((n_parameters,))
    WmWc = np.zeros((n_parameters,))
    for c in range(n_chains):
        for m in range(n_models):
            mask = k[c] == m
            for i in range(n_parameters):
                Vhat[i] += np.sum((theta[c, mask, i] - theta_bar[i])**2)
                Wc[i] += np.sum((theta[c, mask, i] - theta_bar_c[c, i])**2)
                Wm[i] += np.sum((theta[c, mask, i] - theta_bar_m[m, i])**2)
                WmWc[i] += np.sum(
                    (theta[c, mask, i] - theta_bar_cm[c, m, i])**2)

    Vhat /= (n_chains * k.shape[1] - 1.0)
    Wc /= (n_chains * (k.shape[1] - 1.0))
    Wm /= (n_chains * k.shape[1] - n_models)
    WmWc /= (n_chains * (k.shape[1] - n_models))

    expected_psrf1 = Vhat / Wc
    expected_psrf2 = Wm / WmWc

    assert np.allclose(np.diag(rhat_result['Vhat']), Vhat)
    assert np.allclose(np.diag(rhat_result['Wc']), Wc)
    assert np.allclose(np.diag(rhat_result['Wm']), Wm)
    assert np.allclose(np.diag(rhat_result['WmWc']), WmWc)
    assert np.allclose(rhat_result['PSRF1'], expected_psrf1)
    assert np.allclose(rhat_result['PSRF2'], expected_psrf2)

    Vhat = np.zeros((n_parameters, n_parameters))
    Wc = np.zeros((n_parameters, n_parameters))
    Wm = np.zeros((n_parameters, n_parameters))
    WmWc = np.zeros((n_parameters, n_parameters))

    for i in range(n_parameters):
        for j in range(n_parameters):

            for c in range(n_chains):
                for m in range(n_models):
                    mask = k[c] == m

                    Vhat[i, j] += np.sum(
                        (theta[c, mask, i] - theta_bar[i]) *
                        (theta[c, mask, j] - theta_bar[j]))
                    Wc[i, j] += np.sum(
                        (theta[c, mask, i] - theta_bar_c[c, i]) *
                        (theta[c, mask, j] - theta_bar_c[c, j]))
                    Wm[i, j] += np.sum(
                        (theta[c, mask, i] - theta_bar_m[m, i]) *
                        (theta[c, mask, j] - theta_bar_m[m, i]))
                    WmWc[i, j] += np.sum(
                        (theta[c, mask, i] - theta_bar_cm[c, m, i]) *
                        (theta[c, mask, j] - theta_bar_cm[c, m, j]))

    Vhat /= (n_chains * k.shape[1] - 1.0)
    Wc /= (n_chains * (k.shape[1] - 1.0))
    Wm /= (n_chains * k.shape[1] - n_models)
    WmWc /= (n_chains * (k.shape[1] - n_models))

    expected_mpsrf1 = np.max(
        np.linalg.eigvals(np.dot(np.linalg.inv(Wc), Vhat)))
    expected_mpsrf2 = np.max(
        np.linalg.eigvals(np.dot(np.linalg.inv(WmWc), Wm)))

    assert np.allclose(rhat_result['Vhat'], Vhat)
    assert np.allclose(rhat_result['Wc'], Wc)
    assert np.allclose(rhat_result['Wm'], Wm)
    assert np.allclose(rhat_result['WmWc'], WmWc)

    assert np.allclose(rhat_result['MPSRF1'], expected_mpsrf1)
    assert np.allclose(rhat_result['MPSRF2'], expected_mpsrf2)


def test_rjmcmc_rhat_multiple_parameter_split():  # noqa: C901
    """Test RJMCMC with multiple parameters."""

    random_seed = 0
    rng = np.random.default_rng(random_seed)

    n_chains = 5
    n_iter = 20
    n_parameters = 5

    initial_k = np.zeros((n_chains, n_iter))
    initial_k[:, ::2] = 1
    initial_k[:, ::3] = 2
    initial_theta = rng.uniform(size=(n_chains, n_iter, n_parameters))

    rhat_result = rdm.rjmcmc_rhat(initial_k, initial_theta, split=True)

    n_models = np.size(np.unique(initial_k))

    k = np.zeros((2 * n_chains, n_iter // 2))
    theta = np.zeros((2 * n_chains, n_iter // 2, n_parameters))
    for i in range(n_chains):
        k[2 * i] = initial_k[i, :n_iter // 2]
        k[2 * i + 1] = initial_k[i, n_iter // 2:]
        theta[2 * i] = initial_theta[i, :n_iter // 2, :]
        theta[2 * i + 1] = initial_theta[i, n_iter // 2:, :]

    theta_bar_cm = np.zeros((2 * n_chains, n_models, n_parameters))
    theta_bar_c = np.zeros((2 * n_chains, n_parameters))
    theta_bar_m = np.zeros((n_models, n_parameters))
    theta_bar = np.zeros((n_parameters,))

    for c in range(2 * n_chains):
        for m in range(n_models):
            mask = k[c] == m
            for i in range(n_parameters):
                theta_bar_c[c, i] = np.mean(theta[c, :, i])
                theta_bar[i] += np.sum(theta[c, mask, i])
                theta_bar_m[m, i] += np.sum(theta[c, mask, i])
                theta_bar_cm[c, m, i] = np.mean(theta[c, mask, i])

    theta_bar /= (theta.shape[0] * theta.shape[1])
    for m in range(n_models):
        Rm = np.sum(k == m)
        theta_bar_m[m] = theta_bar_m[m] / Rm

    Vhat = np.zeros((n_parameters,))
    Wc = np.zeros((n_parameters,))
    Wm = np.zeros((n_parameters,))
    WmWc = np.zeros((n_parameters,))
    for c in range(2 * n_chains):
        for m in range(n_models):
            mask = k[c] == m
            for i in range(n_parameters):
                Vhat[i] += np.sum((theta[c, mask, i] - theta_bar[i])**2)
                Wc[i] += np.sum((theta[c, mask, i] - theta_bar_c[c, i])**2)
                Wm[i] += np.sum((theta[c, mask, i] - theta_bar_m[m, i])**2)
                WmWc[i] += np.sum(
                    (theta[c, mask, i] - theta_bar_cm[c, m, i])**2)

    Vhat /= (2 * n_chains * k.shape[1] - 1.0)
    Wc /= (2 * n_chains * (k.shape[1] - 1.0))
    Wm /= (2 * n_chains * k.shape[1] - n_models)
    WmWc /= (2 * n_chains * (k.shape[1] - n_models))

    expected_psrf1 = Vhat / Wc
    expected_psrf2 = Wm / WmWc

    assert np.allclose(np.diag(rhat_result['Vhat']), Vhat)
    assert np.allclose(np.diag(rhat_result['Wc']), Wc)
    assert np.allclose(np.diag(rhat_result['Wm']), Wm)
    assert np.allclose(np.diag(rhat_result['WmWc']), WmWc)
    assert np.allclose(rhat_result['PSRF1'], expected_psrf1)
    assert np.allclose(rhat_result['PSRF2'], expected_psrf2)

    Vhat = np.zeros((n_parameters, n_parameters))
    Wc = np.zeros((n_parameters, n_parameters))
    Wm = np.zeros((n_parameters, n_parameters))
    WmWc = np.zeros((n_parameters, n_parameters))

    for i in range(n_parameters):
        for j in range(n_parameters):

            for c in range(2 * n_chains):
                for m in range(n_models):
                    mask = k[c] == m

                    Vhat[i, j] += np.sum(
                        (theta[c, mask, i] - theta_bar[i]) *
                        (theta[c, mask, j] - theta_bar[j]))
                    Wc[i, j] += np.sum(
                        (theta[c, mask, i] - theta_bar_c[c, i]) *
                        (theta[c, mask, j] - theta_bar_c[c, j]))
                    Wm[i, j] += np.sum(
                        (theta[c, mask, i] - theta_bar_m[m, i]) *
                        (theta[c, mask, j] - theta_bar_m[m, i]))
                    WmWc[i, j] += np.sum(
                        (theta[c, mask, i] - theta_bar_cm[c, m, i]) *
                        (theta[c, mask, j] - theta_bar_cm[c, m, j]))

    Vhat /= (2 * n_chains * k.shape[1] - 1.0)
    Wc /= (2 * n_chains * (k.shape[1] - 1.0))
    Wm /= (2 * n_chains * k.shape[1] - n_models)
    WmWc /= (2 * n_chains * (k.shape[1] - n_models))

    expected_mpsrf1 = np.max(
        np.linalg.eigvals(np.dot(np.linalg.inv(Wc), Vhat)))
    expected_mpsrf2 = np.max(
        np.linalg.eigvals(np.dot(np.linalg.inv(WmWc), Wm)))

    assert np.allclose(rhat_result['Vhat'], Vhat)
    assert np.allclose(rhat_result['Wc'], Wc)
    assert np.allclose(rhat_result['Wm'], Wm)
    assert np.allclose(rhat_result['WmWc'], WmWc)

    assert np.allclose(rhat_result['MPSRF1'], expected_mpsrf1)
    assert np.allclose(rhat_result['MPSRF2'], expected_mpsrf2)

    n_chains = 5
    n_iter = 37
    n_parameters = 5

    initial_k = np.zeros((n_chains, n_iter))
    initial_k[:, ::2] = 1
    initial_k[:, ::3] = 2
    initial_k[:, ::5] = 3
    initial_theta = rng.uniform(size=(n_chains, n_iter, n_parameters))

    rhat_result = rdm.rjmcmc_rhat(initial_k, initial_theta, split=True)

    n_models = np.size(np.unique(initial_k))

    k = np.zeros((2 * n_chains, (n_iter - 1) // 2))
    theta = np.zeros((2 * n_chains, (n_iter - 1) // 2, n_parameters))
    for i in range(n_chains):
        k[2 * i] = initial_k[i, 1:(n_iter - 1) // 2 + 1]
        k[2 * i + 1] = initial_k[i, (n_iter - 1) // 2 + 1:]
        theta[2 * i] = initial_theta[i, 1:(n_iter - 1) // 2 + 1, :]
        theta[2 * i + 1] = initial_theta[i, 1 + (n_iter - 1) // 2:, :]

    theta_bar_cm = np.zeros((2 * n_chains, n_models, n_parameters))
    theta_bar_c = np.zeros((2 * n_chains, n_parameters))
    theta_bar_m = np.zeros((n_models, n_parameters))
    theta_bar = np.zeros((n_parameters,))

    for c in range(2 * n_chains):
        for m in range(n_models):
            mask = k[c] == m
            for i in range(n_parameters):
                theta_bar_c[c, i] = np.mean(theta[c, :, i])
                theta_bar[i] += np.sum(theta[c, mask, i])
                theta_bar_m[m, i] += np.sum(theta[c, mask, i])
                theta_bar_cm[c, m, i] = np.mean(theta[c, mask, i])

    theta_bar /= (theta.shape[0] * theta.shape[1])
    for m in range(n_models):
        Rm = np.sum(k == m)
        theta_bar_m[m] = theta_bar_m[m] / Rm

    Vhat = np.zeros((n_parameters,))
    Wc = np.zeros((n_parameters,))
    Wm = np.zeros((n_parameters,))
    WmWc = np.zeros((n_parameters,))
    for c in range(2 * n_chains):
        for m in range(n_models):
            mask = k[c] == m
            for i in range(n_parameters):
                Vhat[i] += np.sum((theta[c, mask, i] - theta_bar[i])**2)
                Wc[i] += np.sum((theta[c, mask, i] - theta_bar_c[c, i])**2)
                Wm[i] += np.sum((theta[c, mask, i] - theta_bar_m[m, i])**2)
                WmWc[i] += np.sum(
                    (theta[c, mask, i] - theta_bar_cm[c, m, i])**2)

    Vhat /= (2 * n_chains * k.shape[1] - 1.0)
    Wc /= (2 * n_chains * (k.shape[1] - 1.0))
    Wm /= (2 * n_chains * k.shape[1] - n_models)
    WmWc /= (2 * n_chains * (k.shape[1] - n_models))

    expected_psrf1 = Vhat / Wc
    expected_psrf2 = Wm / WmWc

    assert np.allclose(np.diag(rhat_result['Vhat']), Vhat)
    assert np.allclose(np.diag(rhat_result['Wc']), Wc)
    assert np.allclose(np.diag(rhat_result['Wm']), Wm)
    assert np.allclose(np.diag(rhat_result['WmWc']), WmWc)
    assert np.allclose(rhat_result['PSRF1'], expected_psrf1)
    assert np.allclose(rhat_result['PSRF2'], expected_psrf2)

    Vhat = np.zeros((n_parameters, n_parameters))
    Wc = np.zeros((n_parameters, n_parameters))
    Wm = np.zeros((n_parameters, n_parameters))
    WmWc = np.zeros((n_parameters, n_parameters))

    for i in range(n_parameters):
        for j in range(n_parameters):

            for c in range(2 * n_chains):
                for m in range(n_models):
                    mask = k[c] == m

                    Vhat[i, j] += np.sum(
                        (theta[c, mask, i] - theta_bar[i]) *
                        (theta[c, mask, j] - theta_bar[j]))
                    Wc[i, j] += np.sum(
                        (theta[c, mask, i] - theta_bar_c[c, i]) *
                        (theta[c, mask, j] - theta_bar_c[c, j]))
                    Wm[i, j] += np.sum(
                        (theta[c, mask, i] - theta_bar_m[m, i]) *
                        (theta[c, mask, j] - theta_bar_m[m, i]))
                    WmWc[i, j] += np.sum(
                        (theta[c, mask, i] - theta_bar_cm[c, m, i]) *
                        (theta[c, mask, j] - theta_bar_cm[c, m, j]))

    Vhat /= (2 * n_chains * k.shape[1] - 1.0)
    Wc /= (2 * n_chains * (k.shape[1] - 1.0))
    Wm /= (2 * n_chains * k.shape[1] - n_models)
    WmWc /= (2 * n_chains * (k.shape[1] - n_models))

    expected_mpsrf1 = np.max(
        np.linalg.eigvals(np.dot(np.linalg.inv(Wc), Vhat)))
    expected_mpsrf2 = np.max(
        np.linalg.eigvals(np.dot(np.linalg.inv(WmWc), Wm)))

    assert np.allclose(rhat_result['Vhat'], Vhat)
    assert np.allclose(rhat_result['Wc'], Wc)
    assert np.allclose(rhat_result['Wm'], Wm)
    assert np.allclose(rhat_result['WmWc'], WmWc)

    assert np.allclose(rhat_result['MPSRF1'], expected_mpsrf1)
    assert np.allclose(rhat_result['MPSRF2'], expected_mpsrf2)


def test_invert_digamma():
    """Test inversion of digamma function."""

    tolerance = 1e-12

    x = 1.0
    y = sp.digamma(x)

    x_est = _invert_digamma(y, tolerance=tolerance)
    assert np.abs(x - x_est) < tolerance

    x = 4.3531
    y = sp.digamma(x)

    x_est = _invert_digamma(y, tolerance=tolerance)
    assert np.abs(x - x_est) < tolerance

    x = 0.2
    y = sp.digamma(x)

    x_est = _invert_digamma(y, tolerance=tolerance)
    assert np.abs(x - x_est) < tolerance

    x = np.ones(())
    y = sp.digamma(x)

    x_est = _invert_digamma(y, tolerance=tolerance)
    assert np.all(np.abs(x - x_est) < tolerance)

    x = np.ones(3)
    y = sp.digamma(x)

    x_est = _invert_digamma(y, tolerance=tolerance)
    assert np.all(np.abs(x - x_est) < tolerance)

    x = np.random.uniform(0.6, 10.0, size=(20,))
    y = sp.digamma(x)

    x_est = _invert_digamma(y, tolerance=tolerance)
    assert np.all(np.abs(x - x_est) < tolerance)

    x = np.random.uniform(0.0001, 0.6, size=(30,))
    y = sp.digamma(x)

    x_est = _invert_digamma(y, tolerance=tolerance)
    assert np.all(np.abs(x - x_est) < tolerance)

    x = np.ones((5, 5))
    y = sp.digamma(x)

    x_est = _invert_digamma(y, tolerance=tolerance)
    assert np.all(np.abs(x - x_est) < tolerance)

    x = np.random.uniform(0.6, 10.0, size=(20, 5, 11))
    y = sp.digamma(x)

    x_est = _invert_digamma(y, tolerance=tolerance)
    assert np.all(np.abs(x - x_est) < tolerance)

    x = np.random.uniform(0.0001, 0.6, size=(30, 20, 4, 8))
    y = sp.digamma(x)

    x_est = _invert_digamma(y, tolerance=tolerance)
    assert np.all(np.abs(x - x_est) < tolerance)


def test_estimate_dirichlet_shape_parameters():
    """Test maximum likelihood fit of Dirichlet distribution."""

    random_seed = 0
    random_state = np.random.default_rng(random_seed)

    def dirichlet_log_likelihood(alpha, p):
        return np.sum([ss.dirichlet.logpdf(pi, alpha) for pi in p])

    n_features = 3
    n_samples = 500

    alpha = np.ones(n_features)
    p = ss.dirichlet.rvs(alpha, size=(n_samples,), random_state=random_state)

    alpha_hat = _estimate_dirichlet_shape_parameters(p)

    alpha_0 = random_state.uniform(0.0, 2.0, size=(n_features,))

    def _objective_one(x):
        return -dirichlet_log_likelihood(x, p)

    tol = 1e-6
    bounds = so.Bounds(0, np.inf)
    sol = so.minimize(_objective_one, alpha_0, bounds=bounds,
                      method='trust-constr', options={'xtol': tol})
    expected_alpha_hat = sol.x

    assert np.allclose(alpha_hat, expected_alpha_hat, atol=tol)

    n_features = 6
    n_samples = 250

    alpha = random_state.uniform(0.5, 4.0, size=(n_features,))
    p = ss.dirichlet.rvs(alpha, size=(n_samples,), random_state=random_state)

    alpha_hat = _estimate_dirichlet_shape_parameters(p)

    alpha_0 = random_state.uniform(0.0, 2.0, size=(n_features,))

    def _objective_two(x):
        return -dirichlet_log_likelihood(x, p)

    tol = 1e-6
    bounds = so.Bounds(0, np.inf)
    sol = so.minimize(_objective_two, alpha_0, bounds=bounds,
                      method='trust-constr', options={'xtol': tol})
    expected_alpha_hat = sol.x

    assert np.allclose(alpha_hat, expected_alpha_hat, atol=tol)


def test_count_model_transitions():
    """Test counting model transitions."""

    k = np.array([[0, 0, 1, 2, 1, 1, 0]], dtype='i8')

    n = _count_model_transitions(k)

    expected_n = np.array([[[1, 1, 0], [1, 1, 1], [0, 1, 0]]])

    assert np.all(n == expected_n)

    n = _count_model_transitions(k, sparse=True)

    expected_n = np.array([[[1, 1, 0], [1, 1, 1], [0, 1, 0]]])

    assert np.all(n.toarray() == expected_n)

    model_indicators = np.array([0, 1, 2, 3])

    n = _count_model_transitions(k, model_indicators=model_indicators)

    expected_n = np.array([[[1, 1, 0, 0],
                            [1, 1, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 0]]])

    assert np.all(n == expected_n)

    n = _count_model_transitions(k, model_indicators=model_indicators,
                                 sparse=True)

    expected_n = np.array([[[1, 1, 0, 0],
                            [1, 1, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 0]]])

    assert np.all(n.toarray() == expected_n)


def test_sample_stationary_distributions():
    """Test sampling of stationary distributions."""

    random_seed = 0
    rng = np.random.default_rng(random_seed)

    P = np.array([[0.2, 0.8], [0.5, 0.5]])
    exact_pi = np.array([5.0 / 13.0, 8.0 / 13.0])

    assert np.allclose(np.dot(exact_pi, P), exact_pi)
    assert np.abs(np.sum(exact_pi) - 1.0) < 1e-10

    n_chains = 4
    n_iter = 500

    k = np.empty((n_chains, n_iter), dtype=np.uint64)
    for i in range(n_chains):
        ki = np.empty((2 * n_iter), dtype=np.uint64)
        ki[0] = rng.choice(2)
        for t in range(1, 2 * n_iter):
            p = P[ki[t - 1]]
            ki[t] = rng.choice(2, p=p)
        k[i] = ki[-n_iter:]

    n_samples = 1000

    samples, epsilon = _sample_stationary_distributions(
        k, epsilon=0.0, n_samples=n_samples)

    assert samples.shape == (n_samples, 2)
    assert np.allclose(np.sum(samples, axis=1), 1.0)
    assert np.allclose(epsilon, np.array([0., 0.]))

    assert np.allclose(np.mean(samples, axis=0), exact_pi, atol=5e-2)

    model_indicators = np.array([0, 1, 2])
    samples, epsilon = _sample_stationary_distributions(
        k, epsilon=1e-6, n_samples=n_samples,
        model_indicators=model_indicators)

    assert samples.shape == (n_samples, 3)
    assert np.allclose(np.sum(samples, axis=1), 1.0)
    assert np.allclose(epsilon, np.array([1e-6, 1e-6, 1e-6]))

    assert np.allclose(np.mean(samples[:, :2], axis=0), exact_pi, atol=5e-2)

    k2 = k.copy()
    for i in range(n_chains):
        k2[i, k2[i] == 1] = 2

    model_indicators = np.array([0, 1, 2])
    samples, epsilon = _sample_stationary_distributions(
        k2, epsilon=1e-6, n_samples=n_samples,
        model_indicators=model_indicators)

    assert samples.shape == (n_samples, 3)
    assert np.allclose(np.sum(samples, axis=1), 1.0)
    assert np.allclose(epsilon, np.array([1e-6, 1e-6, 1e-6]))

    assert np.allclose(np.mean(samples, axis=0)[[0, 2]], exact_pi, atol=5e-2)

    samples, epsilon = _sample_stationary_distributions(
        k, n_samples=n_samples)

    assert samples.shape == (n_samples, 2)
    assert np.allclose(np.sum(samples, axis=1), 1.0)
    assert np.allclose(epsilon, np.array([0.5, 0.5]))

    assert np.allclose(np.mean(samples, axis=0), exact_pi, atol=5e-2)

    model_indicators = np.array([0, 1, 2])
    samples, epsilon = _sample_stationary_distributions(
        k, n_samples=n_samples, model_indicators=model_indicators)

    assert samples.shape == (n_samples, 3)
    assert np.allclose(np.sum(samples, axis=1), 1.0)
    assert np.allclose(epsilon, np.array([0.5, 0.5, 1e-10]))

    assert np.allclose(np.mean(samples[:, :2], axis=0), exact_pi, atol=5e-2)

    samples, epsilon = _sample_stationary_distributions(
        k2, n_samples=n_samples, model_indicators=model_indicators)

    assert samples.shape == (n_samples, 3)
    assert np.allclose(np.sum(samples, axis=1), 1.0)
    assert np.allclose(epsilon, np.array([0.5, 1e-10, 0.5]))

    assert np.allclose(np.mean(samples, axis=0)[[0, 2]], exact_pi, atol=5e-2)


def test_estimate_stationary_distributions():
    """Test estimation of stationary distributions and ESS calculation."""

    random_seed = 0
    rng = np.random.default_rng(random_seed)

    n_chains = 4
    n_iter = 1000
    n_models = 3

    k = np.empty((n_chains, n_iter), dtype=np.uint64)
    for i in range(n_chains):
        for t in range(n_iter):
            k[i, t] = rng.choice(n_models)

    n_samples = 10000
    result = rdm.estimate_stationary_distribution(
        k, tolerance=1e-3, model_indicators=np.arange(3),
        min_epsilon=1e-10, n_samples=n_samples, random_state=random_seed)

    assert result['pi'].shape == (n_samples, 3)
    row_sums = np.sum(result['pi'], axis=1)
    mask = np.isfinite(row_sums)
    assert np.allclose(row_sums[mask], 1.0)
    assert np.abs(result['ess'] - n_chains * n_iter) < 200
    assert result['n_models'] == 3
    assert result['n_observed_models'] == 3

    n_samples = 10000
    result = rdm.estimate_stationary_distribution(
        k, tolerance=1e-3, model_indicators=np.arange(4),
        min_epsilon=1e-10, n_samples=n_samples, random_state=random_seed)

    assert result['pi'].shape == (n_samples, 4)
    row_sums = np.sum(result['pi'], axis=1)
    mask = np.isfinite(row_sums)
    assert np.allclose(row_sums[mask], 1.0)
    assert np.abs(result['ess'] - n_chains * n_iter) < 200
    assert result['n_models'] == 4
    assert result['n_observed_models'] == 3

    P = np.array([[0.2, 0.8], [0.5, 0.5]])
    exact_pi = np.array([5.0 / 13.0, 8.0 / 13.0])

    assert np.allclose(np.dot(exact_pi, P), exact_pi)
    assert np.abs(np.sum(exact_pi) - 1.0) < 1e-10

    n_chains = 4
    n_iter = 1000

    k = np.empty((n_chains, n_iter), dtype=np.uint64)
    for i in range(n_chains):
        ki = np.empty((2 * n_iter), dtype=np.uint64)
        ki[0] = rng.choice(2)
        for t in range(1, 2 * n_iter):
            p = P[ki[t - 1]]
            ki[t] = rng.choice(2, p=p)
        k[i] = ki[-n_iter:]

    n_samples = 10000

    result = rdm.estimate_stationary_distribution(
        k, n_samples=n_samples, tolerance=1e-3, min_epsilon=1e-10)

    assert result['pi'].shape == (n_samples, 2)
    row_sums = np.sum(result['pi'], axis=1)
    mask = np.isfinite(row_sums)
    assert np.allclose(row_sums[mask], 1.0)
    # for chains with significant anti-correlation, ESS should be
    # larger than IID samples
    assert result['ess'] > n_chains * n_iter
    assert result['n_models'] == 2
    assert result['n_observed_models'] == 2

    P = np.array([[0.8, 0.2], [0.7, 0.3]])
    exact_pi = np.array([7.0 / 9.0, 2.0 / 9.0])

    assert np.allclose(np.dot(exact_pi, P), exact_pi)
    assert np.abs(np.sum(exact_pi) - 1.0) < 1e-10

    n_chains = 4
    n_iter = 1000

    k = np.empty((n_chains, n_iter), dtype=np.uint64)
    for i in range(n_chains):
        ki = np.empty((2 * n_iter), dtype=np.uint64)
        ki[0] = rng.choice(2)
        for t in range(1, 2 * n_iter):
            p = P[ki[t - 1]]
            ki[t] = rng.choice(2, p=p)
        k[i] = ki[-n_iter:]

    n_samples = 10000

    result = rdm.estimate_stationary_distribution(
        k, n_samples=n_samples, sparse=True, tolerance=1e-3,
        min_epsilon=1e-10)

    assert result['pi'].shape == (n_samples, 2)
    row_sums = np.sum(result['pi'], axis=1)
    mask = np.isfinite(row_sums)
    assert np.allclose(row_sums[mask], 1.0)
    # for chains with significant autocorrelation, ESS should be
    # smaller than IID samples
    assert result['ess'] < n_chains * n_iter
    assert result['n_models'] == 2
    assert result['n_observed_models'] == 2
