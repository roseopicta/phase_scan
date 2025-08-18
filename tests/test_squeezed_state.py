# MIT License
#
# Copyright (c) 2025 Émilie Gillet [emilie.gillet@etu.sorbonne-universite.fr]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest

from phase_scan import gaussian_utils
from phase_scan.ml_estimation import (
    ml_covariance_estimation,
    ml_covariance_estimation_direct_ls,
)

from functools import partial
import numpy as np
import scipy as sp


@pytest.mark.parametrize(
    "estimation_fn",
    [
        partial(ml_covariance_estimation, lr=0.05, max_iterations=5000),
        ml_covariance_estimation_direct_ls,
    ],
)
def test_squeezed_state(estimation_fn):
    squeezing_dB = 6
    squeezing_angle = np.pi / 4
    squeezing_s = 10 ** (-squeezing_dB / 10)

    sigma = gaussian_utils.squeezed_vacuum(squeezing_s)
    R = gaussian_utils.rotation_symplectic(squeezing_angle)
    sigma = R @ sigma @ R.T

    np.random.seed(0)
    angles = np.linspace(0, 1, 100) * np.pi
    samples = gaussian_utils.generate_scanned_samples(
        sigma, angles, gaussian_utils.generate_samples_parallel, 1000
    )
    sigma_hat = estimation_fn(samples)
    np.testing.assert_allclose(sigma, sigma_hat, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "estimation_fn",
    [
        ml_covariance_estimation_direct_ls,
    ],
)
def test_two_independent_modes(estimation_fn):
    squeezing_dB = 6
    squeezing_angle = np.pi / 4
    squeezing_s = 10 ** (-squeezing_dB / 10)

    sigma = gaussian_utils.squeezed_vacuum(squeezing_s)
    R_1 = gaussian_utils.rotation_symplectic(squeezing_angle)
    R_2 = gaussian_utils.rotation_symplectic(squeezing_angle + np.pi / 2)
    sigma = sp.linalg.block_diag(R_1 @ sigma @ R_1.T, R_2 @ sigma @ R_2.T)

    np.random.seed(100)
    angles = np.tile(np.linspace(0, 1, 1000) * np.pi, (2, 1)).T
    print(angles.shape)
    samples = gaussian_utils.generate_scanned_samples(
        sigma, angles, gaussian_utils.generate_samples_parallel, 1000
    )
    sigma_hat = estimation_fn(samples)
    np.testing.assert_allclose(sigma, sigma_hat, atol=1e-2, rtol=1e-2)
