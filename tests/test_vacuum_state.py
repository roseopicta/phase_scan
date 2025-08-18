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
from phase_scan.ml_estimation import ml_covariance_estimation
from phase_scan.gmm_estimation import gmm_covariance_estimation

from functools import partial
import numpy as np


@pytest.mark.parametrize(
    "estimation_fn",
    [
        partial(ml_covariance_estimation, lr=0.01, max_iterations=1000),
        gmm_covariance_estimation,
    ],
)
def test_vacuum_state(estimation_fn):
    np.random.seed(0)
    sigma = gaussian_utils.vacuum()
    x_xp_p = np.array([0, np.pi / 4, np.pi / 2])
    samples = gaussian_utils.generate_scanned_samples(
        sigma, x_xp_p, gaussian_utils.generate_samples_parallel, 50_000
    )
    sigma_hat = estimation_fn(samples)
    np.testing.assert_allclose(sigma, sigma_hat, atol=1e-2, rtol=1e-2)
