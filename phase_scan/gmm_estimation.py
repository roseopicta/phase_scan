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

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Sequence, Tuple

from .ml_estimation import quadrature_projection_matrix


def gmm_covariance_estimation(
    scanning_data: List[Tuple[np.ndarray, np.ndarray]],
    num_steps=1,
    w_inv_regularization=1e4,
    w_regularization=1e-2,
) -> np.ndarray:
    """
    Estimate the covariance matrix of a Gaussian state from homodyne scanning data,
    using the generalized method of moments.

    Args:
        scanning_data: a list of tuples (angles, quadrature_data)
            - angles: np.ndarray of shape (num_modes), specifying the measurement angles
            - quadratures: np.ndarray of shape (batch_size, num_modes), the quadrature data
        num_steps: number of iterations. If num_steps = 1, compute a least-squares estimate
        w_inv_regularization: regularization for the inversion of the covariance matrix of residuals
        w_regularization: regularization for the weighted least square problem.

    Returns:
        np.ndarray: Estimated covariance matrix of shape (2 * num_modes, 2 * num_modes).
    """
    num_modes = len(scanning_data[0][0])
    angles_array = jnp.stack([item[0] for item in scanning_data])
    quadratures_array = jnp.stack([item[1] for item in scanning_data])

    @jax.jit
    def process_batch(theta, samples):
        V_k = quadrature_projection_matrix(theta)
        S_k = samples.T @ samples
        A_k = jnp.kron(V_k, V_k) * samples.shape[0]
        b_k = S_k.flatten()
        return A_k, b_k

    process_batch = jax.vmap(process_batch, in_axes=(0, 0))
    A_blocks, b_vecs = process_batch(angles_array, quadratures_array)

    # Initial least-squares estimate
    A = A_blocks.reshape(-1, A_blocks.shape[-1])
    b = b_vecs.flatten()
    sigma_vec, *_ = jnp.linalg.lstsq(A, b)

    # Generalized method of moments iterations
    for _ in range(1, num_steps):
        error = b - A @ sigma_vec

        N = len(scanning_data)
        moment_cov = (error[:, None] @ error[None, :]) / N
        W = jnp.linalg.inv(
            moment_cov + w_inv_regularization * jnp.eye(moment_cov.shape[0])
        )
        W += w_regularization * jnp.eye(moment_cov.shape[0])

        # Solve the weighted least-squares problem
        Aw = A.T @ W @ A
        bw = A.T @ W @ b
        eigvals, eigvecs = jnp.linalg.eigh(Aw)
        cutoff = 1e-2
        inv_eig = jnp.where(eigvals > cutoff, 1.0 / eigvals, 0.0)
        sigma_vec = eigvecs @ (inv_eig * (eigvecs.T @ bw))
    sigma = sigma_vec.reshape(2 * num_modes, 2 * num_modes)
    return (sigma + sigma.T) / 2
