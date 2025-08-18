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
import optax
from tqdm.auto import tqdm
from typing import List, Sequence, Tuple


def quadrature_projection_matrix(angles: jnp.ndarray) -> jnp.ndarray:
    """
    Constructs a projection matrix for quadrature measurements based on given angles.

    Args:
        angles (jnp.ndarray): Array of shape (num_modes,) containing measurement angles
                              for each mode in radians.

    Returns:
        jnp.ndarray: A matrix of shape (num_modes, 2 * num_modes) used to project the
                     full covariance matrix onto the measured quadrature directions.
    """
    num_modes = angles.shape[0]

    def make_row(i):
        row = jnp.zeros((2 * num_modes,))
        row = row.at[2 * i].set(jnp.cos(angles[i]))
        row = row.at[2 * i + 1].set(jnp.sin(angles[i]))
        return row

    return jnp.stack([make_row(i) for i in range(num_modes)])


def log_likelihood(
    sigma: jnp.ndarray, angles: jnp.ndarray, q: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the log-likelihood of observing the quadrature measurements `q`
    under a Gaussian model with covariance `sigma` and measurement angles `angles`.

    Args:
        sigma (jnp.ndarray): Covariance matrix of shape (2*num_modes, 2*num_modes).
        angles (jnp.ndarray): Array of shape (2*num_modes,) representing measurement angles.
        q (jnp.ndarray): Observed quadratures of shape (n_samples, num_modes).

    Returns:
        jnp.ndarray: Scalar log-likelihood value.
    """
    num_modes = len(angles) // 2
    p = quadrature_projection_matrix(angles)
    quadrature_cov = p @ sigma @ p.T

    precision = jnp.linalg.inv(quadrature_cov)
    _, logdet = jnp.linalg.slogdet(quadrature_cov)

    q_prec = q @ precision
    mahalanobis_terms = jnp.sum(q_prec * q, axis=1)
    n = mahalanobis_terms.shape[0]

    return -0.5 * (
        jnp.sum(mahalanobis_terms)
        + n * logdet
        + n * num_modes * jnp.log(2 * jnp.pi)
    )


def total_log_likelihood_batched(
    sigma: jnp.ndarray, data: Sequence[Tuple[jnp.ndarray, jnp.ndarray]]
) -> jnp.ndarray:
    """
    Computes the total log-likelihood over a batch of quadrature data and angles.

    Args:
        sigma (jnp.ndarray): Covariance matrix of shape (2*num_modes, 2*num_modes).
        data (Sequence[Tuple[jnp.ndarray, jnp.ndarray]]): A sequence of (angles, quadratures) tuples:
            - angles: jnp.ndarray of shape (2*num_modes,)
            - quadratures: jnp.ndarray of shape (n_samples, num_modes)

    Returns:
        jnp.ndarray: Scalar total log-likelihood summed over the batch.
    """
    angles_array = jnp.stack([item[0] for item in data])
    quadratures_array = jnp.stack([item[1] for item in data])
    batched_log_likelihood = jax.vmap(log_likelihood, in_axes=(None, 0, 0))
    return jnp.sum(
        batched_log_likelihood(sigma, angles_array, quadratures_array)
    )


def full_covariance_matrix(
    l_sigma_values: jnp.ndarray, num_modes: int
) -> jnp.ndarray:
    """
    Reconstructs the full symmetric covariance matrix from its Cholesky factor.

    Args:
        l_sigma_values (jnp.ndarray): Flattened cholesky factor.
        num_modes (int): Number of modes. The full matrix is of shape (2*num_modes, 2*num_modes).

    Returns:
        jnp.ndarray: Symmetric covariance matrix of shape (2*num_modes, 2*num_modes).
    """
    L = jnp.zeros((2 * num_modes, 2 * num_modes))
    indices = jnp.tril_indices(2 * num_modes)
    L = L.at[indices].set(l_sigma_values)
    return L @ L.T


def ml_covariance_estimation(
    scanning_data: List[Tuple[np.ndarray, np.ndarray]],
    lr: float = 0.05,
    max_iterations: int = 5_000,
) -> np.ndarray:
    """
    Performs maximum likelihood estimation (MLE) of a Gaussian state's covariance matrix
    using homodyne scanning data.

    Args:
        scanning_data: a list of tuples (angles, quadrature_data)
            - angles: np.ndarray of shape (num_modes), specifying the measurement angles
            - quadratures: np.ndarray of shape (batch_size, num_modes), the quadrature data
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.05.
        max_iterations (int, optional): Maximum number of optimization steps. Defaults to 5000.

    Returns:
        np.ndarray: Estimated covariance matrix of shape (2 * num_modes, 2 * num_modes).
    """
    num_modes = len(scanning_data[0][0])

    @jax.jit
    def loss_fn(sigma_l_values):
        return -total_log_likelihood_batched(
            full_covariance_matrix(sigma_l_values, num_modes), scanning_data
        )

    @jax.jit
    def update_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jnp.conj(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    sigma_l_0 = jnp.eye(num_modes * 2)
    params = sigma_l_0[jnp.triu_indices(num_modes * 2)]
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)
    progress = tqdm(range(max_iterations), desc="Optimizing...", leave=True)
    for it in progress:
        params, opt_state, loss_value = update_step(params, opt_state)
        if it % 10 == 0:
            progress.set_postfix(log_likelihood=-loss_value)
    return full_covariance_matrix(params, num_modes)


def ml_covariance_estimation_direct_ls(
    scanning_data: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    Performs maximum likelihood estimation (MLE) of a Gaussian state's covariance matrix
    using homodyne scanning data.

    This function uses least square estimation and is suitable for smaller batches
    of data.

    Args:
        scanning_data: a list of tuples (angles, quadrature_data)
            - angles: np.ndarray of shape (num_modes), specifying the measurement angles
            - quadratures: np.ndarray of shape (batch_size, num_modes), the quadrature data

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

    A = A_blocks.reshape(-1, A_blocks.shape[-1])
    b = b_vecs.flatten()
    sigma_vec, *_ = jnp.linalg.lstsq(A, b)
    return sigma_vec.reshape(2 * num_modes, 2 * num_modes)
