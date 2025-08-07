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

import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Optional, Callable, Tuple


# All computations follow the conventions from https://arxiv.org/abs/2102.05748
# Covariance matrices are represented in xpxp convention.


def squeezed_vacuum(s: float) -> np.ndarray:
    """
    Compute the covariance matrix of a squeezed vacuum state.

    Args:
        s: Squeezing parameter.

    Returns:
        2x2 covariance matrix.
    """
    return np.diag([s, 1 / s])


def vacuum() -> np.ndarray:
    """
    Return the covariance matrix of a vacuum state.

    Returns:
        2x2 identity matrix as a NumPy array.
    """
    return np.eye(2, 2)


def T_to_angle(t: float) -> float:
    """
    Convert transmittance to beam-splitter angle.

    Args:
        t: Transmittance (0 ≤ t ≤ 1).

    Returns:
        Corresponding angle in radians.
    """
    return np.acos(np.sqrt(t))


def rotation_symplectic(theta: float) -> np.ndarray:
    """
    Generate the 2x2 symplectic rotation matrix.

    Args:
        theta: Rotation angle in radians.

    Returns:
        2x2 symplectic matrix.
    """
    return np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )


def bs_symplectic(t: float) -> np.ndarray:
    """
    Generate the 4x4 symplectic matrix of a beam-splitter.

    Args:
        t: Transmittance (0 ≤ t ≤ 1).

    Returns:
        4x4 symplectic matrix.
    """
    return np.kron(rotation_symplectic(T_to_angle(t)), np.eye(2, 2))


def plot_covariance_matrix(
    sigma: np.ndarray,
    title: Optional[str] = None,
    show_var_names: bool = False,
    sequential: bool = False,
    axs: Optional[plt.Axes] = None,
):
    """
    Plot a covariance matrix as a heatmap.

    Args:
        sigma: Covariance matrix to plot.
        title: Optional title for the plot.
        show_var_names: Whether to label x and y axes with variable names.
        sequential: Whether to label modes sequentially (e.g., x1, p1, x2, p2, ...).
        axs: Optional matplotlib Axes object to draw on. Creates a new figure if None.
    """
    if axs is None:
        plt.figure(figsize=(4, 4))
        axs = plt.gca()

    n, _ = sigma.shape
    max_v = np.max(sigma)
    pcm = axs.imshow(sigma, cmap="RdBu", vmin=-max_v, vmax=max_v)
    plt.colorbar(pcm, ax=axs, fraction=0.046, pad=0.04)

    if title:
        axs.set_title(title)

    if show_var_names:
        start = n // 4
        if sequential:
            indice = lambda x: f"k{(x - start):+}" if (x - start) else "k"
        else:
            indice = lambda x: str(x)

        var_names = [
            [f"$\\hat{{x}}_{{{indice(i)}}}$", f"$\\hat{{p}}_{{{indice(i)}}}$"]
            for i in range(n // 2)
        ]
        var_names = sum(var_names, [])
        axs.set_xticks(range(n), var_names)
        axs.set_yticks(range(n), var_names)
    else:
        axs.set_xticks([])
        axs.set_yticks([])


def quadrature_covariance(sigma: np.ndarray, theta: float) -> np.ndarray:
    """
    Compute the covariance of a quadrature measurement at angle theta.

    Args:
        sigma: (2n, 2n) covariance matrix in xpxp convention.
        theta: Homodyne measurement angle (in radians).

    Returns:
        Covariance matrix of the quadrature samples for angle theta, size (n,n)
    """

    n, m = sigma.shape
    if n % 2 != 0:
        raise ValueError(
            "The size of the covariance matrix must be a multiple of 2"
        )

    num_modes = n // 2
    if m != 2 * num_modes:
        raise ValueError("Invalid covariance matrix size")

    # Handle a single angle for several modes
    if type(theta) is float:
        theta = np.array([theta] * num_modes)
    elif theta.shape == (1,):
        theta = np.array([theta[0]] * num_modes)

    if len(theta) != num_modes:
        raise ValueError(
            "The size of the angle vector must match the number of modes"
        )

    q = np.zeros((num_modes, num_modes * 2))
    for i in range(num_modes):
        q[i, 2 * i] = np.cos(theta[i])
        q[i, 2 * i + 1] = np.sin(theta[i])
    sigma_rot = q @ sigma @ q.T
    return sigma_rot


def generate_samples_parallel(
    sigma: np.ndarray, num_samples: int
) -> np.ndarray:
    num_modes = sigma.shape[0]
    return np.random.multivariate_normal(
        np.zeros(num_modes), sigma, size=num_samples
    )


def generate_samples_sequential(
    sigma: np.ndarray, num_samples: int
) -> np.ndarray:
    """
    Generate samples given the covariance matrix between consecutive samples

    Args:
        sigma: Covariance matrix defining correlations between samples.
        num_samples: Number of samples to generate.

    Returns:
        Array of shape (num_samples, num_modes) containing generated samples.
    """
    K = sigma.shape[0]
    if K % 8 != 0:
        raise ValueError(
            "The size of the covariance matrix must be a multiple of 8"
        )

    step = K // 8
    half = K // 2

    sigma_11 = sigma[:half, :half]
    sigma_12 = sigma[:half, half:]
    sigma_21 = sigma[half:, :half]
    sigma_22 = sigma[half:, half:]
    inv_sigma_11 = np.linalg.inv(sigma_11)

    # Generate the first K / 2 samples directly
    l = np.linalg.cholesky(sigma)
    block = l @ np.random.randn(K)

    x = list(block[:half])

    while len(x) < num_samples:
        prev_half = x[-half:]

        # Parameters of conditional distribution
        mu_cond = sigma_21 @ inv_sigma_11 @ prev_half
        cov_cond = sigma_22 - sigma_21 @ inv_sigma_11 @ sigma_12

        # Sample from conditional distribution
        l_cond = np.linalg.cholesky(cov_cond)
        next_half = (l_cond @ np.random.randn(half)) + mu_cond

        x.extend(next_half[:step])

    return np.array(x[:num_samples])


def generate_scanned_samples(
    sigma: np.ndarray,
    angles: np.ndarray,
    sampling_fn: Callable[[np.ndarray, int], np.ndarray],
    samples_per_angle: int = 1000,
) -> Sequence[Tuple[float, np.ndarray]]:
    """
    Generate quadrature samples for multiple measurement angles.

    Args:
        sigma: Covariance matrix in xpxp convention.
        angles: Array of measurement angles in radians.
        sampling_fn: Function that generates samples from a covariance matrix.
        samples_per_angle: Number of samples per angle (default is 1000).

    Returns:
        List of (angle, samples) tuples.
    """
    if len(angles.shape) == 1:
        angles = angles.reshape(-1, 1)
    num_angles, num_modes = angles.shape
    return [
        (
            angles[i, :],
            sampling_fn(
                quadrature_covariance(sigma, angles[i, :]), samples_per_angle
            ),
        )
        for i in range(num_angles)
    ]


def collect_adjacent_samples(
    scanning_data: Sequence[Tuple[float, np.ndarray]], num_adjacent_modes: int
) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    """
    Create overlapping windows of adjacent quadrature samples.

    Args:
        scanning_data: List of (angle, samples) tuples.
        num_adjacent_modes: Number of adjacent samples per window.

    Returns:
        List of (angles, sample matrix) tuples. Each sample matrix has shape
        (samples - num_adjacent_modes + 1, num_adjacent_modes).
    """
    new_data = []
    for angles, samples in scanning_data:
        angles = np.array([angles[0]] * num_adjacent_modes)
        samples = np.lib.stride_tricks.sliding_window_view(
            samples, window_shape=num_adjacent_modes
        )
        new_data.append((angles, samples))
    return new_data
