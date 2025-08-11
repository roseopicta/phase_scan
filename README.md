# Covariance matrix reconstruction from quadrature samples

This repository contains a simple to use function to reconstruct the covariance matrix of a gaussian state from scanned quadrature measurements.

Notebooks illustrate its use on several examples of states.

## Setup

```shell
git clone https://github.com/roseopicta/phase_scan.git
cd phase_scan
pip install -e .
```

## Simple example

```python

from phase_scan import gaussian_utils
from phase_scan.ml_estimation import ml_covariance_estimation
import numpy as np


# Compute the covariance matrix of a squeezed vacuum state
squeezing_dB = 6
squeezing_angle = np.pi / 4
squeezing_s = 10 ** (-squeezing_dB / 10)
sigma = gaussian_utils.squeezed_vacuum(squeezing_s)
R = gaussian_utils.rotation_symplectic(squeezing_angle)
sigma = R@sigma@R.T

# Generate quadrature samples with an angle between 0 and pi
angles = np.linspace(0, 1, 100) * np.pi
samples = gaussian_utils.generate_scanned_samples(
    sigma,
    angles,
    gaussian_utils.generate_samples_parallel,
    1000)

# Reconstruct the covariance matrix
sigma_hat = ml_covariance_estimation(samples, lr=0.05, max_iterations=1000)
```

## Dependencies

The code requires the following libraries: numpy, scipy, matplotlib, tqdm and optax.
