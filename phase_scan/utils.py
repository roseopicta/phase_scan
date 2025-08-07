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

import numpy as np


def triangle(t: np.ndarray, frequency: float) -> np.ndarray:
    """
    Generates a triangle wave with a given frequency over time values t.

    Args:
        t (np.ndarray): Array of time values.
        frequency (float): Frequency of the triangle wave in Hz.

    Returns:
        np.ndarray: Array of triangle wave values corresponding to t.
    """
    ramp, _ = np.modf(t * frequency)
    triangle = np.minimum(2 * ramp, 2 - 2 * ramp)
    return triangle
