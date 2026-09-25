"""Standard Hough transform for degree-4 polynomial curves y = f(x).

Curves are written in a normalized x so every coefficient is in pixels:

    t = (x - cx) / cx                    t runs from -1 (left edge) to 1 (right edge)
    y = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4

a0 is the curve's height at the center column, and a_k is how many pixels the
t^k term adds at the image edge.

Voting: every ink pixel (x, y) votes for every quartic passing through it. For
each (a1, a2, a3, a4) on a grid, there is exactly one a0 that makes the curve
hit the pixel:

    a0 = y - (a1 t + a2 t^2 + a3 t^3 + a4 t^4)

so the pixel adds one vote to the accumulator cell (a0, a1, a2, a3, a4).
Cells that collect many votes are quartics lying on many ink pixels. The
curves are the local maxima of the accumulator above a vote threshold.
"""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from numba import njit
from scipy.ndimage import maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector.edge_detector import paper_clean_fast  # noqa: E402


def binarize(gray, max_side=500):
    """Downscale, then clean the paper with the detector's adaptive cleaner.
    Returns (binary image with ink = 0, scale factor back to the input)."""
    scale = max(gray.shape) / max_side
    small = cv.resize(gray, None, fx=1 / scale, fy=1 / scale, interpolation=cv.INTER_AREA)
    return paper_clean_fast(small, inner=1, outer=8, cutoff=8), scale


@njit(cache=True)
def _vote(t, y, A, a0_step, acc):
    """acc[j, b] += 1 for every pixel p and grid row j = (a1, a2, a3, a4),
    where b is the a0 cell that puts the curve through the pixel."""
    n0 = acc.shape[1]
    for p in range(len(t)):
        t1 = t[p]
        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t2 * t2
        for j in range(A.shape[0]):
            a0 = y[p] - (A[j, 0] * t1 + A[j, 1] * t2 + A[j, 2] * t3 + A[j, 3] * t4)
            b = int(np.floor(a0 / a0_step + 0.5))
            if 0 <= b < n0:
                acc[j, b] += 1


class QuarticHough:
    def __init__(self, a0_step=2.0, a1=(-60, 60, 21), a2=(-30, 30, 11),
                 a3=(-30, 30, 11), a4=(-30, 30, 11), min_votes=1.0):
        """a0_step: size of an a0 cell in pixels (a0 covers the image height).
        a1..a4: (low, high, count) grid of values tried for each coefficient.
        min_votes: a peak needs at least this fraction of the image width in votes."""
        self.a0_step = a0_step
        self.grids = [np.linspace(*g) for g in (a1, a2, a3, a4)]
        self.min_votes = min_votes

    def fit(self, binary):
        """Returns curves as a list of (coeffs [a0..a4], votes), most votes first.
        The accumulator is kept in self.acc, indexed [a1, a2, a3, a4, a0]."""
        H, W = binary.shape
        self.cx = W / 2
        ys, xs = np.nonzero(binary == 0)
        t = (xs - self.cx) / self.cx
        self.a0_values = np.arange(0, H, self.a0_step)

        # Every (a1, a2, a3, a4) combination as one row
        A = np.stack(np.meshgrid(*self.grids, indexing="ij"), -1).reshape(-1, 4)
        acc = np.zeros((len(A), len(self.a0_values)), np.int32)
        _vote(t, ys.astype(np.float64), A, self.a0_step, acc)
        self.acc = acc.reshape(*(len(g) for g in self.grids), -1)

        # Peaks: cells that are the maximum of their 3x3x3x3x3 neighborhood
        peak = self.acc == maximum_filter(self.acc, size=3, mode="constant")
        peak &= self.acc >= self.min_votes * W
        cells = np.argwhere(peak)
        cells = cells[np.argsort(-self.acc[tuple(cells.T)])]

        curves = []
        for i1, i2, i3, i4, i0 in cells:
            coeffs = np.array([self.a0_values[i0], self.grids[0][i1], self.grids[1][i2],
                               self.grids[2][i3], self.grids[3][i4]])
            curves.append((coeffs, int(self.acc[i1, i2, i3, i4, i0])))
        return curves

    def evaluate(self, coeffs, x):
        """y of a curve at pixel columns x."""
        t = (np.asarray(x) - self.cx) / self.cx
        return sum(a * t ** k for k, a in enumerate(coeffs))
