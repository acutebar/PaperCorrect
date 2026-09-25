import math

import cv2 as cv
import numpy as np
from numba import njit

from .edge_detector import get_disc_kernel


def _offsets(mask, center):
    """Pixels set in a scratch canvas, as (dy, dx) offsets from its center."""
    dy, dx = np.nonzero(mask)
    return dy - center, dx - center


def _pack(offset_list):
    """Ragged list of (dy, dx) -> padded (n, K) arrays plus per-row counts."""
    K = max(len(dy) for dy, _ in offset_list)
    n = len(offset_list)
    DY = np.zeros((n, K), np.int64)
    DX = np.zeros((n, K), np.int64)
    cnt = np.zeros(n, np.int64)
    for i, (dy, dx) in enumerate(offset_list):
        DY[i, :len(dy)] = dy
        DX[i, :len(dx)] = dx
        cnt[i] = len(dy)
    return DY, DX, cnt


class LineDetector:
    def __init__(self, width=2, height=5, step=5, sweep=30):
        self.kernel = get_disc_kernel(height)
        self.angles = np.arange(0, 360, step).astype(np.int64)
        self.nbrs = max(1, sweep // step)   # neighbors per side in the valley test

        # Sampling templates (curr -> curr + tip) for every angle
        R = int(np.ceil(height)) + 2
        templates, TR, TC = [], [], []
        for angle in self.angles:
            t = math.radians(angle)
            tip_r = height * math.cos(t)
            tip_c = -height * math.sin(t)
            mask = np.zeros((2 * R + 1, 2 * R + 1), np.uint8)
            cv.line(mask, (R, R),
                    (int(round(R + tip_c)), int(round(R + tip_r))),
                    255, thickness=width)
            templates.append(_offsets(mask, R))
            TR.append(int(round(tip_r)))
            TC.append(int(round(tip_c)))
        self.DY, self.DX, self.cnt = _pack(templates)
        self.TR = np.array(TR, np.int64)
        self.TC = np.array(TC, np.int64)

        # Footprints of the visited marks (thick line curr -> curr + tip, and
        # the start circle), rasterized once by OpenCV and stamped at runtime.
        # The canvas is large enough that nothing is clipped, and the image is
        # padded by the same amount so stamps never need bounds checks.
        self.pad = Rf = int(np.ceil(height)) + width + 4
        feet = []
        for tr, tc in zip(TR, TC):
            mask = np.zeros((2 * Rf + 1, 2 * Rf + 1), np.uint8)
            cv.line(mask, (Rf, Rf), (Rf + tc, Rf + tr), 1, thickness=width + 2)
            feet.append(_offsets(mask, Rf))
        self.FY, self.FX, self.fcnt = _pack(feet)

        mask = np.zeros((2 * Rf + 1, 2 * Rf + 1), np.uint8)
        cv.circle(mask, (Rf, Rf), width + 1, 1, -1)
        self.CY, self.CX = _offsets(mask, Rf)

    def findall_lines(self, img):
        f = img.astype(np.float32)
        mean = cv.filter2D(f, -1, self.kernel, borderType=cv.BORDER_REFLECT)
        mean_sq = cv.filter2D(f * f, -1, self.kernel, borderType=cv.BORDER_REFLECT)
        thresh = mean - 15

        # Padding is bright, below any threshold, and pre-marked visited
        p = self.pad
        pimg = cv.copyMakeBorder(img, p, p, p, p, cv.BORDER_CONSTANT, value=255)
        pthresh = cv.copyMakeBorder(thresh, p, p, p, p, cv.BORDER_CONSTANT, value=0)
        visited = np.ones(pimg.shape, np.uint8)
        visited[p:-p, p:-p] = 0
        dark = np.argwhere(img == 0) + p

        pts, offs = _find_all(
            pimg, pthresh, visited, dark, self.angles, self.nbrs,
            self.DY, self.DX, self.cnt, self.TR, self.TC,
            self.FY, self.FX, self.fcnt, self.CY, self.CX,
        )
        pts -= p
        return [list(map(tuple, pts[a:b].tolist())) for a, b in zip(offs[:-1], offs[1:])]


# --------------------------------------------------------------------------
# Compiled core
# --------------------------------------------------------------------------

@njit(cache=True)
def _valid_paths(img, thresh, visited, r, c, nbrs, DY, DX, cnt, TR, TC, means):
    """Sweeps every angle around (r, c). An angle is valid if its template mean
    is below threshold, its tip is unvisited, and it is a local minimum over
    nbrs neighbors on each side. Returns (number valid, darkest valid angle)."""
    n = DY.shape[0]
    for a in range(n):
        s = 0.0
        for k in range(cnt[a]):
            s += img[r + DY[a, k], c + DX[a, k]]
        means[a] = s / cnt[a]

    th = thresh[r, c]
    k = best = 0
    for a in range(n):
        v = means[a]
        if v >= th or visited[r + TR[a], c + TC[a]]:
            continue
        valley = True
        for j in range(1, nbrs + 1):
            if v >= means[(a - j) % n] or v > means[(a + j) % n]:
                valley = False
                break
        if valley:
            if k == 0 or v < means[best]:
                best = a
            k += 1
    return k, best


@njit(cache=True)
def _stamp(visited, r, c, FY, FX):
    for k in range(len(FY)):
        visited[r + FY[k], c + FX[k]] = 1


@njit(cache=True)
def _push(pts, n, r, c):
    if n == pts.shape[0]:
        bigger = np.empty((2 * n, 2), np.int64)
        bigger[:n] = pts
        pts = bigger
    pts[n, 0] = r
    pts[n, 1] = c
    return pts


@njit(cache=True)
def _find_all(img, thresh, visited, dark, angles, nbrs, DY, DX, cnt, TR, TC,
              FY, FX, fcnt, CY, CX):
    means = np.empty(DY.shape[0])
    pts = np.empty((1024, 2), np.int64)
    offs = np.zeros(dark.shape[0] + 1, np.int64)
    npts = nlines = 0

    for i in range(dark.shape[0]):
        r, c = dark[i, 0], dark[i, 1]
        if visited[r, c]:
            continue
        if _valid_paths(img, thresh, visited, r, c, nbrs,
                        DY, DX, cnt, TR, TC, means)[0] != 1:
            continue

        _stamp(visited, r, c, CY, CX)
        start = npts
        pts = _push(pts, npts, r, c)
        npts += 1

        prev = base = 0
        while True:
            k, best = _valid_paths(img, thresh, visited, r, c, nbrs,
                                   DY, DX, cnt, TR, TC, means)
            if k == 0:
                break
            ang = angles[best]
            if npts - start == 1:
                base = prev = ang
            elif (abs((ang - prev + 180) % 360 - 180) > 45 or
                  abs((ang - base + 180) % 360 - 180) > 60):
                break
            prev = ang

            m = fcnt[best]
            _stamp(visited, r, c, FY[best, :m], FX[best, :m])
            r += TR[best]
            c += TC[best]
            pts = _push(pts, npts, r, c)
            npts += 1

        if npts - start > 1:
            nlines += 1
            offs[nlines] = npts
        else:
            npts = start        # drop 1-point line; its start circle stays marked

    return pts[:npts], offs[:nlines + 1]
