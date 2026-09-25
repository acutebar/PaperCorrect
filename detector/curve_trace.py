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
        self.width = width
        self.height = height
        self.angles = np.arange(0, 360, step).astype(np.int64)
        self.pad = R = int(np.ceil(height)) + 2

        templates, tips = [], []
        for angle in self.angles:
            t = math.radians(angle)
            tip_r = height * math.cos(t)
            tip_c = -height * math.sin(t)
            mask = np.zeros((2 * R + 1, 2 * R + 1), np.uint8)
            cv.line(mask, (R, R),
                    (int(round(R + tip_c)), int(round(R + tip_r))),
                    255, thickness=width)
            templates.append(_offsets(mask, R))
            tips.append((int(round(tip_r)), int(round(tip_c))))

        self.DY, self.DX, self.cnt = _pack(templates)
        self.TR = np.array([t[0] for t in tips], np.int64)
        self.TC = np.array([t[1] for t in tips], np.int64)

        # Circular neighbor indices for the valley test
        n = len(self.angles)
        m = max(1, sweep // step)
        i = np.arange(n)[:, None]
        j = np.arange(1, m + 1)[None, :]
        self.PREV = (i - j) % n
        self.NEXT = (i + j) % n

        # Footprints of mark_visited (cv.line, curr -> curr + tip) and the
        # start circle, rasterized once by OpenCV and stamped at runtime.
        # Canvas is large enough that nothing is clipped.
        Rf = int(np.ceil(height)) + width + 4
        feet = []
        for tr, tc in tips:
            mask = np.zeros((2 * Rf + 1, 2 * Rf + 1), np.uint8)
            cv.line(mask, (Rf, Rf), (Rf + tc, Rf + tr), 1, thickness=width + 2)
            feet.append(_offsets(mask, Rf))
        self.FY, self.FX, self.fcnt = _pack(feet)

        mask = np.zeros((2 * Rf + 1, 2 * Rf + 1), np.uint8)
        cv.circle(mask, (Rf, Rf), width + 1, 1, -1)
        self.CY, self.CX = _offsets(mask, Rf)

    def findall_lines(self, img):
        kernel = get_disc_kernel(self.height)
        f = img.astype(np.float32)
        mean = cv.filter2D(f, -1, kernel, borderType=cv.BORDER_REFLECT)
        mean_sq = cv.filter2D(f * f, -1, kernel, borderType=cv.BORDER_REFLECT)
        thresh = mean - 0.5 * np.sqrt(np.maximum(0, mean_sq - mean * mean))

        p = self.pad
        pimg = cv.copyMakeBorder(img, p, p, p, p, cv.BORDER_CONSTANT, value=255)
        pthresh = cv.copyMakeBorder(thresh, p, p, p, p, cv.BORDER_CONSTANT, value=0)
        visited = np.ones(pimg.shape, np.uint8)
        visited[p:-p, p:-p] = 0
        dark = np.argwhere(img == 0).astype(np.int64) + p

        pts, offs = _find_all(
            pimg, pthresh, visited, dark,
            self.DY, self.DX, self.cnt, self.TR, self.TC, self.PREV, self.NEXT,
            self.FY, self.FX, self.fcnt, self.CY, self.CX, self.angles,
        )
        pts -= p
        return [list(map(tuple, pts[a:b].tolist())) for a, b in zip(offs[:-1], offs[1:])]


# --------------------------------------------------------------------------
# Compiled core
# --------------------------------------------------------------------------

@njit(cache=True)
def _valid_paths(img, r, c, visited, thresh, DY, DX, cnt, TR, TC, PREV, NEXT,
                 means, out):
    """Fills means[a] for every angle and out[:k] with valid angle indices
    (ascending). Returns k."""
    n = DY.shape[0]
    for a in range(n):
        s = 0.0
        for k in range(cnt[a]):
            s += img[r + DY[a, k], c + DX[a, k]]
        means[a] = s / cnt[a]

    th = thresh[r, c]
    k = 0
    for a in range(n):
        v = means[a]
        if not v < th or visited[r + TR[a], c + TC[a]]:
            continue
        valley = True
        for j in range(PREV.shape[1]):
            if v >= means[PREV[a, j]] or v > means[NEXT[a, j]]:
                valley = False
                break
        if valley:
            out[k] = a
            k += 1
    return k


@njit(cache=True)
def _stamp(visited, r, c, FY, FX, n):
    H, W = visited.shape
    for k in range(n):
        y = r + FY[k]
        x = c + FX[k]
        if 0 <= y < H and 0 <= x < W:
            visited[y, x] = 1


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
def _find_all(img, thresh, visited, dark, DY, DX, cnt, TR, TC, PREV, NEXT,
              FY, FX, fcnt, CY, CX, angles):
    n = DY.shape[0]
    means = np.empty(n)
    out = np.empty(n, np.int64)
    pts = np.empty((1024, 2), np.int64)
    offs = np.empty(dark.shape[0] + 1, np.int64)
    offs[0] = 0
    npts = 0
    nlines = 0

    for i in range(dark.shape[0]):
        r = dark[i, 0]
        c = dark[i, 1]
        if visited[r, c]:
            continue
        if _valid_paths(img, r, c, visited, thresh, DY, DX, cnt, TR, TC,
                        PREV, NEXT, means, out) != 1:
            continue

        _stamp(visited, r, c, CY, CX, CY.shape[0])
        start = npts
        pts = _push(pts, npts, r, c)
        npts += 1

        first = True
        prev = 0
        base = 0
        while True:
            k = _valid_paths(img, r, c, visited, thresh, DY, DX, cnt, TR, TC,
                             PREV, NEXT, means, out)
            if k == 0:
                break
            best = out[0]                      # first minimum, as min() does
            for q in range(1, k):
                if means[out[q]] < means[best]:
                    best = out[q]
            ang = angles[best]
            if first:
                base = ang
                first = False
            elif (abs((ang - prev + 180) % 360 - 180) > 45 or
                  abs((ang - base + 180) % 360 - 180) > 60):
                break
            prev = ang

            nr = r + TR[best]
            nc = c + TC[best]
            pts = _push(pts, npts, nr, nc)
            npts += 1
            _stamp(visited, r, c, FY[best], FX[best], fcnt[best])
            r = nr
            c = nc

        if npts - start > 1:
            nlines += 1
            offs[nlines] = npts
        else:
            npts = start        # drop 1-point line; its start circle stays marked

    return pts[:npts].copy(), offs[:nlines + 1].copy()
