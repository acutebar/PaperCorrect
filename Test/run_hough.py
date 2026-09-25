"""Front end for the standard degree-4 Hough transform.

Usage (from anywhere):
    python3 Test/run_hough.py                 # cycle through the sample photos
    python3 Test/run_hough.py my_photo.jpeg   # or pass your own images

Keys: right / left = next / previous image.  Click a curve to see its coefficients.
"""
import sys
import time
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from hough import QuarticHough, binarize

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = ["ruled_paper.jpeg", "upcurve.jpeg", "grid1.jpeg", "kindanormalpaper.jpeg",
                  "curvedsurface1.jpeg", "crumpled.jpeg", "curve.jpeg", "straight_line.jpeg"]


def extract(path):
    gray = cv.cvtColor(cv.imread(str(path)), cv.COLOR_BGR2GRAY)
    binary, _ = binarize(gray)
    hough = QuarticHough()
    t = time.perf_counter()
    curves = hough.fit(binary)
    return dict(binary=binary, hough=hough, curves=curves, seconds=time.perf_counter() - t)


class Viewer:
    def __init__(self, paths):
        self.paths, self.i, self.cache = paths, 0, {}
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 8))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.show()
        plt.show()

    def show(self):
        path = self.paths[self.i]
        if path not in self.cache:
            print(f"Running Hough on {path.name} ...")
            self.cache[path] = extract(path)
        r = self.cache[path]
        binary, hough, curves = r["binary"], r["hough"], r["curves"]
        ax_img, ax_acc = self.axes
        for ax in self.axes:
            ax.clear()

        # Left: the binary image with every detected curve
        ax_img.imshow(binary, cmap="gray", alpha=0.4)
        x = np.arange(binary.shape[1])
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        for k, (coeffs, votes) in enumerate(curves):
            line, = ax_img.plot(x, hough.evaluate(coeffs, x), color=colors[k % 20],
                                lw=1.2, picker=4)
            line.curve = (coeffs, votes)
        ax_img.set_xlim(0, binary.shape[1])
        ax_img.set_ylim(binary.shape[0], 0)
        ax_img.set_title(f"{len(curves)} curves (click one to inspect)")

        # Right: the accumulator, showing for each (a1, a0) the best vote count
        # over all (a2, a3, a4). Each curve is a bright spot.
        a1 = hough.grids[0]
        a0 = hough.a0_values
        ax_acc.imshow(hough.acc.max(axis=(1, 2, 3)).T, aspect="auto", cmap="inferno",
                      extent=[a1[0], a1[-1], a0[-1], a0[0]])
        ax_acc.scatter([c[1] for c, _ in curves], [c[0] for c, _ in curves],
                       s=12, facecolors="none", edgecolors="cyan", lw=0.8)
        ax_acc.set_xlabel("a1 (px of tilt at the edge)")
        ax_acc.set_ylabel("a0 (height at the center column, px)")
        ax_acc.set_title("Accumulator, max over a2, a3, a4 (circles = peaks)")

        self.fig.suptitle(f"[{self.i + 1}/{len(self.paths)}] {path.name}: Hough took "
                          f"{r['seconds']:.1f}s    (left/right: change image)")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        step = {"right": 1, "left": -1}.get(event.key)
        if step:
            self.i = (self.i + step) % len(self.paths)
            self.show()

    def on_pick(self, event):
        curve = getattr(event.artist, "curve", None)
        if curve is None:
            return
        (a0, a1, a2, a3, a4), votes = curve
        self.axes[0].set_title(f"y = {a0:.0f} {a1:+.0f}t {a2:+.0f}t² {a3:+.0f}t³ {a4:+.0f}t⁴"
                               f"   ({votes} votes)")
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    args = sys.argv[1:]
    paths = [Path(a).resolve() for a in args] if args else [ROOT / f for f in DEFAULT_IMAGES]
    Viewer([p for p in paths if p.exists()])
