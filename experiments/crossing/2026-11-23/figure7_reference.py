"""Extract VisFly v4 Figure 7's displayed swarm curves from the original PNG."""
import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SHA256 = "2c89a20c13384bec5e2367501782b7d3c368749b1b372360f12fecd2277b8e22"
COLOR = (255, 201, 60)
# Pixel centers of axis spines or labelled gridlines of the original 2100 x 825 image.
# (x at zero, x at 10 million, y at zero, y at one)
AXES = {"success_rate": (1269.5, 2042.5, 519.5, 26.5),
        "reward_on_paper_axis": (191.5, 964.5, 413.5, 26.5)}
WINDOWS = [(900000, 1100000), (9000000, 10000000)]


def read_image(source):
    if tarfile.is_tarfile(source):
        with tarfile.open(source) as archive:
            return archive.extractfile("fig/train_curve.png").read()
    return source.read_bytes()


def extract(image, axes):
    x0, x1, y0, y1 = axes
    mask = np.all(image[27:519, :, :3] == COLOR, axis=2)
    points = []
    for x in range(int(np.ceil(x0)), int(x1) + 1):
        ys = np.flatnonzero(mask[:, x]) + 27
        if ys.size:
            y = .5 * (ys[0] + ys[-1])
            step = 10000000 * (x - x0) / (x1 - x0)
            points.append((step, (y0 - y) / (y0 - y1), x, y))
    return np.array(points)


def window_mean(x, y, lo, hi):
    if np.any(np.diff(x) <= 0):
        raise ValueError("abscissae must be strictly increasing")
    grid = np.r_[lo, x[(x > lo) & (x < hi)], hi]
    samples = np.interp(grid, x, y)
    return float(np.trapz(samples, grid) / (hi - lo))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="arXiv v4 source tarball or fig/train_curve.png")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    png = read_image(args.input)
    digest = hashlib.sha256(png).hexdigest()
    if digest != IMAGE_SHA256:
        raise ValueError(f"Unexpected image hash {digest}")
    image = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    if image.shape != (825, 2100, 3):
        raise ValueError(f"Unexpected image dimensions {image.shape}")
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": "https://arxiv.org/src/2407.14783v4",
        "figure_7_png_sha256": digest, "rgb": COLOR,
        "axes_x0_x10m_y0_y1": AXES,
        "method": "Center of solid yellow row range in each image column, linear interpolation between columns, constant nearest endpoint inside the obscured frame edge; trapezoidal area divided by step width.",
        "type": "raster digitization of plotted curves, not author raw training observations",
        "reward_axis": "The source does not document reward rescaling or smoothing.",
        "windows": {}
    }
    with (args.output / "figure7-digitized.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["series", "step", "plotted_value", "pixel_column", "midline_row"])
        for tag, axes in AXES.items():
            points = extract(image, axes)
            for step, val, x, y in points:
                writer.writerow([tag, f"{step:.6f}", f"{val:.12f}", int(x), y])
            x, y = points[:, 0], points[:, 1]
            meta["windows"][tag] = {
                "visible_step_range": [float(x[0]), float(x[-1])],
                "means": {f"{lo}-{hi}": window_mean(x, y, lo, hi) for lo, hi in WINDOWS},
                "pixel_columns": len(x)
            }
    (args.output / "figure7-reference.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta["windows"], indent=2))


if __name__ == "__main__":
    main()
