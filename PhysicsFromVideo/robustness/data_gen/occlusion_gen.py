import argparse
import os

import imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from noise_background_gen import theta_series


OCCLUSION_LEVELS = {
    1: {
        "mode": "continuous",
        "angle_frac": 0.22,
        "r_outer_frac": 0.72,
        "cell_frac": 0.10,
        "start_side": "left",
    },
    2: {
        "mode": "continuous",
        "angle_frac": 0.38,
        "r_outer_frac": 0.82,
        "cell_frac": 0.11,
        "start_side": "left",
    },
    3: {
        "mode": "continuous",
        "angle_frac": 0.56,
        "r_outer_frac": 0.92,
        "cell_frac": 0.12,
        "start_side": "left",
    },
    4: {
        "mode": "continuous",
        "angle_frac": 0.78,
        "r_outer_frac": 1.04,
        "cell_frac": 0.13,
        "start_side": "left",
    },
    5: {
        "mode": "discrete",
        "segments": 2,
        "segment_frac": 0.13,
        "r_outer_frac": 0.76,
        "cell_frac": 0.10,
    },
    6: {
        "mode": "discrete",
        "segments": 3,
        "segment_frac": 0.15,
        "r_outer_frac": 0.86,
        "cell_frac": 0.11,
    },
    7: {
        "mode": "discrete",
        "segments": 4,
        "segment_frac": 0.17,
        "r_outer_frac": 0.96,
        "cell_frac": 0.12,
    },
    8: {
        "mode": "discrete",
        "segments": 5,
        "segment_frac": 0.20,
        "r_outer_frac": 1.06,
        "cell_frac": 0.13,
    },
    9: {
        "mode": "continuous",
        "angle_frac": 0.16,
        "r_outer_frac": 0.64,
        "cell_frac": 0.09,
        "start_side": "left",
    },
    10: {
        "mode": "continuous",
        "angle_frac": 0.10,
        "r_outer_frac": 0.56,
        "cell_frac": 0.08,
        "start_side": "left",
    },
    11: {
        "mode": "continuous",
        "angle_frac": 0.22,
        "r_outer_frac": 0.72,
        "cell_frac": 0.10,
        "start_side": "center",
    },
    12: {
        "mode": "continuous",
        "angle_frac": 0.34,
        "r_outer_frac": 0.82,
        "cell_frac": 0.10,
        "start_side": "center",
    },
}


def compute_view_bounds(L, pivot_y_frac, zoom):
    margin = 0.15 * L
    span = 2.0 * (L + margin) * zoom
    half = span * 0.5
    p = float(np.clip(pivot_y_frac, 0.0, 1.0))
    xmin, xmax = -half, half
    ymin = -p * span
    ymax = ymin + span
    return xmin, xmax, ymin, ymax


def sample_angle_windows(theta_min, theta_max, count, width_frac):
    span = max(float(theta_max - theta_min), 1e-6)
    width = max(span * float(width_frac), 1e-4)

    if count <= 1:
        centers = [0.5 * (theta_min + theta_max)]
    else:
        centers = np.linspace(theta_min + 0.5 * width, theta_max - 0.5 * width, count)

    windows = []
    for c in centers:
        lo = max(theta_min, float(c) - 0.5 * width)
        hi = min(theta_max, float(c) + 0.5 * width)
        windows.append((lo, hi))
    return windows


def sample_sector_blocks(angle_windows, r_inner, r_outer, cell):
    blocks = []
    if r_outer <= r_inner:
        return blocks

    radial_step = max(cell * 0.74, 1e-4)
    angular_step = max(cell / max(r_outer, 1e-4) * 0.72, 1e-4)
    radii = np.arange(r_inner, r_outer + radial_step * 0.5, radial_step)

    for theta_lo, theta_hi in angle_windows:
        angles = np.arange(theta_lo, theta_hi + angular_step * 0.5, angular_step)
        for r in radii:
            for th in angles:
                x = float(r * np.sin(th))
                y = float(-r * np.cos(th))
                blocks.append((x, y, cell))
    return blocks


def build_occlusion_blocks(level, TH, L, bob_ratio):
    cfg = OCCLUSION_LEVELS[level]
    theta_min = float(np.min(TH))
    theta_max = float(np.max(TH))
    theta_center = 0.5 * (theta_min + theta_max)
    theta_span = max(theta_max - theta_min, 1e-6)

    cell = float(cfg["cell_frac"]) * float(L)
    bob_r = float(bob_ratio) * float(L)
    r_inner = max(0.08 * float(L), 0.5 * cell)
    # Extend past the bob center so the entire bob, including its lower edge,
    # is buried inside the occlusion band instead of peeking out at the ends.
    r_outer = max(float(cfg["r_outer_frac"]) * float(L), float(L) + 1.35 * bob_r + 0.75 * cell)

    if cfg["mode"] == "continuous":
        cover = theta_span * float(cfg["angle_frac"])
        start_side = cfg.get("start_side", "left")
        if start_side == "right":
            angle_windows = [(theta_max - cover, theta_max)]
        elif start_side == "center":
            half = 0.5 * cover
            angle_windows = [(theta_center - half, theta_center + half)]
        else:
            angle_windows = [(theta_min, theta_min + cover)]
    else:
        angle_windows = sample_angle_windows(
            theta_min,
            theta_max,
            int(cfg["segments"]),
            float(cfg["segment_frac"]),
        )

    return sample_sector_blocks(angle_windows, r_inner, r_outer, cell)


def draw_occlusion_blocks(ax, blocks, gray):
    g = str(np.clip(gray, 0, 255) / 255.0)
    for x, y, side in blocks:
        ax.add_patch(
            Rectangle(
                (x - 0.5 * side, y - 0.5 * side),
                side,
                side,
                facecolor=g,
                edgecolor="none",
                zorder=4,
            )
        )


def render_bw_with_occlusion(
    T,
    TH,
    L=1.0,
    out="pendulum.mp4",
    fps=60,
    size=64,
    rod_px=8.0,
    bob_ratio=0.1,
    trail=0,
    pivot_y_frac=0.5,
    zoom=1.0,
    occlusion_level=1,
    occluder_gray=0,
):
    xs = L * np.sin(TH)
    ys = -L * np.cos(TH)

    xmin, xmax, ymin, ymax = compute_view_bounds(L, pivot_y_frac, zoom)
    bob_r = bob_ratio * L
    color = "black"
    occlusion_blocks = build_occlusion_blocks(occlusion_level, TH, L, bob_ratio)

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8)
    try:
        for i in range(len(T)):
            base_res = 512
            fig = plt.figure(
                figsize=(base_res / 100, base_res / 100),
                dpi=size / base_res * 100,
                facecolor="white",
            )
            ax = plt.Axes(fig, [0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)

            ax.set_aspect("equal", adjustable="box")
            ax.set_autoscale_on(False)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            x, y = xs[i], ys[i]
            ax.plot([0, x], [0, y], color=color, linewidth=rod_px, solid_capstyle="round", zorder=2)
            ax.add_patch(Circle((x, y), radius=bob_r, facecolor=color, edgecolor=color, zorder=3))

            if trail > 0:
                j0 = max(0, i - trail)
                ax.plot(
                    xs[j0 : i + 1],
                    ys[j0 : i + 1],
                    color=color,
                    alpha=0.6,
                    linewidth=max(1.0, rod_px * 0.25),
                    zorder=1,
                )

            draw_occlusion_blocks(ax, occlusion_blocks, occluder_gray)

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            frame = rgba[:, :, :3].copy()
            writer.append_data(frame)
            plt.close(fig)
    finally:
        writer.close()

    print(f"[OK] saved -> {out}")


def main(args):
    theta0 = np.deg2rad(args.theta0_deg)
    dtheta0 = np.deg2rad(args.dtheta0_deg_s)

    T, TH = theta_series(
        args.gamma0,
        args.gamma1,
        duration=args.duration,
        fps=args.fps,
        theta0=theta0,
        dtheta0=dtheta0,
    )

    render_bw_with_occlusion(
        T,
        TH,
        L=args.L,
        out=args.out,
        fps=args.fps,
        size=args.size,
        rod_px=args.rod_px,
        bob_ratio=args.bob_ratio,
        trail=args.trail,
        pivot_y_frac=args.pivot_y_frac,
        zoom=args.zoom,
        occlusion_level=args.occlusion_level,
        occluder_gray=args.occluder_gray,
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description="2D pendulum with fixed trajectory occlusion: level1-4 continuous, level5-8 discrete, level9-12 lighter variants"
    )
    ap.add_argument("--gamma0", type=float, default=4.0, help="gamma0 (stiffness-like)")
    ap.add_argument("--gamma1", type=float, default=5.0, help="gamma1 (damping)")
    ap.add_argument("--theta0_deg", type=float, default=90.0, help="initial angle in degrees")
    ap.add_argument("--dtheta0_deg_s", type=float, default=0.0, help="initial angular velocity in degrees per second")
    ap.add_argument("--duration", type=float, default=10.0, help="video seconds")
    ap.add_argument("--fps", type=int, default=60, help="frames per second")
    ap.add_argument("--L", type=float, default=1.0, help="visual rod length")
    ap.add_argument("--size", type=int, default=64, help="square resolution (px)")
    ap.add_argument("--rod_px", type=float, default=8.0, help="rod line width (px)")
    ap.add_argument("--bob_ratio", type=float, default=0.1, help="bob radius = ratio * L")
    ap.add_argument("--trail", type=int, default=0, help="trail length in frames (0 off)")
    ap.add_argument("--pivot_y_frac", type=float, default=0.5, help="pivot vertical position in frame")
    ap.add_argument("--zoom", type=float, default=1.0, help="camera zoom")
    ap.add_argument("--occlusion_level", type=int, default=1, help="occlusion level in [1, 12]")
    ap.add_argument("--occluder_gray", type=int, default=0, help="occlusion gray in [0,255], 0 is black")
    ap.add_argument("-o", "--out", type=str, default="generated_video/pendulum_occlusion.mp4", help="output MP4 path")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_folder = "ablation_data/occlusion_underdamped"
    os.makedirs(save_folder, exist_ok=True)

    for args.theta0_deg in [60]:
        for args.dtheta0_deg_s in [0]:
            for args.duration in [2.5]:
                for args.fps in [20]:
                    duration_pi = args.duration
                    for level in sorted(OCCLUSION_LEVELS.keys()):
                        args.gamma0 = 4.0016
                        args.gamma1 = 0.08
                        args.occlusion_level = level
                        args.out = (
                            f"{save_folder}/{duration_pi:.1f}pi_{int(args.fps)}fps_"
                            f"degree{args.theta0_deg}_degs{args.dtheta0_deg_s}_level{level}.mp4"
                        )
                        args.duration = duration_pi * np.pi
                        main(args)
                        args.duration = duration_pi
