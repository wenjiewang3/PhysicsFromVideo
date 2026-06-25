import argparse
import os

import imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from setting_registry import get_setting_level_params, get_setting_levels, get_setting_save_folder


def theta_series(
    gamma0: float,
    gamma1: float,
    duration: float = 10.0,
    fps: int = 60,
    theta0: float = 1.0,
    dtheta0: float = 0.0,
):
    n = int(round(duration * fps))
    t = np.linspace(0.0, duration, n, endpoint=False)

    D = gamma1**2 - 4.0 * gamma0
    eps = 1e-12

    if D < -eps:
        zeta = gamma1 / 2.0
        omega = np.sqrt(gamma0 - zeta**2)
        A = theta0
        B = (dtheta0 + zeta * theta0) / (omega if omega != 0 else 1.0)
        theta = np.exp(-zeta * t) * (A * np.cos(omega * t) + B * np.sin(omega * t))
    elif D > eps:
        s = np.sqrt(D)
        r1 = (-gamma1 + s) / 2.0
        r2 = (-gamma1 - s) / 2.0
        denom = (r1 - r2) if (r1 - r2) != 0 else 1.0
        C1 = (dtheta0 - r2 * theta0) / denom
        C2 = (r1 * theta0 - dtheta0) / denom
        theta = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)
    else:
        zeta = gamma1 / 2.0
        A = theta0
        B = dtheta0 + zeta * theta0
        theta = (A + B * t) * np.exp(-zeta * t)

    return t, theta


def theta_series_with_derivs(
    gamma0: float,
    gamma1: float,
    duration: float = 10.0,
    fps: int = 60,
    theta0: float = 1.0,
    dtheta0: float = 0.0,
):
    n = int(round(duration * fps))
    t = np.linspace(0.0, duration, n, endpoint=False)

    D = gamma1**2 - 4.0 * gamma0
    eps = 1e-12

    if D < -eps:
        zeta = gamma1 / 2.0
        omega = np.sqrt(max(gamma0 - zeta**2, 0.0))
        A = theta0
        B = (dtheta0 + zeta * theta0) / (omega if omega != 0 else 1.0)
        exp = np.exp(-zeta * t)
        cos = np.cos(omega * t)
        sin = np.sin(omega * t)
        th = exp * (A * cos + B * sin)
        thd = exp * (-zeta * (A * cos + B * sin) + (-A * omega * sin + B * omega * cos))
    elif D > eps:
        s = np.sqrt(D)
        r1 = (-gamma1 + s) / 2.0
        r2 = (-gamma1 - s) / 2.0
        denom = (r1 - r2) if (r1 - r2) != 0 else 1.0
        C1 = (dtheta0 - r2 * theta0) / denom
        C2 = (r1 * theta0 - dtheta0) / denom
        e1 = np.exp(r1 * t)
        e2 = np.exp(r2 * t)
        th = C1 * e1 + C2 * e2
        thd = C1 * r1 * e1 + C2 * r2 * e2
    else:
        zeta = gamma1 / 2.0
        A = theta0
        B = dtheta0 + zeta * theta0
        exp = np.exp(-zeta * t)
        th = (A + B * t) * exp
        thd = (B - zeta * (A + B * t)) * exp

    thdd = -gamma1 * thd - gamma0 * th
    return t, th, thd, thdd


def sample_square_noise(
    rng,
    count,
    xmin,
    xmax,
    ymin,
    ymax,
    size_px,
    min_size_px,
    max_size_px,
    gray_min,
    gray_max,
):
    if count <= 0:
        return []

    min_size_px = max(1, int(min_size_px))
    max_size_px = max(min_size_px, int(max_size_px))
    gray_min = int(np.clip(gray_min, 0, 255))
    gray_max = int(np.clip(gray_max, gray_min, 255))

    span = min(xmax - xmin, ymax - ymin)
    px_to_world = span / float(size_px)
    squares = []

    for _ in range(count):
        side_px = int(rng.integers(min_size_px, max_size_px + 1))
        side_world = side_px * px_to_world
        max_x = max(xmin, xmax - side_world)
        max_y = max(ymin, ymax - side_world)
        x0 = float(rng.uniform(xmin, max_x if max_x > xmin else xmin + 1e-12))
        y0 = float(rng.uniform(ymin, max_y if max_y > ymin else ymin + 1e-12))
        gray = int(rng.integers(gray_min, gray_max + 1))
        squares.append((x0, y0, side_world, gray))

    return squares


def draw_square_noise(ax, squares):
    for x0, y0, side, gray in squares:
        ax.add_patch(
            Rectangle(
                (x0, y0),
                side,
                side,
                facecolor=str(gray / 255.0),
                edgecolor="none",
                zorder=0,
            )
        )


def render_bw_with_noise(
    T,
    TH,
    L=1.0,
    out="pendulum.mp4",
    fps=60,
    size=512,
    rod_px=6.0,
    bob_ratio=0.06,
    trail=0,
    pivot_y_frac=0.5,
    zoom=1.0,
    noise_count=0,
    noise_min_size_px=4,
    noise_max_size_px=12,
    noise_gray_min=220,
    noise_gray_max=245,
    noise_seed=42,
    noise_refresh_every=0,
):
    xs = L * np.sin(TH)
    ys = -L * np.cos(TH)

    margin = 0.15 * L
    span = 2.0 * (L + margin) * zoom
    half = span * 0.5

    p = float(np.clip(pivot_y_frac, 0.0, 1.0))
    xmin, xmax = -half, half
    ymin = -p * span
    ymax = ymin + span

    bob_r = bob_ratio * L
    color = "black"
    rng = np.random.default_rng(noise_seed)

    static_squares = sample_square_noise(
        rng=rng,
        count=noise_count,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        size_px=size,
        min_size_px=noise_min_size_px,
        max_size_px=noise_max_size_px,
        gray_min=noise_gray_min,
        gray_max=noise_gray_max,
    )

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

            if noise_refresh_every and i % noise_refresh_every == 0:
                squares = sample_square_noise(
                    rng=rng,
                    count=noise_count,
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                    size_px=size,
                    min_size_px=noise_min_size_px,
                    max_size_px=noise_max_size_px,
                    gray_min=noise_gray_min,
                    gray_max=noise_gray_max,
                )
            else:
                squares = static_squares

            draw_square_noise(ax, squares)

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

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            frame = rgba[:, :, :3].copy()
            writer.append_data(frame)
            plt.close(fig)
    finally:
        writer.close()

    print(f"[OK] saved -> {out}")


def apply_noise_level(args, level):
    cfg = get_setting_level_params(args.setting_case, level)
    args.noise_count = cfg["noise_count"]
    args.noise_min_size_px = cfg["noise_min_size_px"]
    args.noise_max_size_px = cfg["noise_max_size_px"]
    args.noise_gray_min = cfg["noise_gray_min"]
    args.noise_gray_max = cfg["noise_gray_max"]
    args.noise_refresh_every = cfg["noise_refresh_every"]
    args.noise_level = level
    return args


def main(args):
    if not hasattr(args, "setting_case"):
        args.setting_case = "noise_background_underdamped"

    if getattr(args, "noise_level", None) is not None:
        apply_noise_level(args, args.noise_level)

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

    render_bw_with_noise(
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
        noise_count=args.noise_count,
        noise_min_size_px=args.noise_min_size_px,
        noise_max_size_px=args.noise_max_size_px,
        noise_gray_min=args.noise_gray_min,
        noise_gray_max=args.noise_gray_max,
        noise_seed=args.noise_seed,
        noise_refresh_every=args.noise_refresh_every,
    )

    theta_series_with_derivs(
        args.gamma0,
        args.gamma1,
        duration=args.duration,
        fps=args.fps,
        theta0=theta0,
        dtheta0=dtheta0,
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description="B/W pendulum with square background noise: θ'' + γ1 θ' + γ0 θ = 0"
    )

    ap.add_argument("--gamma0", type=float, default=4.0, help="γ0 (stiffness-like)")
    ap.add_argument("--gamma1", type=float, default=5.0, help="γ1 (damping)")
    ap.add_argument("--theta0_deg", type=float, default=90.0, help="initial angle in degrees")
    ap.add_argument(
        "--dtheta0_deg_s",
        type=float,
        default=0.0,
        help="initial angular velocity in degrees per second",
    )

    ap.add_argument("--duration", type=float, default=10.0, help="video seconds")
    ap.add_argument("--fps", type=int, default=60, help="frames per second")
    ap.add_argument("--L", type=float, default=1.0, help="visual rod length")
    ap.add_argument("--size", type=int, default=64, help="square resolution (px)")
    ap.add_argument("--rod_px", type=float, default=8.0, help="rod line width (px)")
    ap.add_argument("--bob_ratio", type=float, default=0.1, help="bob radius = ratio * L")
    ap.add_argument("--trail", type=int, default=0, help="trail length in frames (0 off)")
    ap.add_argument(
        "--pivot_y_frac",
        type=float,
        default=0.5,
        help="pivot vertical position: 0.0=bottom, 0.5=center, 1.0=top",
    )
    ap.add_argument("--zoom", type=float, default=1.0, help="camera zoom")

    ap.add_argument("--noise_count", type=int, default=8, help="number of square noise patches")
    ap.add_argument("--noise_min_size_px", type=int, default=4, help="min square side length in pixels")
    ap.add_argument("--noise_max_size_px", type=int, default=10, help="max square side length in pixels")
    ap.add_argument("--noise_gray_min", type=int, default=0, help="min grayscale value for noise blocks")
    ap.add_argument("--noise_gray_max", type=int, default=0, help="max grayscale value for noise blocks")
    ap.add_argument("--noise_level", type=int, default=None, help="noise level in [1, 5]; level 2 matches the current default strength")
    ap.add_argument("--noise_seed", type=int, default=42, help="random seed for reproducible noise")
    ap.add_argument(
        "--noise_refresh_every",
        type=int,
        default=0,
        help="regenerate background every N frames; 0 keeps one static background for the full video",
    )

    ap.add_argument(
        "-o",
        "--out",
        type=str,
        default="generated_video/pendulum_10s.mp4",
        help="output MP4 path",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.setting_case = "noise_background_underdamped"
    save_folder = get_setting_save_folder(args.setting_case)
    os.makedirs(save_folder, exist_ok=True)
    for args.theta0_deg in [60]:
        for args.dtheta0_deg_s in [0]:
            for args.duration in [2.5]:
                for args.fps in [20]:
                    duration_pi = args.duration
                    for level in get_setting_levels(args.setting_case):
                        setting_params = get_setting_level_params(args.setting_case, level)
                        args.gamma0 = setting_params["gamma0"]
                        args.gamma1 = setting_params["gamma1"]
                        args.noise_level = level
                        args.out = (
                            f"{save_folder}/{duration_pi:.1f}pi_{int(args.fps)}fps_"
                            f"degree{args.theta0_deg}_degs{args.dtheta0_deg_s}_level{level}.mp4"
                        )
                        args.duration = duration_pi * np.pi
                        main(args)
                        args.duration = duration_pi
