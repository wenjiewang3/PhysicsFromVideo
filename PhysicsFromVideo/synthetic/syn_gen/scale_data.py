

import argparse
import os
import numpy as np
import imageio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def z_series(
    gamma0: float,
    gamma1: float,
    duration: float,
    fps: int,
    z0: float,
    dz0: float,
):
    n = int(round(duration * fps))
    t = np.linspace(0.0, duration, n, endpoint=False)

    D = gamma1 ** 2 - 4.0 * gamma0
    eps = 1e-12

    if D < -eps:  # underdamped
        zeta = gamma1 / 2.0
        omega2 = gamma0 - zeta ** 2
        omega = np.sqrt(max(omega2, 0.0))
        A = z0
        B = (dz0 + zeta * z0) / (omega if omega != 0.0 else 1.0)
        exp = np.exp(-zeta * t)
        z = exp * (A * np.cos(omega * t) + B * np.sin(omega * t))

    elif D > eps:  # overdamped
        s = np.sqrt(D)
        r1 = (-gamma1 + s) / 2.0
        r2 = (-gamma1 - s) / 2.0
        denom = (r1 - r2) if (r1 - r2) != 0.0 else 1.0
        C1 = (dz0 - r2 * z0) / denom
        C2 = (r1 * z0 - dz0) / denom
        z = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)

    else:  # critically damped
        zeta = gamma1 / 2.0
        A = z0
        B = dz0 + zeta * z0
        exp = np.exp(-zeta * t)
        z = (A + B * t) * exp

    return t, z


def map_to_r(z, r_min: float = -10.0, r_max: float = 10.0):
    z = np.asarray(z, dtype=np.float64)
    lo = float(np.min(z))
    hi = float(np.max(z))

    if abs(hi - lo) < 1e-9:
        return np.zeros_like(z)

    r = r_min + (z - lo) / (hi - lo) * (r_max - r_min)
    return np.clip(r, r_min, r_max)


# ---------- Rendering ----------
def render_scale(
    T,
    r_real,
    out: str = "scale/overdamped/2.0_20fps_z1.0_dz0.0.mp4",
    fps: int = 20,
    size: int = 64,
    args_radius_frac: float = 0.25,
    max_fill_frac: float = 0.9,
):
    """
    Render asymmetric circle video.

    Args:
        T:          time stamps (unused visually, only length matters)
        r_real:     array of r(t) in [-10,10]
        out:        output mp4 path
        fps:        frames per second
        size:       frame size (HxW)
        args_radius_frac:
                    R_args = args_radius_frac * size
        max_fill_frac:
                    safety factor so radius never exceeds
                    max_fill_frac * (size/2)
    """
    H = W = size
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    # precompute distance grid
    yy, xx = np.mgrid[0:H, 0:W]
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx ** 2 + dy ** 2)

    R_args = args_radius_frac * size
    R_max_allowed = max_fill_frac * (size / 2.0)

    # f(t) ∈ [-1,1]
    f = np.clip(r_real / 10.0, -1.0, 1.0)

    # ensure directory exists
    out_dir = os.path.dirname(out)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8)

    try:
        for ft in f:
            # asymmetric radii
            R_left = R_args * (1.0 + ft)
            R_right = R_args * (1.0 - ft)

            R_left = np.clip(R_left, 0.0, R_max_allowed)
            R_right = np.clip(R_right, 0.0, R_max_allowed)

            # left: x <= cx uses R_left; right: x > cx uses R_right
            mask_left = (xx <= cx) & (dist <= R_left)
            mask_right = (xx > cx) & (dist <= R_right)
            mask = mask_left | mask_right

            frame = np.zeros((H, W), dtype=np.uint8)
            frame[mask] = 255  # white foreground

            frame_rgb = np.stack([frame, frame, frame], axis=-1)
            writer.append_data(frame_rgb)
    finally:
        writer.close()

    print(f"[OK] saved -> {out}")


# ---------- MAIN ----------
def main(args):
    # 1) solve ODE
    T, z = z_series(
        gamma0=args.gamma0,
        gamma1=args.gamma1,
        duration=args.duration,
        fps=args.fps,
        z0=args.z0,
        dz0=args.dz0,
    )

    # 2) map to r_real in [-10,10]
    r_real = map_to_r(z, r_min=-10.0, r_max=10.0)

    # 3) render
    render_scale(
        T,
        r_real,
        out=args.out,
        fps=args.fps,
        size=args.size,
        args_radius_frac=args.args_radius_frac,
        max_fill_frac=args.max_fill_frac,
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Scale dataset: asymmetric circle driven by z'' + γ1 z' + γ0 z = 0"
    )

    # dynamics
    ap.add_argument("--gamma0", type=float, default=4.0, help="γ0")
    ap.add_argument("--gamma1", type=float, default=0.0, help="γ1")
    ap.add_argument("--z0", type=float, default=1.0, help="z(0)")
    ap.add_argument("--dz0", type=float, default=0.0, help="z'(0)")

    # time & image
    ap.add_argument("--duration", type=float, default=5.0, help="video seconds")
    ap.add_argument("--fps", type=int, default=20, help="frames per second")
    ap.add_argument("--size", type=int, default=64, help="frame size (px)")

    # geometry
    ap.add_argument("--args_radius_frac", type=float, default=0.25,
                    help="R_args as fraction of image size")
    ap.add_argument("--max_fill_frac", type=float, default=0.9,
                    help="max radius fraction of half image size")

    ap.add_argument(
        "-o", "--out",
        type=str,
        default="scale_dataset/test/5.0_20fps_z1.0_dz0.0.mp4",
        help="output MP4 path",
    )

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    save_folder = "scale_dataset/undamped"
    os.makedirs(save_folder, exist_ok=True)
    for z0 in [1.0, -1.0]:
        for dz0 in [0.0, 10.0, -10.0]:
            for duration in [2.0]:
                fps = 60
                args.gamma0 = 4.0
                args.gamma1 = 0.0
                args.z0 = z0
                args.dz0 = dz0
                args.duration = duration
                args.fps = fps
                args.size = 64
                args.out = f"{save_folder}/{duration:.1f}_{fps}fps_z{z0}_dz{dz0}.mp4"
                main(args)

