#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic INTENSITY dataset generator.

We simulate a scalar latent z(t) solving
    z''(t) + γ1 z'(t) + γ0 z(t) = 0
with user-specified γ0, γ1, z(0), z'(0).

For each trajectory:
  1) Solve the ODE analytically.
  2) Linearly normalize this trajectory to
         z_norm(t) ∈ [z_min, z_max]  (default [0.2, 1.0])
     (per-trajectory).
  3) Render a fixed shape (regular or irregular) with uniform
     intensity = z_norm(t) on a black background.

This mirrors the "Intensity" toy dataset.
"""

import argparse
import os
import numpy as np
import imageio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path


# ---------- ODE solution: z'' + γ1 z' + γ0 z = 0 ----------
def z_series(
    gamma0: float,
    gamma1: float,
    duration: float,
    fps: int,
    z0: float,
    dz0: float,
):
    """
    Analytic solution of:
        z'' + γ1 z' + γ0 z = 0
    with initial conditions:
        z(0) = z0, z'(0) = dz0
    """
    n = int(round(duration * fps))
    t = np.linspace(0.0, duration, n, endpoint=False)

    D = gamma1**2 - 4.0 * gamma0
    eps = 1e-12

    if D < -eps:  # under-damped
        zeta = gamma1 / 2.0
        omega2 = gamma0 - zeta**2
        omega = np.sqrt(max(omega2, 0.0))
        A = z0
        B = (dz0 + zeta * z0) / (omega if omega != 0.0 else 1.0)
        exp = np.exp(-zeta * t)
        z = exp * (A * np.cos(omega * t) + B * np.sin(omega * t))

    elif D > eps:  # over-damped
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


# ---------- Per-trajectory normalization ----------
def normalize_z(z, z_min: float = 0.2, z_max: float = 1.0, clip: bool = True):
    """
    Affine-rescale one trajectory z(t) to [z_min, z_max].
    If (almost) constant, return a constant series at the mid-point.
    """
    z = np.asarray(z, dtype=np.float64)
    lo = float(np.min(z))
    hi = float(np.max(z))

    if abs(hi - lo) < 1e-9:
        return np.full_like(z, (z_min + z_max) / 2.0)

    z_scaled = (z - lo) / (hi - lo)
    z_scaled = z_min + z_scaled * (z_max - z_min)
    if clip:
        z_scaled = np.clip(z_scaled, z_min, z_max)
    return z_scaled


# ---------- Shape mask generation ----------
def make_shape_mask(
    size: int,
    mode: str = "random_polygon",
    seed: int = 0,
):
    """
    Return a boolean mask of shape (size, size).

    mode:
      - "circle": centered disk
      - "square": centered axis-aligned square
      - "triangle": centered isosceles triangle
      - "random_polygon": irregular convex-ish polygon
    """
    rng = np.random.RandomState(seed)
    H = W = size

    yy, xx = np.mgrid[0:H, 0:W]
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    if mode == "circle":
        r = 0.35 * min(W, H)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

    elif mode == "square":
        s = int(0.5 * min(W, H))
        x0 = int(cx - s / 2)
        x1 = x0 + s
        y0 = int(cy - s / 2)
        y1 = y0 + s
        mask = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)

    elif mode == "triangle":
        # simple upright isosceles triangle (barycentric test)
        h = int(0.6 * H)
        base_y = int(cy + h / 2)
        top_y = base_y - h
        left_x = int(cx - h / 2)
        right_x = int(cx + h / 2)

        v0 = np.array([right_x - left_x, 0.0])
        v1 = np.array([cx - left_x, top_y - base_y])
        v2x = xx - left_x
        v2y = yy - base_y

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot11 = np.dot(v1, v1)
        dot02 = v2x * v0[0] + v2y * v0[1]
        dot12 = v2x * v1[0] + v2y * v1[1]

        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-9)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        mask = (u >= 0) & (v >= 0) & (u + v <= 1)

    else:
        # "random_polygon": random convex-ish polygon around center
        num_verts = rng.randint(6, 11)
        angles = np.sort(rng.rand(num_verts) * 2.0 * np.pi)
        r_min = 0.25 * min(W, H)
        r_max = 0.45 * min(W, H)
        rs = rng.uniform(r_min, r_max, size=num_verts)
        xs = cx + rs * np.cos(angles)
        ys = cy + rs * np.sin(angles)
        verts = np.stack([xs, ys], axis=1)

        path = Path(verts, closed=True)
        pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        inside = path.contains_points(pts)
        mask = inside.reshape(H, W)

    # Ensure mask not empty
    if not mask.any():
        r = 0.3 * min(W, H)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

    return mask


# ---------- Rendering ----------
def render_intensity_video(
    T,
    z_vals,
    mask,
    out_path: str,
    fps: int,
    invert: bool = False,
):
    """
    Render grayscale video:
      - foreground (mask) intensity = z_vals[t] in [0,1]
      - background = 0 (or inverted if needed)
    """
    H, W = mask.shape

    out_dir = os.path.dirname(out_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)

    try:
        for val in z_vals:
            fg = float(val)
            if invert:
                # shape dark, background bright
                frame = np.where(mask,
                                 (1.0 - fg) * 255.0,
                                 fg * 255.0)
            else:
                frame = np.where(mask,
                                 fg * 255.0,
                                 0.0)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frame_rgb = np.stack([frame, frame, frame], axis=-1)
            writer.append_data(frame_rgb)
    finally:
        writer.close()

    print(f"[OK] saved intensity video -> {out_path}")


# ---------- MAIN ----------
def main(args):
    # 1) ODE trajectory
    T, z = z_series(
        gamma0=args.gamma0,
        gamma1=args.gamma1,
        duration=args.duration,
        fps=args.fps,
        z0=args.z0,
        dz0=args.dz0,
    )

    # 2) Normalize to [z_min, z_max]
    z_norm = normalize_z(z, z_min=args.z_min, z_max=args.z_max, clip=True)

    # 3) Shape mask
    mask = make_shape_mask(
        args.size,
        mode=args.shape_mode,
        seed=args.seed,
    )

    # 4) Render
    render_intensity_video(
        T,
        z_norm,
        mask,
        out_path=args.out,
        fps=args.fps,
        invert=args.invert,
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Synthetic INTENSITY dataset: z'' + γ1 z' + γ0 z = 0 controls gray level."
    )

    # dynamics
    ap.add_argument("--gamma0", type=float, default=4.0, help="γ0")
    ap.add_argument("--gamma1", type=float, default=0.0, help="γ1")
    ap.add_argument("--z0", type=float, default=1.0, help="z(0)")
    ap.add_argument("--dz0", type=float, default=0.0, help="z'(0)")

    # time & resolution
    ap.add_argument("--duration", type=float, default=5.0, help="video seconds")
    ap.add_argument("--fps", type=int, default=20, help="frames per second")
    ap.add_argument("--size", type=int, default=64, help="frame size (HxW)")

    # intensity range
    ap.add_argument("--z_min", type=float, default=0.2,
                    help="min normalized intensity (0-1)")
    ap.add_argument("--z_max", type=float, default=1.0,
                    help="max normalized intensity (0-1)")

    # shape control
    ap.add_argument("--shape_mode",
                    type=str,
                    default="random_polygon",
                    choices=["circle", "square", "triangle", "random_polygon"],
                    help="foreground shape type")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed for random shapes")
    ap.add_argument("--invert", action="store_true",
                    help="invert foreground/background mapping")

    ap.add_argument(
        "-o", "--out",
        type=str,
        default="intensity_dataset/5.0_20fps_z1.0_dz0.0.mp4",
        help="output mp4 path",
    )

    return ap.parse_args()

if __name__ == "__main__":
    base = parse_args()
    save_folder = "intensity_dataset/critical"
    os.makedirs(save_folder, exist_ok=True)
    for z0 in [1.0]:
        for dz0 in [-200, 0, 200]: 
            for duration in [2.0]:
                for fps in [10]:
                    base.gamma0 = 4.0
                    base.gamma1 = 4.0      
                    base.z0 = z0
                    base.dz0 = dz0
                    base.duration = duration
                    base.fps = fps
                    base.size = 64
                    base.out = f"{save_folder}/{duration:.1f}_{fps}fps_z{z0}_dz{dz0}.mp4"
                    main(base)
