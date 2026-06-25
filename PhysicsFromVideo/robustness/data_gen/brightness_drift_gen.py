import argparse
import os

import imageio
import numpy as np

from camera_jitter_gen import render_clean_frame
from noise_background_gen import theta_series, theta_series_with_derivs
from setting_registry import (
    get_setting_level_params,
    get_setting_levels,
    get_setting_save_folder,
)


SETTING_CASE = "brightness_drift_underdamped"


def build_linear_values(num_frames, start_value, end_value):
    if num_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    if num_frames == 1:
        return np.array([float(start_value)], dtype=np.float32)
    return np.linspace(float(start_value), float(end_value), num_frames, dtype=np.float32)


def build_motion_series(num_frames, start_value, end_value, wave_amp=0.0, wave_cycles=0.0, wave_phase_deg=0.0, use_cos=False):
    values = build_linear_values(num_frames, start_value, end_value)
    if num_frames <= 1 or abs(float(wave_amp)) <= 1e-8 or abs(float(wave_cycles)) <= 1e-8:
        return values

    idx = np.arange(num_frames, dtype=np.float32)
    denom = max(num_frames - 1, 1)
    phase = np.deg2rad(float(wave_phase_deg))
    angle = 2.0 * np.pi * float(wave_cycles) * idx / float(denom) + phase
    wave = np.cos(angle) if use_cos else np.sin(angle)
    return values + float(wave_amp) * wave.astype(np.float32)


def build_soft_spot(x, y, center_x, center_y, sigma):
    sigma = max(float(sigma), 1e-3)
    dist2 = ((x - float(center_x)) ** 2 + (y - float(center_y)) ** 2) / (2.0 * sigma * sigma)
    spot = np.exp(-dist2)
    return (spot - spot.min()) / max(float(spot.max() - spot.min()), 1e-6)


def build_rotated_shadow_blob(x, y, center_x, center_y, sigma_x, sigma_y, angle_deg):
    sigma_x = max(float(sigma_x), 1e-3)
    sigma_y = max(float(sigma_y), 1e-3)
    angle = np.deg2rad(float(angle_deg))
    dx = x - float(center_x)
    dy = y - float(center_y)
    xr = np.cos(angle) * dx + np.sin(angle) * dy
    yr = -np.sin(angle) * dx + np.cos(angle) * dy
    dist = (xr / sigma_x) ** 2 + (yr / sigma_y) ** 2
    blob = np.exp(-0.5 * dist)
    return (blob - blob.min()) / max(float(blob.max() - blob.min()), 1e-6)


def build_lighting_mask(
    height,
    width,
    center_x,
    center_y,
    sigma,
    spot_gain,
    shadow_gain,
    vignette_gain,
    floor_gain,
    center_x_2=None,
    center_y_2=None,
    sigma_2=None,
    spot_gain_2=0.0,
    shadow_blob_center_x=None,
    shadow_blob_center_y=None,
    shadow_blob_sigma_x=None,
    shadow_blob_sigma_y=None,
    shadow_blob_angle_deg=0.0,
    shadow_blob_gain=0.0,
    shadow_blob_center_x_2=None,
    shadow_blob_center_y_2=None,
    shadow_blob_sigma_x_2=None,
    shadow_blob_sigma_y_2=None,
    shadow_blob_angle_deg_2=0.0,
    shadow_blob_gain_2=0.0,
):
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (xx + 0.5) / float(width)
    y = (yy + 0.5) / float(height)

    spot = build_soft_spot(x, y, center_x, center_y, sigma)
    diagonal_shadow = 0.62 * x + 0.38 * y
    radial = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    radial = radial / max(float(radial.max()), 1e-6)

    mask = 1.0 + float(spot_gain) * spot
    if center_x_2 is not None and center_y_2 is not None and sigma_2 is not None and float(spot_gain_2) > 0.0:
        mask += float(spot_gain_2) * build_soft_spot(x, y, center_x_2, center_y_2, sigma_2)
    mask -= float(shadow_gain) * diagonal_shadow
    mask -= float(vignette_gain) * radial
    if (
        shadow_blob_center_x is not None
        and shadow_blob_center_y is not None
        and shadow_blob_sigma_x is not None
        and shadow_blob_sigma_y is not None
        and float(shadow_blob_gain) > 0.0
    ):
        shadow_blob = build_rotated_shadow_blob(
            x,
            y,
            shadow_blob_center_x,
            shadow_blob_center_y,
            shadow_blob_sigma_x,
            shadow_blob_sigma_y,
            shadow_blob_angle_deg,
        )
        mask -= float(shadow_blob_gain) * shadow_blob
    if (
        shadow_blob_center_x_2 is not None
        and shadow_blob_center_y_2 is not None
        and shadow_blob_sigma_x_2 is not None
        and shadow_blob_sigma_y_2 is not None
        and float(shadow_blob_gain_2) > 0.0
    ):
        shadow_blob_2 = build_rotated_shadow_blob(
            x,
            y,
            shadow_blob_center_x_2,
            shadow_blob_center_y_2,
            shadow_blob_sigma_x_2,
            shadow_blob_sigma_y_2,
            shadow_blob_angle_deg_2,
        )
        mask -= float(shadow_blob_gain_2) * shadow_blob_2
    return np.clip(mask, float(floor_gain), 1.10).astype(np.float32)


def apply_brightness_lighting(frame, gain, lighting_mask):
    adjusted = np.clip(frame.astype(np.float32) * float(gain) * lighting_mask[..., None], 0.0, 255.0)
    return adjusted.astype(np.uint8)


def build_brightness_factors(num_frames, level_cfg):
    profile = level_cfg.get("brightness_profile", "constant")
    if profile not in {"constant", "linear", "linear_wave"}:
        raise ValueError(f"Unsupported brightness profile: {profile}")

    if profile == "constant":
        base = np.full(num_frames, float(level_cfg["brightness_base_gain"]), dtype=np.float32)
    else:
        base = build_linear_values(
            num_frames=num_frames,
            start_value=level_cfg["brightness_start_gain"],
            end_value=level_cfg["brightness_end_gain"],
        )

    if profile == "linear_wave":
        idx = np.arange(num_frames, dtype=np.float32)
        denom = max(num_frames - 1, 1)
        phase = np.deg2rad(float(level_cfg.get("brightness_wave_phase_deg", 0.0)))
        angle = 2.0 * np.pi * float(level_cfg.get("brightness_wave_cycles", 0.0)) * idx / float(denom) + phase
        base = base + float(level_cfg.get("brightness_wave_amp", 0.0)) * np.sin(angle).astype(np.float32)

    min_gain = float(level_cfg.get("brightness_min_gain", 0.05))
    max_gain = float(level_cfg.get("brightness_max_gain", 1.0))
    return np.clip(base, min_gain, max_gain)


def build_optional_linear_values(num_frames, level_cfg, start_key, end_key):
    if start_key not in level_cfg or end_key not in level_cfg:
        return None
    return build_linear_values(num_frames, level_cfg[start_key], level_cfg[end_key])


def build_optional_motion_values(
    num_frames,
    level_cfg,
    start_key,
    end_key,
    wave_amp_key=None,
    wave_cycles_key=None,
    wave_phase_key=None,
    use_cos=False,
):
    if start_key not in level_cfg or end_key not in level_cfg:
        return None

    wave_amp = level_cfg.get(wave_amp_key, 0.0) if wave_amp_key else 0.0
    wave_cycles = level_cfg.get(wave_cycles_key, 0.0) if wave_cycles_key else 0.0
    wave_phase = level_cfg.get(wave_phase_key, 0.0) if wave_phase_key else 0.0
    return build_motion_series(
        num_frames=num_frames,
        start_value=level_cfg[start_key],
        end_value=level_cfg[end_key],
        wave_amp=wave_amp,
        wave_cycles=wave_cycles,
        wave_phase_deg=wave_phase,
        use_cos=use_cos,
    )


def render_bw_with_brightness_drift(
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
    brightness_level=1,
):
    level_cfg = get_setting_level_params(SETTING_CASE, brightness_level)
    gains = build_brightness_factors(len(T), level_cfg)
    light_center_x = build_motion_series(
        len(T),
        level_cfg["light_center_x_start"],
        level_cfg["light_center_x_end"],
        wave_amp=level_cfg.get("light_wave_amp_x", 0.0),
        wave_cycles=level_cfg.get("light_wave_cycles", 0.0),
        wave_phase_deg=level_cfg.get("light_wave_phase_deg", 0.0),
        use_cos=False,
    )
    light_center_y = build_motion_series(
        len(T),
        level_cfg["light_center_y_start"],
        level_cfg["light_center_y_end"],
        wave_amp=level_cfg.get("light_wave_amp_y", 0.0),
        wave_cycles=level_cfg.get("light_wave_cycles", 0.0),
        wave_phase_deg=level_cfg.get("light_wave_phase_deg", 0.0),
        use_cos=True,
    )
    light_center_x_2 = build_optional_motion_values(
        len(T),
        level_cfg,
        "light_center_x2_start",
        "light_center_x2_end",
        "light_wave_amp_x2",
        "light_wave_cycles_2",
        "light_wave_phase_deg_2",
        False,
    )
    light_center_y_2 = build_optional_motion_values(
        len(T),
        level_cfg,
        "light_center_y2_start",
        "light_center_y2_end",
        "light_wave_amp_y2",
        "light_wave_cycles_2",
        "light_wave_phase_deg_2",
        True,
    )
    shadow_blob_center_x = build_optional_motion_values(
        len(T),
        level_cfg,
        "shadow_blob_center_x_start",
        "shadow_blob_center_x_end",
        "shadow_blob_wave_amp_x",
        "shadow_blob_wave_cycles",
        "shadow_blob_wave_phase_deg",
        False,
    )
    shadow_blob_center_y = build_optional_motion_values(
        len(T),
        level_cfg,
        "shadow_blob_center_y_start",
        "shadow_blob_center_y_end",
        "shadow_blob_wave_amp_y",
        "shadow_blob_wave_cycles",
        "shadow_blob_wave_phase_deg",
        True,
    )
    shadow_blob_center_x_2 = build_optional_motion_values(
        len(T),
        level_cfg,
        "shadow_blob_center_x2_start",
        "shadow_blob_center_x2_end",
        "shadow_blob_wave_amp_x2",
        "shadow_blob_wave_cycles_2",
        "shadow_blob_wave_phase_deg_2",
        False,
    )
    shadow_blob_center_y_2 = build_optional_motion_values(
        len(T),
        level_cfg,
        "shadow_blob_center_y2_start",
        "shadow_blob_center_y2_end",
        "shadow_blob_wave_amp_y2",
        "shadow_blob_wave_cycles_2",
        "shadow_blob_wave_phase_deg_2",
        True,
    )

    xs = L * np.sin(TH)
    ys = -L * np.cos(TH)

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8)
    try:
        for i in range(len(T)):
            trail_xs = trail_ys = None
            if trail > 0:
                j0 = max(0, i - trail)
                trail_xs = xs[j0 : i + 1]
                trail_ys = ys[j0 : i + 1]

            clean_frame = render_clean_frame(
                theta=TH[i],
                L=L,
                size=size,
                rod_px=rod_px,
                bob_ratio=bob_ratio,
                trail_xs=trail_xs,
                trail_ys=trail_ys,
                pivot_y_frac=pivot_y_frac,
                zoom=zoom,
            )
            lighting_mask = build_lighting_mask(
                height=clean_frame.shape[0],
                width=clean_frame.shape[1],
                center_x=light_center_x[i],
                center_y=light_center_y[i],
                sigma=level_cfg["light_sigma"],
                spot_gain=level_cfg["light_spot_gain"],
                shadow_gain=level_cfg["light_shadow_gain"],
                vignette_gain=level_cfg["light_vignette_gain"],
                floor_gain=level_cfg["light_floor_gain"],
                center_x_2=None if light_center_x_2 is None else light_center_x_2[i],
                center_y_2=None if light_center_y_2 is None else light_center_y_2[i],
                sigma_2=level_cfg.get("light_sigma_2"),
                spot_gain_2=level_cfg.get("light_spot_gain_2", 0.0),
                shadow_blob_center_x=None if shadow_blob_center_x is None else shadow_blob_center_x[i],
                shadow_blob_center_y=None if shadow_blob_center_y is None else shadow_blob_center_y[i],
                shadow_blob_sigma_x=level_cfg.get("shadow_blob_sigma_x"),
                shadow_blob_sigma_y=level_cfg.get("shadow_blob_sigma_y"),
                shadow_blob_angle_deg=level_cfg.get("shadow_blob_angle_deg", 0.0),
                shadow_blob_gain=level_cfg.get("shadow_blob_gain", 0.0),
                shadow_blob_center_x_2=None if shadow_blob_center_x_2 is None else shadow_blob_center_x_2[i],
                shadow_blob_center_y_2=None if shadow_blob_center_y_2 is None else shadow_blob_center_y_2[i],
                shadow_blob_sigma_x_2=level_cfg.get("shadow_blob_sigma_x2"),
                shadow_blob_sigma_y_2=level_cfg.get("shadow_blob_sigma_y2"),
                shadow_blob_angle_deg_2=level_cfg.get("shadow_blob_angle_deg2", 0.0),
                shadow_blob_gain_2=level_cfg.get("shadow_blob_gain2", 0.0),
            )
            drifted_frame = apply_brightness_lighting(clean_frame, gains[i], lighting_mask)
            writer.append_data(drifted_frame)
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

    render_bw_with_brightness_drift(
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
        brightness_level=args.brightness_level,
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
        description="B/W pendulum with global brightness drift for binarize=False experiments"
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
    ap.add_argument("--brightness_level", type=int, default=1, help="brightness drift level in [1, 7]")
    ap.add_argument("-o", "--out", type=str, default="generated_video/pendulum_10s.mp4", help="output MP4 path")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_folder = get_setting_save_folder(SETTING_CASE)
    os.makedirs(save_folder, exist_ok=True)

    shared_params = get_setting_level_params(SETTING_CASE, None)
    levels = get_setting_levels(SETTING_CASE)

    for args.theta0_deg in [60]:
        for args.dtheta0_deg_s in [0]:
            for args.duration in [2.5]:
                for args.fps in [20]:
                    duration_pi = args.duration
                    for level in levels:
                        args.gamma0 = shared_params["gamma0"]
                        args.gamma1 = shared_params["gamma1"]
                        args.brightness_level = level
                        args.out = (
                            f"{save_folder}/{duration_pi:.1f}pi_{int(args.fps)}fps_"
                            f"degree{args.theta0_deg}_degs{args.dtheta0_deg_s}_level{level}.mp4"
                        )
                        args.duration = duration_pi * np.pi
                        main(args)
                        args.duration = duration_pi
