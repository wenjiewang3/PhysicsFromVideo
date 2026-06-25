import argparse
import copy
import os

import imageio
import numpy as np

import brightness3d_gen as base
from noise_background_gen import theta_series, theta_series_with_derivs


SETTING_CASE = "brightness3d_lightdom_underdamped"


LIGHTDOM_SCENE_DEFAULTS = copy.deepcopy(base.SCENE_DEFAULTS)
LIGHTDOM_SCENE_DEFAULTS.update(
    {
        "floor_tint": np.array([188.0, 182.0, 174.0], dtype=np.float32),
        "wall_tint": np.array([182.0, 177.0, 170.0], dtype=np.float32),
        "bob_color": np.array([18.0, 18.0, 18.0], dtype=np.float32),
        "rod_color": np.array([44.0, 44.0, 44.0], dtype=np.float32),
    }
)


def build_lightdom_levels():
    levels = copy.deepcopy(base.BRIGHTNESS3D_LEVELS)
    ambient_map = {
        1: 0.24,
        2: 0.20,
        3: 0.22,
        4: 0.18,
        5: 0.16,
        6: 0.13,
        7: 0.14,
        8: 0.11,
    }

    for level, cfg in levels.items():
        cfg["ambient_gain"] = ambient_map[level]
        cfg["shadow_strength"] = float(cfg.get("shadow_strength", 0.16)) * 1.12

        for light_idx, light in enumerate(cfg["lights"]):
            light["strength"] = float(light["strength"]) * (1.36 if light_idx == 0 else 1.18)
            light["radius_px"] = float(light["radius_px"]) * 1.05

            if light.get("pulse_profile") == "smooth_random":
                light["pulse_min_scale"] = max(0.30, float(light.get("pulse_min_scale", 0.6)) - 0.10)
                light["pulse_max_scale"] = float(light.get("pulse_max_scale", 1.4)) + 0.22
            elif "pulse_amp" in light:
                light["pulse_amp"] = float(light.get("pulse_amp", 0.0)) * 1.35

            if light.get("path_profile") == "smooth_random":
                amp = np.asarray(light["path_jitter_amp"], dtype=np.float32) * np.array([1.10, 1.08, 1.05], dtype=np.float32)
                light["path_jitter_amp"] = amp.tolist()

        if level == 1:
            cfg["lights"][0]["pos_start"] = [0.48, -0.40, 2.24]
            cfg["lights"][0]["pos_end"] = [0.48, -0.40, 2.24]
            cfg["lights"][0]["strength"] *= 1.22
            cfg["lights"][0]["radius_px"] *= 1.08
        elif level == 2:
            cfg["lights"][0]["pos_start"] = [0.48, -0.40, 2.24]
            cfg["lights"][0]["pos_end"] = [0.48, -0.40, 2.24]
            cfg["lights"][0]["strength"] *= 1.18
            cfg["lights"][0]["radius_px"] *= 1.08
        elif level == 3:
            cfg["lights"][0]["pos_start"] = [0.62, -0.46, 2.26]
            cfg["lights"][0]["pos_end"] = [0.22, -0.18, 2.12]
            cfg["lights"][0]["strength"] *= 1.16
            cfg["lights"][0]["radius_px"] *= 1.06
        elif level == 4:
            cfg["lights"][0]["pos_start"] = [0.60, -0.44, 2.26]
            cfg["lights"][0]["pos_end"] = [0.20, -0.16, 2.12]
            cfg["lights"][0]["strength"] *= 1.12
            cfg["lights"][0]["radius_px"] *= 1.06

    return levels


LIGHTDOM_LEVELS = build_lightdom_levels()


def render_background_lightdom(size, ambient_gain):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x = xx / max(float(size - 1), 1.0)
    y = yy / max(float(size - 1), 1.0)

    horizon_mix = np.clip((0.62 - y) / 0.62, 0.0, 1.0)[..., None]
    floor = LIGHTDOM_SCENE_DEFAULTS["floor_tint"][None, None, :]
    wall = LIGHTDOM_SCENE_DEFAULTS["wall_tint"][None, None, :]
    base_img = floor * (1.0 - horizon_mix) + wall * horizon_mix

    diagonal = 0.58 * x + 0.42 * y
    vignette = np.sqrt((x - 0.5) ** 2 + (y - 0.45) ** 2)
    vignette = vignette / max(float(vignette.max()), 1e-6)
    texture = 0.7 * np.sin(6.4 * x + 2.0 * y) + 0.5 * np.sin(4.8 * y + 0.7 * x)
    texture = texture[..., None]

    brightness_scale = 0.18 + 0.08 * float(ambient_gain)
    img = base_img * brightness_scale
    img *= (0.968 - 0.120 * diagonal[..., None])
    img *= (1.0 - 0.128 * vignette[..., None])
    img += texture
    return np.clip(img, 0.0, 255.0).astype(np.float32)


def draw_light_pool_lightdom(image, light, size, scene):
    floor_point = np.array([light["pos"][0], max(light["pos"][1], 0.0), 0.0], dtype=np.float32)
    center = base.world_to_image(floor_point[None, :], size, scene)[0, :2]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dist2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2

    sigma_halo = max(0.36 * float(light["radius_px"]), 1.0)
    sigma_core = max(0.11 * float(light["radius_px"]), 1.0)
    halo = np.exp(-dist2 / (2.0 * sigma_halo * sigma_halo)) ** 1.20
    core = np.exp(-dist2 / (2.0 * sigma_core * sigma_core)) ** 1.95

    image = base.blend_additive(
        image,
        halo,
        color=np.array([34.0, 34.0, 32.0], dtype=np.float32),
        amount=1.20 * light["strength"],
    )
    image = base.blend_additive(
        image,
        core,
        color=np.array([176.0, 170.0, 154.0], dtype=np.float32),
        amount=3.10 * light["strength"],
    )
    return image


def render_shadow_from_light_lightdom(image, pivot, bob, light, size, scene, shadow_strength):
    pivot_floor = base.line_light_intersection_on_floor(pivot, light["pos"])
    bob_floor = base.line_light_intersection_on_floor(bob, light["pos"])

    pivot_img = base.world_to_image(pivot_floor[None, :], size, scene)[0, :2]
    bob_img = base.world_to_image(bob_floor[None, :], size, scene)[0, :2]

    ground_bob = np.array([bob[0], bob[1], 0.0], dtype=np.float32)
    ground_img = base.world_to_image(ground_bob[None, :], size, scene)[0, :2]
    shadow_dir = bob_img - ground_img
    shadow_len = float(np.linalg.norm(shadow_dir))
    bob_radius_px = base.project_size_at_point(bob, scene["bob_radius"], size, scene)

    image = base.draw_segment(
        image,
        pivot_img,
        bob_img,
        width_px=max(5.8, bob_radius_px * 1.38),
        color=np.array([8.0, 8.0, 8.0], dtype=np.float32),
        alpha=min(0.16 + 0.12 * light["strength"], 0.30),
        softness=1.24,
    )

    angle_deg = np.rad2deg(np.arctan2(shadow_dir[1], shadow_dir[0] + 1e-6))
    sigma_major = bob_radius_px * (1.42 + 0.20 * shadow_len)
    sigma_minor = bob_radius_px * 0.82
    return base.draw_soft_ellipse_shadow(
        image,
        center_xy=bob_img,
        sigma_major=sigma_major,
        sigma_minor=sigma_minor,
        angle_deg=angle_deg,
        strength=1.85 * shadow_strength * (1.00 + 0.40 * light["strength"]),
    )


def apply_lightdom_rendering():
    base.SCENE_DEFAULTS = LIGHTDOM_SCENE_DEFAULTS
    base.render_background = render_background_lightdom
    base.draw_light_pool = draw_light_pool_lightdom
    base.render_shadow_from_light = render_shadow_from_light_lightdom


def render_brightness3d_lightdom_video(T, TH, out, fps=20, size=64, level=1):
    if level not in LIGHTDOM_LEVELS:
        raise ValueError(f"Unsupported level: {level}")

    apply_lightdom_rendering()
    scene = dict(LIGHTDOM_SCENE_DEFAULTS)
    level_cfg = LIGHTDOM_LEVELS[level]

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8)
    try:
        for frame_idx, theta in enumerate(TH):
            frame = base.render_frame_3d(theta, frame_idx, len(TH), size, scene, level_cfg)
            writer.append_data(frame)
    finally:
        writer.close()

    print(f"[OK] saved -> {out}")


def parse_args():
    ap = argparse.ArgumentParser(description="Pseudo-3D pendulum with light-dominant illumination")
    ap.add_argument("--gamma0", type=float, default=4.0016, help="gamma0 (stiffness-like)")
    ap.add_argument("--gamma1", type=float, default=0.08, help="gamma1 (damping)")
    ap.add_argument("--theta0_deg", type=float, default=60.0, help="initial angle in degrees")
    ap.add_argument("--dtheta0_deg_s", type=float, default=0.0, help="initial angular velocity in degrees per second")
    ap.add_argument("--duration", type=float, default=2.5, help="duration in units of pi before conversion in batch mode")
    ap.add_argument("--fps", type=int, default=20, help="frames per second")
    ap.add_argument("--size", type=int, default=64, help="square resolution (px)")
    ap.add_argument("--level", type=int, default=1, help="brightness3d light-dominant level in [1, 8]")
    ap.add_argument("-o", "--out", type=str, default="generated_video/pendulum_3d_lightdom.mp4", help="output MP4 path")
    return ap.parse_args()


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

    render_brightness3d_lightdom_video(
        T=T,
        TH=TH,
        out=args.out,
        fps=args.fps,
        size=args.size,
        level=args.level,
    )

    theta_series_with_derivs(
        args.gamma0,
        args.gamma1,
        duration=args.duration,
        fps=args.fps,
        theta0=theta0,
        dtheta0=dtheta0,
    )


if __name__ == "__main__":
    args = parse_args()
    save_folder = f"ablation_data/{SETTING_CASE}"
    os.makedirs(save_folder, exist_ok=True)

    for args.theta0_deg in [60]:
        for args.dtheta0_deg_s in [0]:
            for args.duration in [2.5]:
                for args.fps in [20]:
                    duration_pi = args.duration
                    for level in sorted(LIGHTDOM_LEVELS.keys()):
                        args.level = level
                        args.out = (
                            f"{save_folder}/{duration_pi:.1f}pi_{int(args.fps)}fps_"
                            f"degree{args.theta0_deg}_degs{args.dtheta0_deg_s}_level{level}.mp4"
                        )
                        args.duration = duration_pi * np.pi
                        main(args)
                        args.duration = duration_pi
