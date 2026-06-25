import argparse
import os

import imageio
import numpy as np

from noise_background_gen import theta_series, theta_series_with_derivs


SETTING_CASE = "brightness3d_underdamped"

SCENE_DEFAULTS = {
    "camera_pos": np.array([0.0, -3.10, 1.78], dtype=np.float32),
    "camera_target": np.array([0.0, 0.12, 0.86], dtype=np.float32),
    "camera_up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "fov_deg": 32.0,
    "pivot": np.array([0.0, 0.0, 1.42], dtype=np.float32),
    "rod_length": 0.98,
    "bob_radius": 0.19,
    "floor_y_min": -0.15,
    "floor_y_max": 2.50,
    "floor_x_half": 1.55,
    "ambient_gain": 0.62,
    "floor_tint": np.array([198.0, 192.0, 184.0], dtype=np.float32),
    "wall_tint": np.array([192.0, 187.0, 180.0], dtype=np.float32),
    "bob_color": np.array([18.0, 18.0, 18.0], dtype=np.float32),
    "rod_color": np.array([44.0, 44.0, 44.0], dtype=np.float32),
}


BRIGHTNESS3D_LEVELS = {
    1: {
        "description": "single static light",
        "ambient_gain": 0.26,
        "lights": [
            {
                "pos_start": [0.92, -0.66, 2.34],
                "pos_end": [0.92, -0.66, 2.34],
                "radius_px": 32.0,
                "strength": 0.62,
                "pulse_amp": 0.00,
                "pulse_cycles": 0.0,
                "phase_deg": 0.0,
            }
        ],
        "shadow_strength": 0.14,
    },
    2: {
        "description": "single light with intensity variation",
        "ambient_gain": 0.22,
        "lights": [
            {
                "pos_start": [0.92, -0.66, 2.34],
                "pos_end": [0.92, -0.66, 2.34],
                "radius_px": 32.0,
                "strength": 0.64,
                "pulse_profile": "smooth_random",
                "pulse_min_scale": 0.52,
                "pulse_max_scale": 1.48,
                "pulse_knots": 8,
                "pulse_seed": 1202,
            }
        ],
        "shadow_strength": 0.15,
    },
    3: {
        "description": "single moving light with fixed intensity",
        "ambient_gain": 0.24,
        "lights": [
            {
                "pos_start": [0.96, -0.70, 2.36],
                "pos_end": [0.56, -0.42, 2.20],
                "radius_px": 32.0,
                "strength": 0.62,
                "path_profile": "smooth_random",
                "path_jitter_amp": [0.10, 0.08, 0.05],
                "path_knots": 8,
                "path_seed": 1301,
                "pulse_amp": 0.00,
                "pulse_cycles": 0.0,
            },
        ],
        "shadow_strength": 0.15,
    },
    4: {
        "description": "single moving light with intensity variation",
        "ambient_gain": 0.20,
        "lights": [
            {
                "pos_start": [0.94, -0.68, 2.36],
                "pos_end": [0.62, -0.44, 2.24],
                "radius_px": 32.0,
                "strength": 0.64,
                "path_profile": "smooth_random",
                "path_jitter_amp": [0.12, 0.09, 0.06],
                "path_knots": 9,
                "path_seed": 1401,
                "pulse_profile": "smooth_random",
                "pulse_min_scale": 0.48,
                "pulse_max_scale": 1.58,
                "pulse_knots": 9,
                "pulse_seed": 1402,
            },
        ],
        "shadow_strength": 0.18,
    },
    5: {
        "description": "two static lights",
        "ambient_gain": 0.26,
        "lights": [
            {
                "pos_start": [0.96, -0.70, 2.36],
                "pos_end": [0.96, -0.70, 2.36],
                "radius_px": 30.0,
                "strength": 0.56,
                "pulse_amp": 0.00,
                "pulse_cycles": 0.0,
            },
            {
                "pos_start": [-0.86, -0.42, 2.16],
                "pos_end": [-0.86, -0.42, 2.16],
                "radius_px": 24.0,
                "strength": 0.30,
                "pulse_amp": 0.00,
                "pulse_cycles": 0.0,
            },
        ],
        "shadow_strength": 0.16,
    },
    6: {
        "description": "two static lights with intensity variation",
        "ambient_gain": 0.22,
        "lights": [
            {
                "pos_start": [0.96, -0.70, 2.36],
                "pos_end": [0.96, -0.70, 2.36],
                "radius_px": 30.0,
                "strength": 0.56,
                "pulse_profile": "smooth_random",
                "pulse_min_scale": 0.54,
                "pulse_max_scale": 1.40,
                "pulse_knots": 8,
                "pulse_seed": 1601,
            },
            {
                "pos_start": [-0.86, -0.42, 2.16],
                "pos_end": [-0.86, -0.42, 2.16],
                "radius_px": 24.0,
                "strength": 0.30,
                "pulse_profile": "smooth_random",
                "pulse_min_scale": 0.58,
                "pulse_max_scale": 1.34,
                "pulse_knots": 7,
                "pulse_seed": 1602,
            },
        ],
        "shadow_strength": 0.17,
    },
    7: {
        "description": "two moving lights with fixed intensity",
        "ambient_gain": 0.24,
        "lights": [
            {
                "pos_start": [0.96, -0.70, 2.36],
                "pos_end": [0.58, -0.42, 2.20],
                "radius_px": 30.0,
                "strength": 0.56,
                "path_profile": "smooth_random",
                "path_jitter_amp": [0.10, 0.08, 0.05],
                "path_knots": 8,
                "path_seed": 1701,
                "pulse_amp": 0.00,
                "pulse_cycles": 0.0,
            },
            {
                "pos_start": [-0.86, -0.42, 2.16],
                "pos_end": [-0.42, -0.58, 2.22],
                "radius_px": 24.0,
                "strength": 0.30,
                "path_profile": "smooth_random",
                "path_jitter_amp": [0.08, 0.06, 0.04],
                "path_knots": 7,
                "path_seed": 1702,
                "pulse_amp": 0.00,
                "pulse_cycles": 0.0,
            },
        ],
        "shadow_strength": 0.17,
    },
    8: {
        "description": "two moving lights with intensity variation",
        "ambient_gain": 0.20,
        "lights": [
            {
                "pos_start": [0.96, -0.70, 2.36],
                "pos_end": [0.58, -0.42, 2.20],
                "radius_px": 30.0,
                "strength": 0.58,
                "path_profile": "smooth_random",
                "path_jitter_amp": [0.12, 0.09, 0.06],
                "path_knots": 9,
                "path_seed": 1801,
                "pulse_profile": "smooth_random",
                "pulse_min_scale": 0.50,
                "pulse_max_scale": 1.50,
                "pulse_knots": 9,
                "pulse_seed": 1802,
            },
            {
                "pos_start": [-0.86, -0.42, 2.16],
                "pos_end": [-0.42, -0.58, 2.22],
                "radius_px": 24.0,
                "strength": 0.32,
                "path_profile": "smooth_random",
                "path_jitter_amp": [0.10, 0.07, 0.04],
                "path_knots": 8,
                "path_seed": 1811,
                "pulse_profile": "smooth_random",
                "pulse_min_scale": 0.56,
                "pulse_max_scale": 1.42,
                "pulse_knots": 8,
                "pulse_seed": 1812,
            },
        ],
        "shadow_strength": 0.19,
    },
}


def normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return vec / norm


def look_at_basis(camera_pos, target, up):
    forward = normalize(target - camera_pos)
    right = normalize(np.cross(forward, up))
    cam_up = normalize(np.cross(right, forward))
    return right, cam_up, forward


def project_points(points, camera_pos, target, up, fov_deg, aspect):
    points = np.asarray(points, dtype=np.float32)
    right, cam_up, forward = look_at_basis(camera_pos, target, up)
    rel = points - camera_pos[None, :]

    x_cam = rel @ right
    y_cam = rel @ cam_up
    z_cam = rel @ forward
    z_cam = np.maximum(z_cam, 1e-4)

    tan_half = np.tan(np.deg2rad(float(fov_deg)) * 0.5)
    x_ndc = x_cam / (z_cam * tan_half * float(aspect))
    y_ndc = y_cam / (z_cam * tan_half)

    u = 0.5 + 0.5 * x_ndc
    v = 0.5 + 0.5 * y_ndc
    return np.stack([u, v, z_cam], axis=-1)


def world_to_image(points, size, scene):
    pts = project_points(
        points=np.asarray(points, dtype=np.float32),
        camera_pos=scene["camera_pos"],
        target=scene["camera_target"],
        up=scene["camera_up"],
        fov_deg=scene["fov_deg"],
        aspect=1.0,
    )
    x = pts[..., 0] * float(size - 1)
    y = (1.0 - pts[..., 1]) * float(size - 1)
    return np.stack([x, y, pts[..., 2]], axis=-1)


def project_size_at_point(point, size_world, size, scene):
    p = np.asarray(point, dtype=np.float32)
    ref = world_to_image(np.stack([p, p + np.array([size_world, 0.0, 0.0], dtype=np.float32)]), size, scene)
    return max(float(np.linalg.norm(ref[1, :2] - ref[0, :2])), 1.0)


def line_light_intersection_on_floor(point, light_pos):
    p = np.asarray(point, dtype=np.float32)
    l = np.asarray(light_pos, dtype=np.float32)
    denom = p[2] - l[2]
    if abs(float(denom)) < 1e-6:
        return np.array([p[0], p[1], 0.0], dtype=np.float32)
    t = -l[2] / denom
    intersect = l + t * (p - l)
    intersect[2] = 0.0
    return intersect.astype(np.float32)


def blend_additive(image, mask, color, amount):
    image += float(amount) * mask[..., None] * np.asarray(color, dtype=np.float32)[None, None, :]
    return image


def blend_multiply_dark(image, mask, strength):
    image *= 1.0 - float(strength) * mask[..., None]
    return image


def draw_disc(image, center_xy, radius_px, color, alpha=1.0, softness=1.35):
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r = max(float(radius_px), 1.0)
    pad = int(np.ceil(r * softness + 2.0))
    x0 = max(0, int(np.floor(cx - pad)))
    x1 = min(image.shape[1], int(np.ceil(cx + pad + 1)))
    y0 = max(0, int(np.floor(cy - pad)))
    y1 = min(image.shape[0], int(np.ceil(cy + pad + 1)))
    if x1 <= x0 or y1 <= y0:
        return image

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.clip(1.0 - dist / (r * softness), 0.0, 1.0) ** 1.8
    a = float(alpha) * mask[..., None]
    region = image[y0:y1, x0:x1]
    image[y0:y1, x0:x1] = region * (1.0 - a) + np.asarray(color, dtype=np.float32)[None, None, :] * a
    return image


def draw_segment(image, p0, p1, width_px, color, alpha=1.0, softness=1.15):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    width = max(float(width_px), 1.0)
    pad = int(np.ceil(width * softness + 2.0))
    xmin = max(0, int(np.floor(min(x0, x1) - pad)))
    xmax = min(image.shape[1], int(np.ceil(max(x0, x1) + pad + 1)))
    ymin = max(0, int(np.floor(min(y0, y1) - pad)))
    ymax = min(image.shape[0], int(np.ceil(max(y0, y1) + pad + 1)))
    if xmax <= xmin or ymax <= ymin:
        return image

    yy, xx = np.mgrid[ymin:ymax, xmin:xmax].astype(np.float32)
    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy
    if denom < 1e-8:
        return draw_disc(image, p0, width * 0.5, color, alpha=alpha, softness=softness)
    t = ((xx - x0) * dx + (yy - y0) * dy) / denom
    t = np.clip(t, 0.0, 1.0)
    proj_x = x0 + t * dx
    proj_y = y0 + t * dy
    dist = np.sqrt((xx - proj_x) ** 2 + (yy - proj_y) ** 2)
    mask = np.clip(1.0 - dist / (0.5 * width * softness), 0.0, 1.0) ** 1.7
    a = float(alpha) * mask[..., None]
    region = image[ymin:ymax, xmin:xmax]
    image[ymin:ymax, xmin:xmax] = region * (1.0 - a) + np.asarray(color, dtype=np.float32)[None, None, :] * a
    return image


def draw_soft_ellipse_shadow(image, center_xy, sigma_major, sigma_minor, angle_deg, strength):
    cx, cy = float(center_xy[0]), float(center_xy[1])
    s_major = max(float(sigma_major), 1.0)
    s_minor = max(float(sigma_minor), 1.0)
    pad = int(np.ceil(max(s_major, s_minor) * 3.0 + 2.0))
    x0 = max(0, int(np.floor(cx - pad)))
    x1 = min(image.shape[1], int(np.ceil(cx + pad + 1)))
    y0 = max(0, int(np.floor(cy - pad)))
    y1 = min(image.shape[0], int(np.ceil(cy + pad + 1)))
    if x1 <= x0 or y1 <= y0:
        return image

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    ang = np.deg2rad(float(angle_deg))
    xr = np.cos(ang) * dx + np.sin(ang) * dy
    yr = -np.sin(ang) * dx + np.cos(ang) * dy
    mask = np.exp(-0.5 * ((xr / s_major) ** 2 + (yr / s_minor) ** 2))
    return blend_multiply_dark(image, pad_mask_to_image(mask, x0, x1, y0, y1, image.shape[:2]), strength)


def pad_mask_to_image(mask, x0, x1, y0, y1, image_shape):
    full = np.zeros(image_shape, dtype=np.float32)
    full[y0:y1, x0:x1] = mask.astype(np.float32)
    return full


def render_background(size, ambient_gain):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x = xx / max(float(size - 1), 1.0)
    y = yy / max(float(size - 1), 1.0)

    horizon_mix = np.clip((0.62 - y) / 0.62, 0.0, 1.0)[..., None]
    floor = SCENE_DEFAULTS["floor_tint"][None, None, :]
    wall = SCENE_DEFAULTS["wall_tint"][None, None, :]
    base = floor * (1.0 - horizon_mix) + wall * horizon_mix

    diagonal = 0.60 * x + 0.40 * y
    vignette = np.sqrt((x - 0.5) ** 2 + (y - 0.45) ** 2)
    vignette = vignette / max(float(vignette.max()), 1e-6)
    texture = 2.0 * np.sin(7.0 * x + 2.6 * y) + 1.4 * np.sin(5.2 * y + 0.8 * x)
    texture = texture[..., None]

    brightness_scale = 0.28 + 0.12 * float(ambient_gain)
    img = base * brightness_scale
    img *= (0.972 - 0.104 * diagonal[..., None])
    img *= (1.0 - 0.108 * vignette[..., None])
    img += texture
    return np.clip(img, 0.0, 255.0).astype(np.float32)


def evaluate_series(frame_idx, num_frames, start, end, wave_amp=None, wave_cycles=0.0, phase_deg=0.0):
    alpha = 0.0 if num_frames <= 1 else float(frame_idx) / float(num_frames - 1)
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    value = (1.0 - alpha) * start + alpha * end
    if wave_amp is not None and abs(float(wave_cycles)) > 1e-8:
        phase = np.deg2rad(float(phase_deg))
        wave = np.sin(2.0 * np.pi * float(wave_cycles) * alpha + phase)
        value = value + np.asarray(wave_amp, dtype=np.float32) * float(wave)
    return value.astype(np.float32)


def smooth_interp(values, alpha):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 1:
        return values[0]
    x = float(np.clip(alpha, 0.0, 1.0)) * float(len(values) - 1)
    i0 = int(np.floor(x))
    i1 = min(i0 + 1, len(values) - 1)
    frac = x - float(i0)
    frac = frac * frac * (3.0 - 2.0 * frac)
    return (1.0 - frac) * values[i0] + frac * values[i1]


def build_smooth_random_values(seed, num_knots, dims=None, low=0.0, high=1.0):
    rng = np.random.default_rng(int(seed))
    shape = (int(num_knots),) if dims is None else (int(num_knots), int(dims))
    return rng.uniform(float(low), float(high), size=shape).astype(np.float32)


def evaluate_smooth_random_scalar(alpha, seed, num_knots, low, high):
    values = build_smooth_random_values(seed=seed, num_knots=max(int(num_knots), 2), low=low, high=high)
    return float(smooth_interp(values, alpha))


def evaluate_smooth_random_vector(alpha, seed, num_knots, amp):
    amp = np.asarray(amp, dtype=np.float32)
    values = build_smooth_random_values(seed=seed, num_knots=max(int(num_knots), 2), dims=len(amp), low=-1.0, high=1.0)
    return smooth_interp(values, alpha) * amp


def evaluate_light(light_cfg, frame_idx, num_frames):
    alpha = 0.0 if num_frames <= 1 else float(frame_idx) / float(num_frames - 1)
    pos = evaluate_series(
        frame_idx,
        num_frames,
        light_cfg["pos_start"],
        light_cfg["pos_end"],
        light_cfg.get("wave_amp"),
        light_cfg.get("wave_cycles", 0.0),
        light_cfg.get("phase_deg", 0.0),
    )
    if light_cfg.get("path_profile") == "smooth_random":
        pos = pos + evaluate_smooth_random_vector(
            alpha=alpha,
            seed=light_cfg["path_seed"],
            num_knots=light_cfg.get("path_knots", 6),
            amp=light_cfg["path_jitter_amp"],
        )
    if light_cfg.get("pulse_profile") == "smooth_random":
        pulse = evaluate_smooth_random_scalar(
            alpha=alpha,
            seed=light_cfg["pulse_seed"],
            num_knots=light_cfg.get("pulse_knots", 8),
            low=light_cfg.get("pulse_min_scale", 0.6),
            high=light_cfg.get("pulse_max_scale", 1.4),
        )
    else:
        pulse = 1.0 + float(light_cfg.get("pulse_amp", 0.0)) * np.sin(
            2.0 * np.pi * float(light_cfg.get("pulse_cycles", 0.0)) * alpha
            + np.deg2rad(float(light_cfg.get("phase_deg_intensity", 0.0)))
        )
    return {
        "pos": pos,
        "radius_px": float(light_cfg["radius_px"]),
        "strength": max(float(light_cfg["strength"]) * float(pulse), 0.0),
    }


def evaluate_occluder(occ_cfg, frame_idx, num_frames):
    alpha = 0.0 if num_frames <= 1 else float(frame_idx) / float(num_frames - 1)
    center = evaluate_series(
        frame_idx,
        num_frames,
        occ_cfg["center_start"],
        occ_cfg["center_end"],
        occ_cfg.get("wave_amp"),
        occ_cfg.get("wave_cycles", 0.0),
        occ_cfg.get("phase_deg", 0.0),
    )
    if occ_cfg.get("path_profile") == "smooth_random":
        center = center + evaluate_smooth_random_vector(
            alpha=alpha,
            seed=occ_cfg["path_seed"],
            num_knots=occ_cfg.get("path_knots", 6),
            amp=occ_cfg["path_jitter_amp"],
        )
    return {
        "center": center,
        "sigma": np.asarray(occ_cfg["sigma"], dtype=np.float32),
        "angle_deg": float(occ_cfg["angle_deg"]),
        "strength": float(occ_cfg["strength"]),
    }


def draw_occluder_shadow(image, occluder, size, scene):
    center_world = np.array([occluder["center"][0], occluder["center"][1], 0.0], dtype=np.float32)
    center_img = world_to_image(center_world[None, :], size, scene)[0, :2]
    sigma_major = project_size_at_point(center_world, float(occluder["sigma"][1]), size, scene) * 1.6
    sigma_minor = project_size_at_point(center_world, float(occluder["sigma"][0]), size, scene) * 1.1
    return draw_soft_ellipse_shadow(
        image,
        center_xy=center_img,
        sigma_major=sigma_major,
        sigma_minor=sigma_minor,
        angle_deg=occluder["angle_deg"],
        strength=occluder["strength"],
    )


def draw_light_pool(image, light, size, scene):
    floor_point = np.array([light["pos"][0], max(light["pos"][1], 0.0), 0.0], dtype=np.float32)
    center = world_to_image(floor_point[None, :], size, scene)[0, :2]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dist2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    sigma_halo = max(0.42 * float(light["radius_px"]), 1.0)
    sigma_core = max(0.16 * float(light["radius_px"]), 1.0)
    halo = np.exp(-dist2 / (2.0 * sigma_halo * sigma_halo)) ** 1.15
    core = np.exp(-dist2 / (2.0 * sigma_core * sigma_core)) ** 1.85
    image = blend_additive(
        image,
        halo,
        color=np.array([42.0, 40.0, 36.0], dtype=np.float32),
        amount=0.95 * light["strength"],
    )
    image = blend_additive(
        image,
        core,
        color=np.array([118.0, 110.0, 92.0], dtype=np.float32),
        amount=2.15 * light["strength"],
    )
    return image


def render_shadow_from_light(image, pivot, bob, light, size, scene, shadow_strength):
    pivot_floor = line_light_intersection_on_floor(pivot, light["pos"])
    bob_floor = line_light_intersection_on_floor(bob, light["pos"])

    pivot_img = world_to_image(pivot_floor[None, :], size, scene)[0, :2]
    bob_img = world_to_image(bob_floor[None, :], size, scene)[0, :2]

    ground_bob = np.array([bob[0], bob[1], 0.0], dtype=np.float32)
    ground_img = world_to_image(ground_bob[None, :], size, scene)[0, :2]
    shadow_dir = bob_img - ground_img
    shadow_len = float(np.linalg.norm(shadow_dir))
    bob_radius_px = project_size_at_point(bob, scene["bob_radius"], size, scene)

    image = draw_segment(
        image,
        pivot_img,
        bob_img,
        width_px=max(5.6, bob_radius_px * 1.34),
        color=np.array([88.0, 86.0, 84.0], dtype=np.float32),
        alpha=min(0.14 + 0.10 * light["strength"], 0.24),
        softness=1.30,
    )

    angle_deg = np.rad2deg(np.arctan2(shadow_dir[1], shadow_dir[0] + 1e-6))
    sigma_major = bob_radius_px * (1.48 + 0.22 * shadow_len)
    sigma_minor = bob_radius_px * 0.86
    return draw_soft_ellipse_shadow(
        image,
        center_xy=bob_img,
        sigma_major=sigma_major,
        sigma_minor=sigma_minor,
        angle_deg=angle_deg,
        strength=1.62 * shadow_strength * (0.98 + 0.38 * light["strength"]),
    )


def draw_pendulum(image, pivot, bob, size, scene, lights):
    pivot_img = world_to_image(pivot[None, :], size, scene)[0]
    bob_img = world_to_image(bob[None, :], size, scene)[0]
    rod_width = max(4.6, project_size_at_point(bob, 0.070, size, scene))
    bob_radius_px = project_size_at_point(bob, scene["bob_radius"], size, scene)

    image = draw_segment(
        image,
        pivot_img[:2],
        bob_img[:2],
        width_px=rod_width,
        color=scene["rod_color"],
        alpha=1.0,
        softness=1.02,
    )

    bob_color = scene["bob_color"].copy()
    if lights:
        dominant = max(lights, key=lambda item: item["strength"])
        dist = max(float(np.linalg.norm(bob - dominant["pos"])), 0.35)
        highlight = 0.20 * dominant["strength"] / dist
        bob_color = np.clip(bob_color * (1.0 + highlight), 0.0, 255.0)

    image = draw_disc(
        image,
        bob_img[:2],
        bob_radius_px,
        color=bob_color,
        alpha=1.0,
        softness=0.66,
    )

    highlight_center = bob_img[:2] + np.array([-0.25 * bob_radius_px, -0.28 * bob_radius_px], dtype=np.float32)
    image = draw_disc(
        image,
        highlight_center,
        radius_px=max(2.8, 0.38 * bob_radius_px),
        color=np.array([120.0, 120.0, 120.0], dtype=np.float32),
        alpha=0.05,
        softness=0.84,
    )
    return image


def render_frame_3d(theta, frame_idx, num_frames, size, scene, level_cfg):
    img = render_background(size=size, ambient_gain=level_cfg["ambient_gain"])

    pivot = scene["pivot"].astype(np.float32)
    bob = np.array(
        [
            scene["rod_length"] * np.sin(theta),
            0.0,
            pivot[2] - scene["rod_length"] * np.cos(theta),
        ],
        dtype=np.float32,
    )

    lights = [evaluate_light(cfg, frame_idx, num_frames) for cfg in level_cfg["lights"]]
    occluders = [evaluate_occluder(cfg, frame_idx, num_frames) for cfg in level_cfg.get("occluders", [])]

    for light in lights:
        img = draw_light_pool(img, light, size, scene)

    for occluder in occluders:
        img = draw_occluder_shadow(img, occluder, size, scene)

    for light in lights:
        img = render_shadow_from_light(
            img,
            pivot=pivot,
            bob=bob,
            light=light,
            size=size,
            scene=scene,
            shadow_strength=level_cfg.get("shadow_strength", 0.28),
        )

    img = draw_pendulum(img, pivot, bob, size, scene, lights)
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def render_brightness3d_video(
    T,
    TH,
    out,
    fps=20,
    size=64,
    level=1,
):
    if level not in BRIGHTNESS3D_LEVELS:
        raise ValueError(f"Unsupported level: {level}")

    scene = dict(SCENE_DEFAULTS)
    level_cfg = BRIGHTNESS3D_LEVELS[level]

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8)
    try:
        for frame_idx, theta in enumerate(TH):
            frame = render_frame_3d(theta, frame_idx, len(TH), size, scene, level_cfg)
            writer.append_data(frame)
    finally:
        writer.close()

    print(f"[OK] saved -> {out}")


def parse_args():
    ap = argparse.ArgumentParser(description="Pseudo-3D pendulum with lights and floor shadows")
    ap.add_argument("--gamma0", type=float, default=4.0016, help="gamma0 (stiffness-like)")
    ap.add_argument("--gamma1", type=float, default=0.08, help="gamma1 (damping)")
    ap.add_argument("--theta0_deg", type=float, default=60.0, help="initial angle in degrees")
    ap.add_argument("--dtheta0_deg_s", type=float, default=0.0, help="initial angular velocity in degrees per second")
    ap.add_argument("--duration", type=float, default=2.5, help="duration in units of pi before conversion in batch mode")
    ap.add_argument("--fps", type=int, default=20, help="frames per second")
    ap.add_argument("--size", type=int, default=64, help="square resolution (px)")
    ap.add_argument("--level", type=int, default=1, help="brightness3d level in [1, 8]")
    ap.add_argument("-o", "--out", type=str, default="generated_video/pendulum_3d.mp4", help="output MP4 path")
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

    render_brightness3d_video(
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
                    for level in sorted(BRIGHTNESS3D_LEVELS.keys()):
                        args.level = level
                        args.out = (
                            f"{save_folder}/{duration_pi:.1f}pi_{int(args.fps)}fps_"
                            f"degree{args.theta0_deg}_degs{args.dtheta0_deg_s}_level{level}.mp4"
                        )
                        args.duration = duration_pi * np.pi
                        main(args)
                        args.duration = duration_pi
