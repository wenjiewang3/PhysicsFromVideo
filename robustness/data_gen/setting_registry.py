SETTING_CONFIGS = {
    "noise_background_underdamped": {
        "save_folder": "ablation_data/noise_background_underdamped",
        "shared_video_params": {
            "gamma0": 4.0016,
            "gamma1": 0.08,
            "noise_shape": "square",
            "noise_gray_min": 0,
            "noise_gray_max": 0,
            "noise_refresh_every": 0,
        },
        "levels": {
            1: {"noise_count": 4, "noise_min_size_px": 2, "noise_max_size_px": 6},
            2: {"noise_count": 8, "noise_min_size_px": 4, "noise_max_size_px": 10},
            3: {"noise_count": 12, "noise_min_size_px": 6, "noise_max_size_px": 14},
            4: {"noise_count": 20, "noise_min_size_px": 10, "noise_max_size_px": 22},
            5: {"noise_count": 32, "noise_min_size_px": 14, "noise_max_size_px": 30},
        },
    },
    "moving_clutter_underdamped": {
        "save_folder": "ablation_data/moving_clutter_underdamped",
        "shared_video_params": {
            "gamma0": 4.0016,
            "gamma1": 0.08,
            "noise_shape": "square",
            "noise_count": 4,
            "noise_min_size_px": 4,
            "noise_max_size_px": 10,
            "noise_gray_min": 0,
            "noise_gray_max": 0,
            "motion_profile": "global_sinusoid",
            "motion_guard_px": 2,
            "placement_mode": "quadrants",
            "allow_wall_collision": False,
            "allow_pendulum_overlap": False,
            "motion_speed_px": 0.0,
        },
        "levels": {
            1: {"motion_profile": "global_sinusoid", "motion_amp_x_px": 7, "motion_amp_y_px": 4, "motion_cycles_x": 2.50, "motion_cycles_y": 2.00, "motion_harmonic_ratio": 0.18, "allow_wall_collision": True, "allow_pendulum_overlap": True},
            2: {"motion_profile": "global_sinusoid", "motion_amp_x_px": 10, "motion_amp_y_px": 6, "motion_cycles_x": 3.50, "motion_cycles_y": 3.00, "motion_harmonic_ratio": 0.30, "allow_wall_collision": True, "allow_pendulum_overlap": True},
            3: {"motion_profile": "independent_bounce", "motion_speed_px": 1.6, "allow_wall_collision": True, "allow_pendulum_overlap": True},
            4: {"motion_profile": "independent_bounce", "motion_speed_px": 2.4, "allow_wall_collision": True, "allow_pendulum_overlap": True},
            5: {"motion_profile": "independent_bounce", "noise_count": 8, "motion_speed_px": 3.2, "allow_wall_collision": True, "allow_pendulum_overlap": True},
        },
    },
    "camera_jitter_underdamped": {
        "save_folder": "ablation_data/camera_jitter_underdamped",
        "shared_video_params": {
            "gamma0": 4.0016,
            "gamma1": 0.08,
            "jitter_fill": "white",
            "jitter_wrap": False,
        },
        "levels": {
            1: {"jitter_profile": "linear_sine", "jitter_amp_x_px": 1, "jitter_amp_y_px": 1, "jitter_cycles": 1.0, "jitter_direction_deg": 18.0},
            2: {"jitter_profile": "linear_sine", "jitter_amp_x_px": 2, "jitter_amp_y_px": 1, "jitter_cycles": 1.8, "jitter_direction_deg": 18.0},
            3: {"jitter_profile": "random_walk", "jitter_max_x_px": 1, "jitter_max_y_px": 1, "jitter_step_choices": [0, 1], "jitter_step_prob": 0.38, "jitter_momentum": 0.15},
            4: {"jitter_profile": "random_walk", "jitter_max_x_px": 2, "jitter_max_y_px": 2, "jitter_step_choices": [0, 1], "jitter_step_prob": 0.55, "jitter_momentum": 0.35},
            5: {"jitter_profile": "random_walk", "jitter_max_x_px": 3, "jitter_max_y_px": 2, "jitter_step_choices": [0, 1, 2], "jitter_step_prob": 0.72, "jitter_momentum": 0.45},
            6: {"jitter_profile": "staged_horizontal_sweep", "jitter_stage_targets_px": [14, 30], "jitter_sweep_frac": 0.10},
            7: {"jitter_profile": "vertical_shake", "jitter_amp_y_px": 8, "jitter_cycles": 7.0},
        },
    },
    "brightness_drift_underdamped": {
        "save_folder": "ablation_data/brightness_drift_underdamped",
        "shared_video_params": {
            "gamma0": 4.0016,
            "gamma1": 0.08,
            "brightness_profile": "constant",
            "light_center_x_start": 0.28,
            "light_center_y_start": 0.20,
        },
        "levels": {
            1: {"brightness_profile": "constant", "brightness_base_gain": 0.32, "light_center_x_end": 0.28, "light_center_y_end": 0.20, "light_sigma": 0.20, "light_spot_gain": 0.42, "light_shadow_gain": 0.48, "light_vignette_gain": 0.24, "light_floor_gain": 0.10},
            2: {"brightness_profile": "constant", "brightness_base_gain": 0.22, "light_center_x_start": 0.30, "light_center_y_start": 0.21, "light_center_x_end": 0.30, "light_center_y_end": 0.21, "light_sigma": 0.18, "light_spot_gain": 0.44, "light_shadow_gain": 0.56, "light_vignette_gain": 0.30, "light_floor_gain": 0.04},
            3: {"brightness_profile": "constant", "brightness_base_gain": 0.32, "light_center_x_start": 0.22, "light_center_y_start": 0.18, "light_center_x_end": 0.58, "light_center_y_end": 0.36, "light_wave_amp_x": 0.05, "light_wave_amp_y": 0.03, "light_wave_cycles": 1.2, "light_wave_phase_deg": 10.0, "light_sigma": 0.20, "light_spot_gain": 0.42, "light_shadow_gain": 0.48, "light_vignette_gain": 0.24, "light_floor_gain": 0.10},
            4: {"brightness_profile": "linear_wave", "brightness_start_gain": 0.28, "brightness_end_gain": 0.28, "brightness_wave_amp": 0.08, "brightness_wave_cycles": 1.6, "brightness_wave_phase_deg": 90.0, "brightness_min_gain": 0.14, "brightness_max_gain": 0.40, "light_center_x_start": 0.20, "light_center_y_start": 0.20, "light_center_x_end": 0.58, "light_center_y_end": 0.34, "light_wave_amp_x": 0.05, "light_wave_amp_y": 0.04, "light_wave_cycles": 1.6, "light_wave_phase_deg": 20.0, "light_sigma": 0.19, "light_spot_gain": 0.44, "light_shadow_gain": 0.44, "light_vignette_gain": 0.22, "light_floor_gain": 0.08, "light_center_x2_start": 0.82, "light_center_y2_start": 0.22, "light_center_x2_end": 0.46, "light_center_y2_end": 0.56, "light_wave_amp_x2": 0.04, "light_wave_amp_y2": 0.03, "light_wave_cycles_2": 1.9, "light_wave_phase_deg_2": 140.0, "light_sigma_2": 0.16, "light_spot_gain_2": 0.20},
            5: {"brightness_profile": "linear_wave", "brightness_start_gain": 0.24, "brightness_end_gain": 0.24, "brightness_wave_amp": 0.12, "brightness_wave_cycles": 3.2, "brightness_wave_phase_deg": 40.0, "brightness_min_gain": 0.10, "brightness_max_gain": 0.38, "light_center_x_start": 0.18, "light_center_y_start": 0.18, "light_center_x_end": 0.64, "light_center_y_end": 0.38, "light_wave_amp_x": 0.08, "light_wave_amp_y": 0.06, "light_wave_cycles": 2.8, "light_wave_phase_deg": 15.0, "light_sigma": 0.18, "light_spot_gain": 0.48, "light_shadow_gain": 0.50, "light_vignette_gain": 0.26, "light_floor_gain": 0.06, "light_center_x2_start": 0.84, "light_center_y2_start": 0.18, "light_center_x2_end": 0.40, "light_center_y2_end": 0.62, "light_wave_amp_x2": 0.07, "light_wave_amp_y2": 0.05, "light_wave_cycles_2": 3.3, "light_wave_phase_deg_2": 170.0, "light_sigma_2": 0.14, "light_spot_gain_2": 0.24, "shadow_blob_center_x_start": 0.60, "shadow_blob_center_y_start": 0.46, "shadow_blob_center_x_end": 0.38, "shadow_blob_center_y_end": 0.66, "shadow_blob_wave_amp_x": 0.03, "shadow_blob_wave_amp_y": 0.03, "shadow_blob_wave_cycles": 2.0, "shadow_blob_wave_phase_deg": 30.0, "shadow_blob_sigma_x": 0.11, "shadow_blob_sigma_y": 0.22, "shadow_blob_angle_deg": -26.0, "shadow_blob_gain": 0.32},
            6: {"brightness_profile": "linear_wave", "brightness_start_gain": 0.20, "brightness_end_gain": 0.20, "brightness_wave_amp": 0.15, "brightness_wave_cycles": 4.6, "brightness_wave_phase_deg": 55.0, "brightness_min_gain": 0.06, "brightness_max_gain": 0.36, "light_center_x_start": 0.16, "light_center_y_start": 0.16, "light_center_x_end": 0.68, "light_center_y_end": 0.42, "light_wave_amp_x": 0.10, "light_wave_amp_y": 0.07, "light_wave_cycles": 4.0, "light_wave_phase_deg": 25.0, "light_sigma": 0.16, "light_spot_gain": 0.52, "light_shadow_gain": 0.56, "light_vignette_gain": 0.30, "light_floor_gain": 0.04, "light_center_x2_start": 0.86, "light_center_y2_start": 0.14, "light_center_x2_end": 0.34, "light_center_y2_end": 0.66, "light_wave_amp_x2": 0.09, "light_wave_amp_y2": 0.07, "light_wave_cycles_2": 4.4, "light_wave_phase_deg_2": 185.0, "light_sigma_2": 0.13, "light_spot_gain_2": 0.28, "shadow_blob_center_x_start": 0.60, "shadow_blob_center_y_start": 0.44, "shadow_blob_center_x_end": 0.34, "shadow_blob_center_y_end": 0.70, "shadow_blob_wave_amp_x": 0.05, "shadow_blob_wave_amp_y": 0.05, "shadow_blob_wave_cycles": 3.1, "shadow_blob_wave_phase_deg": 40.0, "shadow_blob_sigma_x": 0.10, "shadow_blob_sigma_y": 0.24, "shadow_blob_angle_deg": -30.0, "shadow_blob_gain": 0.36, "shadow_blob_center_x2_start": 0.24, "shadow_blob_center_y2_start": 0.76, "shadow_blob_center_x2_end": 0.36, "shadow_blob_center_y2_end": 0.48, "shadow_blob_wave_amp_x2": 0.04, "shadow_blob_wave_amp_y2": 0.04, "shadow_blob_wave_cycles_2": 3.6, "shadow_blob_wave_phase_deg_2": 120.0, "shadow_blob_sigma_x2": 0.09, "shadow_blob_sigma_y2": 0.18, "shadow_blob_angle_deg2": 16.0, "shadow_blob_gain2": 0.22},
            7: {"brightness_profile": "linear_wave", "brightness_start_gain": 0.18, "brightness_end_gain": 0.18, "brightness_wave_amp": 0.18, "brightness_wave_cycles": 6.2, "brightness_wave_phase_deg": 75.0, "brightness_min_gain": 0.04, "brightness_max_gain": 0.34, "light_center_x_start": 0.14, "light_center_y_start": 0.14, "light_center_x_end": 0.72, "light_center_y_end": 0.46, "light_wave_amp_x": 0.12, "light_wave_amp_y": 0.08, "light_wave_cycles": 5.0, "light_wave_phase_deg": 35.0, "light_sigma": 0.15, "light_spot_gain": 0.56, "light_shadow_gain": 0.60, "light_vignette_gain": 0.34, "light_floor_gain": 0.03, "light_center_x2_start": 0.88, "light_center_y2_start": 0.12, "light_center_x2_end": 0.28, "light_center_y2_end": 0.70, "light_wave_amp_x2": 0.10, "light_wave_amp_y2": 0.08, "light_wave_cycles_2": 5.6, "light_wave_phase_deg_2": 210.0, "light_sigma_2": 0.12, "light_spot_gain_2": 0.32, "shadow_blob_center_x_start": 0.62, "shadow_blob_center_y_start": 0.42, "shadow_blob_center_x_end": 0.30, "shadow_blob_center_y_end": 0.72, "shadow_blob_wave_amp_x": 0.06, "shadow_blob_wave_amp_y": 0.06, "shadow_blob_wave_cycles": 4.2, "shadow_blob_wave_phase_deg": 50.0, "shadow_blob_sigma_x": 0.10, "shadow_blob_sigma_y": 0.26, "shadow_blob_angle_deg": -34.0, "shadow_blob_gain": 0.40, "shadow_blob_center_x2_start": 0.22, "shadow_blob_center_y2_start": 0.78, "shadow_blob_center_x2_end": 0.40, "shadow_blob_center_y2_end": 0.44, "shadow_blob_wave_amp_x2": 0.05, "shadow_blob_wave_amp_y2": 0.05, "shadow_blob_wave_cycles_2": 4.8, "shadow_blob_wave_phase_deg_2": 150.0, "shadow_blob_sigma_x2": 0.08, "shadow_blob_sigma_y2": 0.16, "shadow_blob_angle_deg2": 20.0, "shadow_blob_gain2": 0.24},
        },
    },
    "brightness3d_underdamped": {
        "save_folder": "ablation_data/brightness3d_underdamped",
        "shared_video_params": {
            "gamma0": 4.0016,
            "gamma1": 0.08,
            "render_mode": "pseudo3d_lighting",
            "shadow_model": "floor_projection",
        },
        "levels": {
            1: {"light_count": 1, "light_motion": "static", "light_intensity_mode": "static", "ambient_gain": 0.26, "shadow_strength": 0.14},
            2: {"light_count": 1, "light_motion": "static", "light_intensity_mode": "smooth_random", "ambient_gain": 0.22, "shadow_strength": 0.15},
            3: {"light_count": 1, "light_motion": "smooth_random", "light_intensity_mode": "static", "ambient_gain": 0.24, "shadow_strength": 0.15},
            4: {"light_count": 1, "light_motion": "smooth_random", "light_intensity_mode": "smooth_random", "ambient_gain": 0.20, "shadow_strength": 0.18},
            5: {"light_count": 2, "light_motion": "static", "light_intensity_mode": "static", "ambient_gain": 0.26, "shadow_strength": 0.16},
            6: {"light_count": 2, "light_motion": "static", "light_intensity_mode": "smooth_random", "ambient_gain": 0.22, "shadow_strength": 0.17},
            7: {"light_count": 2, "light_motion": "smooth_random", "light_intensity_mode": "static", "ambient_gain": 0.24, "shadow_strength": 0.17},
            8: {"light_count": 2, "light_motion": "smooth_random", "light_intensity_mode": "smooth_random", "ambient_gain": 0.20, "shadow_strength": 0.19},
        },
    },
    "brightness3d_lightdom_underdamped": {
        "save_folder": "ablation_data/brightness3d_lightdom_underdamped",
        "shared_video_params": {
            "gamma0": 4.0016,
            "gamma1": 0.08,
            "render_mode": "pseudo3d_light_dominant",
            "shadow_model": "floor_projection",
        },
        "levels": {
            1: {"light_count": 1, "light_motion": "static", "light_intensity_mode": "static", "ambient_gain": 0.24, "shadow_strength": 0.1568},
            2: {"light_count": 1, "light_motion": "static", "light_intensity_mode": "smooth_random", "ambient_gain": 0.20, "shadow_strength": 0.168},
            3: {"light_count": 1, "light_motion": "smooth_random", "light_intensity_mode": "static", "ambient_gain": 0.22, "shadow_strength": 0.168},
            4: {"light_count": 1, "light_motion": "smooth_random", "light_intensity_mode": "smooth_random", "ambient_gain": 0.18, "shadow_strength": 0.2016},
            5: {"light_count": 2, "light_motion": "static", "light_intensity_mode": "static", "ambient_gain": 0.16, "shadow_strength": 0.1792},
            6: {"light_count": 2, "light_motion": "static", "light_intensity_mode": "smooth_random", "ambient_gain": 0.13, "shadow_strength": 0.1904},
            7: {"light_count": 2, "light_motion": "smooth_random", "light_intensity_mode": "static", "ambient_gain": 0.14, "shadow_strength": 0.1904},
            8: {"light_count": 2, "light_motion": "smooth_random", "light_intensity_mode": "smooth_random", "ambient_gain": 0.11, "shadow_strength": 0.2128},
        },
    },
    "occlusion_underdamped": {
        "save_folder": "ablation_data/occlusion_underdamped",
        "shared_video_params": {
            "gamma0": 4.0016,
            "gamma1": 0.08,
            "occlusion_color": "black",
            "occlusion_shape": "sector_blocks",
            "occlusion_anchor": "left_to_right",
        },
        "levels": {
            1: {"occlusion_mode": "continuous", "occlusion_angle_frac": 0.22, "occlusion_r_outer_frac": 0.72, "occlusion_cell_frac": 0.10},
            2: {"occlusion_mode": "continuous", "occlusion_angle_frac": 0.38, "occlusion_r_outer_frac": 0.82, "occlusion_cell_frac": 0.11},
            3: {"occlusion_mode": "continuous", "occlusion_angle_frac": 0.56, "occlusion_r_outer_frac": 0.92, "occlusion_cell_frac": 0.12},
            4: {"occlusion_mode": "continuous", "occlusion_angle_frac": 0.78, "occlusion_r_outer_frac": 1.04, "occlusion_cell_frac": 0.13},
            5: {"occlusion_mode": "discrete", "occlusion_segments": 2, "occlusion_segment_frac": 0.13, "occlusion_r_outer_frac": 0.76, "occlusion_cell_frac": 0.10},
            6: {"occlusion_mode": "discrete", "occlusion_segments": 3, "occlusion_segment_frac": 0.15, "occlusion_r_outer_frac": 0.86, "occlusion_cell_frac": 0.11},
            7: {"occlusion_mode": "discrete", "occlusion_segments": 4, "occlusion_segment_frac": 0.17, "occlusion_r_outer_frac": 0.96, "occlusion_cell_frac": 0.12},
            8: {"occlusion_mode": "discrete", "occlusion_segments": 5, "occlusion_segment_frac": 0.20, "occlusion_r_outer_frac": 1.06, "occlusion_cell_frac": 0.13},
            9: {"occlusion_mode": "continuous", "occlusion_angle_frac": 0.16, "occlusion_r_outer_frac": 0.64, "occlusion_cell_frac": 0.09},
            10: {"occlusion_mode": "continuous", "occlusion_angle_frac": 0.10, "occlusion_r_outer_frac": 0.56, "occlusion_cell_frac": 0.08},
            11: {"occlusion_mode": "continuous", "occlusion_anchor": "center", "occlusion_angle_frac": 0.22, "occlusion_r_outer_frac": 0.72, "occlusion_cell_frac": 0.10},
            12: {"occlusion_mode": "continuous", "occlusion_anchor": "center", "occlusion_angle_frac": 0.34, "occlusion_r_outer_frac": 0.82, "occlusion_cell_frac": 0.10},
        },
    },
}


def get_setting_config(setting_case: str):
    if setting_case not in SETTING_CONFIGS:
        raise KeyError(f"Unknown setting_case: {setting_case}")
    return SETTING_CONFIGS[setting_case]


def get_setting_levels(setting_case: str):
    config = get_setting_config(setting_case)
    return sorted(config.get("levels", {}).keys())


def get_available_setting_cases():
    return sorted(SETTING_CONFIGS.keys())


def get_setting_level_params(setting_case: str, level: int | None = None):
    config = get_setting_config(setting_case)
    params = dict(config.get("shared_video_params", {}))

    if level is not None:
        levels = config.get("levels", {})
        if level not in levels:
            raise KeyError(f"Unknown level {level} for setting_case {setting_case}")
        params.update(levels[level])

    return params


def get_setting_save_folder(setting_case: str):
    config = get_setting_config(setting_case)
    return config["save_folder"]


def get_setting_display_lines(setting_case: str, level: int | None = None):
    params = get_setting_level_params(setting_case, level)
    lines = []

    if "noise_shape" in params:
        lines.append(f"noise_shape : {params['noise_shape']}")
    if "noise_count" in params:
        lines.append(f"noise_count : {params['noise_count']}")
    if "noise_min_size_px" in params and "noise_max_size_px" in params:
        lines.append(f"noise_size  : {params['noise_min_size_px']}-{params['noise_max_size_px']} px")
    if "noise_gray_min" in params and "noise_gray_max" in params:
        lines.append(f"noise_gray  : {params['noise_gray_min']}-{params['noise_gray_max']}")
    if "noise_refresh_every" in params:
        lines.append(f"noise_refresh: {params['noise_refresh_every']}")
    if "motion_profile" in params:
        lines.append(f"motion_type : {params['motion_profile']}")
    if "motion_amp_x_px" in params and "motion_amp_y_px" in params:
        lines.append(f"motion_amp  : ({params['motion_amp_x_px']}, {params['motion_amp_y_px']}) px")
    if "motion_cycles_x" in params and "motion_cycles_y" in params:
        lines.append(f"motion_cycle: ({params['motion_cycles_x']}, {params['motion_cycles_y']})")
    if "motion_harmonic_ratio" in params:
        lines.append(f"motion_harm : {params['motion_harmonic_ratio']}")
    if "motion_speed_px" in params and params["motion_speed_px"] > 0:
        lines.append(f"motion_speed: {params['motion_speed_px']} px/frame")
    if "motion_guard_px" in params:
        lines.append(f"motion_guard: {params['motion_guard_px']} px")
    if "placement_mode" in params:
        lines.append(f"placement   : {params['placement_mode']}")
    if "allow_wall_collision" in params:
        lines.append(f"wall_hit    : {params['allow_wall_collision']}")
    if "allow_pendulum_overlap" in params:
        lines.append(f"overlap_ok  : {params['allow_pendulum_overlap']}")
    if "jitter_profile" in params:
        lines.append(f"jitter_type : {params['jitter_profile']}")
    if "jitter_amp_x_px" in params and "jitter_amp_y_px" in params:
        lines.append(f"jitter_amp  : ({params['jitter_amp_x_px']}, {params['jitter_amp_y_px']}) px")
    if "jitter_cycles" in params:
        lines.append(f"jitter_cycle: {params['jitter_cycles']}")
    if "jitter_direction_deg" in params:
        lines.append(f"jitter_dir  : {params['jitter_direction_deg']} deg")
    if "jitter_max_x_px" in params and "jitter_max_y_px" in params:
        lines.append(f"jitter_max  : ({params['jitter_max_x_px']}, {params['jitter_max_y_px']}) px")
    if "jitter_stage_targets_px" in params:
        lines.append(f"jitter_stage: {params['jitter_stage_targets_px']}")
    if "jitter_sweep_frac" in params:
        lines.append(f"jitter_sweep: {params['jitter_sweep_frac']}")
    if "jitter_step_choices" in params:
        lines.append(f"jitter_step : {params['jitter_step_choices']}")
    if "jitter_step_prob" in params:
        lines.append(f"jitter_prob : {params['jitter_step_prob']}")
    if "jitter_momentum" in params:
        lines.append(f"jitter_mom  : {params['jitter_momentum']}")
    if "jitter_fill" in params:
        lines.append(f"jitter_fill : {params['jitter_fill']}")
    if "jitter_wrap" in params:
        lines.append(f"jitter_wrap : {params['jitter_wrap']}")
    if "brightness_profile" in params:
        lines.append(f"bright_type : {params['brightness_profile']}")
    if "brightness_base_gain" in params:
        lines.append(f"bright_base : {params['brightness_base_gain']:.2f}")
    if "brightness_start_gain" in params and "brightness_end_gain" in params:
        lines.append(f"bright_gain : {params['brightness_start_gain']:.2f}->{params['brightness_end_gain']:.2f}")
    if "brightness_wave_amp" in params:
        lines.append(f"bright_wave : {params['brightness_wave_amp']:.2f}@{params.get('brightness_wave_cycles', 0.0):.1f}")
    if "light_sigma" in params:
        lines.append(f"light_sigma : {params['light_sigma']:.2f}")
    if "light_spot_gain" in params:
        lines.append(f"light_spot : {params['light_spot_gain']:.2f}")
    if "light_spot_gain_2" in params:
        lines.append(f"light_spot2: {params['light_spot_gain_2']:.2f}")
    if "light_shadow_gain" in params:
        lines.append(f"light_shadow: {params['light_shadow_gain']:.2f}")
    if "light_vignette_gain" in params:
        lines.append(f"light_vignette: {params['light_vignette_gain']:.2f}")
    if "light_center_x_start" in params and "light_center_y_start" in params and "light_center_x_end" in params and "light_center_y_end" in params:
        lines.append(
            f"light_path : ({params['light_center_x_start']:.2f}, {params['light_center_y_start']:.2f})->"
            f"({params['light_center_x_end']:.2f}, {params['light_center_y_end']:.2f})"
        )
    if "light_wave_amp_x" in params or "light_wave_amp_y" in params:
        lines.append(
            f"light_wave : ({params.get('light_wave_amp_x', 0.0):.2f}, {params.get('light_wave_amp_y', 0.0):.2f})@"
            f"{params.get('light_wave_cycles', 0.0):.1f}"
        )
    if "light_center_x2_start" in params and "light_center_y2_start" in params and "light_center_x2_end" in params and "light_center_y2_end" in params:
        lines.append(
            f"light_path2: ({params['light_center_x2_start']:.2f}, {params['light_center_y2_start']:.2f})->"
            f"({params['light_center_x2_end']:.2f}, {params['light_center_y2_end']:.2f})"
        )
    if "light_wave_amp_x2" in params or "light_wave_amp_y2" in params:
        lines.append(
            f"light_wave2: ({params.get('light_wave_amp_x2', 0.0):.2f}, {params.get('light_wave_amp_y2', 0.0):.2f})@"
            f"{params.get('light_wave_cycles_2', 0.0):.1f}"
        )
    if "shadow_blob_gain" in params:
        lines.append(f"shadow_gain: {params['shadow_blob_gain']:.2f}")
    if "shadow_blob_gain2" in params:
        lines.append(f"shadow_gain2: {params['shadow_blob_gain2']:.2f}")
    if "shadow_blob_sigma_x" in params and "shadow_blob_sigma_y" in params:
        lines.append(f"shadow_size: ({params['shadow_blob_sigma_x']:.2f}, {params['shadow_blob_sigma_y']:.2f})")
    if "shadow_blob_sigma_x2" in params and "shadow_blob_sigma_y2" in params:
        lines.append(f"shadow_size2: ({params['shadow_blob_sigma_x2']:.2f}, {params['shadow_blob_sigma_y2']:.2f})")
    if "shadow_blob_wave_amp_x" in params or "shadow_blob_wave_amp_y" in params:
        lines.append(
            f"shadow_wave: ({params.get('shadow_blob_wave_amp_x', 0.0):.2f}, {params.get('shadow_blob_wave_amp_y', 0.0):.2f})@"
            f"{params.get('shadow_blob_wave_cycles', 0.0):.1f}"
        )
    if "shadow_blob_wave_amp_x2" in params or "shadow_blob_wave_amp_y2" in params:
        lines.append(
            f"shadow_wave2: ({params.get('shadow_blob_wave_amp_x2', 0.0):.2f}, {params.get('shadow_blob_wave_amp_y2', 0.0):.2f})@"
            f"{params.get('shadow_blob_wave_cycles_2', 0.0):.1f}"
        )
    if "render_mode" in params:
        lines.append(f"render_mode: {params['render_mode']}")
    if "shadow_model" in params:
        lines.append(f"shadow_type: {params['shadow_model']}")
    if "light_count" in params:
        lines.append(f"light_count : {params['light_count']}")
    if "light_motion" in params:
        lines.append(f"light_move : {params['light_motion']}")
    if "light_intensity_mode" in params:
        lines.append(f"light_inten: {params['light_intensity_mode']}")
    if "ambient_gain" in params:
        lines.append(f"ambient    : {params['ambient_gain']}")
    if "shadow_strength" in params:
        lines.append(f"shadow_str : {params['shadow_strength']}")
    if "occlusion_color" in params:
        lines.append(f"occ_color  : {params['occlusion_color']}")
    if "occlusion_shape" in params:
        lines.append(f"occ_shape  : {params['occlusion_shape']}")
    if "occlusion_anchor" in params:
        lines.append(f"occ_anchor : {params['occlusion_anchor']}")
    if "occlusion_mode" in params:
        lines.append(f"occ_mode   : {params['occlusion_mode']}")
    if "occlusion_angle_frac" in params:
        lines.append(f"occ_angle  : {params['occlusion_angle_frac']:.2f}")
    if "occlusion_segments" in params:
        lines.append(f"occ_segs   : {params['occlusion_segments']}")
    if "occlusion_segment_frac" in params:
        lines.append(f"occ_segfrac: {params['occlusion_segment_frac']:.2f}")
    if "occlusion_r_outer_frac" in params:
        lines.append(f"occ_outer  : {params['occlusion_r_outer_frac']:.2f}L")
    if "occlusion_cell_frac" in params:
        lines.append(f"occ_cell   : {params['occlusion_cell_frac']:.2f}L")
    if "gamma0" in params:
        lines.append(f"video_gamma0: {params['gamma0']}")
    if "gamma1" in params:
        lines.append(f"video_gamma1: {params['gamma1']}")

    return lines
