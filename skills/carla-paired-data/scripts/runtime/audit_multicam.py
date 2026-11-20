#!/usr/bin/env python3
"""Independent, offline audit of the frozen six-camera pilot.

Reads captures and source snapshots; only the requested JSON/CSV are written.
Does not import the collector or review implementation, and never connects to
CARLA. Partial runs are explicitly labelled; this is not a world-model test.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
if __package__:
    from .render_gate_review import checked_render_gate, expected_sensor_tick, render_history_errors
else:
    from render_gate_review import checked_render_gate, expected_sensor_tick, render_history_errors
from PIL import Image
if __package__:
    from .controllers_v2 import branch_names
    from .audit_auxiliary import inspect_auxiliary, compare_auxiliary
else:
    from controllers_v2 import branch_names
    from audit_auxiliary import inspect_auxiliary, compare_auxiliary

BRANCHES = ("maintain", "brake", "accelerate", "left", "right", "maintain_repeat")
CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
TOLERANCES = {"position_m": .01, "velocity_m_s": .02, "rotation_deg": .1,
              "angular_velocity_deg_s": .1, "acceleration_m_s2": .1}


if __package__:
    from . import semantic_lidar
else:
    import semantic_lidar


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def common_positions(position, world_to_common):
    return position @ world_to_common[:3, :3].T + world_to_common[:3, 3]


def per_role_errors(actual, reference, indices, roles):
    """Absolute wrapped Euler errors plus actual vector and scalar speed errors."""
    output = {}
    for i, role in enumerate(roles):
        p = actual["position"][indices, i] - reference["position"][indices, i]
        record = {"xyz_component_max_m": np.abs(p).max(0).tolist(),
                  "position_m": float(np.linalg.norm(p, axis=-1).max()),
                  "rotation_deg": float(np.abs((actual["rotation_deg"][indices, i] - reference["rotation_deg"][indices, i] + 180) % 360 - 180).max())}
        for key, label in (("velocity", "velocity_m_s"), ("angular_velocity", "angular_velocity_deg_s"), ("acceleration", "acceleration_m_s2")):
            record[label] = float(np.linalg.norm(actual[key][indices, i] - reference[key][indices, i], axis=-1).max())
        record["speed_m_s"] = float(np.abs(np.linalg.norm(actual["velocity"][indices, i], axis=-1) - np.linalg.norm(reference["velocity"][indices, i], axis=-1)).max())
        output[role] = record
    return output


def calibration_errors(sample):
    expected = sample["ego_to_world"][:, None] @ sample["camera_to_ego"][None]
    measured = np.linalg.inv(sample["world_to_camera"])
    cycle = sample["world_to_camera"] @ expected
    return {"translation_max_m": float(np.linalg.norm(measured[..., :3, 3] - expected[..., :3, 3], axis=-1).max()),
            "rotation_matrix_max_error": float(np.abs(measured[..., :3, :3] - expected[..., :3, :3]).max()),
            "cycle_matrix_max_error": float(np.abs(cycle - np.eye(4)).max())}


def equivalent_attribute(actual, expected):
    """CARLA exposes numeric sensor attributes as serialized strings."""
    try:
        return bool(np.isclose(float(actual),float(expected),rtol=1e-6,atol=1e-6))
    except (TypeError,ValueError):
        return str(actual).lower() == str(expected).lower()


def actual_render_checks(meta, config):
    """Verify actor-side attributes and weather, not only requested settings."""
    errors, cameras = [], {}
    expected_weather = config.get('weather',{})
    actual_weather = meta.get('weather_actual',{})
    stages = {'weather_actual':actual_weather}
    if 'weather_warmup_s' in config.get('timing',{}):
        stages.update({name:meta.get(name,{}) for name in ('weather_after_warmup','weather_at_history_start')})
    for stage,observed in stages.items():
        for key,value in expected_weather.items():
            if key not in observed or not equivalent_attribute(observed[key],value):
                errors.append(f'Actual weather differs from frozen config: {stage}/{key}')
    if 'control_dispatch' in config and meta.get('control_dispatch') != config['control_dispatch']:
        errors.append('Actual control-dispatch protocol differs from frozen config')
    for index,spec in enumerate(config['camera_rig']):
        record = meta['rig'][index]
        requested,actual = record.get('attributes',{}),record.get('actor_attributes',{})
        mismatches = [key for key,value in requested.items()
                      if key not in actual or not equivalent_attribute(actual[key],value)]
        for key,value in config['render']['camera_attributes'].items():
            if key not in actual or not equivalent_attribute(actual[key],value):
                if key not in mismatches:mismatches.append(key)
        if mismatches:errors.append(f"{spec['name']}: actual sensor attributes differ: {mismatches}")
        cameras[spec['name']] = {'requested_attributes':requested,'actual_attributes':actual,
                                 'mismatched_attributes':mismatches}
    return {'weather_actual':actual_weather,'weather_requested':expected_weather,
            'weather_readback_stages':stages,'cameras':cameras,'errors':errors}


def renderer_prime_checks(meta, config, recorded_frame_ids=None):
    """Audit discarded render-episode evidence under its declared protocol.

    These checks verify the saved evidence, not visual quality or successful
    individual actor destruction. The latter is not measured in this schema.
    Historical configurations without priming remain independently auditable.
    """
    seconds = config.get('timing', {}).get('renderer_prime_s', 0)
    errors = []
    result = {'required': False, 'errors': errors,
              'scope': 'Recorded priming evidence; image quality and repeatability require capture checks.'}

    def close(actual, expected, tolerance=1e-6):
        try:
            return bool(np.isfinite(float(actual)) and np.isclose(float(actual), expected, rtol=0, atol=tolerance))
        except (ValueError, TypeError):
            return False

    if seconds == 0:
        return result
    if not close(seconds, seconds) or not isinstance(seconds, (int, float)) or seconds < 0:
        errors.append('Invalid configured renderer_prime_s')
        return result
    result['required'] = True
    ticks = round(seconds / .05)
    if ticks < 1 or not close(ticks * .05, seconds, 1e-9):
        errors.append('Renderer prime duration is not a whole number of 20 Hz ticks')
    prime = meta.get('renderer_warmup')
    if not isinstance(prime, dict):
        errors.append('Missing renderer_warmup evidence')
        return result
    result['recorded'] = prime
    for key, expected in {'status': 'complete', 'render_frames': ticks,
                          'images_saved': 0, 'carla_client_version': '0.9.16',
                          'carla_server_version': '0.9.16',
                          'reset_after': 'reload_world(False) before the ordinary branch initialization'}.items():
        if prime.get(key) != expected:
            errors.append(f'Renderer prime {key} differs: expected {expected!r}')
    for key, expected in {'renderer_prime_s': seconds, 'physics_dt_s': .05,
                          'weather_warmup_frames': round(config['timing']['weather_warmup_s'] / .05)}.items():
        if not close(prime.get(key), expected, 1e-9):
            errors.append(f'Renderer prime {key} differs from configured timing')
    first, last = prime.get('first_frame'), prime.get('last_frame')
    frame_ids_valid = all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (first, last))
    if not frame_ids_valid or last - first + 1 != ticks:
        errors.append('Renderer prime first/last frame span does not contain the configured tick count')
    if frame_ids_valid and recorded_frame_ids is not None and np.any(np.asarray(recorded_frame_ids) <= last):
        errors.append('Recorded branch contains frame IDs from or before the discarded prime episode')
    final_time = prime.get('final_simulation_time_s')
    if not close(final_time, final_time) or float(final_time) < seconds:
        errors.append('Renderer prime final simulation timestamp is invalid')
    expected_names = [spec['name'] for spec in config['camera_rig']]
    if len(expected_names) != 6 or set(expected_names) != set(CAMERAS) or prime.get('camera_names') != expected_names:
        errors.append('Renderer prime did not declare the complete ordered six-camera rig')
    samples, final_frames = prime.get('sensor_last_samples', {}), prime.get('sensor_final_frames', {})
    if not isinstance(samples, dict):
        samples = {}
    if not isinstance(final_frames, dict):
        final_frames = {}
    if set(samples) != set(expected_names) or set(final_frames) != set(expected_names):
        errors.append('Renderer prime lacks exactly six final sensor samples')
    rig = prime.get('rig', [])
    if not isinstance(rig, list) or [r.get('name') if isinstance(r, dict) else None for r in rig] != expected_names:
        errors.append('Renderer prime actor rig differs from configured camera order')
        rig = []
    for index, spec in enumerate(config['camera_rig']):
        name = spec['name']
        sample = samples.get(name, {})
        if not isinstance(sample, dict):
            sample = {}
        wanted = {'frame_id': last, 'width': spec['width'], 'height': spec['height'],
                  'buffer_bytes': spec['width'] * spec['height'] * 4}
        if any(sample.get(key) != value for key, value in wanted.items()) or final_frames.get(name) != last:
            errors.append(f'{name}: renderer prime final frame or native BGRA buffer differs')
        if not close(sample.get('timestamp'), final_time):
            errors.append(f'{name}: renderer prime timestamp differs from final world snapshot')
        if rig:
            record = rig[index]
            attributes = record.get('actor_attributes', {})
            required = {**record.get('attributes_requested', {}),
                        **config['render']['camera_attributes'], 'image_size_x': spec['width'],
                        'image_size_y': spec['height'], 'fov': spec['fov_deg'],
                        'sensor_tick': expected_sensor_tick(config), 'enable_postprocess_effects': 'true'}
            if any(key not in attributes or not equivalent_attribute(attributes[key], value)
                   for key, value in required.items()):
                errors.append(f'{name}: renderer prime actual sensor attributes differ from the frozen rig')
    for stage in ('weather_after_warmup', 'weather_actual'):
        observed = prime.get(stage, {})
        if any(key not in observed or not equivalent_attribute(observed[key], value)
               for key, value in config['weather'].items()):
            errors.append(f'Renderer prime {stage} differs from frozen weather')
    return result


def contact_summary(meta):
    events = meta.get("collisions", [])
    times = [float(event["time_s"]) for event in events]
    return {"event_count": len(events), "history_or_t0_event_count": sum(t <= 0 for t in times),
            "future_event_count": sum(t > 0 for t in times), "first_time_s": min(times) if times else None,
            "events": events, "count_note": "Sensor events, not deduplicated physical contacts."}


def reaction_evidence(actual, reference, repeat, world_to_common, times, roles, first_contact=None):
    """Same identity/time; one frozen reference transform for every branch."""
    common = common_positions(actual, world_to_common)
    common_ref = common_positions(reference, world_to_common)
    common_repeat = common_positions(repeat, world_to_common)
    delta = common - common_ref
    world_delta = actual - reference
    future = times > 0
    before = future & (times < first_contact) if first_contact is not None else future
    peak = np.linalg.norm(delta[future, :, :2], axis=-1).max(0)
    world_peak = np.linalg.norm(world_delta[future, :, :2], axis=-1).max(0)
    noise = np.linalg.norm((common_repeat-common_ref)[future, :, :2], axis=-1).max(0)
    pre_peak = np.linalg.norm(delta[before, :, :2], axis=-1).max(0) if before.any() else np.zeros(len(roles))
    output = {"per_role": {}, "thresholds": {},
              "rigid_3d_distance_max_error_m": float(np.abs(np.linalg.norm(delta, axis=-1)-np.linalg.norm(world_delta, axis=-1)).max()),
              "common_xy_vs_world_xy_max_error_m": float(np.abs(np.linalg.norm(delta[..., :2], axis=-1)-np.linalg.norm(world_delta[..., :2], axis=-1)).max()),
              "xy_note": "Common XY is the initial ego plane. Pitch/roll can change XY lengths relative to world XY; full 3D distance must be invariant.",
              "first_contact_cutoff_s": first_contact, "pre_contact_physics_samples": int(before.sum())}
    for i, role in enumerate(roles):
        output["per_role"][role] = {"common_future_peak_xy_m": float(peak[i]), "world_future_peak_xy_m": float(world_peak[i]),
                                    "repeat_future_peak_xy_m": float(noise[i]), "pre_contact_peak_xy_m": float(pre_peak[i])}
    for threshold in (.5, 1.):
        cutoff = np.maximum(threshold, noise+.05)
        chosen = peak > cutoff
        world_chosen = world_peak > cutoff
        ego = roles.index('ego') if 'ego' in roles else None
        if ego is not None:
            chosen[ego] = False
            world_chosen[ego] = False
        output["thresholds"][str(threshold)] = {
            "nominal_threshold_m": threshold, "noise_margin_m": .05,
            "effective_cutoff_m": dict(zip(roles, cutoff.tolist())),
            "roles": [role for i, role in enumerate(roles) if chosen[i]],
            "pre_contact_roles": [role for i, role in enumerate(roles) if chosen[i] and pre_peak[i] > cutoff[i]],
            "world_xy_classification_disagreement_roles": [role for i, role in enumerate(roles) if chosen[i] != world_chosen[i]],
            "ego_is_excluded": ego is None or not bool(chosen[ego])}
    return output


def reaction_timing_evidence(actual, reference, roles, first_contact=None):
    """Record control/trajectory/contact ordering, without inferring causality.

    control[t] is issued after state[t] and before state[t+1]. The final stored
    zero control is never executed and is excluded. Thresholds here detect
    numerical onset, not the separate 0.5/1m reactive-region classification.
    """
    times = actual['time_s']
    valid_control = (times >= 0) & (np.arange(len(times)) < len(times)-1)
    future = times > 0
    pre_contact = times < first_contact if first_contact is not None else np.ones(len(times), dtype=bool)
    result = {'first_contact_in_either_branch_s': first_contact,
              'control_difference_threshold': 1e-5, 'trajectory_difference_threshold_m': .01,
              'scope': 'Temporal evidence only; control[t] precedes the next physics step. Contact-free differences are recorded separately.',
              'per_role': {}}
    for i, role in enumerate(roles):
        control_delta = np.max(abs(actual['control'][:, i]-reference['control'][:, i]), axis=-1)
        trajectory_delta = np.linalg.norm(actual['position'][:, i, :2]-reference['position'][:, i, :2], axis=-1)
        controls = np.flatnonzero(valid_control & (control_delta > 1e-5))
        movement = np.flatnonzero(future & (trajectory_delta > .01))
        control_time = float(times[controls[0]]) if len(controls) else None
        movement_time = float(times[movement[0]]) if len(movement) else None
        before = future & pre_contact
        result['per_role'][role] = {
            'first_control_difference_s': control_time,
            'first_xy_difference_over_1cm_s': movement_time,
            'control_change_precedes_trajectory_onset': bool(control_time is not None and movement_time is not None and control_time < movement_time),
            'control_change_before_contact': bool(control_time is not None and (first_contact is None or control_time < first_contact)),
            'trajectory_onset_before_contact': bool(movement_time is not None and (first_contact is None or movement_time < first_contact)),
            'pre_contact_peak_world_xy_difference_m': float(trajectory_delta[before].max()) if before.any() else 0.,
            'pre_contact_control_difference_max': float(control_delta[valid_control & pre_contact].max()) if (valid_control & pre_contact).any() else 0.,
        }
    return result


def valid_snapshots(run):
    found = []
    for folder in sorted((run / "source").glob("*")):
        if not (folder / "sha256.json").exists():
            continue
        hashes = read(folder / "sha256.json")
        valid = all((folder/name).is_file() and sha(folder/name) == expected for name, expected in hashes.items())
        found.append({"path": str(folder.resolve()), "hashes": hashes, "content_verified": valid})
    return found


def inspect_branch(base, config, name, snapshots, verify_png):
    errors = []
    meta = read(base / "meta.json")
    state, sample = arrays(base / "states.npz"), arrays(base / "samples.npz")
    report = {"path": str(base.resolve()), "captured_utc": meta.get("created_utc"), "errors": errors}
    report['semantic_lidar'] = semantic_lidar.inspect_branch(base, config, meta, state, sample)
    errors.extend(report['semantic_lidar']['errors'])
    report['auxiliary_actors'] = inspect_auxiliary(base, config, meta, state)
    errors.extend(report['auxiliary_actors']['errors'])
    for key in ("spec_sha256", "code_sha256"):
        if meta.get(key) != config.get(key):
            errors.append(f"Metadata {key} differs from frozen configuration")
    if meta.get("status") != "complete" or meta.get("branch") != name:
        errors.append("Capture is incomplete or has a different branch name")
    if meta.get("vehicle_roles") != config["vehicle_roles"] or meta.get("camera_names") != list(CAMERAS):
        errors.append("Recorded role/camera order differs from declared configuration")
    # The relit canonical pilot records actor-side metadata; historical captures
    # predate this field and are only checked under their original protocol.
    if config.get('render',{}).get('lighting_protocol'):
        report['actual_render'] = actual_render_checks(meta,config)
        errors.extend(report['actual_render']['errors'])
    report['renderer_prime'] = renderer_prime_checks(meta, config, state.get('frame_id'))
    errors.extend(report['renderer_prime']['errors'])
    matches = [s["path"] for s in snapshots if s["content_verified"] and all(s["hashes"].get(k) == v for k, v in meta.get("code_sha256", {}).items())]
    report["verified_source_snapshots"] = matches
    if not matches:
        errors.append("No verified archived source matches actual capture metadata")
    n = len(config["vehicle_roles"])
    shapes = {"time_s": (121,), "frame_id": (121,), "simulation_time_s": (121,), "transform_world_vehicle": (121,n,4,4),
              "bbox_extent": (n,3), "bbox_offset": (n,3), "bbox_rotation_deg": (n,3), "target_speed": (121,n)}
    shapes.update({key: (121,n,3) for key in ("position","rotation_deg","velocity","angular_velocity","acceleration","control")})
    sample_shapes = {"time_s": (13,), "physics_indices": (13,), "frame_id": (13,), "sensor_frame_id": (13,7),
                     "world_to_camera": (13,6,4,4), "camera_to_ego": (6,4,4), "intrinsics": (6,3,3), "ego_to_world": (13,4,4)}
    shape_errors = []
    for prefix, value, wanted in (("state", state, shapes), ("sample", sample, sample_shapes)):
        for key, shape in wanted.items():
            if key not in value or value[key].shape != shape or not np.isfinite(value[key]).all():
                shape_errors.append(f"Invalid {prefix}.{key}: expected finite {shape}")
    errors.extend(shape_errors)
    if shape_errors:
        return report, None, None, meta
    checks = {
        "physics_times": np.allclose(state["time_s"], np.arange(121)*.05-2, rtol=0, atol=1e-8),
        "sample_times": np.allclose(sample["time_s"], np.arange(13)*.5-2, rtol=0, atol=1e-8),
        "sample_indices": np.array_equal(sample["physics_indices"], np.arange(0,121,10)),
        "consecutive_physics_frames": np.all(np.diff(state["frame_id"]) == 1),
        "saved_frames_match_states": np.array_equal(sample["frame_id"], state["frame_id"][::10]),
        "all_seven_sensors_synchronized": np.all(sample["sensor_frame_id"] == sample["frame_id"][:,None]),
        "physical_timestamps": np.allclose(np.diff(state["simulation_time_s"]), .05, rtol=0, atol=1e-4),
        "sample_ego_pose": np.allclose(sample["ego_to_world"], state["transform_world_vehicle"][::10,0], rtol=0, atol=1e-5),
        "actor_xyz": np.allclose(state["transform_world_vehicle"][...,:3,3], state["position"], rtol=0, atol=1e-5),
        "t0_frame": meta.get("t0_frame") == int(state["frame_id"][40]),
        "executed_control_range": np.all((state["control"][:120,...,[0,2]] >= 0) & (state["control"][:120,...,[0,2]] <= 1)) and np.all(abs(state["control"][:120,...,1]) <= 1),
    }
    report["frame_and_state_checks"] = {k: bool(v) for k,v in checks.items()}
    errors.extend(k for k,v in checks.items() if not v)
    report["calibration"] = calibration_errors(sample)
    if report["calibration"]["translation_max_m"] > .002 or report["calibration"]["rotation_matrix_max_error"] > 1e-4:
        errors.append("Camera extrinsics disagree with rigid ego-mounted rig")
    for index, spec in enumerate(config["camera_rig"]):
        focal = spec["width"]/(2*np.tan(np.deg2rad(spec["fov_deg"])/2))
        expected_k = np.array([[focal,0,spec["width"]/2],[0,focal,spec["height"]/2],[0,0,1]])
        if not np.allclose(sample["intrinsics"][index], expected_k, atol=1e-6, rtol=0):
            errors.append(f"{spec['name']}: intrinsic matrix differs from frozen camera FOV")
        attrs = meta["rig"][index]["attributes"]
        if abs(float(attrs["sensor_tick"])-expected_sensor_tick(config)) > 1e-8 or attrs["enable_postprocess_effects"].lower() != "true":
            errors.append(f"{spec['name']}: sensor was not configured for 20 Hz postprocessed rendering")
    errors.extend(render_history_errors(meta, state, config))
    environment = read(base/"environment_vehicles.json")
    report["static_environment_vehicle_geometry_records"] = len(environment["objects"])
    if environment.get("queried_before_dynamic_spawn") is not True:
        errors.append("Map vehicle geometry was not distinguished from dynamic actors")
    report["camera_png_count"], report["camera_png_bytes"] = 0, 0
    expected = [f"{i:05d}.png" for i in range(13)]
    for camera in CAMERAS:
        images = sorted((base/"camera"/camera).glob("*.png"))
        report["camera_png_count"] += len(images)
        if [p.name for p in images] != expected:
            errors.append(f"{camera}: not exactly the 13 required PNGs")
        for path in images:
            report["camera_png_bytes"] += path.stat().st_size
            with Image.open(path) as image:
                if image.size != (1600,900) or image.mode != "RGB" or image.format != "PNG":
                    errors.append(f"{path.name}/{camera}: not native 1600x900 RGB PNG")
                if verify_png:
                    image.verify()
    report["png_crc_verified"] = verify_png
    report["contacts"] = contact_summary(meta)
    if report["contacts"]["history_or_t0_event_count"]:
        errors.append("Contact during common history or at t0")
    report["states_file_sha256"] = sha(base/"states.npz")
    report["samples_file_sha256"] = sha(base/"samples.npz")
    return report, state, sample, meta


def action_direction_ok(config, branch, lateral, speed_delta):
    """Coarse executed direction only; lane completion needs its own audit."""
    direction = (branch != 'left' or lateral < -.1) and (branch != 'right' or lateral > .1)
    profile = config.get('action', {}).get('profiles', {}).get(branch, {})
    signed = profile.get('acceleration_m_s2', -1 if branch == 'brake' else 1 if branch == 'accelerate' else 0)
    return bool(direction and (signed >= 0 or speed_delta < -.1) and (signed <= 0 or speed_delta > .1))


def inspect_scene(folder, snapshots, verify_png, run=None):
    config = read(folder/"spec.json")
    roles = config["vehicle_roles"]
    report = {"scene_id": folder.name, "family": config["family"], "map": config["map"], "seed": config["seed"],
              "spec_sha256": config["spec_sha256"], "errors": [], "warnings": [], "missing_branches": [], "branches": {}}
    # A complete/synchronized physical capture is not evidence of repeatable
    # pixels. Bind the independently computed native-RGB gate to current bytes
    # before any early return so a missing gate cannot yield a final audit pass.
    if 'render_consistency_policy' in config:
        if run is None:
            result = {'required': True, 'status': 'pending',
                      'reason': 'Run directory is required to locate the native RGB gate'}
        else:
            result = checked_render_gate(folder.parent, run, folder.name, config)
        report['render_consistency'] = result
        if result['status'] != 'passed':
            report['errors'].append('Native RGB consistency '+result['status']+': '
                                    +result.get('reason', 'gate did not pass'))
    if (not roles or roles[0] != 'ego' or roles.count('ego') != 1 or len(set(roles)) != len(roles)
            or roles != [v.get('role') for v in config['vehicles']]):
        report['errors'].append('Vehicle roles must match config order with sole ego at index0')
        return report
    planned = branch_names(config)
    report['planned_branches'] = list(planned)
    if content_sha({k:v for k,v in config.items() if k != "spec_sha256"}) != config["spec_sha256"]:
        report["errors"].append("Invalid frozen configuration content hash")
    map_path = folder.parent/config["map_geometry_path"]
    if not map_path.is_file() or sha(map_path) != config["provenance"]["opendrive_sha256"]:
        report["errors"].append("Archived OpenDRIVE differs from geometry used to plan scene")
    report["map_geometry_path"] = str(map_path.resolve())
    extra = sorted(p.name for p in (folder/"branches").glob("*") if p.is_dir() and p.name not in planned)
    if extra:
        report["errors"].append(f"Unexpected canonical branch directories: {extra}")
    states, metas = {}, {}
    for branch in planned:
        base = folder/"branches"/branch
        if not (base/"meta.json").is_file():
            report["missing_branches"].append(branch)
            continue
        try:
            info, state, _, meta = inspect_branch(base, config, branch, snapshots, verify_png)
            report["branches"][branch] = info
            if state is not None:
                states[branch], metas[branch] = state, meta
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["branches"][branch] = {"errors": [f"Unreadable capture: {type(exc).__name__}: {exc}"]}
    if "maintain" not in states:
        return report
    ref = states["maintain"]
    anchor = np.linalg.inv(ref["transform_world_vehicle"][40,0])
    report["world_to_common_t0"] = anchor.tolist()
    # CARLA serializes float32-derived transforms. Do not demand float64
    # machine precision of a matrix whose source is only float32 accurate.
    orthogonality_error = float(abs(anchor[:3,:3].T @ anchor[:3,:3]-np.eye(3)).max())
    report["anchor_rotation_orthogonality_error"] = orthogonality_error
    if orthogonality_error > 1e-5 or abs(np.linalg.det(anchor[:3,:3])-1) > 1e-5:
        report["errors"].append("Initial ego anchor is not a proper rigid transform")
    report["anchor_branch"] = "maintain"
    report["anchor_physics_index"] = 40
    for branch, state in states.items():
        info = report["branches"][branch]
        info['auxiliary_common_history'] = compare_auxiliary(folder/'branches'/branch,
                                                            folder/'branches/maintain',config,scope='history')
        info['errors'].extend(info['auxiliary_common_history']['errors'])
        for scope, indices in (("history", np.arange(41)), ("t0", np.array([40]))):
            info[scope] = per_role_errors(state, ref, indices, roles)
            for role, values in info[scope].items():
                for key, limit in TOLERANCES.items():
                    if values[key] > limit:
                        info["errors"].append(f"{scope}/{role}/{key} {values[key]:.6g} exceeds {limit}")
        info["history_control_max_error"] = float(abs(state["control"][:40]-ref["control"][:40]).max())
        if info["history_control_max_error"] > 1e-5:
            report["warnings"].append(f"{branch}: common-history control differences exceed 1e-5; state tolerances are checked separately")
        speed = np.linalg.norm(state["velocity"][:,0,:2], axis=-1)
        refspeed = np.linalg.norm(ref["velocity"][:,0,:2], axis=-1)
        delta = state["position"][-1,0,:2]-ref["position"][-1,0,:2]
        heading = np.deg2rad(ref["rotation_deg"][-1,0,2])
        lateral = float(delta @ np.array([-np.sin(heading), np.cos(heading)]))
        info["ego_executed_action"] = {"speed_t0_m_s": float(speed[40]), "speed_final_m_s": float(speed[-1]),
            "speed_final_minus_reference_m_s": float(speed[-1]-refspeed[-1]),
            "final_offset_relative_to_reference_heading_right_m": lateral,
            "future_speed_deviation_from_t0_max_m_s": float(abs(speed[40:]-speed[40]).max()),
            "final_world_xy_minus_reference_m": delta.tolist()}
        direction = action_direction_ok(config, branch, lateral, float(speed[-1]-refspeed[-1]))
        info["ego_executed_action"]["coarse_direction_check"] = bool(direction)
        if not direction:
            info["errors"].append("Actual ego endpoint/speed does not follow declared intervention direction")
        if branch == "maintain" and abs(speed[40:]-speed[40]).max() > 1:
            report["warnings"].append("Reference cruise speed deviates more than 1 m/s from t0")
        if "maintain_repeat" in states:
            contact_times = [value for value in (info["contacts"]["first_time_s"], report["branches"]["maintain"]["contacts"]["first_time_s"]) if value is not None]
            reaction = reaction_evidence(state["position"], ref["position"], states["maintain_repeat"]["position"], anchor, ref["time_s"], roles, min(contact_times) if contact_times else None)
            info["reaction"] = reaction
            info['reaction_timing'] = reaction_timing_evidence(state, ref, roles, min(contact_times) if contact_times else None)
            if reaction["rigid_3d_distance_max_error_m"] > 1e-4:
                info["errors"].append("Shared reference transform does not preserve 3D trajectory differences")
            if any(x["world_xy_classification_disagreement_roles"] for x in reaction["thresholds"].values()):
                report["warnings"].append(f"{branch}: world-XY and initial-ego-plane-XY threshold labels differ; inspect pitch/roll")
    if "maintain_repeat" in states:
        report['auxiliary_repeat'] = compare_auxiliary(folder/'branches/maintain_repeat',
                                                       folder/'branches/maintain',config,scope='repeat')
        report['errors'].extend(report['auxiliary_repeat']['errors'])
        report["repeat_full_sequence"] = per_role_errors(states["maintain_repeat"], ref, np.arange(121), roles)
        if max(v["position_m"] for v in report["repeat_full_sequence"].values()) > .05:
            report["errors"].append("Repeat full-sequence XYZ error exceeds 0.05 m")
    return report


def dataset_plan(data):
    """Manifest declares missing scenes; each config declares its own actions."""
    manifest_file = data/'manifest.json'
    manifest = read(manifest_file) if manifest_file.exists() else {}
    discovered = sorted(p.parent.name for p in data.glob('scene_*/spec.json'))
    planned = list(manifest.get('scene_ids', discovered))
    errors, scene_branches = [], {}
    if not planned or len(set(planned)) != len(planned):
        errors.append('Expected nonempty distinct planned scene IDs')
    extra = sorted(set(discovered)-set(planned))
    if extra:
        errors.append(f'Unplanned scene configs: {extra}')
    for scene in planned:
        fallback = manifest.get('scene_branches', {}).get(scene, manifest.get('branches', BRANCHES))
        try:
            names = list(branch_names(read(data/scene/'spec.json')))
            if scene in manifest.get('scene_branches', {}) and names != fallback:
                errors.append(f'{scene}: config branches differ from manifest scene branches')
            if 'branches' in manifest and not set(names) <= set(manifest['branches']):
                errors.append(f'{scene}: configured branch absent from manifest union')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f'{scene}: invalid/missing config branch protocol: {exc}')
            names = list(fallback)
        scene_branches[scene] = names
    return planned, scene_branches, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scenes", nargs="+", help="Subset IDs/indices for a partial preflight")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--verify-png", action="store_true", help="Check PNG chunk CRCs, otherwise only native image headers")
    args = parser.parse_args()
    data, run = args.data.resolve(), args.run.resolve()
    out = (args.out or run/"independent_audit.json").resolve()
    planned, scene_branches, plan_errors = dataset_plan(data)
    expected_branch_count = sum(map(len, scene_branches.values()))
    expected_rgb_count = expected_branch_count*13*len(CAMERAS)
    selected = [f"scene_{int(v):04d}" if v.isdigit() else v for v in args.scenes] if args.scenes else planned
    if any(Path(v).name != v or v not in planned for v in selected):
        parser.error("Unknown or unsafe scene ID")
    if len(set(selected)) != len(selected):
        parser.error("Duplicate scene selection")
    started = datetime.now(timezone.utc).isoformat()
    snapshots = valid_snapshots(run)
    report = {"version": "independent-multicam-audit-v1", "data_root": str(data), "run_root": str(run),
              "started_utc": started, "command": sys.argv, "audit_source_sha256": sha(__file__),
              "render_gate_dependencies_sha256": {name: sha(Path(__file__).parent/name) for name in
                                                  ('render_gate_review.py', 'audit_render_consistency.py')},
              "expected": {"scenes": len(planned), "branches": expected_branch_count,
                           "scene_branches": scene_branches, "six_camera_rgb_pngs": expected_rgb_count,
                           "rgb_samples_per_branch": 13, "physics_states_per_branch": 121},
              "initial_and_history_tolerances": TOLERANCES, "repeat_position_tolerance_m": .05,
              "rigid_distance_tolerance_m": 1e-4, "rigid_tolerance_note": "CARLA transforms have float32 source precision; 0.1 mm bound is below state comparison tolerances.",
              "source_snapshots": snapshots, "planned_scene_count": len(planned), "selected_scene_ids": selected,
              "png_validation": "native dimensions/mode plus chunk CRC" if args.verify_png else "native dimensions/mode/header only; no pixel CRC or decode",
              "scientific_scope": "Dataset geometry, provenance and replay audit; no model inference and no action-neglect conclusion.",
              "scenes": [], "errors": plan_errors}
    for name in selected:
        try:
            record = inspect_scene(data/name, snapshots, args.verify_png, run=run)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record = {'scene_id': name, 'family': None, 'branches': {},
                      'missing_branches': scene_branches[name], 'errors': [f'Invalid scene: {exc}']}
        report["scenes"].append(record)
        print(name, "branches", len(record["branches"]), "missing", len(record["missing_branches"]), flush=True)
    errors = report["errors"]
    complete_branches = sum(len(s["branches"]) for s in report["scenes"])
    rgb_count = sum(b.get("camera_png_count",0) for s in report["scenes"] for b in s["branches"].values())
    partial = set(selected) != set(planned) or complete_branches != expected_branch_count or any(s["missing_branches"] for s in report["scenes"])
    for scene in report["scenes"]:
        errors.extend(scene["scene_id"]+": "+e for e in scene["errors"])
        for branch, info in scene["branches"].items():
            errors.extend(scene["scene_id"]+"/"+branch+": "+e for e in info["errors"])
    if partial and not args.allow_partial:
        errors.append("Incomplete data; partial audit requires --allow-partial")
    if not partial and rgb_count != expected_rgb_count:
        errors.append(f"Full dataset must contain exactly {expected_rgb_count} raw six-camera PNGs")
    report["observed"] = {"scenes": len(selected), "branches": complete_branches, "six_camera_rgb_pngs": rgb_count,
                           "six_camera_png_bytes": sum(b.get("camera_png_bytes",0) for s in report["scenes"] for b in s["branches"].values())}
    report["status"] = "failed" if errors else "partial_pass" if partial else "passed"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    fields = ["scene", "family", "branch", "role", "history_xyz_max_m", "t0_xyz_max_m", "repeat_xyz_max_m", "future_peak_xy_m", "pre_contact_peak_xy_m", "reactive_05", "reactive_10", "contact_events", "branch_error_count"]
    with out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scene in report["scenes"]:
            for branch, info in scene["branches"].items():
                for role, values in info.get("history", {}).items():
                    reaction = info.get("reaction", {})
                    writer.writerow({"scene": scene["scene_id"], "family": scene["family"], "branch": branch, "role": role,
                        "history_xyz_max_m": values["position_m"], "t0_xyz_max_m": info["t0"][role]["position_m"],
                        "repeat_xyz_max_m": scene.get("repeat_full_sequence",{}).get(role,{}).get("position_m"),
                        "future_peak_xy_m": reaction.get("per_role",{}).get(role,{}).get("common_future_peak_xy_m"),
                        "pre_contact_peak_xy_m": reaction.get("per_role",{}).get(role,{}).get("pre_contact_peak_xy_m"),
                        "reactive_05": role in reaction.get("thresholds",{}).get("0.5",{}).get("roles",[]),
                        "reactive_10": role in reaction.get("thresholds",{}).get("1.0",{}).get("roles",[]),
                        "contact_events": info["contacts"]["event_count"], "branch_error_count": len(info["errors"])})
    print(report["status"], report["observed"], "errors", len(errors), "output", out, flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
