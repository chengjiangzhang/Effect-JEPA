#!/usr/bin/env python3
"""Collect frozen multicamera plans using one fresh Python process per branch.

Only complete, frame-consistent outputs move from data-disk staging into the
canonical dataset. Failures and their logs remain available for inspection.
Policy-enabled scenes collect maintain and maintain_repeat first. The remaining
actions cannot start until a fresh native-RGB repeat preflight has passed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
if __package__:
    from .controllers_v2 import branch_names
    from . import semantic_lidar
else:
    from controllers_v2 import branch_names
    import semantic_lidar
if __package__:
    from .audit_render_consistency import audit as audit_render_consistency, DEFAULT_THRESHOLDS
else:
    from audit_render_consistency import audit as audit_render_consistency, DEFAULT_THRESHOLDS

HERE = Path(__file__).resolve().parent
BRANCHES = ("maintain", "brake", "accelerate", "left", "right", "maintain_repeat")
SOURCE_FILES = ("collect_multicam.py", "scenarios_v2.py", "controllers_v2.py", "npc_merge.py", "auxiliary_actors.py", "ego_decisions.py", "semantic_lidar.py", "pedestrian_policy.py", "pedestrian_crossing.py")
MIN_FREE_BYTES = int(1.5 * 1024**3)
# A wall-clock resource budget, not a simulation horizon or quality tolerance.
# Six 1600x900 cameras can still be producing valid frames beyond 180 seconds.
BRANCH_TIMEOUT_S = 600
RENDER_POLICY = {'schema_version': 1, 'policy_id': 'native-rgb-repeat-v1',
                 'tile_size_pixels': 100, 'thresholds': dict(DEFAULT_THRESHOLDS)}
RENDER_GATE_SOURCE = 'audit_render_consistency.py'
REFERENCE_PAIR = ('maintain', 'maintain_repeat')


def now():
    return datetime.now(timezone.utc).isoformat()


def run_branch_process(command, stream):
    """Keep a hard upper bound and distinguish expiration from a child exit."""
    try:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=BRANCH_TIMEOUT_S, check=False, start_new_session=True)
        return result.returncode, False
    except subprocess.TimeoutExpired:
        # subprocess.run kills and waits for its child before raising.
        return -1, True


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {name: file_sha(HERE / name) for name in SOURCE_FILES}


def validate_plan(config, hashes):
    errors = []
    if config.get("code_sha256") != hashes:
        errors.append("Collection source SHA256 differs from frozen plan")
    body = {key: value for key, value in config.items() if key != "spec_sha256"}
    if digest(body) != config.get("spec_sha256"):
        errors.append("Config content differs from spec_sha256")
    if len(config.get("camera_rig", [])) != 6:
        errors.append("Frozen plan must declare exactly six cameras")
    if any((cam.get("width"), cam.get("height")) != (1600, 900) for cam in config.get("camera_rig", [])):
        errors.append("Frozen camera resolution must be 1600 x 900")
    if config.get("render", {}).get("quality_level") != "Epic":
        errors.append("Frozen plan must require Epic quality")
    if 'render_consistency_policy' in config:
        try:
            if digest(config['render_consistency_policy']) != digest(RENDER_POLICY):
                errors.append('render_consistency_policy must exactly match native-rgb-repeat-v1; no relaxed or missing fields')
        except (TypeError, ValueError):
            errors.append('render_consistency_policy is not a valid finite JSON policy')
    try:
        branch_names(config)
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(f"Invalid scene action protocol: {exc}")
    try:
        semantic_lidar.validate_spec(config.get('semantic_lidar'))
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def validate_manifest_render_policy(manifest, configs):
    """A dataset-level policy cannot be bypassed by a legacy scene config."""
    if not isinstance(manifest, dict):
        return ['Dataset manifest must be a JSON object']
    if 'render_consistency_policy' not in manifest:
        return []  # Existing per-scene policies still apply to historical plans.
    try:
        expected = digest(RENDER_POLICY)
        if digest(manifest['render_consistency_policy']) != expected:
            return ['Manifest render_consistency_policy must exactly match native-rgb-repeat-v1']
        return [f'{scene}: render_consistency_policy is missing or differs from the dataset manifest'
                for scene, config in configs.items()
                if 'render_consistency_policy' not in config or digest(config['render_consistency_policy']) != expected]
    except (TypeError, ValueError):
        return ['Manifest/config render_consistency_policy is not valid finite JSON']


def render_history_errors(meta, states, names):
    """The new collector must render all ticks, not only the 13 saved RGB ticks."""
    errors = []
    history = meta.get('render_history')
    if not isinstance(history, dict):
        return ['Missing render_history required by render_consistency_policy']
    if history.get('status') != 'complete' or history.get('physics_frames_checked') != 121:
        errors.append('render_history must be complete and check all 121 physics frames')
    tick = history.get('sensor_tick_s')
    if isinstance(tick, bool) or not isinstance(tick, (int, float)) or tick != 0.:
        errors.append('render_history.sensor_tick_s must be zero (render every physics tick)')
    cameras = history.get('cameras')
    expected = names+[semantic_lidar.NAME]
    if (not isinstance(cameras, list) or len(cameras) != 7 or
            any(not isinstance(camera, dict) for camera in cameras) or
            [camera.get('camera') for camera in cameras] != expected):
        return errors+['render_history must contain ordered unique six cameras and LIDAR_TOP']
    for camera in cameras:
        name = camera['camera']
        frames, timestamps = np.asarray(camera.get('frames')), np.asarray(camera.get('timestamps_s'))
        if (frames.shape != (121,) or not np.issubdtype(frames.dtype, np.integer) or
                not np.array_equal(frames, states.get('frame_id'))):
            errors.append(f'render_history.{name}: every sensor frame must match all 121 physical frames')
        if (timestamps.shape != (121,) or not np.issubdtype(timestamps.dtype, np.number) or
                not np.isfinite(timestamps).all() or
                not np.allclose(timestamps, states.get('simulation_time_s'), rtol=0., atol=1e-6)):
            errors.append(f'render_history.{name}: all 121 sensor timestamps must match world timestamps within 1e-6 s')
    return errors


def check_complete(branch_dir, config, branch=None, verify_images=True):
    """Validate file integrity and capture schema; never select on future quality."""
    base = Path(branch_dir)
    errors = []
    branch = branch or base.name
    try:
        if branch not in branch_names(config):
            errors.append('Branch is absent from the frozen scene action set')
        meta = read(base / "meta.json")
        if meta.get("status") != "complete":
            errors.append("meta.status is not complete")
        if meta.get("branch") != branch:
            errors.append("metadata branch differs from expected branch")
        if meta.get("spec_sha256") != config.get("spec_sha256"):
            errors.append("metadata config hash differs from frozen plan")
        if meta.get("code_sha256") != config.get("code_sha256"):
            errors.append("metadata source hashes differ from frozen plan")
        if meta.get("physics_frames") != 121 or meta.get("rgb_samples") != 13:
            errors.append("metadata must declare 121 physics states and 13 RGB samples")
        roles = config["vehicle_roles"]
        if meta.get("vehicle_roles") != roles:
            errors.append("vehicle-role order differs from frozen plan")
        names = [cam["name"] for cam in config["camera_rig"]]
        if len(set(names)) != 6 or meta.get("camera_names") != names:
            errors.append("camera names/order differ from the six-camera plan")
        with np.load(base / "states.npz", allow_pickle=False) as loaded:
            states = {key: loaded[key] for key in loaded.files}
        with np.load(base / "samples.npz", allow_pickle=False) as loaded:
            samples = {key: loaded[key] for key in loaded.files}
        count = len(roles)
        state_shapes = {"time_s": (121,), "frame_id": (121,), "simulation_time_s": (121,),
                        "target_speed": (121, count), "transform_world_vehicle": (121, count, 4, 4),
                        "bbox_extent": (count, 3), "bbox_offset": (count, 3), "bbox_rotation_deg": (count, 3)}
        state_shapes.update({key: (121, count, 3) for key in
                             ("position", "rotation_deg", "velocity", "angular_velocity", "acceleration", "control")})
        sample_shapes = {"time_s": (13,), "physics_indices": (13,), "frame_id": (13,),
                         "sensor_frame_id": (13, 7), "world_to_camera": (13, 6, 4, 4),
                         "camera_to_ego": (6, 4, 4), "intrinsics": (6, 3, 3), "ego_to_world": (13, 4, 4)}
        for label, arrays, shapes in (("states", states, state_shapes), ("samples", samples, sample_shapes)):
            for key, shape in shapes.items():
                if key not in arrays or arrays[key].shape != shape:
                    errors.append(f"{label}.{key} missing or wrong shape; expected {shape}")
                elif not np.isfinite(arrays[key]).all():
                    errors.append(f"{label}.{key} contains non-finite values")
        if not errors:
            if not np.allclose(states["time_s"], np.arange(121)*.05-2, atol=1e-8, rtol=0):
                errors.append("Physics time grid is not -2..4 seconds at 20 Hz")
            if not np.allclose(samples["time_s"], np.arange(13)*.5-2, atol=1e-8, rtol=0):
                errors.append("RGB time grid is not -2..4 seconds at 2 Hz")
            if not np.array_equal(samples["physics_indices"], np.arange(0, 121, 10)):
                errors.append("RGB sample physics indices are incorrect")
            if not np.all(np.diff(states["frame_id"]) == 1):
                errors.append("Physics frame IDs are not consecutive")
            if not np.array_equal(samples["frame_id"], states["frame_id"][::10]):
                errors.append("Sample frame IDs do not match physics frames")
            if not np.all(samples["sensor_frame_id"] == samples["frame_id"][:, None]):
                errors.append("The six cameras/LiDAR are not matched to the saved simulation frame")
            if meta.get("t0_frame") != int(states["frame_id"][40]):
                errors.append("Metadata t0_frame differs from physics index 40")
            if np.any(np.abs(np.diff(states["simulation_time_s"])-.05) > 1e-4):
                errors.append("Simulation timestamps are inconsistent with fixed 20 Hz stepping")
            if 'render_consistency_policy' in config:
                errors.extend(render_history_errors(meta, states, names))
        expected_names = [f"{i:05d}.png" for i in range(13)]
        for camera in config["camera_rig"]:
            folder = base / "camera" / camera["name"]
            actual_names = sorted(path.name for path in folder.glob("*.png"))
            if actual_names != expected_names:
                errors.append(f"{camera['name']}: require exactly 13 expected PNG files")
            elif verify_images:
                for name in expected_names:
                    with Image.open(folder / name) as image:
                        if image.size != (1600, 900) or image.mode != "RGB":
                            errors.append(f"{camera['name']}/{name}: expected 1600 x 900 RGB")
                        image.verify()
        if len(list((base / "camera").glob("*/*.png"))) != 78:
            errors.append("Require exactly 78 six-camera PNG files")
        errors.extend(semantic_lidar.inspect_branch(base, config, meta, states, samples)['errors'])
        diagnostic = read(base / "controller_diagnostics.json")
        if not isinstance(diagnostic, list) or len(diagnostic) != 121:
            errors.append("Controller diagnostics must contain 121 physics records")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"Unreadable/incomplete branch: {type(exc).__name__}: {exc}")
    return errors


def select_scenes(data, explicit=None, start=0, count=None):
    if explicit:
        scenes = []
        for value in explicit:
            for part in value.split(","):
                scenes.append(f"scene_{int(part):04d}" if part.isdigit() else part)
    else:
        available = sorted(path.parent.name for path in data.glob("scene_*/spec.json"))
        scenes = ([scene for scene in available if int(scene.removeprefix("scene_")) >= start]
                  if count is None else [f"scene_{index:04d}" for index in range(start, start+count)])
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("Select at least one scene without duplicates")
    for scene in scenes:
        if Path(scene).name != scene or not scene.startswith("scene_") or not (data / scene / "spec.json").is_file():
            raise ValueError(f"Missing or invalid scene: {scene}")
    return scenes


def save_source_snapshot(out, hashes):
    all_hashes = {**hashes, Path(__file__).name: file_sha(Path(__file__)),
                  RENDER_GATE_SOURCE: file_sha(HERE/RENDER_GATE_SOURCE)}
    folder = out / "source" / digest(all_hashes)[:16]
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in all_hashes.items():
        destination = folder / name
        if destination.exists():
            if file_sha(destination) != expected:
                raise RuntimeError(f"Existing source snapshot is inconsistent: {destination}")
        else:
            shutil.copy2(HERE / name, destination)
    write(folder / "sha256.json", all_hashes)
    return folder


def scene_render_gate(data, out, scene, config, gate_source_sha256):
    """Re-evaluate complete scenes from input bytes, including every resume.

    A partial selection cannot stamp a whole-scene pass. No cached passing report
    is trusted: the auditor hashes every compared input and rehashes on exit.
    """
    if 'render_consistency_policy' not in config:
        return None
    if file_sha(HERE/RENDER_GATE_SOURCE) != gate_source_sha256:
        raise RuntimeError('Render consistency gate source changed during batch')
    required = list(branch_names(config))
    report_path = out/'render_consistency'/f'{scene}.json'
    missing = [name for name in required if not (data/scene/'branches'/name).exists()]
    if missing:
        result = {'status': 'partial', 'scene_id': scene, 'missing_branches': missing,
                  'required_branches': required, 'render_consistency_policy': config['render_consistency_policy'],
                  'audit_source_sha256': gate_source_sha256, 'checked_utc': now(),
                  'note': 'Scene incomplete; no RGB consistency pass was issued.'}
        write(report_path, result)
        return {'status': 'partial', 'report': str(report_path), 'missing_branches': missing}
    invalid = {name: errors for name in required
               if (errors := check_complete(data/scene/'branches'/name, config, name))}
    if invalid:
        result = {'status': 'failed', 'scene_id': scene, 'invalid_branches': invalid,
                  'render_consistency_policy': config['render_consistency_policy'],
                  'audit_source_sha256': gate_source_sha256, 'checked_utc': now()}
    else:
        policy = config['render_consistency_policy']
        result = audit_render_consistency(data, scenes=[scene], thresholds=policy['thresholds'],
                                          tile_size=policy['tile_size_pixels'])
        if file_sha(HERE/RENDER_GATE_SOURCE) != gate_source_sha256:
            result['status'] = 'failed'
            result.setdefault('failures', []).append({'stage': 'source_identity', 'error': 'Gate source changed during audit'})
    write(report_path, result)
    if result['status'] != 'passed':
        raise RuntimeError(f'{scene}: render consistency failed; inspect {report_path}; no subsequent scene will be collected')
    return {'status': 'passed', 'report': str(report_path), 'report_sha256': file_sha(report_path),
            'input_manifest_sha256': result['input_manifest_sha256'],
            'audit_source_sha256': gate_source_sha256, 'checked_utc': now()}


def scene_render_preflight(data, out, scene, config, gate_source_sha256):
    """Fresh same-action check before any intervention; never a delivery gate.

    Missing references remain partial. Failed evidence is preserved and requires
    an explicitly planned replacement scene, never an automatic retry-until-pass.
    """
    if 'render_consistency_policy' not in config:
        return None
    if file_sha(HERE/RENDER_GATE_SOURCE) != gate_source_sha256:
        raise RuntimeError('Render consistency gate source changed during batch')
    path = out/'render_preflight'/f'{scene}.json'
    missing = [name for name in REFERENCE_PAIR if not (data/scene/'branches'/name).exists()]
    result = {'status': 'partial', 'audit_mode': 'repeat_only', 'scene_id': scene,
              'required_branches': list(REFERENCE_PAIR), 'missing_branches': missing,
              'audit_source_sha256': gate_source_sha256, 'checked_utc': now()}
    if not missing:
        invalid = {name: errors for name in REFERENCE_PAIR
                   if (errors := check_complete(data/scene/'branches'/name, config, name))}
        if invalid:
            result.update(status='failed', invalid_branches=invalid)
        else:
            policy = config['render_consistency_policy']
            result = audit_render_consistency(data, scenes=[scene], thresholds=policy['thresholds'],
                                              tile_size=policy['tile_size_pixels'], repeat_only=True)
            if file_sha(HERE/RENDER_GATE_SOURCE) != gate_source_sha256:
                result['status'] = 'failed'
                result.setdefault('failures', []).append({'stage': 'source_identity', 'error': 'Gate source changed during preflight'})
    result['delivery_pass'] = False
    result['action_required'] = {
        'partial': 'collect_missing_reference_and_repeat',
        'passed': 'collect_remaining_actions_then_full_audit',
        'failed': 'replace_scene_and_recollect',
    }[result['status']]
    if result['status'] == 'failed':
        result['replacement_protocol'] = (
            'Preserve rejected evidence. Freeze an explicit new candidate configuration with a different seed '
            'in the same interaction category, then recollect both references before other actions. '
            'Do not select replacements by collision outcomes, interaction strength, or model scores.')
    write(path, result)
    return {'status': result['status'], 'report': str(path), 'report_sha256': file_sha(path),
            'audit_source_sha256': gate_source_sha256,
            'input_manifest_sha256': result.get('input_manifest_sha256'),
            'missing_branches': missing, 'action_required': result['action_required'],
            'delivery_pass': False, 'checked_utc': now()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="System-disk logs, source snapshot, batch progress")
    parser.add_argument("--scenes", nargs="+", help="Explicit IDs or indices, e.g. scene_0000 scene_0004 or 0 4")
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-count", type=int)
    parser.add_argument("--branches", nargs="+", help="Optional explicit subset; defaults to each scene's own feasible actions and repeat")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-branches", type=int,
                        help="Stop successfully after N newly collected, validated and promoted branches; existing skips do not count")
    args = parser.parse_args(argv)
    if args.max_new_branches is not None and args.max_new_branches < 1:
        parser.error("max-new-branches must be positive")
    if args.scene_start < 0 or (args.scene_count is not None and args.scene_count < 1):
        parser.error("scene-start must be nonnegative and scene-count positive")
    if args.scenes and (args.scene_start != 0 or args.scene_count is not None):
        parser.error("Use either explicit --scenes or --scene-start/--scene-count")
    if args.branches is not None and len(set(args.branches)) != len(args.branches):
        parser.error("Branches must not be duplicated")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    data, out = args.data.resolve(), args.out.resolve()
    scenes = select_scenes(data, args.scenes, args.scene_start, args.scene_count)
    hashes = source_hashes()
    configs = {scene: read(data / scene / "spec.json") for scene in scenes}
    dataset_manifest = read(data/'manifest.json') if (data/'manifest.json').is_file() else {}
    policy_errors = validate_manifest_render_policy(dataset_manifest, configs)
    if policy_errors:
        raise RuntimeError('; '.join(policy_errors))
    scene_branches = {}
    for scene, config in configs.items():
        errors = validate_plan(config, hashes)
        available = branch_names(config)
        selected = args.branches if args.branches is not None else list(available)
        if not set(selected).issubset(available):
            errors.append('Requested branch is not feasible in this scene: ' + ', '.join(sorted(set(selected)-set(available))))
        if errors:
            raise RuntimeError(f"{scene}: {'; '.join(errors)}")
        if 'render_consistency_policy' in config:
            selected = [name for name in REFERENCE_PAIR if name in selected] + [
                name for name in selected if name not in REFERENCE_PAIR]
        scene_branches[scene] = selected
    # This lock coordinates wrappers, not the simulator. Calibration must finish
    # before starting collection, as declared in the run protocol.
    lock_name = hashlib.sha256(f"{args.host}:{args.port}".encode()).hexdigest()[:12]
    lock = open(f"/tmp/effect_jepa_multicam_{lock_name}.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another multicamera batch wrapper already owns this server port") from exc
    out.mkdir(parents=True, exist_ok=True)
    snapshot = save_source_snapshot(out, hashes)
    gate_source_sha256 = file_sha(HERE/RENDER_GATE_SOURCE)
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest = {"version": "paired-multicam-batch-v2", "status": "running", "session_id": session,
                "started_utc": now(), "command": sys.argv, "data": str(data), "host": args.host, "port": args.port,
                "scene_ids": scenes, "branches": list(dict.fromkeys(b for values in scene_branches.values() for b in values)),
                "scene_branches": scene_branches, "source_sha256": hashes,
                "source_snapshot": str(snapshot), "spec_sha256": {s: c["spec_sha256"] for s, c in configs.items()},
                "minimum_free_bytes": MIN_FREE_BYTES, "per_branch_timeout_s": BRANCH_TIMEOUT_S,
                "jobs": [], "total_jobs": sum(map(len, scene_branches.values())), "completed_jobs": 0,
                "max_new_branches": args.max_new_branches, "newly_completed_jobs": 0}
    manifest['render_consistency_source_sha256'] = gate_source_sha256
    manifest['render_consistency'] = {}
    manifest['render_preflight'] = {}
    manifest['scene_model_branches'] = {s: list(branch_names(c, include_repeat=False)) for s, c in configs.items()}
    manifest['qa_only_branches'] = ['maintain_repeat']
    manifest['branch_usage_note'] = 'maintain remains a model/reference action; maintain_repeat is QA only. This inventory is not training approval.'
    session_manifest = out / "batch_sessions" / f"{session}.json"

    def save():
        manifest["updated_utc"] = now()
        write(session_manifest, manifest)
        write(out / "batch.json", manifest)

    def preflight(scene, config, require_pass=False):
        if manifest['render_preflight'].get(scene, {}).get('status') == 'passed':
            return  # Only this invocation's fresh audit is reusable within a scene.
        result = scene_render_preflight(data, out, scene, config, gate_source_sha256)
        if result is None:
            return
        manifest['render_preflight'][scene] = result
        save()
        if result['status'] == 'failed':
            manifest.update(action_required=result['action_required'], blocked_scene=scene)
            raise RuntimeError(f'{scene}: RGB repeat preflight failed; replace_scene_and_recollect using an explicit new '
                               f'configuration in the same interaction category; inspect {result["report"]}; '
                               'no intervention or subsequent scene will be collected')
        if require_pass and result['status'] != 'passed':
            manifest.update(action_required=result['action_required'], blocked_scene=scene)
            raise RuntimeError(f'{scene}: collect maintain and maintain_repeat before intervention branches; '
                               f'inspect {result["report"]}; no intervention was started')

    save()
    try:
        for scene in scenes:
            config = configs[scene]
            for branch in scene_branches[scene]:
                if source_hashes() != hashes:
                    raise RuntimeError("Collection source changed during the batch")
                if read(data / scene / "spec.json") != config:
                    raise RuntimeError(f"Frozen config changed during the batch: {scene}")
                if branch not in REFERENCE_PAIR:
                    preflight(scene, config, require_pass=True)
                destination = data / scene / "branches" / branch
                if destination.exists():
                    errors = check_complete(destination, config, branch)
                    if not args.resume:
                        raise FileExistsError(f"Branch already exists; use --resume to validate/skip it: {destination}")
                    if not errors:
                        manifest["jobs"].append({"scene": scene, "branch": branch, "status": "existing_complete", "checked_utc": now()})
                        manifest["completed_jobs"] += 1
                        save()
                        print(f"SKIP {scene}/{branch} ({manifest['completed_jobs']}/{manifest['total_jobs']})", flush=True)
                        if branch in REFERENCE_PAIR:
                            preflight(scene, config)
                        continue
                    rejected = data / "_failed_multicam" / session / scene / branch
                    rejected.parent.mkdir(parents=True, exist_ok=True)
                    destination.rename(rejected)
                    manifest["jobs"].append({"scene": scene, "branch": branch, "status": "invalid_existing_preserved",
                                             "validation_errors": errors, "preserved_at": str(rejected), "time_utc": now()})
                    save()
                free = shutil.disk_usage(data).free
                if free < MIN_FREE_BYTES:
                    raise RuntimeError(f"Data-disk free space {free/1024**3:.2f} GiB is below the 1.5 GiB stop threshold")
                staging = data / "_staging_multicam" / session / scene / branch
                staging.parent.mkdir(parents=True, exist_ok=True)
                log_path = out / "branch_logs" / session / f"{scene}_{branch}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", "-B", str(HERE / "collect_multicam.py"),
                           "--data", str(data), "--scene", scene, "--branch", branch,
                           "--host", args.host, "--port", str(args.port), "--out", str(staging)]
                record = {"scene": scene, "branch": branch, "status": "running", "started_utc": now(),
                          "command": command, "log": str(log_path), "staging": str(staging), "free_bytes_before": free}
                manifest["jobs"].append(record)
                save()
                print(f"START {scene}/{branch} ({manifest['completed_jobs']+1}/{manifest['total_jobs']})", flush=True)
                start = time.monotonic()
                with log_path.open("x") as stream:
                    returncode, timed_out = run_branch_process(command, stream)
                    if timed_out:
                        record["timeout"] = True
                        print(json.dumps({'event': 'branch_wall_timeout', 'scene': scene,
                                          'branch': branch, 'timeout_s': BRANCH_TIMEOUT_S,
                                          'log': str(log_path)}), flush=True)
                record.update(returncode=returncode, elapsed_wall_s=time.monotonic()-start, finished_utc=now())
                errors = check_complete(staging, config, branch)
                if source_hashes() != hashes:
                    errors.append("Collection source changed while this branch ran")
                if read(data / scene / "spec.json") != config:
                    errors.append("Frozen scene config changed while this branch ran")
                if returncode != 0 or errors:
                    record.update(status="failed", validation_errors=errors)
                    save()
                    raise RuntimeError(f"{scene}/{branch} failed or incomplete (exit={returncode}); inspect {log_path}; staging preserved")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise RuntimeError(f"Canonical branch appeared concurrently; refusing replacement: {destination}")
                if os.stat(staging).st_dev != os.stat(destination.parent).st_dev:
                    raise RuntimeError("Staging must be on the same filesystem for atomic promotion")
                staging.rename(destination)
                record.update(status="complete", destination=str(destination), free_bytes_after=shutil.disk_usage(data).free)
                manifest["completed_jobs"] += 1
                manifest["newly_completed_jobs"] += 1
                save()
                print(f"DONE {scene}/{branch}: {record['elapsed_wall_s']:.1f}s ({manifest['completed_jobs']}/{manifest['total_jobs']})", flush=True)
                if branch in REFERENCE_PAIR:
                    preflight(scene, config)
                if (args.max_new_branches is not None and
                        manifest["newly_completed_jobs"] >= args.max_new_branches):
                    gate = scene_render_gate(data, out, scene, config, gate_source_sha256)
                    if gate is not None:
                        manifest['render_consistency'][scene] = gate
                    manifest["status"] = "stopped_after_branch_limit"
                    print(f"STOP after {manifest['newly_completed_jobs']} new branches; batch record: {out / 'batch.json'}", flush=True)
                    return 0
            gate = scene_render_gate(data, out, scene, config, gate_source_sha256)
            if gate is not None:
                manifest['render_consistency'][scene] = gate
                save()
        incomplete = (any(gate['status'] != 'passed' for gate in manifest['render_consistency'].values()) or
                      any(set(scene_branches[s]) != set(branch_names(c)) for s, c in configs.items()))
        manifest["status"] = "partial" if incomplete else "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["finished_utc"] = now()
        save()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    # Keep the declared dataset manifest intact when collecting only a subset.
    print(f"Batch record: {out / 'batch.json'}", flush=True)


if __name__ == "__main__":
    main()
