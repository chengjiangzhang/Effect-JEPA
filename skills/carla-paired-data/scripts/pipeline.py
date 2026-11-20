#!/usr/bin/env python3
"""Self-contained CARLA paired-data production: plan, compile, preflight, capture, audit."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import fcntl
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent
PACKAGE = SCRIPTS.parent
RUNTIME = SCRIPTS / 'runtime'
sys.path.insert(0, str(RUNTIME))


class CaptureInfrastructureError(RuntimeError):
    """Infrastructure failures must not become instructions to redesign a scene."""


def capture_timeout_job(reports):
    batch_path = Path(reports) / 'capture/batch.json'
    if not batch_path.is_file():
        return None
    batch = read(batch_path)
    for job in reversed(batch.get('jobs', [])):
        if job.get('status') == 'failed':
            return job if job.get('timeout') is True else None
    return None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(raw, encoding='utf-8')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_identity():
    return {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in (SCRIPTS, PACKAGE / 'assets') for p in sorted(directory.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py', '.json', '.ini', '.c', '.S', '.so'}}


def event(output, kind, **fields):
    item = {'time_utc': datetime.now(timezone.utc).isoformat(), 'event': kind, **fields}
    with (output / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(item, ensure_ascii=False), flush=True)


def configuration(args):
    root = Path(args.carla_root).expanduser().resolve() if args.carla_root else None
    opendrive = Path(args.opendrive).expanduser().resolve() if args.opendrive else (
        root / 'CarlaUE4/Content/Carla/Maps/OpenDrive' if root else None)
    if opendrive is None or not opendrive.is_dir():
        raise ValueError('Supply --carla-root or --opendrive pointing to installed CARLA maps')
    for name in args.maps:
        if Path(name).name != name or not (opendrive / (name + '.xodr')).is_file():
            raise ValueError(f'Map unavailable: {name}')
    model = read(args.model_config) if args.model_config else None
    managed_config = getattr(args, 'managed_server_config', None)
    managed_value = read(managed_config) if managed_config else None
    managed_ini = (Path(managed_value.get('engine_ini', str(RUNTIME / 'resources/engine.ini')))
                   .expanduser().resolve() if managed_value is not None else None)
    # Persist hashes and provider names, never API credentials/configuration bodies.
    dataset_output = getattr(args, 'dataset_output', None) or args.output.with_name(args.output.name + '_dataset')
    dataset_output = Path(dataset_output).expanduser().resolve()
    work_output = args.output.expanduser().resolve()
    if (dataset_output == work_output or dataset_output.is_relative_to(work_output)
            or work_output.is_relative_to(dataset_output)):
        raise ValueError('Dataset delivery and production work directories must be separate')
    return {'schema_version': 3, 'count': args.count, 'seed': args.seed,
            'dataset_output': str(dataset_output),
            'planning_protocol': 'observed-common-history-before-branch-plan-v1',
            'dataset_purpose': getattr(args, 'dataset_purpose', 'train_eval'),
            'max_repairs': args.max_repairs, 'candidates_per_slot': args.candidates_per_slot,
            'max_candidates': args.max_candidates or args.count * args.candidates_per_slot,
            'maps': args.maps, 'opendrive': str(opendrive),
            'map_sha256': {name: hashlib.sha256((opendrive / (name + '.xodr')).read_bytes()).hexdigest()
                           for name in args.maps},
            'host': args.host, 'port': args.port,
            'grid_sha256': hashlib.sha256(args.grid.read_bytes()).hexdigest(),
            'model_config_sha256': fingerprint(model) if model is not None else None,
            'model_provider': model.get('provider') if model is not None else 'deterministic',
            'managed_server_config_sha256': fingerprint(managed_value) if managed_value is not None else None,
            'managed_engine_ini_sha256': hashlib.sha256(managed_ini.read_bytes()).hexdigest() if managed_ini else None,
            'source_identity': source_identity(),
            **({'job_plan_sha256': hashlib.sha256(Path(args.job_plan).read_bytes()).hexdigest()}
               if getattr(args, 'job_plan', None) is not None else {})}


def requested_plan(args, grid, expected_sha256=None):
    """Read explicit jobs or retain the original balanced-plan path unchanged."""
    from plan_grid import build_plan, load_job_plan
    budget = args.max_candidates or args.count * args.candidates_per_slot
    if getattr(args, 'job_plan', None) is not None:
        if type(args.candidates_per_slot) is not int or args.candidates_per_slot < 1:
            raise ValueError('Explicit job-plan requires positive candidates-per-slot')
        budget = (args.max_candidates if args.max_candidates is not None else
                  args.count * args.candidates_per_slot)
        return load_job_plan(args.job_plan, grid, args.count, args.seed, args.max_repairs,
                             budget, expected_sha256=expected_sha256)[0]
    return build_plan(grid, args.count, args.seed, args.max_repairs, budget)


def verify_explicit_resume(plan, state, requested):
    """The frozen plan and slot identities must still match; never import/reset state."""
    if fingerprint(plan) != fingerprint(requested):
        raise ValueError('Stored plan differs from the frozen job-plan')
    slots = state.get('slots') if isinstance(state, dict) else None
    expected = {job['scene_id']: job['cell_id'] for job in plan['jobs']}
    if not isinstance(slots, dict) or set(slots) != set(expected):
        raise ValueError('Resume state slots differ from the frozen job-plan')
    if any(not isinstance(slots[name], dict) or slots[name].get('cell_id') != cell
           for name, cell in expected.items()):
        raise ValueError('Resume state cell identity differs from the frozen job-plan')


def prepare(args):
    from plan_grid import build_plan
    output = args.output.expanduser().resolve()
    if output == PACKAGE or PACKAGE.is_relative_to(output) or output.is_relative_to(PACKAGE):
        raise ValueError('Output must be separate from the installed skill directory')
    cfg = configuration(args)
    explicit = getattr(args, 'job_plan', None) is not None
    grid = read(args.grid) if explicit else None
    requested = requested_plan(args, grid, cfg['job_plan_sha256']) if explicit else None
    identity_path = output / 'pipeline_config.json'
    if identity_path.exists():
        if not args.resume:
            raise FileExistsError('Output already exists; use --resume with unchanged settings')
        if read(identity_path) != cfg:
            raise ValueError('Resume settings/source differ from frozen pipeline_config.json')
        plan = read(output / 'plan.json')
        state = read(output / 'state.json')
        if explicit:
            verify_explicit_resume(plan, state, requested)
        return output, cfg, plan, state
    if output.exists() and any(p.name != '.pipeline.lock' for p in output.iterdir()):
        raise FileExistsError('Output is nonempty and has no pipeline identity')
    output.mkdir(parents=True, exist_ok=True)
    grid = grid if explicit else read(args.grid)
    plan = (requested if explicit else
            build_plan(grid, cfg['count'], cfg['seed'], cfg['max_repairs'], cfg['max_candidates']))
    state = {'schema_version': 2, 'status': 'planned', 'candidate_specs_started': 0,
             'model_calls': 0, 'slots': {j['scene_id']: {'cell_id': j['cell_id'],
                 'status': 'pending', 'attempts': []} for j in plan['jobs']}}
    write(output / 'grid.json', grid)
    write(output / 'plan.json', plan)
    write(identity_path, cfg)
    write(output / 'state.json', state)
    event(output, 'planned', scene_slots=len(plan['jobs']), cells=len(plan['cells']),
          disabled_pairs=plan.get('disabled_pairs', []))
    return output, cfg, plan, state


def stage_satisfied(slot, stage):
    return (slot['status'] == 'accepted'
            or stage == 'compile' and slot['status'] in {'compiled', 'physics_checked'}
            or stage == 'physics' and slot['status'] == 'physics_checked')


def scheduling_scope(args, plan, state):
    """Operational hold only; never edit the scientific plan or consume retries.

    Every invocation explicitly replaces the prior filter. Resuming without
    --defer-weather releases it, while accepted/exhausted state is unchanged.
    """
    deferred = list(getattr(args, 'defer_weather', None) or [])
    available = {v['id'] for v in plan.get('weather_variants', [{'id': 'daylight'}])}
    if any(not isinstance(value, str) for value in deferred) or len(set(deferred)) != len(deferred):
        raise ValueError('--defer-weather requires distinct weather IDs')
    unknown = set(deferred)-available
    if unknown:
        raise ValueError('Unknown deferred weather IDs: ' + ', '.join(sorted(unknown)))
    held, runnable = [], []
    for job in plan['jobs']:
        slot = state['slots'][job['scene_id']]
        if stage_satisfied(slot, args.stage) or slot['status'] == 'exhausted':
            continue
        if job.get('weather_id', 'daylight') in deferred:
            held.append({'scene_id': job['scene_id'], 'cell_id': job['cell_id'],
                         'weather_id': job.get('weather_id', 'daylight')})
        else:
            runnable.append(job['scene_id'])
    return {'stage': args.stage, 'requested_deferred_weather_ids': sorted(deferred),
            'deferred_scene_ids': [job['scene_id'] for job in held],
            'deferred_scene_count': len(held), 'deferred_scenes': held,
            'runnable_scene_ids': runnable, 'runnable_scene_count': len(runnable),
            'scientific_plan_changed': False, 'candidate_budget_consumed_by_deferral': 0,
            'release_semantics': 'Each invocation replaces the scheduling filter; resume without --defer-weather releases it.'}


def preview_scheduling_scope(args):
    """Read-only planning permits an all-held run without CARLA readiness RPC."""
    output = args.output.expanduser().resolve()
    explicit = getattr(args, 'job_plan', None) is not None
    cfg = configuration(args) if explicit else None
    requested = requested_plan(args, read(args.grid), cfg['job_plan_sha256']) if explicit else None
    if (output / 'pipeline_config.json').exists():
        plan, state = read(output / 'plan.json'), read(output / 'state.json')
        if explicit:
            if not args.resume or read(output / 'pipeline_config.json') != cfg:
                raise ValueError('Explicit job-plan resume requires unchanged frozen settings/source')
            verify_explicit_resume(plan, state, requested)
    else:
        if explicit and output.exists() and any(p.name != '.pipeline.lock' for p in output.iterdir()):
            raise FileExistsError('Output is nonempty and has no pipeline identity')
        plan = requested if explicit else requested_plan(args, read(args.grid))
        state = {'slots': {job['scene_id']: {'status': 'pending'} for job in plan['jobs']}}
    return scheduling_scope(args, plan, state)


def record_scheduling_scope(output, args, plan, state):
    scope = scheduling_scope(args, plan, state)
    lookup = {job['scene_id']: job for job in scope['deferred_scenes']}
    for name, slot in state['slots'].items():
        slot.pop('scheduling_deferral', None)
        if name in lookup:
            slot['scheduling_deferral'] = {
                'weather_id': lookup[name]['weather_id'],
                'reason': 'Explicit invocation weather hold; quota, seed and candidate budget retained.'}
    state['invocation_schedule'] = {**scope, 'recorded_utc': datetime.now(timezone.utc).isoformat()}
    write(output / 'state.json', state)
    event(output, 'scheduling_scope', **scope)
    return scope


def run_command(command, log, timeout_s=1800):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('ab') as stream:
        stream.write(('COMMAND ' + json.dumps([str(x) for x in command]) + '\n').encode())
        result = subprocess.run([str(x) for x in command], cwd=RUNTIME,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                                stdout=stream, stderr=subprocess.STDOUT, timeout=timeout_s)
    if result.returncode:
        raise RuntimeError(f'{Path(str(command[2])).name} exited {result.returncode}; see {log.name}')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def seal_delivery(attempt):
    verify_native_delivery_proofs(attempt)
    files = [p for directory in ('data', 'export') for p in (attempt / directory).rglob('*') if p.is_file()]
    files += sorted((attempt / 'reports/server_epochs').glob('*.json'))
    files += sorted((attempt / 'reports/native_weather').rglob('*.json'))
    files += sorted((attempt / 'reports').glob('interaction_coverage_*.json'))
    if any(read(p).get('pedestrians') for p in (attempt/'data').glob('*/spec.json')):
        for stage in ('physics', 'capture'):
            witness = attempt/'reports'/f'walker_motion_{stage}.json'
            if not read(witness)['passed']:
                raise ValueError('Unpassed native walker execution audit: '+str(witness))
            files.append(witness)
    files += [attempt / 'reports' / name for name in
              ('physics.json', 'physics_execution.json', 'execution.json', 'action_execution.json', 'action_selection.json', 'protocol.json', 'render.json', 'equivalence.json')]
    measured = any('measured_t0_plan' in read(p) for p in (attempt / 'data').glob('*/spec.json'))
    if measured or (attempt / 'reports/t0_planning.json').exists():
        from planning_stage import verify_planning_record
        verify_planning_record(attempt)
        files += [attempt / 'reports' / name for name in
                  ('t0_planning.json', 'history_physics.json', 'history_capture.json', 'physics_window.json')]
        files += sorted((attempt/'reports').glob('encounter_calibration_*.json'))
        files += [p for directory in [attempt / 'initial_plan', *sorted(attempt.glob('calibrated_plan_*')), *sorted(attempt.glob('common_history_*'))]
                  for p in directory.rglob('*') if p.is_file()]
    if not files or any(not p.is_file() for p in files):
        raise ValueError('Cannot seal an incomplete delivery')
    entries = {str(p.relative_to(attempt)): {'size': p.stat().st_size, 'sha256': file_sha(p)}
               for p in sorted(files)}
    path = attempt / 'reports/delivery_files.json'
    write(path, {'schema': 'paired-delivery-sha256-v1', 'files': entries})
    return file_sha(path)


def verify_delivery(output, slot):
    attempt = (output / slot['accepted_data']).parent
    inventory = attempt / 'reports/delivery_files.json'
    if not inventory.is_file() or file_sha(inventory) != slot.get('delivery_inventory_sha256'):
        raise ValueError('Delivery inventory is missing or changed')
    for relative, expected in read(inventory)['files'].items():
        path = (attempt / relative).resolve()
        if not path.is_relative_to(attempt.resolve()):
            raise ValueError('Delivery inventory escapes its attempt directory')
        if not path.is_file() or path.stat().st_size != expected['size'] or file_sha(path) != expected['sha256']:
            raise ValueError(f'Delivery file is missing or changed: {relative}')


def diagnostic(report):
    """Prioritize actual failures over verbose manifests in model repair context."""
    keys = ('status', 'error', 'errors', 'summary', 'protocol_issues')
    result = {k: report[k] for k in keys if k in report}
    if 'failures' in report:
        result['failure_count'] = len(report['failures'])
        result['failures'] = report['failures'][:3]
    scenes = report.get('scenes', {})
    if isinstance(scenes, dict):
        failed = {}
        for name, row in scenes.items():
            branches = {branch: {k: value[k] for k in ('status', 'error',
                'initialization_collisions', 'precontact_dynamics_violations', 'execution_pass', 'failure_reason', 'observed_outcome') if k in value}
                for branch, value in row.get('branches', {}).items()
                if value.get('passed') is False or value.get('execution_pass') is False or value.get('status', '').startswith('failed')}
            replay = row.get('history_and_repeat')
            if branches or (isinstance(replay, dict) and replay.get('passed') is False):
                failed[name] = {'branches': branches}
                if isinstance(replay, dict) and replay.get('passed') is False:
                    failed[name]['history_and_repeat'] = replay
        if failed:
            result['failed_scenes'] = failed
    return result or report


def audit_command(command, report, log):
    """Preserve structured diagnostics even when an auditor exits nonzero."""
    try:
        run_command(command, log)
    except RuntimeError as exc:
        if Path(report).is_file():
            detail = diagnostic(read(report))
            raise ValueError(f'{Path(report).stem}_audit_failed: ' +
                             json.dumps(detail, ensure_ascii=False)[:12000]) from exc
        raise
    return read(report)


def runtime_command(script, *values):
    return [sys.executable, '-B', RUNTIME / script, *values]


def choice_schema(candidates, weather_ids):
    return {'type': 'object', 'additionalProperties': False,
            'required': ['candidate_id', 'layout_seed', 'speed_scale', 'weather_id'],
            'properties': {
                'candidate_id': {'type': 'string', 'enum': [x['candidate_id'] for x in candidates]},
                'layout_seed': {'type': 'integer', 'minimum': 0, 'maximum': 2**31-1},
                'speed_scale': {'type': 'number', 'minimum': .85, 'maximum': 1.15},
                'weather_id': {'type': 'string', 'enum': weather_ids}}}


def current_coverage(output, state, exclude_scene=None):
    """One latest successful attempt per slot, reconstructible on resume."""
    from sampling import coverage_row
    used = []
    for name, slot in state['slots'].items():
        if name == exclude_scene or slot['status'] in {'exhausted', 'invalid_delivery'}:
            continue
        for entry in reversed(slot['attempts']):
            path = output / entry['path'] / 'data' / name / 'spec.json'
            if not path.exists():
                path = output / entry['path'] / 'initial_plan' / name / 'spec.json'
            if entry['status'] in {'compiled', 'physics_checked', 'accepted'} and path.exists():
                used.append({**coverage_row(read(path)), 'stage': entry['status']})
                break
    return used


def select_spec(client, catalog, job, cell, plan, repair, previous, output, state):
    import scene_compiler
    from sampling import location_shortlist
    weather_ids = [v['id'] for v in plan.get('weather_variants', [{'id': 'daylight'}])]
    # Freeze the planned weather for every repair/replacement of this slot.
    assigned_weather = job.get('weather_id', weather_ids[job['component_seeds']['weather'] % len(weather_ids)])
    weather_ids = [assigned_weather]
    candidates = catalog.public(cell['template'])
    if hasattr(scene_compiler, 'eligible_candidates'):
        eligible = {c['candidate_id'] for c in scene_compiler.eligible_candidates(catalog, cell)}
        candidates = [c for c in candidates if c['candidate_id'] in eligible]
    from walker_spawn import feasible_sites
    candidates = feasible_sites(candidates, catalog, job, cell, repair, model_selects_seed=client is not None)
    failed_ids = [(p.get('previous_selection') or {}).get('candidate_id') for p in previous]
    candidates = location_shortlist(candidates, current_coverage(output, state, job['scene_id']),
                                    cell['template'], [job['seed'], repair], failed_ids)
    if not candidates:
        raise ValueError('No map-grounded candidates for this grid cell')
    if client is None:
        return {'candidate_id': candidates[0]['candidate_id'], 'weather_id': assigned_weather}, {
            'provider': 'deterministic', 'model_invoked': False}
    request = {'task': 'Select one map-grounded scene for paired ego-action simulation; return only the schema JSON.',
               'cell': cell, 'scene_seed': job['seed'], 'attempt_index': repair,
               'catalog_candidates': candidates, 'weather_ids': weather_ids,
               'previous_attempts': previous[-3:],
               'rules': ['Select a listed candidate_id; do not invent coordinates.',
                         'All ego branches share this one layout and NPC policy.',
                         'Fix initialization/implementation failures, not valid action collision outcomes.',
                         'Candidates prioritize underused feasible locations; weather is preassigned within the template and cannot be changed.',
                         'Retain the assigned interaction, topology, and NPC style.']}
    try:
        result = client.request_json(request, schema=choice_schema(candidates, weather_ids))
    finally:
        state['model_calls'] = client.calls_used
        write(output / 'state.json', state)
    selection = dict(result.value)
    if selection.get('weather_id') != assigned_weather or selection.get('candidate_id') not in {c['candidate_id'] for c in candidates}:
        raise ValueError('Selection violates the location shortlist or frozen template-balanced weather assignment')
    from scene_spec import authoring_context, authoring_schema
    base = scene_compiler.compile_one(job, cell, catalog, repair, selection)
    request = {'task':'Author the complete scene initial conditions, NPC policies, fork and ego action plans.',
               'cell':cell, 'attempt_index':repair, 'previous_attempts':previous[-3:],
               **authoring_context(base)}
    try:
        authored = client.request_json(request, schema=authoring_schema(base))
    finally:
        state['model_calls'] = client.calls_used
        write(output / 'state.json', state)
    for old in previous:
        prior = old.get('previous_selection') or {}
        if prior.get('candidate_id') == selection['candidate_id'] and 'scene' in prior:
            if set(prior['scene']['unrelated_roles']) != set(authored.value['unrelated_roles']):
                raise ValueError('Repair must preserve unrelated role declarations for the same candidate; fix positions/policies or replace the candidate')
    selection['scene'] = authored.value
    return selection, {'catalog_selection':result.provenance, 'scene_authoring':authored.provenance}


def latest_physics(attempt):
    paths = sorted(attempt.glob('physics_*'))
    for path in reversed(paths):
        if (path / 'manifest.json').exists():
            manifest = read(path / 'manifest.json')
            if all((path / scene / 'branches' / branch / 'meta.json').exists() and
                   read(path / scene / 'branches' / branch / 'meta.json').get('status') == 'complete'
                   for scene in manifest['scene_ids']
                   for branch in manifest['requested_scene_branches'][scene]):
                return path
    return None


def physics_stage(attempt, scene, host, port, manager=None):
    data, reports = attempt / 'data', attempt / 'reports'
    physics = latest_physics(attempt)
    generated = physics is None
    if physics is None:
        physics = attempt / f"physics_{len(list(attempt.glob('physics_*'))):03d}"
        run_command(runtime_command('preflight_physics.py', '--data', data, '--out', physics,
                                    '--host', host, '--port', port), reports / 'physics.log')
    if manager is not None:
        native_physics_proof(attempt, scene, physics, manager if generated else None)
    from planning_stage import audit_measured_prefix
    audit_measured_prefix(attempt, physics, 'physics')
    from runtime.walker_motion import check_scene as check_walker_motion
    check_walker_motion(physics / scene, reports, 'physics')
    from runtime.interaction_coverage import scene_report
    write(reports / 'interaction_coverage_physics.json',scene_report(physics / scene))
    result = audit_command(runtime_command('audit_preflight_progress.py', '--data', physics,
                                '--out', reports / 'physics.json'), reports / 'physics.json',
                                reports / 'physics_audit.log')
    if result['status'] != 'passed_quick_checks':
        raise ValueError('physics_preflight_failed: ' + json.dumps(result['summary']['failed_checks']))
    # Execution correctness is separate from observed interaction outcomes.
    execution = audit_command(runtime_command('audit_execution.py', '--data', physics,
                         '--out', reports / 'physics_execution.json'),
                         reports / 'physics_execution.json', reports / 'physics_execution.log')
    if execution.get('status') != 'passed':
        raise ValueError('execution_failed: ' + json.dumps(diagnostic(execution), ensure_ascii=False)[:12000])
    # This is observed execution after planning, never an input to the planner
    # or an NPC-reaction threshold for choosing scenes.
    from measured_t0 import summarize_observed_window
    window = summarize_observed_window(read(data / scene / 'spec.json'), physics / scene,
                                       execution_report=execution)
    write(reports / 'physics_window.json', window)
    # Retain every raw branch. New scenes must pass all ten action pairs before RGB.
    selection = audit_command(runtime_command('audit_realized_actions.py', '--data', physics,
                              '--out', reports / 'action_execution.json'),
                              reports / 'action_execution.json', reports / 'action_execution.log')
    if selection.get('status') != 'passed':
        raise ValueError('realized_action_audit_failed: ' + json.dumps(selection, ensure_ascii=False)[:12000])
    return physics


def collect_capture_pass(data, reports, host, port, max_new_branches=None):
    extra = ['--max-new-branches', max_new_branches] if max_new_branches is not None else []
    try:
        run_command(runtime_command('collect_multicam_batch.py', '--data', data, '--out', reports / 'capture',
                                    '--host', host, '--port', port, '--resume', *extra),
                    reports / 'capture.log', 4200)
    except subprocess.TimeoutExpired as exc:
        raise CaptureInfrastructureError('Capture stage exceeded its bounded wall-time budget') from exc
    except RuntimeError as exc:
        timeout_job = capture_timeout_job(reports)
        if timeout_job is not None:
            raise CaptureInfrastructureError('capture_branch_timeout: ' + json.dumps(timeout_job)) from exc
        details = {}
        for report_path in sorted((reports / 'capture').glob('**/*.json')):
            if report_path.name == 'batch.json' or 'render_preflight' in report_path.parts or 'render_consistency' in report_path.parts:
                report = read(report_path)
                if report.get('status') not in {'passed', 'complete'}:
                    details[str(report_path.relative_to(reports))] = diagnostic(report)
        if details:
            raise ValueError('capture_failed: ' + json.dumps(details, ensure_ascii=False)[:12000]) from exc
        raise


def native_physics_proof(attempt, scene, physics, manager=None):
    """Fresh preflight: manager required; resume: verify the prior sealed proof."""
    from native_weather_proof import branch_binding, sha, verify_proof
    from managed_server import ManagedServerError
    spec = attempt/'data'/scene/'spec.json'
    from controllers_v2 import branch_names
    binding = {'phase': 'physics', 'scene': scene, 'physics_manifest_sha256': sha(physics/'manifest.json'),
               'branches': [branch_binding(spec, physics/scene/'branches'/name, 'physics')
                            for name in branch_names(read(spec))]}
    path = attempt/'reports/native_weather/physics.json'
    try:
        if manager is not None:
            if path.exists():
                raise ValueError('Refusing to overwrite the original physics epoch proof')
            write(path, manager.weather_proof(False, binding))
        verify_proof(read(path), binding, False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManagedServerError(f'Physics native weather proof failed: {exc}') from exc


def verify_native_branch_proofs(attempt, scene, names):
    from native_weather_proof import branch_binding, rain_required, verify_proof
    from managed_server import ManagedServerError
    spec_path = attempt / 'data' / scene / 'spec.json'
    required = rain_required(read(spec_path))
    try:
        for name in names:
            folder = attempt / 'data' / scene / 'branches' / name
            path = attempt / 'reports/native_weather' / f'{name}.json'
            verify_proof(read(path), branch_binding(spec_path, folder, 'rgb'), required)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManagedServerError(f'Missing or invalid saved native weather proof for {scene}: {exc}') from exc


def save_native_branch_proof(attempt, scene, branch, manager):
    from native_weather_proof import branch_binding, rain_required
    from managed_server import ManagedServerError
    spec_path = attempt / 'data' / scene / 'spec.json'
    folder = attempt / 'data' / scene / 'branches' / branch
    path = attempt / 'reports/native_weather' / f'{branch}.json'
    if path.exists():
        raise ManagedServerError('Refusing to overwrite a native branch proof: ' + str(path))
    proof = manager.weather_proof(rain_required(read(spec_path)), branch_binding(spec_path, folder, 'rgb'))
    write(path, proof)


def verify_native_delivery_proofs(attempt, physics=None):
    from controllers_v2 import branch_names
    from managed_server import ManagedServerError
    try:
        physics = physics if physics is not None else latest_physics(attempt)
        if physics is None:
            raise ValueError('Complete physics output required for native proof verification')
        for scene in read(attempt/'data/manifest.json')['scene_ids']:
            native_physics_proof(attempt, scene, physics)
            verify_native_branch_proofs(attempt, scene, branch_names(read(attempt/'data'/scene/'spec.json')))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManagedServerError(f'Native delivery proof verification failed: {exc}') from exc


def capture_stage(attempt, physics, host, port, manager=None, attempt_identity=None):
    from audit_realized_actions import normalize_motion_report
    data, reports = attempt / 'data', attempt / 'reports'
    if manager is None:
        collect_capture_pass(data, reports, host, port)
    else:
        # Fresh UE process per branch resets renderer/weather process state.
        # The batch wrapper still validates/skips complete branches and runs the
        # unchanged reference-pair gate before any intervention is collected.
        names = read(data / 'manifest.json')['scene_ids']
        if len(names) != 1 or attempt_identity is None:
            raise ValueError('Managed capture requires one scene and an attempt identity')
        scene, candidate, repair = attempt_identity
        if names != [scene]:
            raise ValueError('Managed capture scene identity mismatch')
        from controllers_v2 import branch_names
        branches = branch_names(read(data / scene / 'spec.json'))
        branch_count = len(branches)
        existing = [name for name in branches if (data/scene/'branches'/name/'meta.json').is_file()
                    and read(data/scene/'branches'/name/'meta.json').get('status') == 'complete']
        # A completed output without its actual epoch proof is not retrospectively
        # attributed to a new server. Preserve it and stop for explicit recovery.
        verify_native_branch_proofs(attempt, scene, existing)
        previous_complete = -1
        for part in range(branch_count + 1):
            epoch = manager.before_attempt(scene, candidate, repair)
            write(reports / 'server_epochs' / f"{epoch['epoch']:06d}_rgb.json", epoch)
            collect_capture_pass(data, reports, host, port, max_new_branches=1)
            batch = read(reports / 'capture/batch.json')
            new = [row for row in batch.get('jobs', []) if row.get('status') == 'complete']
            if len(new) > 1 or any(row.get('scene') != scene or row.get('branch') not in branches for row in new):
                raise CaptureInfrastructureError('Managed epoch did not capture one declared branch')
            for row in new:
                save_native_branch_proof(attempt, scene, row['branch'], manager)
            complete = batch.get('completed_jobs', 0)
            if complete == branch_count and batch.get('total_jobs') == branch_count:
                break
            if complete <= previous_complete or batch.get('status') != 'stopped_after_branch_limit':
                raise CaptureInfrastructureError('Managed capture made no completed-branch progress')
            previous_complete = complete
        else:
            raise CaptureInfrastructureError('Managed capture exceeded its finite branch budget')
    verify_native_delivery_proofs(attempt, physics)
    from planning_stage import audit_measured_prefix
    audit_measured_prefix(attempt, data, 'capture')
    from runtime.walker_motion import check_scene as check_walker_motion
    for scene_name in read(data / 'manifest.json')['scene_ids']:
        check_walker_motion(data / scene_name, reports, 'capture')
    for script, args, name in [
        ('audit_protocol.py', ['--data', data], 'protocol'),
        ('audit_render_consistency.py', ['--data', data], 'render'),
        ('audit_capture_equivalence.py', ['--data', data, '--reference', physics], 'equivalence')]:
        result = audit_command(runtime_command(script, *args, '--out', reports / f'{name}.json'),
                               reports / f'{name}.json', reports / f'{name}.log')
        passed = (result.get('complete') is True and result.get('equivalent_exact') is True
                  and result.get('status') == 'equivalent_exact') if name == 'equivalence' else result.get('status') == 'passed'
        if not passed:
            raise ValueError(f'{name}_audit_not_passed: ' + json.dumps(diagnostic(result), ensure_ascii=False)[:12000])
    execution = audit_command(runtime_command('audit_execution.py', '--data', data,
                         '--out', reports / 'execution.json'),
                         reports / 'execution.json', reports / 'execution.log')
    if execution.get('status') != 'passed':
        raise ValueError('execution_failed: ' + json.dumps(diagnostic(execution), ensure_ascii=False)[:12000])
    motion_audit = audit_command(runtime_command('audit_realized_actions.py', '--data', data,
                              '--out', reports / 'action_selection.json'),
                              reports / 'action_selection.json', reports / 'action_selection.log')
    if motion_audit.get('status') != 'passed':
        raise ValueError('captured_action_audit_failed: ' + json.dumps(motion_audit, ensure_ascii=False)[:12000])
    preliminary = read(reports / 'action_execution.json')
    def numerical_motion_audit(report):
        rows = {}
        for name, row in report['scenes'].items():
            normalized = normalize_motion_report(row)
            rows[name] = {key: normalized[key] for key in ('motion_audit_valid', 'model_branches',
                          'duplicate_clusters', 'tolerances', 'pairwise_motion_comparisons')}
            if 'legacy_branch_selection' in normalized:
                rows[name]['legacy_branch_selection'] = normalized['legacy_branch_selection']
            if 'action_separation' in normalized:
                rows[name]['action_separation'] = normalized['action_separation']
        return rows
    if numerical_motion_audit(motion_audit) != numerical_motion_audit(preliminary):
        raise ValueError('Captured motion diagnostics differ from physics preflight')
    from runtime.interaction_coverage import scene_report
    write(reports/'interaction_coverage_capture.json',
          {'scenes':{name:scene_report(data/name) for name in motion_audit['scenes']}})
    if not (attempt / 'export/manifest.json').exists():
        run_command(runtime_command('export_common.py', '--data', data, '--out', attempt / 'export'),
                    reports / 'export.log')
    exported = read(attempt / 'export/manifest.json')
    if (exported.get('schema') != 'paired-common-frame-v1' or
            exported.get('scene_ids') != read(data / 'manifest.json')['scene_ids'] or
            not (attempt / 'export/index.html').is_file()):
        raise ValueError('Common-frame export/review is incomplete')
    expected_branches = {name: normalize_motion_report(row)['model_branches'] for name, row in motion_audit['scenes'].items()}
    if exported.get('scene_model_branches') != expected_branches:
        raise ValueError('Common-frame export does not match audited model branches')
    return data


def motion_diagnostics_summary(report):
    """Normalize report/state compatibility once at the delivery boundary."""
    from audit_realized_actions import normalize_motion_report
    normalized = normalize_motion_report(report)
    keys = ('model_branches', 'motion_audit_valid', 'distinct_motion_count',
            'equivalent_motion_count', 'legacy_branch_selection', 'action_separation')
    return {key: normalized[key] for key in keys if key in normalized}


def delivery(output, plan, state):
    cells = {c['cell_id']: {'target': c['target_count'], 'compiled': 0, 'physics_checked': 0,
                           'accepted': 0, 'exhausted': 0, 'invalid_delivery': 0, 'pending': 0,
                           'deferred': 0, 'attempts': 0} for c in plan['cells']}
    accepted, rows = [], []
    for name, slot in state['slots'].items():
        row = cells[slot['cell_id']]
        row['attempts'] += len(slot['attempts'])
        if slot['status'] in row:
            row[slot['status']] += 1
        if slot.get('scheduling_deferral'):
            row['deferred'] += 1
        if slot['status'] == 'accepted':
            accepted.append({'scene_id': name, 'cell_id': slot['cell_id'], 'data': slot['accepted_data'],
                             **motion_diagnostics_summary(slot), 'qa_only_branches': ['maintain_repeat'],
                             'response_metrics':slot.get('response_metrics',{}),
                             'execution_pass':slot.get('execution_pass'),
                             'censored_branches':slot.get('censored_branches',[]),
                             **slot.get('coverage', {}),
                             'export': str(Path(slot['accepted_data']).parent / 'export'),
                             'review': str(Path(slot['accepted_data']).parent / 'export/index.html'),
                             'reports': str(Path(slot['accepted_data']).parent / 'reports'),
                             'delivery_inventory_sha256': slot['delivery_inventory_sha256']})
        link = html.escape(str(Path(slot['accepted_data']).parent / 'export/index.html')
                           if slot.get('accepted_data') else '', quote=True)
        label = slot['status']
        if slot.get('scheduling_deferral'):
            label += ' (deferred: ' + slot['scheduling_deferral']['weather_id'] + ')'
        rows.append(f'<tr><td>{html.escape(name)}</td><td>{html.escape(slot["cell_id"])}</td>'
                    f'<td>{html.escape(label)}</td><td><a href="{link}">{link}</a></td></tr>')
    report = {'schema_version': 2, 'status': state['status'], 'target_scene_count': plan['count'],
              'accepted_scene_count': len(accepted), 'cells': cells, 'accepted_scenes': accepted,
              'disabled_pairs': plan.get('disabled_pairs', []), 'excluded_pairs': plan.get('excluded_pairs', []),
              'selection_unit': 'whole scene including its action branches',
              'qa_only_branches': ['maintain_repeat'], 'model_calls': state['model_calls']}
    held = [{'scene_id': name, 'cell_id': slot['cell_id'], **slot['scheduling_deferral']}
            for name, slot in state['slots'].items() if slot.get('scheduling_deferral')]
    report.update(deferred_scene_count=len(held), deferred_scenes=held,
                  deferred_weather_ids=sorted({row['weather_id'] for row in held}),
                  invocation_schedule=state.get('invocation_schedule'),
                  remaining_scene_count=plan['count']-len(accepted))
    from sampling import coverage_summary
    coverage = current_coverage(output, state)
    report['coverage'] = {
        'policy': plan.get('coverage_policy'),
        'planned_weather_by_template': plan.get('planned_weather_by_template', {}),
        'compiled_or_better_by_template': coverage_summary(plan, coverage),
        'accepted_by_template': coverage_summary(plan, [r for r in coverage if r['stage'] == 'accepted']),
        'count_unit': 'captured versions and distinct physical bases are separate; same base weather variants share one split',
        'location_definition': 'map+junction_id or map+road_id; connector/lane changes within one site do not count as new locations'}
    write(output / 'dataset_manifest.json', report)
    (output / 'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>CARLA paired data</title><style>body{font:16px sans-serif;margin:32px}td,th{padding:8px;border:1px solid #ddd}'
        'table{border-collapse:collapse}</style><h1>CARLA 成对数据</h1>'
        f'<p>状态：{html.escape(state["status"])}；已验收 {len(accepted)} / {plan["count"]} 景；暂缓 {len(held)} 景，仍计入目标。</p>'
        '<p><a href="dataset_manifest.json">分布与验收清单</a> · <a href="events.jsonl">执行日志</a></p>'
        '<table><tr><th>Scene</th><th>Grid cell</th><th>Status</th><th>Data</th></tr>'
        + ''.join(rows) + '</table></html>', encoding='utf-8')
    return report


def native_server_readiness(host, port, requested_maps):
    """Read-only RPC checks; OpenDRIVE files alone do not install rendered maps."""
    import carla
    connection = carla.Client(host, port)
    connection.set_timeout(15.)
    versions = {'client': connection.get_client_version(), 'server': connection.get_server_version()}
    if any(not version.startswith('0.9.16') for version in versions.values()):
        raise ValueError(f'CARLA 0.9.16 required: {versions}')
    available = sorted(connection.get_available_maps())
    # Town07_Opt does not satisfy a request for the non-layered Town07.
    available_names = {Path(name).name for name in available}
    missing = sorted(set(requested_maps) - available_names)
    if missing:
        raise ValueError('CARLA server missing rendered maps: ' + ', '.join(missing) +
                         '; installed OpenDRIVE is insufficient. Install the matching CARLA 0.9.16 map assets and restart the server. Available: ' +
                         ', '.join(sorted(available_names)))
    return {**versions, 'requested_maps': list(requested_maps), 'available_maps': available}


def _execute(args, manager=None):
    # Fail before freezing a new batch or consuming any candidate/repair budget.
    preview = preview_scheduling_scope(args)
    native = (native_server_readiness(args.host, args.port, args.maps)
              if args.command == 'run' and args.stage != 'compile' and manager is None
              and preview['runnable_scene_count'] else None)
    output, cfg, plan, state = prepare(args)
    for name, slot in state['slots'].items():
        if slot['status'] in {'accepted', 'invalid_delivery'}:
            try:
                verify_delivery(output, slot)
                slot['status'] = 'accepted'
            except (ValueError, OSError, KeyError) as exc:
                slot['status'] = 'invalid_delivery'
                state['status'] = 'failed_delivery_integrity'
                write(output / 'state.json', state)
                delivery(output, plan, state)
                raise ValueError(f'{name}: {exc}') from exc
    scope = record_scheduling_scope(output, args, plan, state)
    if args.command == 'plan':
        write(output / 'state.json', state)
        delivery(output, plan, state)
        return 0
    if not scope['runnable_scene_count']:
        return finish_stage(output, cfg, plan, state, args.stage)
    import scene_compiler
    cells = {c['cell_id']: c for c in plan['cells']}
    templates = {cells[j['cell_id']]['template'] for j in plan['jobs']}
    catalog = scene_compiler.build_catalog(cfg['opendrive'], cfg['maps'], templates)
    eligibility = []
    if hasattr(scene_compiler,'candidate_eligibility'):
        for key in sorted({j['cell_id'] for j in plan['jobs']}):
            cell=cells[key]
            if catalog.candidates.get(cell['template']):
                eligibility.append(scene_compiler.candidate_eligibility(catalog,cell)[1])
    write(output / 'catalog.json', {'candidates': catalog.public(), 'map_sha256': catalog.hashes,
                                    'exclusions': catalog.exclusions,'cell_eligibility':eligibility})
    missing = [t for t in templates if not catalog.candidates[t]]
    if missing:
        raise ValueError('Selected maps have no candidates for: ' + ', '.join(sorted(missing)))
    walker_templates = {cells[j['cell_id']]['template'] for j in plan['jobs']
                        if j['scene_id'] in scope['runnable_scene_ids']
                        and cells[j['cell_id']]['template'].startswith('pedestrian_')}
    if args.stage != 'compile' and walker_templates:
        from walker_spawn import probe_candidates
        probe_scene = next(j['scene_id'] for j in plan['jobs'] if j['scene_id'] in scope['runnable_scene_ids']
                           and cells[j['cell_id']]['template'] in walker_templates)
        catalog.walker_spawns = probe_candidates(catalog, walker_templates, cfg['host'], cfg['port'], output, manager, probe_scene)
    client = None
    if args.model_config:
        from model_api import ModelClient
        model_config = read(args.model_config)
        model_config['calls_used'] = state['model_calls']
        client = ModelClient(model_config)
    if native is not None:
        write(output / 'versions.json', native)
    state['status'] = 'running'
    write(output / 'state.json', state)
    for original_job in plan['jobs']:
        name = original_job['scene_id']
        slot, cell = state['slots'][name], cells[original_job['cell_id']]
        if stage_satisfied(slot, args.stage) or slot.get('scheduling_deferral'):
            continue
        if slot['status'] == 'exhausted':
            continue
        done = False
        from interaction_design import freeze_job_strata
        frozen_job = freeze_job_strata(original_job,cell['template'])
        for candidate in range(cfg['candidates_per_slot']):
            job = dict(frozen_job)
            if candidate:
                job['seed'] = int(fingerprint([original_job['seed'], candidate])[:8], 16) % 2**31
            for repair in range(cfg['max_repairs'] + 1):
                rel = Path('attempts') / name / f'candidate_{candidate:02d}' / f'attempt_{repair:02d}'
                attempt = output / rel
                entry = next((a for a in slot['attempts'] if a['path'] == str(rel)), None)
                if entry and entry['status'] == 'failed':
                    continue
                if entry is None:
                    if repair == 0:
                        if state['candidate_specs_started'] >= cfg['max_candidates']:
                            break
                        state['candidate_specs_started'] += 1
                    entry = {'path': str(rel), 'candidate_index': candidate, 'attempt_index': repair,
                             'seed': job['seed'], 'status': 'started'}
                    slot['attempts'].append(entry)
                    attempt.mkdir(parents=True, exist_ok=True)
                    write(output / 'state.json', state)
                try:
                    if not (attempt / 'initial_plan/manifest.json').exists():
                        if (attempt / 'selection.json').exists():
                            selection = read(attempt / 'selection.json')['selection']
                        else:
                            selection, provenance = select_spec(client, catalog, job, cell, plan, repair,
                                [{'attempt_index': a['attempt_index'], 'reason': a.get('reason', ''),
                                  'previous_selection':read(output / a['path'] / 'selection.json')['selection']
                                  if (output / a['path'] / 'selection.json').exists() else None}
                                 for a in slot['attempts'] if a['status'] == 'failed'], output, state)
                            write(attempt / 'selection.json', {'selection': selection, 'model': provenance})
                        from scene_randomization import intent
                        chosen = scene_compiler.validate_selection(selection,job,cell,catalog,repair)
                        candidate_row = next(c for c in catalog.candidates[cell['template']] if c['candidate_id']==chosen['candidate_id'])
                        proposal = {'intent':intent(job,cell,chosen),'selection':chosen,'map':candidate_row['map']}
                        if (attempt/'sampling_proposal.json').exists():
                            if read(attempt/'sampling_proposal.json')!=proposal:
                                raise ValueError('Frozen sampling proposal changed across resume')
                        else:
                            write(attempt/'sampling_proposal.json',proposal)
                        scene_cfg = scene_compiler.compile_one(job, cell, catalog, repair, selection)
                        scene_compiler.write_plan(attempt / 'initial_plan', [scene_cfg], plan, catalog=catalog)
                        event(output, 'compiled', scene=name, candidate=candidate, repair=repair)
                    entry['status'] = 'compiled'
                    entry.setdefault('completed_stages',{})['compiled']=True
                    if args.stage == 'compile':
                        slot['status'] = 'compiled'; done = True
                    else:
                        if cell['template'].startswith('pedestrian_') and getattr(catalog, 'walker_spawns', None) is not None:
                            from walker_spawn import check_spec
                            witnesses = check_spec(read(attempt / 'initial_plan' / name / 'spec.json'), catalog.walker_spawns)
                            write(attempt / 'reports' / 'walker_spawn_checks.json', witnesses)
                        from planning_stage import prepare_measured_plan
                        prepare_measured_plan(attempt, name, cfg['host'], cfg['port'], plan=plan,
                                              catalog=catalog, manager=manager,
                                              attempt_identity=(name, candidate, repair))
                        event(output, 'measured_t0_planned', scene=name, candidate=candidate, repair=repair)
                        if manager is not None:
                            epoch = manager.before_attempt(name, candidate, repair)
                            write(attempt / 'reports/server_epochs' / f"{epoch['epoch']:06d}_physics.json", epoch)
                            event(output, 'server_ready', scene=name, candidate=candidate,
                                  repair=repair, server=epoch)
                        physics = (physics_stage(attempt, name, cfg['host'], cfg['port'], manager)
                                   if manager is not None else physics_stage(attempt, name, cfg['host'], cfg['port']))
                        entry['status'] = 'physics_checked'
                        entry['completed_stages']['physics_checked']=True
                        write(output / 'state.json', state)
                        event(output, 'physics_checked', scene=name, candidate=candidate, repair=repair)
                        if args.stage == 'physics':
                            slot['status'] = 'physics_checked'; done = True
                        else:
                            if manager is None:
                                data = capture_stage(attempt, physics, cfg['host'], cfg['port'])
                            else:
                                data = capture_stage(attempt, physics, cfg['host'], cfg['port'],
                                                     manager, (name, candidate, repair))
                            final_cfg = read(data / name / 'spec.json')
                            slot['delivery_inventory_sha256'] = seal_delivery(attempt)
                            from sampling import coverage_row
                            slot['coverage'] = coverage_row(final_cfg)
                            entry['status'] = 'accepted'; slot['status'] = 'accepted'
                            slot['accepted_data'] = str(data.relative_to(output))
                            motion_audit = read(attempt / 'reports/action_selection.json')['scenes'][name]
                            slot.update(motion_diagnostics_summary(motion_audit))
                            execution = read(attempt / 'reports/execution.json')['scenes'][name]
                            slot['response_metrics'] = execution['response_metrics']
                            slot['execution_pass'] = execution['execution_pass']
                            slot['censored_branches'] = execution.get('censored_branches',[])
                            event(output, 'accepted', scene=name, data=slot['accepted_data'])
                            done = True
                    write(output / 'state.json', state)
                    if done:
                        break
                except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                    from managed_server import ManagedServerError
                    if isinstance(exc, (ManagedServerError, CaptureInfrastructureError)):
                        entry['status'] = 'interrupted'; entry['reason'] = str(exc)
                        state['status'] = 'blocked_infrastructure'
                        write(output / 'state.json', state)
                        delivery(output, plan, state)
                        event(output, 'infrastructure_interrupted', scene=name, reason=str(exc))
                        raise
                    if client is not None:
                        from model_api import ModelError, SchemaError
                        if isinstance(exc, ModelError) and not isinstance(exc, SchemaError):
                            entry['status'] = 'interrupted'; entry['reason'] = str(exc)
                            state['status'] = 'blocked_model_connection'
                            write(output / 'state.json', state)
                            delivery(output, plan, state)
                            raise
                    entry['status'] = 'failed'; entry['reason'] = str(exc)
                    write(attempt / 'failure.json', entry)
                    event(output, 'candidate_attempt_failed', scene=name, candidate=candidate,
                          repair=repair, reason=str(exc))
                    write(output / 'state.json', state)
            if done:
                break
        if not done:
            slot['status'] = 'exhausted'
        write(output / 'state.json', state)
        delivery(output, plan, state)
    return finish_stage(output, cfg, plan, state, args.stage)


def finish_stage(output, cfg, plan, state, stage):
    """The whole frozen quota determines completion, including deferred jobs."""
    wanted = {'compile': 'compiled', 'physics': 'physics_checked', 'all': 'accepted'}[stage]
    reached = sum(s['status'] in {wanted, 'accepted'} or (wanted == 'compiled' and s['status'] == 'physics_checked')
                  for s in state['slots'].values())
    state['status'] = ('complete' if stage == 'all' else wanted) if reached == plan['count'] else 'incomplete'
    write(output / 'state.json', state)
    report = delivery(output, plan, state)
    from randomization_report import build_report
    write(output/'randomization_report.json',build_report(output,plan,state))
    if stage == 'all' and report['accepted_scene_count']:
        from curate_dataset import curate_dataset
        curated = curate_dataset(output / 'dataset_manifest.json', output / 'curation', seed=cfg['seed'],
                                 dataset_purpose=cfg['dataset_purpose'])
        event(output, 'curated', artifact_path='curation', paired_scenes=report['accepted_scene_count'])
        if curated and curated['retained_scene_count']:
            from publish_dataset import publish_dataset
            published = publish_dataset(output / 'dataset_manifest.json', Path(cfg['dataset_output']),
                                        curation_manifest=output / 'curation/curation_manifest.json')
            event(output, 'dataset_published', dataset_output=cfg['dataset_output'], publication=published)
    event(output, 'finished', stage=stage, status=state['status'], reached=reached,
          target=plan['count'], accepted=report['accepted_scene_count'],
          deferred=report['deferred_scene_count'], remaining=report['remaining_scene_count'])
    return 0 if reached == plan['count'] else 3


def execute(args):
    # Explicit inputs must be rejected before any lock/output or managed CARLA side effects.
    if getattr(args, 'job_plan', None) is not None:
        preview_scheduling_scope(args)
    output = args.output.expanduser().resolve()
    if output == PACKAGE or PACKAGE.is_relative_to(output) or output.is_relative_to(PACKAGE):
        raise ValueError('Output must be separate from the installed skill directory')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.pipeline.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another pipeline process owns this output directory') from exc
        managed = getattr(args, 'managed_server_config', None)
        if managed and args.command == 'run' and args.stage != 'compile':
            from managed_server import ManagedServer
            context = ManagedServer(managed, output, args.maps, args.host, args.port)
        else:
            context = nullcontext(None)
        with context as manager:
            return _execute(args, manager)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('plan', 'run'):
        p = sub.add_parser(command)
        p.add_argument('--output', type=Path, required=True, help='Production work and retry records, separate from final data')
        p.add_argument('--dataset-output', type=Path,
                       help='Clean final scene directory; defaults to the sibling <output>_dataset')
        p.add_argument('--carla-root', default=os.environ.get('CARLA_ROOT'))
        p.add_argument('--opendrive')
        p.add_argument('--maps', nargs='+', default=['Town04', 'Town05', 'Town07', 'Town10HD'])
        p.add_argument('--grid', type=Path, default=PACKAGE / 'assets/grid.example.json')
        p.add_argument('--job-plan', type=Path,
                       help='Planning-only jobs for a new cohort; freeze file SHA256 at creation and require it on resume')
        p.add_argument('--count', type=int, default=200)
        p.add_argument('--dataset-purpose', choices=['train_eval', 'evaluation'], default='train_eval',
                       help='evaluation: all retained scenes are held for evaluation; exact-identity curation without fitting thresholds')
        p.add_argument('--seed', type=int, default=20261028)
        p.add_argument('--max-repairs', type=int, choices=range(4), default=3)
        p.add_argument('--candidates-per-slot', type=int, default=2)
        p.add_argument('--max-candidates', type=int)
        p.add_argument('--model-config', type=Path)
        p.add_argument('--managed-server-config', type=Path,
                       help='Optional local CARLA lifecycle config; recycle the owned server before each attempt')
        p.add_argument('--host', default='127.0.0.1')
        p.add_argument('--port', type=int, default=2000)
        p.add_argument('--resume', action='store_true')
        p.add_argument('--defer-weather', nargs='+', default=[], metavar='ID',
                       help='Scheduling-only weather hold; preserve all scene quotas and retry budgets. Omit on resume to release.')
        p.add_argument('--stage', choices=['compile', 'physics', 'all'], default='all')
    p = sub.add_parser('inspect')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('doctor')
    p.add_argument('--carla-root', default=os.environ.get('CARLA_ROOT'))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--maps', nargs='+', default=['Town04', 'Town05', 'Town07', 'Town10HD'])
    p.add_argument('--connect', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            result = {'python': sys.version.split()[0], 'skill': str(PACKAGE),
                      'packages': {m: importlib.util.find_spec(m) is not None for m in ('carla', 'numpy', 'PIL')}}
            if args.carla_root:
                root = Path(args.carla_root)
                result['carla_executable'] = (root / 'CarlaUE4.sh').is_file()
                result['opendrive'] = (root / 'CarlaUE4/Content/Carla/Maps/OpenDrive').is_dir()
            if args.connect:
                native = native_server_readiness(args.host, args.port, args.maps)
                result.update(client_version=native['client'], server_version=native['server'],
                              requested_maps=native['requested_maps'], available_maps=native['available_maps'])
            print(json.dumps(result, indent=2))
            return 0 if all(result['packages'].values()) and result.get('carla_executable', True) else 2
        if args.command == 'inspect':
            print(json.dumps(read(args.output / 'dataset_manifest.json'), ensure_ascii=False, indent=2))
            return 0
        if args.candidates_per_slot < 1:
            raise ValueError('candidates-per-slot must be positive')
        return execute(args)
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        print(f'Pipeline stopped: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
