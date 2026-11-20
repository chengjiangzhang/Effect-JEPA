"""Observe one immutable shared prefix before freezing any intervention plan.

This stage never chooses a future based on its NPC response. Failures return the
whole scene to the ordinary bounded repair loop. Each subsequent branch replays
the same initialization and is checked against the measured prefix.
"""
from __future__ import annotations

from pathlib import Path
import subprocess


def _inputs(attempt, directories):
    from pipeline import file_sha
    return {str(p.relative_to(attempt)): file_sha(p)
            for folder in directories for p in sorted(folder.rglob('*')) if p.is_file()}


def verify_planning_record(attempt):
    from pipeline import read, file_sha, CaptureInfrastructureError
    attempt = Path(attempt)
    record = read(attempt / 'reports/t0_planning.json')
    if record.get('status') != 'passed':
        raise ValueError('Measured t0 planning did not pass: ' + str(record.get('failure_reasons')))
    for relative, expected in record['frozen_files'].items():
        path = (attempt / relative).resolve()
        if (not path.is_relative_to(attempt.resolve()) or not path.is_file()
                or file_sha(path) != expected):
            raise CaptureInfrastructureError('Measured planning input/output changed: ' + relative)
    history = attempt / record['history_directory']
    if not history.is_relative_to(attempt):
        raise CaptureInfrastructureError('Measured history escaped the scene attempt')
    return record


def prepare_measured_plan(attempt, scene, host, port, *, plan, catalog,
                          manager=None, attempt_identity=None):
    from pipeline import read, write, runtime_command, run_command, CaptureInfrastructureError
    from measured_t0 import plan_from_history, HistoryEvidenceError
    import scene_compiler
    attempt = Path(attempt)
    reports = attempt / 'reports'
    record_path = reports / 't0_planning.json'
    if record_path.exists() and read(record_path).get('status') == 'passed':
        return verify_planning_record(attempt)
    if (attempt / 'data/manifest.json').exists():
        raise CaptureInfrastructureError('Final plan exists without a completed measured-t0 binding; inspect interrupted planning')
    initial = attempt / 'initial_plan'
    calibrated = read(initial/scene/'spec.json').get('encounter_calibration')
    if calibrated:
        from encounter_calibration import prepare_history
        initial,history = prepare_history(attempt,scene,host,port,plan,catalog,manager,attempt_identity)
    histories = [] if calibrated else sorted(attempt.glob('common_history_*'))
    history = history if calibrated else next((p for p in reversed(histories)
                    if (p / 'manifest.json').is_file()
                    and read(p / 'manifest.json').get('status') == 'complete'), None)
    if history is None:
        history = attempt / f'common_history_{len(histories):03d}'
        if manager is not None:
            if attempt_identity is None or attempt_identity[0] != scene:
                raise ValueError('Managed common history requires the scene attempt identity')
            epoch = manager.before_attempt(*attempt_identity)
            write(reports / 'server_epochs' / f"{epoch['epoch']:06d}_history.json", epoch)
        try:
            run_command(runtime_command('preflight_physics.py', '--data', initial,
                        '--out', history, '--scenes', scene, '--history-only',
                        '--host', host, '--port', port), reports / 'common_history.log', 600)
        except subprocess.TimeoutExpired as exc:
            raise CaptureInfrastructureError('Common-history collection timed out; preserve scene for recovery') from exc
    spec = read(initial / scene / 'spec.json')
    try:
        result = plan_from_history(spec, history)
    except (HistoryEvidenceError, FileNotFoundError) as exc:
        raise CaptureInfrastructureError('Measured history evidence is missing or inconsistent; preserve scene: ' + str(exc)) from exc
    # Store the complete failure evidence before raising so the next whole-scene
    # repair receives actual t0 observations, not only its original spawn guess.
    evidence = {key: value for key, value in result.items() if key != 'planned_spec'}
    evidence.update(history_directory=str(history.relative_to(attempt)),
                    initial_plan_directory=str(initial.relative_to(attempt)))
    if result['status'] != 'passed':
        write(record_path, evidence)
        raise ValueError('measured_t0_plan_failed: ' + str(result.get('failure_reasons', evidence)))
    scene_compiler.write_plan(attempt / 'data', [result['planned_spec']], plan, catalog=catalog)
    bound = [attempt/'initial_plan',initial,history,attempt/'data']
    if calibrated:
        bound += sorted(attempt.glob('calibrated_plan_*')) + sorted(attempt.glob('common_history_calibration_*'))
    evidence['frozen_files'] = _inputs(attempt,bound)
    if calibrated:
        from pipeline import file_sha
        evidence['frozen_files'].update({str(p.relative_to(attempt)):file_sha(p)
            for p in sorted(reports.glob('encounter_calibration_*.json'))})
    write(record_path, evidence)
    return verify_planning_record(attempt)


def audit_measured_prefix(attempt, data, phase):
    from pipeline import audit_command, runtime_command
    if phase not in ('physics', 'capture'):
        raise ValueError('Unknown measured-prefix audit phase')
    attempt = Path(attempt)
    record = verify_planning_record(attempt)
    report_path = attempt / 'reports' / f'history_{phase}.json'
    result = audit_command(runtime_command('audit_history_prefix.py', '--data', data,
                           '--reference', attempt / record['history_directory'], '--out', report_path),
                           report_path, report_path.with_suffix('.log'))
    if result.get('status') != 'passed':
        raise ValueError(f'measured_common_history_{phase}_mismatch: {result}')
    return result
