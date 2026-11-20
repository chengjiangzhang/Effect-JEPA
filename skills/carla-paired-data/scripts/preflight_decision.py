#!/usr/bin/env python3
"""Apply retention/repair policy to evidence-backed physics checks; does not audit CARLA data."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

REQUIRED_CHECKS = ('geometry', 'initialization', 'history_state', 'repeat_state',
                   'action_execution', 'record_integrity')
EXIT_CODES = {'ready_for_rgb': 0, 'needs_evidence': 2, 'repair_or_replace': 3, 'reject_candidate': 4}


def decide(record, max_repairs=3):
    if record.get('schema_version') != 1:
        raise ValueError('Require schema_version=1')
    if not isinstance(record.get('scene_id'), str) or not record['scene_id']:
        raise ValueError('scene_id is required')
    digest = record.get('spec_sha256')
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError('spec_sha256 must identify the exact candidate spec')
    attempt = record.get('attempt_index')
    if type(attempt) is not int or attempt < 0 or type(max_repairs) is not int or max_repairs < 0:
        raise ValueError('attempt_index and max_repairs must be nonnegative integers')
    if attempt > max_repairs:
        raise ValueError('attempt_index exceeds the declared repair budget')
    expected = record.get('branches_expected')
    checked = record.get('branches_checked')
    for values in (expected, checked):
        if (not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values)
                or len(set(values)) != len(values)):
            raise ValueError('Branch lists must contain distinct strings')
    if not {'maintain', 'maintain_repeat'}.issubset(expected):
        raise ValueError('Expected branch list must include reference and QA repeat')
    if not set(checked).issubset(expected):
        raise ValueError('Checked branches include an undeclared branch')
    checks = record.get('checks')
    if not isinstance(checks, dict):
        raise ValueError('checks must be an object')
    unknown, failed = [], []
    if set(expected) != set(checked):
        unknown.append('branch_coverage')
    for name in REQUIRED_CHECKS:
        check = checks.get(name, {'status': 'unknown'})
        status = check.get('status')
        if status not in ('pass', 'fail', 'unknown'):
            raise ValueError(f'{name}: invalid check status')
        evidence = check.get('evidence')
        if status == 'unknown' or not isinstance(evidence, str) or not evidence.strip():
            unknown.append(name)
        elif status == 'fail':
            failed.append(name)
    events = record.get('collisions')
    if events is None:
        unknown.append('collision_records')
        events = []
    if not isinstance(events, list):
        raise ValueError('collisions must be a list, empty only when audited absent')
    initial, future = [], []
    for event in events:
        t = event.get('time_s')
        if type(t) not in (int, float) or not math.isfinite(t):
            raise ValueError('Each collision requires finite relative time_s')
        if event.get('branch') not in expected:
            raise ValueError('Each collision must name a declared branch')
        if not event.get('evidence'):
            unknown.append('collision_event_evidence')
        (initial if t <= 0 else future).append(event)
    if initial:
        failed.append('initialization_collision')
    if failed:
        decision = 'repair_or_replace' if attempt < max_repairs else 'reject_candidate'
    elif unknown:
        decision = 'needs_evidence'
    else:
        decision = 'ready_for_rgb'
    return {
        'schema_version': 1, 'scene_id': record['scene_id'], 'spec_sha256': digest,
        'attempt_index': attempt, 'max_repairs': max_repairs,
        'remaining_repairs': max_repairs - attempt, 'decision': decision,
        'failed_checks': sorted(set(failed)), 'missing_evidence': sorted(set(unknown)),
        'initialization_collisions': initial, 'post_intervention_collisions': future,
        'observations': record.get('observations', {}),
        'scope': 'Policy decision from supplied checks; no independent physics/RGB audit performed.',
        'retention_rule': 'Post-intervention collisions and weak reactions alone do not reject a candidate.',
        'render_quality_approved': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--max-repairs', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True, help='New decision JSON; refuses overwrite')
    args = parser.parse_args()
    try:
        raw = args.record.read_bytes()
        result = decide(json.loads(raw), args.max_repairs)
        result['record_sha256'] = hashlib.sha256(raw).hexdigest()
        result['decision_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        parser.exit(64, f'Invalid record/output: {exc}\n')
    print(f"{result['decision']}: {args.out}")
    raise SystemExit(EXIT_CODES[result['decision']])


if __name__ == '__main__':
    main()
