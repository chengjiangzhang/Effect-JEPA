"""Validate native decal ordering evidence without simulator access."""
import hashlib
import json
import math
from pathlib import Path
import re

POLICY = 'stable_decal_paths_v2'
PRIMARY = ['SortOrder_ascending', 'bHasNormal_descending', 'FinalDecalBlendMode_descending']
TIES = ['material_object_path', 'component_object_path']


def require(value, message):
    if not value:
        raise ValueError('Native decal proof: ' + message)


def validate_header(header):
    require(header == {'kind': 'decal_order_start', 'policy': POLICY, 'target': '0x3bf64f0',
                       'entry_verified': True, 'preserved_primary_order': PRIMARY, 'stable_ties': TIES},
            'native ordering header differs')


def launch_evidence(folder, pid):
    path = Path(folder) / 'weather_rng' / f'decal-order-{pid}.jsonl'
    with path.open() as stream:
        header = json.loads(stream.readline())
    validate_header(header)
    return {'engine_pid': pid, 'trace_path': str(path), 'header': header}


def trace_audit(text, header):
    validate_header(header)
    require(text.endswith('\n'), 'incomplete native trace line')
    rows = [json.loads(line) for line in text.splitlines()]
    require(rows and rows[0] == header and sum(r.get('kind') == 'decal_order_start' for r in rows) == 1,
            'header absent, changed or duplicated')
    require(all(r.get('kind') in {'decal_order_start', 'decal_order_apply', 'decal_order_end'} for r in rows),
            'native error or unknown record')
    calls = [r for r in rows if r.get('kind') == 'decal_order_apply']
    require(len(calls) < 100000 and [r['ordinal'] for r in calls] == list(range(1, len(calls) + 1)),
            'call sequence incomplete or at trace limit')
    for row in calls:
        require(all(type(row[k]) is int and row[k] >= 0 for k in ('actor_number', 'frame', 'stage', 'count', 'moved'))
                and 2 <= row['count'] <= 65536 and row['moved'] <= row['count'], 'invalid count or identity')
        require(type(row['time']) in (int, float) and math.isfinite(row['time']) and row['time'] >= 0,
                'invalid native time')
        require(isinstance(row['actor'], str) and isinstance(row['ordered_input_hash'], str)
                and re.fullmatch('[0-9a-f]{16}', row['ordered_input_hash']) is not None, 'invalid native digest')
    ends = [r for r in rows if r.get('kind') == 'decal_order_end']
    changed = sum(r['moved'] > 0 for r in calls)
    require(len(ends) <= 1 and (not ends or rows[-1] == ends[0]), 'invalid footer position')
    require(all(r['calls'] == len(calls) and r['changed_lists'] == changed for r in ends), 'footer count differs')
    return {'status': 'passed', 'calls': len(calls), 'changed_lists': changed,
            'camera_lists': sum(r['actor'] == 'SceneCaptureCamera' for r in calls),
            'footer_present': bool(ends),
            'zero_call_semantics': 'No visible list with at least two decals was recorded; header proves hook installation, not a changed draw.'}


def snapshot(launch, pid):
    require(launch['engine_pid'] == pid and Path(launch['trace_path']).name == f'decal-order-{pid}.jsonl',
            'trace does not bind the owned process')
    text = Path(launch['trace_path']).read_text()
    return {'launch': launch, 'trace_snapshot': text, 'trace_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'audit': trace_audit(text, launch['header'])}


def verify(value, launch, pid):
    require(value['launch'] == launch and launch['engine_pid'] == pid
            and Path(launch['trace_path']).name == f'decal-order-{pid}.jsonl', 'owned-process identity differs')
    text = value['trace_snapshot']
    require(hashlib.sha256(text.encode()).hexdigest() == value['trace_sha256'], 'snapshot changed')
    require(trace_audit(text, launch['header']) == value['audit'], 'stored native audit differs')
