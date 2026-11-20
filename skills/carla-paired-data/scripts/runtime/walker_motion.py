"""Compare each executed walker command with native horizontal translation on the same ticks."""
from pathlib import Path
import json

import numpy as np


POLICY = 'native-walker-command-progress-v1'
MIN_COMMAND_M_S = .4
WINDOW_S = .5
MAX_TRAVEL_M = .05


def examine(records, index, role, contacts=(), vehicle_roles=()):
    times = np.asarray(records['time_s'], dtype=float)
    positions = np.asarray(records['position'][:, index, :2], dtype=float)
    issued = np.asarray(records['control'][:-1, index], dtype=float)
    applied = np.asarray(records['observed_control'][1:, index], dtype=float)
    dt = np.diff(times)
    if (not len(dt) or not np.allclose(dt, .05, atol=1e-6, rtol=0)
            or issued.shape != (len(dt), 5) or applied.shape != issued.shape
            or not all(np.isfinite(x).all() for x in (issued, applied, positions))):
        raise ValueError('Invalid walker motion recording')
    # A real collision with an active traffic participant may physically prevent further travel.
    # A fence or other static prop stays in scope. Do not censor other agents after an unrelated crash.
    peers = set(vehicle_roles)
    contacts_at = [float(c['time_s']) for c in contacts
                   if c['time_s'] > 0 and ((c['role'] == role and c['other_role'] in peers)
                       or (c['other_role'] == role and c['role'] in peers))]
    first = min(contacts_at, default=None)
    delta = np.diff(positions, axis=0)
    steps = np.linalg.norm(delta, axis=1)
    def planar(control):
        return control[:, :2] * control[:, 3, None]
    command = planar(issued)
    native = planar(applied)
    speed = np.linalg.norm(native, axis=1)
    active = (np.linalg.norm(command, axis=1) >= MIN_COMMAND_M_S) & (speed >= MIN_COMMAND_M_S)
    aligned = np.linalg.norm(command-native, axis=1) <= 1e-3
    flags = np.zeros(len(dt), dtype=bool)
    frames = round(WINDOW_S / .05)
    for start in range(len(dt)-frames+1):
        end = start+frames
        if first is not None and times[end] >= first-1e-6:
            continue
        if (active[start:end].all() and aligned[start:end].all()
                and steps[start:end].sum() <= MAX_TRAVEL_M):
            # Deliberate reversing within a window is not a sustained command in one direction.
            vectors = native[start:end] / speed[start:end,None]
            if np.min(vectors @ vectors[0]) > .85:
                flags[start:end] = True
    edges = np.flatnonzero(np.diff(np.r_[False, flags, False]))
    bouts = []
    for a,b in edges.reshape(-1,2):
        bouts.append({'start_s':round(float(times[a]), 4), 'end_s':round(float(times[b]), 4),
                      'commanded_m':float(speed[a:b] @ dt[a:b]), 'actual_path_m':float(steps[a:b].sum())})
    return {'role':role, 'first_agent_contact_s':first, 'sustained_stalls':bouts,
            'native_command_mismatches':int(np.count_nonzero(~aligned)),
            't0_or_final_applied_speed_m_s':float(speed[-1]),
            'passed':not bouts and bool(aligned.all())}


def check_scene(root, reports, stage):
    root = Path(root)
    spec = json.loads((root/'spec.json').read_text())
    if not spec.get('pedestrians'):
        return
    import pipeline
    result = {}
    for branch in spec['branches']:
        directory = root/'branches'/branch
        meta = pipeline.read(directory/'meta.json')
        roles = [p['role'] for p in spec['pedestrians']]
        if roles != meta['pedestrian_roles']:
            raise ValueError('Walker role order differs from the specification')
        with np.load(directory/'pedestrians.npz', allow_pickle=False) as data:
            result[branch] = [examine(data, i, role, meta['collisions'], meta['vehicle_roles']) for i,role in enumerate(roles)]
    passed = all(p['passed'] for people in result.values() for p in people)
    pipeline.write(Path(reports)/f'walker_motion_{stage}.json', {'policy':POLICY, 'passed':passed, 'branches':result})
    if not passed:
        raise ValueError(f'walker_motion_not_executed: {stage}: '+str([b for b,people in result.items() if not all(p['passed'] for p in people)]))
