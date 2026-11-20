"""Check mapped walker capsules in an empty CARLA world before choosing a layout."""
from collections import defaultdict
from pathlib import Path
import json
import subprocess
import sys

import numpy as np


def crossing_path(endpoints, direction):
    start, end = (np.asarray(endpoints[i]['position'], dtype=float) for i in (direction, 1-direction))
    curb = min(.75, max(0., float(endpoints[direction].get('width_m', 0.)) / 2 - .35))
    start = start + (end-start) / np.linalg.norm(end-start) * curb
    points = np.linspace(start, end, max(2, int(np.linalg.norm(end-start) / .5) + 1))
    yaw = float(np.degrees(np.arctan2(end[1]-start[1], end[0]-start[0])))
    return points.tolist(), {'position': (start + [0., 0., .8]).tolist(),
                             'rotation_deg': [0., 0., yaw]}, curb


def direction_for(endpoints, origin, prefer_far):
    return int(bool(prefer_far) and
               np.linalg.norm(np.subtract(endpoints[1]['position'][:2], origin[:2])) <= 5.)


def blueprint_for(seed):
    from interaction_geometry import WALKER_BLUEPRINT_CANDIDATES
    from scene_randomization import stream
    return str(stream(seed, 'walker_blueprint').choice(WALKER_BLUEPRINT_CANDIDATES))


def _probe_map(map_name, host, port, request, out):
    from pipeline import write, CaptureInfrastructureError
    from runtime import collect_multicam as capture
    import carla
    client = carla.Client(host, port, 2)
    client.set_timeout(90.)
    versions = dict(server_version=client.get_server_version(), client_version=client.get_client_version())
    if not all(v.split('+')[0].split('-')[0] == '0.9.16' for v in versions.values()):
        raise CaptureInfrastructureError(f'Walker spawn probe requires CARLA 0.9.16: {versions}')
    world = client.load_world(map_name, reset_settings=False)
    capture.configure_world(world)
    if world.get_map().name.rsplit('/', 1)[-1] != map_name:
        raise CaptureInfrastructureError(f'Walker probe loaded the wrong world for {map_name}')
    library = world.get_blueprint_library()
    rows = []
    for row in request['placements']:
        bp = library.find(row['blueprint_id'])
        bp.set_attribute('is_invincible', 'false')
        try:
            actor = world.spawn_actor(bp, capture.transform(row['transform']))
        except RuntimeError as exc:
            if 'collision at spawn position' not in str(exc).lower():
                raise CaptureInfrastructureError(f'Walker spawn probe did not complete: {exc}') from exc
            spawned = False
        else:
            response, = client.apply_batch_sync([carla.command.DestroyActor(actor.id)], False)
            if response.has_error():
                raise CaptureInfrastructureError(f'Failed to remove walker probe actor: {response.error}')
            spawned = True
        rows.append({**row, 'spawned': spawned})
    write(out, {'map': map_name, 'opendrive_sha256': request['opendrive_sha256'],
                **versions, 'observations': rows})


def probe_candidates(catalog, templates, host, port, output, manager, scene):
    from interaction_geometry import WALKER_BLUEPRINT_CANDIDATES
    from pipeline import read, file_sha, CaptureInfrastructureError
    by_map = defaultdict(list)
    for template in sorted(templates):
        for candidate in catalog.candidates[template]:
            by_map[candidate['map']].append(candidate)
    base = Path(output) / 'walker_spawn_checks'
    base.mkdir(exist_ok=True)
    numbers = [int(p.name[6:]) for p in base.glob('probe_[0-9]*') if p.name[6:].isdigit()]
    directory = base / f'probe_{max(numbers, default=-1)+1:03d}'
    directory.mkdir()
    index = {}
    for n, (map_name, sites) in enumerate(sorted(by_map.items())):
        if manager is not None:
            manager.before_attempt(scene, n, 0)
        placements = []
        for site in sites:
            for direction in (0, 1):
                _, pose, _ = crossing_path(site['pedestrian_endpoints'], direction)
                for blueprint_id in WALKER_BLUEPRINT_CANDIDATES:
                    placements.append({'candidate_id': site['candidate_id'], 'map': map_name,
                                       'direction': direction, 'blueprint_id': blueprint_id, 'transform': pose})
        path = directory / (map_name + '.json')
        request = {'placements': placements, 'opendrive_sha256': catalog.hashes[map_name]}
        # Isolate the native RPC clients so map/worker lifetimes cannot span server epochs.
        with path.with_suffix('.log').open('xb') as log:
            try:
                result = subprocess.run([sys.executable, '-B', __file__, map_name, host, str(port), str(path)],
                        input=json.dumps(request).encode(), stdout=log, stderr=subprocess.STDOUT, timeout=600)
            except subprocess.TimeoutExpired as exc:
                raise CaptureInfrastructureError(f'Walker spawn probe timed out on {map_name}') from exc
        if result.returncode:
            raise CaptureInfrastructureError(f'Walker spawn probe exited {result.returncode}: {path.with_suffix(".log")}')
        rows = read(path)['observations']
        binding = {'file': str(path.relative_to(output)), 'sha256': file_sha(path)}
        for row in rows:
            index[row['candidate_id'], row['direction'], row['blueprint_id']] = {**row, **binding}
    return index


def feasible_sites(candidates, catalog, job, cell, repair, model_selects_seed=False):
    from scene_compiler import validate_selection
    from scene_randomization import intent
    index = getattr(catalog, 'walker_spawns', None)
    if index is None or not cell['template'].startswith('pedestrian_'):
        return candidates
    if model_selects_seed:
        # The model supplies its own layout seed in the next step. Restrict
        # locations here, then check its exact authored pose before native replay.
        viable = {site for (site, _, _), row in index.items() if row['spawned']}
        return [candidate for candidate in candidates if candidate['candidate_id'] in viable]
    sites = {c['candidate_id']: c for c in catalog.candidates[cell['template']]}
    chosen = validate_selection(None, job, cell, catalog, repair)
    preferred = intent(job, cell, chosen)['participants']['preferred_pedestrian_direction']
    bp = blueprint_for(chosen['layout_seed'])
    result = []
    for entry in candidates:
        site = sites[entry['candidate_id']]
        direction = direction_for(site['pedestrian_endpoints'], site['origin'], preferred)
        if index[site['candidate_id'], direction, bp]['spawned']:
            result.append(entry)
    return result


def check_spec(spec, index):
    if not spec.get('pedestrians') or index is None:
        return
    witnesses = []
    for p in spec['pedestrians']:
        key = spec['grid_spec']['candidate_id'], p['crossing_direction_index'], p['blueprint_id']
        row = index.get(key)
        if row is None or not row['spawned'] or any(not np.allclose(
                p['transform'][field], row['transform'][field], rtol=0, atol=1e-6)
                for field in ('position', 'rotation_deg')):
            raise ValueError(f'Walker spawn pose is outside the natively verified set: {key}')
        witnesses.append({k: row[k] for k in ('candidate_id', 'direction', 'blueprint_id', 'transform', 'file', 'sha256')})
    return witnesses


if __name__ == '__main__':
    map_name, host, port, path = sys.argv[1:]
    _probe_map(map_name, host, int(port), json.load(sys.stdin), Path(path))
