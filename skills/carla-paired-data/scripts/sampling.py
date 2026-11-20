"""Reproducible coverage scheduling; no outcome-based selection or CARLA calls."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def weather_variants(grid):
    variants = deepcopy(grid.get('weather_variants', [{'id': 'daylight', 'overrides': {}}]))
    if not isinstance(variants, list) or not variants:
        raise ValueError('weather_variants must be a nonempty list')
    ids = []
    ranges = {k: (0., 100.) for k in ('cloudiness', 'precipitation', 'precipitation_deposits',
              'wind_intensity', 'fog_density', 'wetness', 'dust_storm')}
    ranges.update(sun_altitude_angle=(-90., 90.), sun_azimuth_angle=(0., 360.))
    ranges.update({k: (0., math.inf) for k in ('fog_distance', 'fog_falloff',
                  'scattering_intensity', 'mie_scattering_scale', 'rayleigh_scattering_scale')})
    for item in variants:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id']:
            raise ValueError('Each weather variant needs a nonempty id')
        ids.append(item['id'])
        overrides = item.get('overrides', {})
        if not isinstance(overrides, dict) or set(overrides) - ranges.keys():
            raise ValueError('Unknown weather override; use CARLA 0.9.16 WeatherParameters')
        for key, value in overrides.items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not ranges[key][0] <= value <= ranges[key][1]):
                raise ValueError(f'Invalid weather override {key}')
        camera = item.get('camera_overrides', {})
        if not isinstance(camera, dict) or set(camera) - {'exposure_compensation', 'shutter_speed', 'iso', 'fstop'}:
            raise ValueError('Unknown camera exposure override')
        for key, value in camera.items():
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or (key != 'exposure_compensation' and value <= 0)):
                raise ValueError(f'Invalid camera override {key}')
        if item.get('vehicle_lights', 'default') not in ('default', 'low_beam'):
            raise ValueError('vehicle_lights must be default or low_beam')
    if len(set(ids)) != len(ids):
        raise ValueError('weather variant ids must be unique')
    return variants


def assign_weather(jobs, cells, variants, seed):
    """Keep cell quotas untouched, then balance weather counts within each template."""
    lookup = {c['cell_id']: c for c in cells}
    counts, style_counts = defaultdict(Counter), defaultdict(Counter)
    ids = [v['id'] for v in variants]
    for job in jobs:
        template = lookup[job['cell_id']]['template']
        selected = min(ids, key=lambda w: (counts[template][w], style_counts[job['cell_id']][w],
                                           digest([seed, job['scene_id'], w])))
        job['weather_id'] = selected
        counts[template][selected] += 1
        style_counts[job['cell_id']][selected] += 1
    return {t: {w: counts[t][w] for w in ids} for t in sorted(counts)}


def location_id(candidate):
    """A junction or road is a location; connector/lane variations are not new sites."""
    anchor = candidate.get('anchor', {})
    if 'junction_id' in anchor:
        site = ['junction', anchor['junction_id']]
    elif 'road_id' in anchor:
        site = ['road', anchor['road_id']]
    else:
        site = ['anchor', anchor or candidate['candidate_id']]
    return 'location_' + digest([candidate.get('map'), site])[:20]


def location_shortlist(candidates, used, template, seed, failed_ids=(), limit=24):
    """Choose underrepresented locations before weather. Exhaust failed locations fairly."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError(f'No candidates for {template}')
    fresh = [c for c in candidates if c['candidate_id'] not in set(failed_ids)]
    if fresh:
        candidates = fresh
    local = Counter(r['location_id'] for r in used if r.get('template') == template)
    global_sites = Counter(r['location_id'] for r in used)
    maps = Counter(r.get('map') for r in used if r.get('template') == template)
    anchors = Counter(r.get('candidate_id') for r in used if r.get('template') == template)
    def priority(c):
        site = location_id(c)
        return (local[site], global_sites[site], maps[c.get('map')], anchors[c['candidate_id']])
    best = min(map(priority, candidates))
    tied = [c for c in candidates if priority(c) == best]
    return sorted(tied, key=lambda c: digest([seed, c['candidate_id']]))[:limit]


def coverage_row(cfg):
    grid = cfg.get('grid_spec', {})
    return {'template': grid.get('template'), 'candidate_id': grid.get('candidate_id'),
            'map': cfg.get('map'), 'location_id': grid.get('location_id') or location_id({
                'map': cfg.get('map'), 'candidate_id': grid.get('candidate_id'),
                'anchor': cfg.get('geometry', {}).get('selection', {}).get('anchor', {})}),
            'weather_id': cfg.get('weather_variant_id', grid.get('selection', {}).get('weather_id')),
            'base_scene_id': cfg.get('base_scene_id'),
            'requested_interaction_phase':cfg.get('interaction_design',{}).get('requested_phase')}


def coverage_summary(plan, records):
    ids = [v['id'] for v in plan['weather_variants']]
    result = {}
    for template in sorted({c['template'] for c in plan['cells']}):
        rows = [r for r in records if r.get('template') == template]
        counts = Counter(r.get('weather_id') for r in rows)
        result[template] = {'scene_versions': len(rows),
            'requested_interaction_phase_counts':dict(Counter(r.get('requested_interaction_phase','legacy') for r in rows)),
            'phase_counts_are_design_intents_not_observed_outcomes':True,
            'distinct_base_scenes': len({r['base_scene_id'] for r in rows if r.get('base_scene_id')}),
            'distinct_locations': len({r['location_id'] for r in rows if r.get('location_id')}),
            'weather_counts': {w: counts[w] for w in ids},
            'missing_weather_ids': [w for w in ids if not counts[w]],
            'max_weather_count_gap': max((counts[w] for w in ids), default=0) - min((counts[w] for w in ids), default=0)}
    return result
