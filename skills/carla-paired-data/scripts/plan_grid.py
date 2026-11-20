#!/usr/bin/env python3
"""Allocate reproducible balanced scene jobs for the portable OpenDRIVE compiler."""
import argparse
import hashlib
import json
import platform
import random
import re
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from sampling import assign_weather, weather_variants


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def positive_int(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f'{label} must be a positive integer')
    return value


def identifiers(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f'{label} must be a nonempty list')
    ids = [x.get('id') for x in values]
    if any(not isinstance(x, str) or not x or '__' in x for x in ids):
        raise ValueError(f'{label} needs nonempty ids without double underscores')
    if len(set(ids)) != len(ids):
        raise ValueError(f'{label} ids must be unique')
    return ids


def build_plan(grid, count, seed, max_repairs=3, max_candidates=None):
    if grid.get('schema_version') not in (1, 2):
        raise ValueError('Require grid schema_version=1 or 2')
    positive_int(count, 'count')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if type(max_repairs) is not int or not 0 <= max_repairs <= 3:
        raise ValueError('max_repairs must be an integer in [0,3]')
    max_candidates = count * 2 if max_candidates is None else max_candidates
    positive_int(max_candidates, 'max_candidates')
    if max_candidates < count:
        raise ValueError('max_candidates must cover the initial slot count')
    styles = grid['npc_styles']
    variants = weather_variants(grid)
    identifiers(styles, 'npc_styles')
    pairs = grid['pairs']
    identifiers(pairs, 'pairs')
    cells, excluded, disabled = [], [], []
    seen_pairs = set()
    for pair in pairs:
        key = (pair.get('interaction'), pair.get('topology'))
        if any(not isinstance(v, str) or not v for v in key) or key in seen_pairs:
            raise ValueError('Each interaction/topology pair must be nonempty and unique')
        seen_pairs.add(key)
        if pair.get('semantic_status') == 'excluded':
            if not pair.get('reason'):
                raise ValueError('An excluded pair needs a reason')
            excluded.append(pair)
            continue
        if pair.get('semantic_status') != 'candidate':
            raise ValueError('semantic_status must be candidate or excluded')
        actions = pair.get('candidate_actions')
        if (not isinstance(actions, list) or not actions
                or any(not isinstance(a, str) or not a for a in actions)
                or len(actions) != len(set(actions)) or 'maintain' not in actions
                or 'maintain_repeat' in actions):
            raise ValueError('Use distinct candidate actions including maintain, excluding QA repeat')
        if not pair.get('implementation_status'):
            raise ValueError('Each pair needs an explicit implementation_status')
        if pair.get('enabled', True) is not True:
            if not pair.get('reason'):
                raise ValueError('A disabled pair needs a reason')
            disabled.append(pair)
            continue
        if grid.get('schema_version') == 2 and pair['implementation_status'] != 'implemented':
            raise ValueError('Enabled templates require implementation_status=implemented')
        for style in styles:
            if not style.get('implementation_status'):
                raise ValueError('Each NPC style needs an explicit implementation_status')
            cells.append({
                'cell_id': f"{pair['id']}__{style['id']}",
                'pair_id': pair['id'], 'template': pair.get('template', pair['id']),
                'interaction': pair['interaction'], 'topology': pair['topology'],
                'npc_style': style['id'], 'npc_style_parameters': style.get('parameters', {}),
                'candidate_actions': actions,
                'interaction_implementation': pair['implementation_status'],
                'style_implementation': style['implementation_status'],
                'geometry_validated': False,
            })
    if not cells:
        raise ValueError('No candidate cells remain')
    cells.sort(key=lambda x: x['cell_id'])
    rng = random.Random(seed)
    base, extra = divmod(count, len(cells))
    # At small budgets cover interaction templates before repeating NPC styles.
    by_template = defaultdict(list)
    for i, cell in enumerate(cells):
        by_template[cell['template']].append(i)
    for indices in by_template.values():
        rng.shuffle(indices)
    extra_order = []
    while any(by_template.values()):
        templates = [t for t, indices in by_template.items() if indices]
        rng.shuffle(templates)
        extra_order.extend(by_template[t].pop() for t in templates)
    extra_set = set(extra_order[:extra])
    for i, cell in enumerate(cells):
        cell['target_count'] = base + (i in extra_set)
    slots = [cell['cell_id'] for cell in cells for _ in range(cell['target_count'])]
    rng.shuffle(slots)
    jobs, used_seeds = [], set()
    for i, cell_id in enumerate(slots):
        scene_seed = rng.randrange(2**31)
        while scene_seed in used_seeds:
            scene_seed = rng.randrange(2**31)
        used_seeds.add(scene_seed)
        jobs.append({
            'scene_id': f'scene_{i:04d}', 'cell_id': cell_id, 'seed': scene_seed,
            'component_seeds': {name: int(canonical_hash([scene_seed, name])[:8], 16) % 2**31
                                for name in ('location', 'layout', 'weather', 'npc')},
            'stage': 'proposal', 'executable_config': None, 'attempt_index': 0,
        })
    weather_counts = assign_weather(jobs, cells, variants, seed)
    from interaction_design import assign_phases
    assign_phases(jobs,cells,seed)
    return {
        'schema_version': 1, 'status': 'planning_only', 'execution_ready': False,
        'count': count, 'base_seed': seed, 'grid_sha256': canonical_hash(grid),
        'python_version': platform.python_version(),
        'max_repairs_per_candidate': max_repairs,
        'max_candidate_specs_in_batch': max_candidates,
        'candidate_budget_note': 'Finite engineering budget, configurable; not a completed-scene count.',
        'balance_scope': 'Eligible declared cells, not necessarily equal axis marginals; map feasibility unchecked.',
        'seed_scope': 'Sample each scene once; all its branches reuse the frozen seeds and layout.',
        'randomization_policy': 'joint-scene-randomization-v1: prospective named streams, bounded participant combinations, continuous progress and stable individual policies; proposed versus qualified distributions reported',
        'qa_only_branches': ['maintain_repeat'],
        'cells': cells, 'excluded_pairs': excluded, 'disabled_pairs': disabled, 'jobs': jobs,
        'weather_variants': variants,
        'coverage_policy': 'interaction/style quota first; least-used feasible locations; balanced weather within template; weather frozen through repairs',
        'planned_weather_by_template': weather_counts,
        'limitations': ['No CARLA or map lookup performed.',
                        'Declared NPC profiles are frozen parameters; physical feasibility requires compilation and rollout.',
                        'Do not pass this file to collect_multicam.py as a scene config.'],
    }


def planned_weather_counts(jobs, cells, variants):
    """Count frozen weather assignments; never resample a subset's weather."""
    lookup = {cell['cell_id']: cell for cell in cells}
    counts = {}
    for job in jobs:
        template = lookup[job['cell_id']]['template']
        counts.setdefault(template, {variant['id']: 0 for variant in variants})
        counts[template][job['weather_id']] += 1
    return {template: counts[template] for template in sorted(counts)}


def validate_job_plan(plan, grid, count, seed, max_repairs=3, max_candidates=None):
    """Validate planning-only jobs for a NEW cohort, not imported execution state.

    A subset retains original scene IDs, seed components and weather. The caller
    supplies the new finite budget; cross-cohort budget/acceptance provenance must
    be checked externally. No old attempts, outcomes or accepted state are loaded.
    """
    declared = build_plan(grid, count, seed, max_repairs, max_candidates)
    if not isinstance(plan, dict):
        raise ValueError('job-plan must be a planning-only object')
    # Round-trip rejects non-JSON values/nonfinite numbers for in-memory callers.
    json.dumps(plan, allow_nan=False)
    optional = {'planner_sha256', 'parent_plan_sha256'}
    if set(plan) - set(declared) - optional or set(declared) - set(plan):
        raise ValueError('job-plan has missing or unknown fields; execution state is forbidden')
    for key in optional & set(plan):
        if not isinstance(plan[key], str) or not re.fullmatch(r'[0-9a-f]{64}', plan[key]):
            raise ValueError(f'job-plan {key} must be a SHA256 digest')
    for key in ('schema_version', 'count', 'base_seed', 'max_repairs_per_candidate',
                'max_candidate_specs_in_batch'):
        if type(plan[key]) is not int or plan[key] != declared[key]:
            raise ValueError(f'job-plan {key} differs from requested configuration')
    if plan['status'] != 'planning_only' or plan['execution_ready'] is not False:
        raise ValueError('job-plan must be planning_only and not execution_ready')
    if not isinstance(plan['python_version'], str) or not plan['python_version']:
        raise ValueError('job-plan python_version must identify the planner runtime')
    flexible = {'cells', 'jobs', 'planned_weather_by_template', 'python_version'} | optional
    for key in set(declared) - flexible:
        if canonical_hash(plan[key]) != canonical_hash(declared[key]):
            raise ValueError(f'job-plan {key} differs from grid/planning contract')
    cells = plan['cells']
    expected_cells = {cell['cell_id']: cell for cell in declared['cells']}
    if not isinstance(cells, list) or len(cells) != len(expected_cells):
        raise ValueError('job-plan must retain every declared cell, including zero targets')
    actual_cells = {}
    for cell in cells:
        if not isinstance(cell, dict) or not isinstance(cell.get('cell_id'), str):
            raise ValueError('job-plan cell needs a declared cell_id')
        name = cell['cell_id']
        if name in actual_cells or name not in expected_cells:
            raise ValueError('job-plan cell IDs must be unique and match the grid')
        expected = expected_cells[name]
        if set(cell) != set(expected) or any(canonical_hash(cell[key]) != canonical_hash(expected[key])
                for key in expected if key != 'target_count'):
            raise ValueError(f'job-plan cell {name} differs from grid declaration')
        if type(cell['target_count']) is not int or cell['target_count'] < 0:
            raise ValueError('job-plan target_count must be a nonnegative integer')
        actual_cells[name] = cell
    jobs = plan['jobs']
    if not isinstance(jobs, list) or len(jobs) != count:
        raise ValueError('job-plan count must equal the number of jobs')
    job_keys = {'scene_id', 'cell_id', 'seed', 'component_seeds', 'weather_id',
                'stage', 'executable_config', 'attempt_index'}
    scenes, seeds, cell_counts = set(), set(), defaultdict(int)
    weather_ids = {variant['id'] for variant in declared['weather_variants']}
    for job in jobs:
        if not isinstance(job, dict) or set(job)-{'interaction_phase','pedestrian_pace'} != job_keys:
            raise ValueError('job-plan jobs require exact proposal fields; execution state is forbidden')
        scene = job['scene_id']
        if (not isinstance(scene, str) or not re.fullmatch(r'scene_[0-9]{4,}', scene)
                or scene != f"scene_{int(scene[6:]):04d}" or scene in scenes):
            raise ValueError('job-plan scene IDs must be unique scene_NNNN identifiers')
        scene_seed = job['seed']
        if type(scene_seed) is not int or not 0 <= scene_seed < 2**31 or scene_seed in seeds:
            raise ValueError('job-plan scene seeds must be unique integers in [0, 2**31)')
        components = {name: int(canonical_hash([scene_seed, name])[:8], 16) % 2**31
                      for name in ('location', 'layout', 'weather', 'npc')}
        if canonical_hash(job['component_seeds']) != canonical_hash(components):
            raise ValueError('job-plan component seeds do not match the scene seed')
        if not isinstance(job['cell_id'], str) or job['cell_id'] not in actual_cells:
            raise ValueError('job-plan job references an unknown cell')
        if 'interaction_phase' in job:
            from interaction_design import phase_for, pedestrian_pace_for
            template=actual_cells[job['cell_id']]['template']
            pedestrian_pace_for(job,template,phase_for(job,template))
        elif 'pedestrian_pace' in job:
            raise ValueError('pedestrian_pace requires an explicit interaction_phase')
        if not isinstance(job['weather_id'], str) or job['weather_id'] not in weather_ids:
            raise ValueError('job-plan job references an unknown weather')
        if (job['stage'] != 'proposal' or job['executable_config'] is not None
                or type(job['attempt_index']) is not int or job['attempt_index'] != 0):
            raise ValueError('job-plan cannot contain execution configuration or attempt history')
        scenes.add(scene); seeds.add(scene_seed); cell_counts[job['cell_id']] += 1
    if (sum(cell['target_count'] for cell in cells) != count or
            any(cell_counts[name] != cell['target_count'] for name, cell in actual_cells.items())):
        raise ValueError('job-plan cell targets must exactly equal the job allocation')
    weather_counts = planned_weather_counts(jobs, cells, declared['weather_variants'])
    if canonical_hash(plan['planned_weather_by_template']) != canonical_hash(weather_counts):
        raise ValueError('job-plan weather counts must match the frozen job assignments')
    return deepcopy(plan)


def load_job_plan(path, grid, count, seed, max_repairs=3, max_candidates=None,
                  expected_sha256=None):
    """Hash and parse the same bytes, rejecting duplicate keys and nonfinite JSON."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('job-plan SHA256 differs from frozen configuration')
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'job-plan has duplicate JSON key: {key}')
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f'job-plan has nonfinite JSON value: {value}')
    value = json.loads(raw, object_pairs_hook=object_pairs, parse_constant=invalid_constant)
    return validate_job_plan(value, grid, count, seed, max_repairs, max_candidates), digest


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-repairs', type=int, default=3)
    parser.add_argument('--max-candidates', type=int, help='Finite total candidate budget, default 2*count')
    parser.add_argument('--out', type=Path, required=True, help='New planning-only JSON; refuses overwrite')
    args = parser.parse_args()
    try:
        grid = json.loads(args.grid.read_text(encoding='utf-8'))
        result = build_plan(grid, args.count, args.seed, args.max_repairs, args.max_candidates)
        result['planner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        write_new(args.out, result)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    counts = [c['target_count'] for c in result['cells']]
    print(f"planning_only: {args.count} slots, {len(counts)} cells, {min(counts)}..{max(counts)} per cell -> {args.out}")


if __name__ == '__main__':
    main()
