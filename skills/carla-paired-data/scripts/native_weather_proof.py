"""Read-only native weather RNG identity and per-epoch evidence checks.

No simulator imports, RPCs, ticks, signals, image changes or RNG calls. A proof
contains the exact trace snapshot taken after the worker has returned; a later
engine shutdown footer is not part of that snapshot.
"""
import ctypes
import hashlib
import json
import math
from pathlib import Path
import platform
import dfao_renderer
import native_decal_proof as decal_proof

BUNDLE = Path(__file__).resolve().parent / 'runtime/native/weather_rng'
MANIFEST_SHA256 = 'ad7555cc473976dee54510449dccfb9cccc65829fb9b42975895b8d7529e676b'
POLICY = 'isolated_bp_weather_rain_v1'
FILES = {'weather_rng.c', 'rand_entry.S', 'libweather_rng.so'}
RESOURCES = {'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping',
             'CarlaUE4/Content/Carla/Blueprints/Weather/BP_Weather.uasset',
             'CarlaUE4/Content/Carla/Blueprints/Weather/BP_Weather.uexp'}


def require(condition, message):
    if not condition:
        raise ValueError('Native weather proof: ' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def libc_version():
    library = ctypes.CDLL('libc.so.6')
    library.gnu_get_libc_version.restype = ctypes.c_char_p
    return library.gnu_get_libc_version().decode()


def package_bundle():
    require(sha(BUNDLE/'manifest.json') == MANIFEST_SHA256, 'manifest changed')
    value = read(BUNDLE/'manifest.json')
    require(value['policy'] == POLICY and value['target_platform'] == 'linux_x86_64', 'unknown policy/platform')
    require(set(value['files']) == FILES and set(value['native_resources']) == RESOURCES, 'unexpected native file set')
    require(all(sha(BUNDLE/name) == expected for name, expected in value['files'].items()), 'native C/assembly/library changed')
    require(value['rng']['seed'] == 0 and value['rng']['state_bytes'] == 128 and
            value['rng']['algorithm'] == 'glibc_random_r_type3_deg31_sep3', 'RNG contract changed')
    return {'manifest_sha256': MANIFEST_SHA256, 'manifest': value,
            'library': str((BUNDLE/'libweather_rng.so').resolve())}


def verify_bundle(carla_root):
    """Launcher boundary: validate real native assets before engine execution."""
    bundle = package_bundle()
    require(platform.system() == 'Linux' and platform.machine() == 'x86_64', 'unsupported actual platform')
    require(libc_version() == bundle['manifest']['rng']['libc_version'], 'actual libc version differs')
    root = Path(carla_root).resolve()
    for name, expected in bundle['manifest']['native_resources'].items():
        require(sha(root/name) == expected, 'pinned Shipping/weather asset changed: ' + name)
    return bundle


def validate_header(header, pid, manifest):
    rng = manifest['rng']
    require(header.get('kind') == 'weather_rng_start' and type(pid) is int and pid > 0 and
            header.get('pid') == pid and header.get('policy') == POLICY and
            header.get('seed') == rng['seed'] and header.get('algorithm') == rng['algorithm'] and
            header.get('state_bytes') == rng['state_bytes'] and header.get('libc_version') == rng['libc_version'] and
            header.get('libc_consumption_preserved') is True and header.get('main_thread_only') is True and
            header.get('target_caller') == '0x5ab3d5a' and header.get('test_probe') is False,
            'actual owned-engine header absent or inconsistent')


def process_identity(pid):
    path = Path('/proc')/str(pid)
    raw = (path/'stat').read_text(); fields = raw[raw.rfind(')')+2:].split()
    return {'pid': pid, 'start_ticks': int(fields[19]), 'state': fields[0],
            'argv': [x.decode() for x in (path/'cmdline').read_bytes().split(b'\0') if x]}


def launch_proof(log_dir, carla_root, owned):
    """Verify launcher records plus the header emitted by its actual owned ELF.

    Library identity is demonstrated by the pinned launch environment and native
    header, not claimed as an observation of /proc/maps (cross-user restrictions).
    """
    bundle = package_bundle(); folder = Path(log_dir).resolve()
    launch = read(folder/'launch.json'); metadata = read(folder/'weather_rng.json')
    require(metadata['manifest_sha256'] == MANIFEST_SHA256 and metadata['manifest'] == bundle['manifest'] and
            metadata['library'] == bundle['library'] and metadata['trace_directory'] == str(folder/'weather_rng'),
            'launcher bundle metadata differs')
    env = launch['environment']
    require(env.get('LD_PRELOAD') == bundle['library'] and env.get('CARLA_WEATHER_RNG_SEED') == '0' and
            env.get('CARLA_WEATHER_RNG_LOG_DIR') == str(folder/'weather_rng'), 'actual launch environment differs')
    renderer = read(folder/'dfao_shader_read.json')
    runtime = dfao_renderer.runtime_path(Path(carla_root), folder.parent)
    expected = runtime/dfao_renderer.SHIPPING
    dfao_renderer.verify(renderer, expected)
    require(renderer['stock_root'] == str(Path(carla_root).resolve()), 'original CARLA root differs')
    engines = []
    for old in owned.values():
        try:
            current = process_identity(old['pid'])
        except (FileNotFoundError, ProcessLookupError):
            continue
        if (current['start_ticks'] == old['start_ticks'] and current['state'] != 'Z' and
                current['argv'] and Path(current['argv'][0]).resolve() == expected):
            engines.append(current)
    require(len(engines) == 1, 'one exact owned Shipping process required')
    # Scheduler state (R/S) is live evidence, not stable process identity.
    engine = {key: engines[0][key] for key in ('pid', 'start_ticks', 'argv')}
    trace = folder/'weather_rng'/f"weather-{engine['pid']}.jsonl"
    with trace.open() as stream:
        header = json.loads(stream.readline())
    validate_header(header, engine['pid'], bundle['manifest'])
    decal = (decal_proof.launch_evidence(folder, engine['pid'])
             if bundle['manifest'].get('decal_order_fix') else None)
    return {**bundle, 'engine': engine, 'header': header, 'trace_path': str(trace), 'decal_order': decal,
            'dfao_renderer': renderer,
            'launch_sha256': sha(folder/'launch.json'), 'metadata_sha256': sha(folder/'weather_rng.json'),
            'library_evidence': 'pinned launcher environment and actual owned-engine native header'}


def trace_audit(text, header, require_weather):
    require(type(require_weather) is bool, 'rain requirement must be boolean')
    require(text.endswith('\n'), 'incomplete trace line')
    rows = [json.loads(line) for line in text.splitlines()]
    require(rows and rows[0] == header and sum(x.get('kind') == 'weather_rng_start' for x in rows) == 1,
            'trace header changed or duplicated')
    require(all(x.get('kind') in {'weather_rng_start', 'actual_libc_srand', 'weather_draw', 'weather_rng_end'}
                for x in rows), 'unknown trace record')
    draws = [x for x in rows if x.get('kind') == 'weather_draw']
    require(len(draws) < 1024, 'trace at truncation limit')
    require([x['event_index'] for x in draws] == list(range(1, len(draws)+1)), 'event index discontinuity')
    targets = []
    for row in draws:
        require(row['context_valid'] is True and type(row['weather_match']) is bool, 'unknown call context')
        require(all(type(row[k]) is int and 0 <= row[k] <= 2147483647
                    for k in ('actual_libc_value', 'returned_value')), 'native RNG value out of range')
        if row['weather_match']:
            require(row['status'] == 'isolated_weather_value' and row['node'] == 'Rain' and
                    row['node_outer'] == row['object_class'] == 'BP_Weather_C' and row['script_size'] == 3565 and
                    row['code_offset'] in (1584, 3304) and row['tid'] == header['pid'], 'isolation outside verified context')
            targets.append(row)
        else:
            require(row['status'] == 'unchanged_non_weather' and row['actual_libc_value'] == row['returned_value'] and
                    row['weather_ordinal'] == 0, 'non-weather output changed')
    require([x['weather_ordinal'] for x in targets] == list(range(1, len(targets)+1)), 'private ordinal discontinuity')
    require(not require_weather or bool(targets), 'rain RGB epoch has no isolated weather draws')
    ends = [x for x in rows if x.get('kind') == 'weather_rng_end']
    require(len(ends) <= 1 and (not ends or rows[-1] == ends[0]), 'invalid final trace record')
    require(all(x['truncated'] is False and x['weather_calls'] == len(targets) and x['selected_events'] == len(draws)
                for x in ends), 'trace footer count or truncation mismatch')
    return {'status': 'passed', 'selected_events': len(draws), 'weather_calls': len(targets),
            'require_weather': require_weather, 'footer_present': bool(ends), 'truncated': False}


def epoch_proof(launch, require_weather, binding):
    bundle = package_bundle()
    require(launch['manifest_sha256'] == bundle['manifest_sha256'] and launch['manifest'] == bundle['manifest'] and
            launch['library'] == bundle['library'], 'native bundle changed during worker')
    validate_header(launch['header'], launch['engine']['pid'], bundle['manifest'])
    text = Path(launch['trace_path']).read_text()
    return {'schema': 'native-weather-epoch-proof-v1', 'binding': binding, 'launch': launch,
            'trace_snapshot': text, 'trace_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'audit': trace_audit(text, launch['header'], require_weather),
            'decal_order': (decal_proof.snapshot(launch['decal_order'], launch['engine']['pid'])
                            if bundle['manifest'].get('decal_order_fix') else None)}


def verify_proof(proof, binding, require_weather):
    bundle = package_bundle(); launch = proof['launch']
    require(proof['schema'] == 'native-weather-epoch-proof-v1' and proof['binding'] == binding,
            'proof does not bind this spec/branch/state')
    require(launch['manifest_sha256'] == bundle['manifest_sha256'] and launch['manifest'] == bundle['manifest'] and
            Path(launch['library']).is_absolute(), 'proof bundle identity differs')
    branches = binding.get('branches', [binding])
    if 'dfao_renderer' in launch or any('dfao_cache_sha256' in b for b in branches):
        dfao_renderer.verify(launch['dfao_renderer'], launch['engine']['argv'][0])
        require(all(b.get('dfao_cache_sha256', dfao_renderer.CORRECTED) == dfao_renderer.CORRECTED
                    for b in branches), 'scene specifies a different DFAO renderer')
    validate_header(launch['header'], launch['engine']['pid'], bundle['manifest'])
    require(launch['engine']['start_ticks'] > 0, 'engine process identity absent')
    text = proof['trace_snapshot']
    require(hashlib.sha256(text.encode()).hexdigest() == proof['trace_sha256'], 'sealed trace snapshot changed')
    require(trace_audit(text, launch['header'], require_weather) == proof['audit'], 'stored trace audit differs')
    if bundle['manifest'].get('decal_order_fix'):
        require(launch.get('decal_order') is not None and proof.get('decal_order') is not None,
                'native decal ordering evidence absent')
        decal_proof.verify(proof['decal_order'], launch['decal_order'], launch['engine']['pid'])


def branch_binding(spec_path, branch_path, phase):
    spec_path, branch_path = Path(spec_path), Path(branch_path)
    renderer = read(spec_path).get('render', {}).get('engine', {}).get('dfao_cache_sha256')
    extra = {'dfao_cache_sha256': renderer} if renderer is not None else {}
    return {**extra, 'phase': phase, 'scene': spec_path.parent.name, 'branch': branch_path.name,
            'spec_sha256': sha(spec_path), 'meta_sha256': sha(branch_path/'meta.json'),
            'states_sha256': sha(branch_path/'states.npz')}


def rain_required(spec):
    rain = spec['weather']['precipitation']
    require(type(rain) in (int, float) and math.isfinite(rain) and 0 <= rain <= 100, 'invalid spec precipitation')
    return rain > 0
