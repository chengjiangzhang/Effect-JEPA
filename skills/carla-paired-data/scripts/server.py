#!/usr/bin/env python3
"""Launch a local CARLA 0.9.16 server with the packaged Epic settings."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import signal
import subprocess
import sys
import time

import dfao_renderer
from native_weather_proof import verify_bundle


SYSTEM_PATH = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'


def server_environment(account, inherited=None):
    source = os.environ if inherited is None else inherited
    keys = ('LANG', 'LC_ALL', 'LC_CTYPE', 'VK_ICD_FILENAMES', 'VK_DRIVER_FILES',
            'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_DRIVER_CAPABILITIES')
    env = {key: source[key] for key in keys if key in source}
    env.update(HOME=account.pw_dir, USER=account.pw_name, LOGNAME=account.pw_name, PATH=SYSTEM_PATH)
    env['LD_LIBRARY_PATH'] = ':'.join(x for x in source.get('LD_LIBRARY_PATH', '').split(':')
                                     if x and not any(s in x.lower() for s in ('conda', 'miniforge')))
    home = Path(account.pw_dir)
    env.update(XDG_RUNTIME_DIR=str(home/'xdg-runtime'), XDG_CACHE_HOME=str(home/'.cache'),
               TMPDIR=str(home/'tmp'), TMP=str(home/'tmp'), TEMP=str(home/'tmp'),
               CARLA_CACHE_DIR=str(home/'.cache/carla'))
    return env


def signal_group(child, signum):
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass


def check_account_access(account, root):
    """Check directory traversal as the actual engine user without changing ACLs."""
    runner = shutil.which('runuser', path=SYSTEM_PATH) if os.getuid() == 0 else None
    if os.getuid() == 0 and runner is None:
        raise ValueError('runuser is required for a root launcher')
    def accessible(path, mode):
        if runner:
            result = subprocess.run([runner,'-u',account.pw_name,'--','/usr/bin/test',mode,str(path)],
                                    stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,timeout=10.)
            return result.returncode == 0
        return os.access(path, {'-x':os.X_OK,'-r':os.R_OK,'-w':os.W_OK}[mode])
    for directory in (Path(root),Path(account.pw_dir)):
        for ancestor in [*reversed(directory.parents),directory]:
            if not accessible(ancestor,'-x'):
                raise ValueError(f'Execution account {account.pw_name} cannot traverse directory {ancestor}; '
                                 'grant this account directory-search permission/ACL explicitly, or choose accessible CARLA/home paths. No permissions were changed.')
    executable=Path(root)/'CarlaUE4.sh'
    if not accessible(executable,'-r') or not accessible(executable,'-x'):
        raise ValueError(f'Execution account {account.pw_name} cannot read/execute {executable}')
    if not accessible(Path(account.pw_dir),'-w'):
        raise ValueError(f'Execution account {account.pw_name} cannot write its home {account.pw_dir}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--carla-root', type=Path, required=True)
    p.add_argument('--log-dir', type=Path, required=True)
    p.add_argument('--user', help='Existing non-root execution account; required when launcher runs as root')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--engine-ini', type=Path, default=Path(__file__).parent / 'runtime/resources/engine.ini')
    p.add_argument('--fixed-seed', action='store_true',
                   help='Pass UE4 -FixedSeed to initialize engine RNG with 0; does not reset it between world reloads')
    p.add_argument('--startup-benchmark', action='store_true',
                   help='Request UE4 -benchmark FPS=20; actual startup synchronization/timestep must be established and read back by Python')
    p.add_argument('--stop-timeout', type=float, default=20.)
    args = p.parse_args()
    root = args.carla_root.resolve()
    if not (root / 'CarlaUE4.sh').is_file() or not os.access(root/'CarlaUE4.sh', os.X_OK):
        p.error('--carla-root must contain executable CarlaUE4.sh')
    if not 1 <= args.port <= 65533 or not 0 < args.stop_timeout <= 300:
        p.error('Require port 1..65533 and stop-timeout in (0,300] seconds')
    account = pwd.getpwnam(args.user) if args.user else pwd.getpwuid(os.getuid())
    if account.pw_uid == 0:
        p.error('CARLA requires a non-root account; set --user to an existing account')
    if os.getuid() != 0 and account.pw_uid != os.getuid():
        p.error('A non-root launcher can only use its own account')
    try:
        check_account_access(account,root)
    except ValueError as exc:
        p.error(str(exc))
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log = args.log_dir / 'server.log'
    if log.exists():
        p.error('Use a new --log-dir to preserve the server log')
    account_home = Path(account.pw_dir)
    ini = account_home / '.config/Epic/CarlaUE4/Saved/Config/LinuxNoEditor/Engine.ini'
    if ini.is_symlink():
        p.error('Engine.ini must be a regular file rather than a symbolic link')
    original = None
    original_stat = None
    requested = args.engine_ini.read_bytes()
    new_dirs = []
    directory = ini.parent
    while not directory.exists():
        new_dirs.append(directory); directory = directory.parent
    ini.parent.mkdir(parents=True, exist_ok=True)
    for directory in reversed(new_dirs):
        if os.getuid() == 0:
            os.chown(directory, account.pw_uid, account.pw_gid)
    env = server_environment(account)
    # This boundary is isolated for lifecycle mocks; production always checks
    # the immutable native bundle, Shipping/assets, architecture and libc.
    try:
        stock_root = root
        root = dfao_renderer.prepare(stock_root, args.log_dir.resolve().parent)
        check_account_access(account, root)
        bundle = verify_bundle(root)
    except (ValueError, OSError, KeyError) as exc:
        p.error(str(exc))
    native_root = Path(bundle['library']).parent
    native_manifest = bundle['manifest']
    native_manifest_sha = bundle['manifest_sha256']
    trace = args.log_dir.resolve()/'weather_rng'
    trace.mkdir(exist_ok=False)
    if os.getuid() == 0:
        os.chown(trace, account.pw_uid, account.pw_gid)
    env.update(LD_PRELOAD=str(native_root/'libweather_rng.so'),CARLA_WEATHER_RNG_SEED=str(native_manifest['rng']['seed']),
               CARLA_WEATHER_RNG_LOG_DIR=str(trace))
    (args.log_dir/'weather_rng.json').write_text(json.dumps({
        'manifest_sha256':native_manifest_sha,'manifest':native_manifest,
        'library':str(native_root/'libweather_rng.so'),'trace_directory':str(trace),
        'scope':'Pinned weather RNG policy; actual application and per-scene RGB acceptance are audited separately.'},indent=2))
    for key in ('XDG_RUNTIME_DIR', 'XDG_CACHE_HOME', 'TMPDIR', 'CARLA_CACHE_DIR'):
        directory = Path(env[key])
        directory.mkdir(parents=True, exist_ok=True)
        if os.getuid() == 0:
            os.chown(directory, account.pw_uid, account.pw_gid)
        if key == 'XDG_RUNTIME_DIR':
            directory.chmod(0o700)
    command = [str(root / 'CarlaUE4.sh'), '-RenderOffScreen', '-quality-level=Epic',
               f'-carla-rpc-port={args.port}', '-nosound']
    if args.fixed_seed:
        command.append('-FixedSeed')
    if args.startup_benchmark:
        command.extend(['-benchmark', 'FPS=20'])
    if os.getuid() == 0:
        runner = shutil.which('runuser', path=SYSTEM_PATH)
        if runner is None:
            p.error('runuser is required for a root launcher')
        command = [runner, '-u', account.pw_name, '--', '/usr/bin/env', '-i',
                   *[key+'='+value for key,value in env.items()], *command]
    (args.log_dir / 'launch.json').write_text(json.dumps({'command': command,
        'engine_sha256': hashlib.sha256(requested).hexdigest(), 'engine_path': str(ini),
        'port': args.port, 'execution_account': account.pw_name, 'environment': env,
        'fixed_seed': args.fixed_seed, 'startup_benchmark': args.startup_benchmark,
        'startup_fixed_delta_seconds': .05 if args.startup_benchmark else None,
        'startup_timestep_semantics': 'Requested benchmark timestep only; actual settings and timestamps are read back separately.'}, indent=2))
    child = None
    stop_requested = [False]
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def stop(signum, frame):
        stop_requested[0] = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    lock = (Path(env['XDG_RUNTIME_DIR'])/'carla-engine.lock').open('a+b')
    installed = False
    cache_read = None
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another CARLA launch owns this execution account Engine.ini') from exc
        if os.getuid() == 0:
            os.fchown(lock.fileno(), account.pw_uid, account.pw_gid)
        original = ini.read_bytes() if ini.exists() else None
        original_stat = ini.stat() if ini.exists() else None
        if original is not None:
            (args.log_dir/'engine.before.ini').write_bytes(original)
        # Mark before write so failures while applying ownership are reversible.
        installed = True
        ini.write_bytes(requested)
        if os.getuid() == 0:
            os.chown(ini, account.pw_uid, account.pw_gid)
        cache_read = dfao_renderer.CacheRead(root/dfao_renderer.CACHE)
        read_recorded = False
        with log.open('xb') as stream:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            print(f'CARLA launcher PID={child.pid}; port={args.port}; log={log}', flush=True)
            deadline = None
            while True:
                if not read_recorded and cache_read.observe():
                    evidence = args.log_dir/'dfao_shader_read.json'
                    pending = evidence.with_suffix('.tmp')
                    pending.write_text(json.dumps(dfao_renderer.receipt(stock_root, root), indent=2))
                    pending.replace(evidence)
                    read_recorded = True
                if stop_requested[0] and deadline is None:
                    signal_group(child, signal.SIGTERM)
                    deadline = time.monotonic()+args.stop_timeout
                if deadline is not None and time.monotonic() >= deadline:
                    signal_group(child, signal.SIGKILL)
                try:
                    return child.wait(timeout=.2)
                except subprocess.TimeoutExpired:
                    pass
    finally:
        if cache_read is not None:
            cache_read.close()
        if child is not None and child.poll() is None:
            signal_group(child, signal.SIGTERM)
            try: child.wait(timeout=args.stop_timeout)
            except subprocess.TimeoutExpired:
                signal_group(child, signal.SIGKILL)
                child.wait(timeout=5.)
        if child is not None:
            signal_group(child, signal.SIGKILL)
        restored = False
        if installed and ini.exists() and ini.read_bytes() == requested:
            if original is None: ini.unlink()
            else:
                ini.write_bytes(original)
                os.chmod(ini, original_stat.st_mode & 0o7777)
                if os.getuid() == 0:
                    os.chown(ini, original_stat.st_uid, original_stat.st_gid)
                os.utime(ini, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            restored = True
        (args.log_dir/'shutdown.json').write_text(json.dumps({'engine_restored':restored,
            'returncode':child.returncode if child else None, 'stop_requested':stop_requested[0]},indent=2))
        lock.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    raise SystemExit(main())
