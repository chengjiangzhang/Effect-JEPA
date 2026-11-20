#!/usr/bin/env python3
"""Recycle an owned local CARLA launcher at synchronous pipeline boundaries.

The caller must have finished every world-using child before before_attempt().
This module never adopts an existing server and never kills a process by port.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import signal
import socket
import subprocess
import sys
import time

import native_weather_proof as weather_proof


READY_TIMEOUT_S = 90.
STOP_TIMEOUT_S = 35.
ENGINE_EXIT_TIMEOUT_S = 5.
PORT_RELEASE_TIMEOUT_S = 5.
POLL_SECONDS = .2


class ManagedServerError(RuntimeError):
    """Infrastructure failure: abort production rather than repairing a scene."""


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _record(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return {'pid': int(pid), 'ppid': int(fields[1]), 'pgid': int(fields[2]),
                'state': fields[0], 'start_ticks': int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def _same_process(record):
    current = _record(record['pid'])
    return bool(current and current['start_ticks'] == record['start_ticks'] and current['state'] != 'Z')


def _owned_processes(leader, known):
    """Remember exact descendants even if a child enters a separate session."""
    live = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            row = _record(int(path.name))
            if row:
                live[row['pid']] = row
    owned = {pid for pid, old in known.items() if pid in live
             and live[pid]['start_ticks'] == old['start_ticks']}
    if leader and leader['pid'] in live and live[leader['pid']]['start_ticks'] == leader['start_ticks']:
        owned.add(leader['pid'])
    while True:
        expanded = owned | {pid for pid, row in live.items() if row['ppid'] in owned}
        if expanded == owned:
            break
        owned = expanded
    return {pid: live[pid] for pid in owned if live[pid]['state'] != 'Z'}


def _port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=.5):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def _client(host, port):
    import carla
    return carla.Client(host, port)


def _world_readback(world):
    settings = world.get_settings()
    snapshot = world.get_snapshot()
    return {'world_id': int(world.id), 'map': world.get_map().name,
            'frame': int(snapshot.frame),
            'elapsed_seconds': float(snapshot.timestamp.elapsed_seconds),
            'delta_seconds': float(snapshot.timestamp.delta_seconds),
            'settings': {key: getattr(settings, key) for key in (
                'synchronous_mode', 'fixed_delta_seconds', 'substepping',
                'max_substep_delta_time', 'max_substeps', 'no_rendering_mode')}}


def _ready_sync_matches(readback):
    settings = readback['settings']
    return (settings['synchronous_mode'] is True
            and settings['fixed_delta_seconds'] is not None
            and abs(settings['fixed_delta_seconds'] - .05) < 1e-9
            and settings['substepping'] is True
            and abs(settings['max_substep_delta_time'] - .01) < 1e-9
            and settings['max_substeps'] == 10
            and settings['no_rendering_mode'] is False)


class ManagedServer:
    def __init__(self, config_path, batch_output, maps, host, port):
        self._lock = None
        self._child = None
        self._leader = None
        self._known = {}
        self._current = None
        self._closed = False
        self.config_path = Path(config_path).expanduser().resolve()
        self.batch_output = Path(batch_output).expanduser().resolve()
        self.host, self.port = host, port
        if host not in ('127.0.0.1', 'localhost', '::1'):
            raise ManagedServerError('Managed CARLA must use a loopback host')
        if type(port) is not int or not 1 <= port <= 65533:
            raise ManagedServerError('Managed CARLA port must be an integer in 1..65533')
        if (not isinstance(maps, (list, tuple)) or not maps
                or any(not isinstance(m, str) or not re.fullmatch(r'[A-Za-z0-9_]+', m) for m in maps)):
            raise ManagedServerError('Managed CARLA requires explicit map names')
        self.maps = list(maps)
        try:
            raw = self.config_path.read_bytes()
            config = json.loads(raw)
            required = {'carla_root', 'user', 'log_root'}
            if (not isinstance(config, dict) or not required <= config.keys()
                    or config.keys() - required - {'engine_ini', 'fixed_seed', 'startup_benchmark'}):
                raise ValueError('Config keys: carla_root, user, log_root; optional engine_ini, fixed_seed and startup_benchmark only')
            if any(not isinstance(v, str) or not v or '\x00' in v
                   for key, v in config.items() if key not in ('fixed_seed', 'startup_benchmark')):
                raise ValueError('Path and user config values must be nonempty strings')
            self.fixed_seed = config.get('fixed_seed', False)
            if type(self.fixed_seed) is not bool:
                raise ValueError('fixed_seed must be a boolean')
            self.startup_benchmark = config.get('startup_benchmark', False)
            if type(self.startup_benchmark) is not bool:
                raise ValueError('startup_benchmark must be a boolean')
            for key in ('carla_root', 'log_root', 'engine_ini'):
                if key in config and not Path(config[key]).expanduser().is_absolute():
                    raise ValueError(f'{key} must be an absolute path')
            self.carla_root = Path(config['carla_root']).expanduser().resolve()
            self.log_root = Path(config['log_root']).expanduser().resolve()
            self.user = config['user']
            account = pwd.getpwnam(self.user)
            if account.pw_uid == 0 or (os.getuid() != 0 and account.pw_uid != os.getuid()):
                raise ValueError('Use an existing non-root account available to this launcher')
            executable = self.carla_root / 'CarlaUE4.sh'
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise ValueError('carla_root must contain executable CarlaUE4.sh')
            self.server_script = Path(__file__).resolve().with_name('server.py')
            self.engine_ini = Path(config.get('engine_ini', str(self.server_script.parent / 'runtime/resources/engine.ini'))).expanduser().resolve()
            self.engine_sha256 = _sha(self.engine_ini)
            self.server_sha256 = _sha(self.server_script)
            self.config_sha256 = hashlib.sha256(raw).hexdigest()
            self.log_root.mkdir(parents=True, exist_ok=True)
            self._lock = (self.log_root / '.managed_server.lock').open('a+b')
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ValueError, KeyError) as exc:
            if self._lock is not None:
                self._lock.close()
                self._lock = None
            raise ManagedServerError(f'Invalid managed-server setup: {exc}') from exc

    def __enter__(self):
        if self._closed:
            raise ManagedServerError('Manager is already closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def _event(self, kind, **fields):
        with (self.log_root / 'lifecycle.jsonl').open('a') as stream:
            stream.write(json.dumps({'time_utc': _utc(), 'event': kind,
                'batch_output': str(self.batch_output), **fields}, allow_nan=False) + '\n')

    def _refresh_owned(self):
        self._known = _owned_processes(self._leader, self._known)

    def _allocate_epoch(self):
        indices = [int(p.name.removeprefix('server_')) for p in self.log_root.iterdir()
                   if re.fullmatch(r'server_\d+', p.name)]
        epoch = max(indices, default=0) + 1
        directory = self.log_root / f'server_{epoch:06d}'
        directory.mkdir()  # Never reuse or overwrite a previous engine log.
        return epoch, directory

    def _wait_ready(self):
        deadline = time.monotonic() + READY_TIMEOUT_S
        last_error = None
        startup_trace = []
        while time.monotonic() < deadline:
            self._refresh_owned()
            if self._child.poll() is not None:
                raise ManagedServerError(f'Owned CARLA launcher exited {self._child.returncode}; see {self._current["log_dir"]}')
            try:
                client = _client(self.host, self.port)
                client.set_timeout(min(2., max(.01, deadline - time.monotonic())))
                versions = {'client': client.get_client_version(), 'server': client.get_server_version()}
                if any(not v.startswith('0.9.16') for v in versions.values()):
                    raise ManagedServerError(f'CARLA 0.9.16 required: {versions}')
                # Freeze the native episode before batch auditing or other
                # offline preparation can advance an asynchronous world.
                # Do not tick/pad frames: actual startup clocks are evidence.
                trace = {'started_utc': _utc(), 'policy': 'ready_sync_before_offline_work'}
                startup_trace.append(trace)
                try:
                    world = client.get_world()
                    trace['before_sync'] = _world_readback(world)
                    settings = world.get_settings()
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = .05
                    settings.substepping = True
                    settings.max_substep_delta_time = .01
                    settings.max_substeps = 10
                    settings.no_rendering_mode = False
                    trace['apply_settings_frame'] = int(world.apply_settings(settings))
                    trace['after_sync'] = _world_readback(world)
                    if not _ready_sync_matches(trace['after_sync']):
                        raise ManagedServerError('Owned CARLA did not retain synchronous .05-second startup settings')
                    trace['status'] = 'synchronized'
                except Exception as exc:
                    trace.update(status='failed', error=repr(exc))
                    raise
                finally:
                    _write(Path(self._current['log_dir']) / 'startup_world_readback.json',
                           {'attempts': startup_trace})
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                client.set_timeout(min(2., remaining))
                available = sorted(client.get_available_maps())
                missing = sorted(set(self.maps) - {Path(m).name for m in available})
                if missing:
                    raise ManagedServerError(f'Owned CARLA server lacks rendered maps: {missing}')
                return {**versions, 'available_maps': available,
                        'startup_world_readback': startup_trace}
            except ManagedServerError:
                raise
            except Exception as exc:
                last_error = str(exc)
                time.sleep(min(POLL_SECONDS, max(0., deadline - time.monotonic())))
        raise ManagedServerError(f'Owned CARLA not ready within {READY_TIMEOUT_S:g}s: {last_error}')

    def _stop_current(self):
        if self._child is None:
            return
        self._refresh_owned()
        if self._child.poll() is None:
            if not self._leader or not _same_process(self._leader):
                raise ManagedServerError('Launcher PID identity changed; refusing to signal any process')
            # server.py owns and gracefully terminates its separate engine group,
            # then restores Engine.ini. Do not kill by executable name or port.
            try:
                os.kill(self._leader['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + STOP_TIMEOUT_S
            while self._child.poll() is None and time.monotonic() < deadline:
                self._refresh_owned()
                time.sleep(POLL_SECONDS)
            if self._child.poll() is None:
                raise ManagedServerError('Owned CARLA launcher did not stop cleanly; refusing to start another server')
        self._child.wait()
        deadline = time.monotonic() + ENGINE_EXIT_TIMEOUT_S
        while True:
            self._refresh_owned()
            survivors = [row['pid'] for row in self._known.values() if _same_process(row)]
            if not survivors or time.monotonic() >= deadline:
                break
            time.sleep(POLL_SECONDS)
        shutdown_path = Path(self._current['log_dir']) / 'shutdown.json'
        try:
            shutdown = json.loads(shutdown_path.read_text())
        except (OSError, ValueError):
            shutdown = None
        if survivors or not isinstance(shutdown, dict) or shutdown.get('engine_restored') is not True:
            raise ManagedServerError(f'Owned CARLA cleanup unverified (remaining={survivors}, shutdown={shutdown_path}); refusing restart')
        deadline = time.monotonic() + PORT_RELEASE_TIMEOUT_S
        while _port_open(self.host, self.port):
            if time.monotonic() >= deadline:
                raise ManagedServerError('CARLA port remains occupied after owned launcher exited; no process will be killed by port')
            time.sleep(min(POLL_SECONDS, max(0., deadline - time.monotonic())))
        self._current.update(status='stopped', stopped_utc=_utc(),
                             launcher_returncode=self._child.returncode, shutdown=shutdown)
        _write(Path(self._current['log_dir']) / 'managed.json', self._current)
        self._event('server_stopped', epoch=self._current['epoch'], launcher=self._leader)
        self._child, self._leader, self._current, self._known = None, None, None, {}

    def before_attempt(self, name, candidate, repair):
        """Call only after previous world-using subprocesses have returned."""
        try:
            return self._before_attempt(name, candidate, repair)
        except ManagedServerError:
            raise
        except (OSError, ValueError, KeyError) as exc:
            raise ManagedServerError(f'Managed CARLA infrastructure failure: {exc}') from exc

    def _before_attempt(self, name, candidate, repair):
        if self._closed:
            raise ManagedServerError('Manager is already closed')
        if (not isinstance(name, str) or not re.fullmatch(r'scene_\d+', name)
                or type(candidate) is not int or candidate < 0 or type(repair) is not int or repair < 0):
            raise ManagedServerError('Invalid scene/candidate/repair identity')
        if (_sha(self.config_path) != self.config_sha256 or _sha(self.engine_ini) != self.engine_sha256
                or _sha(self.server_script) != self.server_sha256):
            raise ManagedServerError('Managed-server configuration, Engine.ini or launcher source changed')
        self._stop_current()
        if _port_open(self.host, self.port):
            raise ManagedServerError('CARLA port is already occupied; refusing to adopt or terminate an existing server')
        epoch, directory = self._allocate_epoch()
        command = [sys.executable, '-u', '-B', str(self.server_script),
                   '--carla-root', str(self.carla_root), '--user', self.user,
                   '--port', str(self.port), '--engine-ini', str(self.engine_ini), '--log-dir', str(directory)]
        if self.fixed_seed:
            command.append('--fixed-seed')
        if self.startup_benchmark:
            command.append('--startup-benchmark')
        self._current = {'epoch': epoch, 'status': 'starting', 'scene': name,
                         'candidate': candidate, 'repair': repair, 'started_utc': _utc(),
                         'batch_output': str(self.batch_output), 'log_dir': str(directory),
                         'config_path': str(self.config_path), 'config_sha256': self.config_sha256,
                         'engine_ini': str(self.engine_ini), 'engine_sha256': self.engine_sha256,
                         'server_source_sha256': self.server_sha256, 'fixed_seed': self.fixed_seed,
                         'startup_benchmark': self.startup_benchmark,
                         'startup_fixed_delta_seconds': .05 if self.startup_benchmark else None,
                         'command': command}
        _write(directory / 'managed.json', self._current)
        try:
            with (directory / 'launcher.log').open('xb') as stream:
                self._child = subprocess.Popen(command, cwd=self.server_script.parent,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            self._leader = _record(self._child.pid)
            if self._leader is None:
                raise ManagedServerError('Cannot establish owned CARLA launcher process identity')
            self._known = {self._leader['pid']: self._leader}
            self._current['launcher'] = self._leader
            _write(directory / 'managed.json', self._current)
            self._event('server_starting', epoch=epoch, scene=name, candidate=candidate,
                        repair=repair, launcher=self._leader, log_dir=str(directory))
            ready = self._wait_ready()
            self._refresh_owned()
            self._current['native_weather_launch'] = weather_proof.launch_proof(
                directory, self.carla_root, self._known)
            self._current.update(status='ready', ready_utc=_utc(), readiness=ready)
            _write(directory / 'managed.json', self._current)
            self._event('server_ready', epoch=epoch, scene=name, candidate=candidate, repair=repair)
            return dict(self._current)
        except Exception as exc:
            self._event('server_start_failed', epoch=epoch, error=str(exc))
            try:
                self._stop_current()
            except ManagedServerError as cleanup:
                raise ManagedServerError(f'{exc}; cleanup also failed: {cleanup}') from exc
            raise ManagedServerError(str(exc)) from exc

    def weather_proof(self, require_weather, binding):
        """Read native evidence only after the world-using worker has returned."""
        try:
            if not self._current or self._current.get('status') != 'ready':
                raise ValueError('No ready owned epoch for weather proof')
            self._refresh_owned()
            fresh = weather_proof.launch_proof(self._current['log_dir'], self.carla_root, self._known)
            original = self._current['native_weather_launch']
            if fresh != original:
                raise ValueError('Owned native weather identity changed during worker')
            return weather_proof.epoch_proof(original, require_weather, binding)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ManagedServerError(f'Native weather runtime proof failed: {exc}') from exc

    def close(self):
        if self._closed:
            return
        self._stop_current()
        self._closed = True
        if self._lock is not None:
            self._lock.close()
            self._lock = None
