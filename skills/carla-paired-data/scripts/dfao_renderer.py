"""Apply the pinned global-distance-field shader correction to a private CARLA runtime."""
import base64
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

POLICY = 'dense-distance-field-composition-v1'
STOCK = '67bb5dea997ba04df2cb1f6c3194b0a5482ec2bb72ed547baa3bf9a611c68e97'
CORRECTED = '4be705f6a45f7446902ab546d4e1fb697477a6abe9d83e783d089ad2c034da01'
CACHE = Path('Engine/GlobalShaderCache-SF_VULKAN_SM5.bin')
SHIPPING = Path('CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping')
DELTA = Path(__file__).resolve().parent/'runtime/native/dfao/shader_delta.json'
LAUNCHER = b'''#!/bin/sh
set -eu
root=$(dirname "$(readlink -f "$0")")
exec "$root/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" CarlaUE4 "$@"
'''

def digest(data):
    return hashlib.sha256(data).hexdigest()

def correct(original):
    if digest(original) != STOCK:
        raise ValueError('Unsupported CARLA Vulkan global shader cache SHA-256')
    patch = json.loads(DELTA.read_bytes())
    if (patch['policy'], patch['original_sha256'], patch['corrected_sha256']) != (POLICY, STOCK, CORRECTED):
        raise ValueError('DFAO patch identity differs')
    chunks = []
    previous = 0
    for row in patch['blocks']:
        offset = row['offset']
        end = offset + row['original_bytes']
        if not previous <= offset < end <= len(original):
            raise ValueError('Invalid shader block interval')
        chunks.extend((original[previous:offset], base64.b64decode(row['compressed_block'], validate=True)))
        previous = end
    result = b''.join([*chunks, original[previous:]])
    if digest(result) != CORRECTED:
        raise ValueError('Reconstructed shader cache failed SHA-256')
    return result

def runtime_path(base, logs):
    key = digest(os.fsencode(base.resolve()))[:16]
    return logs.resolve()/('.carla-dfao-' + CORRECTED[:12] + '-' + key)

def prepare(base, logs):
    base = base.resolve()
    runtime = runtime_path(base, logs)
    original = (base/CACHE).read_bytes()
    if digest(original) != STOCK:
        raise ValueError('Unqualified CARLA Vulkan global shader cache')
    with (logs/'.dfao-prepare.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not runtime.exists():
            stage = Path(tempfile.mkdtemp(dir=logs, prefix='.dfao-build-'))
            try:
                for sub in ('Engine', 'CarlaUE4', 'CarlaUE4/Binaries', 'CarlaUE4/Binaries/Linux'):
                    here = stage/sub
                    here.mkdir()
                    for child in (base/sub).iterdir():
                        rel = Path(sub)/child.name
                        if rel in (CACHE, SHIPPING, SHIPPING.parent, SHIPPING.parent.parent):
                            continue
                        (stage/rel).symlink_to(child)
                (stage/CACHE).write_bytes(correct(original))
                shutil.copyfile(base/SHIPPING, stage/SHIPPING)
                (stage/SHIPPING).chmod(0o555)
                (stage/'CarlaUE4.sh').write_bytes(LAUNCHER)
                (stage/'CarlaUE4.sh').chmod(0o555)
                stage.chmod(0o755)
                stage.rename(runtime)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
    if (runtime.is_symlink() or (runtime/CACHE).is_symlink() or (runtime/SHIPPING).is_symlink()
            or digest((runtime/CACHE).read_bytes()) != CORRECTED or (runtime/'CarlaUE4.sh').read_bytes() != LAUNCHER):
        raise ValueError('Private renderer runtime failed validation')
    # The existing native bundle independently checks the copied Shipping binary
    # and the linked weather assets immediately before execution.
    return runtime

class CacheRead:
    """Observe the prepared cache opening and closing during this server launch."""
    OPEN = 0x20
    CLOSE_NOWRITE = 0x10

    def __init__(self, path):
        libc = ctypes.CDLL(None, use_errno=True)
        self.fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), 'inotify_init1 for shader cache')
        if libc.inotify_add_watch(self.fd, os.fsencode(path), self.OPEN | self.CLOSE_NOWRITE) < 0:
            self.close()
            raise OSError(ctypes.get_errno(), 'inotify_add_watch for shader cache')
        self.opened = False
        self.read = False

    def observe(self):
        if self.read:
            return True
        try:
            data = os.read(self.fd, 65536)
        except BlockingIOError:
            return False
        offset = 0
        while offset < len(data):
            watch, mask, _, length = struct.unpack_from('iIII', data, offset)
            offset += 16 + length
            if watch < 0:
                raise RuntimeError('Shader cache observation queue overflow')
            if mask & self.OPEN:
                self.opened = True
            if self.opened and mask & self.CLOSE_NOWRITE:
                self.read = True
        return self.read

    def close(self):
        os.close(self.fd)

def receipt(base, runtime):
    return {'policy': POLICY, 'source_cache_sha256': STOCK, 'cache_sha256': CORRECTED,
            'stock_root': str(base.resolve()), 'runtime_root': str(runtime.resolve()),
            'cache_file': str(runtime.resolve()/CACHE), 'opened_and_closed_readonly': True}

def verify(value, executable):
    base, runtime = Path(value['stock_root']), Path(value['runtime_root'])
    if (not base.is_absolute() or not runtime.is_absolute()
            or value != receipt(base, runtime) or Path(executable) != runtime/SHIPPING):
        raise ValueError('Missing or inconsistent DFAO shader cache launch evidence')
