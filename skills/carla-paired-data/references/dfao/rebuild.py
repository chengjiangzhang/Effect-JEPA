"""Reproduce the shader delta from a local CARLA 0.9.16 installation.

The qualified build used Jammy spirv-cross 2021.01.15-5, glslang-tools
11.8.0+1.3.204.0-1, spirv-tools 2022.1+1.3.204.0-2 and liblz4 1.9.3.
From the skill directory, with these tools on disk, run

    python -B references/dfao/rebuild.py --carla-root /path/to/CARLA \
        --tools /usr/bin --out /tmp/rebuilt-shader-delta.json

The stock installation is opened read-only. The output must be a new file.
"""
import argparse
import base64
import ctypes
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import zlib

HERE = Path(__file__).resolve().parent
PACKAGED = HERE.parents[1]/'scripts/runtime/native/dfao/shader_delta.json'
ENTRY = re.compile(rb'main_[0-9a-f]{8}_[0-9a-f]{8}')
PLACEHOLDER = b'main_00000000_00000000'
MAGIC = bytes.fromhex('03022307')

def digest(value):
    return hashlib.sha256(value).hexdigest()

def entry(code):
    name = ENTRY.search(code)
    if not name:
        raise ValueError('Missing Vulkan shader entry point')
    return name[0]

def named_shader(code):
    neutral = code.replace(entry(code), PLACEHOLDER)
    return neutral.replace(PLACEHOLDER, f'main_{len(code):08x}_{zlib.crc32(neutral):08x}'.encode())

def binding_positions(code):
    words = struct.unpack(f'<{len(code)//4}I', code)
    position = 5
    symbols = {}
    while position < len(words):
        length, opcode = divmod(words[position], 65536)
        if length == 0 or position+length > len(words):
            raise ValueError('Invalid SPIR-V instruction length')
        if opcode == 71 and words[position+2] in (33, 34):  # OpDecorate Binding or DescriptorSet
            symbol, field, value = words[position+1:position+4]
            symbols.setdefault(symbol, {})[field] = (position+3, value)
        position += length
    return {fields[33][1]: (fields[34][0], fields[33][0]) for fields in symbols.values()}

def patch_header(header, original, corrected):
    before, after = binding_positions(original), binding_positions(corrected)
    if before.keys() != after.keys() or named_shader(original) != original:
        raise ValueError('Shader descriptor ABI or native entry-point identity differs')
    output = bytearray(header)
    for slot, old_positions in before.items():
        old = struct.pack('<II', *old_positions)
        if header.count(old) != 1:
            raise ValueError('Unexpected UE Vulkan descriptor relocation')
        struct.pack_into('<II', output, header.index(old), *after[slot])
    old_crc = struct.pack('<I', int(entry(original)[-8:], 16))
    if header.count(old_crc) != 1:
        raise ValueError('Unexpected UE Vulkan shader checksum')
    struct.pack_into('<I', output, header.index(old_crc), int(entry(corrected)[-8:], 16))
    return bytes(output)

def corrected_glsl(native):
    if 'RWCullGridObjectNum' in native:
        # When the cell is full, its last slot carries the number of live
        # clipmap objects. Every compositor variant understands this sentinel.
        index_store = 'uvec4(_101));'
        if native.count(index_store) != 1:
            raise ValueError('Unexpected cooked coarse-grid shader')
        return native.replace(index_store, 'uvec4(_124 == 2047u ? _80 : _101));')
    header, separator, _ = native.partition('shared uint SharedCulledObjectList')
    if not separator:
        raise ValueError('Unexpected cooked compositor')
    if 'ParentGlobalDistanceFieldTexture' in header:
        header += '#define USE_PARENT 1\n'
    return header + (HERE/'compose.comp').read_text()

def rebuild(original_cache, manifest, tools, scratch):
    lz4 = ctypes.CDLL('liblz4.so.1')
    result = dict(manifest, blocks=[])
    for row in manifest['blocks']:
        offset = row['offset']
        packed_size, = struct.unpack_from('<I', original_cache, offset)
        full_size, = struct.unpack_from('<I', original_cache, offset+4+packed_size)
        if packed_size+8 != row['original_bytes']:
            raise ValueError('Invalid cooked block size')
        raw = ctypes.create_string_buffer(full_size)
        size = lz4.LZ4_decompress_safe(original_cache[offset+4:offset+4+packed_size],raw,packed_size,full_size)
        if size != full_size:
            raise ValueError('Failed to unpack native shader')
        raw = raw.raw
        start = raw.index(MAGIC)
        word_count, = struct.unpack_from('<I',raw,start-4)
        end = start+4*word_count
        original = raw[start:end]
        if digest(original) != row['native_shader'] or raw[end:-4] != original:
            raise ValueError('Unexpected original native shader bytecode')
        module = scratch/'native.spv'
        module.write_bytes(original)
        native = subprocess.check_output([tools/'spirv-cross',module,'--vulkan-semantics','--version','450'],text=True)
        source = scratch/'corrected.comp'
        source.write_text(corrected_glsl(native))
        output = scratch/'corrected.spv'
        subprocess.run([tools/'glslangValidator','-V','--target-env','vulkan1.0','-Os','--source-entrypoint',
                        'main','-e',PLACEHOLDER.decode(),'-o',output,source],check=True,stdout=subprocess.DEVNULL)
        corrected = named_shader(output.read_bytes())
        output.write_bytes(corrected)
        subprocess.run([tools/'spirv-val','--target-env','vulkan1.0',output],check=True)
        if digest(corrected) != row['corrected_shader']:
            raise ValueError('Recompiled shader differs from the qualified toolchain output')
        header = patch_header(raw[:start-4],original,corrected)
        raw = header + struct.pack('<I',len(corrected)//4) + corrected*2 + raw[-4:]
        compressed = ctypes.create_string_buffer(len(raw)+256)
        count = lz4.LZ4_compress_default(raw,compressed,len(raw),len(compressed))
        if count == 0:
            raise ValueError('Failed to pack corrected shader')
        block = struct.pack('<I',count)+compressed.raw[:count]+struct.pack('<I',len(raw))
        result['blocks'].append({**row,'compressed_block':base64.b64encode(block).decode()})
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--carla-root',type=Path,required=True)
    parser.add_argument('--tools',type=Path,default=Path('/usr/bin'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(PACKAGED.read_bytes())
    original=(args.carla_root/'Engine/GlobalShaderCache-SF_VULKAN_SM5.bin').read_bytes()
    if digest(original) != manifest['original_sha256']:
        raise ValueError('Unsupported CARLA shader cache')
    with tempfile.TemporaryDirectory() as tmp:
        result=rebuild(original,manifest,args.tools.resolve(),Path(tmp))
    if result != manifest:
        raise ValueError('Rebuilt shader delta differs from qualified payload')
    with args.out.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')

if __name__ == '__main__':
    main()
