#!/usr/bin/env python3
"""Pack real 3D head poses and the tiled background for panic-time rasterization."""
import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('head', type=Path)
parser.add_argument('scene', type=Path)
parser.add_argument('header', type=Path)
args = parser.parse_args()

with args.head.open('rb') as f:
    magic, nv, nt, nf = struct.unpack('<8sIII', f.read(20))
    if magic != b'DNPRM001' or nv > 4096 or nt > 8192 or nf != 131:
        raise SystemExit('invalid head data')
    triangles = f.read(nt * 9)
    if len(triangles) != nt * 9:
        raise SystemExit('truncated triangles')
    indices = np.array([struct.unpack_from('<HHH', triangles, i * 9)
                        for i in range(nt)], dtype=np.int32)
    frames = []
    for _ in range(nf):
        xyz = np.frombuffer(f.read(nv * 12), dtype='<f4').copy()
        lights = np.frombuffer(f.read(24), dtype='<f4').copy()
        if xyz.size != nv * 3 or lights.size != 6:
            raise SystemExit('truncated pose data')
        frames.append((xyz.reshape(nv, 3), lights.reshape(2, 3)))

with args.scene.open('rb') as f:
    if f.read(8) != b'DNPRS005' or struct.unpack('<IIII', f.read(16)) != (1200, 300, 128, 128):
        raise SystemExit('invalid scene data')
    source = np.frombuffer(f.read(4 * 300 * 1200 * 3), dtype=np.uint8)
    if source.size != 4 * 300 * 1200 * 3:
        raise SystemExit('truncated background')
    source = source.reshape(4, 300, 1200, 3)

# Exactly the 16:9 tile mapping used by main.c at 1920x1080.
ys = np.arange(1080)
xs = np.arange(1920)
rows = (ys * 12 // 1080) % 4
sy = (ys * 12 * 300 // 1080) % 300
sx = (xs * 1200 // (1920 // 4)) % 1200
bg = source[rows[:, None], sy[:, None], sx[None, :]]
rgb565 = (((bg[:, :, 0].astype(np.uint16) >> 3) << 11) |
          ((bg[:, :, 1].astype(np.uint16) >> 2) << 5) |
          (bg[:, :, 2].astype(np.uint16) >> 3)).astype('<u2')
payload = bytearray(rgb565.tobytes())
payload += triangles


def normalized(v):
    length = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(length, 1e-20)


fixed = normalized(np.array([[-.45, .75, 1.1]]))[0]
for xyz, stars in frames:
    u = xyz[indices[:, 1]] - xyz[indices[:, 0]]
    v = xyz[indices[:, 2]] - xyz[indices[:, 0]]
    face = np.column_stack((u[:, 2] * v[:, 1] - u[:, 1] * v[:, 2],
                            u[:, 0] * v[:, 2] - u[:, 2] * v[:, 0],
                            u[:, 1] * v[:, 0] - u[:, 0] * v[:, 1]))
    normal = np.zeros_like(xyz)
    for col in range(3):
        np.add.at(normal, indices[:, col], face)
    normal = normalized(normal)
    white = normalized(stars[0] - xyz)
    red = normalized(stars[1] - xyz)
    key = np.maximum(0, normal @ fixed)
    fill = np.maximum(0, np.sum(normal * white, axis=1))
    warm = np.maximum(0, np.sum(normal * red, axis=1))
    base = .42 + key * .76 + fill * .18
    shade = np.column_stack((base + warm * .16,
                             base + warm * .015,
                             base + warm * .015))
    xyz_q = np.clip(np.rint(xyz * 256), -32768, 32767).astype(np.int16)
    shade_q = np.clip(np.rint(shade * 256), 0, 65535).astype(np.uint16)
    normal_q = np.clip(np.rint(normal * 127), -128, 127).astype(np.int8)
    for i in range(nv):
        payload += struct.pack('<hhhHHHbbbB', *xyz_q[i], *shade_q[i], *normal_q[i], 0)

compressor = zlib.compressobj(level=9, wbits=-15)
blob = compressor.compress(payload) + compressor.flush()
with args.header.open('w') as f:
    f.write('/* Generated game-derived mesh and background; do not redistribute. */\n')
    f.write(f'#define DNPR_LIVE_NV {nv}U\n#define DNPR_LIVE_NT {nt}U\n')
    f.write(f'#define DNPR_LIVE_NF {nf}U\n')
    f.write('static const unsigned char dnpr_live_deflate[] = {\n')
    for i in range(0, len(blob), 16):
        f.write('    ' + ', '.join(f'0x{x:02x}' for x in blob[i:i + 16]) + ',\n')
    f.write('};\n')
print(f'{nv} 3D vertices x {nf} poses, {nt} triangles, {len(blob)} compressed bytes')
