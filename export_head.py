#!/usr/bin/env python3
"""Bake SuperGodot64's head JSON and game-over animation for DNPR preview.

The input is user-supplied game-derived data. Keep this converter separate so
the preview does not need Godot, JSON parsing, or runtime skinning.
"""
import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

DEFAULT_SOURCE = Path('/mnt/nvme/SuperGodot64/assets/title_screen/mario_head.json')


def transform(rot, pos, scale):
    x, y, z = np.radians(rot)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    result = np.eye(4)
    result[:3, :3] = rz @ ry @ rx @ np.diag(scale)
    result[:3, 3] = pos
    return result


def sample(node, sequence, frame):
    rot, pos = node['rot'], node['pos']
    animations = node.get('animation', [])
    if animations:
        typ, samples = animations[min(sequence, len(animations) - 1)]
        if typ != 'GD_ANIM_EMPTY' and samples:
            index = max(0, min(int(frame) - 1, len(samples) - 1))
            nxt = (index + 1) % len(samples)
            frac = frame % 1
            value = np.array(samples[index]) * (1 - frac) + np.array(samples[nxt]) * frac
            if typ == 'GD_ANIM_POS3S':
                pos = value
            else:
                rot = value[:3] * .1
                if len(value) == 6:
                    pos = value[3:6]
    return transform(rot, pos, node['scale'])


def apply(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def export(source, output):
    data = json.loads(source.read_text())
    nodes = data['nodes']
    shapes = data['shapes']
    names = {node['name']: node for node in nodes}
    sources = {node['shape']: node['name'] for node in nodes if node.get('shape') in shapes}
    rest = {}
    for node in nodes:
        local = transform(node['rot'], node['pos'], node['scale'])
        rest[node['name']] = np.eye(4) if node['name'] == 'DYNOBJ_MARIO_MAIN_NET' else rest.get(node['parent'], np.eye(4)) @ local

    shape_order = list(shapes)
    offsets = {}
    total_vertices = 0
    triangles = []
    for shape in shape_order:
        definition = shapes[shape]
        offsets[shape] = total_vertices
        total_vertices += len(definition['vertices'])
        for face in definition['faces']:
            color = definition['colors'][face[0]]
            # Godot's mesh uses reversed face indices.
            a, b, c = (offsets[shape] + face[i] for i in (3, 2, 1))
            rgb = [round(max(0, min(1, component)) * 255) for component in color]
            triangles.append((a, b, c, *rgb))

    influences = {}
    for shape in shape_order:
        points = np.array(shapes[shape]['vertices'], dtype=np.float64)
        influences[shape] = [[] for _ in points]
    for node in nodes:
        shape = node.get('skin')
        if shape not in influences:
            continue
        owner = rest[sources[shape]]
        correction = np.linalg.inv(rest[node['name']]) @ owner
        points = np.array(shapes[shape]['vertices'], dtype=np.float64)
        for index, weight in node['weights']:
            influences[shape][index].append((node['name'], float(weight), apply(correction, points[index:index + 1])[0]))

    with output.open('wb') as f:
        f.write(struct.pack('<8sIII', b'DNPRM001', total_vertices, len(triangles), 131))
        for triangle in triangles:
            f.write(struct.pack('<HHHBBB', *triangle))
        for sequence, frame in [(1, i) for i in range(1, 132)]:
            poses = {}
            for node in nodes:
                poses[node['name']] = poses.get(node['parent'], np.eye(4)) @ sample(node, sequence, frame)
            result = []
            for shape in shape_order:
                points = np.array(shapes[shape]['vertices'], dtype=np.float64)
                owner = poses[sources[shape]]
                transformed = apply(owner, points)
                for index, weights in enumerate(influences[shape]):
                    if not weights:
                        continue
                    remaining = 1.0
                    p = np.zeros(3)
                    for joint, weight, local in weights:
                        remaining -= weight
                        p += apply(poses[joint], local.reshape(1, 3))[0] * weight
                    transformed[index] = p + transformed[index] * remaining
                result.append(transformed * .01)
            np.concatenate(result).astype('<f4').tofile(f)
            for name in ('DYNOBJ_SILVER_STAR_LIGHT', 'DYNOBJ_RED_STAR_LIGHT'):
                (poses[name][:3, 3] * .01).astype('<f4').tofile(f)
    print(f'{len(triangles)} triangles, {total_vertices} vertices, 131 posed frames -> {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=Path('mario_head.bin'))
    args = parser.parse_args()
    export(args.source, args.output)
