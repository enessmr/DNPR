#!/usr/bin/env python3
"""Bake the local SuperGodot64 game-over background and star sprites."""
import argparse
import struct
from pathlib import Path

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, default=Path('/mnt/nvme/SuperGodot64'))
parser.add_argument('--output', type=Path, default=Path('scene.bin'))
args = parser.parse_args()
ROOT = args.source_root / 'assets/textures'
ROWS = ('033C0', '04040', '04CC0', '05940')
with args.output.open('wb') as output:
    output.write(struct.pack('<8sIIII', b'DNPRS005', 1200, 300, 128, 128))
    for suffix in ROWS:
        path = ROOT / 'title_screen_bg' / f'title_screen_bg.{suffix}.rgba16.png'
        image = Image.open(path).convert('RGB')
        assert image.size == (1200, 300), path
        output.write(image.tobytes())
    for color in ('white', 'red'):
        for frame in range(8):
            path = ROOT / 'intro_raw' / f'{color}_star_{frame}.rgba16.png'
            image = Image.open(path).convert('RGBA').resize((128, 128), Image.Resampling.LANCZOS)
            output.write(image.tobytes())
    # Segment 2's coloured title glyphs are 16x16 pixels in the N64 source.
    glyphs = {'R': '03600', 'E': '01C00', 'B': '01600', 'O': '03000',
              'T': '03A00', 'Y': '04400', 'D': '01A00', 'V': '03C00',
              'I': '02400', 'C': '01800'}
    # Keep the source glyph untouched and share its V tile with U.
    glyphs['U'] = glyphs['V']
    prompt_sources = []
    for letter in 'REBOTYUDVIC':
        path = ROOT / 'segment2' / f'segment2.{glyphs[letter]}.rgba16.png'
        prompt_sources.append(Image.open(path).convert('RGBA'))
    for image in prompt_sources:
        output.write(image.resize((16, 16), Image.Resampling.NEAREST).tobytes())
    for image in prompt_sources:
        output.write(image.resize((128, 128), Image.Resampling.LANCZOS).tobytes())
    sparkle_sources = [Image.open(ROOT / 'intro_raw' / f'sparkle_{frame}.rgba16.png').convert('RGBA')
                       for frame in range(6)]
    for image in sparkle_sources:
        output.write(image.resize((32, 32), Image.Resampling.NEAREST).tobytes())
    for image in sparkle_sources:
        output.write(image.resize((128, 128), Image.Resampling.LANCZOS).tobytes())
print(f'4 background rows, 16 star frames, 11 low and HD glyphs, 6 low and HD sparkle frames -> {args.output}')
