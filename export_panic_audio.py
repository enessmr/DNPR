#!/usr/bin/env python3
"""Embed the locally rendered game-over cue for panic-time HDA DMA playback."""
import argparse
import struct
import zlib
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, default=Path('gameover_audio.bin'))
p.add_argument('--output', type=Path, default=Path('kernel/dnpr_audio_blob.h'))
a = p.parse_args()
raw = a.input.read_bytes()
magic, rate, channels, frames = struct.unpack_from('<8sIII', raw)
assert magic == b'DNPA0001' and rate == 32000 and channels == 2 and frames == 192000
pcm = raw[20:]
assert len(pcm) == frames * channels * 2
packed = zlib.compress(pcm, 9, wbits=-15)  # raw deflate: zlib_inflate_blob expects no wrapper
lines = ['/* Generated from local game assets; do not redistribute without rights. */',
         '#define DNPR_AUDIO_RATE 32000U',
         f'#define DNPR_AUDIO_BYTES {len(pcm)}U',
         'static const unsigned char dnpr_audio_deflate[] = {']
for pos in range(0, len(packed), 20):
    lines.append('\t' + ','.join(f'0x{b:02x}' for b in packed[pos:pos + 20]) + ',')
lines.append('};')
a.output.write_text('\n'.join(lines) + '\n')
print(f'{len(pcm)} PCM bytes -> {len(packed)} compressed bytes: {a.output}')
