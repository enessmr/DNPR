#!/usr/bin/env python3
"""Bake the SM64 game-over cue through its ending, after the head sigh.

This uses the user's local ROM and SuperGodot64 mixer. The output is a local
preview asset; its redistribution rights are not assumed.
"""
import argparse
import ctypes as ct
import struct
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, default=Path('/mnt/nvme/SuperGodot64'))
parser.add_argument('--output', type=Path, default=Path('gameover_audio.bin'))
args = parser.parse_args()
ROOT = args.source_root
ROM = ROOT / 'Super Mario 64 (USA).z64'
LIBRARY = ROOT / 'addons/libsm64_godot/extension/bin/libsm64.so'
RATE = 32000
# The game-over phrase has a near-silent break at 5.9 s. The next phrase
# begins at 6.0 s and belongs to the title sequence, so stop before it.
SECONDS = 6
FRAMES = SECONDS * RATE

lib = ct.CDLL(str(LIBRARY))
rom_bytes = ROM.read_bytes()
rom = (ct.c_uint8 * len(rom_bytes)).from_buffer_copy(rom_bytes)
texture = (ct.c_uint8 * (704 * 64 * 4))()
lib.sm64_global_init(rom, texture)
lib.sm64_audio_init(rom)
lib.sm64_play_music.argtypes = [ct.c_uint8, ct.c_uint16, ct.c_uint16]
lib.sm64_play_sound_global.argtypes = [ct.c_int32]
lib.sm64_play_music(0, 0x82, 0)  # SEQ_PLAYER_LEVEL, SEQ_MENU_GAME_OVER
buffer = (ct.c_int16 * (544 * 4))()
chunks = []
for tick in range(SECONDS * 30 + 10):
    if tick == 20:
        lib.sm64_play_sound_global(0x2431FF81)  # SOUND_MARIO_GAME_OVER; STARTING status bit
    count = lib.sm64_audio_tick(0, 2200, buffer)
    chunks.append(np.ctypeslib.as_array(buffer)[:count * 4].copy())
lib.sm64_global_terminate()
samples = np.concatenate(chunks)[:FRAMES * 2].astype(np.float32)
assert samples.size == FRAMES * 2
samples *= .65  # Safe output cap; retain source dynamics.
fade_frames = RATE // 4  # Smooth the last tail at the natural break.
samples[-fade_frames * 2:] *= np.repeat(np.linspace(1, 0, fade_frames, dtype=np.float32), 2)
peak = float(np.max(np.abs(samples))) / 32767
samples = np.clip(samples, -32768, 32767).astype('<i2')
with args.output.open('wb') as output:
    output.write(struct.pack('<8sIII', b'DNPA0001', RATE, 2, FRAMES))
    output.write(samples.tobytes())
print(f'{FRAMES} stereo frames at {RATE} Hz, peak {peak:.3f} -> {args.output}')
