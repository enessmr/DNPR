# DNPR panic renderer

DNPR is an experimental animated Linux DRM panic screen built around a software-rendered 3D scene. The kernel rasterizes Mario's head, composes the background and animated overlays, then writes the shared frame to available DRM scanouts. The userspace demo is a simulator and preview; it cannot reproduce every restriction or measure the exact performance of a real kernel panic.

## Renderer and benchmark

By default, `dnpr-test` runs the panic renderer's integer raster, sprite, region, composition, and blit code with persistent CPU workers. Workers and the calling thread claim 32-row raster bands. The simulator uses two scanout buffers and presents a separate SDL copy; it can use mapped DRM dumb buffers or fall back to RAM. Neither path is the GPU driver's actual panic scanout.

```sh
./dnpr-test --benchmark-panic --from 21 --to 25
./dnpr-test --panic-threads 1 --benchmark-panic
./dnpr-test --no-drm-buffer --benchmark-panic
```

Benchmarking measures unthrottled render throughput, not display refresh or live playback FPS. It reports animated and settled frames separately, stage timings, and blit bytes/write counts. Frame numbers sample the 30 Hz head animation clock; a starting frame gets an untimed predecessor to establish incremental state. `--panic-threads N` selects 1–16 threads. `--unshaded` and `--no-specular` are simulator diagnostics only.

## SuperGodot64 assets and preview

The legacy preview (`--legacy-preview`) renders the local SuperGodot64 head geometry and game-over animation with a software depth buffer and Gouraud lighting. `export_head.py` bakes 131 head poses from the project's JSON mesh. Playback interpolates adjacent poses from elapsed time and recomputes smooth normals; after the final sigh pose (frame 130), the head stays still while the overlays continue animating.

`export_scene.py` builds the tiled background strips, star frames, prompt atlases, and sparkle frames from local SuperGodot64 assets. The scene includes silver and red spinning stars, animated sparkles, and the flickering “REBOOT YOUR DEVICE” prompt. HD sparkle art is used at 720p and above; lower resolutions use the smaller atlas. `export_audio.py` extracts the local game-over cue and Mario voice through the project's audio library. Audio continues after the head settles. Keep game-derived assets local unless their distribution rights are established.

Build the demo with `make`. Requirements include Python 3, NumPy, Pillow, SDL2 development files, and local SuperGodot64 assets, ROM, and `libsm64.so`. Set `SUPERGODOT64_ROOT` if the project is outside its default location. Generated assets and preview binaries are not committed.

Useful commands:

```sh
./dnpr-test --displays
./dnpr-test --display 1
./dnpr-test --no-audio
./dnpr-test --frame-step
./dnpr-test --static
./dnpr-test --animation
./dnpr-test --status
./dnpr-test --stop
./dnpr-test --render-frame preview.ppm
./dnpr-test --render-base-frame base.ppm
```

The preview prefers an identified Acer display by default, then falls back to display 0. `--display N` can select any SDL display. `--frame N` and `--time-ms N` render a selected animation time. `--frame-step` pauses at frame 0; press F to advance one source frame, and Escape to exit.

## Kernel backend

Kernel sources, patches, and asset exporters are in `kernel/`. Run `make kernel-assets` to generate the compressed scene, mesh, audio, and specular lookup assets from local inputs. `kernel/dnpr-linux-6.15.2.patch` integrates the backend into Linux 6.15.2; DNPR must be built into the kernel because panic integration is not a loadable-module path.

The renderer uses preallocated memory and integer arithmetic in panic context. It must not depend on allocation, sleeping, desktop services, or normal GPU rendering. Panic audio is optional and uses preloaded HDA DMA data. The optional SMP handoff is experimental and requires `kernel/dnpr-smp-handoff.patch`; test with a boot fallback available.

Generic DRM panic scanout support is limited, and proprietary NVIDIA support requires the separate `kernel/nvidia-drm-panic.patch`. Userspace timing and RAM fallback results do not validate that driver path or real-panic frame pacing.
