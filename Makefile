CC ?= cc
CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra -Wpedantic $(shell pkg-config --cflags sdl2 libdrm) -pthread
LDLIBS += $(shell pkg-config --libs sdl2 libdrm) -lm -pthread
SUPERGODOT64_ROOT ?= /mnt/nvme/SuperGodot64

all: dnpr-test

dnpr-test: main.c panic_sim.o mario_head.bin scene.bin gameover_audio.bin
	$(CC) $(CFLAGS) -o $@ main.c panic_sim.o $(LDLIBS) -lz

panic_sim.o: panic_sim.c panic_sim.h kernel/dnpr_raster_core.h kernel/dnpr_raster_rows.h kernel/dnpr_sprite_core.h kernel/dnpr_regions_core.h kernel/dnpr_shared_core.h kernel/dnpr_blit_core.h kernel/dnpr_frame_blob.h kernel/dnpr_live_blob.h kernel/dnpr_spec_blob.h
	$(CC) $(CFLAGS) -Wno-sign-compare -Wno-unused-function -c -o $@ panic_sim.c

.dnpr-asset-builder: main.c mario_head.bin scene.bin gameover_audio.bin
	$(CC) $(CFLAGS) -DDNPR_ASSET_BUILDER -o $@ main.c $(LDLIBS)

mario_head.bin: export_head.py $(SUPERGODOT64_ROOT)/assets/title_screen/mario_head.json
	python3 export_head.py --source $(SUPERGODOT64_ROOT)/assets/title_screen/mario_head.json --output $@

scene.bin: export_scene.py
	python3 export_scene.py --source-root $(SUPERGODOT64_ROOT) --output $@

gameover_audio.bin: export_audio.py
	python3 export_audio.py --source-root $(SUPERGODOT64_ROOT) --output $@

kernel/dnpr_frame_blob.h kernel/dnpr_live_blob.h kernel/dnpr_audio_blob.h &: export_panic_scene.py export_panic_mesh.py export_panic_audio.py mario_head.bin scene.bin gameover_audio.bin
	python3 export_panic_scene.py scene.bin kernel/dnpr_frame_blob.h
	python3 export_panic_mesh.py mario_head.bin scene.bin kernel/dnpr_live_blob.h
	python3 export_panic_audio.py --input gameover_audio.bin --output kernel/dnpr_audio_blob.h

kernel/dnpr_spec_blob.h: gen_spec_lut.py
	python3 gen_spec_lut.py

kernel-assets: kernel/dnpr_frame_blob.h kernel/dnpr_live_blob.h kernel/dnpr_audio_blob.h kernel/dnpr_spec_blob.h

clean:
	rm -f dnpr-test panic_sim.o .dnpr-asset-builder mario_head.bin scene.bin gameover_audio.bin

.PHONY: all clean kernel-assets
