CC ?= cc
CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra -Wpedantic $(shell pkg-config --cflags sdl2)
LDLIBS += $(shell pkg-config --libs sdl2) -lm
SUPERGODOT64_ROOT ?= /mnt/nvme/SuperGodot64

all: dnpr-test

dnpr-test: main.c mario_head.bin scene.bin gameover_audio.bin
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

mario_head.bin: export_head.py $(SUPERGODOT64_ROOT)/assets/title_screen/mario_head.json
	python3 export_head.py --source $(SUPERGODOT64_ROOT)/assets/title_screen/mario_head.json --output $@

scene.bin: export_scene.py
	python3 export_scene.py --source-root $(SUPERGODOT64_ROOT) --output $@

gameover_audio.bin: export_audio.py
	python3 export_audio.py --source-root $(SUPERGODOT64_ROOT) --output $@

kernel-assets: dnpr-test export_panic_scene.py export_panic_mesh.py export_panic_audio.py
	./dnpr-test --render-base-frame .dnpr-panic-base.ppm
	python3 export_panic_scene.py .dnpr-panic-base.ppm scene.bin kernel/dnpr_frame_blob.h
	python3 export_panic_mesh.py mario_head.bin scene.bin kernel/dnpr_live_blob.h
	python3 export_panic_audio.py --input gameover_audio.bin --output kernel/dnpr_audio_blob.h
	rm .dnpr-panic-base.ppm

clean:
	rm -f dnpr-test mario_head.bin scene.bin gameover_audio.bin

.PHONY: all clean kernel-assets
