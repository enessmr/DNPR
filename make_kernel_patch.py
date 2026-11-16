#!/usr/bin/env python3
"""Create the DNPR integration patch for the installed Linux 6.15.2 tree."""
import argparse
import difflib
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', type=Path, default=Path('/sources/linux-6.15.2'))
parser.add_argument('--output', type=Path, default=Path('kernel/dnpr-linux-6.15.2.patch'))
args = parser.parse_args()
KERNEL = args.kernel
OUT = args.output


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f'kernel source anchor count {text.count(old)} for {old[:60]!r}')
    return text.replace(old, new, 1)


def change(path, edits):
    original = (KERNEL / path).read_text()
    updated = original
    for old, new in edits:
        updated = replace_once(updated, old, new)
    return ''.join(difflib.unified_diff(original.splitlines(keepends=True),
                                        updated.splitlines(keepends=True),
                                        fromfile=f'a/{path}', tofile=f'b/{path}'))


patches = []
patches.append(change('drivers/gpu/drm/Kconfig', [
    ('config DRM_PANIC_FOREGROUND_COLOR\n',
     'config DRM_DNPR\n'
     '\tbool "DNPR preloaded static panic scene"\n'
     '\tdepends on DRM_PANIC\n'
     '\tselect ZLIB_INFLATE\n'
     '\thelp\n'
     '\t  Draw a preloaded DNPR final frame through the driver panic\n'
     '\t  scanout hook. This only works with a driver that implements\n'
     '\t  get_scanout_buffer; it does not add that hook to a driver.\n\n'
     'config DRM_PANIC_FOREGROUND_COLOR\n'),
    ('config DRM_PANIC_SCREEN\n\tstring "Panic screen formatter"\n\tdefault "user"\n',
     'config DRM_PANIC_SCREEN\n\tstring "Panic screen formatter"\n'
     '\tdefault "dnpr" if DRM_DNPR\n\tdefault "user"\n'),
]))
patches.append(change('drivers/gpu/drm/Makefile', [
    ('drm-$(CONFIG_DRM_PANIC) += drm_panic.o\n',
     'drm-$(CONFIG_DRM_PANIC) += drm_panic.o\n'
     'drm-$(CONFIG_DRM_DNPR) += dnpr_panic.o\n'),
]))
patches.append(change('drivers/gpu/drm/drm_panic.c', [
    ('#include "drm_draw_internal.h"\n',
     '#include "drm_draw_internal.h"\n'
     '#ifdef CONFIG_DRM_DNPR\n#include "dnpr_panic.h"\n#endif\n'),
    ('static void draw_panic_dispatch(struct drm_scanout_buffer *sb)\n{\n',
     'static void draw_panic_dispatch(struct drm_scanout_buffer *sb)\n{\n'
     '#ifdef CONFIG_DRM_DNPR\n'
     '\tif (!strcmp(drm_panic_screen, "dnpr") && dnpr_panic_render(sb))\n'
     '\t\treturn;\n'
     '#endif\n'),
    ('void __init drm_panic_init(void)\n{\n\tdrm_panic_qr_init();\n}\n',
     'void __init drm_panic_init(void)\n{\n\tdrm_panic_qr_init();\n'
     '#ifdef CONFIG_DRM_DNPR\n'
     '\tif (dnpr_panic_init())\n'
     '\t\tpr_warn("DNPR: scene preload failed; using standard panic screen\\n");\n'
     '#endif\n}\n'),
    ('void drm_panic_exit(void)\n{\n\tdrm_panic_qr_exit();\n}\n',
     'void drm_panic_exit(void)\n{\n'
     '#ifdef CONFIG_DRM_DNPR\n\tdnpr_panic_exit();\n#endif\n'
     '\tdrm_panic_qr_exit();\n}\n'),
]))
OUT.write_text(''.join(patches))
print(OUT)
