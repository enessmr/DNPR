// SPDX-License-Identifier: GPL-2.0
/*
 * Preloaded DNPR panic scene for the DRM panic scanout path.
 *
 * The scene is inflated into normal kernel memory during boot. The base
 * contains the tiled background; posed 3D vertices and triangle colors
 * are rasterized with a depth buffer on each head frame. Sprite layers
 * contain both stars, both sparkle sets, and prompt glyphs.
 *
 * Nothing animates before a panic. The DRM dumper collects all scanouts,
 * then one shared panic loop redraws them from one clock.
 * Redraws allocate nothing, use no floating point, and end with the
 * same write barrier the driver's panic_flush uses.
 */
#include <asm/barrier.h>
#ifdef CONFIG_X86
#include <asm/cpu.h>
#endif
#include <linux/atomic.h>
#include <linux/cpumask.h>
#include <linux/crc32.h>
#include <linux/errno.h>
#include <linux/delay.h>
#include <linux/init.h>
#include <linux/iosys-map.h>
#include <linux/ktime.h>
#include <linux/math.h>
#include <linux/math64.h>
#include <linux/minmax.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/printk.h>
#include <linux/processor.h>
#include <linux/slab.h>
#include <linux/smp.h>
#include <linux/string.h>
#include <linux/vmalloc.h>
#include <linux/zlib.h>

#include <drm/drm_panic.h>
#include "drm_draw_internal.h"
#include "dnpr_panic.h"
#include "dnpr_audio.h"
#include "dnpr_frame_blob.h"
#include "dnpr_live_blob.h"
#include "dnpr_spec_blob.h"

#define DNPR_WIDTH 1920U
#define DNPR_HEIGHT 1080U
#define DNPR_MAX_WIDTH 4096U
#define DNPR_MAX_HEIGHT 2160U
#define DNPR_FASTEST_FRAME_NS 6944444ULL /* 144 Hz ceiling */

#define DNPR_BASE_BYTES ((size_t)DNPR_WIDTH * DNPR_HEIGHT * 2U)
#define DNPR_SPRITE 128U
#define DNPR_SPRITE_BYTES ((size_t)DNPR_SPRITE * DNPR_SPRITE * 4U)
#define DNPR_STAR_FRAMES 8U
#define DNPR_SPARKLE_FRAMES 6U
#define DNPR_GLYPH_FRAMES 11U
#define DNPR_STAR_OFF 0
#define DNPR_RED_STAR_OFF (DNPR_STAR_FRAMES * DNPR_SPRITE_BYTES)
#define DNPR_SPARKLE_OFF (DNPR_RED_STAR_OFF + DNPR_STAR_FRAMES * DNPR_SPRITE_BYTES)
#define DNPR_RED_SPARKLE_OFF (DNPR_SPARKLE_OFF + DNPR_SPARKLE_FRAMES * DNPR_SPRITE_BYTES)
#define DNPR_GLYPH_OFF (DNPR_RED_SPARKLE_OFF + DNPR_SPARKLE_FRAMES * DNPR_SPRITE_BYTES)
#define DNPR_SCENE_BYTES (DNPR_BASE_BYTES + DNPR_GLYPH_OFF + DNPR_GLYPH_FRAMES * DNPR_SPRITE_BYTES)

/* Mirror of main.c: 3+1 silver, 3+1 red, and REBOOT/YOUR/DEVICE glyphs. */
#define DNPR_MAX_SPRITES 24U
#define DNPR_PROMPT_ORDER "REBOTYUDVIC"

struct dnpr_sprite {
	const u8 *src; /* 128x128 RGBA, tint already baked */
	int left;
	int top;
	int size;
};

static u16 *dnpr_base;
static u8 *dnpr_layers;
static u8 *dnpr_row;
static u32 *dnpr_shared_frame;
static u8 *dnpr_shared_dirty;
static bool dnpr_shared_valid;
static bool dnpr_ready;

/* Opt-in panic-time raster fan-out. The panic CPU parks the other CPUs in
 * IPI callbacks before normal shutdown. Workers and the panic CPU claim
 * disjoint 32-row bands; a timed-out frame stops the helpers before the
 * single-CPU redraw reuses shared vertex and depth buffers. */
static bool dnpr_smp;
module_param_named(smp, dnpr_smp, bool, 0444);
static bool dnpr_smp_selftest;
module_param_named(smp_selftest, dnpr_smp_selftest, bool, 0444);

#define DNPR_BAND_H 32U
#define DNPR_BAND_COUNT DIV_ROUND_UP(DNPR_HEIGHT, DNPR_BAND_H)
#define DNPR_RASTER_TIMEOUT_MS 25U

static atomic_t dnpr_band_next = ATOMIC_INIT(DNPR_BAND_COUNT);
static atomic_t dnpr_raster_done = ATOMIC_INIT(0);
static atomic_t dnpr_raster_alive = ATOMIC_INIT(0);
static atomic_t dnpr_raster_stop = ATOMIC_INIT(0);
static unsigned int dnpr_raster_gen;
static unsigned int dnpr_helper_count;
static unsigned int dnpr_governor_cpu;
static bool dnpr_helpers_active;
static bool dnpr_panic_workers_armed;

/* Quantized 3D geometry. Positions are signed Q8 model coordinates; Z needs
 * 32 bits for the opening poses near -200. RGB
 * lighting factors are unsigned Q8. Triangles are rasterized on every posed
 * panic frame, with depth testing, rather than replaying rendered images.
 */
struct dnpr_live_vertex {
	s16 x, y;
	s32 z;
	u16 lr, lg, lb;
	s8 nx, ny, nz;
	u8 pad;
} __packed;

struct dnpr_live_triangle {
	u16 a, b, c;
	u8 r, g, blue;
} __packed;

struct dnpr_projected {
	s32 x, y;
	u16 z, lr, lg, lb;
	s8 nx, ny, nz;
};

struct dnpr_rect {
	unsigned int x0, y0, x1, y1;
};

#define DNPR_LIVE_BYTES (DNPR_BASE_BYTES + \
	DNPR_LIVE_NT * sizeof(struct dnpr_live_triangle) + \
	DNPR_LIVE_NF * DNPR_LIVE_NV * sizeof(struct dnpr_live_vertex))

static u8 *dnpr_live_data;
static u16 *dnpr_live_depth;
static u16 *dnpr_live_owner;
static u8 *dnpr_spec_lut;
static struct dnpr_projected dnpr_projected[DNPR_LIVE_NV];
static u16 dnpr_triangle_order[DNPR_LIVE_NT];
static u64 dnpr_last_pose_q = ~0ULL;
static struct {
	int x0, y0, x1, y1;
} dnpr_head_bounds, dnpr_head_dirty;

#include "dnpr_raster_core.h"
#include "dnpr_raster_rows.h"

static void dnpr_grab_bands(void)
{
	for (;;) {
		int band = atomic_fetch_inc(&dnpr_band_next);
		int y0 = band * (int)DNPR_BAND_H;

		if (band >= (int)DNPR_BAND_COUNT)
			return;
		dnpr_raster_rows(y0, min(y0 + (int)DNPR_BAND_H - 1,
					 (int)DNPR_HEIGHT - 1));
	}
}

static void dnpr_raster_helper(void *unused)
{
	bool governor = smp_processor_id() == READ_ONCE(dnpr_governor_cpu);
	unsigned int seen = READ_ONCE(dnpr_raster_gen);

	(void)unused;
	atomic_inc(&dnpr_raster_alive);
#ifdef CONFIG_X86
	if (READ_ONCE(dnpr_panic_workers_armed)) {
		smp_wmb();
		cpumask_clear_cpu(smp_processor_id(), &cpus_stop_mask);
	}
#endif
	for (;;) {
		unsigned int gen;

		for (;;) {
			gen = READ_ONCE(dnpr_raster_gen);
			if (gen != seen || atomic_read(&dnpr_raster_stop))
				break;
			cpu_relax();
		}
		if (atomic_read(&dnpr_raster_stop))
			break;
		seen = gen;
		smp_rmb(); /* projected vertices published before the bump */
		if (governor)
			dnpr_audio_maybe_stop(); /* audio owns this CPU */
		else
			dnpr_grab_bands();
		atomic_inc_return(&dnpr_raster_done);
	}
	atomic_dec(&dnpr_raster_alive);
}

static void dnpr_raster_start_helpers(void)
{
	cpumask_t mask;

	if (!dnpr_smp || dnpr_helpers_active || num_online_cpus() < 3)
		return;
	cpumask_copy(&mask, cpu_online_mask);
	cpumask_clear_cpu(smp_processor_id(), &mask);
	if (cpumask_empty(&mask))
		return;
	atomic_set(&dnpr_raster_stop, 0);
	atomic_set(&dnpr_raster_alive, 0);
	/* Exhausted queue: helpers that wake early rasterize nothing. */
	atomic_set(&dnpr_band_next, DNPR_BAND_COUNT);
	WRITE_ONCE(dnpr_governor_cpu, cpumask_first(&mask));
	dnpr_helper_count = cpumask_weight(&mask);
	preempt_disable();
	/* wait=0: the IPIs queue and run as each target can; the raster
	 * deadline covers any target that never services one. */
	smp_call_function_many(&mask, dnpr_raster_helper, NULL, 0);
	preempt_enable();
	dnpr_helpers_active = true;
}

/* Arm the x86 reboot-vector handoff. The architecture's panic stop IPI
 * interrupts each responsive CPU, so this path never calls the normal
 * smp_call_function_many() API with panic interrupts disabled. */
bool dnpr_panic_smp_arm(void)
{
	cpumask_t mask;

	if (!IS_ENABLED(CONFIG_X86) || !dnpr_smp || !READ_ONCE(dnpr_ready) ||
	    num_online_cpus() < 3)
		return false;
	cpumask_copy(&mask, cpu_online_mask);
	cpumask_clear_cpu(smp_processor_id(), &mask);
	WRITE_ONCE(dnpr_governor_cpu, cpumask_first(&mask));
	atomic_set(&dnpr_raster_stop, 0);
	atomic_set(&dnpr_raster_alive, 0);
	atomic_set(&dnpr_band_next, DNPR_BAND_COUNT);
	WRITE_ONCE(dnpr_panic_workers_armed, true);
	return true;
}

bool dnpr_panic_smp_armed(void)
{
	return READ_ONCE(dnpr_panic_workers_armed);
}

void dnpr_panic_smp_worker(void)
{
	dnpr_raster_helper(NULL);
}

void dnpr_panic_smp_finish(void)
{
	smp_rmb();
	dnpr_helper_count = atomic_read(&dnpr_raster_alive);
	dnpr_helpers_active = dnpr_helper_count != 0;
}

static void dnpr_raster_frame(void)
{
	unsigned int done;
	s64 deadline;

	atomic_set(&dnpr_raster_done, 0);
	atomic_set(&dnpr_band_next, 0);
	smp_wmb(); /* projected vertices and cleared buffers are ready */
	WRITE_ONCE(dnpr_raster_gen, dnpr_raster_gen + 1);
	dnpr_grab_bands(); /* the orchestrator joins the queue */
	deadline = ktime_get_mono_fast_ns() +
		(s64)DNPR_RASTER_TIMEOUT_MS * 1000000;
	while ((done = atomic_read_acquire(&dnpr_raster_done)) < dnpr_helper_count &&
	       ktime_get_mono_fast_ns() < deadline)
		cpu_relax();
	if (done < dnpr_helper_count) {
		/* A late helper might still be using the projected vertices.
		 * Quiesce all helpers before recycling the frame buffers. */
		atomic_set(&dnpr_raster_stop, 1);
		WRITE_ONCE(dnpr_panic_workers_armed, false);
		smp_send_stop();
		dnpr_helpers_active = false;
		memcpy(dnpr_base, dnpr_live_data, DNPR_BASE_BYTES);
		memset(dnpr_live_depth, 0xff, DNPR_BASE_BYTES);
		memset(dnpr_live_owner, 0xff, DNPR_BASE_BYTES);
		dnpr_raster_rows(0, DNPR_HEIGHT - 1);
	}
	smp_rmb();
}

static bool dnpr_render_live_head(unsigned int elapsed_ms)
{
	bool head_changed = dnpr_render_live_head_prepare(elapsed_ms);

	if (!head_changed)
		return false;
	if (dnpr_helpers_active)
		dnpr_raster_frame();
	else
		dnpr_raster_rows(0, DNPR_HEIGHT - 1);
	return true;
}

/* Panic-time animation state. Every plane that reports a scanout buffer
 * during the panic draw gets its own snapshot here, and the blink handler
 * redraws all of them: we cannot tell which scanout is actually visible.
 * Snapshots are only written before the panic loop starts, so blink
 * redraws stay allocation-free.
 */
#define DNPR_MAX_SB 8U
static struct drm_scanout_buffer dnpr_sb[DNPR_MAX_SB];
static bool dnpr_sb_valid[DNPR_MAX_SB];
static unsigned int dnpr_sb_count;
static unsigned int dnpr_frame;

#include "dnpr_sprite_core.h"

static void dnpr_composite_rect(struct drm_scanout_buffer *sb,
				const struct dnpr_sprite *sprites, unsigned int n,
				unsigned int frame, unsigned int x0,
				unsigned int y0, unsigned int x1,
				unsigned int y1)
{
	unsigned int width = sb->width, height = sb->height;
	unsigned int cpp = sb->format->cpp[0];
	void (*set_pixel)(struct drm_scanout_buffer *, unsigned int, unsigned int,
			  u32) = sb->set_pixel;
	unsigned int bar_w = ((frame % 40) + 1) * 8;
	unsigned int x, y;

	for (y = y0; y < min(y1, height); y++) {
		const struct dnpr_sprite *active[DNPR_MAX_SPRITES];
		unsigned int row_n = 0, i;

		for (i = 0; i < n; i++)
			if (y >= sprites[i].top && y < sprites[i].top + sprites[i].size)
				active[row_n++] = &sprites[i];

		for (x = x0; x < min(x1, width); x++) {
			unsigned int r, g, b;
			u32 rgb, color;

			dnpr_sample_base(x, y, width, height, &r, &g, &b);
			for (i = 0; i < row_n; i++)
				dnpr_blend_sprite(active[i], x, y, &r, &g, &b);
			if (y < 4 && x < bar_w) {
				r = 255;
				g = 0;
				b = 255;
			}
			rgb = (r << 16) | (g << 8) | b;
			color = drm_draw_color_from_xrgb8888(rgb, sb->format->format);

			if (set_pixel)
				set_pixel(sb, x, y, color);
			else
				memcpy(dnpr_row + x * cpp, &color, cpp);
		}
		if (!set_pixel)
			iosys_map_memcpy_to(&sb->map[0],
					    (size_t)y * sb->pitch[0] + x0 * cpp,
					    dnpr_row + x0 * cpp,
					    (min(x1, width) - x0) * cpp);
	}
}

#include "dnpr_regions_core.h"

static void dnpr_composite(struct drm_scanout_buffer *sb,
			   unsigned int elapsed_ms, unsigned int frame,
			   bool initial, bool head_changed)
{
	struct dnpr_sprite sprites[DNPR_MAX_SPRITES];
	struct dnpr_rect rects[4];
	unsigned int n, nr, i;

	n = dnpr_place_sprites(elapsed_ms, sb->width, sb->height, sprites);
	nr = dnpr_regions(sb->width, sb->height, initial, head_changed, rects);
	for (i = 0; i < nr; i++)
		dnpr_composite_rect(sb, sprites, n, frame, rects[i].x0,
				    rects[i].y0, rects[i].x1, rects[i].y1);
}

/* Compose 1080p pixels once. Both connected monitors currently use this
 * resolution, so their refresh rates do not require separate scene work.
 */
#include "dnpr_shared_core.h"

#include "dnpr_blit_core.h"

static bool dnpr_scanout_ok(struct drm_scanout_buffer *sb)
{
	unsigned int cpp;

	if (!sb || !sb->format)
		return false;
	cpp = sb->format->cpp[0];
	if (!sb->width || !sb->height || sb->width < 48 || sb->width > DNPR_MAX_WIDTH ||
	    sb->height < 48 || sb->height > DNPR_MAX_HEIGHT ||
	    cpp < 2 || cpp > 4 || sb->pitch[0] < sb->width * cpp ||
	    (!sb->set_pixel && iosys_map_is_null(&sb->map[0])))
		return false;
	return true;
}

bool dnpr_panic_render(struct drm_scanout_buffer *sb)
{
	unsigned int idx;

	if (!READ_ONCE(dnpr_ready) || !dnpr_scanout_ok(sb))
		return false;

	/* The DRM dumper calls this for every plane before starting the shared
	 * animation loop. Keep the scanout mapping and display its first frame.
	 */
	if (dnpr_sb_count < DNPR_MAX_SB)
		idx = dnpr_sb_count++;
	else
		idx = DNPR_MAX_SB - 1;
	dnpr_sb[idx] = *sb;
	WRITE_ONCE(dnpr_sb_valid[idx], true);
	dnpr_render_live_head(0);
	if (dnpr_shared_frame && sb->width == DNPR_WIDTH &&
	    sb->height == DNPR_HEIGHT) {
		if (!dnpr_shared_valid)
			dnpr_render_shared(0, 0, true, true);
		dnpr_blit_shared(&dnpr_sb[idx], true, true);
	} else {
		dnpr_composite(&dnpr_sb[idx], 0, 0, true, true);
	}
	wmb();
	return true;
}

void dnpr_panic_animate(void)
{
	u64 start_ns, next_ns, now_ns, wait_us;
	unsigned int k, elapsed_ms;
	bool head_changed;
	bool native_scanout = false;

	if (!READ_ONCE(dnpr_sb_count))
		return;
	for (k = 0; k < dnpr_sb_count; k++)
		if (READ_ONCE(dnpr_sb_valid[k]) &&
		    dnpr_sb[k].width == DNPR_WIDTH &&
		    dnpr_sb[k].height == DNPR_HEIGHT)
			native_scanout = true;
	start_ns = ktime_get_mono_fast_ns();
	next_ns = start_ns + DNPR_FASTEST_FRAME_NS;
	dnpr_audio_start();
	for (;;) {
		now_ns = ktime_get_mono_fast_ns();
		if (now_ns < next_ns) {
			wait_us = (next_ns - now_ns) / 1000;
			if (wait_us >= 1000)
				mdelay(wait_us / 1000);
			if (wait_us % 1000)
				udelay(wait_us % 1000);
		}
		now_ns = ktime_get_mono_fast_ns();
		elapsed_ms = min_t(u64,
			(now_ns - start_ns) / 1000000,
			~0U);
		next_ns = now_ns + DNPR_FASTEST_FRAME_NS;
		dnpr_frame++;
		head_changed = dnpr_render_live_head(elapsed_ms);
		if (native_scanout && dnpr_shared_frame)
			dnpr_render_shared(elapsed_ms, dnpr_frame, false,
					   head_changed);
		for (k = 0; k < dnpr_sb_count; k++)
			if (READ_ONCE(dnpr_sb_valid[k])) {
				if (dnpr_shared_frame &&
				    dnpr_sb[k].width == DNPR_WIDTH &&
				    dnpr_sb[k].height == DNPR_HEIGHT)
					dnpr_blit_shared(&dnpr_sb[k], false,
							 head_changed);
				else
					dnpr_composite(&dnpr_sb[k], elapsed_ms,
						       dnpr_frame, false,
						       head_changed);
			}
		wmb();
		dnpr_audio_maybe_stop();
	}
}

/* Boot-time selftest: render reference poses on one CPU, render the same
 * poses through the SMP band queue, and CRC-compare the shared frame.
 * Helpers are parked afterwards, so a panic later in the same boot falls
 * back to the uniprocessor draw. Boot with dnpr_panic.smp_selftest=1 once
 * and read the results from dmesg.
 */
static int __init dnpr_smp_selftest_run(void)
{
	static const unsigned int poses[3] = { 0, 60, 105 };
	u32 crc[3][2];
	unsigned int k;

	if (!dnpr_smp_selftest || !dnpr_smp || !READ_ONCE(dnpr_ready))
		return 0;
	for (k = 0; k < 3; k++) {
		WRITE_ONCE(dnpr_last_pose_q, ~0ULL);
		dnpr_render_live_head(poses[k] * 1000U / 30U);
		crc[k][0] = crc32_le(~0U, (void *)dnpr_base, DNPR_BASE_BYTES);
	}
	dnpr_raster_start_helpers();
	for (k = 0; k < 3; k++) {
		u64 t0 = ktime_get_mono_fast_ns();
		bool same;

		WRITE_ONCE(dnpr_last_pose_q, ~0ULL);
		dnpr_render_live_head(poses[k] * 1000U / 30U);
		crc[k][1] = crc32_le(~0U, (void *)dnpr_base, DNPR_BASE_BYTES);
		same = crc[k][0] == crc[k][1];
		pr_info("DNPR SMP selftest pose %u: %llu us, helpers %u/%u, %s\n",
			poses[k],
			div64_u64(ktime_get_mono_fast_ns() - t0, 1000),
			atomic_read(&dnpr_raster_alive), dnpr_helper_count,
			same ? "identical" : "MISMATCH");
	}
	atomic_inc(&dnpr_raster_stop);
	dnpr_helpers_active = false;
	return 0;
}
late_initcall(dnpr_smp_selftest_run);

int __init dnpr_panic_init(void)
{
	if (dnpr_frame_meta.magic0 != 0x52504e44u || /* "DNPR" */
	    dnpr_frame_meta.magic1 != 0x3130304bu || /* "K001" */
	    dnpr_frame_meta.base_w != DNPR_WIDTH ||
	    dnpr_frame_meta.base_h != DNPR_HEIGHT ||
	    dnpr_frame_meta.star_frames != DNPR_STAR_FRAMES ||
	    dnpr_frame_meta.star_size != DNPR_SPRITE ||
	    dnpr_frame_meta.sparkle_frames != DNPR_SPARKLE_FRAMES ||
	    dnpr_frame_meta.sparkle_size != DNPR_SPRITE ||
	    dnpr_frame_meta.glyph_frames != DNPR_GLYPH_FRAMES ||
	    dnpr_frame_meta.glyph_size != DNPR_SPRITE)
		return -EINVAL;

	dnpr_base = vmalloc(DNPR_SCENE_BYTES);
	dnpr_row = kmalloc(DNPR_MAX_WIDTH * 4, GFP_KERNEL);
	dnpr_shared_frame = vmalloc(DNPR_BASE_BYTES * 2);
	dnpr_shared_dirty = vmalloc(DNPR_WIDTH * DNPR_HEIGHT);
	if (!dnpr_base || !dnpr_row)
		goto fail;
	if (zlib_inflate_blob(dnpr_base, DNPR_SCENE_BYTES, dnpr_frame_deflate,
			      sizeof(dnpr_frame_deflate)) != DNPR_SCENE_BYTES)
		goto fail;
	dnpr_layers = (u8 *)dnpr_base + DNPR_BASE_BYTES;
	dnpr_live_data = vmalloc(DNPR_LIVE_BYTES);
	dnpr_live_depth = vmalloc(DNPR_BASE_BYTES);
	dnpr_live_owner = vmalloc(DNPR_BASE_BYTES);
	if (!dnpr_live_data || !dnpr_live_depth || !dnpr_live_owner ||
	    zlib_inflate_blob(dnpr_live_data, DNPR_LIVE_BYTES, dnpr_live_deflate,
			      sizeof(dnpr_live_deflate)) != DNPR_LIVE_BYTES)
		goto fail;
	dnpr_spec_lut = vmalloc(1U << 24);
	if (dnpr_spec_lut &&
	    zlib_inflate_blob(dnpr_spec_lut, 1U << 24, dnpr_spec_deflate,
			      sizeof(dnpr_spec_deflate)) != (1U << 24)) {
		vfree(dnpr_spec_lut);
		dnpr_spec_lut = NULL;
	}
	if (!dnpr_spec_lut)
		pr_warn("DNPR highlight lookup unavailable; using arithmetic fallback\n");
	WRITE_ONCE(dnpr_ready, true);
	pr_info("DNPR panic scene preloaded (%zu KiB base + sprite layers); "
		"select drm.panic_screen=dnpr\n", DNPR_SCENE_BYTES / 1024);
	return 0;
fail:
	vfree(dnpr_spec_lut);
	dnpr_spec_lut = NULL;
	vfree(dnpr_live_depth);
	dnpr_live_depth = NULL;
	vfree(dnpr_live_owner);
	dnpr_live_owner = NULL;
	vfree(dnpr_live_data);
	dnpr_live_data = NULL;
	dnpr_layers = NULL;
	vfree(dnpr_base);
	dnpr_base = NULL;
	kfree(dnpr_row);
	dnpr_row = NULL;
	vfree(dnpr_shared_frame);
	dnpr_shared_frame = NULL;
	vfree(dnpr_shared_dirty);
	dnpr_shared_dirty = NULL;
	return -ENOMEM;
}

void dnpr_panic_exit(void)
{
	WRITE_ONCE(dnpr_ready, false);
	vfree(dnpr_spec_lut);
	dnpr_spec_lut = NULL;
	dnpr_sb_count = 0;
	dnpr_frame = 0;
	vfree(dnpr_live_depth);
	dnpr_live_depth = NULL;
	vfree(dnpr_live_owner);
	dnpr_live_owner = NULL;
	vfree(dnpr_live_data);
	dnpr_live_data = NULL;
	memset(dnpr_sb_valid, 0, sizeof(dnpr_sb_valid));
	dnpr_shared_valid = false;
	vfree(dnpr_base);
	dnpr_base = NULL;
	dnpr_layers = NULL;
	kfree(dnpr_row);
	dnpr_row = NULL;
	vfree(dnpr_shared_frame);
	dnpr_shared_frame = NULL;
	vfree(dnpr_shared_dirty);
	dnpr_shared_dirty = NULL;
}

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("DNPR preloaded DRM panic scene with shared live animation");
