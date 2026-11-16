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

#define DNPR_WIDTH 1920U
#define DNPR_HEIGHT 1080U
#define DNPR_MAX_WIDTH 4096U
#define DNPR_MAX_HEIGHT 2160U
#define DNPR_FASTEST_FRAME_NS 6944444ULL /* 144 Hz ceiling */
#define DNPR_AMBIENT_FRAME_NS 83333333ULL /* 12 fps texture clock */

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

/* Panic-time raster fan-out. One helper CPU owns the audio lifecycle and
 * every other CPU plus the orchestrator rasterize 32-row bands of the head
 * from an atomic queue. Rows never straddle CPUs, so the depth buffer needs
 * no locking and every band is deterministic from the same projected
 * vertices. The orchestrator's wait is always bounded by a deadline:
 * helpers that never service their IPI are pruned after the first miss and
 * the frame finishes on one CPU, so the worst case is the uniprocessor
 * draw. A straggler can only re-write rows with identical values.
 */
static bool dnpr_smp = true;
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

/* Quantized 3D geometry. Positions are signed Q8 model coordinates; RGB
 * lighting factors are unsigned Q8. Triangles are rasterized on every posed
 * panic frame, with depth testing, rather than replaying rendered images.
 */
struct dnpr_live_vertex {
	s16 x, y, z;
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
static struct dnpr_projected dnpr_projected[DNPR_LIVE_NV];
static u64 dnpr_last_pose_q = ~0ULL;
static struct {
	int x0, y0, x1, y1;
} dnpr_head_bounds, dnpr_head_dirty;

static s64 dnpr_edge(s32 ax, s32 ay, s32 bx, s32 by, s32 x, s32 y)
{
	return (s64)(x - ax) * (by - ay) - (s64)(y - ay) * (bx - ax);
}

static u16 dnpr_rgb565(unsigned int r, unsigned int g, unsigned int b)
{
	return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}

/* Divide by the triangle's invariant area without a per-pixel DIV. The
 * high half of n * floor(2^64 / area) is either the exact quotient or one
 * below it; the remainder check restores exact truncating division.
 */
static s64 dnpr_div_by_area(s64 n, u64 area, u64 reciprocal)
{
	u64 magnitude, quotient;

	if (area == 1)
		return n;
	magnitude = n < 0 ? 0 - (u64)n : (u64)n;
	quotient = mul_u64_u64_shr(magnitude, reciprocal, 64);
	if (magnitude - quotient * area >= area)
		quotient++;
	return n < 0 ? -(s64)quotient : (s64)quotient;
}

static unsigned int dnpr_specular(s64 w0, s64 w1, s64 w2, s64 area,
				 u64 reciprocal,
				 const struct dnpr_projected *a,
				 const struct dnpr_projected *b,
				 const struct dnpr_projected *c)
{
	s64 nx = dnpr_div_by_area(w0 * a->nx + w1 * b->nx + w2 * c->nx,
				    area, reciprocal);
	s64 ny = dnpr_div_by_area(w0 * a->ny + w1 * b->ny + w2 * c->ny,
				    area, reciprocal);
	s64 nz = dnpr_div_by_area(w0 * a->nz + w1 * b->nz + w2 * c->nz,
				    area, reciprocal);
	u64 length = int_sqrt64(nx * nx + ny * ny + nz * nz);
	s64 h = length ? (-59 * nx + 41 * ny + 106 * nz) / length : 0;
	u64 q;
	int i;

	if (h <= 0)
		return 0;
	q = min_t(u64, h, 127) * 65536 / 127;
	for (i = 0; i < 7; i++)
		q = q * q >> 16; /* h^128, matching the preview highlight */
	return 90 * q >> 16;
}

/* Conservative scanline intersection: the edge test below still decides
 * exact coverage, while these bounds avoid scanning empty triangle boxes.
 */
static void dnpr_span_edge(const struct dnpr_projected *a,
			   const struct dnpr_projected *b, s64 slope,
			   int y, s64 *left, s64 *right, bool *found)
{
	s64 xq;

	if (y < min(a->y, b->y) || y > max(a->y, b->y))
		return;
	xq = (s64)a->x * (1LL << 32) + (s64)(y - a->y) * slope;
	if (!*found) {
		*left = *right = xq;
		*found = true;
	} else {
		*left = min(*left, xq);
		*right = max(*right, xq);
	}
	if (a->y == b->y) {
		xq = (s64)b->x * (1LL << 32);
		*left = min(*left, xq);
		*right = max(*right, xq);
	}
}

static void dnpr_draw_live_triangle(const struct dnpr_projected *a,
				    const struct dnpr_projected *b,
				    const struct dnpr_projected *c,
				    const struct dnpr_live_triangle *tri,
				    int band_y0, int band_y1)
{
	s64 area = dnpr_edge(a->x, a->y, b->x, b->y, c->x, c->y);
	s64 slope_ab = 0, slope_bc = 0, slope_ca = 0;
	s64 step0, step1;
	u64 reciprocal = 0, remainder;
	int x0, x1, y0, y1, x, y, sign;

	if (!area)
		return;
	sign = area < 0 ? -1 : 1;
	area *= sign;
	if (area != 1) {
		reciprocal = div64_u64_rem(U64_MAX, area, &remainder);
		if (remainder == area - 1)
			reciprocal++;
	}
	y0 = clamp_t(int, min3(a->y, b->y, c->y), 0, DNPR_HEIGHT - 1);
	y1 = clamp_t(int, max3(a->y, b->y, c->y), 0, DNPR_HEIGHT - 1);
	if (y0 < band_y0)
		y0 = band_y0;
	if (y1 > band_y1)
		y1 = band_y1;
	if (y0 > y1)
		return; /* The band misses every row of this triangle. */
	x0 = clamp_t(int, min3(a->x, b->x, c->x), 0, DNPR_WIDTH - 1);
	x1 = clamp_t(int, max3(a->x, b->x, c->x), 0, DNPR_WIDTH - 1);
	if (a->y != b->y)
		slope_ab = div_s64((s64)(b->x - a->x) * (1LL << 32),
				 b->y - a->y);
	if (b->y != c->y)
		slope_bc = div_s64((s64)(c->x - b->x) * (1LL << 32),
				 c->y - b->y);
	if (c->y != a->y)
		slope_ca = div_s64((s64)(a->x - c->x) * (1LL << 32),
				 a->y - c->y);
	step0 = (s64)sign * (c->y - b->y);
	step1 = (s64)sign * (a->y - c->y);
	for (y = y0; y <= y1; y++) {
		s64 left = 0, right = 0, w0, w1;
		int span_x0, span_x1;
		bool found = false;

		dnpr_span_edge(a, b, slope_ab, y, &left, &right, &found);
		dnpr_span_edge(b, c, slope_bc, y, &left, &right, &found);
		dnpr_span_edge(c, a, slope_ca, y, &left, &right, &found);
		if (!found)
			continue;
		span_x0 = max(x0, (int)(left >> 32) - 2);
		span_x1 = min(x1, (int)(right >> 32) + 2);
		if (span_x0 > span_x1)
			continue;
		w0 = sign * dnpr_edge(b->x, b->y, c->x, c->y,
					span_x0, y);
		w1 = sign * dnpr_edge(c->x, c->y, a->x, a->y,
					span_x0, y);
		for (x = span_x0; x <= span_x1; x++) {
			s64 w2, weighted;
			unsigned int z, sr, sg, sb, spec, r, g, blue;
			size_t index;

			if (w0 < 0 || w1 < 0)
				goto next_pixel;
			w2 = area - w0 - w1;
			if (w2 < 0)
				goto next_pixel;
			weighted = w0 * a->z + w1 * b->z + w2 * c->z;
			z = dnpr_div_by_area(weighted, area, reciprocal);
			index = (size_t)y * DNPR_WIDTH + x;
			if (z >= dnpr_live_depth[index])
				goto next_pixel;
			dnpr_live_depth[index] = z;
			sr = dnpr_div_by_area(w0 * a->lr + w1 * b->lr + w2 * c->lr,
					  area, reciprocal);
			sg = dnpr_div_by_area(w0 * a->lg + w1 * b->lg + w2 * c->lg,
					  area, reciprocal);
			sb = dnpr_div_by_area(w0 * a->lb + w1 * b->lb + w2 * c->lb,
					  area, reciprocal);
			spec = dnpr_specular(w0, w1, w2, area, reciprocal,
					     a, b, c);
			r = min(255U, (tri->r * sr >> 8) + spec);
			g = min(255U, (tri->g * sg >> 8) + spec);
			blue = min(255U, (tri->blue * sb >> 8) + spec);
			dnpr_base[index] = dnpr_rgb565(r, g, blue);
	next_pixel:
			w0 += step0;
			w1 += step1;
		}
	}
}

static s32 dnpr_lerp_s32(s32 a, s32 b, unsigned int fraction)
{
	return a + ((s64)(b - a) * fraction >> 16);
}

static const struct dnpr_live_triangle *dnpr_tris(void)
{
	return (const void *)(dnpr_live_data + DNPR_BASE_BYTES);
}

static bool dnpr_render_live_head_prepare(unsigned int elapsed_ms)
{
	const struct dnpr_live_vertex *verts, *next_verts;
	u64 pose_q = min_t(u64, (u64)elapsed_ms * 30 * 65536 / 1000,
			   (u64)(DNPR_LIVE_NF - 1) << 16);
	unsigned int pose = pose_q >> 16, fraction = pose_q & 0xffff, i;
	int x0 = DNPR_WIDTH, y0 = DNPR_HEIGHT, x1 = 0, y1 = 0;

	if (pose_q == dnpr_last_pose_q)
		return false;
	dnpr_last_pose_q = pose_q;
	memcpy(dnpr_base, dnpr_live_data, DNPR_BASE_BYTES);
	memset(dnpr_live_depth, 0xff, DNPR_BASE_BYTES);
	verts = (const void *)(dnpr_live_data + DNPR_BASE_BYTES +
		DNPR_LIVE_NT * sizeof(struct dnpr_live_triangle) +
		pose * DNPR_LIVE_NV * sizeof(*verts));
	next_verts = (const void *)(dnpr_live_data + DNPR_BASE_BYTES +
		DNPR_LIVE_NT * sizeof(struct dnpr_live_triangle) +
		min(pose + 1, DNPR_LIVE_NF - 1) * DNPR_LIVE_NV * sizeof(*verts));
	for (i = 0; i < DNPR_LIVE_NV; i++) {
		s32 vx = dnpr_lerp_s32(verts[i].x, next_verts[i].x, fraction);
		s32 vy = dnpr_lerp_s32(verts[i].y, next_verts[i].y, fraction);
		s32 vz = dnpr_lerp_s32(verts[i].z, next_verts[i].z, fraction);
		int cam_z = 20 * 256 - vz;

		if (cam_z <= 0)
			cam_z = 1;
		dnpr_projected[i] = (struct dnpr_projected){
			.x = DNPR_WIDTH / 2 + div_s64((s64)vx * 1365, cam_z),
			.y = DNPR_HEIGHT / 2 - div_s64((s64)(vy - 2 * 256) * 1365, cam_z),
			.z = cam_z,
			.lr = dnpr_lerp_s32(verts[i].lr, next_verts[i].lr, fraction),
			.lg = dnpr_lerp_s32(verts[i].lg, next_verts[i].lg, fraction),
			.lb = dnpr_lerp_s32(verts[i].lb, next_verts[i].lb, fraction),
			.nx = dnpr_lerp_s32(verts[i].nx, next_verts[i].nx, fraction),
			.ny = dnpr_lerp_s32(verts[i].ny, next_verts[i].ny, fraction),
			.nz = dnpr_lerp_s32(verts[i].nz, next_verts[i].nz, fraction),
		};
		x0 = min(x0, dnpr_projected[i].x);
		y0 = min(y0, dnpr_projected[i].y);
		x1 = max(x1, dnpr_projected[i].x);
		y1 = max(y1, dnpr_projected[i].y);
	}
	x0 = clamp(x0 - 2, 0, (int)DNPR_WIDTH);
	y0 = clamp(y0 - 2, 0, (int)DNPR_HEIGHT);
	x1 = clamp(x1 + 3, 0, (int)DNPR_WIDTH);
	y1 = clamp(y1 + 3, 0, (int)DNPR_HEIGHT);
	dnpr_head_dirty.x0 = min(x0, dnpr_head_bounds.x0);
	dnpr_head_dirty.y0 = min(y0, dnpr_head_bounds.y0);
	dnpr_head_dirty.x1 = max(x1, dnpr_head_bounds.x1);
	dnpr_head_dirty.y1 = max(y1, dnpr_head_bounds.y1);
	dnpr_head_bounds.x0 = x0;
	dnpr_head_bounds.y0 = y0;
	dnpr_head_bounds.x1 = x1;
	dnpr_head_bounds.y1 = y1;
	return true;
}

/* Rasterize every triangle over the row range [y0, y1] inclusive. */
static void dnpr_raster_rows(int y0, int y1)
{
	const struct dnpr_live_triangle *t = dnpr_tris();
	unsigned int i;

	for (i = 0; i < DNPR_LIVE_NT; i++) {
		if (t[i].a >= DNPR_LIVE_NV || t[i].b >= DNPR_LIVE_NV ||
		    t[i].c >= DNPR_LIVE_NV)
			continue;
		dnpr_draw_live_triangle(&dnpr_projected[t[i].a],
					&dnpr_projected[t[i].b],
					&dnpr_projected[t[i].c], &t[i],
					y0, y1);
	}
}

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
	unsigned int seen = 0;

	(void)unused;
	atomic_inc(&dnpr_raster_alive);
	for (;;) {
		unsigned int gen;

		for (;;) {
			gen = READ_ONCE(dnpr_raster_gen);
			if (gen != seen || atomic_read(&dnpr_raster_stop))
				break;
			cpu_relax();
		}
		if (atomic_read(&dnpr_raster_stop))
			return;
		seen = gen;
		smp_rmb(); /* projected vertices published before the bump */
		if (governor)
			dnpr_audio_maybe_stop(); /* audio owns this CPU */
		else
			dnpr_grab_bands();
		atomic_inc(&dnpr_raster_done);
	}
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
	while ((done = atomic_read(&dnpr_raster_done)) < dnpr_helper_count &&
	       ktime_get_mono_fast_ns() < deadline)
		cpu_relax();
	if (done < dnpr_helper_count) {
		/* Prune unresponsive helpers once and finish solo; their
		 * late writes, if any, duplicate rows with equal values. */
		unsigned int alive = atomic_read(&dnpr_raster_alive);

		if (alive && alive < dnpr_helper_count)
			dnpr_helper_count = alive;
		dnpr_grab_bands();
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

static void dnpr_sample_base(unsigned int x, unsigned int y, unsigned int width,
			     unsigned int height, unsigned int *r,
			     unsigned int *g, unsigned int *b)
{
	unsigned int sy = (u64)y * DNPR_HEIGHT / height;
	unsigned int sx = (u64)x * DNPR_WIDTH / width;
	u16 p = dnpr_base[sy * DNPR_WIDTH + sx];

	*r = ((p >> 11) & 31) << 3 | ((p >> 11) & 31) >> 2;
	*g = ((p >> 5) & 63) << 2 | ((p >> 5) & 63) >> 4;
	*b = (p & 31) << 3 | (p & 31) >> 2;
}

static void dnpr_blend_sprite(const struct dnpr_sprite *sp, int x, int y,
			      unsigned int *r, unsigned int *g, unsigned int *b)
{
	unsigned int a, sx, sy;
	const u8 *px;

	if (sp->size < 1 || x < sp->left || x >= sp->left + sp->size ||
	    y < sp->top || y >= sp->top + sp->size)
		return;
	sx = (u64)(x - sp->left) * DNPR_SPRITE / sp->size;
	sy = (u64)(y - sp->top) * DNPR_SPRITE / sp->size;
	px = sp->src + ((size_t)sy * DNPR_SPRITE + sx) * 4;
	a = px[3];
	if (!a)
		return;
	if (a >= 255) {
		*r = px[0];
		*g = px[1];
		*b = px[2];
		return;
	}
	*r = (px[0] * a + *r * (255 - a)) / 255;
	*g = (px[1] * a + *g * (255 - a)) / 255;
	*b = (px[2] * a + *b * (255 - a)) / 255;
}

static void dnpr_place(struct dnpr_sprite *out, unsigned int *n, const u8 *src,
		       int center_x, int center_y, int size)
{
	out[(*n)++] = (struct dnpr_sprite){
		src, center_x - size / 2, center_y - size / 2, size };
}

/* Silver branch of star_with_particles: three sparkles then the star. */
static void dnpr_silver_block(struct dnpr_sprite *out, unsigned int *n,
			      const u8 *star, const u8 *sparkle,
			      unsigned int star_frame, unsigned int sparkle_frame,
			      int cx, int cy, int size)
{
	dnpr_place(out, n, sparkle + sparkle_frame * DNPR_SPRITE_BYTES,
		   cx + size / 2, cy - size / 4, size * 5 / 3);
	dnpr_place(out, n,
		   sparkle + ((sparkle_frame + 3) % DNPR_SPARKLE_FRAMES) * DNPR_SPRITE_BYTES,
		   cx - size / 2, cy + size / 4, size);
	dnpr_place(out, n,
		   sparkle + ((sparkle_frame + 1) % DNPR_SPARKLE_FRAMES) * DNPR_SPRITE_BYTES,
		   cx + size / 2, cy - size, size / 2);
	dnpr_place(out, n, star + star_frame * DNPR_SPRITE_BYTES, cx, cy, size);
}

/* Red branch: tinted sparkles sit at different offsets, and the third
 * sparkle keeps the base size instead of shrinking to half.
 */
static void dnpr_red_block(struct dnpr_sprite *out, unsigned int *n,
			   const u8 *star, const u8 *sparkle,
			   unsigned int star_frame, unsigned int sparkle_frame,
			   int cx, int cy, int size)
{
	dnpr_place(out, n, sparkle + sparkle_frame * DNPR_SPRITE_BYTES,
		   cx + size, cy - size / 2, size * 3 / 2);
	dnpr_place(out, n,
		   sparkle + ((sparkle_frame + 3) % DNPR_SPARKLE_FRAMES) * DNPR_SPRITE_BYTES,
		   cx - size, cy + size / 2, size);
	dnpr_place(out, n,
		   sparkle + ((sparkle_frame + 1) % DNPR_SPARKLE_FRAMES) * DNPR_SPRITE_BYTES,
		   cx + size * 2, cy, size);
	dnpr_place(out, n, star + star_frame * DNPR_SPRITE_BYTES, cx, cy, size);
}

/*
 * One live frame: star and sparkle textures advance on a 12 fps clock;
 * the prompt changes visibility every 15 frames of the 30 fps game clock.
 */
static unsigned int dnpr_place_sprites(unsigned int elapsed_ms, unsigned int width,
				       unsigned int height,
				       struct dnpr_sprite *out)
{
	unsigned int n = 0, star_size = height / 24;
	const u8 *star = dnpr_layers + DNPR_STAR_OFF;
	const u8 *red_star = dnpr_layers + DNPR_RED_STAR_OFF;
	const u8 *sparkle = dnpr_layers + DNPR_SPARKLE_OFF;
	const u8 *red_sparkle = dnpr_layers + DNPR_RED_SPARKLE_OFF;
	const u8 *glyph = dnpr_layers + DNPR_GLYPH_OFF;
	unsigned int sprite_tick = (u64)elapsed_ms * 12 / 1000;

	dnpr_silver_block(out, &n, star, sparkle, (sprite_tick + 1) % DNPR_STAR_FRAMES,
			  (sprite_tick + 4) % DNPR_SPARKLE_FRAMES,
			  width * 21 / 100, height * 97 / 100, star_size);
	dnpr_red_block(out, &n, red_star, red_sparkle, (sprite_tick + 5) % DNPR_STAR_FRAMES,
		       (sprite_tick + 2) % DNPR_SPARKLE_FRAMES,
		       width * 92 / 100, height * 16 / 100, star_size / 2);

	/* 15 SM64 frames = 500 ms at the original 30 fps. */
	if (!((elapsed_ms / 500) & 1)) {
		static const char *const lines[3] = { "REBOOT", "YOUR", "DEVICE" };
		unsigned int line, size = height / 21;
		int advance = size * 5 / 6;
		int x = width * 195 / 1000, y = height * 77 / 100, i;

		for (line = 0; line < 3; line++) {
			x = width * 195 / 1000;
			for (i = 0; lines[line][i]; i++) {
				const char *ch = lines[line] + i;
				const char *at = strchr(DNPR_PROMPT_ORDER, *ch);

				if (at)
					out[n++] = (struct dnpr_sprite){
						glyph + (at - DNPR_PROMPT_ORDER) * DNPR_SPRITE_BYTES,
						x, y, size };
				x += advance;
			}
			y += size + height / 120;
		}
	}
	return n;
}

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

static unsigned int dnpr_regions(unsigned int width, unsigned int height,
				 bool initial, bool head_changed,
				 struct dnpr_rect out[4])
{
	if (initial) {
		out[0] = (struct dnpr_rect){ 0, 0, width, height };
		return 1;
	}
	if (head_changed) {
		unsigned int x0 = clamp_t(int,
			dnpr_head_dirty.x0 * width / DNPR_WIDTH - 2, 0, width);
		unsigned int y0 = clamp_t(int,
			dnpr_head_dirty.y0 * height / DNPR_HEIGHT - 2, 0, height);
		unsigned int x1 = min(width,
			DIV_ROUND_UP(dnpr_head_dirty.x1 * width, DNPR_WIDTH) + 2);
		unsigned int y1 = min(height,
			DIV_ROUND_UP(dnpr_head_dirty.y1 * height, DNPR_HEIGHT) + 2);

		out[0] = (struct dnpr_rect){ x0, y0, x1, y1 };
	}
	/* Restore and redraw animated overlays; the rest remains on scanout. */
	out[head_changed] = (struct dnpr_rect){ 0, height * 70 / 100,
						   width / 2, height };
	out[head_changed + 1] = (struct dnpr_rect){ width * 80 / 100, 0,
						       width, height * 30 / 100 };
	out[head_changed + 2] = (struct dnpr_rect){ 0, 0,
						       min(width, 320U),
						       min(height, 4U) };
	return head_changed + 3;
}

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
static void dnpr_render_shared_rect(const struct dnpr_sprite *sprites,
				    unsigned int n, unsigned int frame,
				    const struct dnpr_rect *rect, bool initial)
{
	unsigned int x, y;
	unsigned int bar_w = ((frame % 40) + 1) * 8;

	for (y = rect->y0; y < rect->y1; y++) {
		const struct dnpr_sprite *active[DNPR_MAX_SPRITES];
		unsigned int row_n = 0, i;

		for (i = 0; i < n; i++)
			if (y >= sprites[i].top && y < sprites[i].top + sprites[i].size)
				active[row_n++] = &sprites[i];
		for (x = rect->x0; x < rect->x1; x++) {
			unsigned int r, g, b;
			u32 rgb;
			size_t index = (size_t)y * DNPR_WIDTH + x;

			dnpr_sample_base(x, y, DNPR_WIDTH, DNPR_HEIGHT, &r, &g, &b);
			for (i = 0; i < row_n; i++)
				dnpr_blend_sprite(active[i], x, y, &r, &g, &b);
			if (y < 4 && x < bar_w) {
				r = 255;
				g = 0;
				b = 255;
			}
			rgb = (r << 16) | (g << 8) | b;
			if (!initial && dnpr_shared_dirty &&
			    dnpr_shared_frame[index] != rgb)
				dnpr_shared_dirty[index] = 1;
			dnpr_shared_frame[index] = rgb;
		}
	}
}

static void dnpr_render_shared(unsigned int elapsed_ms, unsigned int frame,
			       bool initial, bool head_changed)
{
	struct dnpr_sprite sprites[DNPR_MAX_SPRITES];
	struct dnpr_rect rects[4];
	unsigned int n, nr, i;

	n = dnpr_place_sprites(elapsed_ms, DNPR_WIDTH, DNPR_HEIGHT, sprites);
	nr = dnpr_regions(DNPR_WIDTH, DNPR_HEIGHT, initial, head_changed, rects);
	if (dnpr_shared_dirty && !initial)
		memset(dnpr_shared_dirty, 0, DNPR_WIDTH * DNPR_HEIGHT);
	for (i = 0; i < nr; i++)
		dnpr_render_shared_rect(sprites, n, frame, &rects[i], initial);
	dnpr_shared_valid = true;
}

static void dnpr_blit_shared(struct drm_scanout_buffer *sb, bool initial,
			     bool head_changed)
{
	struct dnpr_rect rects[4];
	unsigned int nr, i, x, y, cpp = sb->format->cpp[0];
	void (*set_pixel)(struct drm_scanout_buffer *, unsigned int, unsigned int,
			  u32) = sb->set_pixel;

	nr = dnpr_regions(DNPR_WIDTH, DNPR_HEIGHT, initial, head_changed, rects);
	for (i = 0; i < nr; i++) {
		for (y = rects[i].y0; y < rects[i].y1; y++) {
			for (x = rects[i].x0; x < rects[i].x1; x++) {
				size_t index = (size_t)y * DNPR_WIDTH + x;
				u32 rgb, color;

				if (set_pixel && !initial && dnpr_shared_dirty &&
				    !dnpr_shared_dirty[index])
					continue;
				rgb = dnpr_shared_frame[index];
				color = drm_draw_color_from_xrgb8888(rgb,
								 sb->format->format);

				if (set_pixel)
					set_pixel(sb, x, y, color);
				else
					memcpy(dnpr_row + x * cpp, &color, cpp);
			}
			if (!set_pixel)
				iosys_map_memcpy_to(&sb->map[0],
					(size_t)y * sb->pitch[0] + rects[i].x0 * cpp,
					dnpr_row + rects[i].x0 * cpp,
					(rects[i].x1 - rects[i].x0) * cpp);
		}
	}
}

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
	/* panic() stopped the other CPUs before the DRM kmsg dumper reached us.
	 * x86 marks them offline, so the helper startup would return without
	 * dispatching work. Keep the boot-time SMP selftest separate.
	 */
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
		next_ns = now_ns +
			(elapsed_ms < DIV_ROUND_UP((DNPR_LIVE_NF - 1) * 1000, 30) ?
			 DNPR_FASTEST_FRAME_NS : DNPR_AMBIENT_FRAME_NS);
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
	if (!dnpr_live_data || !dnpr_live_depth ||
	    zlib_inflate_blob(dnpr_live_data, DNPR_LIVE_BYTES, dnpr_live_deflate,
			      sizeof(dnpr_live_deflate)) != DNPR_LIVE_BYTES)
		goto fail;
	WRITE_ONCE(dnpr_ready, true);
	pr_info("DNPR panic scene preloaded (%zu KiB base + sprite layers); "
		"select drm.panic_screen=dnpr\n", DNPR_SCENE_BYTES / 1024);
	return 0;
fail:
	vfree(dnpr_live_depth);
	dnpr_live_depth = NULL;
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
	dnpr_sb_count = 0;
	dnpr_frame = 0;
	vfree(dnpr_live_depth);
	dnpr_live_depth = NULL;
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
