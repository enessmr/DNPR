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
#include <linux/errno.h>
#include <linux/delay.h>
#include <linux/init.h>
#include <linux/iosys-map.h>
#include <linux/ktime.h>
#include <linux/math.h>
#include <linux/math64.h>
#include <linux/minmax.h>
#include <linux/module.h>
#include <linux/printk.h>
#include <linux/slab.h>
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
static bool dnpr_ready;

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

#define DNPR_LIVE_BYTES (DNPR_BASE_BYTES + \
	DNPR_LIVE_NT * sizeof(struct dnpr_live_triangle) + \
	DNPR_LIVE_NF * DNPR_LIVE_NV * sizeof(struct dnpr_live_vertex))

static u8 *dnpr_live_data;
static u16 *dnpr_live_depth;
static struct dnpr_projected dnpr_projected[DNPR_LIVE_NV];
static unsigned int dnpr_last_pose = ~0U;

static s64 dnpr_edge(s32 ax, s32 ay, s32 bx, s32 by, s32 x, s32 y)
{
	return (s64)(x - ax) * (by - ay) - (s64)(y - ay) * (bx - ax);
}

static u16 dnpr_rgb565(unsigned int r, unsigned int g, unsigned int b)
{
	return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}

static unsigned int dnpr_specular(s64 w0, s64 w1, s64 w2, s64 area,
				 const struct dnpr_projected *a,
				 const struct dnpr_projected *b,
				 const struct dnpr_projected *c)
{
	s64 nx = div64_s64(w0 * a->nx + w1 * b->nx + w2 * c->nx, area);
	s64 ny = div64_s64(w0 * a->ny + w1 * b->ny + w2 * c->ny, area);
	s64 nz = div64_s64(w0 * a->nz + w1 * b->nz + w2 * c->nz, area);
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

static void dnpr_draw_live_triangle(const struct dnpr_projected *a,
				    const struct dnpr_projected *b,
				    const struct dnpr_projected *c,
				    const struct dnpr_live_triangle *tri)
{
	s64 area = dnpr_edge(a->x, a->y, b->x, b->y, c->x, c->y);
	int x0, x1, y0, y1, x, y, sign;

	if (!area)
		return;
	sign = area < 0 ? -1 : 1;
	area *= sign;
	x0 = clamp_t(int, min3(a->x, b->x, c->x), 0, DNPR_WIDTH - 1);
	x1 = clamp_t(int, max3(a->x, b->x, c->x), 0, DNPR_WIDTH - 1);
	y0 = clamp_t(int, min3(a->y, b->y, c->y), 0, DNPR_HEIGHT - 1);
	y1 = clamp_t(int, max3(a->y, b->y, c->y), 0, DNPR_HEIGHT - 1);
	for (y = y0; y <= y1; y++) {
		for (x = x0; x <= x1; x++) {
			s64 w0 = sign * dnpr_edge(b->x, b->y, c->x, c->y, x, y);
			s64 w1 = sign * dnpr_edge(c->x, c->y, a->x, a->y, x, y);
			s64 w2, weighted;
			unsigned int z, sr, sg, sb, spec, r, g, blue;
			size_t index;

			if (w0 < 0 || w1 < 0)
				continue;
			w2 = area - w0 - w1;
			if (w2 < 0)
				continue;
			weighted = w0 * a->z + w1 * b->z + w2 * c->z;
			z = div64_s64(weighted, area);
			index = (size_t)y * DNPR_WIDTH + x;
			if (z >= dnpr_live_depth[index])
				continue;
			dnpr_live_depth[index] = z;
			sr = div64_s64(w0 * a->lr + w1 * b->lr + w2 * c->lr, area);
			sg = div64_s64(w0 * a->lg + w1 * b->lg + w2 * c->lg, area);
			sb = div64_s64(w0 * a->lb + w1 * b->lb + w2 * c->lb, area);
			spec = dnpr_specular(w0, w1, w2, area, a, b, c);
			r = min(255U, (tri->r * sr >> 8) + spec);
			g = min(255U, (tri->g * sg >> 8) + spec);
			blue = min(255U, (tri->blue * sb >> 8) + spec);
			dnpr_base[index] = dnpr_rgb565(r, g, blue);
		}
	}
}

static void dnpr_render_live_head(unsigned int elapsed_ms)
{
	const struct dnpr_live_triangle *tris;
	const struct dnpr_live_vertex *verts;
	unsigned int pose = min_t(u64, (u64)elapsed_ms * 30 / 1000,
				  DNPR_LIVE_NF - 1), i;

	if (pose == dnpr_last_pose)
		return;
	dnpr_last_pose = pose;
	memcpy(dnpr_base, dnpr_live_data, DNPR_BASE_BYTES);
	memset(dnpr_live_depth, 0xff, DNPR_BASE_BYTES);
	tris = (const void *)(dnpr_live_data + DNPR_BASE_BYTES);
	verts = (const void *)(dnpr_live_data + DNPR_BASE_BYTES +
		DNPR_LIVE_NT * sizeof(*tris) + pose * DNPR_LIVE_NV * sizeof(*verts));
	for (i = 0; i < DNPR_LIVE_NV; i++) {
		int cam_z = 20 * 256 - verts[i].z;

		if (cam_z <= 0)
			cam_z = 1;
		dnpr_projected[i] = (struct dnpr_projected){
			.x = DNPR_WIDTH / 2 + div_s64((s64)verts[i].x * 1365, cam_z),
			.y = DNPR_HEIGHT / 2 - div_s64((s64)(verts[i].y - 2 * 256) * 1365, cam_z),
			.z = cam_z,
			.lr = verts[i].lr, .lg = verts[i].lg, .lb = verts[i].lb,
			.nx = verts[i].nx, .ny = verts[i].ny, .nz = verts[i].nz,
		};
	}
	for (i = 0; i < DNPR_LIVE_NT; i++) {
		const struct dnpr_live_triangle *t = &tris[i];

		if (t->a >= DNPR_LIVE_NV || t->b >= DNPR_LIVE_NV ||
		    t->c >= DNPR_LIVE_NV)
			continue;
		dnpr_draw_live_triangle(&dnpr_projected[t->a],
					&dnpr_projected[t->b],
					&dnpr_projected[t->c], t);
	}
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

static void dnpr_composite(struct drm_scanout_buffer *sb,
			   unsigned int elapsed_ms, unsigned int frame)
{
	struct dnpr_sprite sprites[DNPR_MAX_SPRITES];
	unsigned int n;
	unsigned int width = sb->width, height = sb->height;

	dnpr_render_live_head(elapsed_ms);
	n = dnpr_place_sprites(elapsed_ms, width, height, sprites);
	if (elapsed_ms <= 4500) {
		dnpr_composite_rect(sb, sprites, n, frame, 0, 0, width, height);
		return;
	}
	/* Once Mario reaches the sigh pose, only these overlay regions change.
	 * Rebuild them from the cached base instead of rewriting both screens.
	 */
	dnpr_composite_rect(sb, sprites, n, frame, 0, height * 70 / 100,
			    width / 2, height);
	dnpr_composite_rect(sb, sprites, n, frame, width * 80 / 100, 0,
			    width, height * 30 / 100);
	dnpr_composite_rect(sb, sprites, n, frame, 0, 0,
			    min(width, 320U), min(height, 4U));
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
	dnpr_composite(&dnpr_sb[idx], 0, 0);
	wmb();
	return true;
}

void dnpr_panic_animate(void)
{
	u64 start_ns;
	unsigned int k, elapsed_ms;

	if (!READ_ONCE(dnpr_sb_count))
		return;
	start_ns = ktime_get_mono_fast_ns();
	dnpr_audio_start();
	for (;;) {
		mdelay(80);
		elapsed_ms = min_t(u64,
			(ktime_get_mono_fast_ns() - start_ns) / 1000000,
			~0U);
		dnpr_frame++;
		for (k = 0; k < dnpr_sb_count; k++)
			if (READ_ONCE(dnpr_sb_valid[k]))
				dnpr_composite(&dnpr_sb[k], elapsed_ms, dnpr_frame);
		wmb();
		dnpr_audio_maybe_stop();
	}
}

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
	vfree(dnpr_base);
	dnpr_base = NULL;
	dnpr_layers = NULL;
	kfree(dnpr_row);
	dnpr_row = NULL;
}

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("DNPR preloaded DRM panic scene with shared live animation");
