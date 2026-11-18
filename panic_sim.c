#include "panic_sim.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>

typedef uint8_t u8;
typedef int8_t s8;
typedef uint16_t u16;
typedef int16_t s16;
typedef uint32_t u32;
typedef int32_t s32;
typedef uint64_t u64;
typedef int64_t s64;

#define __packed __attribute__((packed))
#define U64_MAX UINT64_MAX
#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min(min(a, b), c)
#define max3(a, b, c) max(max(a, b), c)
#define min_t(type, a, b) min((type)(a), (type)(b))
#define clamp(value, low, high) min(max(value, low), high)
#define clamp_t(type, value, low, high) clamp((type)(value), (type)(low), (type)(high))
#define DIV_ROUND_UP(value, divisor) (((value) + (divisor) - 1) / (divisor))
#define ARRAY_SIZE(array) (sizeof(array) / sizeof((array)[0]))
#define div_s64(value, divisor) ((value) / (divisor))
#define div64_u64(value, divisor) ((value) / (divisor))

static u64 div64_u64_rem(u64 value, u64 divisor, u64 *remainder)
{
	*remainder = value % divisor;
	return value / divisor;
}

static u64 mul_u64_u64_shr(u64 first, u64 second, unsigned int shift)
{
	return (u64)(((__uint128_t)first * second) >> shift);
}

static u64 int_sqrt64(u64 value)
{
	u64 root = (u64)sqrtl((long double)value);

	while (root && root > value / root)
		root--;
	while (root < UINT32_MAX && root + 1 <= value / (root + 1))
		root++;
	return root;
}

#include "kernel/dnpr_frame_blob.h"
#include "kernel/dnpr_live_blob.h"
#include "kernel/dnpr_spec_blob.h"

#define DNPR_WIDTH 1920U
#define DNPR_HEIGHT 1080U
#define DNPR_BAND_H 32U
#define DNPR_BAND_COUNT DIV_ROUND_UP(DNPR_HEIGHT, DNPR_BAND_H)
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
#define DNPR_MAX_SPRITES 24U
#define DNPR_PROMPT_ORDER "REBOTYUDVIC"
#define DNPR_LIVE_BYTES (DNPR_BASE_BYTES + \
	DNPR_LIVE_NT * sizeof(struct dnpr_live_triangle) + \
	DNPR_LIVE_NF * DNPR_LIVE_NV * sizeof(struct dnpr_live_vertex))

struct dnpr_sprite {
	const u8 *src;
	int left;
	int top;
	int size;
};

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

struct dnpr_map {
	u8 *vaddr;
};

struct dnpr_format {
	u32 format;
	unsigned int cpp[1];
};

struct drm_scanout_buffer {
	unsigned int width, height, pitch[1];
	const struct dnpr_format *format;
	struct dnpr_map map[1];
	void (*set_pixel)(struct drm_scanout_buffer *, unsigned int, unsigned int, u32);
	void (*set_pixels)(struct drm_scanout_buffer *, unsigned int, unsigned int,
			   const u32 *, unsigned int);
};

static u32 drm_draw_color_from_xrgb8888(u32 color, u32 format)
{
	(void)format;
	return color;
}

static void iosys_map_memcpy_to(struct dnpr_map *map, size_t offset,
				const void *source, size_t bytes)
{
	memcpy(map->vaddr + offset, source, bytes);
}

static u16 *dnpr_base;
static u8 *dnpr_layers;
static u8 *dnpr_row;
static u32 *dnpr_shared_frame;
static u8 *dnpr_shared_dirty;
static bool dnpr_shared_valid;
static struct dnpr_rect dnpr_shared_rects[DNPR_MAX_SPRITES * 2 + 3];
static unsigned int dnpr_shared_rect_count;
static struct dnpr_sprite dnpr_shared_prev_sprites[DNPR_MAX_SPRITES];
static unsigned int dnpr_shared_prev_sprite_count;
static unsigned int dnpr_shared_prev_bar_w;
static bool dnpr_shared_prev_valid;
static u8 *dnpr_live_data;
static u16 *dnpr_live_depth;
static u16 *dnpr_live_owner;
static u8 *dnpr_spec_lut;
static struct dnpr_projected dnpr_projected[DNPR_LIVE_NV];
static u16 dnpr_triangle_order[DNPR_LIVE_NT];
static u16 dnpr_triangle_depths[DNPR_LIVE_NT];
static u16 dnpr_band_triangles[DNPR_BAND_COUNT][DNPR_LIVE_NT];
static u16 dnpr_band_triangle_count[DNPR_BAND_COUNT];
static bool dnpr_unshaded;
static bool dnpr_no_specular;
#define DNPR_UNSHADED dnpr_unshaded
#define DNPR_NO_SPECULAR dnpr_no_specular
static u64 dnpr_last_pose_q = U64_MAX;
static struct {
	int x0, y0, x1, y1;
} dnpr_head_bounds, dnpr_head_dirty;
static bool dnpr_head_bounds_valid;

#include "kernel/dnpr_raster_core.h"
#include "kernel/dnpr_raster_rows.h"
#include "kernel/dnpr_sprite_core.h"
#include "kernel/dnpr_regions_core.h"
#include "kernel/dnpr_shared_core.h"
#include "kernel/dnpr_blit_core.h"

void panic_sim_set_unshaded(bool enabled)
{
	dnpr_unshaded = enabled;
}

void panic_sim_set_no_specular(bool enabled)
{
	dnpr_no_specular = enabled;
}

static struct dnpr_format sim_format = { .cpp = { 4 } };
static struct drm_scanout_buffer sim_scanouts[2];
static bool sim_scanout_owned[2];

#define SIM_BAND_H 32U
#define SIM_BAND_COUNT DIV_ROUND_UP(DNPR_HEIGHT, SIM_BAND_H)
#define SIM_MAX_WORKERS 15U
static pthread_t sim_workers[SIM_MAX_WORKERS];
static pthread_mutex_t sim_raster_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t sim_raster_start = PTHREAD_COND_INITIALIZER;
static pthread_cond_t sim_raster_done = PTHREAD_COND_INITIALIZER;
static atomic_uint sim_next_band;
static unsigned int sim_requested_threads;
static unsigned int sim_worker_count;
static unsigned int sim_completed;
static unsigned int sim_generation;
static bool sim_stop;

void panic_sim_set_threads(unsigned int threads)
{
	sim_requested_threads = clamp(threads, 1U, SIM_MAX_WORKERS + 1U);
}

unsigned int panic_sim_thread_count(void)
{
	return sim_worker_count + 1;
}

static void sim_raster_bands(void)
{
	for (;;) {
		unsigned int band = atomic_fetch_add(&sim_next_band, 1);
		int y0;

		if (band >= SIM_BAND_COUNT)
			return;
		y0 = band * SIM_BAND_H;
		dnpr_raster_rows(y0, min(y0 + SIM_BAND_H - 1,
					       DNPR_HEIGHT - 1));
	}
}

static void *sim_raster_worker(void *unused)
{
	unsigned int seen = 0;

	(void)unused;
	pthread_mutex_lock(&sim_raster_lock);
	for (;;) {
		while (!sim_stop && seen == sim_generation)
			pthread_cond_wait(&sim_raster_start, &sim_raster_lock);
		if (sim_stop)
			break;
		seen = sim_generation;
		pthread_mutex_unlock(&sim_raster_lock);
		sim_raster_bands();
		pthread_mutex_lock(&sim_raster_lock);
		sim_completed++;
		pthread_cond_signal(&sim_raster_done);
	}
	pthread_mutex_unlock(&sim_raster_lock);
	return NULL;
}

static void sim_raster_draw(void)
{
	if (!sim_worker_count) {
		dnpr_raster_rows(0, DNPR_HEIGHT - 1);
		return;
	}
	pthread_mutex_lock(&sim_raster_lock);
	sim_completed = 0;
	atomic_store(&sim_next_band, 0);
	sim_generation++;
	pthread_cond_broadcast(&sim_raster_start);
	pthread_mutex_unlock(&sim_raster_lock);
	sim_raster_bands();
	pthread_mutex_lock(&sim_raster_lock);
	while (sim_completed < sim_worker_count)
		pthread_cond_wait(&sim_raster_done, &sim_raster_lock);
	pthread_mutex_unlock(&sim_raster_lock);
}

static void sim_set_pixel(struct drm_scanout_buffer *scanout,
			  unsigned int x, unsigned int y, u32 color)
{
	u32 columns = DIV_ROUND_UP(scanout->pitch[0], 64U);
	u32 block_x = (x * 4) / 64;
	u32 block_y = y / 128;
	u32 block_offset = (block_y * columns + block_x) * (64 * 128);
	u32 offset = block_offset + (x % 16) * 4 + (y % 128) * 64;

	memcpy(scanout->map[0].vaddr + offset, &color, sizeof(color));
}

static bool inflate_blob(void *destination, size_t size,
			 const u8 *source, size_t source_size)
{
	z_stream stream = { 0 };
	int result;

	stream.next_in = (Bytef *)source;
	stream.avail_in = source_size;
	stream.next_out = destination;
	stream.avail_out = size;
	if (inflateInit2(&stream, -15) != Z_OK)
		return false;
	result = inflate(&stream, Z_FINISH);
	inflateEnd(&stream);
	return result == Z_STREAM_END && stream.total_out == size;
}

void panic_sim_destroy(void)
{
	pthread_mutex_lock(&sim_raster_lock);
	sim_stop = true;
	pthread_cond_broadcast(&sim_raster_start);
	pthread_mutex_unlock(&sim_raster_lock);
	for (unsigned int index = 0; index < sim_worker_count; index++)
		pthread_join(sim_workers[index], NULL);
	sim_worker_count = 0;
	for (unsigned int index = 0; index < 2; index++) {
		if (sim_scanout_owned[index])
			free(sim_scanouts[index].map[0].vaddr);
		sim_scanouts[index].map[0].vaddr = NULL;
		sim_scanout_owned[index] = false;
	}
	free(dnpr_spec_lut);
	free(dnpr_live_depth);
	free(dnpr_live_owner);
	free(dnpr_live_data);
	free(dnpr_shared_dirty);
	free(dnpr_shared_frame);
	free(dnpr_row);
	free(dnpr_base);
	dnpr_spec_lut = NULL;
	dnpr_live_depth = NULL;
	dnpr_live_owner = NULL;
	dnpr_live_data = NULL;
	dnpr_shared_dirty = NULL;
	dnpr_shared_frame = NULL;
	dnpr_row = NULL;
	dnpr_base = NULL;
	dnpr_shared_valid = false;
	dnpr_last_pose_q = U64_MAX;
}

bool panic_sim_init(void)
{
	size_t scanout_size = (size_t)DNPR_WIDTH * 4 * 1152;

	if (dnpr_frame_meta.base_w != DNPR_WIDTH ||
	    dnpr_frame_meta.base_h != DNPR_HEIGHT)
		return false;
	dnpr_base = malloc(DNPR_SCENE_BYTES);
	dnpr_row = malloc(DNPR_WIDTH * 4);
	dnpr_shared_frame = malloc(DNPR_WIDTH * DNPR_HEIGHT * sizeof(u32));
	dnpr_shared_dirty = malloc(DNPR_WIDTH * DNPR_HEIGHT);
	dnpr_live_data = malloc(DNPR_LIVE_BYTES);
	dnpr_live_depth = malloc(DNPR_BASE_BYTES);
	dnpr_live_owner = malloc(DNPR_BASE_BYTES);
	dnpr_spec_lut = malloc(1U << 24);
	for (unsigned int index = 0; index < 2; index++) {
		sim_scanouts[index] = (struct drm_scanout_buffer){
			.width = DNPR_WIDTH,
			.height = DNPR_HEIGHT,
			.pitch = { DNPR_WIDTH * 4 },
			.format = &sim_format,
			.set_pixel = sim_set_pixel,
		};
		sim_scanouts[index].map[0].vaddr = malloc(scanout_size);
		sim_scanout_owned[index] = true;
	}
	if (!dnpr_base || !dnpr_row || !dnpr_shared_frame ||
	    !dnpr_shared_dirty || !dnpr_live_data || !dnpr_live_depth ||
	    !dnpr_live_owner ||
	    !dnpr_spec_lut || !sim_scanouts[0].map[0].vaddr ||
	    !sim_scanouts[1].map[0].vaddr ||
	    !inflate_blob(dnpr_base, DNPR_SCENE_BYTES, dnpr_frame_deflate,
			 sizeof(dnpr_frame_deflate)) ||
	    !inflate_blob(dnpr_live_data, DNPR_LIVE_BYTES, dnpr_live_deflate,
			 sizeof(dnpr_live_deflate)) ||
	    !inflate_blob(dnpr_spec_lut, 1U << 24, dnpr_spec_deflate,
			 sizeof(dnpr_spec_deflate))) {
		panic_sim_destroy();
		return false;
	}
	dnpr_layers = (u8 *)dnpr_base + DNPR_BASE_BYTES;
	long cpus = sysconf(_SC_NPROCESSORS_ONLN);
	unsigned int threads = sim_requested_threads ? sim_requested_threads :
		(cpus > 0 ? min((unsigned long)cpus,
				SIM_MAX_WORKERS + 1U) : 1U);
	sim_stop = false;
	sim_generation = 0;
	sim_completed = 0;
	for (unsigned int index = 0; index + 1 < threads; index++) {
		if (pthread_create(&sim_workers[index], NULL,
				   sim_raster_worker, NULL))
			break;
		sim_worker_count++;
	}
	return true;
}

bool panic_sim_set_scanout(unsigned int index, void *map, size_t size,
			   unsigned int pitch)
{
	size_t required;

	if (index >= 2 || !map || pitch != DNPR_WIDTH * 4)
		return false;
	required = (size_t)DIV_ROUND_UP(pitch, 64U) * 64 * 1152;
	if (size < required)
		return false;
	if (sim_scanout_owned[index])
		free(sim_scanouts[index].map[0].vaddr);
	sim_scanouts[index].map[0].vaddr = map;
	sim_scanout_owned[index] = false;
	return true;
}

static double sim_now_ms(void)
{
	struct timespec now;

	clock_gettime(CLOCK_MONOTONIC, &now);
	return now.tv_sec * 1000.0 + now.tv_nsec / 1000000.0;
}

void panic_sim_render_timed(unsigned int elapsed_ms, unsigned int frame,
			    uint32_t *pixels, struct panic_sim_timing *timing)
{
	bool initial = !dnpr_shared_valid;
	double begin = timing ? sim_now_ms() : 0;
	bool head_changed = dnpr_render_live_head_prepare(elapsed_ms);
	double after_prepare = timing ? sim_now_ms() : 0;

	if (head_changed)
		sim_raster_draw();
	double after_raster = timing ? sim_now_ms() : 0;
	dnpr_render_shared(elapsed_ms, frame, initial, head_changed);
	double after_compose = timing ? sim_now_ms() : 0;
	for (unsigned int index = 0; index < 2; index++)
		dnpr_blit_shared(&sim_scanouts[index], initial, head_changed);
	double after_blit = timing ? sim_now_ms() : 0;
	for (size_t index = 0; index < (size_t)DNPR_WIDTH * DNPR_HEIGHT; index++)
		pixels[index] = dnpr_shared_frame[index] | 0xff000000u;
	if (timing) {
		*timing = (struct panic_sim_timing){
			.head_changed = head_changed,
			.prepare_ms = after_prepare - begin,
			.raster_ms = after_raster - after_prepare,
			.compose_ms = after_compose - after_raster,
			.blit_ms = after_blit - after_compose,
			.copy_ms = sim_now_ms() - after_blit,
		};
	}
}

void panic_sim_render(unsigned int elapsed_ms, unsigned int frame,
			      uint32_t *pixels)
{
	panic_sim_render_timed(elapsed_ms, frame, pixels, NULL);
}
