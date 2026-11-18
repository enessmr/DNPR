#ifndef DNPR_BLIT_ACCOUNT
#define DNPR_BLIT_ACCOUNT(bytes, calls) do { } while (0)
#endif

static void dnpr_blit_shared(struct drm_scanout_buffer *sb, bool initial,
			     bool head_changed)
{
	unsigned int i, x, y, cpp = sb->format->cpp[0];
	void (*set_pixel)(struct drm_scanout_buffer *, unsigned int, unsigned int,
			  u32) = sb->set_pixel;
	void (*set_pixels)(struct drm_scanout_buffer *, unsigned int, unsigned int,
			   const u32 *, unsigned int) = sb->set_pixels;
	bool native_xrgb8888 = sb->format->format == DRM_FORMAT_XRGB8888;

	(void)initial;
	(void)head_changed;
	for (i = 0; i < dnpr_shared_rect_count; i++) {
		for (y = dnpr_shared_rects[i].y0;
		     y < dnpr_shared_rects[i].y1; y++) {
			if (set_pixels) {
				x = dnpr_shared_rects[i].x0;
				while (x < dnpr_shared_rects[i].x1) {
					size_t index = (size_t)y * DNPR_WIDTH + x;
					unsigned int run, j;

					if (!initial && dnpr_shared_dirty &&
					    !dnpr_shared_dirty[index]) {
						x++;
						continue;
					}
					run = x;
					while (x < dnpr_shared_rects[i].x1) {
						index = (size_t)y * DNPR_WIDTH + x;
						if (!initial && dnpr_shared_dirty &&
						    !dnpr_shared_dirty[index])
							break;
						x++;
					}
					if (native_xrgb8888) {
						set_pixels(sb, run, y,
							   &dnpr_shared_frame[(size_t)y *
									      DNPR_WIDTH + run],
							   x - run);
						DNPR_BLIT_ACCOUNT((size_t)(x - run) * cpp, 1);
					} else {
						for (j = run; j < x; j++) {
							u32 rgb = dnpr_shared_frame[(size_t)y *
										 DNPR_WIDTH + j];
							u32 color = drm_draw_color_from_xrgb8888(
									rgb, sb->format->format);

							memcpy(dnpr_row + j * cpp, &color, cpp);
						}
						set_pixels(sb, run, y,
							   (const u32 *)(dnpr_row + run * cpp),
							   x - run);
						DNPR_BLIT_ACCOUNT((size_t)(x - run) * cpp, 1);
					}
				}
				continue;
			}
			if (!set_pixel && native_xrgb8888) {
				unsigned int x0 = dnpr_shared_rects[i].x0;
				unsigned int x1 = dnpr_shared_rects[i].x1;

				iosys_map_memcpy_to(&sb->map[0],
					(size_t)y * sb->pitch[0] + x0 * cpp,
					&dnpr_shared_frame[(size_t)y * DNPR_WIDTH + x0],
					(x1 - x0) * cpp);
				DNPR_BLIT_ACCOUNT((size_t)(x1 - x0) * cpp, 1);
				continue;
			}
			for (x = dnpr_shared_rects[i].x0;
			     x < dnpr_shared_rects[i].x1; x++) {
				size_t index = (size_t)y * DNPR_WIDTH + x;
				u32 rgb, color;

				if (set_pixel && !initial && dnpr_shared_dirty &&
				    !dnpr_shared_dirty[index])
					continue;
				rgb = dnpr_shared_frame[index];
				color = native_xrgb8888 ? rgb :
					drm_draw_color_from_xrgb8888(rgb,
								     sb->format->format);

				if (set_pixel) {
					set_pixel(sb, x, y, color);
					DNPR_BLIT_ACCOUNT(cpp, 1);
				} else {
					memcpy(dnpr_row + x * cpp, &color, cpp);
				}
			}
			if (!set_pixel) {
				unsigned int x0 = dnpr_shared_rects[i].x0;
				unsigned int x1 = dnpr_shared_rects[i].x1;

				iosys_map_memcpy_to(&sb->map[0],
					(size_t)y * sb->pitch[0] + x0 * cpp,
					dnpr_row + x0 * cpp, (x1 - x0) * cpp);
				DNPR_BLIT_ACCOUNT((size_t)(x1 - x0) * cpp, 1);
			}
		}
	}
}
