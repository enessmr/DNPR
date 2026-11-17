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
