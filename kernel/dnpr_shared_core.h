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
			u16 p = dnpr_base[index];

			r = ((p >> 11) & 31) << 3 | ((p >> 11) & 31) >> 2;
			g = ((p >> 5) & 63) << 2 | ((p >> 5) & 63) >> 4;
			b = (p & 31) << 3 | (p & 31) >> 2;
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
