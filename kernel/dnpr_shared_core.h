static u32 dnpr_rgb565_to_xrgb8888[1U << 16];

static void dnpr_prepare_rgb565_lut(void)
{
	unsigned int p;

	for (p = 0; p < ARRAY_SIZE(dnpr_rgb565_to_xrgb8888); p++) {
		unsigned int r = (p >> 11) & 31;
		unsigned int g = (p >> 5) & 63;
		unsigned int b = p & 31;

		dnpr_rgb565_to_xrgb8888[p] =
			(((r << 3) | (r >> 2)) << 16) |
			(((g << 2) | (g >> 4)) << 8) |
			((b << 3) | (b >> 2));
	}
}

static bool dnpr_shared_rects_touch(const struct dnpr_rect *a,
				    const struct dnpr_rect *b)
{
	return a->x0 <= b->x1 && b->x0 <= a->x1 &&
	       a->y0 <= b->y1 && b->y0 <= a->y1;
}

static void dnpr_shared_add_rect(struct dnpr_rect rect)
{
	unsigned int i;

	if (rect.x0 >= rect.x1 || rect.y0 >= rect.y1)
		return;
	for (i = 0; i < dnpr_shared_rect_count; ) {
		struct dnpr_rect *old = &dnpr_shared_rects[i];

		if (!dnpr_shared_rects_touch(old, &rect)) {
			i++;
			continue;
		}
		rect.x0 = min(rect.x0, old->x0);
		rect.y0 = min(rect.y0, old->y0);
		rect.x1 = max(rect.x1, old->x1);
		rect.y1 = max(rect.y1, old->y1);
		dnpr_shared_rects[i] = dnpr_shared_rects[--dnpr_shared_rect_count];
		/* The enlarged rect may now touch an earlier entry. */
		i = 0;
	}
	if (dnpr_shared_rect_count < ARRAY_SIZE(dnpr_shared_rects))
		dnpr_shared_rects[dnpr_shared_rect_count++] = rect;
}

static void dnpr_shared_add_sprite_rect(const struct dnpr_sprite *sp)
{
	struct dnpr_rect rect = {
		.x0 = clamp_t(int, sp->left, 0, DNPR_WIDTH),
		.y0 = clamp_t(int, sp->top, 0, DNPR_HEIGHT),
		.x1 = clamp_t(int, sp->left + sp->size, 0, DNPR_WIDTH),
		.y1 = clamp_t(int, sp->top + sp->size, 0, DNPR_HEIGHT),
	};

	dnpr_shared_add_rect(rect);
}

static void dnpr_shared_build_regions(const struct dnpr_sprite *sprites,
				      unsigned int n, bool initial,
				      bool head_changed, unsigned int frame)
{
	unsigned int i, bar_w = ((frame % 40) + 1) * 8;

	dnpr_shared_rect_count = 0;
	if (initial) {
		dnpr_shared_add_rect((struct dnpr_rect){
			0, 0, DNPR_WIDTH, DNPR_HEIGHT });
	} else {
		if (head_changed) {
			struct dnpr_rect head = {
				.x0 = clamp_t(int, dnpr_head_dirty.x0 - 2, 0, DNPR_WIDTH),
				.y0 = clamp_t(int, dnpr_head_dirty.y0 - 2, 0, DNPR_HEIGHT),
				.x1 = clamp_t(int, dnpr_head_dirty.x1 + 2, 0, DNPR_WIDTH),
				.y1 = clamp_t(int, dnpr_head_dirty.y1 + 2, 0, DNPR_HEIGHT),
			};

			dnpr_shared_add_rect(head);
		}
		for (i = 0; i < dnpr_shared_prev_sprite_count; i++)
			dnpr_shared_add_sprite_rect(&dnpr_shared_prev_sprites[i]);
		for (i = 0; i < n; i++)
			dnpr_shared_add_sprite_rect(&sprites[i]);
		bar_w = max(bar_w, dnpr_shared_prev_bar_w);
		dnpr_shared_add_rect((struct dnpr_rect){ 0, 0, bar_w, 4 });
	}
	memcpy(dnpr_shared_prev_sprites, sprites,
	       n * sizeof(dnpr_shared_prev_sprites[0]));
	dnpr_shared_prev_sprite_count = n;
	dnpr_shared_prev_bar_w = ((frame % 40) + 1) * 8;
	dnpr_shared_prev_valid = true;
}

static void dnpr_render_shared_rect(const struct dnpr_sprite *sprites,
				    unsigned int n, unsigned int frame,
				    const struct dnpr_rect *rect, bool initial)
{
	unsigned int x, y, i;
	unsigned int bar_w = ((frame % 40) + 1) * 8;

	/* Restore the background once. The old compositor tested every sprite
	 * for every pixel in the rectangle, even though sprite coverage is sparse.
	 * Paint each clipped sprite span over this base in the original layer order. */
	for (y = rect->y0; y < rect->y1; y++) {
		for (x = rect->x0; x < rect->x1; x++) {
			size_t index = (size_t)y * DNPR_WIDTH + x;
			u32 rgb = dnpr_rgb565_to_xrgb8888[dnpr_base[index]];

			if (!initial && dnpr_shared_dirty &&
			    dnpr_shared_frame[index] != rgb)
				dnpr_shared_dirty[index] = 1;
			dnpr_shared_frame[index] = rgb;
		}
	}
	for (i = 0; i < n; i++) {
		const struct dnpr_sprite *sp = &sprites[i];
		int x0 = (int)rect->x0 > sp->left ? (int)rect->x0 : sp->left;
		int y0 = (int)rect->y0 > sp->top ? (int)rect->y0 : sp->top;
		int x1 = (int)rect->x1 < sp->left + sp->size ?
			(int)rect->x1 : sp->left + sp->size;
		int y1 = (int)rect->y1 < sp->top + sp->size ?
			(int)rect->y1 : sp->top + sp->size;

		if (x0 >= x1 || y0 >= y1)
			continue;
		for (y = y0; y < y1; y++) {
			unsigned int sy = (u64)(y - sp->top) * DNPR_SPRITE /
				sp->size;
			unsigned int dx = x0 - sp->left;
			unsigned int sx = (u64)dx * DNPR_SPRITE / sp->size;
			unsigned int sx_rem = (u64)dx * DNPR_SPRITE % sp->size;
			unsigned int sx_step = DNPR_SPRITE / sp->size;
			unsigned int sx_step_rem = DNPR_SPRITE % sp->size;

			for (x = x0; x < x1; x++) {
				size_t index = (size_t)y * DNPR_WIDTH + x;
				const u8 *px = sp->src +
					((size_t)sy * DNPR_SPRITE + sx) * 4;
				u32 old = dnpr_shared_frame[index];
				unsigned int r = (old >> 16) & 255;
				unsigned int g = (old >> 8) & 255;
				unsigned int b = old & 255;
				unsigned int a = px[3];
				u32 rgb;

				if (a >= 255) {
					r = px[0];
					g = px[1];
					b = px[2];
				} else if (a) {
					r = (px[0] * a + r * (255 - a)) / 255;
					g = (px[1] * a + g * (255 - a)) / 255;
					b = (px[2] * a + b * (255 - a)) / 255;
				}
				rgb = (r << 16) | (g << 8) | b;
				if (!initial && dnpr_shared_dirty && old != rgb)
					dnpr_shared_dirty[index] = 1;
				dnpr_shared_frame[index] = rgb;
				sx += sx_step;
				sx_rem += sx_step_rem;
				if (sx_rem >= sp->size) {
					sx++;
					sx_rem -= sp->size;
				}
			}
		}
	}
	/* The status bar is intentionally on top of every sprite. */
	if (rect->y0 < 4 && rect->x0 < bar_w) {
		unsigned int y1 = min(rect->y1, 4U);
		unsigned int x1 = min(rect->x1, bar_w);

		for (y = rect->y0; y < y1; y++)
			for (x = rect->x0; x < x1; x++) {
				size_t index = (size_t)y * DNPR_WIDTH + x;

				if (!initial && dnpr_shared_dirty &&
				    dnpr_shared_frame[index] != 0x00ff00ff)
					dnpr_shared_dirty[index] = 1;
				dnpr_shared_frame[index] = 0x00ff00ff;
			}
	}
}

static void dnpr_render_shared(unsigned int elapsed_ms, unsigned int frame,
			       bool initial, bool head_changed)
{
	struct dnpr_sprite sprites[DNPR_MAX_SPRITES];
	unsigned int n, i;
	bool full_redraw = initial || !dnpr_shared_prev_valid;

	n = dnpr_place_sprites(elapsed_ms, DNPR_WIDTH, DNPR_HEIGHT, sprites);
	dnpr_shared_build_regions(sprites, n, full_redraw, head_changed, frame);
	/* The blitter only examines these same rectangles. Clear dirty state
	 * there instead of sweeping the full 1920x1080 map on every frame. Do
	 * all clears before rendering: rectangles can overlap, and clearing an
	 * overlap after an earlier rectangle was composed would lose its marks. */
	if (dnpr_shared_dirty && !full_redraw)
		for (i = 0; i < dnpr_shared_rect_count; i++)
			for (unsigned int y = dnpr_shared_rects[i].y0;
			     y < dnpr_shared_rects[i].y1; y++)
				memset(dnpr_shared_dirty + (size_t)y * DNPR_WIDTH +
				       dnpr_shared_rects[i].x0, 0,
				       dnpr_shared_rects[i].x1 -
				       dnpr_shared_rects[i].x0);
	for (i = 0; i < dnpr_shared_rect_count; i++)
		dnpr_render_shared_rect(sprites, n, frame,
					&dnpr_shared_rects[i], full_redraw);
	dnpr_shared_valid = true;
}
