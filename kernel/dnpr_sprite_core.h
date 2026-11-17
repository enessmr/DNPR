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
