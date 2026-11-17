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
