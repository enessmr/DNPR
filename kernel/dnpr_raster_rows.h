static void dnpr_raster_rows(int y0, int y1)
{
	unsigned int band;

	for (band = y0 / DNPR_BAND_H; band <= y1 / DNPR_BAND_H; band++) {
		int band_y0 = max(y0, (int)(band * DNPR_BAND_H));
		int band_y1 = min(y1, (int)((band + 1) * DNPR_BAND_H - 1));
		unsigned int i;

		for (i = 0; i < dnpr_band_triangle_count[band]; i++) {
			unsigned int index = dnpr_band_triangles[band][i];

			dnpr_draw_live_triangle(index, band_y0, band_y1);
		}
	}
}
