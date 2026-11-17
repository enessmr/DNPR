static void dnpr_raster_rows(int y0, int y1)
{
	const struct dnpr_live_triangle *t = dnpr_tris();
	unsigned int i;

	for (i = 0; i < DNPR_LIVE_NT; i++) {
		unsigned int index = dnpr_triangle_order[i];

		if (t[index].a >= DNPR_LIVE_NV || t[index].b >= DNPR_LIVE_NV ||
		    t[index].c >= DNPR_LIVE_NV)
			continue;
		dnpr_draw_live_triangle(&dnpr_projected[t[index].a],
					&dnpr_projected[t[index].b],
					&dnpr_projected[t[index].c], &t[index],
					y0, y1);
	}
}
