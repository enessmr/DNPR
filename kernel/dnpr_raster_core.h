static s64 dnpr_edge(s32 ax, s32 ay, s32 bx, s32 by, s32 x, s32 y)
{
	return (s64)(x - ax) * (by - ay) - (s64)(y - ay) * (bx - ax);
}

static u16 dnpr_rgb565(unsigned int r, unsigned int g, unsigned int b)
{
	return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}

/* Q24 plane values let the scan converter step all attributes across a row
 * with additions instead of doing a reciprocal multiply for every pixel. */
static s64 dnpr_div_by_area_q24(s64 n, u64 area, u64 reciprocal)
{
	u64 magnitude, quotient;

	if (!n || area == 1)
		return n * (1LL << 24);
	magnitude = n < 0 ? 0 - (u64)n : (u64)n;
	if (area <= 0xffffffffULL) {
		quotient = mul_u64_u64_shr(magnitude, reciprocal, 24);
		if ((magnitude << 24) - quotient * area >= area)
			quotient++;
	} else {
		quotient = div64_u64(magnitude << 24, area);
	}
	return n < 0 ? -(s64)quotient : (s64)quotient;
}

static s64 dnpr_q24_to_s64(s64 value)
{
	return value < 0 ? -((-value) >> 24) : value >> 24;
}

static unsigned int dnpr_specular(s64 nx, s64 ny, s64 nz)
{
	u64 length, q;
	s64 h;
	int i;

	if (dnpr_spec_lut && nx >= -128 && nx <= 127 &&
	    ny >= -128 && ny <= 127 && nz >= -128 && nz <= 127)
		return dnpr_spec_lut[((unsigned int)(nx + 128) << 16) |
				     ((unsigned int)(ny + 128) << 8) |
				     (unsigned int)(nz + 128)];
	length = int_sqrt64(nx * nx + ny * ny + nz * nz);
	h = length ? (-59 * nx + 41 * ny + 106 * nz) / (s64)length : 0;

	if (h <= 0)
		return 0;
	q = min_t(u64, h, 127) * 65536 / 127;
	for (i = 0; i < 7; i++)
		q = q * q >> 16; /* h^128, matching the preview highlight */
	return 90 * q >> 16;
}

/* Conservative scanline intersection: the edge test below still decides
 * exact coverage, while these bounds avoid scanning empty triangle boxes.
 */
static void dnpr_span_edge(const struct dnpr_projected *a,
			   const struct dnpr_projected *b, s64 slope,
			   int y, s64 *left, s64 *right, bool *found)
{
	s64 xq;

	if (y < min(a->y, b->y) || y > max(a->y, b->y))
		return;
	xq = (s64)a->x * (1LL << 32) + (s64)(y - a->y) * slope;
	if (!*found) {
		*left = *right = xq;
		*found = true;
	} else {
		*left = min(*left, xq);
		*right = max(*right, xq);
	}
	if (a->y == b->y) {
		xq = (s64)b->x * (1LL << 32);
		*left = min(*left, xq);
		*right = max(*right, xq);
	}
}

static const struct dnpr_live_triangle *dnpr_tris(void);

#ifndef DNPR_UNSHADED
#define DNPR_UNSHADED false
#endif

#ifndef DNPR_NO_SPECULAR
#define DNPR_NO_SPECULAR false
#endif

/* Projection changes once per frame, but tall triangles are visited by
 * several row bands. Cache the expensive triangle setup once for that frame.
 */
struct dnpr_triangle_setup {
	s64 area, slope_ab, slope_bc, slope_ca;
	s64 step0, step1, dz, dr, dg, db, dnx, dny, dnz;
	u64 reciprocal;
	int x0, x1, y0, y1;
	unsigned int nearest_z;
	int sign;
	bool unshaded, no_specular, valid;
};

static struct dnpr_triangle_setup dnpr_triangle_setup[DNPR_LIVE_NT];

static void dnpr_prepare_triangle_setup(unsigned int index)
{
	const struct dnpr_live_triangle *tri = &dnpr_tris()[index];
	const struct dnpr_projected *a = &dnpr_projected[tri->a];
	const struct dnpr_projected *b = &dnpr_projected[tri->b];
	const struct dnpr_projected *c = &dnpr_projected[tri->c];
	struct dnpr_triangle_setup *setup = &dnpr_triangle_setup[index];
	s64 area = dnpr_edge(a->x, a->y, b->x, b->y, c->x, c->y);
	s64 slope_ab = 0, slope_bc = 0, slope_ca = 0;
	s64 step0, step1, dz, dr, dg, db, dnx, dny, dnz;
	u64 reciprocal = 0, remainder;
	int x0, x1, y0, y1, sign;
	bool unshaded = DNPR_UNSHADED;
	unsigned int nearest_z = min3(a->z, b->z, c->z);
	bool no_specular = max3(-59 * a->nx + 41 * a->ny + 106 * a->nz,
				-59 * b->nx + 41 * b->ny + 106 * b->nz,
				-59 * c->nx + 41 * c->ny + 106 * c->nz) <= -207;

	setup->valid = false;
	if (!area)
		return;
	sign = area < 0 ? -1 : 1;
	area *= sign;
	if (area != 1) {
		if (area <= 0xffffffffULL) {
			reciprocal = div64_u64_rem(1ULL << 48, area,
						 &remainder);
		} else {
			reciprocal = div64_u64_rem(U64_MAX, area, &remainder);
			if (remainder == area - 1)
				reciprocal++;
		}
	}
	y0 = clamp_t(int, min3(a->y, b->y, c->y), 0, DNPR_HEIGHT - 1);
	y1 = clamp_t(int, max3(a->y, b->y, c->y), 0, DNPR_HEIGHT - 1);
	x0 = clamp_t(int, min3(a->x, b->x, c->x), 0, DNPR_WIDTH - 1);
	x1 = clamp_t(int, max3(a->x, b->x, c->x), 0, DNPR_WIDTH - 1);
	if (a->y != b->y)
		slope_ab = div_s64((s64)(b->x - a->x) * (1LL << 32),
				 b->y - a->y);
	if (b->y != c->y)
		slope_bc = div_s64((s64)(c->x - b->x) * (1LL << 32),
				 c->y - b->y);
	if (c->y != a->y)
		slope_ca = div_s64((s64)(a->x - c->x) * (1LL << 32),
				 a->y - c->y);
	step0 = (s64)sign * (c->y - b->y);
	step1 = (s64)sign * (a->y - c->y);
	dz = dnpr_div_by_area_q24(
			step0 * ((s64)a->z - c->z) +
			step1 * ((s64)b->z - c->z), area, reciprocal);
	dr = dg = db = dnx = dny = dnz = 0;
	if (!unshaded) {
		dr = dnpr_div_by_area_q24(
			step0 * ((s64)a->lr - c->lr) +
			step1 * ((s64)b->lr - c->lr), area, reciprocal);
		dg = dnpr_div_by_area_q24(
			step0 * ((s64)a->lg - c->lg) +
			step1 * ((s64)b->lg - c->lg), area, reciprocal);
		db = dnpr_div_by_area_q24(
			step0 * ((s64)a->lb - c->lb) +
			step1 * ((s64)b->lb - c->lb), area, reciprocal);
		dnx = dnpr_div_by_area_q24(
			step0 * ((s64)a->nx - c->nx) +
			step1 * ((s64)b->nx - c->nx), area, reciprocal);
		dny = dnpr_div_by_area_q24(
			step0 * ((s64)a->ny - c->ny) +
			step1 * ((s64)b->ny - c->ny), area, reciprocal);
		dnz = dnpr_div_by_area_q24(
			step0 * ((s64)a->nz - c->nz) +
			step1 * ((s64)b->nz - c->nz), area, reciprocal);
	}
	*setup = (struct dnpr_triangle_setup){
		.area = area,
		.slope_ab = slope_ab, .slope_bc = slope_bc,
		.slope_ca = slope_ca,
		.step0 = step0, .step1 = step1,
		.dz = dz, .dr = dr, .dg = dg, .db = db,
		.dnx = dnx, .dny = dny, .dnz = dnz,
		.reciprocal = reciprocal,
		.x0 = x0, .x1 = x1, .y0 = y0, .y1 = y1,
		.nearest_z = nearest_z, .sign = sign,
		.unshaded = unshaded, .no_specular = no_specular,
		.valid = true,
	};
}

static void dnpr_draw_live_triangle(unsigned int index,
				    int band_y0, int band_y1)
{
	const struct dnpr_live_triangle *tri = &dnpr_tris()[index];
	const struct dnpr_projected *a = &dnpr_projected[tri->a];
	const struct dnpr_projected *b = &dnpr_projected[tri->b];
	const struct dnpr_projected *c = &dnpr_projected[tri->c];
	const struct dnpr_triangle_setup *setup = &dnpr_triangle_setup[index];
	s64 area = setup->area;
	s64 slope_ab = setup->slope_ab, slope_bc = setup->slope_bc;
	s64 slope_ca = setup->slope_ca;
	s64 step0 = setup->step0, step1 = setup->step1;
	s64 dz = setup->dz, dr = setup->dr, dg = setup->dg, db = setup->db;
	s64 dnx = setup->dnx, dny = setup->dny, dnz = setup->dnz;
	u64 reciprocal = setup->reciprocal;
	int x0 = setup->x0, x1 = setup->x1;
	int y0 = max(setup->y0, band_y0);
	int y1 = min(setup->y1, band_y1);
	int x, y, sign = setup->sign;
	bool unshaded = setup->unshaded;
	unsigned int nearest_z = setup->nearest_z;
	bool no_specular = setup->no_specular;

	if (!setup->valid || y0 > y1)
		return;
	for (y = y0; y <= y1; y++) {
		s64 left = 0, right = 0, w0, w1;
		s64 wz, wr, wg, wb, wnx, wny, wnz;
		int span_x0, span_x1;
		bool found = false;

		dnpr_span_edge(a, b, slope_ab, y, &left, &right, &found);
		dnpr_span_edge(b, c, slope_bc, y, &left, &right, &found);
		dnpr_span_edge(c, a, slope_ca, y, &left, &right, &found);
		if (!found)
			continue;
		span_x0 = max(x0, (int)(left >> 32) - 2);
		span_x1 = min(x1, (int)(right >> 32) + 2);
		if (span_x0 > span_x1)
			continue;
		for (x = span_x0; x <= span_x1; x++) {
			size_t index = (size_t)y * DNPR_WIDTH + x;

			if (dnpr_live_depth[index] > nearest_z ||
			    (dnpr_live_depth[index] == nearest_z &&
			     dnpr_live_owner[index] > tri - dnpr_tris()))
				break;
		}
		if (x > span_x1)
			continue;
		w0 = sign * dnpr_edge(b->x, b->y, c->x, c->y,
					span_x0, y);
		w1 = sign * dnpr_edge(c->x, c->y, a->x, a->y,
					span_x0, y);
		wz = dnpr_div_by_area_q24((s64)area * c->z +
			w0 * ((s64)a->z - c->z) + w1 * ((s64)b->z - c->z),
			area, reciprocal);
		wr = unshaded ? 0 : dnpr_div_by_area_q24((s64)area * c->lr +
			w0 * ((s64)a->lr - c->lr) + w1 * ((s64)b->lr - c->lr),
			area, reciprocal);
		wg = unshaded ? 0 : dnpr_div_by_area_q24((s64)area * c->lg +
			w0 * ((s64)a->lg - c->lg) + w1 * ((s64)b->lg - c->lg),
			area, reciprocal);
		wb = unshaded ? 0 : dnpr_div_by_area_q24((s64)area * c->lb +
			w0 * ((s64)a->lb - c->lb) + w1 * ((s64)b->lb - c->lb),
			area, reciprocal);
		wnx = unshaded ? 0 : dnpr_div_by_area_q24((s64)area * c->nx +
			w0 * ((s64)a->nx - c->nx) + w1 * ((s64)b->nx - c->nx),
			area, reciprocal);
		wny = unshaded ? 0 : dnpr_div_by_area_q24((s64)area * c->ny +
			w0 * ((s64)a->ny - c->ny) + w1 * ((s64)b->ny - c->ny),
			area, reciprocal);
		wnz = unshaded ? 0 : dnpr_div_by_area_q24((s64)area * c->nz +
			w0 * ((s64)a->nz - c->nz) + w1 * ((s64)b->nz - c->nz),
			area, reciprocal);
		for (x = span_x0; x <= span_x1; x++) {
			s64 w2;
			unsigned int z, sr, sg, sb, spec, r, g, blue;
			size_t index;

			if (w0 < 0 || w1 < 0)
				goto next_pixel;
			w2 = area - w0 - w1;
			if (w2 < 0)
				goto next_pixel;
			index = (size_t)y * DNPR_WIDTH + x;
			z = dnpr_q24_to_s64(wz);
			if (z > dnpr_live_depth[index] ||
			    (z == dnpr_live_depth[index] &&
			     (dnpr_live_owner[index] == 0xffff ||
			      tri - dnpr_tris() >= dnpr_live_owner[index])))
				goto next_pixel;
			dnpr_live_depth[index] = z;
			dnpr_live_owner[index] = tri - dnpr_tris();
			if (unshaded) {
				dnpr_base[index] = dnpr_rgb565(tri->r, tri->g,
							tri->blue);
				goto next_pixel;
			}
			sr = dnpr_q24_to_s64(wr);
			sg = dnpr_q24_to_s64(wg);
			sb = dnpr_q24_to_s64(wb);
			spec = no_specular || DNPR_NO_SPECULAR ? 0 :
				dnpr_specular(dnpr_q24_to_s64(wnx),
					      dnpr_q24_to_s64(wny),
					      dnpr_q24_to_s64(wnz));
			r = min(255U, (tri->r * sr >> 8) + spec);
			g = min(255U, (tri->g * sg >> 8) + spec);
			blue = min(255U, (tri->blue * sb >> 8) + spec);
			dnpr_base[index] = dnpr_rgb565(r, g, blue);
	next_pixel:
			w0 += step0;
			w1 += step1;
			wz += dz;
			wr += dr;
			wg += dg;
			wb += db;
			wnx += dnx;
			wny += dny;
			wnz += dnz;
		}
	}
}

static s32 dnpr_lerp_s32(s32 a, s32 b, unsigned int fraction)
{
	return a + ((s64)(b - a) * fraction >> 16);
}

static const struct dnpr_live_triangle *dnpr_tris(void)
{
	return (const void *)(dnpr_live_data + DNPR_BASE_BYTES);
}

static unsigned int dnpr_triangle_depth(unsigned int index)
{
	const struct dnpr_live_triangle *tri = &dnpr_tris()[index];

	if (tri->a >= DNPR_LIVE_NV || tri->b >= DNPR_LIVE_NV ||
	    tri->c >= DNPR_LIVE_NV)
		return ~0U;
	return min3(dnpr_projected[tri->a].z,
		    dnpr_projected[tri->b].z, dnpr_projected[tri->c].z);
}

static void dnpr_restore_live_head(void)
{
	int y;

	for (y = dnpr_head_dirty.y0; y < dnpr_head_dirty.y1; y++) {
		size_t offset = (size_t)y * DNPR_WIDTH + dnpr_head_dirty.x0;
		unsigned int width = dnpr_head_dirty.x1 - dnpr_head_dirty.x0;

		memcpy(dnpr_base + offset,
		       (const u16 *)dnpr_live_data + offset,
		       width * sizeof(*dnpr_base));
	}
	for (y = dnpr_head_bounds.y0; y < dnpr_head_bounds.y1; y++) {
		size_t offset = (size_t)y * DNPR_WIDTH + dnpr_head_bounds.x0;
		unsigned int width = dnpr_head_bounds.x1 - dnpr_head_bounds.x0;

		memset(dnpr_live_depth + offset, 0xff,
		       width * sizeof(*dnpr_live_depth));
		memset(dnpr_live_owner + offset, 0xff,
		       width * sizeof(*dnpr_live_owner));
	}
}

static bool dnpr_render_live_head_prepare(unsigned int elapsed_ms)
{
	const struct dnpr_live_vertex *verts, *next_verts;
	u64 pose_q = min_t(u64, (u64)elapsed_ms * 30 * 65536 / 1000,
			   (u64)(DNPR_LIVE_NF - 1) << 16);
	unsigned int pose = pose_q >> 16, fraction = pose_q & 0xffff, i;
	int x0 = DNPR_WIDTH, y0 = DNPR_HEIGHT, x1 = 0, y1 = 0;

	if (pose_q == dnpr_last_pose_q)
		return false;
	dnpr_last_pose_q = pose_q;
	verts = (const void *)(dnpr_live_data + DNPR_BASE_BYTES +
		DNPR_LIVE_NT * sizeof(struct dnpr_live_triangle) +
		pose * DNPR_LIVE_NV * sizeof(*verts));
	next_verts = (const void *)(dnpr_live_data + DNPR_BASE_BYTES +
		DNPR_LIVE_NT * sizeof(struct dnpr_live_triangle) +
		min(pose + 1, DNPR_LIVE_NF - 1) * DNPR_LIVE_NV * sizeof(*verts));
	for (i = 0; i < DNPR_LIVE_NV; i++) {
		s32 vx = dnpr_lerp_s32(verts[i].x, next_verts[i].x, fraction);
		s32 vy = dnpr_lerp_s32(verts[i].y, next_verts[i].y, fraction);
		s32 vz = dnpr_lerp_s32(verts[i].z, next_verts[i].z, fraction);
		int cam_z = 20 * 256 - vz;

		if (cam_z <= 0)
			cam_z = 1;
		dnpr_projected[i] = (struct dnpr_projected){
			.x = DNPR_WIDTH / 2 + div_s64((s64)vx * 1365, cam_z),
			.y = DNPR_HEIGHT / 2 - div_s64((s64)(vy - 2 * 256) * 1365, cam_z),
			.z = cam_z,
			.lr = dnpr_lerp_s32(verts[i].lr, next_verts[i].lr, fraction),
			.lg = dnpr_lerp_s32(verts[i].lg, next_verts[i].lg, fraction),
			.lb = dnpr_lerp_s32(verts[i].lb, next_verts[i].lb, fraction),
			.nx = dnpr_lerp_s32(verts[i].nx, next_verts[i].nx, fraction),
			.ny = dnpr_lerp_s32(verts[i].ny, next_verts[i].ny, fraction),
			.nz = dnpr_lerp_s32(verts[i].nz, next_verts[i].nz, fraction),
		};
		x0 = min(x0, dnpr_projected[i].x);
		y0 = min(y0, dnpr_projected[i].y);
		x1 = max(x1, dnpr_projected[i].x);
		y1 = max(y1, dnpr_projected[i].y);
	}
	x0 = clamp(x0 - 2, 0, (int)DNPR_WIDTH);
	y0 = clamp(y0 - 2, 0, (int)DNPR_HEIGHT);
	x1 = clamp(x1 + 3, 0, (int)DNPR_WIDTH);
	y1 = clamp(y1 + 3, 0, (int)DNPR_HEIGHT);
	dnpr_head_dirty.x0 = dnpr_head_bounds_valid ?
		min(x0, dnpr_head_bounds.x0) : x0;
	dnpr_head_dirty.y0 = dnpr_head_bounds_valid ?
		min(y0, dnpr_head_bounds.y0) : y0;
	dnpr_head_dirty.x1 = dnpr_head_bounds_valid ?
		max(x1, dnpr_head_bounds.x1) : x1;
	dnpr_head_dirty.y1 = dnpr_head_bounds_valid ?
		max(y1, dnpr_head_bounds.y1) : y1;
	dnpr_head_bounds.x0 = x0;
	dnpr_head_bounds.y0 = y0;
	dnpr_head_bounds.x1 = x1;
	dnpr_head_bounds.y1 = y1;
	dnpr_head_bounds_valid = true;
	/* Restore the old and new head area, clearing depth only where current
	 * geometry can land. */
	dnpr_restore_live_head();
	memset(dnpr_band_triangle_count, 0, sizeof(dnpr_band_triangle_count));
	for (i = 0; i < DNPR_LIVE_NT; i++) {
		dnpr_triangle_order[i] = i;
		dnpr_triangle_depths[i] = dnpr_triangle_depth(i);
	}
	for (unsigned int gap = DNPR_LIVE_NT / 2; gap; gap /= 2)
		for (i = gap; i < DNPR_LIVE_NT; i++) {
			u16 item = dnpr_triangle_order[i];
			u16 depth = dnpr_triangle_depths[item];
			unsigned int cursor = i;

			while (cursor >= gap &&
			       dnpr_triangle_depths[
				       dnpr_triangle_order[cursor - gap]] > depth) {
				dnpr_triangle_order[cursor] = dnpr_triangle_order[cursor - gap];
				cursor -= gap;
			}
			dnpr_triangle_order[cursor] = item;
		}
	/* Keep the near-to-far order within each band, while excluding triangles
	 * whose projected bounds cannot touch that band. */
	for (i = 0; i < DNPR_LIVE_NT; i++) {
		const struct dnpr_live_triangle *t = dnpr_tris();
		unsigned int index = dnpr_triangle_order[i];
		int first, last, band;

		dnpr_triangle_setup[index].valid = false;
		if (t[index].a >= DNPR_LIVE_NV || t[index].b >= DNPR_LIVE_NV ||
		    t[index].c >= DNPR_LIVE_NV)
			continue;
		/* Godot's face indices are reversed in the exported mesh. For this
		 * camera and edge convention, negative area faces the camera. Cull
		 * back faces before they enter any worker's band list. */
		if (dnpr_edge(dnpr_projected[t[index].a].x,
			      dnpr_projected[t[index].a].y,
			      dnpr_projected[t[index].b].x,
			      dnpr_projected[t[index].b].y,
			      dnpr_projected[t[index].c].x,
			      dnpr_projected[t[index].c].y) >= 0)
			continue;
		dnpr_prepare_triangle_setup(index);
		if (!dnpr_triangle_setup[index].valid)
			continue;
		first = clamp_t(int, min3(dnpr_projected[t[index].a].y,
					 dnpr_projected[t[index].b].y,
					 dnpr_projected[t[index].c].y),
				 0, DNPR_HEIGHT - 1) / DNPR_BAND_H;
		last = clamp_t(int, max3(dnpr_projected[t[index].a].y,
					 dnpr_projected[t[index].b].y,
					 dnpr_projected[t[index].c].y),
				 0, DNPR_HEIGHT - 1) / DNPR_BAND_H;
		for (band = first; band <= last; band++)
			dnpr_band_triangles[band][dnpr_band_triangle_count[band]++] =
				index;
	}
	return true;
}

/* Rasterize every triangle over the row range [y0, y1] inclusive. */
