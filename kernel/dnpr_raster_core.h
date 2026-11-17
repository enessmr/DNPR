static s64 dnpr_edge(s32 ax, s32 ay, s32 bx, s32 by, s32 x, s32 y)
{
	return (s64)(x - ax) * (by - ay) - (s64)(y - ay) * (bx - ax);
}

static u16 dnpr_rgb565(unsigned int r, unsigned int g, unsigned int b)
{
	return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}

/* Divide by the triangle's invariant area without a per-pixel DIV. The
 * 48-bit reciprocal handles screen-sized triangles with a 64-bit product;
 * larger triangles use the 64-bit reciprocal. The remainder restores the
 * exact truncating quotient.
 */
static s64 dnpr_div_by_area(s64 n, u64 area, u64 reciprocal)
{
	u64 magnitude, quotient;

	if (area == 1)
		return n;
	magnitude = n < 0 ? 0 - (u64)n : (u64)n;
	quotient = area <= 0xffffffffULL ?
		(magnitude * reciprocal >> 48) :
		mul_u64_u64_shr(magnitude, reciprocal, 64);
	if (magnitude - quotient * area >= area)
		quotient++;
	return n < 0 ? -(s64)quotient : (s64)quotient;
}

static unsigned int dnpr_specular(s64 weighted_nx, s64 weighted_ny,
				 s64 weighted_nz, s64 area, u64 reciprocal)
{
	s64 nx = dnpr_div_by_area(weighted_nx, area, reciprocal);
	s64 ny = dnpr_div_by_area(weighted_ny, area, reciprocal);
	s64 nz = dnpr_div_by_area(weighted_nz, area, reciprocal);
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

static void dnpr_draw_live_triangle(const struct dnpr_projected *a,
				    const struct dnpr_projected *b,
				    const struct dnpr_projected *c,
				    const struct dnpr_live_triangle *tri,
				    int band_y0, int band_y1)
{
	s64 area = dnpr_edge(a->x, a->y, b->x, b->y, c->x, c->y);
	s64 slope_ab = 0, slope_bc = 0, slope_ca = 0;
	s64 step0, step1, dz, dr, dg, db, dnx, dny, dnz;
	u64 reciprocal = 0, remainder;
	int x0, x1, y0, y1, x, y, sign;
	bool unshaded = DNPR_UNSHADED;
	unsigned int nearest_z = min3(a->z, b->z, c->z);
	bool no_specular = max3(-59 * a->nx + 41 * a->ny + 106 * a->nz,
				-59 * b->nx + 41 * b->ny + 106 * b->nz,
				-59 * c->nx + 41 * c->ny + 106 * c->nz) <= -207;

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
	if (y0 < band_y0)
		y0 = band_y0;
	if (y1 > band_y1)
		y1 = band_y1;
	if (y0 > y1)
		return; /* The band misses every row of this triangle. */
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
	dz = step0 * ((s64)a->z - c->z) + step1 * ((s64)b->z - c->z);
	dr = dg = db = dnx = dny = dnz = 0;
	if (!unshaded) {
		dr = step0 * ((s64)a->lr - c->lr) + step1 * ((s64)b->lr - c->lr);
		dg = step0 * ((s64)a->lg - c->lg) + step1 * ((s64)b->lg - c->lg);
		db = step0 * ((s64)a->lb - c->lb) + step1 * ((s64)b->lb - c->lb);
		dnx = step0 * ((s64)a->nx - c->nx) + step1 * ((s64)b->nx - c->nx);
		dny = step0 * ((s64)a->ny - c->ny) + step1 * ((s64)b->ny - c->ny);
		dnz = step0 * ((s64)a->nz - c->nz) + step1 * ((s64)b->nz - c->nz);
	}
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
		wz = (s64)area * c->z + w0 * ((s64)a->z - c->z) +
			w1 * ((s64)b->z - c->z);
		wr = (s64)area * c->lr + w0 * ((s64)a->lr - c->lr) +
			w1 * ((s64)b->lr - c->lr);
		wg = (s64)area * c->lg + w0 * ((s64)a->lg - c->lg) +
			w1 * ((s64)b->lg - c->lg);
		wb = (s64)area * c->lb + w0 * ((s64)a->lb - c->lb) +
			w1 * ((s64)b->lb - c->lb);
		wnx = (s64)area * c->nx + w0 * ((s64)a->nx - c->nx) +
			w1 * ((s64)b->nx - c->nx);
		wny = (s64)area * c->ny + w0 * ((s64)a->ny - c->ny) +
			w1 * ((s64)b->ny - c->ny);
		wnz = (s64)area * c->nz + w0 * ((s64)a->nz - c->nz) +
			w1 * ((s64)b->nz - c->nz);
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
			z = dnpr_div_by_area(wz, area, reciprocal);
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
			sr = dnpr_div_by_area(wr, area, reciprocal);
			sg = dnpr_div_by_area(wg, area, reciprocal);
			sb = dnpr_div_by_area(wb, area, reciprocal);
			spec = no_specular || DNPR_NO_SPECULAR ? 0 :
				dnpr_specular(wnx, wny, wnz, area, reciprocal);
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
	memcpy(dnpr_base, dnpr_live_data, DNPR_BASE_BYTES);
	memset(dnpr_live_depth, 0xff, DNPR_BASE_BYTES);
	memset(dnpr_live_owner, 0xff, DNPR_BASE_BYTES);
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
	dnpr_head_dirty.x0 = min(x0, dnpr_head_bounds.x0);
	dnpr_head_dirty.y0 = min(y0, dnpr_head_bounds.y0);
	dnpr_head_dirty.x1 = max(x1, dnpr_head_bounds.x1);
	dnpr_head_dirty.y1 = max(y1, dnpr_head_bounds.y1);
	dnpr_head_bounds.x0 = x0;
	dnpr_head_bounds.y0 = y0;
	dnpr_head_bounds.x1 = x1;
	dnpr_head_bounds.y1 = y1;
	for (i = 0; i < DNPR_LIVE_NT; i++)
		dnpr_triangle_order[i] = i;
	for (unsigned int gap = DNPR_LIVE_NT / 2; gap; gap /= 2)
		for (i = gap; i < DNPR_LIVE_NT; i++) {
			u16 item = dnpr_triangle_order[i];
			unsigned int depth = dnpr_triangle_depth(item);
			unsigned int cursor = i;

			while (cursor >= gap &&
			       dnpr_triangle_depth(dnpr_triangle_order[cursor - gap]) > depth) {
				dnpr_triangle_order[cursor] = dnpr_triangle_order[cursor - gap];
				cursor -= gap;
			}
			dnpr_triangle_order[cursor] = item;
		}
	return true;
}

/* Rasterize every triangle over the row range [y0, y1] inclusive. */
