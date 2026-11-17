/* SPDX-License-Identifier: GPL-2.0 */
#ifndef DNPR_PANIC_H
#define DNPR_PANIC_H

#include <linux/types.h>

struct drm_scanout_buffer;

bool dnpr_panic_render(struct drm_scanout_buffer *sb);
bool dnpr_panic_smp_arm(void);
bool dnpr_panic_smp_armed(void);
void dnpr_panic_smp_worker(void);
void dnpr_panic_smp_finish(void);
void dnpr_panic_animate(void);
int dnpr_panic_init(void);
void dnpr_panic_exit(void);

#endif
