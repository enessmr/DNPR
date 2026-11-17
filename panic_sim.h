#ifndef DNPR_PANIC_SIM_H
#define DNPR_PANIC_SIM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool panic_sim_init(void);
void panic_sim_set_threads(unsigned int threads);
unsigned int panic_sim_thread_count(void);
void panic_sim_set_unshaded(bool enabled);
void panic_sim_set_no_specular(bool enabled);
bool panic_sim_set_scanout(unsigned int index, void *map, size_t size, unsigned int pitch);
struct panic_sim_timing {
	bool head_changed;
	double prepare_ms;
	double raster_ms;
	double compose_ms;
	double blit_ms;
	double copy_ms;
};
void panic_sim_render(unsigned int elapsed_ms, unsigned int frame, uint32_t *pixels);
void panic_sim_render_timed(unsigned int elapsed_ms, unsigned int frame,
			    uint32_t *pixels, struct panic_sim_timing *timing);
void panic_sim_destroy(void);

#endif
