/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#ifndef __THREADS_H__
#define __THREADS_H__

#include "ifftw.h"
#include "ct.h"
#include "hc2hc.h"

typedef struct {
     int min, max, thr_num;
     void *data;
} spawn_data;

typedef void *(*spawn_function) (spawn_data *);

void X(spawn_loop)(int loopmax, int nthreads,
		   spawn_function proc, void *data);
int X(ithreads_init)(void);
void X(threads_cleanup)(void);

/* configurations */

void X(dft_thr_vrank_geq1_register)(planner *p);
void X(rdft_thr_vrank_geq1_register)(planner *p);
void X(rdft2_thr_vrank_geq1_register)(planner *p);

ct_solver *X(mksolver_ct_threads)(size_t size, INT r, int dec, 
				  ct_mkinferior mkcldw,
				  ct_force_vrecursion force_vrecursionp);
hc2hc_solver *X(mksolver_hc2hc_threads)(size_t size, INT r, hc2hc_mkinferior mkcldw);

void X(threads_conf_standard)(planner *p);
void X(threads_register_hooks)(void);
void X(threads_unregister_hooks)(void);
#endif /* __THREADS_H__ */
