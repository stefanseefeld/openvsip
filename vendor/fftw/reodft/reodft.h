/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#ifndef __REODFT_H__
#define __REODFT_H__

#include "ifftw.h"
#include "rdft.h"

#define REODFT_KINDP(k) ((k) >= REDFT00 && (k) <= RODFT11)

void X(redft00e_r2hc_register)(planner *p);
void X(redft00e_r2hc_pad_register)(planner *p);
void X(rodft00e_r2hc_register)(planner *p);
void X(rodft00e_r2hc_pad_register)(planner *p);
void X(reodft00e_splitradix_register)(planner *p);
void X(reodft010e_r2hc_register)(planner *p);
void X(reodft11e_r2hc_register)(planner *p);
void X(reodft11e_radix2_r2hc_register)(planner *p);
void X(reodft11e_r2hc_odd_register)(planner *p);

/* configurations */
void X(reodft_conf_standard)(planner *p);

#endif /* __REODFT_H__ */
