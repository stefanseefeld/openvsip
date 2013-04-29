/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "dft.h"
#include "rdft.h"
#include "reodft.h"

void X(configure_planner)(planner *plnr)
{
     X(dft_conf_standard)(plnr);
     X(rdft_conf_standard)(plnr);
     X(reodft_conf_standard)(plnr);
#if HAVE_CELL
     X(dft_conf_cell)(plnr);
#endif
}
