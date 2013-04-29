/*
 * Copyright (c) 2000 Matteo Frigo
 * Copyright (c) 2000 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include "bench.h"

void useropt(const char *arg)
{
     ovtpvt_err("unknown user option: %s.  Ignoring.\n", arg);
}
