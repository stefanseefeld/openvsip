//
// Copyright (c) 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is made available under the GPL.
// See the accompanying file LICENSE.GPL for details.

#include <vsip.h>
#include "output.h"
#include "test.h"

int
main()
{
  vsip_init(0);
  vsip_blockdestroy_f(0);
  vsip_talldestroy_f(0);
  vsip_ctalldestroy_f(0);
  vsip_tdestroy_f(0);
  vsip_ctdestroy_f(0);
  vsip_malldestroy_f(0);
  vsip_cmalldestroy_f(0);
  vsip_mdestroy_f(0);
  vsip_cmdestroy_f(0);
  vsip_valldestroy_f(0);
  vsip_cvalldestroy_f(0);
  vsip_vdestroy_f(0);
  vsip_cvdestroy_f(0);
  vsip_finalize(0);
  return 0;
};
