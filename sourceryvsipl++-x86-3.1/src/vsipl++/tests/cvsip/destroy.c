/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    tests/cvsip/destroy.c
    @author  Stefan Seefeld
    @date    2008-05-02
    @brief   view support tests: Make sure all destroy functions accept null pointers.
*/
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
