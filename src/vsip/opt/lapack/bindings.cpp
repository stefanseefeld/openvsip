/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/lapack/bindings.cpp
    @author  Jules Bergmann
    @date    2005-10-11
    @brief   VSIPL++ Library: Lapack interface
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/opt/lapack/bindings.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

extern "C"
{

/// LAPACK error handler.  Called by LAPACK functions if illegal
/// argument is passed.

void
xerbla_(char* name, int* info)
{
  char copy[8];
  char msg[256];

  strncpy(copy, name, 6);
  copy[6] = 0;
  sprintf(msg, "lapack -- illegal arg (name=%s  info=%d)", copy, *info);

  VSIP_IMPL_THROW(vsip::impl::unimplemented(msg));
}

} // extern "C"
