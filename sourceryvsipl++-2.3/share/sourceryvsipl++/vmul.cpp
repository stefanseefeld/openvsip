/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vmul.cpp
    @author  Stefan Seefeld
    @date    2007-09-20
    @brief   VSIPL++ Library: Simple VSIPL++ program.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/math.hpp>
#include <vsip/core/profile.hpp>

#include <vsip_csl/error_db.hpp>
#include <vsip_csl/ref_dft.hpp>


/***********************************************************************
  Definitions
***********************************************************************/

using namespace vsip;
using namespace vsip_csl;
using namespace vsip::impl::profile;


int
main(int argc, char **argv)
{
  vsipl init(argc, argv);
  
  Vector<float> a(16);
  Vector<float> b(16);
  Vector<float> c(16);

  c = a * b;

  return 0;
}
