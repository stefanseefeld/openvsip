/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.
*/
/** @file    dda/dense.cpp
    @author  Stefan Seefeld
    @date    2007-06-12
    @brief   VSIPL++ Library: Simple example for direct data access.
*/

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip_csl/dda.hpp>

using namespace vsip;
using namespace vsip_csl;

void ramp(float *data, ptrdiff_t stride, size_t size)
{
  for (size_t i = 0; i < size; ++i)
    data[stride * i] = i;
}

int 
main(int argc, char **argv)
{
  vsipl init(argc, argv);

  // Create a (dense) vector of size 8.
  Vector<float> view(8);
  // Create an external data access object for it.
  dda::Ext_data<Vector<float>::block_type> ext(view.block());
  // Pass raw pointer to VSIPL++-oblivious function.
  ramp(ext.data(), ext.stride(0), ext.size());
  return 0;
}
