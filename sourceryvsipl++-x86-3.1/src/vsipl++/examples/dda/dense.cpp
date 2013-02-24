/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.
*/
/** @file    dda/dense.cpp
    @author  Stefan Seefeld
    @date    2007-06-12
    @brief   VSIPL++ Library: Simple example for direct data access.
*/

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/dda.hpp>

using namespace vsip;

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
  dda::Data<Vector<float>::block_type, dda::out> data(view.block());
  // Pass raw pointer to a VSIPL++-oblivious function.
  ramp(data.ptr(), data.stride(0), data.size());
  return 0;
}
