/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.
*/
/** @file    dda/stride.cpp
    @author  Stefan Seefeld
    @date    2007-06-12
    @brief   VSIPL++ Library: Simple example for direct data access.
*/

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/dda.hpp>

using namespace vsip;

void ramp(float *data, size_t size)
{
  for (size_t i = 0; i < size; ++i)
    data[i] = i;
}

void ramp(float *data, ptrdiff_t stride, size_t size)
{
  for (size_t i = 0; i < size; ++i)
    data[i*stride] = i;
}

int 
main(int argc, char **argv)
{
  vsipl init(argc, argv);

  // Create a (dense) vector of size 8.
  Vector<float> view(8);
  // Create a subview refering to every other element of the above.
  Vector<float>::subview_type subview = view(Domain<1>(0, 2, 4));

  // First case: access data with non-unit stride.
  {
    typedef Vector<float>::subview_type::block_type block_type;
    dda::Data<block_type, dda::out> data(subview.block());
    ramp(data.ptr(), data.stride(0), data.size());
  }

  // Second case: force unit-stride access by means of a temporary.
  {
    // Define an alias for the subview block.
    typedef Vector<float>::subview_type::block_type block_type;
    // Define a 1D unit-stride layout.
    typedef Layout<1, row1_type, unit_stride> layout_type;

    dda::Data<block_type, dda::out, layout_type> data(subview.block());
    ramp(data.ptr(), data.size());
  }

  // Third case: attempt unit-stride, but avoid copy.
  {
    // Define an alias for the subview block.
    typedef Vector<float>::subview_type::block_type block_type;
    // Define a 1D unit-stride layout.
    typedef Layout<1, row1_type, unit_stride> layout_type;
    // Check for the cost of creating a unit-stride data accessor.
    if (dda::Data<block_type, dda::out, layout_type>::ct_cost != 0)
    {
      // If unit-stride access would require a copy,
      // choose non-unit stride access
      dda::Data<block_type, dda::out> data(subview.block());
      ramp(data.ptr(), data.stride(0), data.size());
    }
    else
    {
      // Create a unit-stride data accessor that will be synched with the subview
      // block when it is being destructed.
      dda::Data<block_type, dda::out, layout_type> data(subview.block());
      ramp(data.ptr(), data.size());
    }

  }
  return 0;
}
