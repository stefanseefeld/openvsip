/* Copyright (c) 2007, 2011 CodeSourcery, Inc.  All rights reserved. */

/* Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
         copyright notice, this list of conditions and the following
         disclaimer in the documentation and/or other materials
         provided with the distribution.
       * Neither the name of CodeSourcery nor the names of its
         contributors may be used to endorse or promote products
         derived from this software without specific prior written
         permission.

   THIS SOFTWARE IS PROVIDED BY CODESOURCERY, INC. "AS IS" AND ANY
   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CODESOURCERY BE LIABLE FOR
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
   BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
   EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  */

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
