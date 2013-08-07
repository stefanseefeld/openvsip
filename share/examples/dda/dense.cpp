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
