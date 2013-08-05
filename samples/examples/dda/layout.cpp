/* Copyright (c) 2009, 2011 CodeSourcery, Inc.  All rights reserved. */

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

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/dda.hpp>

using namespace vsip;

int 
main(int argc, char **argv)
{
  vsipl init(argc, argv);

  // Create a dense row-major matrix of size 8x8.
  Matrix<float> view(8,8);
  // Create a submatrix refering to every other row in the above.
  Matrix<float>::subview_type subview = view(Domain<2>(Domain<1>(0, 2, 4), 8));

  // First case: access non-dense data.
  {
    dda::Data<Matrix<float>::subview_type::block_type, dda::inout> data(subview.block());
    // ...
  }

  // Second case: access dense data.
  {
    // Define an alias for the subview block.
    typedef Matrix<float>::subview_type::block_type block_type;
    // Define a 2D unit-stride layout.
    typedef Layout<2, row2_type, dense> layout_type;
    dda::Data<block_type, dda::in, layout_type> data(subview.block());
    // ...
  }
  // Third case: access dense column-major data.
  {
    // Define a row-major matrix type.
    typedef Matrix<float>::block_type block_type;
    // Define a 2D column-major dense layout.
    typedef Layout<2, col2_type, dense> layout_type;
    dda::Data<block_type, dda::in, layout_type> data(view.block());
    // ...
  }
  return 0;
}
