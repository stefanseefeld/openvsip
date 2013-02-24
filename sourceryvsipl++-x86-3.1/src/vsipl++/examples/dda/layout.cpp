/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.
*/

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
