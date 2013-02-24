/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.
*/

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip_csl/dda.hpp>

using namespace vsip;
using namespace vsip_csl;

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
    dda::Ext_data<Matrix<float>::subview_type::block_type> ext(subview.block());
    // ...
  }

  // Second case: access dense data.
  {
    // Define an alias for the subview block.
    typedef Matrix<float>::subview_type::block_type block_type;
    // Define a 2D unit-stride layout.
    typedef dda::Layout<2, row2_type, dda::Stride_unit_dense> layout_type;
    dda::Ext_data<block_type, layout_type> ext(subview.block(), dda::SYNC_IN);
    // ...
  }
  // Third case: access dense column-major data.
  {
    // Define an alias for the subview block.
    typedef Matrix<float>::block_type block_type;
    // Define a 2D unit-stride layout.
    typedef dda::Layout<2, col2_type, dda::Stride_unit_dense> layout_type;
    dda::Ext_data<block_type, layout_type> ext(view.block(), dda::SYNC_IN);
    // ...
  }
  return 0;
}
