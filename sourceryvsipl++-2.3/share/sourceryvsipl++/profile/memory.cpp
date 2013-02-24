/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   Trace memory allocation and library-internal block copying

// Defining these will turn on profiling support 
// for 'memory' and 'user' categories.
#define VSIP_PROFILE_MEMORY 1
#define VSIP_PROFILE_USER 1

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip_csl/dda.hpp>
#include <vsip_csl/profile.hpp>

using namespace vsip;
using namespace vsip_csl;

int
main(int argc, char **argv)
{
  vsipl init(argc, argv);
  {
    typedef Dense<2, float> block_type;
    // Allocate an 4x4 row-major matrix
    Matrix<float, block_type> matrix(4, 4);
  
    // Convert to column-major storage for direct data access.
    typedef dda::Layout<2, col2_type, dda::Stride_unit_dense> layout_type;
    dda::Ext_data<block_type, layout_type> ext(matrix.block(), dda::SYNC_INOUT);
  }
  profile::event<profile::user>("marker 1");
  {
    typedef Dense<2, complex<float> > block_type;

    float real[32];
    float imag[16];

    block_type block(Domain<2>(4, 4), real, imag);
    block.admit();
    Matrix<complex<float>, block_type> matrix(block);
    block.release();
    profile::event<profile::user>("marker 2");
    block.rebind(real);
    block.admit();
    block.release();
  }
  return 0;
}
