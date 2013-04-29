/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   Trace memory allocation and library-internal block copying

// report memory (de-)allocation
#define VSIP_PROFILE_MEMORY 1
// report data transfers and other copies.
#define VSIP_PROFILE_COPY 1
// report user events
#define VSIP_PROFILE_USER 1

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/dda.hpp>
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
    typedef Layout<2, col2_type, dense> layout_type;
    dda::Data<block_type, dda::inout, layout_type> data(matrix.block());
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
