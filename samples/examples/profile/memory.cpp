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
