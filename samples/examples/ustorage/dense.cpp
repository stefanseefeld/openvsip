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
///   Demonstrate the use of user storage with ``Dense`` blocks.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/dense.hpp>

using namespace vsip;

int
main(int argc, char **argv)
{
  vsipl init(argc, argv);
  {
    // Operate on real-valued user-storage
    float data[8] = {0};
    Dense<1, float> block(8, data);
    block.admit();
    block.put(0, 1.);
    block.release();
    assert(data[0] == 1.);
  }
  {
    // Operate on interleaved-complex user-storage
    float data[16] = {0};
    Dense<1, complex<float> > block(8, data);
    block.admit();
    block.put(0, complex<float>(1., 2.));
    block.release();
    assert(data[0] == 1. && data[1] == 2.);
  }
  {
    // Operate on interleaved-complex user-storage
    float real[8] = {0};
    float imag[8] = {0};
    Dense<1, complex<float> > block(8, real, imag);
    block.admit();
    block.put(0, complex<float>(1., 2.));
    block.release();
    assert(real[0] == 1. && imag[0] == 2.);
  }
  {
    // Rebind to alternate user storage
    float data[16] = {0};
    Dense<1, complex<float> > block(8, data);
    float real[8] = {0};
    float imag[8] = {0};
    block.rebind(real, imag);
    block.admit();
    block.put(0, complex<float>(1., 2.));
    block.release();
    assert(real[0] == 1. && imag[0] == 2.);
  }
}
