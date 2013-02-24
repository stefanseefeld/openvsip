/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

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
