/* Copyright (c) 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    tests/regressions/large_vmul.cpp
    @author  Jules Bergmann
    @date    2007-04-13
    @brief   VSIPL++ Library: Regression for large complex vmul.

    Caused segfault when run with 1 SPE.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define VERBOSE 0

#if VERBOSE
#  include <iostream>
#endif

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/random.hpp>

#include <vsip_csl/test.hpp>

using namespace vsip;
using vsip_csl::equal;
using vsip_csl::almost_equal;

/***********************************************************************
  Definitions - Utility Functions
***********************************************************************/

template <typename T, storage_format_type C>
void
test_vmul(length_type size)
{
  typedef Layout<1, row1_type, dense, C> LP;
  typedef impl::Strided<1, T, LP> block_type;

  Vector<T, block_type> A(size, T(3));
  Vector<T, block_type> B(size, T(4));
  Vector<T, block_type> Z(size);

  Rand<T> gen(0, 0);
  A = gen.randu(size);
  B = gen.randu(size);

  Z = A * B;
  for (index_type i=0; i<size; ++i)
  {
    // Note: almost_equal is necessary for Cbe since SPE and PPE will not
    //       compute idential results.
#if VERBOSE
    if (!almost_equal(Z(i), A(i) * B(i)))
    {
      std::cout << "Z(i)        = " << Z(i) << std::endl;
      std::cout << "A(i) * B(i) = " << A(i) * B(i) << std::endl;
    }
#endif
    test_assert(almost_equal(Z.get(i), A(i) * B(i)));
  }
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_vmul<complex<float>, interleaved_complex>(2048);
  test_vmul<complex<float>, interleaved_complex>(2048+16);
  test_vmul<complex<float>, interleaved_complex>(2048+16+1);

  // Hit stride bug for CBE float backend (081224)
  test_vmul<float, interleaved_complex>(65536);
  test_vmul<complex<float>, interleaved_complex>(65536);
  test_vmul<complex<float>, split_complex>(65536);
}
