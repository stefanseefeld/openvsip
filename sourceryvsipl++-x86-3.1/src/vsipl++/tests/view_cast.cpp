/* Copyright (c) 2005, 2006, 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/view_cast.cpp
    @author  Jules Bergmann
    @date    2006-10-31
    @brief   VSIPL++ Library: Test View_cast.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/random.hpp>
#include <vsip/selgen.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/view_cast.hpp>

using namespace std;
using namespace vsip;
using vsip_csl::equal;


/***********************************************************************
  Definitions
***********************************************************************/

template <typename T,
	  typename BlockT>
T
expect_vector(const_Vector<T, BlockT> view)
{
  Index<1> idx;
  return maxval(view, idx);
}


template <typename T1,
	  typename T2>
void
test_view_cast(length_type size)
{
  Vector<T1> src(size);
  Vector<T2> dst(size);

  Rand<T1> rand(0);

  src = T1(100) * rand.randu(size);

  dst = vsip_csl::view_cast<T2>(src);

  for (index_type i=0; i<size; ++i)
    test_assert(equal(dst.get(i),
		      static_cast<T2>(src.get(i))));

  src = ramp(T1(0), T1(1), size);

  dst = vsip_csl::view_cast<T2>(src);

  T1 src_sum = expect_vector<T1>(src);
  T2 dst_sum = expect_vector<T2>(vsip_csl::view_cast<T2>(src));

  for (index_type i=0; i<size; ++i)
    test_assert(equal(dst.get(i),
		      static_cast<T2>(src.get(i))));

  test_assert(equal(dst_sum, static_cast<T2>(src_sum)));
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_view_cast<float,  int>(100);
  test_view_cast<double, float>(100);

  // SAL dispatch exists for these
  test_view_cast<float, long>(100);
  test_view_cast<float, short>(100);
  test_view_cast<float, char>(100);
  test_view_cast<float, signed long>(100);
  test_view_cast<float, signed short>(100);
  test_view_cast<float, signed char>(100);
  test_view_cast<float, unsigned long>(100);
  test_view_cast<float, unsigned short>(100);
  test_view_cast<float, unsigned char>(100);

  // these too.
  test_view_cast<long,           float>(100);
  test_view_cast<short,          float>(100);
  test_view_cast<char,           float>(100);
  test_view_cast<signed long,    float>(100);
  test_view_cast<signed short,   float>(100);
  test_view_cast<signed char,    float>(100);
  test_view_cast<unsigned long,  float>(100);
  test_view_cast<unsigned short, float>(100);
  test_view_cast<unsigned char,  float>(100);

  return 0;
}
