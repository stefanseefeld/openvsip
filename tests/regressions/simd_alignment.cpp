//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#define VERBOSE 0

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>
#include <vsip/selgen.hpp>
#include <vsip/parallel.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;
using vsip_csl::equal;



/***********************************************************************
  Definitions - Misaligned
***********************************************************************/

// Test an unaligned negation, where all views have same relative
// alignment.
//
// Source and destination are not SIMD aligned, but they both have the
// same relative alignement.  Hence they can be processed with aligned
// SIMD code after the initial misalignment is cleaned up.

template <typename T>
void
do_negate_same(length_type size, length_type align, length_type cleanup)
{
  Vector<T> src(size);
  Vector<T> dst(size);

  Domain<1> dom(align, 1, size-align-cleanup);

  src = ramp(T(0), T(1), size);

  dst(dom) = -src(dom);

  for (index_type i=0; i<dom.size(); ++i)
  {
#if VERBOSE
    if (!(dst(dom)(i) == -src(dom)(i)))
    {
      cout << "src:\n" << src << endl;
      cout << "dst:\n" << dst << endl;
    }
#endif
    test_assert(dst(dom)(i) == -src(dom)(i));
  }
}



template <typename T>
void
test_unaligned_same()
{
  do_negate_same<T>(16, 0, 0);

  do_negate_same<T>(16, 1, 0);
  do_negate_same<T>(16, 2, 0);
  do_negate_same<T>(16, 3, 0);

  do_negate_same<T>(16, 1, 1);
  do_negate_same<T>(16, 2, 1);
  do_negate_same<T>(16, 3, 1);

  do_negate_same<T>(16, 1, 2);
  do_negate_same<T>(16, 2, 2);
  do_negate_same<T>(16, 3, 2);

  do_negate_same<T>(16, 1, 3);
  do_negate_same<T>(16, 2, 3);
  do_negate_same<T>(16, 3, 3);
}



/***********************************************************************
  Unaligned test
***********************************************************************/

// Test an "unaligned" negation, with different relative alignment.
//
// Source and destination are not SIMD aligned, and they have the
// different relative alignement.  They must be processed with
// unaligned SIMD code.

template <typename T>
void
do_negate_diff(
  length_type total_size,
  length_type process_size,
  index_type  offset1,
  index_type  offset2)
{
  Vector<T> src(total_size);
  Vector<T> dst(total_size);

  Domain<1> dom1(offset1, 1, process_size);
  Domain<1> dom2(offset2, 1, process_size);

  src = ramp(T(0), T(1), total_size);

  dst(dom1) = -src(dom2);

  for (index_type i=0; i<process_size; ++i)
  {
#if VERBOSE
    if (!(dst(dom1)(i) == -src(dom2)(i)))
    {
      cout << "src:\n" << src << endl;
      cout << "dst:\n" << dst << endl;
    }
#endif
    test_assert(dst(dom1)(i) == -src(dom2)(i));
  }
}



template <typename T>
void
test_unaligned_diff()
{
  do_negate_diff<T>(32, 16, 0, 0);
  do_negate_diff<T>(32, 16, 1, 1);
  do_negate_diff<T>(32, 16, 0, 1);
  do_negate_diff<T>(32, 16, 1, 2); // doesn't dispatch to SIMD UALF
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_unaligned_same<float>();
  test_unaligned_diff<float>();

  return 0;
}
