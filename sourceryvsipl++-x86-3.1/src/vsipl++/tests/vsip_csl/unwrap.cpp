/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/signal.hpp>

#include <vsip_csl/unwrap.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>

using namespace std;
using namespace vsip;


/***********************************************************************
  Test driver
***********************************************************************/

template <typename ViewType>
void
test_unwrap(
  ViewType v_in,
  length_type rate,
  int sign)
{
  typedef typename ViewType::value_type T;
  length_type size = v_in.size();

  Vector<T> v_ref(size);
  Vector<T> v_out(size);

  // Set up reference output and input.
  v_ref = ramp(T(0.0), sign * T(2*M_PI/rate), size);
  v_in = fmod((v_ref + M_PI), 2*M_PI) - M_PI;

  // Test out-of-place unwrap.
  vsip_csl::unwrap(v_out, v_in);
  if (!vsip_csl::view_equal(v_out, v_ref))
  {
    cout << "v_out\n" << v_out << "v_ref\n" << v_ref;
    std::abort();
  }
  
  // Test in-place unwrap.
  vsip_csl::unwrap(v_in);
  if (!vsip_csl::view_equal(v_in, v_ref))
  {
    cout << "v_in\n" << v_in << "v_ref\n" << v_ref;
    std::abort();
  }
}


template <typename T>
void
test_unwrap_aligned(
  length_type size,
  length_type rate,
  int sign)
{
  Vector<T> v_in(size);
  test_unwrap(v_in, rate, sign);
}


template <typename T>
void
test_unwrap_unaligned(
  length_type size,
  length_type rate,
  int sign)
{
  Vector<T> v_in(size+1);
  test_unwrap(v_in(Domain<1>(1,1,size)), rate, sign);
}


template <typename T>
void
test_unwrap_strided(
  length_type size,
  length_type rate,
  int sign)
{
  Vector<T> v_in(2*size);
  test_unwrap(v_in(Domain<1>(0,2,size)), rate, sign);
}


template <typename T>
void
test_unwrap_all(
  length_type size,
  length_type rate,
  int sign)
{
  cout << "Testing: " << size << ", " << rate << ", " << sign << "\n";
  test_unwrap_aligned<T>(size, rate, sign);
  test_unwrap_unaligned<T>(size, rate, sign);
  test_unwrap_strided<T>(size, rate, sign);
}


/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_unwrap_all<float>(16, 3, 1);
  test_unwrap_all<float>(16, 8, 1);
  test_unwrap_all<float>(16, 64, 1);
  test_unwrap_all<float>(16, 8, -1);

  test_unwrap_all<float>(64, 128, 1);
  test_unwrap_all<float>(65, 128, 1);

  test_unwrap_all<float>(1024, 3, 1);
  test_unwrap_all<float>(1024, 8, 1);
  test_unwrap_all<float>(1024, 32, 1);
  test_unwrap_all<float>(1024, 512, 1);
  test_unwrap_all<float>(1024, 4096, 1);
  test_unwrap_all<float>(1024, 8, -1);
  test_unwrap_all<float>(1024, 32, -1);
}
