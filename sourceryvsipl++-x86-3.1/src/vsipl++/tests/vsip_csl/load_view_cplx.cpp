/* Copyright (c) 2006-2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/vsip_csl/load_view_cplx.cpp
    @author  Jules Bergmann
    @date    2006-09-28
    @brief   VSIPL++ Library: Unit-tests for vsip_csl/load_view.hpp
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define DEBUG 0

#include <iostream>
#if DEBUG
#include <unistd.h>
#endif
#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/test-storage.hpp>
#include <vsip_csl/load_view.hpp>
#include <vsip_csl/save_view.hpp>

#include "load_save.hpp"
#include "test_common.hpp"

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions
***********************************************************************/


template <typename T>
void
test_complex_ls(
  length_type size)
{
  char const* filename = "test.load_view.tmpfile";

  typedef Vector<complex<T> > view_type;
  view_type s_view(size);
  setup(s_view, 1);

  bool swap_bytes = true;
  save_view(filename, s_view, swap_bytes);


  // Make sure the view didn't change (this is a regression test
  // against the view being swapped in place prior to being written
  // to a file).
  view_type ref_view(size);
  setup(ref_view, 1);
  test_assert(view_equal(ref_view, s_view));


  // Now read the data back in scalar form and ensure that the real
  // and imaginary parts were not swapped (also a regression test)
  typedef Vector<T> scalar_view_type;
  scalar_view_type l_view(size * 2, T());
  load_view(filename, l_view, !swap_bytes);

  T real = l_view.get(2); // second real value
  T imag = l_view.get(3); // second imaginary value
  matlab::Swap_value<T,true>::swap(&real);
  matlab::Swap_value<T,true>::swap(&imag);

  test_assert(equal(real, ref_view.real().get(1)));
  test_assert(equal(imag, ref_view.imag().get(1)));
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if DEBUG
  // Enable this section for easier debugging.
  impl::Communicator& comm = impl::default_communicator();
  pid_t pid = getpid();

  cout << "rank: "   << comm.rank()
       << "  size: " << comm.size()
       << "  pid: "  << pid
       << endl;

  // Stop each process, allow debugger to be attached.
  if (comm.rank() == 0) fgetc(stdin);
  comm.barrier();
  cout << "start\n";
#endif

  // Note: Complex versions of these tests are found in the module
  // 'load_view_cplx.cpp'.  The tests were split to improve compile time.
  test_type<complex<float> >();
  test_type<complex<double> >();

  // The tests below handle additional checks for particular bug fixes
  // (regressions).
  test_complex_ls<float>(10);
  test_complex_ls<double>(10);
}


