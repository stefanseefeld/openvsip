//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#define VERBOSE 0

#if VERBOSE
#  include <iostream>
#endif

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using vsip_csl::equal;

/***********************************************************************
  Definitions
***********************************************************************/

// Test FFTM by-reference out-of-place with given alignment.

// Requires
//   ROWS to be the number of rows
//   SIZE to be the number of elements per row.
//   GAP to be the extra space between rows.
//   ALIGN to be the alignment of the subview (ALIGN <= GAP)

template <typename T, storage_format_type C>
void
test_fftm_op_align(
  length_type rows,
  length_type size,
  length_type gap,
  length_type align)
{
#if VERBOSE
  std::cout << "fftm_op_align(" << rows << " x " << size
	    << ", gap " << gap
	    << ", align " << align << ")\n";
#endif

  typedef Layout<2, row2_type, dense, C> lp_type;

  typedef impl::Strided<2, T, lp_type> block_type;

  typedef Fftm<T, T, row, fft_fwd, by_reference, 1, alg_space> fftm_type;

  fftm_type fftm(Domain<2>(rows, size), 1.f);

  Matrix<T, block_type> in (rows, size + gap);
  Matrix<T, block_type> out(rows, size + gap);

  for (index_type i=0; i<rows; ++i)
    in.row(i) = T(i+1);

  fftm(in (Domain<2>(rows, Domain<1>(align, 1, size))),
       out(Domain<2>(rows, Domain<1>(align, 1, size))));

  for (index_type i=0; i<rows; ++i)
  {
    T got = out.get(i, align);
#if VERBOSE
    if (!equal(got, T((i+1)*size)))
    {
      std::cout << "Error: row " << i << std::endl
		<< "     : expected " << T((i+1)*size) << std::endl
		<< "     : got      " << got << std::endl
		<< "     : got      " << out.get(i, align) << std::endl
	;
    }
#endif
    test_assert(equal(got, T((i+1)*size)));
  }
}



// Test FFTM by-reference in-place with given alignment.

// Requires
//   ROWS to be the number of rows
//   SIZE to be the number of elements per row.
//   GAP to be the extra space between rows.
//   ALIGN to be the alignment of the subview (ALIGN <= GAP)

template <typename T, storage_format_type C>
void
test_fftm_ip_align(
  length_type rows,
  length_type size,
  length_type gap,
  length_type align)
{
#if VERBOSE
  std::cout << "fftm_ip_align(" << rows << " x " << size
	    << ", gap " << gap
	    << ", align " << align << ")\n";
#endif

  typedef Layout<2, row2_type, dense, C> lp_type;

  typedef impl::Strided<2, T, lp_type> block_type;

  typedef Fftm<T, T, row, fft_fwd, by_reference, 1, alg_space> fftm_type;

  fftm_type fftm(Domain<2>(rows, size), 1.f);

  Matrix<T, block_type> inout(rows, size + gap);

  inout = T(1);

  fftm(inout(Domain<2>(rows, Domain<1>(align, 1, size))));

  for (index_type i=0; i<rows; ++i)
    test_assert(inout.get(i, align) == T(size));
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  for (int i=0; i<100; ++i)
  {
  test_fftm_op_align<complex<float>, interleaved_complex>(64, 256, 0, 0);
  test_fftm_op_align<complex<float>, interleaved_complex>(64, 256, 1, 1);
  test_fftm_op_align<complex<float>, interleaved_complex>(64, 256, 1, 0);

  test_fftm_ip_align<complex<float>, interleaved_complex>(64, 256, 0, 0);
  test_fftm_ip_align<complex<float>, interleaved_complex>(64, 256, 1, 1);
  test_fftm_ip_align<complex<float>, interleaved_complex>(64, 256, 1, 0);
  }

  return 0;
}
