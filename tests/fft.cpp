//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

// Set to 1 to enable verbose output.
#define VERBOSE     0
// Set to 0 to disble use of random values.
#define FILL_RANDOM 1

#include <cmath>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include <test.hpp>

#include "fft_common.hpp"

using namespace ovxx;

template <typename View1,
	  typename View2>
inline void
check_error(
  char const* where,
  View1       v1,
  View2       v2,
  double      epsilon)
{
  double error = error_db(v1, v2);
#if VERBOSE
  if (error >= epsilon)
  {
    std::cout << "check_error " << where << ": error >= epsilon" << std::endl;
    std::cout << "  error   = " << error   << std::endl;
    std::cout << "  epsilon = " << epsilon << std::endl;
    std::cout << "  v1 =\n" << v1;
    std::cout << "  v2 =\n" << v2;
  }
#else
  (void)where;
#endif
  test_assert(error < epsilon);
}



// Setup input data for Fft.

template <typename T,
	  typename Block>
void
setup_ptr(int set, Vector<T, Block> in, float scale = 1)
{
  length_type const N = in.size();

  switch(set)
  {
  default:
  case 0:
    in    = T();
    break;
  case 1:
    in    = T();
    in(0) = T(scale);
    break;
  case 2:
    in    = T();
    in(0) = T(1);
    if (N >  1) in(Domain<1>(0, 1, N))    += T(3);
    if (N >  4) in(Domain<1>(0, 4, N/4))  += T(-2);
    if (N > 13) in(Domain<1>(0, 13, N/13)) += T(7);
    if (N > 27) in(Domain<1>(0, 27, N/27)) += T(-15);
    if (N > 37) in(Domain<1>(0, 37, N/37)) += T(31);
    break;
  case 3:
    in    = T(scale);
    break;
  }
}



/// Test complex by-reference Fft (out-of-place and in-place).

template <typename T, storage_format_type Complex_format>
void
test_complex_by_ref(int set, length_type N)
{
  typedef std::complex<T> CT;
  typedef Fft<const_Vector, CT, CT, fft_fwd, by_reference, 1, alg_space>
	f_fft_type;
  typedef Fft<const_Vector, CT, CT, fft_inv, by_reference, 1, alg_space>
	i_fft_type;

  f_fft_type f_fft(Domain<1>(N), 1.0);
  i_fft_type i_fft(Domain<1>(N), 1.0/N);

  test_assert(f_fft.input_size().size() == N);
  test_assert(f_fft.output_size().size() == N);

  test_assert(i_fft.input_size().size() == N);
  test_assert(i_fft.output_size().size() == N);

  typedef vsip::impl::Strided<1, CT,
    vsip::Layout<1, row1_type,
    vsip::dense, Complex_format> >
    block_type;

  Vector<CT, block_type> in(N, CT());
  Vector<CT, block_type> out(N);
  Vector<CT, block_type> ref(N);
  Vector<CT, block_type> inv(N);

  setup_ptr(set, in);

  test::ref::dft(in, ref, -1);
  f_fft(in, out);
  i_fft(out, inv);

  test_assert(error_db(ref, out) < -100);
  check_error("test_complex_by_ref", inv, in, -100);

  out = in;  f_fft(out);
  inv = out; i_fft(inv);

  test_assert(error_db(ref, out) < -100);
  test_assert(error_db(inv, in) < -100);
}



/// Test complex by-value Fft.

template <typename T, storage_format_type Complex_format>
void
test_complex_by_val(int set, length_type N)
{
  typedef std::complex<T> CT;
  typedef Fft<const_Vector, CT, CT, fft_fwd, by_value, 1, alg_space>
	f_fft_type;
  typedef Fft<const_Vector, CT, CT, fft_inv, by_value, 1, alg_space>
	i_fft_type;

  f_fft_type f_fft(Domain<1>(N), 1.0);
  i_fft_type i_fft(Domain<1>(N), 1.0/N);

  test_assert(f_fft.input_size().size() == N);
  test_assert(f_fft.output_size().size() == N);

  test_assert(i_fft.input_size().size() == N);
  test_assert(i_fft.output_size().size() == N);

  typedef vsip::impl::Strided<1, CT,
    vsip::Layout<1, row1_type, vsip::dense, Complex_format> >
    block_type;

  Vector<CT, block_type> in(N, CT());
  Vector<CT, block_type> out(N);
  Vector<CT, block_type> ref(N);
  Vector<CT, block_type> inv(N);

  setup_ptr(set, in);

  test::ref::dft(in, ref, -1);
  out = f_fft(in);
  inv = i_fft(out);

  test_assert(error_db(ref, out) < -100);
  test_assert(error_db(inv, in) < -100);
}



/// Test r->c and c->r by-value Fft.

template <typename T>
void
test_real(const int set, const length_type N)
{
  typedef Fft<const_Vector, T, std::complex<T>, 0, by_value, 1, alg_space>
	f_fft_type;
  typedef Fft<const_Vector, std::complex<T>, T, 0, by_value, 1, alg_space>
	i_fft_type;
  const length_type N2 = N/2 + 1;

  f_fft_type f_fft(Domain<1>(N), 1.0);
  i_fft_type i_fft(Domain<1>(N), 1.0/(N));

  test_assert(f_fft.input_size().size() == N);
  test_assert(f_fft.output_size().size() == N2);

  test_assert(i_fft.input_size().size() == N2);
  test_assert(i_fft.output_size().size() == N);

  test_assert(f_fft.scale() == 1.0);  // can represent exactly
  test_assert(i_fft.scale() > 1.0/(N + 1) && i_fft.scale() < 1.0/(N - 1));
  test_assert(f_fft.forward() == true);
  test_assert(i_fft.forward() == false);

  Vector<T> in(N, T());
  Vector<std::complex<T> > out(N2);
  Vector<std::complex<T> > ref(N2);
  Vector<T> inv(N);
  Vector<T> inv2(N);

  setup_ptr(set, in, 3.0);
  out = f_fft(in);

  if (set == 1)
  {
    setup_ptr(3, ref, 3.0);
    test_assert(error_db(ref, out) < -100);
  }
  if (set == 3)
  {
    setup_ptr(1, ref, 3.0 * N);
    test_assert(error_db(ref, out) < -100);
  }

  ref = out;
  inv = i_fft(out);
  check_error("test_real", inv, in, -100);

  // make sure out has not been scribbled in during the conversion.
  test_assert(error_db(ref,out) < -100);
}



// Check 1D 

template <typename T>
void
test_1d()
{
  test_complex_by_ref<T, vsip::interleaved_complex>(2, 64);
  test_complex_by_ref<T, vsip::split_complex>(2, 64);

  test_complex_by_ref<T, vsip::interleaved_complex>(2, 256);
  test_complex_by_ref<T, vsip::split_complex>(2, 256);

  test_complex_by_val<T, vsip::interleaved_complex>(1, 128);
  test_complex_by_val<T, vsip::split_complex>(1, 128);
  test_complex_by_val<T, vsip::interleaved_complex>(2, 256);
  test_complex_by_val<T, vsip::split_complex>(2, 256);
  test_complex_by_val<T, vsip::interleaved_complex>(3, 512);
  test_complex_by_val<T, vsip::split_complex>(3, 512);

  test_real<T>(1, 128);
  test_real<T>(3, 16);

#if TEST_NON_POWER_OF_2
  test_complex_by_ref<T, vsip::interleaved_complex>(1, 68);
  test_complex_by_ref<T, vsip::split_complex>(1, 68);
  test_complex_by_ref<T, vsip::interleaved_complex>(2, 252);
  test_complex_by_ref<T, vsip::split_complex>(2, 252);
  test_complex_by_ref<T, vsip::interleaved_complex>(3, 17);
  test_complex_by_ref<T, vsip::split_complex>(3, 17);
  test_real<T>(2, 242);
#endif
}


int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  // show_config();

#if VSIP_IMPL_PROVIDE_FFT_FLOAT
  test_1d<float>();
#endif 

#if VSIP_IMPL_PROVIDE_FFT_DOUBLE && VSIP_IMPL_TEST_DOUBLE
  test_1d<double>();
#endif 

#if VSIP_IMPL_PROVIDE_FFT_LONG_DOUBLE
  test_1d<long double>();
#endif

  return 0;
}
