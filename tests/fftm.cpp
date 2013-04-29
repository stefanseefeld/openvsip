//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#define VERBOSE 0

#if VERBOSE
#  include <iostream>
#  include <iomanip>
#endif

#include <cmath>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/math.hpp>
#include <vsip/matrix.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include <vsip_csl/error_db.hpp>
#include <vsip_csl/ref_dft.hpp>

#if VSIP_IMPL_SAL_FFT
#  define TEST_NON_REALCOMPLEX 0
#  define TEST_NON_POWER_OF_2  0
#else
#  define TEST_NON_REALCOMPLEX 1
#  define TEST_NON_POWER_OF_2  1
#endif



/***********************************************************************
  Definitions
***********************************************************************/

using namespace std;
using namespace vsip;
using vsip_csl::error_db;
namespace ref = vsip_csl::ref;


template <typename View1,
	  typename View2>
inline void
check_error(
  View1  v1,
  View2  v2,
  double epsilon)
{
  double error = error_db(v1, v2);
#if VERBOSE
  if (error >= epsilon)
  {
    std::cout << "check_error: error >= epsilon" << std::endl;
    std::cout << "  error   = " << error   << std::endl;
    std::cout << "  epsilon = " << epsilon << std::endl;
    std::cout << "  v1 =\n" << v1;
    std::cout << "  v2 =\n" << v2;
  }
#endif
  test_assert(error < epsilon);
}



// Set up input data for Fftm.

template <typename T,
	  typename Block>
void
setup_data_x(Matrix<T, Block> in, float scale = 1)
{
  test_assert(in.size(0) == 5);
  length_type const N = in.size(1);

  in.row(0)    = T();

  in.row(1)    = T();
  in.row(1)(0) = T(scale);

  in.row(2)    = T();
  in.row(2)(0) = T(1);
  in.row(2)(Domain<1>(0, 1, N))    += T(3);
  if (in.size(1) > 4)
    in.row(2)(Domain<1>(0, 4, N/4))  += T(-2);
  if (in.size(1) > 13)
    in.row(2)(Domain<1>(0, 13, N/13)) += T(7);
  if (in.size(1) > 27)
    in.row(2)(Domain<1>(0, 27, N/27)) += T(-15);
  if (in.size(1) > 37)
    in.row(2)(Domain<1>(0, 37, N/37)) += T(31);

  in.row(3)    = T(scale);

  for (unsigned i = 0; i < N; ++i)
    in.row(4)(i)    = T(std::sin(3.1415926535898*i*4/N));
}


/// Test by-reference Fftm (out-of-place and in-place).

template <typename T>
void
test_by_ref_x(length_type N)
{
  typedef Fftm<T, T, row, fft_fwd, by_reference, 1, alg_space>
	f_fftm_type;
  typedef Fftm<T, T, row, fft_inv, by_reference, 1, alg_space>
	i_fftm_type;

  f_fftm_type f_fftm(Domain<2>(Domain<1>(5),Domain<1>(N)), 1.0);
  i_fftm_type i_fftm(Domain<2>(Domain<1>(5),Domain<1>(N)), 1.0/N);

  test_assert(f_fftm.input_size().size() == 5*N);
  test_assert(f_fftm.input_size()[1].size() == N);
  test_assert(f_fftm.output_size().size() == 5*N);
  test_assert(f_fftm.output_size()[1].size() == N);

  test_assert(i_fftm.input_size().size() == 5*N);
  test_assert(i_fftm.input_size()[1].size() == N);
  test_assert(i_fftm.output_size().size() == 5*N);
  test_assert(i_fftm.output_size()[1].size() == N);

  Matrix<T> in(5, N, T());
  Matrix<T> out(5, N, T());
  Matrix<T> ref(5, N, T());
  Matrix<T> inv(5, N, T());

  setup_data_x(in);

  ref::dft_x(in, ref, -1);

  f_fftm(in, out);

#if 0
  for (unsigned i = 0; i < 5; ++i)
  {
    for (unsigned j = 0; j < N; ++j) 
      { cout << in.get(i,j)  << " " << out.get(i,j) << "\n"; }
    cout << "\n";
  }
#endif

  i_fftm(out, inv);

  check_error(ref, out, -100);
  check_error(inv, in,  -100);

  out = in;  f_fftm(out);
  inv = out; i_fftm(inv);

  check_error(ref, out, -100);
  check_error(inv, in,  -100);
}



/// Test by-value Fft.

template <typename T>
void
test_by_val_x(length_type N)
{
  typedef Fftm<T, T, row, fft_fwd, by_value, 1, alg_space>
	f_fftm_type;
  typedef Fftm<T, T, row, fft_inv, by_value, 1, alg_space>
	i_fftm_type;

  f_fftm_type f_fftm(Domain<2>(Domain<1>(5),Domain<1>(N)), 1.0);
  i_fftm_type i_fftm(Domain<2>(Domain<1>(5),Domain<1>(N)), 1.0/N);

  test_assert(f_fftm.input_size().size() == 5*N);
  test_assert(f_fftm.output_size().size() == 5*N);

  test_assert(i_fftm.input_size().size() == 5*N);
  test_assert(i_fftm.output_size().size() == 5*N);

  Matrix<T> in(5, N, T());
  Matrix<T> out(5, N);
  Matrix<T> ref(5, N);
  Matrix<T> inv(5, N);

  setup_data_x(in);

  ref::dft_x(in, ref, -1);
  out = f_fftm(in);
  inv = i_fftm(out);

  test_assert(error_db(ref, out) < -100);
  test_assert(error_db(inv, in) < -100);
}



// Set up input data for Fftm.

template <typename T,
	  typename Block>
void
setup_data_y(Matrix<T, Block> in, float scale = 1)
{
  test_assert(in.size(1) == 5);
  length_type const N = in.size(0);

  in.col(0)    = T();

  in.col(1)    = T();
  in.col(1)(0) = T(scale);

  in.col(2)    = T();
  in.col(2)(0) = T(1);
  in.col(2)(Domain<1>(0, 1, N))    += T(3);
  in.col(2)(Domain<1>(0, 4, N/4))  += T(-2);
  if (in.size(0) > 16)
    in.col(2)(Domain<1>(0, 13, N/13)) += T(7);
  if (in.size(0) > 27)
    in.col(2)(Domain<1>(0, 27, N/27)) += T(-15);
  if (in.size(0) > 37)
    in.col(2)(Domain<1>(0, 37, N/37)) += T(31);

  in.col(3)    = T(scale);

  for (unsigned i = 0; i < N; ++i)
    in.col(4)(i)    = T(std::sin(3.1415926535898*i*4/N));
}


/// Test by-reference Fftm (out-of-place and in-place).

template <typename T>
void
test_by_ref_y(length_type N)
{
  typedef Fftm<T, T, col, fft_fwd, by_reference, 1, alg_space>
	f_fftm_type;
  typedef Fftm<T, T, col, fft_inv, by_reference, 1, alg_space>
	i_fftm_type;

  f_fftm_type f_fftm(Domain<2>(Domain<1>(N),Domain<1>(5)), 1.0);
  i_fftm_type i_fftm(Domain<2>(Domain<1>(N),Domain<1>(5)), 1.0/N);

  test_assert(f_fftm.input_size().size() == 5*N);
  test_assert(f_fftm.output_size().size() == 5*N);

  test_assert(i_fftm.input_size().size() == 5*N);
  test_assert(i_fftm.output_size().size() == 5*N);

  Matrix<T> in(N, 5, T());
  Matrix<T> out(N, 5);
  Matrix<T> ref(N, 5);
  Matrix<T> inv(N, 5);

  setup_data_y(in);

  ref::dft_y(in, ref, -1);

#if 0
  cout.precision(3);
  cout.setf(ios_base::fixed);
#endif

#if 0
  cout << "\n";
  for (unsigned i = 0; i < N; ++i)
  {
    for (unsigned j = 0; j < 5; ++j) 
      { cout << in.get(i,j)  << " "; }
    cout << "\n";
  }
  cout << "\n";
  for (unsigned i = 0; i < N; ++i)
  {
    for (unsigned j = 0; j < 5; ++j) 
      { cout << ref.get(i,j)  << " "; }
    cout << "\n";
  }
#endif
  f_fftm(in, out);
  i_fftm(out, inv);

  test_assert(error_db(ref, out) < -100);
  test_assert(error_db(inv, in) < -100);

  out = in;  f_fftm(out);
  inv = out; i_fftm(inv);

  test_assert(error_db(ref, out) < -100);
  test_assert(error_db(inv, in) < -100);
}



/// Test by-value Fft.

template <typename T>
void
test_by_val_y(length_type N)
{
  typedef Fftm<T, T, col, fft_fwd, by_value, 1, alg_space>
	f_fftm_type;
  typedef Fftm<T, T, col, fft_inv, by_value, 1, alg_space>
	i_fftm_type;

  f_fftm_type f_fftm(Domain<2>(Domain<1>(N),Domain<1>(5)), 1.0);
  i_fftm_type i_fftm(Domain<2>(Domain<1>(N),Domain<1>(5)), 1.0/N);

  test_assert(f_fftm.input_size().size() == 5*N);
  test_assert(f_fftm.output_size().size() == 5*N);

  test_assert(i_fftm.input_size().size() == 5*N);
  test_assert(i_fftm.output_size().size() == 5*N);

  Matrix<T> in(N, 5, T());
  Matrix<T> out(N, 5);
  Matrix<T> ref(N, 5);
  Matrix<T> inv(N, 5);

  setup_data_y(in);
  ref::dft_y(in, ref, -1);
  out = f_fftm(in);
  inv = i_fftm(out);

  test_assert(error_db(ref, out) < -100);
  test_assert(error_db(inv, in) < -100);
}



/// Test r->c and c->r by-value Fft.

template <typename T>
void
test_real(const length_type N)
{
  typedef Fftm<T, std::complex<T>, col, fft_fwd, by_value, 1, alg_space>
	f_fftm_type;
  typedef Fftm<std::complex<T>, T, col, fft_inv, by_value, 1, alg_space>
	i_fftm_type;
  const length_type N2 = N/2 + 1;

  f_fftm_type f_fftm(Domain<2>(Domain<1>(N),Domain<1>(5)), 1.0);
  i_fftm_type i_fftm(Domain<2>(Domain<1>(N),Domain<1>(5)), 1.0/N);

  test_assert(f_fftm.input_size().size() == 5*N);
  test_assert(f_fftm.output_size().size() == 5*N2);

  test_assert(i_fftm.input_size().size() == 5*N2);
  test_assert(i_fftm.output_size().size() == 5*N);

  test_assert(f_fftm.scale() == 1.0);  // can represent exactly
  test_assert(i_fftm.scale() > 1.0/(N + 1) && i_fftm.scale() < 1.0/(N - 1));
  test_assert(f_fftm.forward() == true);
  test_assert(i_fftm.forward() == false);

  Matrix<T> in(N, 5, T());
  Matrix<std::complex<T> > out(N2, 5);
  Matrix<std::complex<T> > ref(N2, 5);
  Matrix<T> inv(N, 5);

  setup_data_y(in);
  ref::dft_y_real(in, ref);
  out = f_fftm(in);
  inv = i_fftm(out);

  test_assert(error_db(ref, out) < -100);
  test_assert(error_db(inv, in) < -100);
}


template <typename T>
void
test()
{
  // Test powers-of-2
  test_by_ref_x<complex<T> >(64);
  test_by_ref_x<complex<T> >(256);

  test_by_ref_y<complex<T> >(256);

  test_by_val_x<complex<T> >(128);
  test_by_val_x<complex<T> >(256);
  test_by_val_x<complex<T> >(512);

  test_by_val_y<complex<T> >(256);


#if TEST_NON_REALCOMPLEX
  test_real<T>(128);
  test_real<T>(16);
#endif

#if TEST_NON_POWER_OF_2
  // Test non-powers-of-2
  test_by_ref_x<complex<T> >(18);
  test_by_ref_x<complex<T> >(68);
  test_by_ref_x<complex<T> >(252);

  test_by_ref_y<complex<T> >(68);

  test_by_val_y<complex<T> >(18);

  // Tests for test r->c, c->r.
  test_real<T>(242);
#endif
};



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_PROVIDE_FFT_FLOAT
  test<float>();
#endif

#if VSIP_IMPL_PROVIDE_FFT_DOUBLE
  test<double>();
#endif

#if VSIP_IMPL_PROVIDE_FFT_LONG_DOUBLE
#  if ! defined(VSIP_IMPL_IPP_FFT)
  test<long double>();
#  endif /* VSIP_IMPL_IPP_FFT */
#endif

  return 0;
}
