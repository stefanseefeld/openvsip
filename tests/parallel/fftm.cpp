//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/math.hpp>
#include <vsip/matrix.hpp>
#include <vsip/map.hpp>
#include <test.hpp>
#include "util.hpp"
#include <test/ref/dft.hpp>

#define VERBOSE 0

#if VSIP_IMPL_SAL_FFT
#  define TEST_NON_REALCOMPLEX 0
#  define TEST_NON_POWER_OF_2  0
#else
#  define TEST_NON_REALCOMPLEX 0
#  define TEST_NON_POWER_OF_2  1
#endif

using namespace ovxx;
namespace p = ovxx::parallel;

int number_of_processors;

template <typename Block>
void 
dump_matrix(Block& block, unsigned n, int axis)
{
  p::Communicator comm = p::default_communicator();
  int dummy1;
  int dummy2;

  if (comm.rank() != 0)
    comm.recv(0, &dummy1, 1);

  std::cout << "\n";
  for (unsigned i = 0; i < n; ++i)
  {
    for (unsigned j = 0; j < block.get_local_block().size(2,axis); ++j) 
      { std::cout << block.get_local_block().get(i,j)  << " "; }
    std::cout << "\n";
  }
  std::cout << "\n";

  if (comm.rank() == 0)
  {
    for (unsigned i = 1; i < comm.size(); ++i)
    {
      comm.buf_send(i, &dummy1, 1);
      comm.recv(i, &dummy2, 1);
    }
    for (unsigned i = 1; i < comm.size(); ++i)
      comm.buf_send(i, &dummy1, 1);
  }
  else
  {
    comm.buf_send(0, &dummy2, 1);
    comm.recv(0, &dummy1, 1);
  }
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
  if (subblock(in.row(2)) != no_subblock)
  {
    in.row(2).local()(Domain<1>(0, 1, N))    += T(3);
    if (in.size(1) > 4)  in.row(2).local()(Domain<1>(0, 4, N/4))   += T(-2);
    if (in.size(1) > 13) in.row(2).local()(Domain<1>(0, 13, N/13)) += T(7);
    if (in.size(1) > 27) in.row(2).local()(Domain<1>(0, 27, N/27)) += T(-15);
    if (in.size(1) > 37) in.row(2).local()(Domain<1>(0, 37, N/37)) += T(31);
  }

  in.row(3)    = T(scale);

  for (unsigned i = 0; i < N; ++i)
    in.row(4)(i) = T(std::sin(3.1415926535898*i*4/N));
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

  Domain<2>  domain(Domain<1>(5), Domain<1>(N));

  f_fftm_type f_fftm(domain, 1.0);
  i_fftm_type i_fftm(domain, 1.0/N);

  test_assert(f_fftm.input_size().size() == 5*N);
  test_assert(f_fftm.input_size()[1].size() == N);
  test_assert(f_fftm.output_size().size() == 5*N);
  test_assert(f_fftm.output_size()[1].size() == N);

  test_assert(i_fftm.input_size().size() == 5*N);
  test_assert(i_fftm.input_size()[1].size() == N);
  test_assert(i_fftm.output_size().size() == 5*N);
  test_assert(i_fftm.output_size()[1].size() == N);

  typedef Map<Block_dist,Block_dist>  map_type;
  map_type  map(Block_dist(::number_of_processors), Block_dist(1));
  typedef Dense<2,T,tuple<0,1,2>,map_type>  dist_block_type;
  typedef Matrix<T,dist_block_type>  dist_matrix_type;

  dist_block_type in_block(domain, map);
  dist_block_type out_block(domain, map);
  dist_block_type ref_block(domain, map);
  dist_block_type inv_block(domain, map);

  dist_matrix_type in(in_block);
  dist_matrix_type out(out_block);
  dist_matrix_type ref(ref_block);
  dist_matrix_type inv(inv_block);

  setup_data_x(in);
  test::ref::dft_x(in, ref, -1);

  f_fftm(in, out);
  i_fftm(out, inv);

  double error_ref_out = error_db(ref, out);
  double error_inv_in  = error_db(inv, in);

#if VERBOSE
  std::cout << "out-of-place: 5 x " << N 
	    << "  ref_out: " << error_ref_out
	    << "  inv_in: " << error_inv_in
	    << std::endl;
#endif

  test_assert(error_ref_out < -100);
  test_assert(error_inv_in  < -100);

  out = in;
  f_fftm(out);
  inv = out;
  i_fftm(inv);

  error_ref_out = error_db(ref, out);
  error_inv_in  = error_db(inv, in);

#if VERBOSE
  std::cout << "in-place    : 5 x " << N 
	    << "  ref_out: " << error_ref_out
	    << "  inv_in: " << error_inv_in
	    << std::endl;
#endif

  test_assert(error_ref_out < -100);
  test_assert(error_inv_in  < -100);
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

  Domain<2>  domain(Domain<1>(5), Domain<1>(N));

  f_fftm_type f_fftm(domain, 1.0);
  i_fftm_type i_fftm(domain, 1.0/N);

  test_assert(f_fftm.input_size().size() == 5*N);
  test_assert(f_fftm.output_size().size() == 5*N);

  test_assert(i_fftm.input_size().size() == 5*N);
  test_assert(i_fftm.output_size().size() == 5*N);

  typedef Map<Block_dist,Block_dist>  map_type;
  map_type  map(Block_dist(::number_of_processors), Block_dist(1));
  typedef Dense<2,T,tuple<0,1,2>,map_type>  dist_block_type;
  typedef Matrix<T,dist_block_type>  dist_matrix_type;

  dist_block_type in_block(domain, map);
  dist_block_type out_block(domain, map);
  dist_block_type ref_block(domain, map);
  dist_block_type inv_block(domain, map);

  dist_matrix_type in(in_block);
  dist_matrix_type out(out_block);
  dist_matrix_type ref(ref_block);
  dist_matrix_type inv(inv_block);

  setup_data_x(in);

  test::ref::dft_x(in, ref, -1);
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
  if (subblock(in.col(2)) != no_subblock)
  {
    in.col(2).local()(Domain<1>(0, 1, N))    += T(3);
    if (in.size(0) > 4)  in.col(2).local()(Domain<1>(0, 4, N/4))  += T(-2);
    if (in.size(0) > 13) in.col(2).local()(Domain<1>(0, 13, N/13)) += T(7);
    if (in.size(0) > 27) in.col(2).local()(Domain<1>(0, 27, N/27)) += T(-15);
    if (in.size(0) > 37) in.col(2).local()(Domain<1>(0, 37, N/37)) += T(31);
  }

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

  Domain<2>  domain(Domain<1>(N), Domain<1>(5));

  f_fftm_type  f_fftm(domain, 1.0);
  i_fftm_type  i_fftm(domain, 1.0/N);

  test_assert(f_fftm.input_size().size() == 5*N);
  test_assert(f_fftm.output_size().size() == 5*N);

  test_assert(i_fftm.input_size().size() == 5*N);
  test_assert(i_fftm.output_size().size() == 5*N);

  typedef Map<Block_dist,Block_dist>  map_type;
  map_type  map(Block_dist(1), Block_dist(::number_of_processors));
  typedef Dense<2,T,tuple<0,1,2>,map_type>  dist_block_type;
  typedef Matrix<T,dist_block_type>  dist_matrix_type;

  dist_block_type in_block(domain, map);
  dist_block_type out_block(domain, map);
  dist_block_type ref_block(domain, map);
  dist_block_type inv_block(domain, map);

  dist_matrix_type in(in_block);
  dist_matrix_type out(out_block);
  dist_matrix_type ref(ref_block);
  dist_matrix_type inv(inv_block);

  setup_data_y(in);
  test::ref::dft_y(in, ref, -1);

#if VERBOSE >= 2
  std::cout.precision(3);
  std::cout.setf(std::ios_base::fixed);
#endif

#if VERBOSE >= 2
  dump_matrix(in.block(), N, 1);
  dump_matrix(ref.block(), N, 1);
#endif

  f_fftm(in, out);

#if VERBOSE >= 2
  dump_matrix(in.block(), N, 1);
  dump_matrix(out.block(), N, 1);
#endif

  i_fftm(out, inv);

#if VERBOSE >= 2
  dump_matrix(out.block(), N, 1);
  dump_matrix(inv.block(), N, 1);
#endif

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

  Domain<2>  domain(Domain<1>(N), Domain<1>(5));

  f_fftm_type  f_fftm(domain, 1.0);
  i_fftm_type  i_fftm(domain, 1.0/N);

  test_assert(f_fftm.input_size().size() == 5*N);
  test_assert(f_fftm.output_size().size() == 5*N);

  test_assert(i_fftm.input_size().size() == 5*N);
  test_assert(i_fftm.output_size().size() == 5*N);

  typedef Map<Block_dist,Block_dist>  map_type;
  map_type  map(Block_dist(1), Block_dist(::number_of_processors));
  typedef Dense<2,T,tuple<0,1,2>,map_type>  dist_block_type;
  typedef Matrix<T,dist_block_type>  dist_matrix_type;

  dist_block_type in_block(domain, map);
  dist_block_type out_block(domain, map);
  dist_block_type ref_block(domain, map);
  dist_block_type inv_block(domain, map);

  dist_matrix_type in(in_block);
  dist_matrix_type out(out_block);
  dist_matrix_type ref(ref_block);
  dist_matrix_type inv(inv_block);

  setup_data_y(in);

  test::ref::dft_y(in, ref, -1);
  out = f_fftm(in);
  inv = i_fftm(out);

  test_assert(error_db(ref, out) < -100);
  test_assert(error_db(inv, in) < -100);
}


#if 0

/// Test r->c and c->r by-value Fft.

template <typename T>
void
test_real(const int set, const length_type N)
{
  typedef Fftm<T, std::complex<T>, col, 0, by_value, 1, alg_space>
	f_fftm_type;
  typedef Fftm<std::complex<T>, T, col, 0, by_value, 1, alg_space>
	i_fftm_type;
  const length_type N2 = N/2 + 1;

  f_fftm_type f_fftm(Domain<1>(N), 1.0);
  i_fftm_type i_fftm(Domain<1>(N), 1.0/(N));

  test_assert(f_fftm.input_size().size() == N);
  test_assert(f_fftm.output_size().size() == N2);

  test_assert(i_fftm.input_size().size() == N2);
  test_assert(i_fftm.output_size().size() == N);

  test_assert(f_fftm.scale() == 1.0);  // can represent exactly
  test_assert(i_fftm.scale() > 1.0/(N + 1) && i_fftm.scale() < 1.0/(N - 1));
  test_assert(f_fftm.forward() == true);
  test_assert(i_fftm.forward() == false);

  Matrix<T> in(N, T());
  Matrix<std::complex<T> > out(N2);
  Matrix<std::complex<T> > ref(N2);
  Matrix<T> inv(N);
  Matrix<T> inv2(N);

  setup_ptr(set, in, 3.0);
  out = f_fftm(in);

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
  inv = i_fftm(out);

  test_assert(error_db(inv, in) < -100);

  // make sure out has not been scribbled in during the conversion.
  test_assert(error_db(ref,out) < -100);
}

#endif



template <typename T>
void
test_cases()
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
#  if TEST_NON_REALCOMPLEX
  test_real<T>(242);
#  endif
#endif
};

int
main(int argc, char** argv)
{
  
  vsipl init(argc, argv);

  ::number_of_processors = vsip::num_processors();

  // include debug.hpp to use this function.  It waits for input
  // to allow a debugger to be attached.
  // debug_stub();
#if 0
  // Enable this section for easier debugging.
  p::Communicator& comm = p::default_communicator();
  pid_t pid = getpid();

  std::cout << "rank: "   << comm.rank()
	    << "  size: " << comm.size()
	    << "  pid: "  << pid
	    << std::endl;

  // Stop each process, allow debugger to be attached.
  if (comm.rank() == 0) fgetc(stdin);
  comm.barrier();
  std::cout << "start\n";
#endif

#if defined(VSIP_IMPL_FFT_USE_FLOAT)
  test_cases<float>();
#endif

#if defined(VSIP_IMPL_FFT_USE_DOUBLE) && VSIP_IMPL_TEST_DOUBLE
  test_cases<double>();
#endif

  return 0;
}
