//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef TEST_CONVOLUTION_CONVOLUTION_HPP
#define TEST_CONVOLUTION_CONVOLUTION_HPP

#define VERBOSE 0

#include <vsip/vector.hpp>
#include <vsip/signal.hpp>
#include <vsip/initfin.hpp>
#include <vsip/random.hpp>
#include <vsip/parallel.hpp>
#include <vsip/core/metaprogramming.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/ref_conv.hpp>
#include <vsip_csl/error_db.hpp>

#if VERBOSE
#  include <iostream>
#  include <vsip_csl/output.hpp>
#endif

using namespace std;
using namespace vsip;
using namespace vsip_csl;

double const ERROR_THRESH = -70;



/***********************************************************************
  Definitions
***********************************************************************/

length_type
expected_kernel_size(
  vsip::symmetry_type symmetry,
  vsip::length_type   coeff_size)
{
  if (symmetry == vsip::nonsym)
    return coeff_size;
  else if (symmetry == vsip::sym_even_len_odd)
    return 2*coeff_size-1;
  else /* (symmetry == vsip::sym_even_len_even) */
    return 2*coeff_size;
}
		     

length_type
expected_output_size(
  support_region_type supp,
  length_type         M,    // kernel length
  length_type         N,    // input  length
  length_type         D)    // decimation factor
{
  if      (supp == support_full)
    return ((N + M - 2)/D) + 1;
  else if (supp == support_same)
    return ((N - 1)/D) + 1;
  else //(supp == support_min)
  {
#if VSIP_IMPL_CONV_CORRECT_MIN_SUPPORT_SIZE
    return ((N - M + 1) / D) + ((N - M + 1) % D == 0 ? 0 : 1);
#else
    return ((N - 1)/D) - ((M-1)/D) + 1;
#endif
  }
}



length_type
expected_shift(
  support_region_type supp,
  length_type         M,     // kernel length
  length_type         /*D*/) // decimation factor
{
  if      (supp == support_full)
    return 0;
  else if (supp == support_same)
    return (M/2);
  else //(supp == support_min)
    return (M-1);
}



/// Test convolution with nonsym symmetry.

template <typename            T,
	  support_region_type support>
void
test_conv_nonsym(
  length_type N,
  length_type M,
  index_type  c1,
  index_type  c2,
  int         k1,
  int         k2)
{
  symmetry_type const        symmetry = nonsym;

  typedef Convolution<const_Vector, symmetry, support, T> conv_type;

  // length_type const M = 5;				// filter size
  // length_type const N = 100;				// input size
  length_type const D = 1;				// decimation
  length_type const P = expected_output_size(support, M, N, D);
  // ((N-1)/D) - ((M-1)/D) + 1;	// output size

  int shift = expected_shift(support, M, D);

  Vector<T> coeff(M, T());

  coeff(c1) = T(k1);
  coeff(c2) = T(k2);

  conv_type conv(coeff, Domain<1>(N), D);

  test_assert(conv.symmetry() == symmetry);
  test_assert(conv.support()  == support);

  test_assert(conv.kernel_size().size()  == M);
  test_assert(conv.filter_order().size() == M);
  test_assert(conv.input_size().size()   == N);
  test_assert(conv.output_size().size()  == P);


  Vector<T> in(N);
  Vector<T> out(P, T(100));
  Vector<T> exp(P, T(201));

  for (index_type i=0; i<N; ++i)
    in(i) = T(i);

  conv(in, out);

  for (index_type i=0; i<P; ++i)
  {
    T val1, val2;

    if ((int)i + shift - (int)c1 < 0 || i + shift - c1 >= in.size())
      val1 = T();
    else
      val1 = in(i + shift - c1);

    if ((int)i + shift - (int)c2 < 0 || i + shift - c2 >= in.size())
      val2 = T();
    else
      val2 = in(i + shift - c2);

    exp.put(i, T(k1) * val1 + T(k2) * val2);
  }

  double error = error_db(out, exp);
#if VERBOSE
  std::cout << "error-nonsym: " << error
	    << "  M/N/P " << M << "/" << N << "/" << P
	    << (support == vsip::support_full  ? " full"  :
		support == vsip::support_same  ? " same"  :
		support == vsip::support_min   ? " min"   :
		                                 " *unknown*" )
	    << std::endl;
#endif
  test_assert(error < ERROR_THRESH);
}



/// Test general 1-D convolution.

template <symmetry_type       symmetry,
	  support_region_type support,
	  typename            T,
	  typename            Block1,
	  typename            Block2,
	  typename            Block3>
void
test_conv_base(
  Vector<T, Block1>        in,
  Vector<T, Block2>        out,
  const_Vector<T, Block3>  coeff,	// coefficients
  length_type              D,		// decimation
  length_type const        n_loop = 2)
{
  using vsip::impl::conditional;
  using vsip::impl::is_global_map;

  typedef Convolution<const_Vector, symmetry, support, T> conv_type;

  length_type M = expected_kernel_size(symmetry, coeff.size());
  length_type N = in.size();
  length_type P = out.size();

  length_type expected_P = expected_output_size(support, M, N, D);

  test_assert(P == expected_P);

  conv_type conv(coeff, Domain<1>(N), D);

  test_assert(conv.symmetry() == symmetry);
  test_assert(conv.support()  == support);

  test_assert(conv.kernel_size().size()  == M);
  test_assert(conv.filter_order().size() == M);
  test_assert(conv.input_size().size()   == N);
  test_assert(conv.output_size().size()  == P);

  // Determine type of map to use for expected result.
  // If 'out's map is global, make it global_map.
  // Otherwise, make it a local_map.
  typedef typename
          conditional<is_global_map<typename Block2::map_type>::value,
		      Replicated_map<1>,
		      Local_map>::type
          map_type;
  typedef Dense<1, T, row1_type, map_type> block_type;

  Vector<T, block_type> exp(P);

  for (index_type loop=0; loop<n_loop; ++loop)
  {
    for (index_type i=0; i<N; ++i)
      in(i) = T(3*loop+i);

    conv(in, out);

    ref::conv(symmetry, support, coeff, in, exp, D);

    // Check result
    Index<1> idx;
    double error   = error_db(out, exp);
    double maxdiff = maxval(magsq(out - exp), idx);

#if VERBOSE
    std::cout << "error: " << error
	      << "  M/N/P " << M << "/" << N << "/" << P
	      << (symmetry == vsip::sym_even_len_odd  ? " odd"  :
	          symmetry == vsip::sym_even_len_even ? " even" :
	          symmetry == vsip::nonsym            ? " nonsym" :
		                                        " *unknown*" )
	      << (support == vsip::support_full  ? " full"  :
	          support == vsip::support_same  ? " same"  :
	          support == vsip::support_min   ? " min"   :
		                                   " *unknown*" )
	      << std::endl;

    if (error > ERROR_THRESH)
    {
      cout << "exp = \n" << exp;
      cout << "out = \n" << out;
      cout << "diff = \n" << mag(exp-out);
    }
#endif

    test_assert(error < ERROR_THRESH || maxdiff < 1e-4);
  }
}



/// Test convolution for non-unit strides.

template <typename            T,
          support_region_type support>
void
test_conv_nonunit_stride(
  length_type N,
  length_type M,
  stride_type stride)
{
  symmetry_type const         symmetry = nonsym;
  length_type const           D = 1; // decimation

  typedef typename Vector<T>::subview_type vector_subview_type;

  length_type const P = expected_output_size(support, M, N, D);

  Vector<T> kernel(M, T());

  Rand<T> rgen(0);
  kernel = rgen.randu(M);

  Vector<T> in_base(N * stride);
  Vector<T> out_base(P * stride, T(100));

  vector_subview_type  in =  in_base( Domain<1>(0, stride, N) );
  vector_subview_type out = out_base( Domain<1>(0, stride, P) );

  test_conv_base<symmetry, support>(in, out, kernel, D, 1);
}



/// Test general 1-D convolution.

template <typename            T,
	  symmetry_type       symmetry,
	  support_region_type support,
	  typename            T1,
	  typename            Block1>
void
test_conv(
  length_type              N,		// input size
  length_type              D,		// decimation
  const_Vector<T1, Block1> coeff,	// coefficients
  length_type const        n_loop = 2)
{
  length_type M = expected_kernel_size(symmetry, coeff.size());
  length_type P = expected_output_size(support, M, N, D);

  Vector<T> in(N);
  Vector<T> out(P, T(100));

  test_conv_base<symmetry, support>(in, out, coeff, D, n_loop);
}



/// Test general 1-D convolution, with distributed arguments.

template <typename            T,
	  symmetry_type       symmetry,
	  support_region_type support,
	  typename            MapT>
void
test_conv_dist(
  length_type              N,		// input size
  length_type              M,		// coeff size
  length_type              D,		// decimation
  length_type const        n_loop = 2)
{
  length_type const P = expected_output_size(support, M, N, D);

  typedef Dense<1, T, row1_type, MapT> block_type;
  typedef Vector<T, block_type>        view_type;

  MapT map(num_processors());

  view_type coeff(M, map);
  view_type in(N, map);
  view_type out(P, T(100), map);

  Rand<T> rgen(0);
  vsip::impl::assign_local(coeff, rgen.randu(M));

  test_conv_base<symmetry, support>(in, out, coeff, D, n_loop);
}



// Run a set of convolutions for given type and size
//   (with symmetry = nonsym and decimation = 1).

template <typename T>
void
cases_nonsym(length_type size)
{
  test_conv_nonsym<T, support_min>(size, 4, 0, 1, +1, +1);
  test_conv_nonsym<T, support_min>(size, 5, 0, 1, +1, -1);

  test_conv_nonsym<T, support_same>(size, 4, 0, 1, +1, +1);
  test_conv_nonsym<T, support_same>(size, 5, 0, 1, +1, -1);

  test_conv_nonsym<T, support_full>(size, 4, 0, 1, +1, +1);
  test_conv_nonsym<T, support_full>(size, 5, 0, 1, +1, -1);
}



// Run a set of convolutions for given type and size
//   (using vectors with strides other than one).

template <typename T>
void
cases_nonunit_stride(length_type size)
{
  test_conv_nonunit_stride<T, support_min>(size, 4, 3);
  test_conv_nonunit_stride<T, support_min>(size, 5, 2);

  test_conv_nonunit_stride<T, support_full>(size, 4, 3);
  test_conv_nonunit_stride<T, support_full>(size, 5, 2);

  test_conv_nonunit_stride<T, support_same>(size, 4, 3);
  test_conv_nonunit_stride<T, support_same>(size, 5, 2);
}



// Run a set of convolutions for given type, symmetry, input size, coeff size
// and decmiation.

template <typename      T,
	  symmetry_type Sym>
void
cases_conv(length_type size, length_type M, length_type D, bool rand)
{
  Vector<T> coeff(M, T());

  if (rand)
  {
    Rand<T> rgen(0);
    coeff = rgen.randu(M);
  }
  else
  {
    coeff(0)   = T(-1);
    coeff(M-1) = T(2);
  }

  test_conv<T, Sym, support_min>(size, D, coeff);
  test_conv<T, Sym, support_same>(size, D, coeff);
  test_conv<T, Sym, support_full>(size, D, coeff);
}



// Run a set of convolutions for given type, symmetry, input size, coeff size
// and decmiation, using distributed arugments.

template <typename      T>
void
cases_conv_dist(length_type size, length_type M, length_type D)
{
  symmetry_type const sym = nonsym;

  typedef Map<Block_dist> map_type;

  test_conv_dist<T, sym, support_min, map_type> (size, M, D);
  test_conv_dist<T, sym, support_same, map_type>(size, M, D);
  test_conv_dist<T, sym, support_full, map_type>(size, M, D);
}



// Run a single convolutions for given type, symmetry, support, input
// size, coeff size and decmiation.

template <typename            T,
	  symmetry_type       Sym,
	  support_region_type Sup>
void
single_conv(length_type size, length_type M, length_type D,
	    length_type n_loop, bool rand)
{
  Vector<T> coeff(M, T());

  if (rand)
  {
    Rand<T> rgen(0);
    coeff = rgen.randu(M);
  }
  else
  {
    coeff(0)   = T(-1);
    coeff(M-1) = T(2);
  }

  test_conv<T, Sym, Sup>(size, D, coeff, n_loop);
}



template <typename T>
void
cases(bool rand)
{
  // check that M == N works
  cases_conv<T, nonsym>(8, 8, 1, rand);
  cases_conv<T, nonsym>(5, 5, 1, rand);
  cases_conv<T, sym_even_len_even>(8, 4, 1, rand);
  cases_conv<T, sym_even_len_odd>(7, 4, 1, rand);

  cases_conv<T, nonsym>(5, 4, 1, rand);
  cases_conv<T, nonsym>(5, 4, 2, rand);
  cases_conv<T, nonsym>(5, 4, 3, rand);
  cases_conv<T, nonsym>(5, 4, 4, rand);

  cases_nonsym<T>(100);

  for (length_type size=32; size<=1024; size *= 4)
  {
    cases_nonsym<T>(size);
    cases_nonsym<T>(size+3);
    cases_nonsym<T>(2*size);

    cases_nonunit_stride<T>(size);

    cases_conv<T, nonsym>(size,      8,  1, rand);
    cases_conv<T, nonsym>(2*size,    7,  2, rand);
    cases_conv<T, nonsym>(size+4,    6,  3, rand);

    cases_conv<T, sym_even_len_even>(size,   5,  1, rand);
    cases_conv<T, sym_even_len_even>(size+1, 6,  2, rand);

    cases_conv<T, sym_even_len_odd>(size,   4,  1, rand);
    cases_conv<T, sym_even_len_odd>(size+3, 3,  2, rand);
  }
}

#endif // TEST_CONVOLUTION_CONVOLUTION_HPP
