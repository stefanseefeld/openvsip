//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip/selgen.hpp>
#include <test.hpp>
#include <test/ref/corr.hpp>

#define VERBOSE 0

using namespace ovxx;

// On some compute devices, the numerical accuracy is slightly
// reduced, hence the slightly relaxed threshold used when 
// comparing views for equality
#ifdef VSIP_IMPL_CBE_SDK
float threshold = -80;
#elif VSIP_IMPL_CUDA_FFT
float threshold = -80;
#else
float threshold = -100;
#endif



/***********************************************************************
  Definitions
***********************************************************************/

/// Test general 1-D correlation.

template <typename            T,
	  support_region_type support>
void
test_corr(
  bias_type                bias,
  length_type              M,		// reference size
  length_type              N,		// input size
  length_type const        n_loop = 3)
{
  typedef typename scalar_of<T>::type scalar_type;
  typedef Correlation<const_Vector, support, T> corr_type;

  length_type const P = test::ref::corr_output_size(support, M, N);
  corr_type corr((Domain<1>(M)), Domain<1>(N));

  test_assert(corr.support()  == support);

  test_assert(corr.reference_size().size() == M);
  test_assert(corr.input_size().size()     == N);
  test_assert(corr.output_size().size()    == P);

  Rand<T> rand(0);

  Vector<T> ref(M);
  Vector<T> in(N);
  Vector<T> out(P, T(100));
  Vector<T> chk(P, T(101));

  for (index_type loop=0; loop<n_loop; ++loop)
  {
    if (loop == 0)
    {
      ref = T(1);
      in  = ramp(T(0), T(1), N);
    }
    else if (loop == 1)
    {
      ref = rand.randu(M);
      in  = ramp(T(0), T(1), N);
    }
    else
    {
      ref = rand.randu(M);
      in  = rand.randu(N);
    }

    corr(bias, ref, in, out);

    test::ref::corr(bias, support, ref, in, chk);

    double error = test::diff(out, chk);

#if VERBOSE
    if (error > threshold)
    {
      for (index_type i=0; i<P; ++i)
      {
	cout << i << ":  out = " << out(i)
	     << "  chk = " << chk(i)
	     << endl;
      }
      std::cout << "corr<"
	// << vsip::impl::diag_detail::Class_name<T>::name()
		<< ">("
		<< (bias == biased ? "biased" : "unbiased") << " "
		<< (support == support_min  ? "support_min"  :
		    support == support_same ? "support_same" :
		    support == support_full ? "support_full" : "other")
		<< " M: " << M
		<< " N: " << N << ") = "
		<< error << "\n";
    }
#endif

    test_assert(error < threshold);
  }
}



/// Test general 1-D correlation.

template <typename            Tag,
	  typename            T,
	  support_region_type support>
void
test_impl_corr(
  bias_type                bias,
  length_type              M,		// reference size
  length_type              N,		// input size
  length_type const        n_loop = 3)
{
  using namespace dispatcher;
  typedef typename scalar_of<T>::type scalar_type;
  typedef typename Dispatcher<op::corr<1, support, T, 0, alg_time> >::type
    corr_type;

  length_type const P = test::ref::corr_output_size(support, M, N);

  corr_type corr((Domain<1>(M)), Domain<1>(N));

  // Correlation_impl doesn't define support():
  // test_assert(corr.support()  == support);

  test_assert(corr.reference_size().size() == M);
  test_assert(corr.input_size().size()     == N);
  test_assert(corr.output_size().size()    == P);

  Rand<T> rand(0);

  Vector<T> ref(M);
  Vector<T> in(N);
  Vector<T> out(P, T(100));
  Vector<T> chk(P, T(101));

  for (index_type loop=0; loop<n_loop; ++loop)
  {
    if (loop == 0)
    {
      ref = T(1);
      in  = ramp(T(0), T(1), N);
    }
    else if (loop == 1)
    {
      ref = rand.randu(M);
      in  = ramp(T(0), T(1), N);
    }
    else
    {
      ref = rand.randu(M);
      in  = rand.randu(N);
    }

    corr.impl_correlate(bias, ref, in, out);

    test::ref::corr(bias, support, ref, in, chk);

    double error = test::diff(out, chk);

#if VERBOSE
    if (error > threshold)
    {
      for (index_type i=0; i<P; ++i)
      {
	cout << i << ":  out = " << out(i)
	     << "  chk = " << chk(i)
	     << endl;
      }
    }
    std::cout << "impl_core<"
	// << vsip::impl::diag_detail::Class_name<Tag>::name() << ", "
	// << vsip::impl::diag_detail::Class_name<T>::name()
		<< ">("
		<< (bias == biased ? "biased" : "unbiased") << " "
		<< (support == support_min ? "support_min" :
		    support == support_same ? "support_same" :
		    support == support_full ? "support_full" : "other")
		<< " M: " << M
		<< " N: " << N << ") = "
		<< error << "\n";
#endif

    test_assert(error < threshold);
  }
}



template <typename T>
void
corr_cases(length_type M, length_type N)
{
  test_corr<T, support_min>(biased,   M, N);
  test_corr<T, support_min>(unbiased, M, N);

  test_corr<T, support_same>(biased,   M, N);
  test_corr<T, support_same>(unbiased, M, N);

  test_corr<T, support_full>(biased,   M, N);
  test_corr<T, support_full>(unbiased, M, N);
}


template <typename T>
void
corr_cover()
{
  corr_cases<T>(8, 8);

  corr_cases<T>(1, 128);
  corr_cases<T>(7, 128);
  corr_cases<T>(8, 128);
  corr_cases<T>(9, 128);

  corr_cases<T>(7, 127);
  corr_cases<T>(8, 127);
  corr_cases<T>(9, 127);

  corr_cases<T>(7, 129);
  corr_cases<T>(8, 129);
  corr_cases<T>(9, 129);
}



template <typename Tag,
	  typename T>
void
impl_corr_cases(length_type M, length_type N)
{
  test_impl_corr<Tag, T, support_min>(biased,   M, N);
  test_impl_corr<Tag, T, support_min>(unbiased, M, N);

  test_impl_corr<Tag, T, support_same>(biased,   M, N);
  test_impl_corr<Tag, T, support_same>(unbiased, M, N);

  test_impl_corr<Tag, T, support_full>(biased,   M, N);
  test_impl_corr<Tag, T, support_full>(unbiased, M, N);
}



template <typename Tag,
	  typename T>
void
impl_corr_cover()
{
  impl_corr_cases<Tag, T>(8, 8);

  impl_corr_cases<Tag, T>(1, 128);
  impl_corr_cases<Tag, T>(7, 128);
  impl_corr_cases<Tag, T>(8, 128);
  impl_corr_cases<Tag, T>(9, 128);

  impl_corr_cases<Tag, T>(32, 128);
  impl_corr_cases<Tag, T>(64, 128);
  impl_corr_cases<Tag, T>(96, 128);

  impl_corr_cases<Tag, T>(7, 127);
  impl_corr_cases<Tag, T>(8, 127);
  impl_corr_cases<Tag, T>(9, 127);

  impl_corr_cases<Tag, T>(7, 129);
  impl_corr_cases<Tag, T>(8, 129);
  impl_corr_cases<Tag, T>(9, 129);

  impl_corr_cases<Tag, T>(12, 96);
  impl_corr_cases<Tag, T>(12, 97);
  impl_corr_cases<Tag, T>(11, 97);
  impl_corr_cases<Tag, T>(12, 98);
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0
  corr_cover<float>();
#else
  // Test user-visible correlation
  corr_cover<float>();
  corr_cover<complex<float> >();

  // Test optimized implementation
  impl_corr_cover<be::opt, float>();
  impl_corr_cover<be::opt, complex<float> >();

# if VSIP_IMPL_HAVE_CVSIP
  // Test C-VSIPL implementation
  impl_corr_cover<be::cvsip, float>();
  impl_corr_cover<be::cvsip, complex<float> >();
# endif
  // Test generic implementation
  impl_corr_cover<be::generic, float>();
  impl_corr_cover<be::generic, complex<float> >();

# if VSIP_IMPL_TEST_DOUBLE
  // Test user-visible correlation
  corr_cover<double>();
  corr_cover<complex<double> >();

  // Test optimized implementation
  impl_corr_cover<be::opt, double>();
  impl_corr_cover<be::opt, complex<double> >();

#  if VSIP_IMPL_HAVE_CVSIP
  // Test generic implementation
  impl_corr_cover<be::cvsip, double>();
  impl_corr_cover<be::cvsip, complex<double> >();
#  endif

  // Test generic implementation
  impl_corr_cover<be::generic, double>();
  impl_corr_cover<be::generic, complex<double> >();
# endif // VSIP_IMPL_TEST_DOUBLE
#endif
}
