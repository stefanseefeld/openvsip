//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#define VERBOSE 0

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/math.hpp>
#include <vsip/matrix.hpp>
#include <test.hpp>
#include <cmath>

using namespace ovxx;

template <typename T, vsip::symmetry_type sym>
void
test_fir(
  vsip::length_type D,
  vsip::length_type M,
  vsip::length_type N)
{
  const unsigned  insize = 2 * M * N;
  const unsigned  outsize = ((2 * M * N) + D - 1) / D + 1;
  vsip::Vector<T> input(insize);
  vsip::Vector<T> output1(outsize);
  vsip::Vector<T> output2(2 * M * (N+D-1)/D);
  vsip::Vector<T> output3(2 * M * (N+D-1)/D);

  vsip::Vector<T> convinput(insize+M, T(0));  // room for initial state
  vsip::Vector<T> convout((insize+M-1)/D + 1, T(0)); // per spec
  vsip::Vector<T> kernel(M);

  for (vsip::length_type i = 0; i < insize; ++i)
    input.put(i, T(i+1));
  for (vsip::length_type i = 0; i < M; ++i)
    kernel.put(i, T(2*i+1));

  vsip::Convolution<vsip::const_Vector,sym,vsip::support_same,T,1>  conv(
    kernel, vsip::Domain<1>(convinput.size()), D);
  
  const vsip::length_type  pad = (sym == vsip::nonsym) ? M/2 :
    (sym == vsip::sym_even_len_even) ? M : M - 1;
  convinput(vsip::Domain<1>(pad, 1, insize)) = input;
  conv(convinput, convout);  // emulate chained FIR

  vsip::Fir<>  dummy(
    vsip::const_Vector<>(vsip::length_type(3),vsip::scalar_f(1)), N*10);
  test_assert(dummy.decimation() == 1);
  vsip::Fir<T,sym,vsip::state_save,1>  fir1a(kernel, N, D);
  vsip::Fir<T,sym,vsip::state_save,1>  fir1b(fir1a);
  vsip::Fir<T,sym,vsip::state_no_save,1>  fir2(kernel, N, D);

  test_assert(fir1a.symmetry == sym);
  test_assert(fir2.symmetry == sym);
  test_assert(fir1a.continuous_filter == vsip::state_save);
  test_assert(fir2.continuous_filter == vsip::state_no_save);
 
  const vsip::length_type  order = (sym == vsip::nonsym) ? M :
    (sym == vsip::sym_even_len_even) ? 2 * M : (2 * M) - 1;
  test_assert(fir1a.kernel_size() == order);
  test_assert(fir1b.kernel_size() == order);
  test_assert(fir1a.filter_order() == order);
  test_assert(fir1b.filter_order() == order);
  // test_assert(fir1a.symmetry()
  test_assert(fir1a.input_size() == N);
  test_assert(fir1b.input_size() == N);
  test_assert(fir1a.output_size() == (N+D-1)/D);
  test_assert(fir1b.output_size() == (N+D-1)/D);
  test_assert(fir1a.continuous_filtering() == fir1a.continuous_filter);
  test_assert(fir2.continuous_filtering() == fir2.continuous_filter);
  test_assert(fir1a.decimation() == D);
  test_assert(fir1b.decimation() == D);

  vsip::length_type got1a = 0;
  for (vsip::length_type i = 0; i < 2 * M; ++i) // chained
  {
    got1a += fir1a(
      input(vsip::Domain<1>(i * N, 1, N)),
      output1(vsip::Domain<1>(got1a, 1, (N + D - 1) / D)));
  }

  // vsip::Vector<T> o1(output1.size(), T(0));
  // o1 = convout(vsip::Domain<1>(output1.size())) - output1;
  
  vsip::length_type got1b = 0;
  vsip::length_type got2 = 0;
  for (vsip::length_type i = 0; i < 2 * M; ++i)  // not
  {
    got1b += fir1b(input(vsip::Domain<1>(i * N, 1, N)),
          output2(vsip::Domain<1>(got1b, 1, (N+D-1)/D)));
    fir1b.reset();
    got2 += fir2(input(vsip::Domain<1>(i * N, 1, N)),
         output3(vsip::Domain<1>(got2, 1, (N+D-1)/D)));
  }

  vsip::Vector<T>  reference(convout(vsip::Domain<1>(got1a)));
  vsip::Vector<T>  result(output1(vsip::Domain<1>(got1a)));

  test_assert(outsize - got1a <= 1);
  double error_rr = test::diff(result, reference);
  double error_23 = test::diff(output2(vsip::Domain<1>(got1b)),
			     output3(vsip::Domain<1>(got1b)));
#if VERBOSE
  using vsip::is_same;
  std::cout << "error_rr: " << error_rr
	    << "  error_23: " << error_23
	    << " " << D << "/" << M << "/" << N
	    << " "
	    << (is_same<T, float>::value ? "float" :
		is_same<T, double>::value ? "double" :
		is_same<T, std::complex<float> >::value ? "complex<float>" :
		is_same<T, std::complex<double> >::value ? "complex<double>" :
		"*unknown*")

	    << " " << (sym == vsip::sym_even_len_even ? "even" :
		       sym == vsip::sym_even_len_odd ? "odd" : "nonsym")
	    << std::endl;
  if (!(error_rr < -100))
  {
    std::cout << "result: " << result;
    std::cout << "reference: " << reference;
  }
  if (!(error_23 < -100))
  {
    std::cout << "- (num chunks)     2*M      : " << (2*M) << std::endl;
    std::cout << "- (in chunk size)  N        : " << (N) << std::endl;
    std::cout << "- (out chunk size) (N+D-1)/D: " << ((N+D-1)/D) << std::endl;
    std::cout << "output2:\n" << output2(vsip::Domain<1>(got1b));
    std::cout << "output3:\n" << output3(vsip::Domain<1>(got1b));
  }
#endif
  test_assert(error_rr < -100);
  // NOTE: got1a may not equal got1b.  fir1b is reset between each frame.
  // if N % D != 0, this may lead to different number of outputs.
  test_assert(got1a == got1b || N % D != 0);

  test_assert(got1b == got2);
  test_assert(error_23 < -100);
}
  
int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  test_fir<float,vsip::nonsym>(1,2,3);
  test_fir<float,vsip::nonsym>(1,3,5);
  test_fir<float,vsip::nonsym>(1,3,9);
  test_fir<float,vsip::nonsym>(1,4,8);
  test_fir<float,vsip::nonsym>(1,23,31);
  test_fir<float,vsip::nonsym>(1,32,1024);

  test_fir<float,vsip::nonsym>(2,3,5);
  test_fir<float,vsip::nonsym>(2,3,9);
  test_fir<float,vsip::nonsym>(2,4,8);
  test_fir<float,vsip::nonsym>(2,23,31);
  test_fir<float,vsip::nonsym>(2,32,1024);

#if VSIP_IMPL_TEST_DOUBLE
  test_fir<double,vsip::nonsym>(2,3,5);
  test_fir<double,vsip::nonsym>(2,3,9);
  test_fir<double,vsip::nonsym>(2,4,8);
  test_fir<double,vsip::nonsym>(2,23,31);
  test_fir<double,vsip::nonsym>(2,32,1024);
#endif

  test_fir<std::complex<float>,vsip::nonsym>(2,3,5);
  test_fir<std::complex<float>,vsip::nonsym>(2,3,9);
  test_fir<std::complex<float>,vsip::nonsym>(2,4,8);
  test_fir<std::complex<float>,vsip::nonsym>(2,23,31);
  test_fir<std::complex<float>,vsip::nonsym>(2,32,1024);

#if VSIP_IMPL_TEST_DOUBLE
  test_fir<std::complex<double>,vsip::nonsym>(2,3,5);
  test_fir<std::complex<double>,vsip::nonsym>(2,3,9);
  test_fir<std::complex<double>,vsip::nonsym>(2,4,8);
  test_fir<std::complex<double>,vsip::nonsym>(2,23,31);
  test_fir<std::complex<double>,vsip::nonsym>(2,32,1024);
#endif

  test_fir<float,vsip::nonsym>(3,4,8);
  test_fir<float,vsip::nonsym>(3,4,21);
  test_fir<float,vsip::nonsym>(3,9,27);
  test_fir<float,vsip::nonsym>(3,23,31);
  test_fir<float,vsip::nonsym>(3,32,1024);

  test_fir<float,vsip::nonsym>(4,5,13);
  test_fir<float,vsip::nonsym>(4,7,31);
  test_fir<float,vsip::nonsym>(4,8,32);
  test_fir<float,vsip::nonsym>(4,23,31);
  test_fir<float,vsip::nonsym>(4,32,1024);

  test_fir<float,vsip::sym_even_len_even>(1,1,3);
  test_fir<float,vsip::sym_even_len_even>(1,2,3);
  test_fir<float,vsip::sym_even_len_even>(1,3,5);
  test_fir<float,vsip::sym_even_len_even>(1,3,9);
  test_fir<float,vsip::sym_even_len_even>(1,4,8);
  test_fir<float,vsip::sym_even_len_even>(1,23,57);
  test_fir<float,vsip::sym_even_len_even>(1,32,1024);

  test_fir<float,vsip::sym_even_len_even>(2,2,3);
  test_fir<float,vsip::sym_even_len_even>(2,3,5);
  test_fir<float,vsip::sym_even_len_even>(2,3,9);
  test_fir<float,vsip::sym_even_len_even>(2,4,8);
  test_fir<float,vsip::sym_even_len_even>(2,23,57);
  test_fir<float,vsip::sym_even_len_even>(2,32,1024);

  test_fir<float,vsip::sym_even_len_even>(3,3,5);
  test_fir<float,vsip::sym_even_len_even>(3,4,8);
  test_fir<float,vsip::sym_even_len_even>(3,23,57);
  test_fir<float,vsip::sym_even_len_even>(3,32,1024);

  test_fir<float,vsip::sym_even_len_odd>(1,2,3);
  test_fir<float,vsip::sym_even_len_odd>(1,3,5);
  test_fir<float,vsip::sym_even_len_odd>(1,3,9);
  test_fir<float,vsip::sym_even_len_odd>(1,4,9);
  test_fir<float,vsip::sym_even_len_odd>(1,23,57);
  test_fir<float,vsip::sym_even_len_odd>(1,32,1024);

  test_fir<float,vsip::sym_even_len_odd>(2,2,3);
  test_fir<float,vsip::sym_even_len_odd>(2,3,5);
  test_fir<float,vsip::sym_even_len_odd>(2,3,9);
  test_fir<float,vsip::sym_even_len_odd>(2,4,10);
  test_fir<float,vsip::sym_even_len_odd>(2,23,57);
  test_fir<float,vsip::sym_even_len_odd>(2,32,1024);

  test_fir<float,vsip::sym_even_len_odd>(3,3,5);
  test_fir<float,vsip::sym_even_len_odd>(3,4,9);
  test_fir<float,vsip::sym_even_len_odd>(3,23,55);
  test_fir<float,vsip::sym_even_len_odd>(3,32,1024);

  return 0;
}
