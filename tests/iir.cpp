/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    iir.cpp
    @author  Jules Bergmann
    @date    2005-12-19
    @brief   VSIPL++ Library: Unit tests for [signal.iir] items.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/vector.hpp>
#include <vsip/signal.hpp>
#include <vsip/initfin.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/error_db.hpp>

#define VERBOSE 0

#ifdef VERBOSE
#  include <vsip_csl/output.hpp>
#endif


using namespace std;
using namespace vsip;
using vsip_csl::error_db;


/***********************************************************************
  Test IIR as single FIR -- no recursion
***********************************************************************/

template <obj_state State,
	  typename  T,
	  typename  Block>
void
single_iir_as_fir_case(
  length_type      size,
  length_type      chunk,
  Vector<T, Block> weights)
{
  length_type order = 1;

  Matrix<T> b(order, 3);
  Matrix<T> a(order, 2);

  test_assert(weights.size() == 3);	// IIR is only 2nd order

  test_assert(chunk <= size);
  test_assert(size % chunk == 0);

  b.row(0) = weights;

  a(0, 0) = T(0);
  a(0, 1) = T(0);

  Iir<T, State> iir(b, a, chunk);

  test_assert(iir.kernel_size()  == 2*order);
  test_assert(iir.filter_order() == 2*order);
  test_assert(iir.input_size()   == chunk);
  test_assert(iir.output_size()  == chunk);
  test_assert(iir.continuous_filtering == State);

  Fir<T, nonsym, State> fir(weights, chunk, 1);

  Vector<T> data(size);
  Vector<T> out_iir(size);
  Vector<T> out_fir(size);

  data = ramp(T(1), T(1), size);

  index_type pos = 0;
  while (pos < size)
  {
    iir(data(Domain<1>(pos, 1, chunk)), out_iir(Domain<1>(pos, 1, chunk)));
    fir(data(Domain<1>(pos, 1, chunk)), out_fir(Domain<1>(pos, 1, chunk)));
    pos += chunk;
  }

  float error = error_db(out_iir, out_fir);

#if VERBOSE
  if (error >= -150)
  {
    std::cout << "iir =\n" << out_iir;
    std::cout << "fir =\n" << out_fir;
  }
#endif

  test_assert(error < -150);
}



template <obj_state State,
	  typename  T,
	  typename  Block>
void
iir_as_fir_case(
  length_type      size,
  length_type      chunk,
  Matrix<T, Block> b)
{
  length_type order = b.size(0);

  Matrix<T> a(order, 2, T());

  test_assert(b.size(1) == 3);	// IIR is only 2nd order

  test_assert(chunk <= size);
  test_assert(size % chunk == 0);

  Iir<T, State> iir(b, a, chunk);

  test_assert(iir.kernel_size()  == 2*order);
  test_assert(iir.filter_order() == 2*order);
  test_assert(iir.input_size()   == chunk);
  test_assert(iir.output_size()  == chunk);
  test_assert(iir.continuous_filtering == State);

  Fir<T, nonsym, State>** fir;

  fir = new Fir<T, nonsym, State>*[order];

  for (length_type m=0; m<order; ++m)
    fir[m] = new Fir<T, nonsym, State>(b.row(m), chunk, 1);

  Vector<T> data(size);
  Vector<T> out_iir(size);
  Vector<T> out_fir(size);
  Vector<T> tmp(chunk);

  data = ramp(T(1), T(1), size);

  index_type pos = 0;
  while (pos < size)
  {
    iir(data(Domain<1>(pos, 1, chunk)), out_iir(Domain<1>(pos, 1, chunk)));

    tmp = data(Domain<1>(pos, 1, chunk));

    for (index_type m=0; m<order; ++m)
    {
      fir[m]->operator()(tmp, out_fir(Domain<1>(pos, 1, chunk)));
      tmp = out_fir(Domain<1>(pos, 1, chunk));
    }

    pos += chunk;
  }

  float error = error_db(out_iir, out_fir);

#if VERBOSE
  using vsip::is_same;
  std::cout << "error: " << error
	    << " " << size << "/" << chunk 
	    << " "
	    << (is_same<T, int>::value ? "int" :
		is_same<T, float>::value ? "float" :
		is_same<T, double>::value ? "double" :
		is_same<T, std::complex<float> >::value ? "complex<float>" :
		is_same<T, std::complex<double> >::value ? "complex<double>" :
		"*unknown*")

	    << " " << (State == vsip::state_save ? "state_save" :
		       State == vsip::state_no_save ? "state_no_save" : "*unknown*")
	    << std::endl;
  if (error >= -150)
  {
    std::cout << "iir =\n" << out_iir;
    std::cout << "fir =\n" << out_fir;
  }
#endif

  test_assert(error < -150);

  for (length_type m=0; m<order; ++m)
    delete fir[m];
  delete[] fir;
}



template <typename T>
void
test_iir_as_fir()
{
  Matrix<T> w(4, 3);

  w(0, 0) = T(1);
  w(0, 1) = T(-2);
  w(0, 2) = T(3);

  w(1, 0) = T(3);
  w(1, 1) = T(-1);
  w(1, 2) = T(1);

  w(2, 0) = T(1);
  w(2, 1) = T(0);
  w(2, 2) = T(-1);

  w(3, 0) = T(-1);
  w(3, 1) = T(2);
  w(3, 2) = T(-2);

  length_type size = 128;

  iir_as_fir_case<state_save>(size, size, w);
  iir_as_fir_case<state_save>(size, size/2, w);
  iir_as_fir_case<state_save>(size, size/4, w);

  iir_as_fir_case<state_no_save>(size, size, w);
  iir_as_fir_case<state_no_save>(size, size/2, w);
  iir_as_fir_case<state_no_save>(size, size/4, w);
}



/***********************************************************************
  Test IIR as summation
***********************************************************************/

///
// Test:
//  [1] iir copy cons
//  [2] iir assignment
template <typename  T>
void
sum_case(
  length_type      size,
  length_type      chunk)
{
  test_assert(chunk <= size);
  test_assert(size % chunk == 0);

  length_type order = 1;

  Matrix<T> b(order, 3);
  Matrix<T> a(order, 2);
  Matrix<T> b3(order, 3, T());
  Matrix<T> a3(order, 2, T());

  b(0, 0) = T(1);
  b(0, 1) = T(0);
  b(0, 2) = T(0);

  a(0, 0) = T(-1);
  a(0, 1) = T(0);

  Iir<T, state_save> iir1(b, a, chunk);
  Iir<T, state_save> iir2 = iir1;
  Iir<T, state_save> iir3(b3, a3, chunk); // [1]

  test_assert(iir1.kernel_size()  == 2*order);
  test_assert(iir1.filter_order() == 2*order);
  test_assert(iir1.input_size()   == chunk);
  test_assert(iir1.output_size()  == chunk);
  test_assert(iir1.continuous_filtering == state_save);

  test_assert(iir2.kernel_size()  == 2*order);
  test_assert(iir2.filter_order() == 2*order);
  test_assert(iir2.input_size()   == chunk);
  test_assert(iir2.output_size()  == chunk);
  test_assert(iir2.continuous_filtering == state_save);

  test_assert(iir3.kernel_size()  == 2*order);
  test_assert(iir3.filter_order() == 2*order);
  test_assert(iir3.input_size()   == chunk);
  test_assert(iir3.output_size()  == chunk);
  test_assert(iir3.continuous_filtering == state_save);

  Vector<T> data(size);
  Vector<T> out1(size);
  Vector<T> out2(size);
  Vector<T> out3(size);
  Vector<T> exp(size);

  data = ramp(T(1), T(1), size);

  index_type pos = 0;
  while (pos < size)
  {
    iir1(data(Domain<1>(pos, 1, chunk)), out1(Domain<1>(pos, 1, chunk)));
    iir2(data(Domain<1>(pos, 1, chunk)), out2(Domain<1>(pos, 1, chunk)));

    if (pos == 0)
    {
      out3(Domain<1>(pos, 1, chunk)) = out2(Domain<1>(pos, 1, chunk));
      iir3 = iir1; // [2]
      test_assert(iir3.kernel_size()  == 2*order);
      test_assert(iir3.filter_order() == 2*order);
      test_assert(iir3.input_size()   == chunk);
      test_assert(iir3.output_size()  == chunk);
      test_assert(iir3.continuous_filtering == state_save);
    }
    else
      iir3(data(Domain<1>(pos, 1, chunk)), out3(Domain<1>(pos, 1, chunk)));

    pos += chunk;
  }

  iir1.reset();

  T accum = T();
  for (index_type i=0; i<size; ++i)
  {
    accum += data(i);
    exp(i) = accum;
  }

  float error1 = error_db(out1, exp);
  float error2 = error_db(out2, exp);
  float error3 = error_db(out3, exp);

#if VERBOSE
  if (error1 >= -150 || error2 >= -150 || error3 >= -150)
  {
    std::cout << "out1 =\n" << out1;
    std::cout << "out2 =\n" << out2;
    std::cout << "out3 =\n" << out3;
    std::cout << "exp  =\n" << exp;
  }
#endif

  test_assert(error1 < -150);
  test_assert(error2 < -150);
  test_assert(error3 < -150);
}



template <typename T>
void
test_sum()
{
  sum_case<T>(128, 32);
  sum_case<T>(16, 16);
}



/***********************************************************************
  main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_iir_as_fir<int>();
  test_iir_as_fir<float>();
  test_iir_as_fir<complex<float> >();

  test_sum<int>();
  test_sum<float>();
  test_sum<complex<float> >();
}
