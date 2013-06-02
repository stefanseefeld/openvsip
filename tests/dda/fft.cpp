//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   VSIPL++ Library: Test for extdata for a SP object.
///   This file illustrates how data access may be used to implement
///   a signal processing object.

#include <iostream>
#include <cassert>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/dense.hpp>
#include <vsip/vector.hpp>
#include "test.hpp"
#include "output.hpp"

using namespace std;
using namespace vsip;
using ovxx::equal;

// Dummy interleaved FFT function, passes input through to output.

template <typename T>
void
fft_unit_stride(
  T const* in,
  T*       out,
  unsigned size)
{
  for (index_type i=0; i<size; ++i)
    out[i] = in[i];
}



// Dummy split FFT function, passes input through to output.

template <typename T>
void
fft_unit_stride_split(
  T const* in_real,
  T const* in_imag,
  T*       out_real,
  T*       out_imag,
  unsigned size)
{
  for (index_type i=0; i<size; ++i)
  {
    out_real[i] = in_real[i];
    out_imag[i] = in_imag[i];
  }
}



struct FFTFor {};
struct FFTInv {};



/// Dummy FFT object that requires access to interleaved complex.

template <typename T,
	  typename Dir>
class Test_FFT_inter
{
  // Constructors.
public:
  Test_FFT_inter(length_type size)
    : size_  (size),
      buffer_(new T[2*size]),
      verbose_(false)
  {}

  ~Test_FFT_inter()
  { delete[] buffer_; }

  template <typename Block1,
	    typename Block2>
  void operator()(
    char const *            str,
    const_Vector<T, Block1> vin,
    Vector      <T, Block2> vout)
  {
    typedef vsip::Layout<1, row1_type,
			 vsip::unit_stride, vsip::array>
      LP;
    typedef vsip::dda::Data<Block1, dda::in, LP> layout1;
    typedef vsip::dda::Data<Block2, dda::out, LP> layout2;

    // PROFILE: Check sizes, check layout costs.

    // Important to check size.  If vectors are too large, our
    // buffers will overflow.
    test_assert(vin.size()  == size_);
    test_assert(vout.size() == size_);

    if (verbose_)
    {
      cout << "Test_FFT_inter: " << str << endl;
      cout << "get_block_layout<Block1>:\n";
      print_layout<typename vsip::get_block_layout<Block1>::type>(cout);
      cout << endl;

      cout << "LP:\n";
      print_layout<LP>(cout);
      cout << endl;

      cout << "  access_type<LP>(vin.block()) = "
	   <<    access_type<LP>(vin.block()) << endl;
      cout << "  access_type<LP>(vout.block()) = "
	   <<    access_type<LP>(vout.block()) << endl;
      
      cout << "  required_buffer_size<LP>(vin.block()) = "
	   <<    vsip::dda::required_buffer_size<LP>(vin.block()) << endl;
      cout << "  required_buffer_size<LP>(vout.block()) = "
	   <<    vsip::dda::required_buffer_size<LP>(vout.block()) << endl;
    }

    test_assert(vsip::dda::required_buffer_size<LP>(vin.block())  <= sizeof(T)*size_);
    test_assert(vsip::dda::required_buffer_size<LP>(vout.block()) <= sizeof(T)*size_);

    layout1 rin (vin.block(), buffer_ + 0);
    layout2 rout(vout.block(), buffer_ + size_);

    test_assert(rin.stride(0) == 1);
    test_assert(rin.size(0) == size_);

    test_assert(rout.stride(0) == 1);
    test_assert(rout.size(0) == size_);

    fft_unit_stride(rin.ptr(), rout.ptr(), size_);
  }

private:
  length_type size_;
  T*	      buffer_;
  bool        verbose_;
};



/// Dummy FFT object that requires access to split complex.

template <typename T,
	  typename Dir>
class Test_FFT_split
{
  // Constructors.
public:
  Test_FFT_split(length_type size)
    : size_  (size),
      buffer_(new typename T::value_type[4*size]),
      verbose_(false)
  {}

  ~Test_FFT_split()
  { delete[] buffer_; }

  template <typename Block1,
	    typename Block2>
  void operator()(
    char const *            str,
    const_Vector<T, Block1> vin,
    Vector      <T, Block2> vout)
  {
    typedef vsip::Layout<1, row1_type,
      vsip::unit_stride, vsip::split_complex>
		LP;
    typedef vsip::dda::Data<Block1, dda::in, LP> layout1;
    typedef vsip::dda::Data<Block2, dda::out, LP> layout2;

    // PROFILE: Check sizes, check layout costs.

    // Important to check size.  If vectors are too large, our
    // buffers will overflow.
    test_assert(vin.size()  == size_);
    test_assert(vout.size() == size_);

    if (verbose_)
    {
      cout << "Test_FFT_split: " << str << endl;
      cout << "get_block_layout<Block1>:\n";
      print_layout<typename vsip::get_block_layout<Block1>::type>(cout);
      cout << endl;
      
      cout << "LP:\n";
      print_layout<LP>(cout);
      cout << endl;
      
      cout << "  access_type<LP>(vin.block()) = "
	   <<    access_type<LP>(vin.block()) << endl;
      cout << "  access_type<LP>(vout.block()) = "
	   <<    access_type<LP>(vout.block()) << endl;
      
      cout << "  required_buffer_size<LP>(vin.block()) = "
	   <<    vsip::dda::required_buffer_size<LP>(vin.block()) << endl;
      cout << "  required_buffer_size<LP>(vout.block()) = "
	   <<    vsip::dda::required_buffer_size<LP>(vout.block()) << endl;
    }

    test_assert(vsip::dda::required_buffer_size<LP>(vin.block())  <= sizeof(T)*size_);
    test_assert(vsip::dda::required_buffer_size<LP>(vout.block()) <= sizeof(T)*size_);

    layout1 rin (vin.block(), make_pair(buffer_ + 0, buffer_ + size_));
    layout2 rout(vout.block(), make_pair(buffer_ + 2*size_, buffer_ + 3*size_));

    test_assert(rin.stride(0) == 1);
    test_assert(rin.size(0) == size_);

    test_assert(rout.stride(0) == 1);
    test_assert(rout.size(0) == size_);

    fft_unit_stride_split(rin.ptr().first,  rin.ptr().second,
			  rout.ptr().first, rout.ptr().second,
			  size_);
  }

private:
  length_type             size_;
  typename T::value_type* buffer_;
  bool                    verbose_;
};



/***********************************************************************
  Definitions
***********************************************************************/



// Fill vector with sequence of values.

template <typename T,
	  typename Block>
void
fill_view(
  Vector<complex<T>, Block> view,
  int                       k,
  Index<1>	            offset,
  Domain<1>                 /* dom */)
{
  for (index_type i=0; i<view.size(0); ++i)
    view.put(i, complex<T>(T(k*(i + offset[0])+1),
			   T(k*(i + offset[0])+2)));
}



template <typename T,
	  typename Block>
void
fill_view(
  Vector<T, Block> view,
  int              k)
{
  fill_view(view, k, Index<1>(0), Domain<1>(view.size(0)));
}



// Test values in view against sequence.

template <typename T,
	  typename Block>
void
test_view(const_Vector<complex<T>, Block> vec, int k)
{
  for (index_type i=0; i<vec.size(0); ++i)
  {
    if (!equal(vec.get(i), complex<T>(T(k*i+1), T(k*i+2))))
    {
      cout << "ERROR: i        = " << i << endl
	   << "       Got      = " << vec.get(i) << endl
	   << "       expected = " << vec.get(i) << endl;
    }
    test_assert(equal(vec.get(i), complex<T>(T(k*i+1), T(k*i+2))));
  }
}



template <template <typename, typename> class FFT,
	  typename                            Block>
void
test_fft_1d(length_type size, int k)
{
  typedef typename Block::value_type T;
  Vector<T, Block> in (size);
  Vector<T, Block> out(size);

  FFT<T, FFTFor> fft(size);

  fill_view(in, k);
  fft("vector", in, out);
  test_view(out, k);

  fft("subvector", in(Domain<1>(size)), out(Domain<1>(size)));
}

int
main()
{
  vsip::vsipl init;
  test_fft_1d<Test_FFT_inter, ovxx::Strided<1, complex<float> > >(256, 3);
  test_fft_1d<Test_FFT_split, ovxx::Strided<1, complex<float> > >(256, 3);
}
