//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <cmath>
#include <iostream>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>
#include <vsip/core/subblock.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>

using namespace std;
using namespace vsip;
using vsip_csl::equal;
  

template <typename T>
void 
Test_add( T a, T b)
{
  // unit strides
  {
    Vector<T> v1(10, a);
    Vector<T> v2(10, b);
    Vector<T> result(10, T());
    
    // add them
    result = v1 + v2;
    
    // check the result
    for ( index_type i = 0; i < result.size(); ++i )
      assert( equal( result.get(i), a + b ) );
  }

  // non-unit strides
  {
    typedef typename Vector<T>::subview_type vector_subview_type;

    Vector<T> v1_(20, a);
    Vector<T> v2_(30, b);
    Vector<T> result_(50, T());
    vector_subview_type v1 = v1_(Domain<1>(0, 2, 10));
    vector_subview_type v2 = v2_(Domain<1>(0, 3, 10));
    vector_subview_type result = result_(Domain<1>(0, 5, 10));

    // add them
    result = v1 + v2;

    // check the result
    for ( index_type i = 0; i < result.size(); ++i )
      assert( equal( result.get(i), a + b ) );
  }
}


template <typename T>
void 
Test_split_add( complex<T> a, complex<T> b)
{
  typedef vsip::impl::Strided<1, complex<T>,
    vsip::Layout<1, row1_type,
    vsip::dense,
    vsip::split_complex> > split_type;
  
  Vector<complex<T>, split_type>  v1(10, a);
  Vector<complex<T>, split_type>  v2(10, b);
  Vector<complex<T>, split_type>  result(10, T());
  
  // add them
  result = v1 + v2;
  
    // check the result
  for ( index_type i = 0; i < result.size(); ++i )
    assert( equal( result.get(i), a + b ) );
}


template <typename T>
void 
Test_sub( T a, T b)
{
  // unit strides
  {
    Vector<T> v1(10, a);
    Vector<T> v2(10, b);
    Vector<T> result(10, T());
    
    // subtract them
    result = v1 - v2;
    
    // check the result
    for ( index_type i = 0; i < result.size(); ++i )
      assert( equal( result.get(i), a - b ) );
  }

  // non-unit strides
  {
    typedef typename Vector<T>::subview_type vector_subview_type;

    Vector<T> v1_(20, a);
    Vector<T> v2_(30, b);
    Vector<T> result_(50, T());
    vector_subview_type v1 = v1_(Domain<1>(0, 2, 10));
    vector_subview_type v2 = v2_(Domain<1>(0, 3, 10));
    vector_subview_type result = result_(Domain<1>(0, 5, 10));

    // subtract them
    result = v1 - v2;

    // check the result
    for ( index_type i = 0; i < result.size(); ++i )
      assert( equal( result.get(i), a - b ) );
  }
}


template <typename T>
void 
Test_split_sub( complex<T> a, complex<T> b)
{
  typedef vsip::impl::Strided<1, complex<T>,
    vsip::Layout<1, row1_type,
    vsip::dense,
    vsip::split_complex> > split_type;

  Vector<complex<T>, split_type>  v1(10, a);
  Vector<complex<T>, split_type>  v2(10, b);
  Vector<complex<T>, split_type>  result(10, T());

  // subtract them
  result = v1 - v2;

  // check the result
  for ( index_type i = 0; i < result.size(); ++i )
    assert( equal( result.get(i), a - b ) );
}



template <typename T>
void 
Test_mul( T a, T b)
{
  // unit strides
  {
    Vector<T> v1(10, a);
    Vector<T> v2(10, b);
    Vector<T> result(10, T());
    
    // multiply them
    result = v1 * v2;
  
    // check the result
    for ( index_type i = 0; i < result.size(); ++i )
      assert( equal( result.get(i), a * b ) );
  }

  // non-unit strides
  {
    typedef typename Vector<T>::subview_type vector_subview_type;

    Vector<T> v1_(20, a);
    vector_subview_type v1 = v1_(Domain<1>(0, 2, 10));
    Vector<T> v2_(30, b);
    vector_subview_type v2 = v2_(Domain<1>(0, 3, 10));
    Vector<T> result_(50, T());
    vector_subview_type result = result_(Domain<1>(0, 5, 10));

    // multiply them
    result = v1 * v2;

    // check the result
    for ( index_type i = 0; i < result.size(); ++i )
      assert( equal( result.get(i), a * b ) );
  }
}


template <typename T>
void 
Test_split_mul( complex<T> a, complex<T> b)
{
  typedef vsip::impl::Strided<1, complex<T>,
    vsip::Layout<1, row1_type,
    vsip::dense,
    vsip::split_complex> > split_type;

  Vector<complex<T>, split_type>  v1(10, a);
  Vector<complex<T>, split_type>  v2(10, b);
  Vector<complex<T>, split_type>  result(10, T());

  // multiply them
  result = v1 * v2;

  // check the result
  for ( index_type i = 0; i < result.size(); ++i )
    assert( equal( result.get(i), a * b ) );
}


template <typename T>
void 
Test_div( T a, T b)
{
  // unit strides
  {
    Vector<T> v1(10, a);
    Vector<T> v2(10, b);
    Vector<T> result(10, T());
    
    // divide them
    result = v1 / v2;
    
    // check the result
    for ( index_type i = 0; i < result.size(); ++i )
      assert( equal( result.get(i), a / b ) );
  }

  // non-unit strides
  {
    typedef typename Vector<T>::subview_type vector_subview_type;

    Vector<T> v1_(20, a);
    Vector<T> v2_(30, b);
    Vector<T> result_(50, T());
    vector_subview_type v1 = v1_(Domain<1>(0, 2, 10));
    vector_subview_type v2 = v2_(Domain<1>(0, 3, 10));
    vector_subview_type result = result_(Domain<1>(0, 5, 10));

    // divide them
    result = v1 / v2;

    // check the result
    for ( index_type i = 0; i < result.size(); ++i )
      assert( equal( result.get(i), a / b ) );
  }
}


template <typename T>
void 
Test_split_div( complex<T> a, complex<T> b )
{
  typedef vsip::impl::Strided<1, complex<T>,
    vsip::Layout<1, row1_type,
    vsip::dense,
    vsip::split_complex> > split_type;

  Vector<complex<T>, split_type>  v1(10, a);
  Vector<complex<T>, split_type>  v2(10, b);
  Vector<complex<T>, split_type>  result(10, T());

  // divide them
  result = v1 / v2;

  // check the result
  for ( index_type i = 0; i < result.size(); ++i )
    assert( equal( result.get(i), a / b ) );
}





void
Test_add()
{
  Test_add<float>( float(1.0), float(2.0));
  Test_add<double>( double(1.0), double(2.0));
  Test_add<complex<float> >( complex<float>(1.0, 2.0),
    complex<float>(3.0, 4.0) );
  Test_add<complex<double> >( complex<double>(1.0, 2.0), 
    complex<double>(3.0, 4.0) );
  Test_split_add<float>( complex<float>(1.0, 2.0), 
    complex<float>(3.0, 4.0) );
  Test_split_add<double>( complex<double>(1.0, 2.0), 
    complex<double>(3.0, 4.0) );
}

void
Test_sub()
{
  Test_sub<float>( float(3.0), float(2.0));
  Test_sub<double>( double(3.0), double(2.0));
  Test_sub<complex<float> >( complex<float>(4.0, 3.0),
    complex<float>(2.0, 1.0) );
  Test_sub<complex<double> >( complex<double>(4.0, 3.0), 
    complex<double>(2.0, 1.0) );
  Test_split_sub<float>( complex<float>(4.0, 3.0), 
    complex<float>(2.0, 1.0) );
  Test_split_sub<double>( complex<double>(4.0, 3.0), 
    complex<double>(2.0, 1.0) );
}

void
Test_mul()
{
  Test_mul<float>( float(1.0), float(2.0));
  Test_mul<double>( double(1.0), double(2.0));
  Test_mul<complex<float> >( complex<float>(1.0, 2.0),
    complex<float>(3.0, 4.0) );
  Test_mul<complex<double> >( complex<double>(1.0, 2.0), 
    complex<double>(3.0, 4.0) );
  Test_split_mul<float>( complex<float>(1.0, 2.0), 
    complex<float>(3.0, 4.0) );
  Test_split_mul<double>( complex<double>(1.0, 2.0), 
    complex<double>(3.0, 4.0) );
}

void
Test_div()
{
  Test_div<float>( float(1.0), float(2.0));
  Test_div<double>( double(1.0), double(2.0));
  Test_div<complex<float> >( complex<float>(1.0, 2.0),
    complex<float>(3.0, 4.0) );
  Test_div<complex<double> >( complex<double>(1.0, 2.0), 
    complex<double>(3.0, 4.0) );
  Test_split_div<float>( complex<float>(1.0, 2.0), 
    complex<float>(3.0, 4.0) );
  Test_split_div<double>( complex<double>(1.0, 2.0), 
    complex<double>(3.0, 4.0) );
}



int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  Test_add();
  Test_sub();
  Test_mul();
  Test_div();
}

