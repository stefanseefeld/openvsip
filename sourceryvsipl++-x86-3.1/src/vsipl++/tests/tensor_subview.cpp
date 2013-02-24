/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/tensor_subview.cpp
    @author  Jules Bergmann
    @date    2005-07-01
    @brief   VSIPL++ Library: Unit tests for tensor subviews.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/tensor.hpp>

#include <vsip_csl/test.hpp>

using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

class Index_value
{
public:
  Index_value(int k, int i,
	      length_type M,
	      length_type N,
	      length_type P)
    : k_(k), i_(i), M_(M), N_(N), P_(P) {}

  float operator()(
    index_type  m,
    index_type  n,
    index_type  p)
    const
  { return k_*(m*(N_*P_)+n*P_+p)+i_; }

private:
  int         k_;
  int         i_;
  length_type M_;
  length_type N_;
  length_type P_;
};


class Bind_01
{
public:
  Bind_01(Index_value obj, index_type a, index_type b)
    : obj_(obj), a_(a), b_(b) {}

  float operator()(index_type i) const { return obj_(a_, b_, i); }

private:
  Index_value obj_;
  index_type  a_;
  index_type  b_;
};

class Bind_02
{
public:
  Bind_02(Index_value obj, index_type a, index_type b)
    : obj_(obj), a_(a), b_(b) {}

  float operator()(index_type i) const { return obj_(a_, i, b_); }

private:
  Index_value obj_;
  index_type  a_;
  index_type  b_;
};

class Bind_12
{
public:
  Bind_12(Index_value obj, index_type a, index_type b)
    : obj_(obj), a_(a), b_(b) {}

  float operator()(index_type i) const { return obj_(i, a_, b_); }

private:
  Index_value obj_;
  index_type  a_;
  index_type  b_;
};



template <typename VectorT,
	  typename IndexObj>
void
check_v(
  VectorT         vec,
  IndexObj const& obj)
{
  for (index_type p=0; p<vec.size(); ++p)
    test_assert(vec(p) == obj(p));
}



template <typename VectorT,
	  typename IndexObj>
void
set_v(
  VectorT         vec,
  IndexObj const& obj)
{
  for (index_type p=0; p<vec.size(); ++p)
    vec(p) = obj(p);
}



template <typename VectorT,
	  typename IndexObj>
void
check_ext_v(
  VectorT         vec,
  IndexObj const& obj)
{
  typedef typename VectorT::block_type block_type;
  typedef typename VectorT::value_type value_type;

  typedef typename get_block_layout<block_type>::layout_type LP;

  dda::Data<block_type, dda::in> ext(vec.block());

  value_type* ptr    = ext.ptr();
  stride_type stride = ext.stride(0);
  length_type size   = ext.size(0);

  for (index_type p=0; p<size; ++p)
  {
    test_assert(*ptr == obj(p));
    ptr += stride;
  }
}



template <typename Vector1,
	  typename Vector2,
	  typename Vector3,
	  typename IndexObj1,
	  typename IndexObj2,
	  typename IndexObj3>
void
t_subvector(
  Vector1          vec1,
  Vector2          vec2,
  Vector3          vec3,
  IndexObj1 const& obj1,
  IndexObj2 const& obj2,
  IndexObj3 const& obj3)
{
  typedef typename Vector1::value_type T;

  // Check initial values.
  check_v(vec1, obj1);
  check_v(vec2, obj1);
  check_v(vec3, obj1);
  
  // Rotate through values.
  set_v  (vec1, obj2);
  check_v(vec2, obj2);
  set_v  (vec2, obj3);
  check_v(vec3, obj3);
  set_v  (vec3, obj1);

  // Check vector lengths.
  length_type size = vec1.size();

  test_assert(vec1.size() == size);
  test_assert(vec2.size() == size);
  test_assert(vec3.size() == size);
  test_assert(vec1.size(0) == size);
  test_assert(vec2.size(0) == size);
  test_assert(vec3.size(0) == size);

  // Test vector assignment.
  Vector<T> new1(size); set_v(new1, obj1);
  Vector<T> new2(size); set_v(new2, obj2);
  Vector<T> new3(size); set_v(new3, obj3);

  check_v(vec1, obj1);
  check_v(vec2, obj1);
  check_v(vec3, obj1);

  vec1 = new2;

  check_v(vec1, obj2);
  check_v(vec2, obj2);
  check_v(vec3, obj2);

  vec2 = new3;

  check_v(vec1, obj3);
  check_v(vec2, obj3);
  check_v(vec3, obj3);

  vec3 = new1;

  check_v(vec1, obj1);
  check_v(vec2, obj1);
  check_v(vec3, obj1);
}




/// Test vector subviews of a tensor.

void
test_tensor_vector()
{
  typedef float T;

  length_type const M = 10;
  length_type const N = 12;
  length_type const P = 14;

  Index_value val1(1, 0, M, N, P);
  Index_value val2(2, 1, M, N, P);
  Index_value val3(3, 2, M, N, P);

  Tensor<T> ten(M, N, P);

  Tensor<T>::whole_domain_type whole = Tensor<T>::whole_domain;

  for (index_type m=0; m<M; ++m)
    for (index_type n=0; n<N; ++n)
      for (index_type p=0; p<P; ++p)
	ten(m, n, p) = val1(m,n,p);

  test_assert(ten(whole, 0, 0).size() == M);
  test_assert(ten(0, whole, 0).size() == N);
  test_assert(ten(0, 0, whole).size() == P);

  test_assert(ten(whole, 0, 0).size(0) == M);
  test_assert(ten(0, whole, 0).size(0) == N);
  test_assert(ten(0, 0, whole).size(0) == P);


  for (index_type m=0; m<M; ++m)
    for (index_type n=0; n<N; ++n)
      t_subvector(ten(m, n, whole),
		  ten(m, whole, whole).row(n),
		  ten(whole, n, whole).row(m),
		  Bind_01(val1, m, n),
		  Bind_01(val2, m, n),
		  Bind_01(val3, m, n));

  for (index_type m=0; m<M; ++m)
    for (index_type p=0; p<P; ++p)
      t_subvector(ten(m, whole, p),
		  ten(m, whole, whole).col(p),
		  ten(whole, whole, p).row(m),
		  Bind_02(val1, m, p),
		  Bind_02(val2, m, p),
		  Bind_02(val3, m, p));

  for (index_type n=0; n<N; ++n)
    for (index_type p=0; p<P; ++p)
      t_subvector(ten(whole, n, p),
		  ten(whole, whole, p).col(n),
		  ten(whole, n, whole).col(p),
		  Bind_12(val1, n, p),
		  Bind_12(val2, n, p),
		  Bind_12(val3, n, p));
}



/// Test matrix subviews of a tensor

void
test_tensor_matrix()
{
  typedef float T;

  length_type const M = 10;
  length_type const N = 12;
  length_type const P = 14;

  Index_value val1(1, 0, M, N, P);

  Tensor<T> ten(M, N, P);

  Tensor<T>::whole_domain_type whole = Tensor<T>::whole_domain;

  for (index_type m=0; m<M; ++m)
    for (index_type n=0; n<N; ++n)
      for (index_type p=0; p<P; ++p)
	ten(m, n, p) = val1(m,n,p);

  test_assert(ten(whole, whole, 0).size() == M*N);
  test_assert(ten(whole, 0, whole).size() == M*P);
  test_assert(ten(0, whole, whole).size() == N*P);

  test_assert(ten(whole, whole, 0).size(0) == M);
  test_assert(ten(whole, whole, 0).size(1) == N);

  test_assert(ten(whole, 0, whole).size(0) == M);
  test_assert(ten(whole, 0, whole).size(1) == P);

  test_assert(ten(0, whole, whole).size(0) == N);
  test_assert(ten(0, whole, whole).size(1) == P);
}
  


int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  test_tensor_vector();
  test_tensor_matrix();
}
