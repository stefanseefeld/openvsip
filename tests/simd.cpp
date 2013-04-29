//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#define VERBOSE 0

#if VERBOSE
#  include <iostream>
#endif
#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/opt/simd/simd.hpp>
#include <vsip/core/metaprogramming.hpp>

#include <vsip_csl/test.hpp>

using namespace vsip;
using namespace vsip_csl;

/***********************************************************************
  Definitions
***********************************************************************/

template <typename T>
void
test_zero()
{
  using namespace vsip::impl;
  typedef simd::Simd_traits<T> traits;

  typedef typename traits::value_type value_type;
  typedef typename traits::simd_type  simd_type;

  length_type const vec_size = traits::vec_size;

  union
  {
    simd_type  vec;
    value_type val[vec_size];
  } u;

  u.vec = traits::zero();

  for (index_type i=0; i<vec_size; ++i)
    test_assert(u.val[i] == value_type());
}



template <typename T>
void
test_load_scalar()
{
  typedef vsip::impl::simd::Simd_traits<T> traits;

  typedef typename traits::value_type value_type;
  typedef typename traits::simd_type  simd_type;

  length_type const vec_size = traits::vec_size;

  union
  {
    simd_type  vec;
    value_type val[vec_size];
  } u;

  // Zero out the vector.
  u.vec = traits::zero();

  // Confirm it has been zero'd.
  for (index_type i=0; i<vec_size; ++i)
    test_assert(u.val[i] == value_type());

  // Load a scalar.
  u.vec = traits::load_scalar(value_type(1));

#if VERBOSE
  std::cout << "load_test_scalar<" << typeid(T).name() << ">\n";
  for (index_type i=0; i<vec_size; ++i)
    std::cout << "u.val[" << i << "]: " << u.val[i] << std::endl;
#endif

  // Check that value is loaded into the 'scalar position'.
  test_assert(u.val[traits::scalar_pos] == value_type(1));
  for (index_type i=0; i<vec_size; ++i)
    if (i != traits::scalar_pos)
      test_assert(u.val[i] == value_type());
}



template <typename T>
void
test_load_scalar_all()
{
  typedef vsip::impl::simd::Simd_traits<T> traits;

  typedef typename traits::value_type value_type;
  typedef typename traits::simd_type  simd_type;

  length_type const vec_size = traits::vec_size;

  union
  {
    simd_type  vec;
    value_type val[vec_size];
  } u;

  u.vec = traits::load_scalar_all(value_type(1));

  for (index_type i=0; i<vec_size; ++i)
    test_assert(u.val[i] == value_type(1));
}



template <typename T>
void
test_load_unaligned()
{
  typedef vsip::impl::simd::Simd_traits<T> traits;

  typedef typename traits::value_type value_type;
  typedef typename traits::simd_type  simd_type;

  length_type const vec_size = traits::vec_size;

  union
  {
    simd_type  vec;
    value_type val[vec_size];
  } u;

  value_type val[8*vec_size];

  for (index_type i=0; i<8*vec_size; ++i)
    val[i] = value_type(i);

  for (index_type i=0; i<1*vec_size; ++i)
  {
    u.vec = traits::load_unaligned(val + i);

#if VERBOSE
    std::cout << "unaligned load offset: " << i << std::endl;
    for (index_type j=0; j<vec_size; ++j)
      std::cout  << "  - " << j << ": "
		 << u.val[j] << "  " << value_type(i+j)
		 << std::endl;
#endif

    for (index_type j=0; j<vec_size; ++j)
      test_assert(u.val[j] == value_type(i+j));
  }
}



template <typename T>
void
test_interleaved(true_type)
{
  typedef vsip::impl::simd::Simd_traits<T> traits;

  typedef typename traits::value_type value_type;
  typedef typename traits::simd_type  simd_type;

  length_type const vec_size = traits::vec_size;

  union
  {
    simd_type  vec;
    value_type val[vec_size];
  } ri1, ri2, rr, ii, new_ri1, new_ri2;

  for (index_type i=0; i<vec_size/2; ++i)
  {
    ri1.val[2*i]   = (2*i);
    ri1.val[2*i+1] = (3*i+1);
    ri2.val[2*i]   = (2*(i+vec_size/2));
    ri2.val[2*i+1] = (3*(i+vec_size/2)+1);
  }

  rr.vec = traits::real_from_interleaved(ri1.vec, ri2.vec);
  ii.vec = traits::imag_from_interleaved(ri1.vec, ri2.vec);
  new_ri1.vec = traits::interleaved_lo_from_split(rr.vec, ii.vec);
  new_ri2.vec = traits::interleaved_hi_from_split(rr.vec, ii.vec);

#if VERBOSE
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "test_interleaved: " << std::endl;
  std::cout << "  vec_size: " << vec_size << std::endl;
  std::cout << "  accelerated: " << (traits::is_accel ? "yes" : "no") << std::endl;
  for (index_type i=0; i<vec_size; ++i)
    std::cout << "ri1.val[" << i << "]: " << ri1.val[i] << std::endl;
  for (index_type i=0; i<vec_size; ++i)
    std::cout << "ri2.val[" << i << "]: " << ri2.val[i] << std::endl;
  for (index_type i=0; i<vec_size; ++i)
    std::cout << "rr.val[" << i << "]: " << rr.val[i] << std::endl;
  for (index_type i=0; i<vec_size; ++i)
    std::cout << "ii.val[" << i << "]: " << ii.val[i] << std::endl;
  for (index_type i=0; i<vec_size; ++i)
    std::cout << "new_ri1.val[" << i << "]: " << new_ri1.val[i] << std::endl;
  for (index_type i=0; i<vec_size; ++i)
    std::cout << "new_ri2.val[" << i << "]: " << new_ri2.val[i] << std::endl;
#endif

  for (index_type i=0; i<vec_size; ++i)
  {
    test_assert(rr.val[i] == value_type(2*i));
    test_assert(ii.val[i] == value_type(3*i)+1);
  }

  for (index_type i=0; i<vec_size/2; ++i)
  {
    test_assert(new_ri1.val[2*i]   == (2*i));
    test_assert(new_ri1.val[2*i+1] == (3*i+1));
    test_assert(new_ri2.val[2*i]   == (2*(i+vec_size/2)));
    test_assert(new_ri2.val[2*i+1] == (3*(i+vec_size/2)+1));
  }
}



template <typename T>
void
test_interleaved(false_type)
{
}



template <typename T>
void
test_complex(true_type)
{
  typedef vsip::impl::simd::Simd_traits<T> traits;

  typedef typename traits::value_type value_type;
  typedef typename traits::simd_type  simd_type;

  length_type const vec_size = traits::vec_size;

  union
  {
    simd_type           vec;
    value_type          val[vec_size];
  } ri1, ri2, rr, ii, new_ri1, new_ri2;

  complex<value_type> cval1[vec_size/2];
  complex<value_type> cval2[vec_size/2];

  for (index_type i=0; i<vec_size/2; ++i)
  {
    cval1[i] = complex<value_type>(2*i, 3*i+1);
    cval2[i] = complex<value_type>(2*(i+vec_size/2), 3*(i+vec_size/2)+1);
  }

  ri1.vec = traits::load((value_type*)cval1);
  ri2.vec = traits::load((value_type*)cval2);

  for (index_type i=0; i<vec_size/2; ++i)
  {
    test_assert(ri1.val[2*i]   == (2*i));
    test_assert(ri1.val[2*i+1] == (3*i+1));
    test_assert(ri2.val[2*i]   == (2*(i+vec_size/2)));
    test_assert(ri2.val[2*i+1] == (3*(i+vec_size/2)+1));
  }

  rr.vec = traits::real_from_interleaved(ri1.vec, ri2.vec);
  ii.vec = traits::imag_from_interleaved(ri1.vec, ri2.vec);
  new_ri1.vec = traits::interleaved_lo_from_split(rr.vec, ii.vec);
  new_ri2.vec = traits::interleaved_hi_from_split(rr.vec, ii.vec);

#if VERBOSE
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "test_complex: " << std::endl;
  std::cout << "  vec_size: " << vec_size << std::endl;
  std::cout << "  accelerated: " << (traits::is_accel ? "yes" : "no") << std::endl;
  for (index_type i=0; i<vec_size; ++i)
    std::cout << "ri1.val[" << i << "]: " << ri1.val[i] << std::endl;
  for (index_type i=0; i<vec_size; ++i)
    std::cout << "ri2.val[" << i << "]: " << ri2.val[i] << std::endl;
  for (index_type i=0; i<vec_size; ++i)
    std::cout << "rr.val[" << i << "]: " << rr.val[i] << std::endl;
  for (index_type i=0; i<vec_size; ++i)
    std::cout << "ii.val[" << i << "]: " << ii.val[i] << std::endl;
  for (index_type i=0; i<vec_size; ++i)
    std::cout << "new_ri1.val[" << i << "]: " << new_ri1.val[i] << std::endl;
  for (index_type i=0; i<vec_size; ++i)
    std::cout << "new_ri2.val[" << i << "]: " << new_ri2.val[i] << std::endl;
#endif

  for (index_type i=0; i<vec_size; ++i)
  {
    test_assert(rr.val[i] == value_type(2*i));
    test_assert(ii.val[i] == value_type(3*i)+1);
  }

  for (index_type i=0; i<vec_size/2; ++i)
  {
    test_assert(new_ri1.val[2*i]   == (2*i));
    test_assert(new_ri1.val[2*i+1] == (3*i+1));
    test_assert(new_ri2.val[2*i]   == (2*(i+vec_size/2)));
    test_assert(new_ri2.val[2*i+1] == (3*(i+vec_size/2)+1));
  }
}



template <typename T>
void
test_complex(false_type)
{
}



template <typename T>
void
test_add()
{
  typedef vsip::impl::simd::Simd_traits<T> traits;

  typedef typename traits::value_type value_type;
  typedef typename traits::simd_type  simd_type;

  length_type const vec_size = traits::vec_size;

  union
  {
    simd_type  vec;
    value_type val[vec_size];
  } u, a, b;

  for (index_type i=0; i<vec_size; ++i)
  {
    a.val[i] = T(2*i);
    b.val[i] = T(3*i);
  }

  u.vec = traits::add(a.vec, b.vec);

  for (index_type i=1; i<vec_size; ++i)
    test_assert(u.val[i] == a.val[i] + b.val[i]);
}



template <typename T>
void
test_mul()
{
  typedef vsip::impl::simd::Simd_traits<T> traits;

  typedef typename traits::value_type value_type;
  typedef typename traits::simd_type  simd_type;

  length_type const vec_size = traits::vec_size;

  union
  {
    simd_type  vec;
    value_type val[vec_size];
  } u, a, b;

  for (index_type i=0; i<vec_size; ++i)
  {
    a.val[i] = T(2*i);
    b.val[i] = T(3*i);
  }

  u.vec = traits::mul(a.vec, b.vec);

  for (index_type i=1; i<vec_size; ++i)
    test_assert(u.val[i] == a.val[i] * b.val[i]);
}



template <typename T>
void
test_all()
{
  typedef vsip::impl::simd::Simd_traits<T> traits;

  bool const do_complex = traits::is_accel && 
    (is_same<T, float>::value || is_same<T, double>::value);

  test_zero<T>();
  test_load_scalar<T>();
  test_load_scalar_all<T>();
  test_load_unaligned<T>();
  test_add<T>();

  test_interleaved<T>(integral_constant<bool, do_complex>());
  test_complex<T>(integral_constant<bool, do_complex>());
}



template <typename T>
void
test_arith()
{
  // typedef vsip::impl::simd::Simd_traits<T> traits;
  // test_assert(traits::is_accel);
  test_add<T>();
  test_mul<T>();
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_all<signed char>();
  test_all<short>();
  test_all<int>();
  test_all<float>();
  test_all<double>();

  // test_all<complex<float> >();

  test_arith<float>();
  test_arith<double>();
}
