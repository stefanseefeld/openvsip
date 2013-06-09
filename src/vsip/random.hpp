//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

// The core portion of the random number generator is taken 
// from TASP VSIPL.  See http://www.vsipl.org/software/ for
// copyright notice and additional information.

#ifndef vsip_random_hpp_
#define vsip_random_hpp_

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <ovxx/expr/generator.hpp>
#include <vsip/map.hpp>

namespace vsip
{

namespace impl
{

/// Base class for random number generation
template <typename T>
class Rand_base
{
  typedef unsigned int uint_type;

public:
  Rand_base(index_type seed, index_type numprocs, index_type id, 
    bool portable = true) VSIP_THROW((std::bad_alloc));
  ~Rand_base() VSIP_NOTHROW
    {}

  T randn() VSIP_NOTHROW;
  T randu() VSIP_NOTHROW;

protected:
  std::complex<T> randn_complex() VSIP_NOTHROW;

private:
  uint_type a_;     // multiplier in LCG
  uint_type c_;     // adder in LCG
  uint_type a1_;
  uint_type c1_;
  uint_type X_;     // Last or initial X
  uint_type X1_;
  uint_type X2_;
  
  bool portable_;
};


/// Base class implementation
template <typename T>
Rand_base<T>::Rand_base(index_type seed, index_type numprocs, index_type id, 
  bool portable) VSIP_THROW((std::bad_alloc))
{
  assert( id > 0 );

  const uint_type A0 = 1664525;
  const uint_type C0 = 1013904223;
  const uint_type A1 = 69069;

  uint_type x0 = (uint_type) seed;
  uint_type k  = (uint_type) numprocs;
  portable_ = portable;

  if ( !portable )
  {
    // create non-portable generator
    uint_type i;
    for ( i = 0; i < id; i++ )
      x0 = A0 * x0 + C0;

    X_ = x0;                    // find the seed to start out for id
    uint_type n = 0;
    uint_type k0 = k;
    while ( (k0 % 2) == 0 )
    {
      k0 = k0 / 2;
      n++;
    }

    i = k - 1;
    uint_type a = A0;
    while ( i-- > 0 )
    {
      a *= A0;                  // find a for numseqs
    }

    uint_type c = 1;            // find c for numseqs
    uint_type t = 1;
    for ( i = 0; i < k0; i++ )
    {
      t *= A0;
    }

    while ( n-- > 0 )
    {
      c *= (t + 1);
      t *= t;
    }

    t = 1;
    n = A0;
    for ( i = 1; i < k0; i++ )
    {
      t += n;
      n *= A0;
    }

    c *= (t * C0);
    a_ = a;
    c_ = c;
    // unused in non-portable case 
    a1_ = 0;    
    c1_ = 0;    
    X1_ = 0;
    X2_ = 0;
  } 
  else 
  {
    // create portable generator
    assert( id <= 100 );
    const uint_type c[]=             // 100 prime numbers
      {    3,   5,   7,  11,  13,  17,  19,  23,  29,  31,
          37,  41,  43,  47,  53,  59,  61,  67,  71,  73,
          79,  83,  89,  97, 101, 103, 107, 109, 113, 127,
         131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
         181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
         239, 241, 251, 257, 263, 269, 271, 277, 281, 283,
         293, 307, 311, 313, 317, 331, 337, 347, 349, 353,
         359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
         421, 431, 433, 439, 443, 449, 457, 461, 463, 467,
         479, 487, 491, 499, 503, 509, 521, 523, 541, 547  };
    const uint_type c1 = c[id-1];

    if ( id > 1 )
    { 
      uint_type a0 = A0;
      uint_type c0 = C0;
      uint_type mask = 1;
      uint_type big      = 4294967295ul;
      uint_type skip = (big / k) * (id - 1);
      int i; 

      for ( i = 0; i < 32; i++ )
      {
        if ( mask & skip )
        {
          x0 = a0 * x0 + c0;
        }
        c0 = (a0 + 1) * c0;
        a0 = a0 * a0;
        mask <<= 1;
      } 
    }

    a_  = A0;
    c_  = C0;
    a1_ = A1; 
    c1_ = c1;
    X_  = x0;
    X1_ = 1;
    X2_ = 1;
  }
}

template <typename T>
inline T 
Rand_base<T>::randn() VSIP_NOTHROW
{
  index_type i;
  T rp = T();

  if ( !portable_ )
  {
    // non-portable generator

    for ( i = 0; i < 12; i++ )
    {
      X_ = a_ * X_ + c_;
      rp  += T(X_ / 4294967296.0);
    }
    rp -= 6.0;
  } 
  else
  {
    // portable generator
    uint_type itemp;

    for ( i = 0; i < 12; i++ )
    {
      X_ = X_ * a_ + c_;
      X1_ = X1_ * a1_ + c1_;
      itemp = X_ - X1_;
      if ( X1_ == X2_ )
      {
        X1_++;
        X2_++;
      }
      rp  += T(itemp / 4294967296.0);
    }
    rp = 6.0 - rp;
  }
  return rp;
}

template <typename T>
inline T 
Rand_base<T>::randu() VSIP_NOTHROW
{
  if ( !portable_ )
  {
    // non-portable generator

    X_  = a_ * X_ + c_;
    return ( T(X_ / 4294967296.0) );
  } 
  else
  {
    // portable generator
    uint_type itemp;

    X_  = X_ * a_ + c_;
    X1_ = X1_ * a1_ + c1_;
    itemp = X_ - X1_;
    if ( X1_ == X2_ )
    {
      X1_++;
      X2_++;
    }
    return( T(itemp / 4294967296.0) );
  }
}

template <typename T>
inline std::complex<T> 
Rand_base<T>::randn_complex() VSIP_NOTHROW
{
  T real = T();
  T imag = T();
  uint_type i;

  if ( !portable_ )
  { 
    // non-portable generator

    for ( i = 0; i < 3; i++ )
    {
      X_ = a_ * X_ + c_;
      real += T(X_ / 4294967296.0);
    }

    T t2 = T();
    for ( i = 0; i < 3; i++ )
    {
      X_ = a_ * X_ + c_;
      t2 += T(X_ / 4294967296.0);
    }

    imag = real - t2;
    real = 3 - t2 - real;
  } 
  else
  {
    // portable generator
    uint_type itemp;

    for ( i = 0; i < 3; i++ )
    {
      X_ =      X_ * a_ + c_;
      X1_ = X1_ * a1_ + c1_;
      itemp = X_ - X1_;
      if ( X1_ == X2_ )
      {
        X1_++;
        X2_++;
      }
      real += T(itemp / 4294967296.0);
    }
        
    T t2 = T();
    for ( i = 0; i < 3; i++ )
    {
      X_ =      X_ * a_ + c_;
      X1_ = X1_ * a1_ + c1_;
      itemp = X_ - X1_;
      if ( X1_ == X2_ )
      {
        X1_++;
        X2_++;
      }
      t2 += T(itemp / 4294967296.0);
    }

    imag = real - t2;
    real = 3 - t2 - real;
  }
  return std::complex<T>(real, imag);
}


/// specialization for complex types
template <typename T>
class Rand_base<std::complex<T> > : public Rand_base<T>
{
  typedef Rand_base<T> base_type;

public:
  Rand_base(index_type seed, index_type numprocs, index_type id, 
    bool portable = true) VSIP_THROW((std::bad_alloc))
    : base_type(seed, numprocs, id, portable)
    {}
  ~Rand_base() VSIP_NOTHROW
    {}

  std::complex<T> randn() VSIP_NOTHROW
    {
      return base_type::randn_complex();
    }

  std::complex<T> randu()
    {
      T re = base_type::randu();
      T im = base_type::randu();
      return std::complex<T>(re, im);
    }
};

} // namespace impl



/// Rand class definition
template <typename T = VSIP_DEFAULT_VALUE_TYPE>
class Rand : public impl::Rand_base<T>
{
  typedef impl::Rand_base<T> base_type;

  struct Uniform_generator
  {
    typedef T result_type;
    Uniform_generator(Rand &r) : rng(r) {}
    T operator()(index_type) const { return rng.randu();}
    T operator()(index_type, index_type) const { return rng.randu();}
    T operator()(index_type, index_type, index_type) const { return rng.randu();}
    Rand<T> &rng;
  };

  struct Normal_generator
  {
    typedef T result_type;
    Normal_generator(Rand &r) : rng(r) {}
    T operator()(index_type) const { return rng.randn();}
    T operator()(index_type, index_type) const { return rng.randn();}
    T operator()(index_type, index_type, index_type) const { return rng.randn();}
    Rand<T> &rng;
  };

  typedef ovxx::expr::Generator<1, Uniform_generator> const uniform1d_block_type;
  typedef ovxx::expr::Generator<1, Normal_generator> const normal1d_block_type;
  typedef ovxx::expr::Generator<2, Uniform_generator> const uniform2d_block_type;
  typedef ovxx::expr::Generator<2, Normal_generator> const normal2d_block_type;
  typedef ovxx::expr::Generator<3, Uniform_generator> const uniform3d_block_type;
  typedef ovxx::expr::Generator<3, Normal_generator> const normal3d_block_type;

  typedef Dense<1, T, row1_type, ovxx::parallel::local_or_global_map<1> > block1_type;
  typedef Dense<2, T, row2_type, ovxx::parallel::local_or_global_map<2> > block2_type;
  typedef Dense<3, T, row3_type, ovxx::parallel::local_or_global_map<3> > block3_type;

public:
  // View types [random.rand.view types]
  typedef Vector<T> vector_type;
  typedef Matrix<T> matrix_type;
  typedef Tensor<T> tensor_type;

  typedef const_Vector<T, uniform1d_block_type> uniform_vector_type;
  typedef const_Vector<T, normal1d_block_type> normal_vector_type;
  typedef const_Matrix<T, uniform2d_block_type> uniform_matrix_type;
  typedef const_Matrix<T, normal2d_block_type> normal_matrix_type;
  typedef const_Tensor<T, uniform3d_block_type> uniform_tensor_type;
  typedef const_Tensor<T, normal3d_block_type> normal_tensor_type;
  
  // Constructors, copy, assignment, and destructor 
  //   [random.rand.constructors]
  Rand(index_type seed, bool portable = true) 
      VSIP_THROW((std::bad_alloc)) 
    : base_type(seed, 1, 1, portable)
    {}
  Rand(index_type seed, index_type numprocs, index_type id, 
    bool portable = true) VSIP_THROW((std::bad_alloc))
    : base_type(seed, numprocs, id, portable)
    {}

private:
  Rand(Rand const&) VSIP_NOTHROW {}
  Rand& operator=(Rand const&) VSIP_NOTHROW {}

public:
  ~Rand() VSIP_NOTHROW {}

  // Number generators [random.rand.generate]
  T randu() VSIP_NOTHROW
  {
    return base_type::randu();
  }
  T randn() VSIP_NOTHROW
  {
    return base_type::randn();
  }

  uniform_vector_type 
  randu(length_type len) VSIP_NOTHROW
  {
    Uniform_generator gen(*this);
    uniform1d_block_type block(ovxx::Length<1>(len), gen);
    return const_Vector<T, uniform1d_block_type>(block);
  }
  uniform_matrix_type 
  randu(length_type rows, length_type columns) VSIP_NOTHROW
  {
    Uniform_generator gen(*this);
    uniform2d_block_type block(ovxx::Length<2>(rows, columns), gen);
    return const_Matrix<T, uniform2d_block_type>(block);
  }

  uniform_tensor_type 
  randu(length_type z, length_type y, length_type x) VSIP_NOTHROW
  {
    Uniform_generator gen(*this);
    uniform3d_block_type block(ovxx::Length<3>(z, y, x), gen);
    return const_Tensor<T, uniform3d_block_type>(block);
  }

  normal_vector_type
  randn(length_type len) VSIP_NOTHROW
  {
    Normal_generator gen(*this);
    normal1d_block_type block(ovxx::Length<1>(len), gen);
    return const_Vector<T, normal1d_block_type>(block);
  }

  normal_matrix_type
  randn(length_type rows, length_type columns) VSIP_NOTHROW
  {
    Normal_generator gen(*this);
    normal2d_block_type block(ovxx::Length<2>(rows, columns), gen);
    return const_Matrix<T, normal2d_block_type>(block);
  }

  normal_tensor_type
  randn(length_type z, length_type y, length_type x) VSIP_NOTHROW
  {
    Normal_generator gen(*this);
    normal3d_block_type block(ovxx::Length<3>(z, y, x), gen);
    return const_Tensor<T, normal3d_block_type>(block);
  }
};


} // namespace vsip

#endif

