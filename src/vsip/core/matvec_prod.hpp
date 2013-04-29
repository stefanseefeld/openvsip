//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

/// Description
///   Matrix-Vector product operations

#ifndef VSIP_CORE_MATVEC_PROD_HPP
#define VSIP_CORE_MATVEC_PROD_HPP

#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/matvec.hpp>
#if VSIP_IMPL_CVSIP_FFT
# include <vsip/core/cvsip/matvec.hpp>
#endif
#ifndef VSIP_IMPL_REF_IMPL
# ifdef VSIP_IMPL_CBE_SDK
#  include <vsip/opt/cbe/cml/matvec.hpp>
# endif
# ifdef VSIP_IMPL_HAVE_BLAS
#  include <vsip/opt/lapack/matvec.hpp>
# endif
# ifdef VSIP_IMPL_HAVE_SAL
#  include <vsip/opt/sal/eval_misc.hpp>
# endif
#endif
#include <iostream>

namespace vsip_csl
{
namespace dispatcher
{

#ifndef VSIP_IMPL_REF_IMPL
template<>
struct List<op::prod>
{
  typedef Make_type_list<be::user,
			 be::cml,
			 be::cuda,
			 be::blas,
			 be::mercury_sal, 
			 be::cvsip,
			 be::generic>::type type;
};

template<>
struct List<op::prodj>
{
  typedef Make_type_list<be::user,
			 be::cml,
                         be::cuda,
			 be::blas,
			 be::mercury_sal, 
			 be::cvsip,
			 be::generic>::type type;
};
#endif

/// Generic evaluator for matrix-matrix products.
template <typename Block0,
	  typename Block1,
	  typename Block2>
struct Evaluator<op::prod, be::generic, 
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 2 && Block2::dim == 2>::type>
{
  static bool const ct_valid = true;
  static bool rt_valid(Block0&, Block1 const&, Block2 const&)
  { return true; }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    typedef typename Block0::value_type RT;

    for (index_type i=0; i<r.size(2, 0); ++i)
      for (index_type j=0; j<r.size(2, 1); ++j)
      {
	RT sum = RT();
	for (index_type k=0; k<a.size(2, 1); ++k)
	{
	  sum += a.get(i, k) * b.get(k, j);
	}
	r.put(i, j, sum);
    }
  }
};

/// Generic evaluator for matrix-matrix conjugate products.
template <typename Block0,
	  typename Block1,
	  typename Block2>
struct Evaluator<op::prodj, be::generic,
                 void(Block0&, Block1 const&, Block2 const&)>
{
  static bool const ct_valid = true;
  static bool rt_valid(Block0&, Block1 const&, Block2 const&)
  { return true; }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    typedef typename Block0::value_type RT;

    for (index_type i=0; i<r.size(2, 0); ++i)
      for (index_type j=0; j<r.size(2, 1); ++j)
      {
	RT sum = RT();
	for (index_type k=0; k<a.size(2, 1); ++k)
	{
	  sum += a.get(i, k) * conj(b.get(k, j));
	}
	r.put(i, j, sum);
    }
  }
};

/// Generic evaluator for matrix-vector products.
template <typename Block0,
	  typename Block1,
	  typename Block2>
struct Evaluator<op::prod, be::generic,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 2 && Block2::dim == 1>::type>
{
  static bool const ct_valid = true;
  static bool rt_valid(Block0&, Block1 const&, Block2 const&)
  { return true; }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    typedef typename Block0::value_type RT;

    for (index_type i=0; i<r.size(1, 0); ++i)
    {
      RT sum = RT();
      for (index_type k=0; k<a.size(2, 1); ++k)
      {
        sum += a.get(i, k) * b.get(k);
      }
      r.put(i, sum);
    }
  }
};

/// Generic evaluator for vector-matrix products.
template <typename Block0,
	  typename Block1,
	  typename Block2>
struct Evaluator<op::prod, be::generic,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 1 && Block2::dim == 2>::type>
{
  static bool const ct_valid = true;
  static bool rt_valid(Block0&, Block1 const&, Block2 const&)
  { return true; }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    typedef typename Block0::value_type RT;

    for (index_type i=0; i<r.size(); ++i)
    {
      RT sum = RT();
      for (index_type k=0; k<b.size(2, 0); ++k)
      {
        sum += a.get(k) * b.get(k, i);
      }
      r.put(i, sum);
    }
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip
{
namespace impl
{

/// Generic matrix-matrix product.
template <typename Block0, typename Block1, typename Block2>
void
generic_prod(Block0 const &a, Block1 const &b, Block2 &r)
{
  using namespace vsip_csl::dispatcher;
#ifdef VSIP_IMPL_REF_IMPL
  Evaluator<op::prod, be::cvsip,
    void(Block2&, Block0 const&, Block1 const&)>::exec(r, a, b);
#else
  vsip_csl::dispatch<op::prod, void,
    Block2&, Block0 const&, Block1 const&>(r, a, b);
#endif
}

/// Generic matrix-matrix conjugate product.
template <typename Block0, typename Block1, typename Block2>
void
generic_prodj(Block0 const &a, Block1 const &b, Block2 &r)
{
  using namespace vsip_csl::dispatcher;
#ifdef VSIP_IMPL_REF_IMPL
  Evaluator<op::prodj, be::cvsip,
    void(Block2&, Block0 const&, Block1 const&)>::exec(r, a, b);
#else
  vsip_csl::dispatch<vsip_csl::dispatcher::op::prodj, void,
    Block2&, Block0 const&, Block1 const&>(r, a, b);
#endif
}

template <typename A1, typename A2>
class Product_base
{
  typedef typename View_block_storage<A1>::expr_type arg1_storage;
  typedef typename View_block_storage<A2>::expr_type arg2_storage;
public:
  static dimension_type const dim = (A1::dim == 2 && A2::dim == 2) ? 2 : 1;
  typedef Local_map map_type;
  typedef typename Promotion<typename A1::value_type, 
			     typename A2::value_type>::type result_type;

  Product_base(A1 const &a1, A2 const &a2) : arg1_(a1), arg2_(a2) {}
  A1 const &arg1() const { return arg1_;}
  A2 const &arg2() const { return arg2_;}
  vsip::length_type size() const 
  {
    if (A1::dim == 2 && A2::dim == 2) // matrix-matrix
      return arg1_.size(2, 0) * arg2_.size(2, 1);
    else if (A1::dim == 1) // vector-matrix
      return arg2_.size(2, 1);
    else // matrix-vector
      return arg1_.size(2, 0);
  }
  vsip::length_type size(vsip::dimension_type block_dim,
			 vsip::dimension_type d) const
  { return d == 0 ? arg1_.size(block_dim, 0) : arg2_.size(block_dim, 1);}
  map_type const &map() const { return map_;}

private:
  Local_map map_;
  arg1_storage arg1_;
  arg2_storage arg2_;
};

template <typename A1, typename A2>
class Product : public Product_base<A1, A2>
{
public:
  Product(A1 const &a1, A2 const &a2) : Product_base<A1, A2>(a1, a2) {}

  template <typename B>
  void apply(B &block) const { generic_prod(this->arg1(), this->arg2(), block);}
};

template <typename A1, typename A2>
class Productj : public Product_base<A1, A2>
{
public:
  Productj(A1 const &a1, A2 const &a2) : Product_base<A1, A2>(a1, a2) {}

  template <typename B>
  void apply(B &block) const { generic_prodj(this->arg1(), this->arg2(), block);}
};

} // namespace vsip::impl

/// Matrix-matrix product dispatch.
template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
const_Matrix<typename Promotion<T0, T1>::type,
	     vsip_csl::expr::Binary<impl::Product, Block0, Block1> const>
prod(const_Matrix<T0, Block0> a, const_Matrix<T1, Block1> b)
{
  typedef typename Promotion<T0, T1>::type value_type;
  typedef vsip_csl::expr::Binary<impl::Product, Block0, Block1> block_type;

  assert(a.size(1) == b.size(0));

  impl::Product<Block0, Block1> operation(a.block(), b.block());
  return const_Matrix<value_type, block_type const>(operation);
}

/// Matrix-vector product dispatch.
template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
const_Vector<typename Promotion<T0, T1>::type,
	     vsip_csl::expr::Binary<impl::Product, Block0, Block1> const>
prod(const_Matrix<T0, Block0> a, const_Vector<T1, Block1> b)
{
  typedef typename Promotion<T0, T1>::type value_type;
  typedef vsip_csl::expr::Binary<impl::Product, Block0, Block1> block_type;

  assert(a.size(1) == b.size());

  impl::Product<Block0, Block1> operation(a.block(), b.block());
  return const_Vector<value_type, block_type const>(operation);
}

/// Vector-Matrix product dispatch.
template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
const_Vector<typename Promotion<T0, T1>::type,
	     vsip_csl::expr::Binary<impl::Product, Block0, Block1> const>
prod(const_Vector<T0, Block0> a, const_Matrix<T1, Block1> b)
{
  typedef typename Promotion<T0, T1>::type value_type;
  typedef vsip_csl::expr::Binary<impl::Product, Block0, Block1> block_type;

  assert(a.size() == b.size(0));

  impl::Product<Block0, Block1> operation(a.block(), b.block());
  return const_Vector<value_type, block_type const>(operation);
}


/// [3x3] Matrix-matrix product dispatch.
template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
const_Matrix<typename Promotion<T0, T1>::type,
	     vsip_csl::expr::Binary<impl::Product, Block0, Block1> const>
prod3(const_Matrix<T0, Block0> a, const_Matrix<T1, Block1> b)
{
  assert(a.size(0) == 3);
  assert(a.size(1) == 3);
  typedef typename Promotion<T0, T1>::type value_type;
  typedef vsip_csl::expr::Binary<impl::Product, Block0, Block1> block_type;
  impl::Product<Block0, Block1> operation(a.block(), b.block());
  return const_Matrix<value_type, block_type const>(operation);
}


/// [3x3] Matrix-vector product dispatch.
template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
const_Vector<typename Promotion<T0, T1>::type,
	     vsip_csl::expr::Binary<impl::Product, Block0, Block1> const>
prod3(const_Matrix<T0, Block0> a, const_Vector<T1, Block1> b)
{
  assert(a.size(0) == 3);
  assert(a.size(1) == 3);
  typedef typename Promotion<T0, T1>::type value_type;
  typedef vsip_csl::expr::Binary<impl::Product, Block0, Block1> block_type;
  impl::Product<Block0, Block1> operation(a.block(), b.block());
  return const_Vector<value_type, block_type const>(operation);
}


/// [4x4] Matrix-matrix product dispatch.
template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
const_Matrix<typename Promotion<T0, T1>::type,
	     vsip_csl::expr::Binary<impl::Product, Block0, Block1> const>
prod4(const_Matrix<T0, Block0> a, const_Matrix<T1, Block1> b)
{
  assert(a.size(0) == 4);
  assert(a.size(1) == 4);
  typedef typename Promotion<T0, T1>::type value_type;
  typedef vsip_csl::expr::Binary<impl::Product, Block0, Block1> block_type;
  impl::Product<Block0, Block1> operation(a.block(), b.block());
  return const_Matrix<value_type, block_type const>(operation);
}


/// [4x4] Matrix-vector product dispatch.
template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
const_Vector<typename Promotion<T0, T1>::type,
	     vsip_csl::expr::Binary<impl::Product, Block0, Block1> const>
prod4(const_Matrix<T0, Block0> a, const_Vector<T1, Block1> b)
{
  assert(a.size(0) == 4);
  assert(a.size(1) == 4);
  typedef typename Promotion<T0, T1>::type value_type;
  typedef vsip_csl::expr::Binary<impl::Product, Block0, Block1> block_type;
  impl::Product<Block0, Block1> operation(a.block(), b.block());
  return const_Vector<value_type, block_type const>(operation);
}


/// Matrix-Matrix (with hermitian) product dispatch.
template <typename T0,
          typename T1,
          typename Block0,
          typename Block1>
const_Matrix<typename Promotion<complex<T0>, complex<T1> >::type,
  vsip_csl::expr::Binary<impl::Productj, Block0, impl::Transposed_block<Block1> > const>
prodh(const_Matrix<complex<T0>, Block0> a,
      const_Matrix<complex<T1>, Block1> b) 
    VSIP_NOTHROW
{
  typedef typename Promotion<complex<T0>, complex<T1> >::type value_type;
  typedef impl::Transposed_block<Block1> tblock1_type;
  typedef vsip_csl::expr::Binary<impl::Productj, Block0, tblock1_type> block_type;
  impl::Productj<Block0, tblock1_type> operation(a.block(), b.transpose().block());
  return const_Matrix<value_type, block_type const>(operation);
}


/// Matrix-Matrix (with complex conjugate) product dispatch.
template <typename T0,
          typename T1,
          typename Block0,
          typename Block1>
const_Matrix<typename Promotion<complex<T0>, complex<T1> >::type,
  vsip_csl::expr::Binary<impl::Productj, Block0, Block1> const>
prodj(const_Matrix<complex<T0>, Block0> a,
      const_Matrix<complex<T1>, Block1> b)
    VSIP_NOTHROW
{
  typedef typename Promotion<complex<T0>, complex<T1> >::type value_type;
  typedef vsip_csl::expr::Binary<impl::Productj, Block0, Block1> block_type;
  impl::Productj<Block0, Block1> operation(a.block(), b.block());
  return const_Matrix<value_type, block_type const>(operation);
}


/// Matrix-Matrix (with transpose) product dispatch.
template <typename T0,
          typename T1,
          typename Block0,
          typename Block1>
const_Matrix<typename Promotion<T0, T1>::type,
  vsip_csl::expr::Binary<impl::Product, Block0, impl::Transposed_block<Block1> > const>
prodt(const_Matrix<T0, Block0> a, const_Matrix<T1, Block1> b) VSIP_NOTHROW
{
  typedef typename Promotion<T0, T1>::type value_type;
  typedef impl::Transposed_block<Block1> tblock1_type;
  typedef vsip_csl::expr::Binary<impl::Product, Block0, tblock1_type> block_type;
  impl::Product<Block0, tblock1_type> operation(a.block(), b.transpose().block());
  return const_Matrix<value_type, block_type const>(operation);
}

} // namespace vsip

#endif // VSIP_IMPL_MATVEC_PROD_HPP
