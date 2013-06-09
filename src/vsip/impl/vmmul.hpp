//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_vmmul_hpp_
#define vsip_impl_vmmul_hpp_

#include <ovxx/block_traits.hpp>
#include <ovxx/expr.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/impl/promotion.hpp>

namespace vsip
{
namespace impl
{

/// Traits class to determines return type for vmmul.
template <dimension_type Dim,
	  typename       T0,
	  typename       T1,
	  typename       Block0,
	  typename       Block1>
struct vmmul_traits
{
  typedef typename vsip::Promotion<T0, T1>::type value_type;
  typedef ovxx::expr::Vmmul<Dim, Block0, Block1> const block_type;
  typedef Matrix<value_type, block_type> view_type;
};

} // namespace vsip::impl
} // namespace vsip

namespace ovxx
{
namespace dispatcher
{
/// Evaluator for vector-matrix multiply.
///
/// Reduces vmmul into either vector element-wise multiply, or
/// scalar-vector multiply, depending on the dimension-order and
/// requested orientation.  These reduced cases are then
/// re-dispatched, allowing them to be handled by a vendor library,
template <typename LHS, typename VBlock, typename MBlock, dimension_type D>
struct Evaluator<op::assign<2>, be::op_expr,
		 void(LHS &, expr::Vmmul<D, VBlock, MBlock> const &)>
{
  static char const* name() { return "Expr_Loop_Vmmul"; }

  typedef expr::Vmmul<D, VBlock, MBlock> RHS;

  typedef typename LHS::value_type lhs_value_type;
  typedef typename VBlock::value_type v_value_type;
  typedef typename MBlock::value_type m_value_type;

  static bool const ct_valid = 
    !is_expr_block<VBlock>::value &&
    !is_expr_block<MBlock>::value;

  static bool rt_valid(LHS &, RHS const &) { return true;}
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    VBlock const& vblock = rhs.get_vblk();
    MBlock const& mblock = rhs.get_mblk();

    typedef typename get_block_layout<LHS>::order_type order_type;

    Matrix<lhs_value_type, LHS> m_dst(lhs);
    const_Vector<v_value_type, VBlock> v(const_cast<VBlock&>(vblock));
    const_Matrix<m_value_type, MBlock> m(const_cast<MBlock&>(mblock));

    if (is_same<order_type, row2_type>::value)
    {
      if (D == row)
      {
	for (index_type r = 0; r < lhs.size(2, 0); ++r)
	  m_dst.row(r) = v * m.row(r);
      }
      else
      {
	for (index_type r = 0; r < lhs.size(2, 0); ++r)
	  m_dst.row(r) = v.get(r) * m.row(r);
      }
    }
    else // col2_type
    {
      if (D == row)
      {
	for (index_type c = 0; c < lhs.size(2, 1); ++c)
	  m_dst.col(c) = v.get(c) * m.col(c);
      }
      else
      {
	for (index_type c = 0; c < lhs.size(2, 1); ++c)
	  m_dst.col(c) = v * m.col(c);
      }
    }
  }
};

} // namespace ovxx::dispatcher
} // namespace ovxx

namespace vsip
{

/// Vector-matrix element-wise multiplication
template <dimension_type D,
	  typename       T0,
	  typename       T1,
	  typename       Block0,
	  typename       Block1>
typename impl::vmmul_traits<D, T0, T1, Block0, Block1>::view_type
vmmul(const_Vector<T0, Block0> v, const_Matrix<T1, Block1> m) VSIP_NOTHROW
{
  typedef impl::vmmul_traits<D, T0, T1, Block0, Block1> traits;
  typedef typename traits::block_type block_type;
  typedef typename traits::view_type  view_type;

  return view_type(block_type(v.block(), m.block()));
}

} // namespace vsip

#endif
