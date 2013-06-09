//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_expr_hpp_
#define ovxx_parallel_expr_hpp_

#include <ovxx/strided.hpp>
#include <ovxx/domain_utils.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/view/traits.hpp>
#include <ovxx/parallel/distributed_block.hpp>
#include <ovxx/parallel/map_subset_block.hpp>
#include <ovxx/parallel/assign_fwd.hpp>
#include <ovxx/parallel/subset_map.hpp>
#include <ovxx/expr/transform.hpp>
#include <iostream>

namespace ovxx
{
namespace parallel
{

template <dimension_type D, typename M, typename B,
	  typename I = typename choose_peb<B>::type>
class Expr_block;

template <dimension_type D, typename M>
class create_expr;

template <dimension_type D, typename M, typename B>
typename expr::transform::return_type<create_expr<D, M>, B>::type
get_expr_block(M const &map, B const &block);

template <dimension_type D, typename M, typename B>
class Expr_block<D, M, B, peb_reorg> : ovxx::detail::noncopyable
{
public:
  static dimension_type const dim = D;

  typedef typename B::value_type           value_type;
  typedef typename B::reference_type       reference_type;
  typedef typename B::const_reference_type const_reference_type;
  typedef M                                map_type;

  // The layout of the reorg block should have the same dimension-
  // order and complex format as the source block.  Packing format
  // should either be unit-stride-dense or unit-stride-aligned.
  // It should not be taken directly from BlockT since it may have
  // a non realizable packing format such as packing::unknown.
  typedef typename get_block_layout<B>::order_type order_type;
  static pack_type const packing = dense;
  static storage_format_type const storage_format = get_block_layout<B>::storage_format;
  typedef Layout<D, order_type, packing, storage_format> layout_type;

  typedef Strided<D, value_type, layout_type>  local_block_type;
  typedef distributed_block<local_block_type, M> dst_block_type;

  typedef typename block_traits<local_block_type>::plain_type local_block_ret_type;

  typedef typename view_of<dst_block_type>::type dst_view_type;
  typedef typename view_of<B>::const_type src_view_type;

  typedef typename
  choose_par_assign_impl<D, dst_block_type, B, false>::type par_assign_type;

  Expr_block(M const &map, B const &block)
  : map_(map),
    dom_(block_domain<D>(block)),
    dst_block_(new dst_block_type(dom_, map_), false),
    dst_(*dst_block_),
    src_(const_cast<B&>(block)),
    assign_(dst_, src_)
  {
  }
  ~Expr_block() {}

  void exec() { this->assign_();}
  length_type size() const VSIP_NOTHROW { return src_.block().size();}
  length_type size(dimension_type block_dim, dimension_type d) const VSIP_NOTHROW
  { return src_.block().size(block_dim, d);}

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}

  local_block_type &get_local_block() const
  { return dst_block_->get_local_block();}

private:
  M const &map_;
  Domain<dim> dom_;
  ovxx::refcounted_ptr<dst_block_type> dst_block_;
  dst_view_type   dst_;
  src_view_type   src_;
  Assignment<dim, dst_block_type, B, par_assign_type> assign_;
};

template <dimension_type D, typename M, typename B>
class Expr_block<D, M, B, peb_reuse> : ovxx::detail::noncopyable
{
public:
  static dimension_type const dim = D;

  typedef typename B::value_type           value_type;
  typedef typename B::reference_type       reference_type;
  typedef typename B::const_reference_type const_reference_type;
  typedef M                                map_type;


  typedef B const                          local_block_type;
  typedef typename block_traits<local_block_type>::plain_type local_block_ret_type;
  typedef distributed_block<local_block_type, M> dst_block_type;

  typedef typename view_of<dst_block_type>::type dst_view_type;
  typedef typename view_of<B>::const_type src_view_type;

  Expr_block(M const &map, B const &block)
    : map_(map),
      block_(const_cast<B&>(block))
  {}
  ~Expr_block() {}
  void exec() {}

  length_type size() const VSIP_NOTHROW { return block_.size();}
  length_type size(dimension_type block_dim, dimension_type d) const VSIP_NOTHROW
  { return block_.size(block_dim, d);}

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}

  local_block_type &get_local_block() const { return block_;}

private:
  M const &map_;
  typename block_traits<B>::expr_type block_;
};

template <typename B>
struct create_subblock;

template <typename B>
struct create_subblock<expr::Subset<B> >
{
  template <typename M>
  static expr::Subset<B> exec(M const &map, B const &block)
  {
    dimension_type const dim = B::dim;
    return expr::Subset<B>
      (map.template impl_global_domain<dim>(map.subblock(),0),block);
  }
};

template <typename M, typename B>
struct create_subblock<map_subset_block<B,M> >
{
  static map_subset_block<B,M> exec(M const &map, B const &block)
  {
    return map_subset_block<B,M>(block,map);
  }
};

template <typename M, typename B>
struct choose_local_block;

template <typename B, dimension_type D>
struct choose_local_block<Replicated_map<D>, B>
{
  typedef expr::Subset<B> block_type;
};

template <typename B>
struct choose_local_block<Map<Block_dist,Block_dist,Block_dist>, B>
{
  typedef expr::Subset<B> block_type;
};

template <typename M, typename B>
struct choose_local_block
{
  typedef map_subset_block<B,M> block_type;
};

template <dimension_type D, typename M, typename B>
class Expr_block<D, M, B, peb_remap> : ovxx::detail::noncopyable
{
public:
  static dimension_type const dim = D;

  typedef typename B::value_type           value_type;
  typedef typename B::reference_type       reference_type;
  typedef typename B::const_reference_type const_reference_type;
  typedef M                                map_type;


  typedef typename choose_local_block<M, B const>::block_type
                                           local_block_type;
  typedef typename block_traits<local_block_type>::plain_type local_block_ret_type;

  Expr_block(M const &map, B const &block)
    : map_(map),
      block_(block),
      subblock_(create_subblock<local_block_type>::exec(map_,block_))
  {}
  ~Expr_block() {}
  void exec() {}

  length_type size() const VSIP_NOTHROW { return block_.size();}
  length_type size(dimension_type block_dim, dimension_type d) const VSIP_NOTHROW
  { return block_.size(block_dim, d);}

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}

  local_block_ret_type get_local_block() const { return subblock_;}

private:
  M const &map_;
  typename block_traits<B const>::expr_type block_;
  local_block_type                          subblock_;
};

template <typename T> struct wrapper {};

/// Construct an expression of Expr_blocks from an
/// expression of distributed blockes.
///
/// get_expr_block() is a convenience function to apply this
/// functor.
template <dimension_type D, typename M>
class create_expr
{
public:
  template <typename B>
  struct tree_type
  {
    typedef Expr_block<D, M, B> type;
  };

  template <typename B>
  struct return_type
  {
    typedef Expr_block<D, M, B> &type;
  };

  create_expr(M const &map) : map_(map) {}

  // Leaf combine function.
  template <typename B>
  typename return_type<B>::type
  apply(B const &block) const
  {
    return *(new Expr_block<D, M, B>(map_, block));
  }

private:
  M const &map_;
};

/// 'Visitor' functor to call 'exec()' member function for each
/// leaf in an expression.
/// Expected to be used on expressions of Expr_blocks.
struct exec_expr
{
  template <typename B>
  void apply(B const &block) const
  {
    // const_cast necessary because Expr_binary_block accessors for
    // left() and right() return const.
    const_cast<B&>(block).exec();
  }
};

/// 'Visitor' functor to delete each leaf block in an expression.
/// Expected to be used on expressions of Expr_blocks.
struct free_expr
{
  template <typename B>
  void apply(B const &block) const { delete &block;}
};

/// Represent and evaluate a parallel expression.
template <dimension_type D, typename LHS, typename RHS>
class Expression
{
public:
  typedef typename LHS::value_type value1_type;
  typedef typename RHS::value_type value2_type;

  typedef typename LHS::map_type   lhs_map_type;
  typedef typename distributed_local_block<LHS>::type lhs_lblock_type;

  typedef typename
  expr::transform::return_type<create_expr<D, lhs_map_type>, RHS>
    ::tree_type rhs_peb_type;

  typedef typename distributed_local_block<rhs_peb_type>::type rhs_lblock_type;

public:
  Expression(typename view_of<LHS>::type lhs,
	     typename view_of<RHS>::const_type rhs)
  : lhs_(lhs),
    rhs_remap_(get_expr_block<D>(lhs.block().map(), rhs.block()))
  {}

  ~Expression()
  { free_expr_block(rhs_remap_.block());}

  void operator()()
  {
    typedef typename view_of<lhs_lblock_type>::type lhs_lview_type;
    typedef typename view_of<rhs_lblock_type>::const_type rhs_lview_type;

    // Reorganize data to owner-computes
    exec_expr_block(rhs_remap_.block());

    LHS &lhs_block = lhs_.block();

    if (lhs_block.map().subblock() != no_subblock)
    {
      lhs_lview_type lhs_lview = ovxx::get_local_view(lhs_);
      rhs_lview_type rhs_lview = ovxx::get_local_view(rhs_remap_);
      
#if 0
      // This is valid if the local view is element conformant to the
      // union of patches in corresponding subblock.
      //
      // Subset_maps do not meet this requirement currently.
      // (061212).
      lhs_lview = rhs_lview;
#else
      typedef typename LHS::map_type map_type;
      map_type const &map = lhs_block.map();

      index_type sb = map.subblock();

      for (index_type p=0; p<map.impl_num_patches(sb); ++p)
      {
	Domain<D> dom = map.template impl_local_domain<D>(sb, p);
	lhs_lview(dom) = rhs_lview(dom);
      }
#endif
    }
  }

private:
  typename view_of<LHS>::type lhs_;
  typename view_of<rhs_peb_type>::const_type rhs_remap_;
};

/// Overload of get_local_block for Expr_block.
template <dimension_type D, typename M, typename B, typename I>
typename Expr_block<D, M, B, I>::local_block_ret_type
get_local_block(Expr_block<D, M, B, I> const &block)
{
  return block.get_local_block();
}

/// Convert an expression of distributed blocks into an expression
/// of Expr_blocks
template <dimension_type D, typename M, typename B>
typename expr::transform::return_type<create_expr<D, M>, B>::type
get_expr_block(M const &map, B const &block)
{
  create_expr<D, M> func(map);
  return expr::transform::combine(func, block);
}

/// Perform data reorganization for an expression of Expr_blocks
template <typename B>
void
exec_expr_block(B const &block)
{
  exec_expr func;
  expr::transform::apply(func, block);
}

/// Deallocate leaf Expr_blocks in expression.
template <typename B>
void
free_expr_block(B const &block)
{
  free_expr func;
  expr::transform::apply(func, block);
}

/// Evaluate a parallel expression where LHS and RHS have same mapping.
///
/// Arguments:
///   :lhs: a distributed, non-const view to assign to.
///   :rhs: a distributed view of the same dimension as DST that holds
///         the values to assign into DST.
template <template <typename, typename> class View1,
	  typename                            T1,
	  typename                            Block1,
	  template <typename, typename> class View2,
	  typename                            T2,
	  typename                            Block2>
void
expr_simple(View1<T1, Block1> lhs, View2<T2, Block2> rhs)
{
  OVXX_CT_ASSERT((View1<T1, Block1>::dim == View2<T2, Block2>::dim));

  typedef typename distributed_local_block<Block1>::type lhs_lblock_type;
  typedef typename distributed_local_block<Block2>::type rhs_lblock_type;

  View1<T1, lhs_lblock_type> lhs_lview = ovxx::get_local_view(lhs);
  View2<T2, rhs_lblock_type> rhs_lview = ovxx::get_local_view(rhs);

  lhs_lview = rhs_lview;
}

/// Evaluate a parallel expression where LHS and RHS have different
/// mappings.
///
/// Arguments:
///   :lhs: a distributed, non-const view to assign to.
///   :rhs: a distributed view of the same dimension as DST that holds
///         the values to assign into DST.
template <template <typename, typename> class View1,
	  typename                            T1,
	  typename                            Block1,
	  template <typename, typename> class View2,
	  typename                            T2,
	  typename                            Block2>
void
expr(View1<T1, Block1> lhs, View2<T2, Block2> rhs)
{
  using namespace vsip::impl;
  OVXX_CT_ASSERT((View1<T1, Block1>::dim == View2<T2, Block2>::dim));
  dimension_type const dim = View1<T1, Block1>::dim;
  Expression<dim, Block1, Block2> expr(lhs, rhs);
  expr();
}

} // namespace ovxx::parallel

template <dimension_type D, typename M, typename B>
struct is_sized_block<parallel::Expr_block<D, M, B, parallel::peb_reuse> >
{ static bool const value = is_sized_block<B>::value;};

/// Specialize distributed_local_block traits class for Expr_block.
template <dimension_type D, typename M, typename B>
struct distributed_local_block<parallel::Expr_block<D, M, B> >
{
  typedef typename parallel::Expr_block<D, M, B>::local_block_type type;
  typedef typename parallel::Expr_block<D, M, B>::local_block_type proxy_type;
};
} // namespace ovxx

#endif

