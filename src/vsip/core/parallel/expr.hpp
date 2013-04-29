//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_PARALLEL_EXPR_HPP
#define VSIP_CORE_PARALLEL_EXPR_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/strided.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/parallel/block.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/parallel/assign.hpp>
#include <vsip/core/parallel/choose_assign_impl.hpp>
#include <vsip/core/expr/map_subset_block.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

// Forward.
template <dimension_type Dim,
	  typename       MapT>
class Create_par_expr;

// Forward.
template <dimension_type Dim,
	  typename       MapT,
	  typename       BlockT>
typename Combine_return_type<Create_par_expr<Dim, MapT>, BlockT>::type
get_par_expr_block(
  MapT const&   map,
  BlockT const& block);



template <dimension_type Dim,
	  typename       MapT,
	  typename       BlockT>
class Par_expr_block<Dim, MapT, BlockT, Peb_reorg_tag> : Non_copyable
{
public:
  static dimension_type const dim = Dim;

  typedef typename BlockT::value_type           value_type;
  typedef typename BlockT::reference_type       reference_type;
  typedef typename BlockT::const_reference_type const_reference_type;
  typedef MapT                                  map_type;

  // The layout of the reorg block should have the same dimension-
  // order and complex format as the source block.  Packing format
  // should either be unit-stride-dense or unit-stride-aligned.
  // It should not be taken directly from BlockT since it may have
  // a non realizable packing format such as packing::unknown.
  typedef typename get_block_layout<BlockT>::order_type        order_type;
  static pack_type const packing = dense;
  static storage_format_type const storage_format = get_block_layout<BlockT>::storage_format;
  typedef Layout<Dim, order_type, packing, storage_format> layout_type;

  typedef Strided<Dim, value_type, layout_type>  local_block_type;
  typedef Distributed_block<local_block_type, MapT> dst_block_type;

  typedef typename View_block_storage<local_block_type>::plain_type
                                                local_block_ret_type;

  typedef typename view_of<dst_block_type>::type dst_view_type;
  typedef typename view_of<BlockT>::const_type src_view_type;

  typedef typename
    Choose_par_assign_impl<Dim, dst_block_type, BlockT, false>::type
    par_assign_type;


public:
  Par_expr_block(MapT const& map, BlockT const& block);
  ~Par_expr_block() {}

  void exec() { this->assign_(); }

  // Accessors.
public:
  length_type size() const VSIP_NOTHROW { return src_.block().size(); }
  length_type size(dimension_type blk_dim, dimension_type d) const VSIP_NOTHROW
  { return src_.block().size(blk_dim, d); }

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}

  // Distributed Accessors
public:
  local_block_type& get_local_block() const
    { return dst_block_->get_local_block(); }

  // Member data.
private:
  MapT const&     map_;
  Domain<Dim>     dom_;
  Ref_counted_ptr<dst_block_type> dst_block_;
  dst_view_type   dst_;
  src_view_type   src_;
  Par_assign<Dim, value_type, value_type, dst_block_type, BlockT,
             par_assign_type>
		  assign_;
};



template <dimension_type Dim,
	  typename       MapT,
	  typename       BlockT>
class Par_expr_block<Dim, MapT, BlockT, Peb_reuse_tag> : Non_copyable
{
public:
  static dimension_type const dim = Dim;

  typedef typename BlockT::value_type           value_type;
  typedef typename BlockT::reference_type       reference_type;
  typedef typename BlockT::const_reference_type const_reference_type;
  typedef MapT                                  map_type;


  typedef BlockT const                              local_block_type;
  typedef typename View_block_storage<local_block_type>::plain_type
                                                local_block_ret_type;
  typedef Distributed_block<local_block_type, MapT> dst_block_type;

  typedef typename view_of<dst_block_type>::type dst_view_type;
  typedef typename view_of<BlockT>::const_type src_view_type;


public:
  Par_expr_block(MapT const& map, BlockT const& block)
    : map_ (map),
      blk_ (const_cast<BlockT&>(block))
  {}

  ~Par_expr_block() {}

  void exec() {}

  // Accessors.
public:
  length_type size() const VSIP_NOTHROW { return blk_.size(); }
  length_type size(dimension_type blk_dim, dimension_type d) const VSIP_NOTHROW
  { return blk_.size(blk_dim, d); }

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}

  // Distributed Accessors
public:
  local_block_type& get_local_block() const
    { return blk_; }

  // Member data.
private:
  MapT const&     map_;
  typename View_block_storage<BlockT>::expr_type blk_;
};



template <dimension_type Dim,
	  typename       MapT,
	  typename       BlockT>
struct Is_sized_block<Par_expr_block<Dim, MapT, BlockT, Peb_reuse_tag> >
{ static bool const value = Is_sized_block<BlockT>::value; };



template <typename RetBlock>
struct Create_subblock;

template <typename BlockT>
struct Create_subblock<Subset_block<BlockT> >
{
  template <typename MapT>
  static Subset_block<BlockT> exec(MapT const& map, BlockT const& blk)
  {
    dimension_type const dim = BlockT::dim;
    return Subset_block<BlockT>
      (map.template impl_global_domain<dim>(map.subblock(),0),blk);
  }
};

template <typename MapT, typename BlockT>
struct Create_subblock<Map_subset_block<BlockT,MapT> >
{
  static Map_subset_block<BlockT,MapT> exec(MapT const& map, BlockT const& blk)
  {
    return Map_subset_block<BlockT,MapT>(blk,map);
  }
};

template <typename MapT, typename BlockT>
struct Choose_local_block;

template <typename BlockT, dimension_type Dim>
struct Choose_local_block<Replicated_map<Dim>, BlockT>
{
  typedef Subset_block<BlockT> block_type;
};

template <typename BlockT, dimension_type Dim>
struct Choose_local_block<Global_map<Dim>, BlockT>
{
  typedef Subset_block<BlockT> block_type;
};

template <typename BlockT>
struct Choose_local_block<Map<Block_dist,Block_dist,Block_dist>, BlockT>
{
  typedef Subset_block<BlockT> block_type;
};

template <typename MapT, typename BlockT>
struct Choose_local_block
{
  typedef Map_subset_block<BlockT,MapT> block_type;
};


template <dimension_type Dim,
	  typename       MapT,
	  typename       BlockT>
class Par_expr_block<Dim, MapT, BlockT, Peb_remap_tag> : Non_copyable
{
public:
  static dimension_type const dim = Dim;

  typedef typename BlockT::value_type           value_type;
  typedef typename BlockT::reference_type       reference_type;
  typedef typename BlockT::const_reference_type const_reference_type;
  typedef MapT                                  map_type;


  typedef typename Choose_local_block<MapT, BlockT const>::block_type
                                                local_block_type;
  typedef typename View_block_storage<local_block_type>::plain_type
                                                local_block_ret_type;

public:
  Par_expr_block(MapT const& map, BlockT const& block)
    : map_     (map),
      blk_     (block),
      subblock_(Create_subblock<local_block_type>::exec(map_,blk_))
  {}

  ~Par_expr_block() {}

  void exec() {}

  // Accessors.
public:
  length_type size() const VSIP_NOTHROW { return blk_.size(); }
  length_type size(dimension_type blk_dim, dimension_type d) const VSIP_NOTHROW
  { return blk_.size(blk_dim, d); }

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}

  // Distributed Accessors
public:
  local_block_ret_type get_local_block() const
  {
    return subblock_;
  }


  // Member data.
private:
  MapT const&                                          map_;
  typename View_block_storage<BlockT const>::expr_type blk_;
  local_block_type                                     subblock_;
};


/// 'Combine' functor to construct an expression of Par_expr_blocks from an
/// expression of distributed blockes.

/// get_par_expr_block() is a convenience function to apply this
/// functor.

template <dimension_type Dim,
	  typename       MapT>
class Create_par_expr
{
public:
  template <typename BlockT>
  struct tree_type
  {
    typedef Par_expr_block<Dim, MapT, BlockT> type;
  };

  template <typename BlockT>
  struct return_type
  {
    typedef Par_expr_block<Dim, MapT, BlockT>& type;
  };

  // Constructors.
public:
  Create_par_expr(MapT const& map) : map_(map) {}

  // Leaf combine function.
  template <typename BlockT>
  typename return_type<BlockT>::type
  apply(BlockT const& block) const
  {
    return *(new Par_expr_block<Dim, MapT, BlockT>(map_, block));
  }

  // Member data.
private:
  MapT const& map_;

};



/// 'Visitor' functor to call 'exec()' member function for each
/// leaf in an expression.

/// Expected to be used on expressions of Par_expr_blocks.

class Exec_par_expr
{
  // Constructors.
public:
  Exec_par_expr() {}

  // Leaf visitor function.
  template <typename BlockT>
  void
  apply(BlockT const& block) const
  {
    // const_cast necessary because Expr_binary_block accessors for
    // left() and right() return const.
    const_cast<BlockT&>(block).exec();
  }
};



/// 'Visitor' functor to delete each leaf block in an expression.

/// Expected to be used on expressions of Par_expr_blocks.

class Free_par_expr
{
  // Constructors.
public:
  Free_par_expr() {}

  // Leaf visitor function.
  template <typename BlockT>
  void
  apply(BlockT const& block) const
  {
    delete &block;
  }
};



/// Represent and evaluate a parallel expression.
template <dimension_type Dim,
	  typename       DstBlock,
	  typename       SrcBlock>
class Par_expr
{
  // Compile-time values and typedefs.
  typedef typename DstBlock::value_type value1_type;
  typedef typename SrcBlock::value_type value2_type;

  typedef typename DstBlock::map_type   dst_map_type;
  typedef typename Distributed_local_block<DstBlock>::type dst_lblock_type;

  typedef typename
    Combine_return_type<Create_par_expr<Dim, dst_map_type>, SrcBlock>
    ::tree_type src_peb_type;

  typedef typename Distributed_local_block<src_peb_type>::type src_lblock_type;

  // Constructors
public:
  Par_expr(typename view_of<DstBlock>::type dst,
	   typename view_of<SrcBlock>::const_type src)
  : dst_      (dst),
    src_remap_(get_par_expr_block<Dim>(dst.block().map(), src.block()))
  {}

  ~Par_expr()
  { free_par_expr_block(src_remap_.block()); }

  void operator()()
  {
    typedef typename view_of<dst_lblock_type>::type dst_lview_type;
    typedef typename view_of<src_lblock_type>::const_type src_lview_type;

    // Reorganize data to owner-computes
    exec_par_expr_block(src_remap_.block());

    DstBlock&      dst_block = dst_.block();

    if (dst_block.map().subblock() != no_subblock)
    {
      dst_lview_type dst_lview = get_local_view(dst_);
      src_lview_type src_lview = get_local_view(src_remap_);
      
#if 0
      // This is valid if the local view is element conformant to the
      // union of patches in corresponding subblock.
      //
      // Subset_maps do not meet this requirement currently.
      // (061212).
      dst_lview = src_lview;
#else
      typedef typename DstBlock::map_type map_type;
      map_type const& map = dst_block.map();

      index_type sb = map.subblock();

      for (index_type p=0; p<map.impl_num_patches(sb); ++p)
      {
	Domain<Dim> dom = map.template impl_local_domain<Dim>(sb, p);
	dst_lview(dom) = src_lview(dom);
      }
#endif
    }
  }

  // Member data.
private:
  typename view_of<DstBlock>::type dst_;
  typename view_of<src_peb_type>::const_type src_remap_;
};



/// Specialize Distributed_local_block traits class for Par_expr_block.
template <dimension_type Dim,
	  typename       MapT,
	  typename       BlockT>
struct Distributed_local_block<Par_expr_block<Dim, MapT, BlockT> >
{
  typedef typename Par_expr_block<Dim, MapT, BlockT>::local_block_type
		type;
  typedef typename Par_expr_block<Dim, MapT, BlockT>::local_block_type
		proxy_type;
};



#if VSIP_IMPL_USE_GENERIC_VISITOR_TEMPLATES==0

/// Determine return type for combining a leaf Par_expr_block.
template <typename       CombineT,
	  dimension_type Dim,
	  typename       MapT,
	  typename       BlockT>
struct Combine_return_type<CombineT, Par_expr_block<Dim, MapT, BlockT> >
{
  typedef Par_expr_block<Dim, MapT, BlockT> block_type;
  typedef typename CombineT::template return_type<block_type>::type
		type;
  typedef typename CombineT::template tree_type<block_type>::type
		tree_type;
};



/// Specialize apply_combine for Par_expr_block leaves.
template <typename       CombineT,
	  dimension_type Dim,
	  typename       MapT,
	  typename       BlockT>
typename Combine_return_type<CombineT,
			     Par_expr_block<Dim, MapT, BlockT> >::type
apply_combine(
  CombineT const&                          combine,
  Par_expr_block<Dim, MapT, BlockT> const& block)
{
  return combine.apply(block);
}



/// Specialize apply_leaf for Par_expr_block leaves.
template <typename       VisitorT,
	  dimension_type Dim,
	  typename       MapT,
	  typename       BlockT>
void
apply_leaf(
  VisitorT const&                          visitor,
  Par_expr_block<Dim, MapT, BlockT> const& block)
{
  visitor.apply(block);
}
#endif



/***********************************************************************
  Definitions
***********************************************************************/

template <dimension_type Dim,
	  typename       MapT,
	  typename       BlockT>
Par_expr_block<Dim, MapT, BlockT, Peb_reorg_tag>::Par_expr_block(
  MapT const&   map,
  BlockT const& block)
  : map_      (map),
    dom_      (block_domain<Dim>(block)),
    dst_block_(new dst_block_type(dom_, map_), noincrement),
    dst_      (*dst_block_),
    src_      (const_cast<BlockT&>(block)),
    assign_   (dst_, src_)
{
}



/// Overload of get_local_block for Par_expr_block.
template <dimension_type Dim,
	  typename       MapT,
	  typename       BlockT,
	  typename       ImplTag>
typename Par_expr_block<Dim, MapT, BlockT, ImplTag>::local_block_ret_type
get_local_block(
  Par_expr_block<Dim, MapT, BlockT, ImplTag> const& block)
{
  return block.get_local_block();
}



/// Convert an expression of distributed blocks into an expression
/// of Par_expr_blocks
template <dimension_type Dim,
	  typename       MapT,
	  typename       BlockT>
typename Combine_return_type<Create_par_expr<Dim, MapT>, BlockT>::type
get_par_expr_block(
  MapT const&   map,
  BlockT const& block)
{
  Create_par_expr<Dim, MapT> combine(map);
  return apply_combine(combine, block);
}



/// Perform data reorganization for an expression of Par_expr_blocks
template <typename BlockT>
void
exec_par_expr_block(
  BlockT const& block)
{
  Exec_par_expr visitor;
  apply_leaf(visitor, block);
}



/// Deallocate leaf Par_expr_blocks in expression.
template <typename BlockT>
void
free_par_expr_block(
  BlockT const& block)
{
  Free_par_expr visitor;
  apply_leaf(visitor, block);
}




/// Evaluate a parallel expression where LHS and RHS have same mapping.
///
/// Arguments:
///   :dst: a distributed, non-const view to assign to.
///   :src: a distributed view of the same dimension as DST that holds
///         the values to assign into DST.
template <template <typename, typename> class View1,
	  template <typename, typename> class View2,
	  typename                            T1,
	  typename                            Block1,
	  typename                            T2,
	  typename                            Block2>
void
par_expr_simple(View1<T1, Block1> dst,
		View2<T2, Block2> src)
{
  VSIP_IMPL_STATIC_ASSERT((View1<T1, Block1>::dim == View2<T2, Block2>::dim));

  typedef typename Distributed_local_block<Block1>::type dst_lblock_type;
  typedef typename Distributed_local_block<Block2>::type src_lblock_type;

  // Already checked by Dispatch_assign prior to calling:
  //
  // assert(Is_par_same_map<map_t, Block2>::value(dst.block().map(),
  //                                              src.block()));

  View1<T1, dst_lblock_type> dst_lview = get_local_view(dst);
  View2<T2, src_lblock_type> src_lview = get_local_view(src);

  dst_lview = src_lview;
}



/// Evaluate a parallel expression where LHS and RHS have different
/// mappings.
///
/// Arguments:
///   :dst: a distributed, non-const view to assign to.
///   :src: a distributed view of the same dimension as DST that holds
///         the values to assign into DST.
template <template <typename, typename> class View1,
	  template <typename, typename> class View2,
	  typename                            T1,
	  typename                            Block1,
	  typename                            T2,
	  typename                            Block2>
void
par_expr(View1<T1, Block1> dst,
	 View2<T2, Block2> src)
{
  VSIP_IMPL_STATIC_ASSERT((View1<T1, Block1>::dim == View2<T2, Block2>::dim));
  dimension_type const dim = View1<T1, Block1>::dim;

  Par_expr<dim, Block1, Block2> par_expr(dst, src);

  par_expr();
}

} // namespace vsip::impl
} // namespace vsip




#endif // VSIP_CORE_PARALLEL_EXPR_HPP

