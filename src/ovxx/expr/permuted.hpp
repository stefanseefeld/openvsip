//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_permuted_hpp_
#define ovxx_expr_permuted_hpp_

namespace ovxx
{
namespace expr
{
namespace op
{
/// The Permutor policy class provides functions which do the actual
/// argument reorder.  There is no general version, but below are
/// partial specializations for all permutations of <0,1,2>.
/// Block is the underlying block, Ordering is a tuple.  This class
/// is not used directly.
template <typename Block, typename Ordering> struct Permutor;

#define OVXX_PERMUTOR_SPECIALIZATION(n0,n1,n2)				\
template <typename Block>                                               \
struct Permutor<Block, tuple<n0, n1, n2> >                              \
{                                                                       \
  static dimension_type dimension_order(dimension_type d) VSIP_NOTHROW  \
  {									\
    static const dimension_type permutation[3] = { n0, n1, n2};		\
    return permutation[d];						\
  }									\
  static typename Block::value_type                                     \
  get(Block const& block, index_type i##n0, index_type i##n1,		\
      index_type i##n2) VSIP_NOTHROW                                    \
  { return block.get(i0, i1, i2);}					\
  static void                                                           \
  put(Block& block, index_type i##n0, index_type i##n1, index_type i##n2, \
      typename Block::value_type val) VSIP_NOTHROW                      \
  { block.put(i0, i1, i2, val);}					\
  static typename Block::reference_type                                 \
  ref(Block& block, index_type i##n0, index_type i##n1,			\
      index_type i##n2) VSIP_NOTHROW                                    \
  { return block.ref(i0, i1, i2);}					\
}

OVXX_PERMUTOR_SPECIALIZATION(0,1,2);
OVXX_PERMUTOR_SPECIALIZATION(1,0,2);
OVXX_PERMUTOR_SPECIALIZATION(1,2,0);
OVXX_PERMUTOR_SPECIALIZATION(2,1,0);
OVXX_PERMUTOR_SPECIALIZATION(2,0,1);
OVXX_PERMUTOR_SPECIALIZATION(0,2,1);

#undef OVXX_PERMUTOR_SPECIALIZATION

}

/// The Permuted class reorders the indices to a 3-dimensional
/// block, and the dimensions visible via 2-argument size().  The
/// permutation is specified as a tuple (see [support]).
template <typename Block, typename Ordering>
class Permuted : ct_assert<Block::dim == 3>, ovxx::detail::nonassignable
{
protected:
  // Policy class.
  typedef op::Permutor<Block, Ordering> perm_type;

public:
  static dimension_type const dim = Block::dim;
  typedef typename Block::value_type value_type;
  typedef value_type&       reference_type;
  typedef value_type const& const_reference_type;
  typedef typename Block::map_type map_type;

  Permuted(Block &block) VSIP_NOTHROW
  : block_ (&block) {}
  Permuted(Permuted const& pb) VSIP_NOTHROW
    : block_ (&*pb.block_) {}
  ~Permuted() VSIP_NOTHROW {}

  length_type size() const VSIP_NOTHROW
  { return block_->size();}
  length_type size(dimension_type block_d, dimension_type d) const VSIP_NOTHROW
  { return block_->size(block_d, perm_type::dimension_order(d));}
  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const { return block_->map();}

  value_type get(index_type i, index_type j, index_type k) const VSIP_NOTHROW
  { return perm_type::get(*block_, i, j, k);}

  void put(index_type i, index_type j, index_type k, value_type val) VSIP_NOTHROW
  { perm_type::put(*block_, i, j, k, val);}

  reference_type ref(index_type i, index_type j, index_type k) VSIP_NOTHROW
  { return perm_type::ref(*block_, i, j, k);}

  Block const &block() const { return *this->block_;}

  typedef storage_traits<value_type, get_block_layout<Block>::storage_format> storage;
  typedef typename storage::ptr_type ptr_type;
  typedef typename storage::const_ptr_type const_ptr_type;

  ptr_type ptr() VSIP_NOTHROW
  { return block_->ptr();}

  const_ptr_type ptr() const VSIP_NOTHROW
  { return block_->ptr();}

  stride_type stride(dimension_type Dim, dimension_type d)
     const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(Dim == dim && d<dim);
    return block_->stride(dim,  perm_type::dimension_order(d));
  }

 private:
  typename block_traits<Block>::ptr_type block_;
};


/// Take permutation of dimension-order.
template <typename PermutionT,		// Tuple expressing permutation
	  typename OrderT>		// Original dim-order
struct Permute_order;

template <dimension_type Dim0, dimension_type Dim1, dimension_type Dim2>
struct Permute_order<tuple<0, 1, 2>, tuple<Dim0, Dim1, Dim2> >
{ typedef tuple<Dim0, Dim1, Dim2> type; };

template <dimension_type Dim0, dimension_type Dim1, dimension_type Dim2>
struct Permute_order<tuple<1, 0, 2>, tuple<Dim0, Dim1, Dim2> >
{ typedef tuple<Dim1, Dim0, Dim2> type; };

template <dimension_type Dim0, dimension_type Dim1, dimension_type Dim2>
struct Permute_order<tuple<0, 2, 1>, tuple<Dim0, Dim1, Dim2> >
{ typedef tuple<Dim0, Dim2, Dim1> type; };

template <dimension_type Dim0, dimension_type Dim1, dimension_type Dim2>
struct Permute_order<tuple<1, 2, 0>, tuple<Dim0, Dim1, Dim2> >
{ typedef tuple<Dim1, Dim2, Dim0> type; };

template <dimension_type Dim0, dimension_type Dim1, dimension_type Dim2>
struct Permute_order<tuple<2, 1, 0>, tuple<Dim0, Dim1, Dim2> >
{ typedef tuple<Dim2, Dim1, Dim0> type; };

template <dimension_type Dim0, dimension_type Dim1, dimension_type Dim2>
struct Permute_order<tuple<2, 0, 1>, tuple<Dim0, Dim1, Dim2> >
{ typedef tuple<Dim2, Dim0, Dim1> type; };


} // namespace ovxx::expr

template <typename B, typename O>
struct block_traits<expr::Permuted<B, O> >
  : by_value_traits<expr::Permuted<B, O> > {};

template <typename B, typename O>
struct is_modifiable_block<expr::Permuted<B, O> > : is_modifiable_block<B>
{};

template <typename B, typename O>
struct lvalue_factory_type<expr::Permuted<B, O> >
{
  typedef typename lvalue_factory_type<B>::
   template rebind<expr::Permuted<B, O> >::type type;
  template <typename B1>
  struct rebind 
  {
    typedef typename lvalue_factory_type<B>::
      template rebind<B1>::type type;
  };
};

} // namespace ovxx

namespace vsip
{

/// dimension-order is permuted, pack-type is set to unknown.
template <typename B, typename P>
struct get_block_layout<ovxx::expr::Permuted<B, P> >
{
  static dimension_type const dim = B::dim;

  typedef typename ovxx::expr::Permute_order<P, 
    typename get_block_layout<B>::order_type>::type
    order_type;
  static pack_type const packing = any_packing;
  static storage_format_type const storage_format = get_block_layout<B>::storage_format;

  typedef Layout<dim, order_type, packing, storage_format> type;
};

template <typename B, typename P>
struct supports_dda<ovxx::expr::Permuted<B, P> >
{ static bool const value = supports_dda<B>::value;};

} // namespace vsip

#endif
