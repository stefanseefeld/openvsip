//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_diag_hpp_
#define ovxx_expr_diag_hpp_

#include <ovxx/support.hpp>

namespace ovxx
{
namespace expr
{

/// The Diag class is similar to the Sliced class, 
/// producing diagonal slices based on an offset.  The center diagonal
/// is obtained with offset zero.   Positive values refer to diagonals
/// above the center and negative values to diagonals below.
/// Note the length of the resultant vector is affected by the size of
/// the original matrix as well as the offset from the main diagonal.
template <typename B> 
class Diag : ct_assert<(B::dim == 2)>, ovxx::detail::nonassignable
{
 public:
  static dimension_type const dim = B::dim - 1;
  typedef typename B::value_type value_type;
  typedef value_type &reference_type;
  typedef value_type const &const_reference_type;
  typedef typename B::map_type map_type;
  typedef storage_traits<value_type, get_block_layout<B>::storage_format> storage;
  typedef typename storage::ptr_type ptr_type;
  typedef typename storage::const_ptr_type const_ptr_type;


  Diag(Diag const &sb) VSIP_NOTHROW
  : block_(&*sb.block_), offset_(sb.offset_)
  {}
  Diag(B &block, index_difference_type offset) VSIP_NOTHROW
  : block_(&block), offset_(offset)
  {}
  ~Diag() VSIP_NOTHROW {}

  length_type size() const VSIP_NOTHROW
  { 
    OVXX_PRECONDITION(dim == 1);
    length_type size = 0;
    length_type limit = std::min(this->block_->size(2, 0), this->block_->size(2, 1));
    if ( this->offset_ >= 0 )
      size = std::min( limit, this->block_->size(2, 1) - this->offset_ );
    else
      size = std::min( limit, this->block_->size(2, 0) + this->offset_ );
    
    return size;
  }
  length_type size(dimension_type block_d OVXX_UNUSED, dimension_type d OVXX_UNUSED) const VSIP_NOTHROW
  { 
    OVXX_PRECONDITION(block_d == 1 && dim == 1 && d == 0);
    return this->size();
  }

  // These are noops as Diag_block is held by-value.
  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const { return this->block_->map(); }

  // Data accessors.
  value_type get(index_type i) const VSIP_NOTHROW
  { 
    if ( this->offset_ >= 0 )
      return this->block_->get(i, i + this->offset_);
    else 
      return this->block_->get(i - this->offset_, i);
  }

  void put(index_type i, value_type val) VSIP_NOTHROW
  {
    if ( this->offset_ >= 0 )
      return this->block_->put(i, i + this->offset_, val);
    else 
      return this->block_->put(i - this->offset_, i, val);
  }
  
  reference_type ref(index_type i) VSIP_NOTHROW
  {
    if ( this->offset_ >= 0 )
      return this->block_->ref(i, i + this->offset_);
    else
      return this->block_->ref(i - this->offset_, i);
  }
  
  ptr_type ptr() VSIP_NOTHROW
  { 
    if (this->offset_ >= 0)
      return storage::offset(this->block_->ptr(),
			     + this->offset_ * this->block_->stride(2, 1));
    else
      return storage::offset(this->block_->ptr(),
			     - this->offset_ * this->block_->stride(2, 0));
  }

  const_ptr_type ptr() const VSIP_NOTHROW
  { 
    if (this->offset_ >= 0)
      return storage::offset(this->block_->ptr(),
			     + this->offset_ * this->block_->stride(2, 1));
    else
      return storage::offset(this->block_->ptr(),
			     - this->offset_ * this->block_->stride(2, 0));
  }

  stride_type stride(dimension_type Dim OVXX_UNUSED, dimension_type d)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(Dim == dim && d<dim);
    return this->block_->stride(2, 0) + this->block_->stride(2, 1);
  }
  
 private:
  typename block_traits<B>::ptr_type block_;
  index_difference_type const offset_;
};

} // namespace ovxx::expr

template <typename B>
struct block_traits<expr::Diag<B> > : by_value_traits<expr::Diag<B> >
{};

template <typename B>
struct is_modifiable_block<expr::Diag<B> > : is_modifiable_block<B> {};

template <typename B, dimension_type D>
struct lvalue_factory_type<expr::Diag<B>, D>
{
  typedef typename lvalue_factory_type<B, D>
    ::template rebind<expr::Diag<B> >::type type;
  template <typename O>
  struct rebind 
  {
    typedef typename lvalue_factory_type<B, D>
      ::template rebind<O>::type type;
  };
};

} // namespace ovxx

namespace vsip
{
template <typename B>
struct get_block_layout<ovxx::expr::Diag<B> >
{
  static dimension_type const dim = 1;

  typedef row1_type order_type;
  static pack_type const packing = any_packing;
  static storage_format_type const storage_format = get_block_layout<B>::storage_format;

  typedef Layout<dim, order_type, packing, storage_format> type;
};

template <typename B>
struct supports_dda<ovxx::expr::Diag<B> >
{ static bool const value = supports_dda<B>::value;};

} // namespace vsip

#endif
