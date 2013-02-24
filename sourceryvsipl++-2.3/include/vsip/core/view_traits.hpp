/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/view_traits.hpp
    @author  Stefan Seefeld
    @date    2005-04-16
    @brief   VSIPL++ Library: helper templates related to views.

*/

#ifndef VSIP_CORE_VIEW_TRAITS_HPP
#define VSIP_CORE_VIEW_TRAITS_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/view_fwd.hpp>
#include <vsip/core/subblock.hpp>
#include <vsip/core/parallel/get_local_view.hpp>
#include <complex>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

template <template <typename, typename> class View,
	  typename                            T,
	  typename                            Block>
struct ViewConversion;

namespace impl
{

template <template <typename, typename> class View,
	  typename                            T,
	  typename                            Block>
View<T, typename Distributed_local_block<Block>::type>
get_local_view(View<T, Block> v);

/// Template class to help instantiate a view of a given dimension.
template <dimension_type Dim,
	  typename       T,
	  typename       Block>
struct View_of_dim;

/// Trait that provides typedef 'type' iff VIEW is a valid view.
///
/// Views should provide specializations similar to:
///    template <> struct Is_view_type<Matrix>       { typedef void type; };
///    template <> struct Is_view_type<const_Matrix> { typedef void type; };
template <typename> struct Is_view_type { static bool const value = false;};


/// Trait that provides typedef 'type' iff VIEW is a valid const view.
template <typename> struct Is_const_view_type
{ static bool const value = false;};

// Dim_of_view<V>

template <template <typename,typename> class View> 
            struct Dim_of_view;

template <> struct Dim_of_view<vsip::Vector>
  { static const dimension_type dim = 1; };

template <> struct Dim_of_view<vsip::Matrix>
  { static const dimension_type dim = 2; };

template <> struct Dim_of_view<vsip::Tensor> 
  { static const dimension_type dim = 3; };

template <> struct Dim_of_view<vsip::const_Vector>
  { static const dimension_type dim = 1; };

template <> struct Dim_of_view<vsip::const_Matrix>
  { static const dimension_type dim = 2; };

template <> struct Dim_of_view<vsip::const_Tensor> 
  { static const dimension_type dim = 3; };

// Const_of_view<V,B>

template <template <typename,typename> class View, typename Block> 
                          struct Const_of_view;

template <typename Block> struct Const_of_view<vsip::Vector,Block>
  { typedef vsip::const_Vector<typename Block::value_type,Block> view_type; };

template <typename Block> struct Const_of_view<vsip::Matrix,Block>
  { typedef vsip::const_Matrix<typename Block::value_type,Block> view_type; };

template <typename Block> struct Const_of_view<vsip::Tensor,Block> 
  { typedef vsip::const_Tensor<typename Block::value_type,Block> view_type; };

} // namespace impl

template <template <typename,typename> class View,
	  typename Block, typename T = typename Block::value_type,
	  dimension_type Dim = impl::Dim_of_view<View>::dim >
struct impl_const_View
{ 
  static const bool impl_is_view = true;
  typedef View<T,Block> impl_vsip_view_type;
  typedef impl_const_View impl_const_view_type;
  typedef typename impl::View_block_storage<Block>::type impl_storage_type;

  // Types for local views of distributed objects.
  typedef typename impl::Distributed_local_block<Block>::type
		 impl_localblock_type;
  typedef View<T, impl_localblock_type> local_type;
    
  impl_const_View(Block* blk)
    : impl_blk(blk) {}
  impl_const_View(Block* blk, impl::noincrement_type)
    : impl_blk(blk,impl::noincrement) {}

  View<T,Block>& impl_view()
    { return static_cast<View<T,Block>&>(*this); }
  View<T,Block> const& impl_view() const
    { return static_cast<View<T,Block> const&>(*this); }

  // [view.vector.accessors]
  // [view.matrix.accessors]
  // [view.tensor.accessors]
  Block& block() const VSIP_NOTHROW
    { return *this->impl_blk; }
  vsip::length_type size() const VSIP_NOTHROW
    { return this->impl_blk->size(); }
  vsip::length_type size(vsip::dimension_type d) const VSIP_NOTHROW
    { return this->impl_blk->size(Dim, d); }

  // [parview.vector.accessors]
  local_type local() const VSIP_NOTHROW
    { return vsip::impl::get_local_view(this->impl_view()); }

protected:
   impl_storage_type impl_blk;
};

// specialize for element type std::complex<T>

template <template <typename,typename> class View,
	  typename Block, typename T, dimension_type Dim>
struct impl_const_View<View,Block,std::complex<T>,Dim>
{ 
  static const bool impl_is_view = true;
  typedef View<std::complex<T>,Block> impl_vsip_view_type;
  typedef impl_const_View impl_const_view_type;
  typedef typename impl::View_block_storage<Block>::type impl_storage_type;

  // Types for local views of distributed objects.
  typedef typename impl::Distributed_local_block<Block>::type
		 impl_localblock_type;
  typedef View<std::complex<T>, impl_localblock_type> local_type;

  typedef impl::Component_block<Block,impl::Real_extractor>
    impl_rblock_type;
  typedef impl::Component_block<Block,impl::Imag_extractor>
    impl_iblock_type;
  typedef T impl_scalar_type;

  typedef View<T,impl_rblock_type> realview_type;
  typedef View<T,impl_iblock_type> imagview_type;
  typedef View<T,impl_rblock_type> const_realview_type;
  typedef View<T,impl_iblock_type> const_imagview_type;

  impl_const_View(Block* blk)
    : impl_blk(blk) {}
  impl_const_View(Block* blk, impl::noincrement_type)
    : impl_blk(blk,impl::noincrement) {}

  const_realview_type real() const VSIP_THROW((std::bad_alloc))
  {
    impl_rblock_type block(*this->impl_blk);
    return const_realview_type(block);
  }
  const_imagview_type imag() const VSIP_THROW((std::bad_alloc))
  {
    impl_iblock_type block(*this->impl_blk);
    return const_imagview_type(block);
  }

  View<std::complex<T>,Block>& impl_view()
    { return static_cast<View<std::complex<T>,Block>&>(*this); }
  View<std::complex<T>,Block> const& impl_view() const
    { return static_cast<View<std::complex<T>,Block> const&>(*this); }

  // [view.vector.accessors]
  // [view.matrix.accessors]
  // [view.tensor.accessors]
  Block& block() const VSIP_NOTHROW
    { return *this->impl_blk; }
  vsip::length_type size() const VSIP_NOTHROW
    { return this->impl_blk->size(); }
  vsip::length_type size(vsip::dimension_type d) const VSIP_NOTHROW
    { return this->impl_blk->size(Dim, d); }

  // [parview.vector.accessors]
  local_type local() const VSIP_NOTHROW
    { return vsip::impl::get_local_view(this->impl_view()); }

protected:
   impl_storage_type impl_blk;
};

namespace impl
{
  enum disambiguator_type { disambiguate }; 
}

template <template <typename,typename> class View,
	  typename Block, typename T = typename Block::value_type,
	  dimension_type Dim = impl::Dim_of_view<View>::dim>
struct impl_View : impl::Const_of_view<View,Block>::view_type
{ 
  static const bool impl_is_nonconst_view = true;
  typedef View<T,Block> impl_vsip_view_type;
  typedef typename impl::Const_of_view<View,Block>::view_type impl_base_type;
  typedef typename Block::map_type impl_map_type;
  typedef typename
    impl::Const_of_view<View,Block>::view_type::impl_const_view_type
      impl_const_view_type;

  // Types for local views of distributed objects.
  typedef typename impl_base_type::impl_localblock_type impl_localblock_type;
  typedef View<T, impl_localblock_type> local_type;

  impl_View(length_type len, T const& value,
       impl_map_type const& map, impl::disambiguator_type)
    : impl_base_type(len, value, map) {}
  impl_View(length_type len1, length_type len2, T const& value,
       impl_map_type const& map, impl::disambiguator_type)
    : impl_base_type(len1, len2, value, map) {}
  impl_View(length_type len1, length_type len2, length_type len3,
       T const& value, impl_map_type const& map, impl::disambiguator_type)
    : impl_base_type(len1, len2, len3, value, map) {}
  impl_View(length_type len, impl_map_type const& map)
    : impl_base_type(len, map) {}
  impl_View(length_type len1, length_type len2, impl_map_type const& map)
    : impl_base_type(len1, len2, map) {}
  impl_View(length_type len1, length_type len2, length_type len3,
       impl_map_type const& map)
    : impl_base_type(len1, len2, len3, map) {}
  explicit impl_View(Block& blk) VSIP_NOTHROW
    : impl_base_type(blk) {}
  impl_View(impl_View const& v) VSIP_NOTHROW
    : impl_base_type(v.block()) {}

  View<T,Block>& impl_view() { return static_cast<View<T,Block>&>(*this); }
  View<T,Block> const& impl_view() const
    { return static_cast<View<T,Block> const&>(*this); }

  // [parview.vector.accessors]
  local_type local() const VSIP_NOTHROW
    { return vsip::impl::get_local_view(this->impl_view()); }
};

// specialize for element type std::complex<T>

template <template <typename,typename> class View,
	  typename Block, typename T, dimension_type Dim>
struct impl_View<View,Block,std::complex<T>,Dim>
  : impl::Const_of_view<View,Block>::view_type
{ 
  static const bool impl_is_nonconst_view = true;
  typedef View<std::complex<T>,Block> impl_vsip_view_type;
  typedef typename
    impl::Const_of_view<View,Block>::view_type::impl_const_view_type
      impl_const_view_type;
  typedef typename impl::Const_of_view<View,Block>::view_type impl_base_type;
  typedef typename Block::map_type impl_map_type;

  // Types for local views of distributed objects.
  typedef typename impl_base_type::impl_localblock_type impl_localblock_type;
  typedef View<std::complex<T>, impl_localblock_type> local_type;

  typedef impl::Component_block<Block,impl::Real_extractor> impl_rblock_type;
  typedef impl::Component_block<Block,impl::Imag_extractor> impl_iblock_type;
  typedef T impl_scalar_type;

  // complex
  typedef typename impl::Const_of_view<View,impl_rblock_type>::view_type
    const_realview_type;
  typedef typename impl::Const_of_view<View,impl_iblock_type>::view_type 
    const_imagview_type;
  typedef View<T,impl_rblock_type> realview_type;
  typedef View<T,impl_iblock_type> imagview_type;

  impl_View(length_type len, std::complex<T> const& value,
       impl_map_type const& map, impl::disambiguator_type)
    : impl_base_type(len, value, map) {}
  impl_View(length_type len1, length_type len2, std::complex<T> const& value,
       impl_map_type const& map, impl::disambiguator_type)
    : impl_base_type(len1, len2, value, map) {}
  impl_View(length_type len1, length_type len2, length_type len3,
      std::complex<T> const& value, impl_map_type const& map,
      impl::disambiguator_type)
    : impl_base_type(len1, len2, len3, value, map) {}
  impl_View(length_type len, impl_map_type const& map)
    : impl_base_type(len, map) {}
  impl_View(length_type len1, length_type len2, impl_map_type const& map)
    : impl_base_type(len1, len2, map) {}
  impl_View(length_type len1, length_type len2, length_type len3,
       impl_map_type const& map)
    : impl_base_type(len1, len2, len3, map) {}
  explicit impl_View(Block& blk) VSIP_NOTHROW
    : impl_base_type(blk) {}
  impl_View(impl_View const& v) VSIP_NOTHROW
    : impl_base_type(v.block()) {}
  const_realview_type real() const VSIP_THROW((std::bad_alloc))
    { return this->impl_const_view_type::real(); }
  realview_type real() VSIP_THROW((std::bad_alloc))
  {
    impl_rblock_type block(this->block());
    return realview_type(block);
  }
  const_imagview_type imag() const VSIP_THROW((std::bad_alloc))
    { return this->impl_const_view_type::imag(); }
  imagview_type imag() VSIP_THROW((std::bad_alloc))
  {
    impl_iblock_type block(this->block());
    return imagview_type(block);
  }

  View<std::complex<T>,Block>& impl_view()
    { return static_cast<View<std::complex<T>,Block>&>(*this); }
  View<std::complex<T>,Block> const& impl_view() const
    { return static_cast<View<std::complex<T>,Block> const&>(*this); }

  // [parview.vector.accessors]
  local_type local() const VSIP_NOTHROW
    { return vsip::impl::get_local_view(this->impl_view()); }
};

} // namespace vsip

#endif
