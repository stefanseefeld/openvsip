//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_view_view_hpp_
#define ovxx_view_view_hpp_

#include <vsip/support.hpp>
#include <ovxx/view/traits.hpp>
#include <ovxx/expr.hpp>
#include <ovxx/parallel/get_local_view.hpp>
#include <vsip/impl/complex_decl.hpp>

namespace ovxx
{
namespace detail
{
enum disambiguator_type { disambiguate};
} // namespace ovxx::detail

template <template <typename,typename> class V,
	  typename B, typename T = typename B::value_type,
	  dimension_type D = dim_of_view<V>::dim >
class const_View : detail::nonassignable//, ovxx::assert_proper_block<B>
{
protected:
  typedef V<T,B> impl_vsip_view_type;
  typedef const_View impl_const_view_type;
  typedef typename block_traits<B>::ptr_type block_ptr_type;

  typedef typename distributed_local_block<B>::type impl_localblock_type;
    
public:
  typedef V<T, impl_localblock_type> local_type;

  const_View(B *block, bool add_ref = true) : block_(block, add_ref) {}

  V<T, B> &view() { return static_cast<V<T, B>&>(*this);}
  V<T, B> const &view() const { return static_cast<V<T, B> const&>(*this);}

  // [view.vector.accessors]
  // [view.matrix.accessors]
  // [view.tensor.accessors]
  B &block() const VSIP_NOTHROW { return *this->block_;}
  length_type size() const VSIP_NOTHROW { return this->block_->size();}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return this->block_->size(D, d);}

  // [parview.vector.accessors]
  local_type local() const VSIP_NOTHROW { return get_local_view(this->view());}

private:
   block_ptr_type block_;
};

template <template <typename,typename> class V,
	  typename B, typename T, dimension_type D>
class const_View<V, B, complex<T>, D> : detail::nonassignable//, assert_proper_block<B>
{ 
protected:
  typedef V<complex<T>, B> impl_vsip_view_type;
  typedef const_View impl_const_view_type;
  typedef typename block_traits<B>::ptr_type block_ptr_type;

  typedef typename distributed_local_block<B>::type impl_localblock_type;
  typedef expr::Component<B, expr::op::RealC> impl_rblock_type;
  typedef expr::Component<B, expr::op::ImagC> impl_iblock_type;
  typedef T impl_scalar_type;

public:
  typedef V<complex<T>, impl_localblock_type> local_type;

  typedef V<T,impl_rblock_type> realview_type;
  typedef V<T,impl_iblock_type> imagview_type;
  typedef V<T,impl_rblock_type> const_realview_type;
  typedef V<T,impl_iblock_type> const_imagview_type;

  const_View(B *block, bool add_ref = true) : block_(block, add_ref) {}

  const_realview_type real() const VSIP_THROW((std::bad_alloc))
  {
    impl_rblock_type block(*this->block_);
    return const_realview_type(block);
  }
  const_imagview_type imag() const VSIP_THROW((std::bad_alloc))
  {
    impl_iblock_type block(*this->block_);
    return const_imagview_type(block);
  }

  V<complex<T>, B> &view() { return static_cast<V<complex<T>, B>&>(*this);}
  V<complex<T>, B> const &view() const
  { return static_cast<V<complex<T>, B> const&>(*this);}

  // [view.vector.accessors]
  // [view.matrix.accessors]
  // [view.tensor.accessors]
  B &block() const VSIP_NOTHROW { return *this->block_;}
  length_type size() const VSIP_NOTHROW { return this->block_->size();}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return this->block_->size(D, d);}

  // [parview.vector.accessors]
  local_type local() const VSIP_NOTHROW { return ovxx::get_local_view(this->view());}

private:
  block_ptr_type block_;
};

template <template <typename,typename> class V,
	  typename B, typename T = typename B::value_type,
	  dimension_type D = dim_of_view<V>::dim>
class View : public const_of_view<V, B>::type
{ 
protected:
  static const bool impl_is_nonconst_view = true;
  typedef V<T, B> impl_vsip_view_type;
  typedef typename const_of_view<V, B>::type base_type;
  typedef typename B::map_type map_type;
  typedef typename const_of_view<V, B>::type::impl_const_view_type
    impl_const_view_type;

  typedef typename base_type::impl_localblock_type impl_localblock_type;

public:
  typedef V<T, impl_localblock_type> local_type;

  View(length_type len, T const &value, map_type const &map, detail::disambiguator_type)
    : base_type(len, value, map) {}
  View(length_type len1, length_type len2, T const &value,
       map_type const &map, detail::disambiguator_type)
    : base_type(len1, len2, value, map) {}
  View(length_type len1, length_type len2, length_type len3,
       T const &value, map_type const &map, detail::disambiguator_type)
    : base_type(len1, len2, len3, value, map) {}
  View(length_type len, map_type const& map)
    : base_type(len, map) {}
  View(length_type len1, length_type len2, map_type const& map)
    : base_type(len1, len2, map) {}
  View(length_type len1, length_type len2, length_type len3, map_type const& map)
    : base_type(len1, len2, len3, map) {}
  explicit View(B &block) VSIP_NOTHROW : base_type(block) {}
  View(View const &v) VSIP_NOTHROW : base_type(v.block()) {}

  V<T,B> &view() { return static_cast<V<T, B>&>(*this);}
  V<T,B> const &view() const { return static_cast<V<T, B> const&>(*this);}

  // [parview.vector.accessors]
  local_type local() const VSIP_NOTHROW { return get_local_view(this->view());}
};

template <template <typename,typename> class V,
	  typename B, typename T, dimension_type D>
class View<V, B, complex<T>, D> : public const_of_view<V, B>::type
{ 
protected:
  static const bool impl_is_nonconst_view = true;
  typedef V<complex<T>, B> impl_vsip_view_type;
  typedef typename const_of_view<V, B>::type::impl_const_view_type
      impl_const_view_type;
  typedef typename const_of_view<V, B>::type base_type;
  typedef typename B::map_type map_type;

  typedef typename base_type::impl_localblock_type impl_localblock_type;

  typedef expr::Component<B, expr::op::RealC> impl_rblock_type;
  typedef expr::Component<B, expr::op::ImagC> impl_iblock_type;
  typedef T impl_scalar_type;

public:
  typedef V<complex<T>, impl_localblock_type> local_type;
  typedef V<T, impl_rblock_type> realview_type;
  typedef V<T, impl_iblock_type> imagview_type;
  typedef typename const_of_view<V, impl_rblock_type>::type const_realview_type;
  typedef typename const_of_view<V, impl_iblock_type>::type const_imagview_type;

  View(length_type len, complex<T> const &value, map_type const &map, detail::disambiguator_type)
    : base_type(len, value, map) {}
  View(length_type len1, length_type len2, complex<T> const &value,
       map_type const &map, detail::disambiguator_type)
    : base_type(len1, len2, value, map) {}
  View(length_type len1, length_type len2, length_type len3, complex<T> const &value, 
       map_type const &map, detail::disambiguator_type)
    : base_type(len1, len2, len3, value, map) {}
  View(length_type len, map_type const &map) : base_type(len, map) {}
  View(length_type len1, length_type len2, map_type const &map)
    : base_type(len1, len2, map) {}
  View(length_type len1, length_type len2, length_type len3, map_type const &map)
    : base_type(len1, len2, len3, map) {}
  explicit View(B &block) VSIP_NOTHROW : base_type(block) {}
  View(View const &v) VSIP_NOTHROW : base_type(v.block()) {}
  const_realview_type real() const VSIP_THROW((std::bad_alloc))
  { return this->impl_const_view_type::real();}
  realview_type real() VSIP_THROW((std::bad_alloc))
  {
    impl_rblock_type block(this->block());
    return realview_type(block);
  }
  const_imagview_type imag() const VSIP_THROW((std::bad_alloc))
  { return this->impl_const_view_type::imag();}
  imagview_type imag() VSIP_THROW((std::bad_alloc))
  {
    impl_iblock_type block(this->block());
    return imagview_type(block);
  }

  V<complex<T>, B> &view() { return static_cast<V<complex<T>, B>&>(*this);}
  V<complex<T>, B> const &view() const 
  { return static_cast<V<complex<T>, B> const&>(*this);}

  // [parview.vector.accessors]
  local_type local() const VSIP_NOTHROW
  { return get_local_view(this->view());}
};

} // namespace ovxx

#endif
