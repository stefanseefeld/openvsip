//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_view_cast_hpp_
#define ovxx_view_cast_hpp_

#include <ovxx/support.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/view/traits.hpp>
//#include <ovxx/expr.hpp>

namespace ovxx
{
namespace detail
{

template <typename T>
struct cast_closure
{
  template <typename Operand>
  struct cast
  {
    typedef T result_type;

    static result_type apply(Operand op) { return static_cast<result_type>(op); }
    result_type operator()(Operand op) const { return apply(op); }
  };
};

template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            T1,
	  typename                            Block1>
struct view_cast
{
  typedef expr::Unary<cast_closure<T>::template cast,
		      Block1, true> const block_type;
  typedef typename ViewConversion<ViewT, T, block_type>::const_view_type
    view_type;

  static view_type cast(ViewT<T1, Block1> const& view)
  {
    block_type blk(view.block());
    return view_type(blk);
  }
};

template <typename                            T1,
	  template <typename, typename> class ViewT,
	  typename                            Block1>
struct view_cast<T1, ViewT, T1, Block1>
{
  typedef ViewT<T1, Block1> view_type;
  static view_type cast(ViewT<T1, Block1> const& v)
  {
    return v;
  }
};

} // namespace ovxx::detail

template <typename                            T,
	  template <typename, typename> class V,
	  typename                            T1,
	  typename                            B>
typename detail::view_cast<T, V, T1, B>::view_type
view_cast(V<T1, B> const &view)
{
  return detail::view_cast<T, V, T1, B>::cast(view);
}

} // namespace ovxx

#endif
