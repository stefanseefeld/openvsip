/* Copyright (c) 2005, 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/view_cast.hpp
    @author  Jules Bergmann
    @date    2005-06-15
    @brief   VSIPL++ Library: View cast class.

*/

#ifndef VSIP_CORE_VIEW_CAST_HPP
#define VSIP_CORE_VIEW_CAST_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/noncopyable.hpp>
#include <vsip/core/view_traits.hpp>
#include <vsip/core/expr/unary_block.hpp>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

// Cast operator 'Cast'.  Return type of cast is set through Cast_closuere.

template <typename T>
struct Cast_closure
{
  template <typename Operand>
  struct Cast
  {
    typedef T result_type;

    static char const* name() { return "cast"; }
    static result_type apply(Operand op) { return static_cast<result_type>(op); }
    result_type operator()(Operand op) const { return apply(op); }
  };
};



// Helper class to determine the return type of view_cast function
// and handle actual casting.

template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            T1,
	  typename                            Block1>
struct View_cast
{
  typedef expr::Unary<Cast_closure<T>::template Cast,
		      Block1, true> const block_type;

  typedef typename ViewConversion<ViewT, T, block_type>::const_view_type
    view_type;

  static view_type cast(ViewT<T1, Block1> const& view)
  {
    block_type blk(view.block());
    return view_type(blk);
  }
};



/// Specialization to avoid unnecessary cast when T == T1.

template <typename                            T1,
	  template <typename, typename> class ViewT,
	  typename                            Block1>
struct View_cast<T1, ViewT, T1, Block1>
{
  typedef ViewT<T1, Block1> view_type;

  static view_type cast(ViewT<T1, Block1> const& v)
  {
    return v;
  }
};



/***********************************************************************
  Definitions
***********************************************************************/

template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            T1,
	  typename                            Block1>
typename View_cast<T, ViewT, T1, Block1>::view_type
view_cast(ViewT<T1, Block1> const& view)
{
  return View_cast<T, ViewT, T1, Block1>::cast(view);
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_CAST_BLOCK_HPP
