//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

//    Used for working with distributed data by replicating a copy locally
//    to each processor.
//     - Assign_local() transfers data between local and distributed views.
//     - Working_view_holder creates a local working view of an argument,
//       either replicating a distributed view to a local copy, or aliasing
//       a local view.

#ifndef VSIP_CORE_WORKING_VIEW_HPP
#define VSIP_CORE_WORKING_VIEW_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/parallel/services.hpp>
#include <vsip/core/static_assert.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/assign_local.hpp>


/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{

namespace impl
{

// Create a new view of type 'ViewT' that has the same dimensions as the
// existing view.

template <typename ViewT,
	  typename T,
	  typename Block>
ViewT
clone_view(const_Vector<T, Block> view)
{
  ViewT ret(view.size(0));
  return ret;
}

template <typename ViewT,
	  typename T,
	  typename Block>
ViewT
clone_view(const_Matrix<T, Block> view)
{
  ViewT ret(view.size(0), view.size(1));
  return ret;
}

template <typename ViewT,
	  typename T,
	  typename Block>
ViewT
clone_view(const_Tensor<T, Block> view)
{
  ViewT ret(view.size(0), view.size(1), view.size(2));
  return ret;
}



template <typename ViewT,
	  typename T,
	  typename BlockT,
	  typename MapT>
ViewT
clone_view(const_Vector<T, BlockT> view, MapT const& map)
{
  ViewT ret(view.size(0), map);
  return ret;
}

template <typename ViewT,
	  typename T,
	  typename BlockT,
	  typename MapT>
ViewT
clone_view(const_Matrix<T, BlockT> view, MapT const& map)
{
  ViewT ret(view.size(0), view.size(1), map);
  return ret;
}

template <typename ViewT,
	  typename T,
	  typename BlockT,
	  typename MapT>
ViewT
clone_view(const_Tensor<T, BlockT> view, MapT const& map)
{
  ViewT ret(view.size(0), view.size(1), view.size(2), map);
  return ret;
}

/***********************************************************************
  Definitions
***********************************************************************/

template <typename ViewT,
	  typename MapT = typename ViewT::block_type::map_type>
struct As_local_view
{
  static bool const is_copy = true;
  static dimension_type const dim = ViewT::dim;

  typedef typename ViewT::value_type                    value_type;
  typedef typename ViewT::block_type                    block_type;
  typedef typename get_block_layout<block_type>::order_type order_type;

  typedef Dense<dim, value_type, order_type, Local_map> r_block_type;

  typedef typename 
    conditional<Is_const_view_type<ViewT>::value,
      typename view_of<r_block_type>::type,
      typename view_of<r_block_type>::const_type
      >::type type;

  static type exec(ViewT view)
  {
    // The internal view needs to be non-const, even if the function
    // return type is const.
    typedef typename view_of<r_block_type>::type view_type;

    view_type ret(clone_view<view_type>(view));
    assign_local(ret, view);
    return ret;
  }
};

template <typename ViewT>
struct As_local_view<ViewT, Local_map>
{
  static bool const is_copy = false;
  typedef ViewT type;

  static type exec(ViewT view) { return view; }
};



template <typename       ViewT,
	  dimension_type Dim>
struct As_local_view<ViewT, Replicated_map<Dim> >
{
  static bool const is_copy = false;
  typedef typename ViewT::local_type type;

  static type exec(ViewT view) { return view.local(); }
};



template <template <typename, typename> class ViewT,
	  typename                            T,
	  typename                            Block>
typename As_local_view<ViewT<T, Block> >::type
convert_to_local(ViewT<T, Block> view)
{
  return As_local_view<ViewT<T, Block> >::exec(view);
}


template <typename ViewT,
	  bool     is_const = Is_const_view_type<ViewT>::value>
struct Working_view_holder
{
public:
  typedef typename As_local_view<ViewT>::type type;

public:
  Working_view_holder(ViewT v)
    : orig_view(v), view(convert_to_local(v))
  {}

  ~Working_view_holder()
  {
    if (As_local_view<ViewT>::is_copy)
    {
      assign_local(orig_view, view);
    }
  }

  // Member data.  This is intentionally public.
private:
  ViewT orig_view;

public:
  type  view;
};



template <typename ViewT>
struct Working_view_holder<ViewT, true>
{
public:
  typedef typename As_local_view<ViewT>::type type;

public:
  Working_view_holder(ViewT v)
    : view(convert_to_local(v))
  {}

  ~Working_view_holder() {}

  // Member data.  This is intentionally public.
public:
  type  view;
};

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_WORKING_VIEW_HPP
