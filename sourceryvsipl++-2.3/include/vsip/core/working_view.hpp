/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/working-view.hpp
    @author  Jules Bergmann
    @date    2005-12-27
    @brief   VSIPL++ Library: Utilities for local working views.

    Used for working with distributed data by replicating a copy locally
    to each processor.
     - Assign_local() transfers data between local and distributed views.
     - Working_view_holder creates a local working view of an argument,
       either replicating a distributed view to a local copy, or aliasing
       a local view.
*/

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



// Helper class for assigning between local and distributed views.

template <typename View1,
	  typename View2,
	  bool     IsLocal1
	           = Is_local_map<typename View1::block_type::map_type>::value,
	  bool     IsLocal2
	           = Is_local_map<typename View2::block_type::map_type>::value>
struct Assign_local {};

// Local to local case.  Both views are local, just copy the data.

template <typename View1,
	  typename View2>
struct Assign_local<View1, View2, true, true>
{
  static void exec(View1 dst, View2 src)
  {
    dst = src;
  }
};

// Global to local case.  Destination is local, so copy source into
// replicated view, then copy from local view.

template <typename View1,
	  typename View2>
struct Assign_local<View1, View2, true, false>
{
  static void exec(View1 dst, View2 src)
  {
    dimension_type const dim = View1::dim;

    typedef typename View1::value_type                     T;
    typedef typename View1::block_type                     block1_type;
    typedef Global_map<dim>                                map_type;
    typedef typename Block_layout<block1_type>::order_type order_type;
    typedef Dense<dim, T, order_type, map_type>            block_type;
    typedef typename View_of_dim<dim, T, block_type>::type view_type;

    view_type view(clone_view<view_type>(dst));

    view = src;
    dst  = view.local();
  }
};

// Local to global case.  Source is local, so copy source into
// replicated view, then copy to destination view.

template <typename View1,
	  typename View2>
struct Assign_local<View1, View2, false, true>
{
  static void exec(View1 dst, View2 src)
  {
    dimension_type const dim = View1::dim;

    typedef typename View1::value_type                     T;
    typedef typename View1::block_type                     block2_type;
    typedef Global_map<dim>                                map_type;
    typedef typename Block_layout<block2_type>::order_type order_type;
    typedef Dense<dim, T, order_type, map_type>            block_type;
    typedef typename View_of_dim<dim, T, block_type>::type view_type;

    view_type view(clone_view<view_type>(dst));

    view.local() = src;
    dst          = view;
  }
};



/// Guarded Assign_local

template <bool     PerformAssignment,
	  typename ViewT1,
	  typename ViewT2>
struct Assign_local_if
{
  static void exec(ViewT1 dst, ViewT2 src)
  {
    Assign_local<ViewT1, ViewT2>::exec(dst, src);
  }
};



template <typename ViewT1,
	  typename ViewT2>
struct Assign_local_if<false, ViewT1, ViewT2>
{
  static void exec(ViewT1, ViewT2)
  { /* no assignment */ }
};



/// Assign between local and distributed views.

template <typename ViewT1,
	  typename ViewT2>
void assign_local(
  ViewT1 dst,
  ViewT2 src)
{
  VSIP_IMPL_STATIC_ASSERT((Is_view_type<ViewT1>::value));
  VSIP_IMPL_STATIC_ASSERT((Is_view_type<ViewT2>::value));
  VSIP_IMPL_STATIC_ASSERT(ViewT1::dim == ViewT2::dim);

  Assign_local<ViewT1, ViewT2>
    ::exec(dst, src);
}



/// Guarded assign between local and distributed views.

template <bool     PerformAssignment,
	  typename ViewT1,
	  typename ViewT2>
void assign_local_if(
  ViewT1 dst,
  ViewT2 src)
{
  VSIP_IMPL_STATIC_ASSERT((Is_view_type<ViewT1>::value));
  VSIP_IMPL_STATIC_ASSERT((Is_view_type<ViewT2>::value));
  VSIP_IMPL_STATIC_ASSERT(ViewT1::dim == ViewT2::dim);

  Assign_local_if<PerformAssignment, ViewT1, ViewT2>::exec(dst, src);
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
  typedef typename Block_layout<block_type>::order_type order_type;

  typedef Dense<dim, value_type, order_type, Local_map> r_block_type;

  typedef typename 
    ITE_Type<Is_const_view_type<ViewT>::value,
      As_type<typename View_of_dim<dim, value_type, r_block_type>::type>,
      As_type<typename View_of_dim<dim, value_type, r_block_type>::const_type>
      >::type type;

  static type exec(ViewT view)
  {
    // The internal view needs to be non-const, even if the function
    // return type is const.
    typedef typename
      View_of_dim<dim, value_type, r_block_type>::type view_type;

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
struct As_local_view<ViewT, Global_map<Dim> >
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
