/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/parallel/foreach.hpp
    @author  Jules Bergmann
    @date    2005-06-08
    @brief   VSIPL++ Library: Parallel foreach.

*/

#ifndef VSIP_OPT_PARALLEL_FOREACH_HPP
#define VSIP_OPT_PARALLEL_FOREACH_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/parallel/services.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <vsip/core/parallel/block.hpp>
#include <vsip/core/parallel/util.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{

namespace impl
{

template <dimension_type Dim,
	  typename       Order>
struct subview;

template <>
struct subview<2, tuple<0, 1, 2> >
{
  template <typename T,
	    typename BlockT>
  static
  typename Matrix<T, BlockT>::row_type
  vector(Matrix<T, BlockT> view, index_type i)
  {
    return view.row(i);
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Matrix<T, BlockT>::row_type
  vector(const_Matrix<T, BlockT> view, index_type i)
  {
    return view.row(i);
  }
};



template <>
struct subview<2, tuple<1, 0, 2> >
{
  template <typename T,
	    typename BlockT>
  static
  typename Matrix<T, BlockT>::col_type
  vector(Matrix<T, BlockT> view, index_type i)
  {
    return view.col(i);
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Matrix<T, BlockT>::col_type
  vector(const_Matrix<T, BlockT> view, index_type i)
  {
    return view.col(i);
  }
};



template <>
struct subview<3, tuple<0, 1, 2> >
{
  template <typename T,
	    typename BlockT>
  static
  typename Tensor<T, BlockT>::template subvector<0, 1>::impl_type
  vector(Tensor<T, BlockT> view, index_type i, index_type j)
  {
    return view(i, j, whole_domain);
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template subvector<0, 1>::impl_type
  vector(const_Tensor<T, BlockT> view, index_type i, index_type j)
  {
    return view(i, j, whole_domain);
  }

  template <typename T,
	    typename BlockT>
  static
  typename Tensor<T, BlockT>::template submatrix<0>::impl_type
  matrix(Tensor<T, BlockT> view, index_type i)
  {
    return view(i, whole_domain, whole_domain);
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template submatrix<0>::impl_type
  matrix(const_Tensor<T, BlockT> view, index_type i)
  {
    return view(i, whole_domain, whole_domain);
  }

  static index_type first (index_type i, index_type) { return i; }
  static index_type second(index_type, index_type j) { return j; }
};



template <>
struct subview<3, tuple<0, 2, 1> >
{
  template <typename T,
	    typename BlockT>
  static
  typename Tensor<T, BlockT>::template subvector<0, 2>::impl_type
  vector(Tensor<T, BlockT> view, index_type i, index_type j)
  {
    return view(i, whole_domain, j);
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template subvector<0, 2>::impl_type
  vector(const_Tensor<T, BlockT> view, index_type i, index_type j)
  {
    return view(i, whole_domain, j);
  }

  template <typename T,
	    typename BlockT>
  static
  typename Tensor<T, BlockT>::template submatrix<0>::impl_type
  matrix(Tensor<T, BlockT> view, index_type i)
  {
    assert(0);
    // return view(i, whole_domain, whole_domain).transpose();
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template submatrix<0>::impl_type
  matrix(const_Tensor<T, BlockT> view, index_type i)
  {
    assert(0);
    // return view(i, whole_domain, whole_domain).transpose();
  }

  static index_type first (index_type i, index_type) { return i; }
  static index_type second(index_type, index_type j) { return j; }
};



template <>
struct subview<3, tuple<1, 0, 2> >
{
  template <typename T,
	    typename BlockT>
  static
  typename Tensor<T, BlockT>::template subvector<0, 1>::impl_type
  vector(Tensor<T, BlockT> view, index_type i, index_type j)
  {
    return view(j, i, whole_domain);
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template subvector<0, 1>::impl_type
  vector(const_Tensor<T, BlockT> view, index_type i, index_type j)
  {
    return view(j, i, whole_domain);
  }

  template <typename T,
	    typename BlockT>
  static
  typename Tensor<T, BlockT>::template submatrix<1>::impl_type
  matrix(Tensor<T, BlockT> view, index_type i)
  {
    return view(whole_domain, i, whole_domain);
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template submatrix<1>::impl_type
  matrix(const_Tensor<T, BlockT> view, index_type i)
  {
    return view(whole_domain, i, whole_domain);
  }

  static index_type first (index_type, index_type j) { return j; }
  static index_type second(index_type i, index_type) { return i; }
};



template <>
struct subview<3, tuple<1, 2, 0> >
{
  template <typename T,
	    typename BlockT>
  static
  typename Tensor<T, BlockT>::template subvector<0, 2>::impl_type
  vector(Tensor<T, BlockT> view, index_type i, index_type j)
  {
    return view(j, whole_domain, i);
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template subvector<0, 2>::impl_type
  vector(const_Tensor<T, BlockT> view, index_type i, index_type j)
  {
    return view(j, whole_domain, i);
  }

  template <typename T,
	    typename BlockT>
  static
  typename Tensor<T, BlockT>::template submatrix<1>::impl_type
  matrix(Tensor<T, BlockT> view, index_type i)
  {
    return view(whole_domain, whole_domain, i);
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template submatrix<1>::impl_type
  matrix(const_Tensor<T, BlockT> view, index_type i)
  {
    return view(whole_domain, whole_domain, i);
  }

  static index_type first (index_type, index_type j) { return j; }
  static index_type second(index_type i, index_type) { return i; }
};



template <>
struct subview<3, tuple<2, 0, 1> >
{
  template <typename T,
	    typename BlockT>
  static
  typename Tensor<T, BlockT>::template subvector<1, 2>::impl_type
  vector(Tensor<T, BlockT> view, index_type i, index_type j)
  {
    return view(whole_domain, i, j);
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template subvector<1, 2>::impl_type
  vector(const_Tensor<T, BlockT> view, index_type i, index_type j)
  {
    return view(whole_domain, i, j);
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template submatrix<2>::impl_type
  matrix(Tensor<T, BlockT> view, index_type i)
  {
    assert(0);
    // return view(whole_domain, i, whole_domain).transpose();
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template submatrix<2>::impl_type
  matrix(const_Tensor<T, BlockT> view, index_type i)
  {
    assert(0);
    // return view(whole_domain, i, whole_domain).transpose();
  }

  static index_type first (index_type i, index_type) { return i; }
  static index_type second(index_type, index_type j) { return j; }
};



template <>
struct subview<3, tuple<2, 1, 0> >
{
  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template subvector<1, 2>::impl_type
  vector(Tensor<T, BlockT> view, index_type i, index_type j)
  {
    return view(whole_domain, j, i);
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template subvector<1, 2>::impl_type
  vector(const_Tensor<T, BlockT> view, index_type i, index_type j)
  {
    return view(whole_domain, j, i);
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template submatrix<2>::impl_type
  matrix(Tensor<T, BlockT> view, index_type i)
  {
    assert(0);
    // return view(whole_domain, whole_domain, i).transpose();
  }

  template <typename T,
	    typename BlockT>
  static
  typename const_Tensor<T, BlockT>::template submatrix<2>::impl_type
  matrix(const_Tensor<T, BlockT> view, index_type i)
  {
    assert(0);
    // return view(whole_domain, whole_domain, i).transpose();
  }
  
  static index_type first (index_type, index_type j) { return j; }
  static index_type second(index_type i, index_type) { return i; }
};



template <dimension_type Dim,
	  typename       Order,
	  typename       InView,
	  typename       OutView,
	  typename       FuncT>
struct Foreach_vector;



template <typename Order,
	  typename InView,
	  typename OutView,
	  typename FuncT>
struct Foreach_vector<2, Order, InView, OutView, FuncT>
{
  static void exec(
    FuncT&  fcn,
    InView  in,
    OutView out)
  {
    typedef typename OutView::block_type::map_type map_t;

    static dimension_type const Dim0 = Order::impl_dim0;
    static dimension_type const Dim1 = Order::impl_dim1;
    // static dimension_type const Dim2 = Order::impl_dim2;

    map_t const& map = out.block().map();
    Domain<2>    dom = global_domain(out);

    if (map.num_subblocks(Dim1) != 1)
    {
      VSIP_IMPL_THROW(impl::unimplemented(
        "foreach_vector requires the dimension being processed to be undistributed"));
    }

    if (Is_par_same_map<2, map_t, typename InView::block_type>
	::value(map, in.block()))
    {
      typename InView::local_type  l_in  = get_local_view(in);
      typename OutView::local_type l_out = get_local_view(out);

      for (index_type i=0; i<l_out.size(Dim0); ++i)
      {
	index_type global_i = dom[Dim0].impl_nth(i);
	  
	fcn(subview<2, Order>::vector(l_in, i),
	    subview<2, Order>::vector(l_out, i),
	    Index<1>(global_i));
      }
    }
    else
    {
      typedef typename InView::value_type             value_type;
      typedef typename get_block_layout<typename InView::block_type>::order_type
	                                              order_type;
      typedef Dense<2, value_type, order_type, map_t> block_type;

      Matrix<value_type, block_type> in_copy(in.size(0), in.size(1), map);

      // Rearrange data.
      in_copy = in;

      // Force view to be const.
      const_Matrix<value_type, block_type> in_const = in_copy;

      typename InView::local_type  l_in  = get_local_view(in_const);
      typename OutView::local_type l_out = get_local_view(out);

      for (index_type i=0; i<l_out.size(Dim0); ++i)
      {
	index_type global_i = dom[Dim0].impl_nth(i);
	  
	fcn(subview<2, Order>::vector(l_in, i),
	    subview<2, Order>::vector(l_out, i),
	    Index<1>(global_i));
      }
    }
  }
};



template <typename Order,
	  typename InView,
	  typename OutView,
	  typename FuncT>
struct Foreach_vector<3, Order, InView, OutView, FuncT>
{
  static void exec(
    FuncT&  fcn,
    InView  in,
    OutView out)
  {
    typedef typename OutView::block_type::map_type map_t;

    static dimension_type const Dim0 = Order::impl_dim0;
    static dimension_type const Dim1 = Order::impl_dim1;
    static dimension_type const Dim2 = Order::impl_dim2;

    map_t const& map = out.block().map();
    Domain<3>    dom = global_domain(out);

    if (map.num_subblocks(Dim2) != 1)
    {
      VSIP_IMPL_THROW(impl::unimplemented(
        "foreach_vector requires the dimension being processed to be undistributed"));
    }

    if (Is_par_same_map<3, map_t, typename InView::block_type>
	::value(map, in.block()))
    {
      typename InView::local_type  l_in  = get_local_view(in);
      typename OutView::local_type l_out = get_local_view(out);

      for (index_type i=0; i<l_out.size(Dim0); ++i)
	for (index_type j=0; j<l_out.size(Dim1); ++j)
	{
	  index_type global_i = dom[Dim0].impl_nth(i);
	  index_type global_j = dom[Dim1].impl_nth(j);
	  
	  fcn(subview<3, Order>::vector(l_in, i, j),
	      subview<3, Order>::vector(l_out, i, j),
	      subview<3, Order>::first(global_i, global_j),
	      subview<3, Order>::second(global_i, global_j));
	}
    }
    else
    {
      typedef typename InView::value_type             value_type;
      typedef typename get_block_layout<typename InView::block_type>::order_type
	                                              order_type;
      typedef Dense<3, value_type, order_type, map_t> block_type;

      Tensor<value_type, block_type> in_copy(in.size(0),in.size(1),in.size(2),
					     map);

      // Rearrange data.
      in_copy = in;

      // Force view to be const.
      const_Tensor<value_type, block_type> in_const = in_copy;

      typename InView::local_type  l_in  = get_local_view(in_const);
      typename OutView::local_type l_out = get_local_view(out);

      for (index_type i=0; i<l_out.size(Dim0); ++i)
	for (index_type j=0; j<l_out.size(Dim1); ++j)
	{
	  index_type global_i = dom[Dim0].impl_nth(i);
	  index_type global_j = dom[Dim1].impl_nth(j);
	  
	  fcn(subview<3, Order>::vector(l_in, i, j),
	      subview<3, Order>::vector(l_out, i, j),
	      subview<3, Order>::first(global_i, global_j),
	      subview<3, Order>::second(global_i, global_j));
	}
    }
  }
};

template <dimension_type Dim1, dimension_type Dim>
struct Foreach_order;

template <> struct Foreach_order<3, 0> { typedef tuple<1, 2, 0> type; };
template <> struct Foreach_order<3, 1> { typedef tuple<0, 2, 1> type; };
template <> struct Foreach_order<3, 2> { typedef tuple<0, 1, 2> type; };

template <> struct Foreach_order<2, 0> { typedef tuple<1, 0, 2> type; };
template <> struct Foreach_order<2, 1> { typedef tuple<0, 1, 2> type; };

} // namespace vsip::impl



template <dimension_type                      Dim,
	  template <typename, typename> class View1,
	  template <typename, typename> class View2,
	  typename                            T1,
	  typename                            T2,
	  typename                            Block1,
	  typename                            Block2,
	  typename                            FuncT>
void
foreach_vector(
  FuncT&            fcn,
  View1<T1, Block1> in,
  View2<T2, Block2> out)
{
  dimension_type const dim = View1<T1, Block1>::dim;

  impl::Foreach_vector<dim, typename impl::Foreach_order<dim, Dim>::type,
    View1<T1, Block1>,
    View2<T2, Block2>,
    FuncT>::exec(fcn, in, out);
}



template <dimension_type                      Dim,
	  template <typename, typename> class View1,
	  typename                            T1,
	  typename                            Block1,
	  typename                            FuncT>
void
foreach_vector(
  FuncT&            fcn,
  View1<T1, Block1> inout)
{
  dimension_type const dim = View1<T1, Block1>::dim;

  impl::Foreach_vector<dim, typename impl::Foreach_order<dim, Dim>::type,
    View1<T1, Block1>,
    View1<T1, Block1>,
    FuncT>::exec(fcn, inout, inout);
}



template <typename                            TraverseOrder,
	  template <typename, typename> class View1,
	  typename                            T1,
	  typename                            Block1,
	  typename                            FuncT>
void
foreach_vector(
  FuncT&            fcn,
  View1<T1, Block1> inout)
{
  dimension_type const dim = View1<T1, Block1>::dim;

  impl::Foreach_vector<dim, TraverseOrder,
    View1<T1, Block1>,
    View1<T1, Block1>,
    FuncT>::exec(fcn, inout, inout);
}

} // namespace vsip

#endif // VSIP_IMPL_PAR_FOREACH_HPP
