/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/reductions/reductions.hpp
    @author  Jules Bergmann
    @date    2005-07-12
    @brief   VSIPL++ Library: Reduction functions.
	     [math.fns.reductions].

*/

#ifndef VSIP_CORE_REDUCTIONS_REDUCTIONS_HPP
#define VSIP_CORE_REDUCTIONS_REDUCTIONS_HPP

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <vsip/core/reductions/functors.hpp>
#include <vsip/core/parallel/services.hpp>
#include <vsip/core/dispatch.hpp>
#ifndef VSIP_IMPL_REF_IMPL
# include <vsip/opt/reductions/reductions.hpp>
#endif
#if VSIP_IMPL_HAVE_CVSIP
# include <vsip/core/cvsip/eval_reductions.hpp>
#endif

namespace vsip
{
namespace impl
{

template <template <typename> class ReduceT,
	  typename                  ViewT>
typename ReduceT<typename ViewT::value_type>::result_type
reduce(ViewT v)
{
  using namespace vsip_csl::dispatcher;

  typedef typename ViewT::value_type T;
  typedef typename ReduceT<T>::result_type result_type;
  typedef typename ViewT::block_type block_type;
  typedef typename Block_layout<typename ViewT::block_type>::order_type
    order_type;
  typedef Int_type<ViewT::dim> dim_type;

  result_type r;

#if VSIP_IMPL_REF_IMPL
  Evaluator<op::reduce<ReduceT>, be::cvsip, 
    void(result_type&, block_type const&, order_type, dim_type)>::
    exec(r, v.block(), order_type(), dim_type());
#else

  // Don't use the default dispatch list here, as that doesn't include the
  // 'parallel' backend. (The latter then uses the default list for local
  // dispatches.
  typedef Make_type_list<be::parallel,
                         be::cbe_sdk,
                         be::cvsip,
                         be::mercury_sal, 
                         be::generic>::type list_type;

  Dispatcher<op::reduce<ReduceT>, 
    void(result_type&, block_type const&, order_type, dim_type), list_type>::
    dispatch(r, v.block(), order_type(), dim_type());
#endif

  return r;
}

} // namespace vsip::impl

template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            BlockT>
typename impl::All_true<T>::result_type
alltrue(ViewT<T, BlockT> v)
{
  return impl::reduce<impl::All_true>(v);
}

template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            BlockT>
typename impl::Any_true<T>::result_type
anytrue(ViewT<T, BlockT> v)
{
  return impl::reduce<impl::Any_true>(v);
}

template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            BlockT>
typename impl::Mean_value<T>::result_type
meanval(ViewT<T, BlockT> v)
{
  return impl::reduce<impl::Mean_value>(v);
}

// Note: meansqval computes the mean of the magnitude square

template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            BlockT>
typename impl::Mean_magsq_value<T>::result_type
meansqval(ViewT<T, BlockT> v)
{
  return impl::reduce<impl::Mean_magsq_value>(v);
}

template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            BlockT>
typename impl::Sum_value<T>::result_type
sumval(ViewT<T, BlockT> v)
{
  return impl::reduce<impl::Sum_value>(v);
}

template <typename                            T,
	  template <typename, typename> class ViewT,
	  typename                            BlockT>
typename impl::Sum_sq_value<T>::result_type
sumsqval(ViewT<T, BlockT> v)
{
  return impl::reduce<impl::Sum_sq_value>(v);
}

} // namespace vsip

#endif
