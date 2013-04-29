//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_REDUCTIONS_REDUCTIONS_HPP
#define VSIP_CORE_REDUCTIONS_REDUCTIONS_HPP

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <vsip/dda.hpp>
#include <vsip/core/reductions/functors.hpp>
#include <vsip/core/parallel/services.hpp>
#include <vsip/core/dispatch.hpp>
#ifndef VSIP_IMPL_REF_IMPL
# include <vsip/opt/reductions/reductions.hpp>
#endif
#if VSIP_IMPL_HAVE_CVSIP
# include <vsip/core/cvsip/eval_reductions.hpp>
#endif
#include <vsip/opt/expr/redim_block.hpp>
#include <vsip/opt/expr/eval_dense.hpp>

namespace vsip
{
namespace impl
{

/// This helper class is needed because Redim_block cannot be instantiated
/// on a view with a dimension of one.  It handles cases where there is no need
/// to redimension the view, so dispatch is called directly.
template <template <typename> class ReduceT,
          typename ViewT,
          bool Possibly_redimensionable = false>
struct Reduction_dispatch_helper
{
  typedef typename ViewT::value_type T;
  typedef typename ReduceT<T>::result_type result_type;
  typedef typename ViewT::block_type block_type;
  typedef typename get_block_layout<block_type>::order_type order_type;
  typedef Int_type<ViewT::dim> dim_type;

  static result_type
  apply(ViewT& v)
  {
    using namespace vsip_csl::dispatcher;
    result_type r;

    // Don't use the default dispatch list here, as that doesn't include the
    // 'parallel' backend. (The latter then uses the default list for local
    // dispatches.
    typedef Make_type_list<
      be::parallel,
      be::cbe_sdk,
      be::cuda,
      be::cvsip,
      be::mercury_sal, 
      be::generic>::type list_type;

    Dispatcher<op::reduce<ReduceT>, 
      void(result_type&, block_type const&, order_type, dim_type), list_type>::
      dispatch(r, v.block(), order_type(), dim_type());

    return r;
  }
};


/// This handles the case where the input is either 2- or 3-D view that 
/// may be re-dimensioned to a 1-D view.  Because Redim_block requires
/// direct data access to all expression blocks, including elementwise
/// blocks that don't support dda, this specialization likewise cannot
/// handle expressions.
template <template <typename> class ReduceT,
          typename ViewT>
struct Reduction_dispatch_helper<ReduceT, ViewT, true>
{
  typedef typename ViewT::value_type T;
  typedef typename ReduceT<T>::result_type result_type;
  typedef typename ViewT::block_type block_type;
  typedef typename get_block_layout<block_type>::order_type order_type;
  typedef Int_type<ViewT::dim> dim_type;

  typedef Redim_block<block_type, ViewT::dim> new_block_type;
  typedef row1_type new_order_type;
  typedef Int_type<1> new_dim_type;

  static result_type
  apply(ViewT& v)
  {
    using namespace vsip_csl::dispatcher;
    result_type r;

    // Don't use the default dispatch list here, as that doesn't include the
    // 'parallel' backend. (The latter then uses the default list for local
    // dispatches.
    typedef Make_type_list<
      be::parallel,
      be::cbe_sdk,
      be::cuda,
      be::cvsip,
      be::mercury_sal, 
      be::generic>::type list_type;

    if (is_expr_dense(v.block()))
    {
      Dispatcher<op::reduce<ReduceT>, 
        void(result_type&, new_block_type const&, new_order_type, new_dim_type), list_type>::
        dispatch(r, new_block_type(const_cast<block_type&>(v.block())), new_order_type(), new_dim_type());
    }
    else
    {
      Dispatcher<op::reduce<ReduceT>, 
        void(result_type&, block_type const&, order_type, dim_type), list_type>::
      dispatch(r, v.block(), order_type(), dim_type());
    }

    return r;
  }
};



template <template <typename> class ReduceT,
	  typename                  ViewT>
typename ReduceT<typename ViewT::value_type>::result_type
reduce(ViewT v)
{
  using namespace vsip_csl::dispatcher;

  typedef typename ViewT::value_type T;
  typedef typename ReduceT<T>::result_type result_type;
  typedef typename ViewT::block_type block_type;
  typedef typename get_block_layout<block_type>::order_type order_type;
  typedef Int_type<ViewT::dim> dim_type;

  result_type r;

#if VSIP_IMPL_REF_IMPL
  Evaluator<op::reduce<ReduceT>, be::cvsip, 
    void(result_type&, block_type const&, order_type, dim_type)>::
    exec(r, v.block(), order_type(), dim_type());
#else

  // This optimization is only applicable to target platforms that provide a
  // backend that uses direct data access rather than redim_get/put().
  // This includes CUDA and Cell backends, but not x86 (as of August 2010).
#if defined(VSIP_IMPL_CBE_SDK) || defined(VSIP_IMPL_HAVE_CUDA)
  bool const redimensionable = 
    (ViewT::dim != 1) && (!is_expr_block<block_type>::value);
#else
  bool const redimensionable = false;
#endif

  r = Reduction_dispatch_helper<ReduceT, ViewT, redimensionable>::apply(v);
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
