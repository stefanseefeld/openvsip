//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_reductions_reductions_hpp_
#define ovxx_reductions_reductions_hpp_

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <vsip/dda.hpp>
#include <ovxx/reductions/functors.hpp>
#include <ovxx/parallel/service.hpp>
#include <ovxx/dispatch.hpp>
#include <ovxx/length.hpp>
#if OVXX_HAVE_CVSIP
# include <ovxx/cvsip/reductions.hpp>
#endif

// This value is a tradeoff based on the fact that floating point 
// values only have 24 bits of precision.  Adding lots of small
// values can affect the accuracy of the result and this limit (4 K)
// helps prevent these losses for very large reductions.
#define OVXX_MAX_SUMMATION_LENGTH  (1 << 12)

namespace ovxx
{
namespace reduction
{
/// This helper class is needed because Redim_block cannot be instantiated
/// on a view with a dimension of one.  It handles cases where there is no need
/// to redimension the view, so dispatch is called directly.
template <template <typename> class R,
          typename V,
          bool Possibly_redimensionable = false>
struct dispatcher
{
  typedef typename V::value_type T;
  typedef typename R<T>::result_type result_type;
  typedef typename V::block_type block_type;
  typedef typename get_block_layout<block_type>::order_type order_type;
  typedef integral_constant<dimension_type, V::dim> dim_type;

  static result_type
  apply(V &v)
  {
    using namespace ovxx::dispatcher;
    result_type r;

    // Don't use the default dispatch list here, as that doesn't include the
    // 'parallel' backend. (The latter then uses the default list for local
    // dispatches.
    typedef make_type_list<
      be::parallel,
      be::cuda,
      be::cvsip,
      be::generic>::type list_type;

    Dispatcher<op::reduce<R>, 
      void(result_type&, block_type const&, order_type, dim_type), list_type>::
      dispatch(r, v.block(), order_type(), dim_type());

    return r;
  }
};
// FIXME: Reimplement block redimension
#if 0
/// This handles the case where the input is either 2- or 3-D view that 
/// may be re-dimensioned to a 1-D view.  Because Redim_block requires
/// direct data access to all expression blocks, including elementwise
/// blocks that don't support dda, this specialization likewise cannot
/// handle expressions.
template <template <typename> class R,
          typename V>
struct dispatcher<R, V, true>
{
  typedef typename V::value_type T;
  typedef typename R<T>::result_type result_type;
  typedef typename V::block_type block_type;
  typedef typename get_block_layout<block_type>::order_type order_type;
  typedef integral_constant<dimension_type, V::dim> dim_type;

  typedef vsip::impl::Redim_block<block_type, V::dim> new_block_type;
  typedef row1_type new_order_type;
  typedef integral_constant<dimension_type, 1> new_dim_type;

  static result_type
  apply(V &v)
  {
    using namespace ovxx::dispatcher;
    result_type r;

    // Don't use the default dispatch list here, as that doesn't include the
    // 'parallel' backend. (The latter then uses the default list for local
    // dispatches.
    typedef make_type_list<
      be::parallel,
      be::cuda,
      be::cvsip,
      be::generic>::type list_type;

    if (is_expr_dense(v.block()))
    {
      Dispatcher<op::reduce<R>, 
        void(result_type&, new_block_type const&, new_order_type, new_dim_type), list_type>::
        dispatch(r, new_block_type(const_cast<block_type&>(v.block())), new_order_type(), new_dim_type());
    }
    else
    {
      Dispatcher<op::reduce<R>, 
        void(result_type&, block_type const&, order_type, dim_type), list_type>::
      dispatch(r, v.block(), order_type(), dim_type());
    }

    return r;
  }
};
#endif

// Is this reduction a summation?
template <template <typename> class R>
struct is_summation { static bool const value = false;};

template<> struct is_summation<Sum_value> { static bool const value = true;};
template<> struct is_summation<Sum_sq_value> { static bool const value = true;};
template<> struct is_summation<Mean_value> { static bool const value = true;};
template<> struct is_summation<Mean_magsq_value> { static bool const value = true;};

// Are intermediate values squared before summing?
template <template <typename> class R>
struct is_sum_squared_based { static bool const value = false;};

template<> struct is_sum_squared_based<Sum_sq_value> { static bool const value = true;};
template<> struct is_sum_squared_based<Mean_magsq_value> { static bool const value = true;};

// Is the result averaged after summing?
template <template <typename> class R>
struct is_mean_based { static bool const value = false;};

template<> struct is_mean_based<Mean_value> { static bool const value = true;};
template<> struct is_mean_based<Mean_magsq_value> { static bool const value = true;};

// Get either only the real part, or both real and imaginary parts, depending
// on the destination value type.
template <typename T1, typename T2>
struct extract
{
  static void apply(T1 const& src, T2& dst) { dst = src;}
};

template <typename T1, typename T2>
struct extract<complex<T1>, T2>
{
  static void apply(complex<T1> const& src, T2& dst) { dst = src.real();}
};

template <template <typename> class R, typename T>
T partial_sum(T const *data, std::ptrdiff_t offset, length_type len)
{
  T result = T();
  data += offset;
  if (is_sum_squared_based<R>::value)
    for (index_type i = 0; i < len; ++i)
      result += data[i]*data[i];
  else
    for (index_type i = 0; i < len; ++i)
      result += data[i];
  return result;
}

template <template <typename> class R, typename T>
complex<T> partial_sum(std::pair<T const*, T const*> const &data, std::ptrdiff_t offset, length_type len)
{
  complex<T> result = complex<T>();
  if (is_sum_squared_based<R>::value)
    for (index_type i = 0; i < len; ++i)
    {
      complex<T> val = 
        complex<T>(data.first[i + offset], data.second[i + offset]);
      result += val*val;
    }
  else
    for (index_type i = 0; i < len; ++i)
      result += complex<T>(data.first[i + offset], data.second[i + offset]);

  return result;
}

} // namespace ovxx::reduction

template <template <typename> class R, typename V>
typename R<typename V::value_type>::result_type
reduce(V view)
{
  using namespace ovxx::dispatcher;

  typedef typename V::value_type T;
  typedef typename R<T>::result_type result_type;
  typedef typename V::block_type block_type;
  typedef typename get_block_layout<block_type>::order_type order_type;
  typedef integral_constant<dimension_type, V::dim> dim_type;

  result_type r;

  // This optimization is only applicable to target platforms that provide a
  // backend that uses direct data access rather than redim_get/put().
  // This includes CUDA and Cell backends, but not x86 (as of August 2010).
#if defined(VSIP_IMPL_CBE_SDK) || defined(VSIP_IMPL_HAVE_CUDA)
  bool const redimensionable = 
    (V::dim != 1) && (!is_expr_block<block_type>::value);
#else
  bool const redimensionable = false;
#endif

  r = reduction::dispatcher<R, V, redimensionable>::apply(view);

  return r;
}

namespace dispatcher
{

template<template <typename> class R>
struct List<op::reduce<R> >
{
  typedef make_type_list<be::user,
			 be::cuda,
			 be::cvsip,
			 be::generic>::type type;
};

template <template <typename> class R,
	  typename T, typename B>
struct Evaluator<op::reduce<R>, be::generic, 
  void(T&, B const &, row1_type, integral_constant<dimension_type, 1>)>
{
  static char const* name() { return "generic";}

  static bool const ct_valid = true;
  static bool rt_valid(T&, B const&, row1_type, integral_constant<dimension_type, 1>)
  { return true;}

  static void exec(T& r, B const& a, row1_type, integral_constant<dimension_type, 1>)
  {
    using namespace reduction;
    length_type length = a.size(1, 0);
    length_type const max_length = OVXX_MAX_SUMMATION_LENGTH;

    // Anything under the maximum length, not a summation (min/max)
    // or not dealing with single-precision floating point is 
    // handled in the usual way.
    if (!is_summation<R>::value ||
        !is_same<typename scalar_of<T>::type, float>::value ||
        (length < max_length))
    {
      typedef typename B::value_type V;
      typename R<V>::accum_type state = R<V>::initial();

      PRAGMA_IVDEP
        for (index_type i=0; i<length; ++i)
        {
          state = R<V>::update(state, a.get(i));
          if (R<V>::done(state)) break;
        }
      r = R<V>::value(state, length);
    }
    else
    {
      // Special handling is provided for very large summations by breaking
      // them into smaller blocks and saving intermediate results.  This avoids
      // accuracy issues related to round-off error when single-precision
      // is used.
      length_type const subblocks = length % max_length ?
        length / max_length + 1 : length / max_length;
      length_type const last_subblock_size = length % max_length ?
        length % max_length : max_length;

      vsip::dda::Data<B, vsip::dda::in> data(a);

      typename B::value_type result = 0;
      index_type sb = 0;
      for (; sb < subblocks - 1; ++sb)
      {
        result += partial_sum<R>(data.ptr(), sb * max_length, max_length);
      }
      result += partial_sum<R>(data.ptr(), sb * max_length, last_subblock_size);

      // Partial sums are accumlated in the view's natural type, even
      // for magnitude-of-squared values where the imaginary part is zero.
      // The result type is real in those cases, so we extract the desired
      // part depending on the type being returned.
      extract<typename B::value_type, T>::apply(result, r);

      if (is_mean_based<R>::value)
        r /= length;
    }
  }
};

template <template <typename> class R, 
	  typename T, typename B,
	  dimension_type D1, dimension_type D2>
struct Evaluator<op::reduce<R>, be::generic,
  void(T&, B const&, tuple<D1, D2>, integral_constant<dimension_type, 2>)>
{
  static char const* name() { return "generic";}

  static bool const ct_valid = true;
  static bool rt_valid(T&, B const&, tuple<D1, D2>, integral_constant<dimension_type, 2>)
  { return true;}

  static void exec(T& r, B const &a, tuple<D1, D2>, integral_constant<dimension_type, 2>)
  {
    typedef typename B::value_type V;
    typename R<T>::accum_type state = R<V>::initial();

    Length<2> length = extent<2>(a);
    Index<2> i;
    for (i[D1]=0; i[D1] != length[D1]; ++i[D1])
      PRAGMA_IVDEP
      for (i[D2]=0; i[D2] != length[D2]; ++i[D2])
      {
	state = R<V>::update(state, a.get(i[0], i[1]));
	if (R<V>::done(state)) break;
      }

    r = R<V>::value(state, total_size(length));
  }
};

template <template <typename> class R,
	  typename T, typename B,
	  dimension_type D1, dimension_type D2, dimension_type D3>
struct Evaluator<op::reduce<R>, be::generic,
  void(T&, B const&, tuple<D1, D2, D3>, integral_constant<dimension_type, 3>)>
{
  static char const* name() { return "generic";}

  static bool const ct_valid = true;
  static bool rt_valid(T&, B const&, tuple<D1, D2, D3>, integral_constant<dimension_type, 3>)
  { return true;}

  static void exec(T& r, B const &a, tuple<D1, D2, D3>, integral_constant<dimension_type, 3>)
  {
    typedef typename B::value_type V;
    typename R<T>::accum_type state = R<V>::initial();

    Length<3> length = extent<3>(a);
    Index<3> i;

    for (i[D1] = 0; i[D1] != length[D1]; ++i[D1])
      for (i[D2] = 0; i[D2] != length[D2]; ++i[D2])
	for (i[D3] = 0; i[D3] != length[D3]; ++i[D3])
	{
	  state = R<V>::update(state, a.get(i[0], i[1], i[2]));
	  if (R<V>::done(state)) break;
	}

    r = R<V>::value(state, total_size(length));
  }
};

} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
