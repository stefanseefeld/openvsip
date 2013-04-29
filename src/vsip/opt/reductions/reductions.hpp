/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_OPT_REDUCTIONS_REDUCTIONS_HPP
#define VSIP_OPT_REDUCTIONS_REDUCTIONS_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/dda.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <vsip/core/reductions/functors.hpp>
#include <vsip/core/parallel/services.hpp>
#include <vsip/opt/dispatch.hpp>
#ifdef VSIP_IMPL_CBE_SDK
# include <vsip/opt/cbe/ppu/reductions.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_SAL
# include <vsip/opt/sal/eval_reductions.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_CUDA
# include <vsip/opt/cuda/reductions.hpp>
#endif


// This value is a tradeoff based on the fact that floating point 
// values only have 24 bits of precision.  Adding lots of small
// values can affect the accuracy of the result and this limit (4 K)
// helps prevent these losses for very large reductions.
#define VSIP_IMPL_MAX_SUMMATION_LENGTH  (1 << 12)


namespace vsip
{
namespace impl
{
template <template <typename> class ReduceT,
          typename                  T,
	  typename                  Block,
	  typename                  OrderT,
	  int                       Dim>
struct Par_reduction_eval_base
{
  static bool const ct_valid = 
    !impl::Is_local_map<typename Block::map_type>::value &&
    impl::Reduction_supported<ReduceT<typename Block::value_type>::rtype,
			      typename Block::value_type>::value;

  static bool rt_valid(T&, Block const&, OrderT, impl::Int_type<Dim>)
  { return true; }
};


// Is this reduction a summation?
template <template <typename> class ReduceT>
struct is_summation { static bool const value = false; };

template<> struct is_summation<Sum_value> { static bool const value = true; };
template<> struct is_summation<Sum_sq_value> { static bool const value = true; };
template<> struct is_summation<Mean_value> { static bool const value = true; };
template<> struct is_summation<Mean_magsq_value> { static bool const value = true; };

// Are intermediate values squared before summing?
template <template <typename> class ReduceT>
struct is_sum_squared_based { static bool const value = false; };

template<> struct is_sum_squared_based<Sum_sq_value> { static bool const value = true; };
template<> struct is_sum_squared_based<Mean_magsq_value> { static bool const value = true; };

// Is the result averaged after summing?
template <template <typename> class ReduceT>
struct is_mean_based { static bool const value = false; };

template<> struct is_mean_based<Mean_value> { static bool const value = true; };
template<> struct is_mean_based<Mean_magsq_value> { static bool const value = true; };


// Get either only the real part, or both real and imaginary parts, depending
// on the destination value type.
template <typename T1, typename T2>
struct extract
{
  static void apply(T1 const& src, T2& dst) { dst = src; }
};

template <typename T1, typename T2>
struct extract<complex<T1>, T2>
{
  static void apply(complex<T1> const& src, T2& dst) { dst = src.real(); }
};



template <template <typename> class ReduceT,
          typename T>
T partial_sum(T const *A, ptrdiff_t offset, length_type len)
{
  T result = T();
  A += offset;
  if (is_sum_squared_based<ReduceT>::value)
    for (index_type i = 0; i < len; ++i)
      result += A[i]*A[i];
  else
    for (index_type i = 0; i < len; ++i)
      result += A[i];
  return result;
}

template <template <typename> class ReduceT,
          typename T>
std::complex<T> partial_sum(std::pair<T const*, T const*> const &A, ptrdiff_t offset, length_type len)
{
  std::complex<T> result = std::complex<T>();
  if (is_sum_squared_based<ReduceT>::value)
    for (index_type i = 0; i < len; ++i)
    {
      std::complex<T> val = 
        std::complex<T>(A.first[i + offset], A.second[i + offset]);
      result += val*val;
    }
  else
    for (index_type i = 0; i < len; ++i)
      result += std::complex<T>(A.first[i + offset], A.second[i + offset]);

  return result;
}

} // namespace vsip::impl
} // namespace vsip


namespace vsip_csl
{
namespace dispatcher
{

template<template <typename> class ReduceT>
struct List<op::reduce<ReduceT> >
{
  typedef Make_type_list<be::cbe_sdk,
			 be::cuda,
			 be::cvsip,
			 be::mercury_sal,
			 be::generic>::type type;
};

/// Generic evaluator for vector reductions.
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic, 
                 void(T&, Block const&, row1_type, impl::Int_type<1>)>
{
  static char const* name() { return "be_generic_vector_reduction ";}

  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, row1_type, impl::Int_type<1>)
  { return true; }

  static void exec(T& r, Block const& a, row1_type, impl::Int_type<1>)
  {
    length_type length = a.size(1, 0);
    length_type const max_length = VSIP_IMPL_MAX_SUMMATION_LENGTH;


    // Anything under the maximum length, not a summation (min/max)
    // or not dealing with single-precision floating point is 
    // handled in the usual way.
    if (!vsip::impl::is_summation<ReduceT>::value ||
        !vsip::impl::is_same<typename vsip::impl::scalar_of<T>::type, float>::value ||
        (length < max_length))
    {
      typedef typename Block::value_type VT;
      typename ReduceT<VT>::accum_type state = ReduceT<VT>::initial();

      PRAGMA_IVDEP
        for (index_type i=0; i<length; ++i)
        {
          state = ReduceT<VT>::update(state, a.get(i));
          if (ReduceT<VT>::done(state)) break;
        }
      r = ReduceT<VT>::value(state, length);
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

      vsip::dda::Data<Block, vsip::dda::in> data(a);

      typename Block::value_type result = 0;
      index_type sb = 0;
      for (; sb < subblocks - 1; ++sb)
      {
        result += vsip::impl::partial_sum<ReduceT>(data.ptr(), sb * max_length, max_length);
      }
      result += vsip::impl::partial_sum<ReduceT>(data.ptr(), sb * max_length, last_subblock_size);

      // Partial sums are accumlated in the view's natural type, even
      // for magnitude-of-squared values where the imaginary part is zero.
      // The result type is real in those cases, so we extract the desired
      // part depending on the type being returned.
      vsip::impl::extract<typename Block::value_type, T>::apply(result, r);

      if (vsip::impl::is_mean_based<ReduceT>::value)
        r /= length;
    }
  }
};


/// Generic evaluator for matrix reductions (tuple<0, 1, 2>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, row2_type, impl::Int_type<2>)>
{
  static char const* name() { return "be_generic_matrix_012_reduction ";}

  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, row2_type, impl::Int_type<2>)
  { return true; }

  static void exec(T& r, Block const& a, row2_type, impl::Int_type<2>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_i = a.size(2, 0);
    length_type length_j = a.size(2, 1);

    for (index_type i=0; i<length_i; ++i)
      PRAGMA_IVDEP
      for (index_type j=0; j<length_j; ++j)
      {
	state = ReduceT<VT>::update(state, a.get(i, j));
	if (ReduceT<VT>::done(state)) break;
      }

    r = ReduceT<VT>::value(state, length_i*length_j);
  }
};


/// Generic evaluator for matrix reductions (tuple<2, 1, 0>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, col2_type, impl::Int_type<2>)>
{
  static char const* name() { return "be_generic_matrix_210_reduction ";}

  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, col2_type, impl::Int_type<2>)
  { return true; }

  static void exec(T& r, Block const& a, col2_type, impl::Int_type<2>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_i = a.size(2, 0);
    length_type length_j = a.size(2, 1);

    for (index_type j=0; j<length_j; ++j)
    PRAGMA_IVDEP
    for (index_type i=0; i<length_i; ++i)
    {
      state = ReduceT<VT>::update(state, a.get(i, j));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length_i*length_j);
  }
};


/// Generic evaluator for tensor reductions (tuple<0, 1, 2>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, tuple<0, 1, 2>, impl::Int_type<3>)>
{
  static char const* name() { return "be_generic_tensor_012_reduction ";}

  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, tuple<0, 1, 2>, impl::Int_type<3>)
  { return true; }

  static void exec(T& r, Block const& a, tuple<0, 1, 2>, impl::Int_type<3>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_0 = a.size(3, 0);
    length_type length_1 = a.size(3, 1);
    length_type length_2 = a.size(3, 2);

    for (index_type i=0; i<length_0; ++i)
    for (index_type j=0; j<length_1; ++j)
    for (index_type k=0; k<length_2; ++k)
    {
      state = ReduceT<VT>::update(state, a.get(i, j, k));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length_0*length_1*length_2);
  }
};


/// Generic evaluator for tensor reductions (tuple<0, 2, 1>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, tuple<0, 2, 1>, impl::Int_type<3>)>
{
  static char const* name() { return "be_generic_tensor_021_reduction ";}

  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, tuple<0, 2, 1>, impl::Int_type<3>)
  { return true; }

  static void exec(T& r, Block const& a, tuple<0, 2, 1>, impl::Int_type<3>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_0 = a.size(3, 0);
    length_type length_1 = a.size(3, 1);
    length_type length_2 = a.size(3, 2);

    for (index_type i=0; i<length_0; ++i)
    for (index_type k=0; k<length_2; ++k)
    for (index_type j=0; j<length_1; ++j)
    {
      state = ReduceT<VT>::update(state, a.get(i, j, k));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length_0*length_1*length_2);
  }
};


/// Generic evaluator for tensor reductions (tuple<1, 0, 2>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, tuple<1, 0, 2>, impl::Int_type<3>)>
{
  static char const* name() { return "be_generic_tensor_102_reduction ";}

  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, tuple<1, 0, 2>, impl::Int_type<3>)
  { return true; }

  static void exec(T& r, Block const& a, tuple<1, 0, 2>, impl::Int_type<3>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_0 = a.size(3, 0);
    length_type length_1 = a.size(3, 1);
    length_type length_2 = a.size(3, 2);

    for (index_type j=0; j<length_1; ++j)
    for (index_type i=0; i<length_0; ++i)
    for (index_type k=0; k<length_2; ++k)
    {
      state = ReduceT<VT>::update(state, a.get(i, j, k));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length_0*length_1*length_2);
  }
};


/// Generic evaluator for tensor reductions (tuple<1, 2, 0>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, tuple<1, 2, 0>, impl::Int_type<3>)>
{
  static char const* name() { return "be_generic_tensor_120_reduction ";}

  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, tuple<1, 2, 0>, impl::Int_type<3>)
  { return true; }

  static void exec(T& r, Block const& a, tuple<1, 2, 0>, impl::Int_type<3>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_0 = a.size(3, 0);
    length_type length_1 = a.size(3, 1);
    length_type length_2 = a.size(3, 2);

    for (index_type j=0; j<length_1; ++j)
    for (index_type k=0; k<length_2; ++k)
    for (index_type i=0; i<length_0; ++i)
    {
      state = ReduceT<VT>::update(state, a.get(i, j, k));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length_0*length_1*length_2);
  }
};


/// Generic evaluator for tensor reductions (tuple<2, 0, 1>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, tuple<2, 0, 1>, impl::Int_type<3>)>
{
  static char const* name() { return "be_generic_tensor_201_reduction ";}

  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, tuple<2, 0, 1>, impl::Int_type<3>)
  { return true; }

  static void exec(T& r, Block const& a, tuple<2, 0, 1>, impl::Int_type<3>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_0 = a.size(3, 0);
    length_type length_1 = a.size(3, 1);
    length_type length_2 = a.size(3, 2);

    for (index_type k=0; k<length_2; ++k)
    for (index_type i=0; i<length_0; ++i)
    for (index_type j=0; j<length_1; ++j)
    {
      state = ReduceT<VT>::update(state, a.get(i, j, k));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length_0*length_1*length_2);
  }
};


/// Generic evaluator for tensor reductions (tuple<2, 1, 0>).
template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::generic,
                 void(T&, Block const&, tuple<2, 1, 0>, impl::Int_type<3>)>
{
  static char const* name() { return "be_generic_tensor_210_reduction ";}

  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, tuple<2, 1, 0>, impl::Int_type<3>)
  { return true; }

  static void exec(T& r, Block const& a, tuple<2, 1, 0>, impl::Int_type<3>)
  {
    typedef typename Block::value_type VT;
    typename ReduceT<T>::accum_type state = ReduceT<VT>::initial();

    length_type length_0 = a.size(3, 0);
    length_type length_1 = a.size(3, 1);
    length_type length_2 = a.size(3, 2);

    for (index_type k=0; k<length_2; ++k)
    for (index_type j=0; j<length_1; ++j)
    for (index_type i=0; i<length_0; ++i)
    {
      state = ReduceT<VT>::update(state, a.get(i, j, k));
      if (ReduceT<VT>::done(state)) break;
    }

    r = ReduceT<VT>::value(state, length_0*length_1*length_2);
  }
};

template <typename                  T,
	  typename                  Block,
	  typename                  OrderT,
	  int                       Dim>
struct Evaluator<op::reduce<impl::Mean_value>, be::parallel,
                 void(T&, Block const&, OrderT, impl::Int_type<Dim>)>
  : vsip::impl::Par_reduction_eval_base<impl::Mean_value, T, Block, OrderT, Dim>
{
  static void exec(T& r, Block const& a, OrderT, impl::Int_type<Dim>)
  {
    typedef typename Block::value_type VT;
    T l_r;
    typedef typename impl::Distributed_local_block<Block>::type local_block_type;
    typedef typename get_block_layout<local_block_type>::order_type order_type;
    typedef impl::Int_type<Dim>                                       dim_type;
    typedef impl::Mean_value<VT>                                      reduce_type;
    typedef typename Block::map_type                            map_type;

    dispatch<op::reduce<impl::Sum_value>, void,
             typename impl::Sum_value<VT>::result_type&,
             local_block_type const&,
             order_type,
             dim_type>
      (l_r, get_local_block(a), order_type(), dim_type());

    if (!is_same<map_type, Replicated_map<Block::dim> >::value)
	r = a.map().impl_comm().allreduce(reduce_type::rtype, l_r);
    else
      r = l_r;

    r /= static_cast<typename reduce_type::accum_type>(a.size());
  }
};



template <typename                  T,
	  typename                  Block,
	  typename                  OrderT,
	  int                       Dim>
struct Evaluator<op::reduce<impl::Mean_magsq_value>, be::parallel,
                 void(T&, Block const&, OrderT, impl::Int_type<Dim>)>
  : vsip::impl::Par_reduction_eval_base<impl::Mean_magsq_value, T, Block, OrderT, Dim>
{
  static void exec(T& r, Block const& a, OrderT, impl::Int_type<Dim>)
  {
    typedef typename Block::value_type VT;
    T l_r;
    typedef typename impl::Distributed_local_block<Block>::type local_block_type;
    typedef typename get_block_layout<local_block_type>::order_type order_type;
    typedef impl::Int_type<Dim>                                       dim_type;
    typedef impl::Mean_magsq_value<VT>                                reduce_type;
    typedef typename Block::map_type                            map_type;

    dispatch<op::reduce<impl::Sum_magsq_value>, void,
      typename impl::Sum_magsq_value<VT>::result_type&,
             local_block_type const&,
             order_type,
             dim_type>
      (l_r, get_local_block(a), order_type(), dim_type());

    if (!is_same<map_type, Replicated_map<Block::dim> >::value)
      r = a.map().impl_comm().allreduce(reduce_type::rtype, l_r);
    else
      r = l_r;

    r /= static_cast<typename reduce_type::accum_type>(a.size());
  }
};


template <template <typename> class ReduceT,
	  typename                  T,
	  typename                  Block,
	  typename                  OrderT,
	  int                       Dim>
struct Evaluator<op::reduce<ReduceT>, be::parallel, 
                 void(T&, Block const&, OrderT, impl::Int_type<Dim>)>
  : vsip::impl::Par_reduction_eval_base<ReduceT, T, Block, OrderT, Dim>
{
  static void exec(T& r, Block const& a, OrderT, impl::Int_type<Dim>)
  {
    typedef typename Block::value_type VT;
    T l_r = T();
    typedef typename impl::Distributed_local_block<Block>::type local_block_type;
    typedef typename get_block_layout<local_block_type>::order_type order_type;
    typedef impl::Int_type<Dim>                                       dim_type;
    typedef ReduceT<VT>                                         reduce_type;
    typedef typename Block::map_type                            map_type;

    dispatch<op::reduce<ReduceT>, void,
             typename ReduceT<VT>::result_type&,
	     local_block_type const&,
             order_type,
             dim_type>
      (l_r, get_local_block(a), order_type(), dim_type());

    if (!is_same<map_type, Replicated_map<Block::dim> >::value)
      r = a.map().impl_comm().allreduce(ReduceT<T>::rtype, l_r);
    else
      r = l_r;
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
