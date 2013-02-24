/* Copyright (c) 2009 by CodeSourcery, Inc.  All rights reserved. */

/** @file    vsip_csl/sort.hpp
    @author  Mike LeBlanc
    @date    2009-04-22
    @brief   VSIPL++ Library: Sort support classes.

*/

#ifndef VSIP_CSL_SORT_HPP
#define VSIP_CSL_SORT_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/selgen.hpp>
#include <vsip/core/metaprogramming.hpp>

namespace vsip_csl
{
namespace impl
{
/// A comparison functor that compares data via an index vector.
template <typename T, typename BlockT, typename CompT>
struct compare_indices : std::binary_function<vsip::index_type, vsip::index_type, bool>
{
  typedef vsip::const_Vector<T, BlockT> vector_type;
  vector_type data;
  CompT &compare;
  compare_indices(vector_type d, CompT &c) : data(d), compare(c) {}
  bool
  operator() (vsip::index_type x, vsip::index_type y) const
  { return compare(data.get(x), data.get(y));}
};

/// Disambiguate between view-type argument and comparison functor.
template <typename ViewT,
          typename Arg2 = ViewT,
          bool arg2_is_view = vsip::impl::Is_view_type<Arg2>::value>
struct Sort_data_helper;

/// In-place sorter
template <typename ViewT>
struct Sort_data_helper<ViewT, ViewT, true>
{
  static void exec(ViewT inout)
  {
    typedef vsip::impl::Layout<1, vsip::row1_type,
      vsip::impl::Stride_unit_dense> layout_type;
    vsip::impl::Ext_data<typename ViewT::block_type, layout_type>
      raw(inout.block(), vsip::impl::SYNC_INOUT);
    std::sort(raw.data(), raw.data() + raw.size(0), std::less<typename ViewT::value_type>());
  }
  static void exec(ViewT in, ViewT out)
  {
    out = in;
    exec(out);
  }
};

/// Out-of-place sorter
template <typename View1T, typename View2T>
struct Sort_data_helper<View1T, View2T, true>
{
  static void exec(View1T in, View2T out)
  {
    out = in;
    Sort_data_helper<View2T>::exec(out);
  }
};

/// In-place sorter with comparison functor.
template <typename ViewT, typename Comp>
struct Sort_data_helper<ViewT, Comp, false>
{
  static void exec(ViewT inout, Comp comp)
  {
    typedef vsip::impl::Layout<1, vsip::row1_type,
      vsip::impl::Stride_unit_dense> layout_type;
    vsip::impl::Ext_data<typename ViewT::block_type, layout_type>
      raw(inout.block(), vsip::impl::SYNC_INOUT);
    std::sort(raw.data(), raw.data() + raw.size(0), comp);
  }
};

}



/// Return sorted indices in user-supplied vector.
///  :indices:      Overwritten with sorted indices such that
///                 sort_functor(data(indices(i)), data(indices(j))) == false iff i > j
///  :data:         data vector, not modified.
///  :comp:         a functor object like less<T>() to compare
///                 two T items for sorted order.
template <typename T,
          typename B1,
          typename B2,
          typename FunctorT>
inline
void
sort_indices(vsip::Vector<vsip::index_type, B1> indices,
             vsip::const_Vector<T, B2> data,
             FunctorT c)
{
  indices = vsip::ramp(0,1,indices.size(0));
  typedef vsip::impl::Layout<1, vsip::row1_type, vsip::impl::Stride_unit_dense> layout_type;
  impl::compare_indices<T, B2, FunctorT> comp(data, c);
  vsip::impl::Ext_data<B1, layout_type> raw(indices.block(), vsip::impl::SYNC_INOUT);
  std::sort(raw.data(), raw.data() + raw.size(0), comp);
}

/// Return sorted indices in user-supplied vector.
///  :indices:      Overwritten with sorted indices such that
///                 data(indices(i)) < data(indices(j)) == false iff i > j
///  :data:         data vector, not modified.
template <typename T, typename B1, typename B2>
inline void
sort_indices(vsip::Vector<vsip::index_type, B1> indices,
             vsip::const_Vector<T, B2> data)
{
  indices = vsip::ramp(0,1,indices.size(0));
  typedef vsip::impl::Layout<1, vsip::row1_type, vsip::impl::Stride_unit_dense> layout_type;
  std::less<T> less;
  impl::compare_indices<T, B2, std::less<T> > comp(data, less);
  vsip::impl::Ext_data<B1, layout_type> raw(indices.block(), vsip::impl::SYNC_INOUT);
  std::sort(raw.data(), raw.data() + raw.size(0), comp);
}

/// in-place sort.
template <typename ViewT>
void
sort_data(ViewT inout)
{
  impl::Sort_data_helper<ViewT>::exec(inout);
}

template <typename A1, typename A2>
void
sort_data(A1 a1, A2 a2)
{
  impl::Sort_data_helper<A1, A2>::exec(a1, a2);
}

template <typename ViewT1, typename ViewT2, typename Comp>
void
sort_data(ViewT1 in, ViewT2 out, Comp comp)
{
  out = in;
  sort_data(out, comp);
}

} // namespace vsip_csl

#endif
