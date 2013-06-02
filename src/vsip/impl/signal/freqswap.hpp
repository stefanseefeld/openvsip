//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_signal_freqswap_hpp_
#define vsip_impl_signal_freqswap_hpp_

#include <ovxx/signal/freqswap.hpp>

namespace vsip
{

/// Swaps halves of a vector, or quadrants of a matrix, to remap zero 
/// frequencies from the origin to the middle.
template <template <typename, typename> class const_View,
          typename T,
          typename B>
const_View<T, ovxx::expr::Unary<ovxx::expr::op::Freqswap, B> const>
freqswap(const_View<T, B> in) VSIP_NOTHROW
{
  typedef ovxx::expr::Unary<ovxx::expr::op::Freqswap, B> const block_type;
  ovxx::expr::op::Freqswap<B> fs(in.block());
  return const_View<T, block_type>(block_type(fs));
}

} // namespace vsip

namespace ovxx
{
namespace signal
{
template <template <typename, typename> class const_View,
          typename T1, typename B1,
	  template <typename, typename> class View,
	  typename T2, typename B2>
View<T2, B2>
freqswap(const_View<T1, B1> in, View<T2, B2> out)
{
  out = vsip::freqswap(in);
  return out;
}

// Only swap halves along one axis
template <vsip::dimension_type D,
	  typename T1, typename B1,
	  typename T2, typename B2>
vsip::Matrix<T2, B2>
freqswap(const_Matrix<T1, B1> in, Matrix<T2, B2> out)
{
  OVXX_PRECONDITION(in.size(0) == out.size(0));
  OVXX_PRECONDITION(in.size(1) == out.size(1));
  length_type rows = in.size(0);
  length_type cols = in.size(1);
  if (D == vsip::row)
  {
    for (index_type r = 0; r != rows; ++r)
      freqswap(in.row(r), out.row(r));
  }
  else
  {
    for (index_type c = 0; c != rows; ++c)
      freqswap(in.col(c), out.col(c));
  }
  return out;
}

} // namespace ovxx::signal
} // namespace ovxx

#endif
