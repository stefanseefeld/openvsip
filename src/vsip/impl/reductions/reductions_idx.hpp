//
// Copyright (c) 2005, 2006 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_reductions_reductions_idx_hpp_
#define vsip_impl_reductions_reductions_idx_hpp_

#include <ovxx/reductions/reductions_idx.hpp>

namespace vsip
{

template <template <typename, typename> class V,
	  typename T, typename B>
T
maxval(V<T, B> view, Index<V<T, B>::dim> &idx)
  VSIP_NOTHROW
{
  typedef typename get_block_layout<B>::order_type order_type;
  return ovxx::reduce_idx<ovxx::Max_value>(view, idx);
}

template <template <typename, typename> class V,
	  typename T, typename B>
T
minval(V<T, B> view, Index<V<T, B>::dim> &idx)
  VSIP_NOTHROW
{
  typedef typename get_block_layout<B>::order_type order_type;
  return ovxx::reduce_idx<ovxx::Min_value>(view, idx);
}

template <template <typename, typename> class V,
	  typename T, typename B>
typename ovxx::scalar_of<T>::type
maxmgval(V<T, B> view, Index<V<T, B>::dim> &idx)
  VSIP_NOTHROW
{
  typedef typename get_block_layout<B>::order_type order_type;
  return ovxx::reduce_idx<ovxx::Max_mag_value>(view, idx);
}

template <template <typename, typename> class V,
	  typename T, typename B>
typename ovxx::scalar_of<T>::type
minmgval(V<T, B> view, Index<V<T, B>::dim> &idx)
  VSIP_NOTHROW
{
  typedef typename get_block_layout<B>::order_type order_type;
  return ovxx::reduce_idx<ovxx::Min_mag_value>(view, idx);
}

template <template <typename, typename> class V,
	  typename T, typename B>
T
maxmgsqval(V<complex<T>, B> view, Index<V<complex<T>, B>::dim> &idx)
  VSIP_NOTHROW
{
  typedef typename get_block_layout<B>::order_type order_type;
  return ovxx::reduce_idx<ovxx::Max_magsq_value>(view, idx);
}

template <template <typename, typename> class V,
	  typename T, typename B>
T
minmgsqval(V<complex<T>, B> view, Index<V<complex<T>, B>::dim> &idx)
  VSIP_NOTHROW
{
  typedef typename get_block_layout<B>::order_type order_type;
  return ovxx::reduce_idx<ovxx::Min_magsq_value>(view, idx);
}

} // namespace vsip

#endif
