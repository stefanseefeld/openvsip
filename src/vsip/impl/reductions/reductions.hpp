//
// Copyright (c) 2005, 2006 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_reductions_reductions_hpp_
#define vsip_impl_reductions_reductions_hpp_

#include <ovxx/reductions/reductions.hpp>

namespace vsip
{
template <template <typename, typename> class V,
	  typename T, typename B>
typename ovxx::All_true<T>::result_type
alltrue(V<T, B> view)
{
  return ovxx::reduce<ovxx::All_true>(view);
}

template <template <typename, typename> class V,
	  typename T, typename B>
typename ovxx::Any_true<T>::result_type
anytrue(V<T, B> view)
{
  return ovxx::reduce<ovxx::Any_true>(view);
}

template <template <typename, typename> class V,
	  typename T, typename B>
typename ovxx::Mean_value<T>::result_type
meanval(V<T, B> view)
{
  return ovxx::reduce<ovxx::Mean_value>(view);
}

// Note: meansqval computes the mean of the magnitude square
template <template <typename, typename> class V,
	  typename T, typename B>
typename ovxx::Mean_magsq_value<T>::result_type
meansqval(V<T, B> view)
{
  return ovxx::reduce<ovxx::Mean_magsq_value>(view);
}

template <template <typename, typename> class V,
	  typename T, typename B>
typename ovxx::Sum_value<T>::result_type
sumval(V<T, B> view)
{
  return ovxx::reduce<ovxx::Sum_value>(view);
}

template <template <typename, typename> class V,
	  typename T, typename B>
typename ovxx::Sum_sq_value<T>::result_type
sumsqval(V<T, B> view)
{
  return ovxx::reduce<ovxx::Sum_sq_value>(view);
}

} // namespace vsip

#endif
