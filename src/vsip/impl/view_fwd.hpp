//
// Copyright (c) 2007 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_view_fwd_hpp_
#define vsip_impl_view_fwd_hpp_

#include <vsip/impl/dense_fwd.hpp>

namespace vsip
{

template <template <typename, typename> class V, typename T, typename B>
struct ViewConversion;

template <typename T = VSIP_DEFAULT_VALUE_TYPE,
	  typename B = Dense<1, T> > struct Vector;
template <typename T = VSIP_DEFAULT_VALUE_TYPE,
	  typename B = Dense<2, T> > struct Matrix;
template <typename T = VSIP_DEFAULT_VALUE_TYPE,
	  typename B = Dense<3, T> > struct Tensor;
template <typename T = VSIP_DEFAULT_VALUE_TYPE,
	  typename B = Dense<1, T> > struct const_Vector;
template <typename T = VSIP_DEFAULT_VALUE_TYPE,
	  typename B = Dense<2, T> > struct const_Matrix;
template <typename T = VSIP_DEFAULT_VALUE_TYPE,
	  typename B = Dense<3, T> > struct const_Tensor;

} // namespace vsip

#endif
