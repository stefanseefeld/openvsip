//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_VIEW_FWD_HPP
#define VSIP_CORE_VIEW_FWD_HPP

#include <vsip/core/dense_fwd.hpp>

/***********************************************************************
  Forward Declarations
***********************************************************************/

namespace vsip
{

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

#endif // VSIP_CORE_VIEW_FWD_HPP
