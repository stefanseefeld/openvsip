//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_MATH_ENUM_HPP
#define VSIP_CORE_MATH_ENUM_HPP

/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{

enum mat_op_type
{
  mat_ntrans,
  mat_trans,
  mat_herm,
  mat_conj
};

enum product_side_type
{
  mat_lside,
  mat_rside
};

enum storage_type
{
  qrd_nosaveq,
  qrd_saveq1,
  qrd_saveq,
  svd_uvnos,
  svd_uvpart,
  svd_uvfull
};
   
} // namespace vsip

#endif // VSIP_CORE_MATH_ENUM_HPP
