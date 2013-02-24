/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/math_enum.hpp
    @author  Jules Bergmann
    @date    2005-08-19
    @brief   VSIPL++ Library: Mathematical enumerations [math.enum].

*/

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
