/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/diag/class_name.hpp
    @author  Jules Bergmann
    @date    2007-03-06
    @brief   VSIPL++ Library: Class name utility for diags.
*/

#ifndef VSIP_OPT_DIAG_CLASS_NAME_HPP
#define VSIP_OPT_DIAG_CLASS_NAME_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <string>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace diag_detail
{

// Helper class to return the name corresponding to a dispatch tag.

template <typename T> 
struct Class_name
{
  static std::string name() { return "unknown"; }
};

#define VSIP_IMPL_CLASS_NAME(TYPE)				\
  template <>							\
  struct Class_name<TYPE> {					\
    static std::string name() { return "" # TYPE; }		\
  }

VSIP_IMPL_CLASS_NAME(float);


} // namespace vsip::impl::diag_detail
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_DIAG_EXTDATA_HPP
