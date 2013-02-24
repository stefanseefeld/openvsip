/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/coverage.hpp
    @author  Jules Bergmann
    @date    2006-05-31
    @brief   VSIPL++ Library: Coverage utilities.
*/

#ifndef VSIP_CORE_COVERAGE_HPP
#define VSIP_CORE_COVERAGE_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/config.hpp>

#ifndef VSIP_IMPL_DO_COVERAGE
#  define VSIP_IMPL_DO_COVERAGE 0
#endif

#if VSIP_IMPL_DO_COVERAGE
#  include <iostream>
#  include <typeinfo>
#endif



/***********************************************************************
  Macros
***********************************************************************/


#if VSIP_IMPL_DO_COVERAGE
#  define VSIP_IMPL_COVER_FCN(TYPE, FCN)				\
  std::cout << "FCN," << TYPE << "," << #FCN << std::endl;
#else
#  define VSIP_IMPL_COVER_FCN(TYPE, FCN)
#endif

#if VSIP_IMPL_DO_COVERAGE
#  define VSIP_IMPL_COVER_BLK(TYPE, BLK)				\
     std::cout << "BLK," << TYPE << "," << typeid(BLK).name() << std::endl;
#else
#  define VSIP_IMPL_COVER_BLK(TYPE, BLK)
#endif

#if VSIP_IMPL_DO_COVERAGE
#  define VSIP_IMPL_COVER_TAG(TYPE, TAG)				\
     std::cout << "TAG," << TYPE << "," << typeid(TAG).name() << std::endl;
#else
#  define VSIP_IMPL_COVER_TAG(TYPE, TAG)
#endif

#endif // VSIP_CORE_COVERAGE_HPP
