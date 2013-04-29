/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/expr/scalar_block.cpp
    @author  Jules Bergmann
    @date    2006-11-27
    @brief   VSIPL++ Library: Scalar block class definitions.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/expr/scalar_block.hpp>
#include <vsip/core/parallel/global_map.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{
namespace impl
{

template <> Scalar_block_shared_map<1>::type
  Scalar_block_shared_map<1>::map = Scalar_block_shared_map<1>::type();

template <> Scalar_block_shared_map<2>::type
  Scalar_block_shared_map<2>::map = Scalar_block_shared_map<2>::type();

template <> Scalar_block_shared_map<3>::type
  Scalar_block_shared_map<3>::map = Scalar_block_shared_map<3>::type();

} // namespace vsip::impl
} // namespace vsip
