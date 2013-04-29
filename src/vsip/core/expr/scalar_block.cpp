//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

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
