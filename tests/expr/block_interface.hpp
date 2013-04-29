//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/support.hpp>
#include <vsip_csl/test.hpp>

using vsip_csl::use_variable;

namespace vsip
{

/***********************************************************************
  Declarations
***********************************************************************/

template <typename Block>
void
block_1d_interface_test(Block const& block)
{
  typename Block::value_type value;
  typename Block::reference_type ref = value;
  typename Block::const_reference_type cref = value;
  typename Block::map_type map;
  use_variable(ref);
  use_variable(cref);
  (void) block.get(0);
  map = block.map();
}

template <typename Block>
void
block_2d_interface_test(Block const& block)
{
  typename Block::value_type value;
  typename Block::reference_type ref = value;
  typename Block::const_reference_type cref = value;
  typename Block::map_type map;
  (void) block.get(0, 0);
  map = block.map();
}

template <typename Block>
void
block_3d_interface_test(Block const& block)
{
  typename Block::value_type value;
  typename Block::reference_type ref = value;
  typename Block::const_reference_type cref = value;
  typename Block::map_type map;
  (void) block.get(0, 0, 0);
  map = block.map();
}

}
