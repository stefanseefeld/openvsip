/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/block_interface.cpp
    @author  Stefan Seefeld
    @date    2005-02-15
    @brief   VSIPL++ Library: Unit tests for block API [view.block].

    This file has unit tests for block requirements (table 6.1 / 6.2).
*/

/***********************************************************************
  Included Files
***********************************************************************/

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
