/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/diag/block.hpp
    @author  Jules Bergmann
    @date    2007-08-21
    @brief   VSIPL++ Library: Diagnostics for blocks.
*/

#ifndef VSIP_OPT_DIAG_BLOCK_HPP
#define VSIP_OPT_DIAG_BLOCK_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <iomanip>

#include <vsip/opt/diag/extdata.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{

template <typename BlockT>
void
diagnose_block(char const* str, BlockT const& /*blk*/)
{
  using std::cout;
  using std::endl;
  using vsip::impl::diag_detail::Class_name;

  typedef typename get_block_layout<BlockT>::access_type AT;
  dimension_type const dim = get_block_layout<BlockT>::dim;
  bool const is_split = is_split_block<BlockT>::value;

  cout << "diagnose_block(" << str << "):" << std::endl
       << "  BlockT        : " << typeid(BlockT).name() << endl
       << "  Dim           : " << dim << endl
       << "  is_split_block: " << (is_split ? "yes" : "no") << endl
       << "  DDA ct_cost : " << dda::Data<BlockT, dda::out>::ct_cost << endl
       << "  access_type   : " << Class_name<AT>::name() << endl
    ;
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_DIAG_BLOCK_HPP
