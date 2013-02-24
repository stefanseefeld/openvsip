/* Copyright (c) 2005, 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/extdata_common.hpp
    @author  Jules Bergmann
    @date    2006-11-29
    @brief   VSIPL++ Library: Common Decls for Direct Data Access.

*/

#ifndef VSIP_CORE_EXTDATA_COMMON_HPP
#define VSIP_CORE_EXTDATA_COMMON_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/static_assert.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/layout.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

/// Enum to indicate data interface syncronization necessary for
/// correctness.
enum sync_action_type
{
  /// syncronize data interface on creation
  SYNC_IN              = 0x01,
  /// syncronize data interface on destruction
  SYNC_OUT             = 0x02,
  /// SYNC_INOUT syncronize data interface on creation and destruction
  SYNC_INOUT           = SYNC_IN | SYNC_OUT,		// 0x03
  SYNC_NOPRESERVE_impl = 0x04,
  /// syncronize data interface on creation
  /// with guarantee that changes are not preserved
  /// (usually by forcing a copy).
  SYNC_IN_NOPRESERVE   = SYNC_IN | SYNC_NOPRESERVE_impl	// 0x05
};

namespace data_access 
{

/// Low-level data access class.
///
/// Template parameters:
///
///   :AT: is a valid data access tag,
///   :Block: is a block that supports the data access interface indicated
///           by `AT`.
///   :LP:    is a layout policy compatible with access tag `AT` and block 
///           `Block`.
///
/// (Each specializtion may provide additional requirements).
///
/// Member Functions:
///    ...
///
/// Notes:
///   Low_level_data_access does not hold a block reference/pointer, it
///   is provided to each member function by the caller.  This allows
///   the caller to make policy decisions, such as reference counting.
template <typename AT,
          typename Block,
	  typename LP>
class Low_level_data_access;

template <typename AT> struct Cost { static int const value = 10; };

} // namespace vsip::impl::data_access
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_EXTDATA_COMMON_HPP
