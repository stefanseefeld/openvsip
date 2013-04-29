/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/dda.hpp
    @author  Stefan Seefeld
    @date    2009-29-30
    @brief   Sourcery VSIPL++: Direct Data Access API. 
*/

#ifndef VSIP_CSL_DDA_HPP
#define VSIP_CSL_DDA_HPP

#include <vsip/layout.hpp>
#include <vsip/dda.hpp>

namespace vsip_csl
{
namespace dda
{
  using namespace vsip::dda;
  using vsip::impl::Rt_layout;
  using vsip::dda::impl::Pointer;

  using vsip::impl::Applied_layout;

/// Determine if an dda::Data object refers to a dense (contiguous,
/// unit-stride) region of memory.
template <typename OrderT,
	  typename DataT>
bool
is_data_dense(dimension_type dim,
	      DataT const &data)
{
  using vsip::dimension_type;
  using vsip::stride_type;

  dimension_type const dim0 = OrderT::impl_dim0;
  dimension_type const dim1 = OrderT::impl_dim1;
  dimension_type const dim2 = OrderT::impl_dim2;

  assert(dim <= VSIP_MAX_DIMENSION);

  if (dim == 1)
  {
    return (data.stride(dim0) == 1);
  }
  else if (dim == 2)
  {
    return (data.stride(dim1) == 1) &&
           (data.stride(dim0) == static_cast<stride_type>(data.size(dim1)) ||
	    data.size(dim0) == 1);
  }
  else /*  if (dim == 2) */
  {
    return (data.stride(dim2) == 1) &&
           (data.stride(dim1) == static_cast<stride_type>(data.size(dim2)) ||
	    (data.size(dim0) == 1 && data.size(dim1) == 1)) &&
           (data.stride(dim0) == static_cast<stride_type>(data.size(dim1)  *
							  data.size(dim2)) ||
	    data.size(dim0) == 1);
  }
}

}
}

#endif
