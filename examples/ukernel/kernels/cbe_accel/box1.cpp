/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description:
///   1D Box filter ukernel.

#include <kernels/cbe_accel/box1.hpp>

typedef example::Box1_kernel kernel_type;

#include <vsip_csl/ukernel/cbe_accel/alf_base.hpp>
