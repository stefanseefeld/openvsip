/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   CUDA utilities.

#ifndef vsip_opt_cuda_util_hpp_
#define vsip_opt_cuda_util_hpp_

namespace vsip
{
namespace impl
{
namespace cuda
{

/// Size threshold for elementwise operations.
template <typename Operation, bool Split = false>
struct Size_threshold
{
  static int const value = 0;
};

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif // vsip_opt_cuda_util_hpp_

 
