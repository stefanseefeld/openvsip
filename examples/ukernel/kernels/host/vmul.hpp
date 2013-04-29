/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/// Description: Vector element-wise multiply Ukernel

#ifndef KERNELS_HOST_VMUL_HPP
#define KERNELS_HOST_VMUL_HPP

#include <vsip_csl/ukernel/host/ukernel.hpp>

namespace example
{
namespace uk = vsip_csl::ukernel;

// Host-side vector elementwise multiply ukernel.
// Template parameters:
//  - The first parameter specifies the number of pre-input streams.
//  - The second parameter specifies the number of input streams.
//  - The third parameter specifies the number of output streams.
class Vmul_proxy : public uk::Kernel_proxy<0, 2, 1>
{
public:

  // Host-side ukernel object initialization.
  //
  // Streaming pattern divides linear vector into blocks, with minimum
  // size 256, maximum size 1024.  (Blocksize_sdist has an implicit
  // quantum of 16 elements).
  Vmul_proxy() : sp_(uk::Blocksize_sdist(1024, 256)) {}

  // Host-side compute kernel.  Used if accelerator is not available.
  template <typename View0, typename View1, typename View2>
  void compute(View0 in0, View1 in1, View2 out)
  {
    out = in0 * in1;
  }

  // Query API:  in_spatt()/out_spatt() allow VSIPL++ to determine
  // streaming pattern for user-kernel.  Since both input and output
  // have same streaming pattern, simply return 'sp'
  uk::Stream_pattern const& in_spatt(vsip::index_type) const
  { return sp_;}

  uk::Stream_pattern const& out_spatt(vsip::index_type) const
  { return sp_;}

private:
  uk::Stream_pattern sp_;	
};
}

#endif
