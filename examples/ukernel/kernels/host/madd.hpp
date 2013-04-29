/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

#ifndef KERNELS_HOST_MADD_HPP
#define KERNELS_HOST_MADD_HPP

#include <vsip_csl/ukernel/host/ukernel.hpp>

namespace example
{
// Host-side vector elementwise multiply-add ukernel.
class Madd : public vsip_csl::ukernel::Host_kernel
{
  // Parameters.
  //  - 'tag_type' is used to select the appropriate kernel (via
  //    Task_manager's Task_map)
  //  - in_argc and out_argc describe the number of input and output
  //    streams.
  //  - param_type (inherited) defaults to 'Empty_params'.
public:
  static unsigned int const in_argc  = 3;
  static unsigned int const out_argc = 1;

  // Host-side ukernel object initialization.
  //
  // Streaming pattern divides matrix into whole, single rows.

  Madd()
    : sp_(vsip_csl::ukernel::Blocksize_sdist(1),
	  vsip_csl::ukernel::Whole_sdist())
  {}



  // Host-side compute kernel.  Used if accelerator is not available.
  template <typename View0, typename View1, typename View2, typename View3>
  void compute(View0 in0, View1 in1, View2 in2, View3 out)
  {
    out = in0 * in1 + in2;
  }

  vsip_csl::ukernel::Stream_pattern const& in_spatt(vsip::index_type i) const
  { return sp_; }

  vsip_csl::ukernel::Stream_pattern const& out_spatt(vsip::index_type) const
  { return sp_; }

private:
  vsip_csl::ukernel::Stream_pattern sp_;
};

}

#endif
