/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

#ifndef KERNELS_HOST_VMMUL_HPP
#define KERNELS_HOST_VMMUL_HPP

#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <kernels/vmmul_param.hpp>

namespace example
{
namespace uk = vsip_csl::ukernel;

// Host-side vector elementwise copy ukernel.
class Vmmul_proxy : public uk::Kernel_proxy<1, 1, 1, Vmmul_params>
{
public:
  // Host-side ukernel object initialization.
  //
  // Streaming pattern divides linear vector into blocks, with minimum
  // size 256, maximum size 1024.  (Blocksize_sdist has an implicit
  // quantum of 16 elements).
  Vmmul_proxy(unsigned int size)
    : size_(size),
      pre_sp_(uk::Whole_sdist()),
      io_sp_(uk::Blocksize_sdist(1), uk::Whole_sdist())
  {}

  // Host-side compute kernel.  Used if accelerator is not available.
  template <typename View0, typename View1, typename View2>
  void compute(View0 in0, View1 in1, View2 out)
  {
    out = in0 * in1;
  }

  void fill_params(param_type& param) const
  {
    param.size  = size_;
  }

  uk::Stream_pattern const& in_spatt(vsip::index_type i) const
  { return (i == 0) ? pre_sp_ : io_sp_; }

  uk::Stream_pattern const& out_spatt(vsip::index_type) const
  { return io_sp_; }

private:
  unsigned int size_;
  uk::Stream_pattern pre_sp_;	
  uk::Stream_pattern io_sp_;	
};

} // namespace example

#endif
