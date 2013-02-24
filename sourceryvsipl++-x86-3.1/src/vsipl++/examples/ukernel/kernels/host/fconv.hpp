/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

#ifndef KERNELS_HOST_FCONV_HPP
#define KERNELS_HOST_FCONV_HPP

#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <kernels/fconv_params.hpp>

namespace example
{
namespace uk = vsip_csl::ukernel;

class Fconv_proxy : public uk::Kernel_proxy<1, 1, 1, Fconv_params>
{
public:
  // Host-side ukernel object initialization.
  //
  // Streaming pattern divides linear vector into blocks, with minimum
  // size 256, maximum size 1024.  (Blocksize_sdist has an implicit
  // quantum of 16 elements).
  Fconv_proxy(unsigned int size)
    : size_(size),
      pre_sp_(uk::Whole_sdist()),
      io_sp_(uk::Blocksize_sdist(1), uk::Whole_sdist())
  {}

  // Host-side compute kernel.  Used if accelerator is not available.
  template <typename View0, typename View1, typename View2>
  void compute( View0 in0, View1 in1, View2 out)
  {
    out = in0 * in1;
  }

  // Queury API:
  // - fill_params() fills the parameter block to be passed to the
  //   accelerators.
  // - in_spatt()/out_spatt() allow VSIPL++ to determine streaming
  //   pattern for user-kernel.  Since both input and output have same
  //   streaming pattern, simply return 'sp'
  void fill_params(param_type& param) const
  {
    param.fwd_fft_params.size  = size_;
    param.fwd_fft_params.dir   = -1;
    param.fwd_fft_params.scale = 1.f;
    param.vmmul_params.size  = size_;
    param.inv_fft_params.size  = size_;
    param.inv_fft_params.dir   = +1;
    param.inv_fft_params.scale = 1.f / size_;
  }

  uk::Stream_pattern const& in_spatt(vsip::index_type i) const
  { return (i == 0) ? pre_sp_ : io_sp_;}

  uk::Stream_pattern const& out_spatt(vsip::index_type) const
  { return io_sp_;}

private:
  unsigned int size_;
  uk::Stream_pattern pre_sp_;
  uk::Stream_pattern io_sp_;
};

} // namespace example

#endif
