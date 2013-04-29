/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

#ifndef KERNELS_HOST_FFT_HPP
#define KERNELS_HOST_FFT_HPP

#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <kernels/fft_param.hpp>

namespace example
{
namespace uk = vsip_csl::ukernel;

class Fft_proxy : public uk::Kernel_proxy<0, 1, 1, Fft_params>
{
public:
  // Host-side ukernel object initialization.
  //
  // Streaming pattern divides matrix into blocks of 1 row, keeping
  // all elements in a row together.
public:
  Fft_proxy(unsigned int size, int dir, float scale)
    : size_(size),
      dir_(dir),
      scale_(scale),
      sp_(uk::Blocksize_sdist(1), uk::Whole_sdist())
  {}

  // Host-side compute kernel.  Used if accelerator is not available.
  template <typename View1, typename View2>
  void compute(View1 in, View2 out)
  {
    assert(0); // TODO
  }


  // Query API:
  // - fill_params() fills the parameter block to be passed to the
  //   accelerators.
  // - in_spatt()/out_spatt() allow VSIPL++ to determine streaming
  //   pattern for user-kernel.  Since both input and output have same
  //   streaming pattern, simply return 'sp'
  void fill_params(param_type& param) const
  {
    param.size  = size_;
    param.dir   = dir_;
    param.scale = scale_;
  }

  uk::Stream_pattern const& in_spatt(vsip::index_type) const
  { return sp_;}

  uk::Stream_pattern const& out_spatt(vsip::index_type) const
  { return sp_;}

  vsip::length_type stack_size() const { return 4096; }

  // Member data.
private:
  unsigned int size_;
  int          dir_;
  float        scale_;
  uk::Stream_pattern sp_;
};

}

#endif
