/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/// Description: User-defined polar to rectangular
///              interpolation kernel for SSAR images.

#ifndef CBE_HOST_INTERP_HPP
#define CBE_HOST_INTERP_HPP

#include <vsip_csl/ukernel/host/ukernel.hpp>

// Host-side interpolation kernel
class Interp_proxy : public vsip_csl::ukernel::Kernel_proxy<0, 3, 1>
{
public:
  Interp_proxy()
    : sp_(vsip_csl::ukernel::Blocksize_sdist(1), 
          vsip_csl::ukernel::Whole_sdist())
  {}



  // Host-side compute kernel.  Used if accelerator is not available.
  //
  // View sizes:
  //   in0 is N x M
  //   in1 is N x M x I
  //   in2 is N x M
  //   out is NX x M    
  // 
  template <typename View0, typename View1, typename View2, typename View3>
  void compute(View0 in0, View1 in1, View2 in2, View3 out)
  {
    out = View3::value_type();

    for (vsip::index_type j = 0; j < in0.size(1); ++j)
      for (vsip::index_type i = 0; i < in0.size(0); ++i)
        for (vsip::index_type h = 0; h < in1.size(2); ++h)
        {
          vsip::index_type ikxrows = in0.get(i, j) + h;

          out.put(ikxrows, j, out.get(ikxrows, j) + 
            (in2.get(i, j) * in1.get(i, j, h)));
        }
  }


  // Query API:
  // - in_spatt()/out_spatt() allow VSIPL++ to determine streaming
  //   pattern for user-kernel.  Since both input and output have same
  //   streaming pattern, simply return 'sp'
  vsip_csl::ukernel::Stream_pattern const& in_spatt(vsip::index_type) const
  { return sp_; }

  vsip_csl::ukernel::Stream_pattern const& out_spatt(vsip::index_type) const
  { return sp_; }

private:
  vsip_csl::ukernel::Stream_pattern sp_;
};

namespace vsip_csl
{
namespace ukernel
{
template <>
struct Task_map<Interp_proxy,
  void(uint32_t*, float*, std::complex<float>*, std::complex<float>*)>
{
  static char const *plugin() { return "cinterp.plg";}
};

template <>
struct Task_map<Interp_proxy,
  void(uint32_t*, float*, std::pair<float*,float*>, std::pair<float*,float*>)>
{
  static char const *plugin() { return "zinterp.plg";}
};
} // namespace vsip_csl::ukernel
} // namespace vsip_csl

#endif
