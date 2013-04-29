/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip_csl/ukernel/kernels/host/interp.hpp
    @author  Don McCoy
    @date    2008-08-26
    @brief   VSIPL++ Library: User-defined polar to rectangular
               interpolation kernel for SSAR images.
*/

#ifndef VSIP_CSL_UKERNEL_KERNELS_HOST_INTERP_HPP
#define VSIP_CSL_UKERNEL_KERNELS_HOST_INTERP_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip_csl/ukernel/host/ukernel.hpp>

namespace vsip_csl
{
namespace ukernel
{


/***********************************************************************
  Definitions
***********************************************************************/

// Host-side interpolation kernel
class Interp : public Host_kernel
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
  // Streaming pattern divides matrix into whole, single columns (for now).
  //
  Interp()
    : sp_(Blocksize_sdist(1), 
          Whole_sdist())
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
  Stream_pattern const& in_spatt(vsip::index_type) const
  { return sp_; }

  Stream_pattern const& out_spatt(vsip::index_type) const
  { return sp_; }

private:
  Stream_pattern sp_;
};

template <>
struct Task_map<Interp,
  void(uint32_t*, float*, std::complex<float>*, std::complex<float>*)>
{
  static char const *plugin() { return "uk_plugin/interp_f.plg";}
};

template <>
struct Task_map<Interp,
  void(uint32_t*, float*, std::pair<float*, float*>, std::pair<float*, float*>)>
{
  static char const *plugin() { return "uk_plugin/zinterp_f.plg";}
};

}
}

#endif
