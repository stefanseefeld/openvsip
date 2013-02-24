/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/ipp/fir.cpp
    @author  Stefan Seefeld
    @date    2006-11-02
    @brief   VSIPL++ Library: FIR implementation.
*/

#ifndef VSIP_OPT_IPP_FIR_HPP
#define VSIP_OPT_IPP_FIR_HPP

#include <vsip/core/signal/fir_backend.hpp>
#include <ipps.h>
#include <ippi.h>

namespace vsip
{
namespace impl
{
namespace ipp
{

template <typename T, typename I,
          IppStatus (VSIP_IMPL_IPP_CALL *F)
            (I const*,I*,int,I const*,int,I*,int*),
          IppStatus (VSIP_IMPL_IPP_CALL *FD)
            (I const*,I*,int,I const*,int,int,int,int,int,I*)>
struct Fir_base
{
  typedef Fir_base  base_type;

  static void call(T const *xkernel, length_type kernel_size,  
                   T const *xin, T *xout, length_type out_size,
                   T *xstate, length_type *xstate_size,
                   length_type decimation)
  {
    I const *const in = reinterpret_cast<I const*>(xin);
    I *const out = reinterpret_cast<I*>(xout);
    I const *const  kernel = reinterpret_cast<I const*>(xkernel);
    I* const state = reinterpret_cast<I*>(xstate);
    int state_size = *xstate_size;
    IppStatus stat = (decimation == 1) ?
      F(in, out, out_size, kernel, kernel_size, state, &state_size) :
      FD(in, out, out_size, kernel, kernel_size, 1, 0, decimation, 0, state);
    assert(stat == ippStsNoErr);
    *xstate_size = state_size;
  }
};

template <typename T> struct Fir_traits;

template<>
struct Fir_traits<float> : Fir_base<float, Ipp32f,
                                    ippsFIR_Direct_32f, ippsFIRMR_Direct_32f>
{};

template<> 
struct Fir_traits<double> : Fir_base<double, Ipp64f,
                                     ippsFIR_Direct_64f, ippsFIRMR_Direct_64f> 
{};

template<> 
struct Fir_traits<std::complex<float> > : Fir_base<std::complex<float>, Ipp32fc,
                                                   ippsFIR_Direct_32fc,
                                                   ippsFIRMR_Direct_32fc>
{};

template<> 
struct Fir_traits<std::complex<double> > : Fir_base<std::complex<double>, Ipp64fc,
                                                    ippsFIR_Direct_64fc,
                                                    ippsFIRMR_Direct_64fc> 
{};

template <typename T>
void 
fir_call(T const *kernel, length_type kernel_size,
         T const *in, T *out, length_type out_size,
         T *state, length_type *state_size,
         length_type decimation)
{
  Fir_traits<T>::call(kernel, kernel_size, in, out, out_size,
                      state, state_size, decimation);
}

// Instantiate it for the supported types.

template
void 
fir_call(float const *, length_type, float const *, float *, length_type,
         float *, length_type *, length_type);

template
void 
fir_call(std::complex<float> const *, length_type, std::complex<float> const *,
         std::complex<float> *, length_type,
         std::complex<float> *, length_type *, length_type);

template
void 
fir_call(double const *, length_type, double const *, double *, length_type,
         double *, length_type *, length_type);

template
void 
fir_call(std::complex<double> const *, length_type, std::complex<double> const *,
         std::complex<double> *, length_type,
         std::complex<double> *, length_type *, length_type);


} // namespace vsip::impl::ipp
} // namespace vsip::impl
} // namespace vsip

#endif
