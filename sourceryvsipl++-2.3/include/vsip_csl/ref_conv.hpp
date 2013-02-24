/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/ref_conv.cpp
    @author  Jules Bergmann
    @date    2005-12-28
    @brief   VSIPL++ CodeSourcery Library: Reference implementation of 
             convolution function.
*/

#ifndef VSIP_CSL_REF_CORR_HPP
#define VSIP_CSL_REF_CORR_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/vector.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip/selgen.hpp>
#include <vsip/parallel.hpp>


namespace vsip_csl
{

/***********************************************************************
  Definitions
***********************************************************************/

namespace ref
{

vsip::length_type
conv_output_size(
  vsip::support_region_type supp,
  vsip::length_type         M,    // kernel length
  vsip::length_type         N,    // input  length
  vsip::length_type         D)    // decimation factor
{
  if      (supp == vsip::support_full)
    return ((N + M - 2)/D) + 1;
  else if (supp == vsip::support_same)
    return ((N - 1)/D) + 1;
  else //(supp == vsip::support_min)
  {
#if VSIP_IMPL_CONV_CORRECT_MIN_SUPPORT_SIZE
    return ((N - M + 1) / D) + ((N - M + 1) % D == 0 ? 0 : 1);
#else
    return ((N - 1)/D) - ((M-1)/D) + 1;
#endif
  }
}



vsip::stride_type
conv_expected_shift(
  vsip::support_region_type supp,
  vsip::length_type         M)     // kernel length
{
  if      (supp == vsip::support_full)
    return 0;
  else if (supp == vsip::support_same)
    return (M/2);
  else //(supp == vsip::support_min)
    return (M-1);
}



/// Generate full convolution kernel from coefficients.

template <typename T,
	  typename Block>
vsip::Vector<T>
kernel_from_coeff(
  vsip::symmetry_type          symmetry,
  vsip::const_Vector<T, Block> coeff)
{
  using vsip::Domain;
  using vsip::length_type;

  length_type M2 = coeff.size();
  length_type M;

  if (symmetry == vsip::nonsym)
    M = coeff.size();
  else if (symmetry == vsip::sym_even_len_odd)
    M = 2*coeff.size()-1;
  else /* (symmetry == vsip::sym_even_len_even) */
    M = 2*coeff.size();

  vsip::Vector<T> kernel(M, T());

  if (symmetry == vsip::nonsym)
  {
    kernel = coeff;
  }
  else if (symmetry == vsip::sym_even_len_odd)
  {
    kernel(Domain<1>(0,  1, M2))   = coeff;
    kernel(Domain<1>(M2, 1, M2-1)) = coeff(Domain<1>(M2-2, -1, M2-1));
  }
  else /* (symmetry == sym_even_len_even) */
  {
    kernel(Domain<1>(0,  1, M2)) = coeff;
    kernel(Domain<1>(M2, 1, M2)) = coeff(Domain<1>(M2-1, -1, M2));
  }

  return kernel;
}



template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
conv(
  vsip::symmetry_type           sym,
  vsip::support_region_type     sup,
  vsip::const_Vector<T, Block1> coeff,
  vsip::const_Vector<T, Block2> in,
  vsip::Vector<T, Block3>       out,
  vsip::length_type             D)
{
  using vsip::index_type;
  using vsip::length_type;
  using vsip::stride_type;
  using vsip::Vector;
  using vsip::const_Vector;
  using vsip::Domain;
  using vsip::unbiased;

  using vsip::impl::convert_to_local;
  using vsip::impl::Working_view_holder;

  typedef typename vsip::impl::Scalar_of<T>::type scalar_type;

  Working_view_holder<const_Vector<T, Block1> > w_coeff(coeff);
  Working_view_holder<const_Vector<T, Block2> > w_in(in);
  Working_view_holder<Vector<T, Block3> >       w_out(out);

  Vector<T> kernel = kernel_from_coeff(sym, w_coeff.view);

  length_type M = kernel.size(0);
  length_type N = in.size(0);
  length_type P = out.size(0);

  stride_type shift      = conv_expected_shift(sup, M);

  // expected_P == conv_output_size(sup, M, N, D) == P;
  assert(conv_output_size(sup, M, N, D) == P);

  Vector<T> sub(M);

  // Check result
  for (index_type i=0; i<P; ++i)
  {
    sub = T();
    index_type pos = i*D + shift;

    if (pos+1 < M)
      sub(Domain<1>(0, 1, pos+1)) = w_in.view(Domain<1>(pos, -1, pos+1));
    else if (pos >= N)
    {
      index_type start = pos - N + 1;
      sub(Domain<1>(start, 1, M-start)) = w_in.view(Domain<1>(N-1, -1, M-start));
    }
    else
      sub = w_in.view(Domain<1>(pos, -1, M));
      
    w_out.view(i) = dot(kernel, sub);
  }
}

} // namespace ref
} // namespace vsip_csl

#endif // VSIP_CSL_REF_CORR_HPP
