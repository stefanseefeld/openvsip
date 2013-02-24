/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   User-defined polar to rectangular interpolation kernel for SSAR images.

#ifndef cuda_interp_hpp_
#define cuda_interp_hpp_

#include <vsip/support.hpp>
#include <vsip_csl/udk.hpp>
#include <vsip/opt/cuda/dda.hpp>

namespace cuda
{

extern void
interpolate(unsigned int const *indices,  // m x n
	    float const *       window,   // m x n x I
	    std::complex<float> const *in,// m x n
	    std::complex<float> *out,     // m x nx
	    size_t              depth,    // I
	    size_t              wstride,  // I + pad bytes
	    size_t              rows,     // m
	    size_t              cols_in,  // n
	    size_t              cols_out);// nx

extern void
interpolate_with_shift(unsigned int const *indices,  // m x n
		       float const *       window,   // m x n x I
		       std::complex<float> const *in,// m x n
		       std::complex<float> *out,     // m x nx
		       size_t              depth,    // I
		       size_t              wstride,  // I + pad bytes
		       size_t              rows,     // m
		       size_t              cols_in,  // n
		       size_t              cols_out);// nx

} // namespace cuda

template <typename B1, typename B2, typename B3, typename B4>
void interpolate(vsip::impl::cuda::dda::Data<B1 const> &indices,
		 vsip::impl::cuda::dda::Data<B2 const> &window,
		 vsip::impl::cuda::dda::Data<B3 const> &in,
		 vsip::impl::cuda::dda::Data<B4> &out)
{
  cuda::interpolate(indices.ptr(), window.ptr(), in.ptr(), out.ptr(),
		    window.size(2), window.stride(0),
		    in.size(1), in.size(0), out.size(0));
}

template <typename B1, typename B2, typename B3, typename B4>
void interpolate_with_shift(vsip::impl::cuda::dda::Data<B1 const> &indices,
			    vsip::impl::cuda::dda::Data<B2 const> &window,
			    vsip::impl::cuda::dda::Data<B3 const> &in,
			    vsip::impl::cuda::dda::Data<B4> &out)
{
  cuda::interpolate_with_shift(indices.ptr(), window.ptr(), in.ptr(), out.ptr(),
			       window.size(2), window.stride(0),
			       in.size(1), in.size(0), out.size(0));
}

#endif
