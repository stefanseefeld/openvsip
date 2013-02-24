/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/fft/ct_workspace.cpp
    @author  Stefan Seefeld
    @date    2006-11-30
    @brief   VSIPL++ Library: FFT common infrastructure used by all 
    implementations.
*/

#ifndef VSIP_CORE_FFT_CT_WORKSPACE_HPP
#define VSIP_CORE_FFT_CT_WORKSPACE_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/fft/backend.hpp>
#include <vsip/core/view_traits.hpp>
#include <vsip/core/adjust_layout.hpp>
#include <vsip/core/allocation.hpp>
#include <vsip/core/equal.hpp>
#include <vsip/dda.hpp>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace fft
{

template <typename InT,
	  typename OutT>
struct Select_fft_size
{
  static length_type exec(length_type /*in_size*/, length_type out_size)
  { return out_size; }
};

template <typename T>
struct Select_fft_size<T, std::complex<T> >
{
  static length_type exec(length_type in_size, length_type /*out_size*/)
  { return in_size; }
};

template <typename InT,
	  typename OutT>
inline length_type
select_fft_size(length_type in_size, length_type out_size)
{
  return Select_fft_size<InT, OutT>::exec(in_size, out_size);
}



/// This provides the temporary data as well as the
/// conversion logic from blocks to arrays as expected
/// by fft backends.
template <dimension_type D, typename I, typename O>
class Ct_workspace;

template <typename InT,
	  typename OutT>
class Ct_workspace<1, InT, OutT>
{
  typedef typename scalar_of<OutT>::type scalar_type;

public:
  template <typename BE>
  Ct_workspace(BE* /*backend*/, Domain<1> const &in, Domain<1> const &out,
	       scalar_type scale)
    : scale_(scale),
      input_buffer_(in.size()),
      output_buffer_(out.size())
  {
  }
  
  template <typename BE, typename Block0, typename Block1>
  void out_of_place(BE *backend,
		    const_Vector<InT, Block0>& in,
		    Vector<OutT, Block1>& out)
  {
    {
      dda::Data<Block0, dda::in> in_data(in.block());
      dda::Data<Block1, dda::out> out_data(out.block());

      backend->out_of_place(in_data.ptr(),  in_data.stride(0),
			    out_data.ptr(), out_data.stride(0),
			    select_fft_size<InT, OutT>(in_data.size(0), out_data.size(0)));
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && !almost_equal(scale_, scalar_type(1.)))
      out *= scale_;
  }

  template <typename BE, typename BlockT>
  void in_place(BE *backend, Vector<OutT, BlockT> inout)
  {
    {
      // Create a 'direct data accessor', adjusting the block layout if
      // necessary.
      dda::Data<BlockT, dda::inout> inout_data(inout.block()); // input_buffer_.get());
    
      // Call the backend.
      backend->in_place(inout_data.ptr(),
			inout_data.stride(0), inout_data.size(0));
    }
    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && !almost_equal(scale_, scalar_type(1.)))
      inout *= scale_;
  }

private:
  scalar_type scale_;
  aligned_array<InT> input_buffer_;
  aligned_array<OutT> output_buffer_;
};



template <typename InT,
	  typename OutT>
class Ct_workspace<2, InT, OutT>
{
  typedef typename scalar_of<OutT>::type scalar_type;

public:
  template <typename BE>
  Ct_workspace(BE* /*backend*/, Domain<2> const &in, Domain<2> const &out,
	       scalar_type scale)
    : scale_(scale),
      input_buffer_(in.size()),
      output_buffer_(out.size())
  {
  }
  
  template <typename BE, typename Block0, typename Block1>
  void out_of_place(BE *backend,
		    const_Matrix<InT, Block0> in,
		    Matrix<OutT, Block1> out)
  {
    {
      dda::Data<Block0, dda::in> in_data(in.block());
      dda::Data<Block1, dda::out> out_data(out.block());

      backend->out_of_place(in_data.ptr(), in_data.stride(0), in_data.stride(1),
			    out_data.ptr(), out_data.stride(0), out_data.stride(1),
			    select_fft_size<InT, OutT>(in_data.size(0), out_data.size(0)),
			    select_fft_size<InT, OutT>(in_data.size(1), out_data.size(1)));
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && !almost_equal(scale_, scalar_type(1.)))
      out *= scale_;
  }

  template <typename BE, typename BlockT>
  void in_place(BE *backend, Matrix<OutT, BlockT> inout)
  {
    {
      // Create a 'direct data accessor', adjusting the block layout if
      // necessary.
      dda::Data<BlockT, dda::inout> inout_data(inout.block()); // FIXME (split): input_buffer_.get());
    
      // Call the backend.
      backend->in_place(inout_data.ptr(),
			inout_data.stride(0), inout_data.stride(1),
			inout_data.size(0), inout_data.size(1));
    }
    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && !almost_equal(scale_, scalar_type(1.)))
      inout *= scale_;
  }

private:
  scalar_type scale_;
  aligned_array<InT> input_buffer_;
  aligned_array<OutT> output_buffer_;
};


template <typename InT,
	  typename OutT>
class Ct_workspace<3, InT, OutT>
{
  typedef typename scalar_of<OutT>::type scalar_type;

public:
  template <typename BE>
  Ct_workspace(BE* /*backend*/, Domain<3> const &in, Domain<3> const &out,
	       scalar_type scale)
    : scale_(scale),
      input_buffer_(in.size()),
      output_buffer_(out.size())
  {
  }
  
  template <typename BE, typename Block0, typename Block1>
  void out_of_place(BE *backend,
		    const_Tensor<InT, Block0> in,
		    Tensor<OutT, Block1> out)
  {
    {
      dda::Data<Block0, dda::in> in_data(in.block());
      dda::Data<Block1, dda::out> out_data(out.block());

      backend->out_of_place(in_data.ptr(), 
			    in_data.stride(0), in_data.stride(1), in_data.stride(2),
			    out_data.ptr(),
			    out_data.stride(0), out_data.stride(1), out_data.stride(2),
			    select_fft_size<InT, OutT>(in_data.size(0), out_data.size(0)),
			    select_fft_size<InT, OutT>(in_data.size(1), out_data.size(1)),
			    select_fft_size<InT, OutT>(in_data.size(2), out_data.size(2)));
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && !almost_equal(scale_, scalar_type(1.)))
      out *= scale_;
  }

  template <typename BE, typename BlockT>
  void in_place(BE *backend, Vector<OutT, BlockT> inout)
  {
    {
      // Create a 'direct data accessor', adjusting the block layout if
      // necessary.
      dda::Data<BlockT, dda::inout> inout_data(inout.block(), input_buffer_.get());
    
      // Call the backend.
      backend->in_place(inout_data.ptr(),
			inout_data.stride(0), inout_data.stride(1), inout_data.stride(2),
			inout_data.size(0), inout_data.size(1), inout_data.size(2));
    }
    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && !almost_equal(scale_, scalar_type(1.)))
      inout *= scale_;
  }

private:
  scalar_type scale_;
  aligned_array<InT> input_buffer_;
  aligned_array<OutT> output_buffer_;
};

} // namespace vsip::impl::fft
} // namespace vsip::impl
} // namespace vsip

#endif
