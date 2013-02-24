/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/fft/workspace.cpp
    @author  Stefan Seefeld
    @date    2006-02-21
    @brief   VSIPL++ Library: FFT common infrastructure used by all 
    implementations.
*/

#ifndef VSIP_OPT_FFT_WORKSPACE_HPP
#define VSIP_OPT_FFT_WORKSPACE_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/fft/backend.hpp>
#include <vsip/core/view_traits.hpp>
#include <vsip/core/adjust_layout.hpp>
#include <vsip/core/allocation.hpp>
#include <vsip/core/equal.hpp>
#include <vsip/core/expr/binary_operators.hpp>
#include <vsip/opt/rt_extdata.hpp>

#if VSIP_IMPL_CUDA_FFT
#include <vsip/opt/cuda/dda.hpp>
#endif

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace fft
{

template <typename T, typename Block>
inline void
scale_block(T scalar, Block &block)
{
  typedef expr::Scalar<Block::dim, T> scalar_block_type;
  typedef expr::Binary<expr::op::Mult, Block, scalar_block_type, true> expr_block_type;

  scalar_block_type scalar_block(scalar);
  expr_block_type   expr_block(block, scalar_block);
  assign<Block::dim>(block, expr_block);
}



template <typename       T,
	  bool           ComputeInSize,
	  typename       BE,
	  dimension_type Dim>
inline length_type
inout_size(BE* backend, Domain<Dim> const& dom)
{
  Rt_layout<Dim> rtl_in, rtl_out;

  rtl_in.pack    = stride_unit_dense;
  rtl_in.order   = Rt_tuple(0, 1, 2);
  rtl_in.complex = cmplx_inter_fmt;
  rtl_in.align   = 0;

  rtl_out.pack    = stride_unit_dense;
  rtl_out.order   = Rt_tuple(0, 1, 2);
  rtl_out.complex = cmplx_inter_fmt;
  rtl_out.align   = 0;

  backend->query_layout(rtl_in, rtl_out);

  if (ComputeInSize)
  {
    Rt_layout<Dim> rtl_inout;

    rtl_inout.pack    = stride_unit_dense;
    rtl_inout.order   = Rt_tuple(0, 1, 2);
    rtl_inout.complex = cmplx_inter_fmt;
    rtl_inout.align   = 0;

    backend->query_layout(rtl_inout);

    Applied_layout<Rt_layout<Dim> > l_in(rtl_in, extent(dom), sizeof(T));
    Applied_layout<Rt_layout<Dim> > l_inout(rtl_inout, extent(dom), sizeof(T));

    return std::max(l_in.total_size(), l_inout.total_size());
  }
  else
  {
    Applied_layout<Rt_layout<Dim> > layout(rtl_out, extent(dom), sizeof(T));
    return layout.total_size();
  }
}



// Determine in-out sizes, for Out-of-Place Only

template <typename       T,
	  bool           ComputeInSize,
	  typename       BE,
	  dimension_type Dim>
inline length_type
inout_size_opo(BE* backend, Domain<Dim> const& dom)
{
  Rt_layout<Dim> rtl_in, rtl_out;

  rtl_in.pack    = stride_unit_dense;
  rtl_in.order   = Rt_tuple(0, 1, 2);
  rtl_in.complex = cmplx_inter_fmt;
  rtl_in.align   = 0;

  rtl_out.pack    = stride_unit_dense;
  rtl_out.order   = Rt_tuple(0, 1, 2);
  rtl_out.complex = cmplx_inter_fmt;
  rtl_out.align   = 0;

  backend->query_layout(rtl_in, rtl_out);

  if (ComputeInSize)
  {
    Applied_layout<Rt_layout<Dim> > layout(rtl_in, extent(dom), sizeof(T));
    return layout.total_size();
  }
  else
  {
    Applied_layout<Rt_layout<Dim> > layout(rtl_out, extent(dom), sizeof(T));
    return layout.total_size();
  }
}



/// This provides the temporary data as well as the
/// conversion logic from blocks to arrays as expected
/// by fft backends.
template <dimension_type D, typename I, typename O>
class workspace;

template <typename T>
class workspace<1, std::complex<T>, std::complex<T> >
{
public:
  template <typename BE>
  workspace(BE* backend, Domain<1> const &in, Domain<1> const &out, T scale)
    : scale_(scale),
      input_buffer_(in.size()),
      output_buffer_(out.size())
  {
    // Check if Backend supports interleaved unit-stride fastpath:
    {
      Rt_layout<1> ref;
      ref.pack    = stride_unit_dense;
      ref.order   = Rt_tuple(row1_type());
      ref.complex = cmplx_inter_fmt;
      ref.align   = 0;
      Rt_layout<1> rtl_in(ref);
      Rt_layout<1> rtl_out(ref);
      backend->query_layout(rtl_in, rtl_out);
      this->inter_fastpath_ok_ = 
	rtl_in.pack == ref.pack       && rtl_out.pack == ref.pack &&
	// rtl_in.order == ref.order     && rtl_out.order == ref.order &&
	rtl_in.complex == ref.complex && rtl_out.complex == ref.complex &&
	!backend->requires_copy(rtl_in);
    }
    // Check if Backend supports split unit-stride fastpath:
    {
      Rt_layout<1> ref;
      ref.pack    = stride_unit_dense;
      ref.order   = Rt_tuple(row1_type());
      ref.complex = cmplx_split_fmt;
      ref.align   = 0;
      Rt_layout<1> rtl_in(ref);
      Rt_layout<1> rtl_out(ref);
      backend->query_layout(rtl_in, rtl_out);
      this->split_fastpath_ok_ = 
	rtl_in.pack == ref.pack       && rtl_out.pack == ref.pack &&
	// rtl_in.order == ref.order     && rtl_out.order == ref.order &&
	rtl_in.complex == ref.complex && rtl_out.complex == ref.complex &&
	!backend->requires_copy(rtl_in);
    }
  }
  
  template <typename BE, typename Block0, typename Block1>
  void out_of_place_blk(
    BE*           backend,
    Block0 const& in,
    Block1&       out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<1> rtl_in  = block_layout<1>(in);
    Rt_layout<1> rtl_out = block_layout<1>(out);
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);

#if VSIP_IMPL_CUDA_FFT
    if (backend->supports_cuda_memory())
    {
      // The data is either already in device memory or must be brought 
      // over before calling the backend.
      cuda::dda::Rt_data_ptr<Block0 const> dev_in(in, rtl_in, SYNC_IN);
      cuda::dda::Rt_data_ptr<Block1> dev_out(out, rtl_out, SYNC_OUT);

      if (rtl_in.complex == cmplx_inter_fmt) 
	backend->out_of_place(dev_in.data().as_inter(), 1,
			      dev_out.data().as_inter(), 1,
			      out.size());
      else
	backend->out_of_place(dev_in.data().as_split(), 1,
			      dev_out.data().as_split(), 1,
			      out.size());
    }
    else 
#endif
    {
      typedef typename Proper_type_of<Block0>::type proper_block0_type;

      // Check whether the input buffer will be destroyed (and hence
      // should be a copy), or whether the input block aliases the
      // output (and hence should be a copy).
      //
      // The input and output may alias when using return-block
      // optimization for by-value Fft: 'A = fft(A)'.
      sync_action_type in_sync =
        (backend->requires_copy(rtl_in) || is_alias(in, out))
        ? SYNC_IN_NOPRESERVE
        : SYNC_IN; 

      {
        // Create a 'direct data accessor', adjusting the block layout if necessary.
        Rt_ext_data<proper_block0_type> in_ext(in, rtl_in, in_sync,
          input_buffer_.get());
        Rt_ext_data<Block1> out_ext(out, rtl_out, SYNC_OUT,
          output_buffer_.get());
    
        // Call the backend.
        assert(rtl_in.complex == rtl_out.complex);
        if (rtl_in.complex == cmplx_inter_fmt) 
          backend->out_of_place(in_ext.data().as_inter(), in_ext.stride(0),
            out_ext.data().as_inter(), out_ext.stride(0),
            in_ext.size(0));
        else
          backend->out_of_place(in_ext.data().as_split(), in_ext.stride(0),
            out_ext.data().as_split(), out_ext.stride(0),
            in_ext.size(0));
      }
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      scale_block(scale_, out);
  }

  template <typename BE, typename Block0, typename Block1>
  void out_of_place(BE *backend,
		    const_Vector<std::complex<T>, Block0>& in,
		    Vector<std::complex<T>, Block1>& out)
    const
  {
#if 0
    // Unfortunately, calling out_of_place_blk adds performance
    // overhead for small FFT sizes.
    //
    // 0703012: On a 2 GHz PPC 970FX, with FFTW 3.1.2, using 'fft -1'
    //  - implement inline.      16-point FFT: 1751 MFLOP/s (baseline)
    //  - call out_of_place_blk. 16-point FFT: 1186 MFLOP/s (-32.2%)

    out_of_place_blk<BE, Block0, Block1>(backend, in.block(), out.block());
#else
    // Find out about the blocks's actual layout.
    Rt_layout<1> rtl_in = block_layout<1>(in.block()); 
    Rt_layout<1> rtl_out = block_layout<1>(out.block()); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);

#if VSIP_IMPL_CUDA_FFT
    if (backend->supports_cuda_memory())
    {
      // The data is either already in device memory or must be brought 
      // over before calling the backend.
      cuda::dda::Rt_data_ptr<Block0 const> dev_in(in.block(), rtl_in, SYNC_IN);
      cuda::dda::Rt_data_ptr<Block1> dev_out(out.block(), rtl_out, SYNC_OUT);
      
      if (rtl_in.complex == cmplx_inter_fmt) 
	backend->out_of_place(dev_in.data().as_inter(),
			      dev_in.stride(0),
			      dev_out.data().as_inter(),
			      dev_out.stride(0),
			      dev_out.size(0));
      else
	backend->out_of_place(dev_in.data().as_split(),
			      dev_in.stride(0),
			      dev_out.data().as_split(),
			      dev_out.stride(0),
			      dev_out.size(0));
    }
    else
#endif
    {
      typedef typename Block_layout<Block0>::complex_type complex_type;
      typedef Layout<1, row1_type, Stride_unit, complex_type> LP;

      if (Ext_data_cost<Block0, LP>::value == 0 &&
	Ext_data_cost<Block1, LP>::value == 0 &&
	Type_equal<complex_type, Cmplx_inter_fmt>::value ?
        this->inter_fastpath_ok_ : this->split_fastpath_ok_)
      {
        Ext_data<Block0, LP> in_ext (in.block(),  SYNC_IN);
        Ext_data<Block1, LP> out_ext(out.block(), SYNC_OUT);

        backend->out_of_place(in_ext.data(),  1,
          out_ext.data(), 1,
          in_ext.size(0));
      }
      else
      {
        // General-path (using RT Ext_data).

        // Check whether the input buffer will be destroyed.
        sync_action_type in_sync = backend->requires_copy(rtl_in)
          ? SYNC_IN_NOPRESERVE
          : SYNC_IN; 

        {
          // Create a 'direct data accessor', adjusting the block layout if
          // necessary.
          Rt_ext_data<Block0> in_ext(in.block(), rtl_in, in_sync,
            input_buffer_.get());
          Rt_ext_data<Block1> out_ext(out.block(), rtl_out, SYNC_OUT,
            output_buffer_.get());
    
          // Call the backend.
          assert(rtl_in.complex == rtl_out.complex);
          if (rtl_in.complex == cmplx_inter_fmt) 
            backend->out_of_place(in_ext.data().as_inter(), in_ext.stride(0),
              out_ext.data().as_inter(), out_ext.stride(0),
              in_ext.size(0));
          else
            backend->out_of_place(in_ext.data().as_split(), in_ext.stride(0),
              out_ext.data().as_split(), out_ext.stride(0),
              in_ext.size(0));
        }
      }
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      out *= scale_;
#endif
  }

  template <typename BE, typename BlockT>
  void in_place(BE *backend, Vector<std::complex<T>,BlockT> inout)
  {
    // Find out about the block's actual layout.
    Rt_layout<1> rtl_inout = block_layout<1>(inout.block()); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_inout);

#if VSIP_IMPL_CUDA_FFT
    if (backend->supports_cuda_memory())
    {
      // The data is either already on GPU or must be brought over before
      // calling the backend.
      cuda::dda::Rt_data_ptr<BlockT> dev_inout(inout.block(), rtl_inout, SYNC_INOUT);

      if (rtl_inout.complex == cmplx_inter_fmt) 
	backend->in_place(dev_inout.data().as_inter(),
			  dev_inout.stride(0), dev_inout.size(0));
      else
	backend->in_place(dev_inout.data().as_split(),
			  dev_inout.stride(0), dev_inout.size(0));
    }
    else
#endif
    {
      // Create a 'direct data accessor', adjusting the block layout if necessary.
      Rt_ext_data<BlockT> inout_ext(inout.block(), rtl_inout, SYNC_INOUT,
				    input_buffer_.get());
    
      // Call the backend.
      if (rtl_inout.complex == cmplx_inter_fmt) 
	backend->in_place(inout_ext.data().as_inter(),
			  inout_ext.stride(0), inout_ext.size(0));
      else
	backend->in_place(inout_ext.data().as_split(),
			  inout_ext.stride(0), inout_ext.size(0));
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      inout *= scale_;
  }

  T scale() const { return scale_; }

private:
  T scale_;
  aligned_array<std::complex<T> > input_buffer_;
  aligned_array<std::complex<T> > output_buffer_;
  bool inter_fastpath_ok_;
  bool split_fastpath_ok_;
};


template <typename T>
class workspace<1, T, std::complex<T> >
{
public:
  template <typename BE>
  workspace(BE*, Domain<1> const &in, Domain<1> const &out, T scale)
    : scale_(scale),
      input_buffer_(in.size()),
      output_buffer_(out.size())
  {}
  
  template <typename BE, typename Block0, typename Block1>
  void out_of_place(BE *backend,
		    const_Vector<T, Block0> in,
		    Vector<std::complex<T>, Block1> out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<1> rtl_in = block_layout<1>(in.block()); 
    Rt_layout<1> rtl_out = block_layout<1>(out.block()); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);

#if VSIP_IMPL_CUDA_FFT
    if (backend->supports_cuda_memory())
    {
      // The data is either already in device memory or must be brought 
      // over before calling the backend.
      cuda::dda::Rt_data_ptr<Block0 const> dev_in(in.block(), rtl_in, SYNC_IN);
      cuda::dda::Rt_data_ptr<Block1> dev_out(out.block(), rtl_out, SYNC_OUT);

      if (rtl_in.complex == cmplx_inter_fmt) 
	backend->out_of_place(dev_in.data().as_real(),
			      dev_in.stride(0),
			      dev_out.data().as_inter(),
			      dev_out.stride(0),
			      dev_out.size(0));
      else
	backend->out_of_place(dev_in.data().as_real(),
			      dev_in.stride(0),
			      dev_out.data().as_split(),
			      dev_out.stride(0),
			      dev_out.size(0));
    }
    else
#endif
    {
      // Check whether the input buffer will be destroyed.
      sync_action_type in_sync = backend->requires_copy(rtl_in)
        ? SYNC_IN_NOPRESERVE
        : SYNC_IN; 
      {
        // Create a 'direct data accessor', adjusting the block layout if necessary.
        Rt_ext_data<Block0> in_ext(in.block(), rtl_in, in_sync,
          input_buffer_.get());
        Rt_ext_data<Block1> out_ext(out.block(), rtl_out, SYNC_OUT,
          output_buffer_.get());
    
        // Call the backend.
        if (rtl_out.complex == cmplx_inter_fmt) 
          backend->out_of_place(in_ext.data().as_real(), in_ext.stride(0),
            out_ext.data().as_inter(), out_ext.stride(0),
            in_ext.size(0));
        else
          backend->out_of_place(in_ext.data().as_real(), in_ext.stride(0),
            out_ext.data().as_split(), out_ext.stride(0),
            in_ext.size(0));
      }
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      out *= scale_;
  }

  template <typename BE, typename Block0, typename Block1>
  void out_of_place_blk(
    BE*           backend,
    Block0 const& in,
    Block1&       out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<1> rtl_in = block_layout<1>(in); 
    Rt_layout<1> rtl_out = block_layout<1>(out); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);

#if VSIP_IMPL_CUDA_FFT
    if (backend->supports_cuda_memory())
    {
      // The data is either already in device memory or must be brought 
      // over before calling the backend.
      cuda::dda::Rt_data_ptr<Block0 const> dev_in(in, rtl_in, SYNC_IN);
      cuda::dda::Rt_data_ptr<Block1> dev_out(out, rtl_out, SYNC_OUT);

      if (rtl_in.complex == cmplx_inter_fmt) 
	backend->out_of_place(dev_in.data().as_real(),
			      dev_in.stride(0),
			      dev_out.data().as_inter(),
			      dev_out.stride(0),
			      dev_out.size(0));
      else
	backend->out_of_place(dev_in.data().as_real(),
			      dev_in.stride(0),
			      dev_out.data().as_split(),
			      dev_out.stride(0),
			      dev_out.size(0));
    }
    else 
#endif
    {
      typedef typename Proper_type_of<Block0>::type block0_type;

      // Check whether the input buffer will be destroyed.
      sync_action_type in_sync = backend->requires_copy(rtl_in)
        ? SYNC_IN_NOPRESERVE
        : SYNC_IN; 
      {
        // Create a 'direct data accessor', adjusting the block layout if necessary.
        Rt_ext_data<block0_type> in_ext(in, rtl_in, in_sync,
          input_buffer_.get());
        Rt_ext_data<Block1> out_ext(out, rtl_out, SYNC_OUT,
          output_buffer_.get());
    
        // Call the backend.
        if (rtl_out.complex == cmplx_inter_fmt) 
          backend->out_of_place(in_ext.data().as_real(), in_ext.stride(0),
            out_ext.data().as_inter(), out_ext.stride(0),
            in_ext.size(0));
        else
          backend->out_of_place(in_ext.data().as_real(), in_ext.stride(0),
            out_ext.data().as_split(), out_ext.stride(0),
            in_ext.size(0));
      }
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      scale_block(scale_, out);
  }

  T scale() const { return scale_; }

private:
  T scale_;
  aligned_array<T> input_buffer_;
  aligned_array<std::complex<T> > output_buffer_;
};

template <typename T>
class workspace<1, std::complex<T>, T>
{
public:
  template <typename BE>
  workspace(BE*, Domain<1> const &in, Domain<1> const &out, T scale)
    : scale_(scale),
      input_buffer_(in.size()),
      output_buffer_(out.size())
  {}
  
  template <typename BE, typename Block0, typename Block1>
  void out_of_place(BE *backend,
		    const_Vector<std::complex<T>, Block0> in,
		    Vector<T, Block1> out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<1> rtl_in = block_layout<1>(in.block()); 
    Rt_layout<1> rtl_out = block_layout<1>(out.block()); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);

#if VSIP_IMPL_CUDA_FFT
    if (backend->supports_cuda_memory())
    {
      cuda::dda::Rt_data_ptr<Block0 const> dev_in(in.block(), rtl_in, SYNC_IN);
      cuda::dda::Rt_data_ptr<Block1> dev_out(out.block(), rtl_out, SYNC_OUT);

      if (rtl_in.complex == cmplx_inter_fmt) 
	backend->out_of_place(dev_in.data().as_inter(),
			      dev_in.stride(0),
			      dev_out.data().as_real(),
			      dev_out.stride(0),
			      dev_out.size(0));
      else
	backend->out_of_place(dev_in.data().as_split(),
			      dev_in.stride(0),
			      dev_out.data().as_real(),
			      dev_out.stride(0),
			      dev_out.size(0));
    }
    else
#endif
    {
      // Check whether the input buffer will be destroyed.
      sync_action_type in_sync = backend->requires_copy(rtl_in)
        ? SYNC_IN_NOPRESERVE
        : SYNC_IN;
      { 
        // Create a 'direct data accessor', adjusting the block layout if necessary.
        Rt_ext_data<Block0> in_ext(in.block(), rtl_in, in_sync,
          input_buffer_.get());
        Rt_ext_data<Block1> out_ext(out.block(), rtl_out, SYNC_OUT,
          output_buffer_.get());
    
        // Call the backend.
        if (rtl_in.complex == cmplx_inter_fmt) 
          backend->out_of_place(in_ext.data().as_inter(), in_ext.stride(0),
            out_ext.data().as_real(), out_ext.stride(0),
            out_ext.size(0));
        else
          backend->out_of_place(in_ext.data().as_split(), in_ext.stride(0),
            out_ext.data().as_real(), out_ext.stride(0),
            out_ext.size(0));
      }
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      out *= scale_;
  }

  template <typename BE, typename Block0, typename Block1>
  void out_of_place_blk(
    BE*           backend,
    Block0 const& in,
    Block1&       out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<1> rtl_in = block_layout<1>(in); 
    Rt_layout<1> rtl_out = block_layout<1>(out); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);

#if VSIP_IMPL_CUDA_FFT
    if (backend->supports_cuda_memory())
    {
      cuda::dda::Rt_data_ptr<Block0 const> dev_in(in, rtl_in, SYNC_IN);
      cuda::dda::Rt_data_ptr<Block1> dev_out(out, rtl_out, SYNC_OUT);

      if (rtl_in.complex == cmplx_inter_fmt) 
	backend->out_of_place(dev_in.data().as_inter(), 1,
			      dev_out.data().as_real(), 1,
			      out.size());
      else
	backend->out_of_place(dev_in.data().as_split(), 1,
			      dev_out.data().as_real(), 1,
			      out.size());
    }
    else 
#endif
    {
      // Check whether the input buffer will be destroyed.
      sync_action_type in_sync = backend->requires_copy(rtl_in)
        ? SYNC_IN_NOPRESERVE
        : SYNC_IN;
      { 
        // Create a 'direct data accessor', adjusting the block layout if necessary.
        Rt_ext_data<Block0> in_ext(in, rtl_in, in_sync,
          input_buffer_.get());
        Rt_ext_data<Block1> out_ext(out, rtl_out, SYNC_OUT,
          output_buffer_.get());
    
        // Call the backend.
        if (rtl_in.complex == cmplx_inter_fmt) 
          backend->out_of_place(in_ext.data().as_inter(), in_ext.stride(0),
            out_ext.data().as_real(), out_ext.stride(0),
            out_ext.size(0));
        else
          backend->out_of_place(in_ext.data().as_split(), in_ext.stride(0),
            out_ext.data().as_real(), out_ext.stride(0),
            out_ext.size(0));
      }
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      scale_block(scale_, out);
  }

  T scale() const { return scale_; }

private:
  T scale_;
  aligned_array<std::complex<T> > input_buffer_;
  aligned_array<T> output_buffer_;
};

template <typename T>
class workspace<2, std::complex<T>, std::complex<T> >
{
public:
  template <typename BE>
  workspace(BE* backend, Domain<2> const& in, Domain<2> const& out, T scale)
    : scale_(scale),
      input_buffer_size_ (inout_size<std::complex<T>, true> (backend, in)),
      output_buffer_size_(inout_size<std::complex<T>, false>(backend, out)),
      input_buffer_      (input_buffer_size_),
      output_buffer_     (output_buffer_size_)
  {}
  
  template <typename BE, typename Block0, typename Block1>
  void out_of_place(BE *backend,
		    const_Matrix<std::complex<T>, Block0> in,
		    Matrix<std::complex<T>, Block1> out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<2> rtl_in = block_layout<2>(in.block()); 
    Rt_layout<2> rtl_out = block_layout<2>(out.block()); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);

#if VSIP_IMPL_CUDA_FFT
    if (backend->supports_cuda_memory())
    {
      cuda::dda::Rt_data_ptr<Block0 const> dev_in(in.block(), rtl_in, SYNC_IN);
      cuda::dda::Rt_data_ptr<Block1> dev_out(out.block(), rtl_out, SYNC_OUT);
      if (rtl_in.complex == cmplx_inter_fmt) 
	backend->out_of_place(dev_in.data().as_inter(),
			      dev_in.stride(0), dev_in.stride(1),
			      dev_out.data().as_inter(),
			      dev_out.stride(0), dev_out.stride(1),
			      dev_out.size(0), dev_out.size(1));
      else
	backend->out_of_place(dev_in.data().as_split(),
			      dev_in.stride(0), dev_in.stride(1),
			      dev_out.data().as_split(),
			      dev_out.stride(0), dev_out.stride(1),
			      dev_out.size(0), dev_out.size(1));
    }
    else
#endif
    {
      // Check whether the input buffer will be destroyed.
      sync_action_type in_sync = backend->requires_copy(rtl_in)
        ? SYNC_IN_NOPRESERVE
        : SYNC_IN; 
      {
        // Create a 'direct data accessor', adjusting the block layout if necessary.
        Rt_ext_data<Block0> in_ext(in.block(), rtl_in, in_sync,
                                   input_buffer_.get());
        Rt_ext_data<Block1> out_ext(out.block(), rtl_out, SYNC_OUT,
                                    output_buffer_.get());
    
        // Call the backend.
        assert(rtl_in.complex == rtl_out.complex);
        if (rtl_in.complex == cmplx_inter_fmt) 
          backend->out_of_place(in_ext.data().as_inter(),
                                in_ext.stride(0), in_ext.stride(1),
                                out_ext.data().as_inter(),
                                out_ext.stride(0), out_ext.stride(1),
                                in_ext.size(0), in_ext.size(1));
        else
          backend->out_of_place(in_ext.data().as_split(),
                                in_ext.stride(0), in_ext.stride(1),
                                out_ext.data().as_split(),
                                out_ext.stride(0), out_ext.stride(1),
                                in_ext.size(0), in_ext.size(1));
      }
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      out *= scale_;
  }

  template <typename BE, typename Block0, typename Block1>
  void out_of_place_blk(
    BE*           backend,
    Block0 const& in,
    Block1&       out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<2> rtl_in  = block_layout<2>(in);
    Rt_layout<2> rtl_out = block_layout<2>(out);
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);

#if VSIP_IMPL_CUDA_FFT
    if (backend->supports_cuda_memory())
    {
      cuda::dda::Rt_data_ptr<Block0 const> dev_in(in, rtl_in, SYNC_IN);
      cuda::dda::Rt_data_ptr<Block1> dev_out(out, rtl_out, SYNC_OUT);

      if (rtl_in.complex == cmplx_inter_fmt)
	backend->out_of_place(dev_in.data().as_inter(),
			      dev_in.stride(0), dev_in.stride(1),
			      dev_out.data().as_inter(),
			      dev_out.stride(0), dev_out.stride(1),
			      dev_out.size(0), dev_out.size(1));
      else
	backend->out_of_place(dev_in.data().as_split(),
			      dev_in.stride(0), dev_in.stride(1),
			      dev_out.data().as_split(),
			      dev_out.stride(0), dev_out.stride(1),
			      dev_out.size(0), dev_out.size(1));
    }
    else
#endif
    {
      typedef typename Proper_type_of<Block0>::type proper_block0_type;

      // Check whether the input buffer will be destroyed (and hence
      // should be a copy), or whether the input block aliases the
      // output (and hence should be a copy).
      //
      // The input and output may alias when using return-block
      // optimization for by-value Fft: 'A = fft(A)'.
      sync_action_type in_sync =
        (backend->requires_copy(rtl_in) || is_alias(in, out))
        ? SYNC_IN_NOPRESERVE
        : SYNC_IN; 
      {
        // Create a 'direct data accessor', adjusting the block layout if necessary.
        Rt_ext_data<proper_block0_type> in_ext(in, rtl_in, in_sync,
                                               input_buffer_.get());
        Rt_ext_data<Block1> out_ext(out, rtl_out, SYNC_OUT,
                                    output_buffer_.get());
    
        // Call the backend.
        assert(rtl_in.complex == rtl_out.complex);
        if (rtl_in.complex == cmplx_inter_fmt) 
          backend->out_of_place(in_ext.data().as_inter(),
                                in_ext.stride(0), in_ext.stride(1),
                                out_ext.data().as_inter(),
                                out_ext.stride(0), out_ext.stride(1),
                                in_ext.size(0), in_ext.size(1));
        else
          backend->out_of_place(in_ext.data().as_split(),
                                in_ext.stride(0), in_ext.stride(1),
                                out_ext.data().as_split(),
                                out_ext.stride(0), out_ext.stride(1),
                                in_ext.size(0), in_ext.size(1));
      }
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      scale_block(scale_, out);
  }

  template <typename BE, typename BlockT>
  void in_place(BE *backend, Matrix<std::complex<T>,BlockT> inout)
  {
    // Find out about the block's actual layout.
    Rt_layout<2> rtl_inout = block_layout<2>(inout.block()); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_inout);

#if VSIP_IMPL_CUDA_FFT
    if (backend->supports_cuda_memory())
    {
      cuda::dda::Rt_data_ptr<BlockT> dev_inout(inout.block(), rtl_inout, SYNC_INOUT);

      if (rtl_inout.complex == cmplx_inter_fmt)
	backend->in_place(dev_inout.data().as_inter(),
			  dev_inout.stride(0), dev_inout.stride(1),
			  dev_inout.size(0), dev_inout.size(1));
      else
	backend->in_place(dev_inout.data().as_split(),
			  dev_inout.stride(0), dev_inout.stride(1),
			  dev_inout.size(0), dev_inout.size(1));
    }
    else 
#endif
    {
      // Create a 'direct data accessor', adjusting the block layout if necessary.
      Rt_ext_data<BlockT> inout_ext(inout.block(), rtl_inout, SYNC_INOUT,
        input_buffer_.get(),
        input_buffer_size_);
    
      // Call the backend.
      if (rtl_inout.complex == cmplx_inter_fmt) 
        backend->in_place(inout_ext.data().as_inter(),
          inout_ext.stride(0), inout_ext.stride(1), 
          inout_ext.size(0), inout_ext.size(1));
      else
        backend->in_place(inout_ext.data().as_split(),
          inout_ext.stride(0), inout_ext.stride(1),
          inout_ext.size(0), inout_ext.size(1));
    }
     
    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      inout *= scale_;
  }

  T scale() const { return scale_; }

private:
  T scale_;
  length_type input_buffer_size_;
  length_type output_buffer_size_;
  aligned_array<std::complex<T> > input_buffer_;
  aligned_array<std::complex<T> > output_buffer_;
};

template <typename T>
class workspace<2, T, std::complex<T> >
{
public:
  template <typename BE>
  workspace(BE* backend, Domain<2> const &in, Domain<2> const &out, T scale)
    : scale_(scale),
      input_buffer_ (inout_size_opo<T,               true> (backend, in)),
      output_buffer_(inout_size_opo<std::complex<T>, false>(backend, out))
  {}
  
  template <typename BE, typename Block0, typename Block1>
  void out_of_place(BE *backend,
		    const_Matrix<T, Block0> in,
		    Matrix<std::complex<T>, Block1> out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<2> rtl_in = block_layout<2>(in.block()); 
    Rt_layout<2> rtl_out = block_layout<2>(out.block()); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);

#if VSIP_IMPL_CUDA_FFT
    if (backend->supports_cuda_memory())
    {
      cuda::dda::Rt_data_ptr<Block0 const> dev_in(in.block(), rtl_in, SYNC_IN);
      cuda::dda::Rt_data_ptr<Block1> dev_out(out.block(), rtl_out, SYNC_OUT);

      if (rtl_in.complex == cmplx_inter_fmt)
	backend->out_of_place(dev_in.data().as_real(),
			      dev_in.stride(0), dev_in.stride(1),
			      dev_out.data().as_inter(),
			      dev_out.stride(0), dev_out.stride(1),
			      dev_out.size(0), dev_out.size(1));
      else
	backend->out_of_place(dev_in.data().as_real(),
			      dev_in.stride(0), dev_in.stride(1),
			      dev_out.data().as_split(),
			      dev_out.stride(0), dev_out.stride(1),
			      dev_out.size(0), dev_out.size(1));
    }
    else
#endif
    {
      // Check whether the input buffer will be destroyed.
      sync_action_type in_sync = backend->requires_copy(rtl_in)
        ? SYNC_IN_NOPRESERVE
        : SYNC_IN; 
      {
        // Create a 'direct data accessor', adjusting the block layout if necessary.
        Rt_ext_data<Block0> in_ext(in.block(), rtl_in, in_sync, 
                                   input_buffer_.get());
        Rt_ext_data<Block1> out_ext(out.block(), rtl_out, SYNC_OUT,
                                    output_buffer_.get());
    
        // Call the backend.
        if (rtl_out.complex == cmplx_inter_fmt) 
          backend->out_of_place(in_ext.data().as_real(),
                                in_ext.stride(0), in_ext.stride(1),
                                out_ext.data().as_inter(),
                                out_ext.stride(0), out_ext.stride(1),
                                in_ext.size(0), in_ext.size(1));
        else
          backend->out_of_place(in_ext.data().as_real(),
                                in_ext.stride(0), in_ext.stride(1),
                                out_ext.data().as_split(),
                                out_ext.stride(0), out_ext.stride(1),
                                in_ext.size(0), in_ext.size(1));
      }
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      out *= scale_;
  }

  template <typename BE, typename Block0, typename Block1>
  void out_of_place_blk(
    BE*           backend,
    Block0 const& in,
    Block1&       out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<2> rtl_in = block_layout<2>(in);
    Rt_layout<2> rtl_out = block_layout<2>(out);
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);

#if VSIP_IMPL_CUDA_FFT
    if (backend->supports_cuda_memory())
    {
      cuda::dda::Rt_data_ptr<Block0 const> dev_in(in, rtl_in, SYNC_IN);
      cuda::dda::Rt_data_ptr<Block1> dev_out(out, rtl_out, SYNC_OUT);

      if (rtl_in.complex == cmplx_inter_fmt)
	backend->out_of_place(dev_in.data().as_real(),
			      dev_in.stride(0), dev_in.stride(1),
			      dev_out.data().as_inter(),
			      dev_out.stride(0), dev_out.stride(1),
			      dev_out.size(0), dev_out.size(1));
      else
	backend->out_of_place(dev_in.data().as_real(),
			      dev_in.stride(0), dev_in.stride(1),
			      dev_out.data().as_split(),
			      dev_out.stride(0), dev_out.stride(1),
			      dev_out.size(0), dev_out.size(1));
    }
    else
#endif
    {
      // Check whether the input buffer will be destroyed.
      sync_action_type in_sync = backend->requires_copy(rtl_in)
        ? SYNC_IN_NOPRESERVE
        : SYNC_IN; 
      {
        // Create a 'direct data accessor', adjusting the block layout if necessary.
        Rt_ext_data<Block0> in_ext(in, rtl_in, in_sync, 
                                   input_buffer_.get());
        Rt_ext_data<Block1> out_ext(out, rtl_out, SYNC_OUT,
                                    output_buffer_.get());
    
        // Call the backend.
        if (rtl_out.complex == cmplx_inter_fmt) 
          backend->out_of_place(in_ext.data().as_real(),
                                in_ext.stride(0), in_ext.stride(1),
                                out_ext.data().as_inter(),
                                out_ext.stride(0), out_ext.stride(1),
                                in_ext.size(0), in_ext.size(1));
        else
          backend->out_of_place(in_ext.data().as_real(),
                                in_ext.stride(0), in_ext.stride(1),
                                out_ext.data().as_split(),
                                out_ext.stride(0), out_ext.stride(1),
                                in_ext.size(0), in_ext.size(1));
      }
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      scale_block(scale_, out);
  }

  T scale() const { return scale_; }

private:
  T scale_;
  aligned_array<T> input_buffer_;
  aligned_array<std::complex<T> > output_buffer_;
};

template <typename T>
class workspace<2, std::complex<T>, T>
{
public:
  template <typename BE>
  workspace(BE* backend, Domain<2> const &in, Domain<2> const &out, T scale)
    : scale_(scale),
      input_buffer_ (inout_size_opo<std::complex<T>, true> (backend, in)),
      output_buffer_(inout_size_opo<T,               false>(backend, out))
  {}
  
  template <typename BE, typename Block0, typename Block1>
  void out_of_place(BE *backend,
		    const_Matrix<std::complex<T>, Block0> in,
		    Matrix<T, Block1> out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<2> rtl_in = block_layout<2>(in.block()); 
    Rt_layout<2> rtl_out = block_layout<2>(out.block()); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);

#if VSIP_IMPL_CUDA_FFT
    if (backend->supports_cuda_memory())
    {
      cuda::dda::Rt_data_ptr<Block0 const> dev_in(in.block(), rtl_in, SYNC_IN);
      cuda::dda::Rt_data_ptr<Block1> dev_out(out.block(), rtl_out, SYNC_OUT);

      if (rtl_in.complex == cmplx_inter_fmt)      
	backend->out_of_place(dev_in.data().as_inter(),
			      dev_in.stride(0), dev_in.stride(1),
			      dev_out.data().as_real(),
			      dev_out.stride(0), dev_out.stride(1),
			      dev_out.size(0), dev_out.size(1));
      else
	backend->out_of_place(dev_in.data().as_split(),
			      dev_in.stride(0), dev_in.stride(1),
			      dev_out.data().as_real(),
			      dev_out.stride(0), dev_out.stride(1),
			      dev_out.size(0), dev_out.size(1));
    }
    else
#endif
    {
      // Check whether the input buffer will be destroyed.
      sync_action_type in_sync = backend->requires_copy(rtl_in)
        ? SYNC_IN_NOPRESERVE
        : SYNC_IN;
      { 
        // Create a 'direct data accessor', adjusting the block layout if necessary.
        Rt_ext_data<Block0> in_ext(in.block(), rtl_in, in_sync,
                                   input_buffer_.get());
        Rt_ext_data<Block1> out_ext(out.block(), rtl_out, SYNC_OUT,
                                    output_buffer_.get());
        // Call the backend.
        if (rtl_in.complex == cmplx_inter_fmt) 
          backend->out_of_place(in_ext.data().as_inter(),
                                in_ext.stride(0), in_ext.stride(1),
                                out_ext.data().as_real(),
                                out_ext.stride(0), out_ext.stride(1),
                                out_ext.size(0), out_ext.size(1));
        else
          backend->out_of_place(in_ext.data().as_split(),
                                in_ext.stride(0), in_ext.stride(1),
                                out_ext.data().as_real(),
                                out_ext.stride(0), out_ext.stride(1),
                                out_ext.size(0), out_ext.size(1));
      }
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      out *= scale_;
  }

  template <typename BE, typename Block0, typename Block1>
  void out_of_place_blk(
    BE*           backend,
    Block0 const& in,
    Block1&       out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<2> rtl_in = block_layout<2>(in); 
    Rt_layout<2> rtl_out = block_layout<2>(out); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);

#if VSIP_IMPL_CUDA_FFT
    if (backend->supports_cuda_memory())
    {
      cuda::dda::Rt_data_ptr<Block0 const> dev_in(in, rtl_in, SYNC_IN);
      cuda::dda::Rt_data_ptr<Block1> dev_out(out, rtl_out, SYNC_OUT);

      if (rtl_in.complex == cmplx_inter_fmt)
	backend->out_of_place(dev_in.data().as_inter(),
			      dev_in.stride(0), dev_in.stride(1),
			      dev_out.data().as_real(),
			      dev_out.stride(0), dev_out.stride(1),
			      dev_out.size(0), dev_out.size(1));
      else
	backend->out_of_place(dev_in.data().as_split(),
			      dev_in.stride(0), dev_in.stride(1),
			      dev_out.data().as_real(),
			      dev_out.stride(0), dev_out.stride(1),
			      dev_out.size(0), dev_out.size(1));
    }
    else
#endif
    {
      // Check whether the input buffer will be destroyed.
      sync_action_type in_sync = backend->requires_copy(rtl_in)
        ? SYNC_IN_NOPRESERVE
        : SYNC_IN;
      { 
        // Create a 'direct data accessor', adjusting the block layout if necessary.
        Rt_ext_data<Block0> in_ext(in, rtl_in, in_sync,
                                   input_buffer_.get());
        Rt_ext_data<Block1> out_ext(out, rtl_out, SYNC_OUT,
                                    output_buffer_.get());
    
        // Call the backend.
        if (rtl_in.complex == cmplx_inter_fmt) 
          backend->out_of_place(in_ext.data().as_inter(),
                                in_ext.stride(0), in_ext.stride(1),
                                out_ext.data().as_real(),
                                out_ext.stride(0), out_ext.stride(1),
                                out_ext.size(0), out_ext.size(1));
        else
          backend->out_of_place(in_ext.data().as_split(),
                                in_ext.stride(0), in_ext.stride(1),
                                out_ext.data().as_real(),
                                out_ext.stride(0), out_ext.stride(1),
                                out_ext.size(0), out_ext.size(1));
      }
    }

    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      scale_block(scale_, out);
  }

  T scale() const { return scale_; }

private:
  T scale_;
  aligned_array<std::complex<T> > input_buffer_;
  aligned_array<T> output_buffer_;
};

template <typename T>
class workspace<3, std::complex<T>, std::complex<T> >
{
public:
  template <typename BE>
  workspace(BE*, Domain<3> const &in, Domain<3> const &out, T scale)
    : scale_(scale),
      input_buffer_(in.size()),
      output_buffer_(out.size())
  {}
  
  template <typename BE, typename Block0, typename Block1>
  void out_of_place_blk(
    BE*           backend,
    Block0 const& in,
    Block1&       out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<3> rtl_in = block_layout<3>(in); 
    Rt_layout<3> rtl_out = block_layout<3>(out); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);

    // Check whether the input buffer will be destroyed.
    sync_action_type in_sync = backend->requires_copy(rtl_in)
      ? SYNC_IN_NOPRESERVE
      : SYNC_IN; 
    {
      // Create a 'direct data accessor', adjusting the block layout if necessary.
      Rt_ext_data<Block0> in_ext(in, rtl_in, in_sync,
				 input_buffer_.get());
      Rt_ext_data<Block1> out_ext(out, rtl_out, SYNC_OUT,
				  output_buffer_.get());
    
      // Call the backend.
      assert(rtl_in.complex == rtl_out.complex);
      if (rtl_in.complex == cmplx_inter_fmt) 
	backend->out_of_place(in_ext.data().as_inter(),
			      in_ext.stride(0),
			      in_ext.stride(1),
			      in_ext.stride(2),
			      out_ext.data().as_inter(),
			      out_ext.stride(0),
			      out_ext.stride(1),
			      out_ext.stride(2),
			      in_ext.size(0),
			      in_ext.size(1),
			      in_ext.size(2));
      else
	backend->out_of_place(in_ext.data().as_split(),
			      in_ext.stride(0),
			      in_ext.stride(1),
			      in_ext.stride(2),
			      out_ext.data().as_split(),
			      out_ext.stride(0),
			      out_ext.stride(1),
			      out_ext.stride(2),
			      in_ext.size(0),
			      in_ext.size(1),
			      in_ext.size(2));
    }
    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      scale_block(scale_, out);
  }

  template <typename BE, typename Block0, typename Block1>
  void out_of_place(BE *backend,
		    const_Tensor<std::complex<T>, Block0> in,
		    Tensor<std::complex<T>, Block1> out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<3> rtl_in = block_layout<3>(in.block()); 
    Rt_layout<3> rtl_out = block_layout<3>(out.block()); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);

    // Check whether the input buffer will be destroyed.
    sync_action_type in_sync = backend->requires_copy(rtl_in)
      ? SYNC_IN_NOPRESERVE
      : SYNC_IN; 
    {
      // Create a 'direct data accessor', adjusting the block layout if necessary.
      Rt_ext_data<Block0> in_ext(in.block(), rtl_in, in_sync,
				 input_buffer_.get());
      Rt_ext_data<Block1> out_ext(out.block(), rtl_out, SYNC_OUT,
				  output_buffer_.get());
    
      // Call the backend.
      assert(rtl_in.complex == rtl_out.complex);
      if (rtl_in.complex == cmplx_inter_fmt) 
	backend->out_of_place(in_ext.data().as_inter(),
			      in_ext.stride(0),
			      in_ext.stride(1),
			      in_ext.stride(2),
			      out_ext.data().as_inter(),
			      out_ext.stride(0),
			      out_ext.stride(1),
			      out_ext.stride(2),
			      in_ext.size(0),
			      in_ext.size(1),
			      in_ext.size(2));
      else
	backend->out_of_place(in_ext.data().as_split(),
			      in_ext.stride(0),
			      in_ext.stride(1),
			      in_ext.stride(2),
			      out_ext.data().as_split(),
			      out_ext.stride(0),
			      out_ext.stride(1),
			      out_ext.stride(2),
			      in_ext.size(0),
			      in_ext.size(1),
			      in_ext.size(2));
    }
    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      out *= scale_;
  }

  template <typename BE, typename BlockT>
  void in_place(BE *backend, Tensor<std::complex<T>,BlockT> inout)
  {
    // Find out about the block's actual layout.
    Rt_layout<3> rtl_inout = block_layout<3>(inout.block()); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_inout);
    {
      // Create a 'direct data accessor', adjusting the block layout if necessary.
      Rt_ext_data<BlockT> inout_ext(inout.block(), rtl_inout, SYNC_INOUT,
				    input_buffer_.get());
    
      // Call the backend.
      if (rtl_inout.complex == cmplx_inter_fmt) 
	backend->in_place(inout_ext.data().as_inter(),
			  inout_ext.stride(0),
			  inout_ext.stride(1),
			  inout_ext.stride(2), 
			  inout_ext.size(0),
			  inout_ext.size(1),
			  inout_ext.size(2));
      else
	backend->in_place(inout_ext.data().as_split(),
			  inout_ext.stride(0),
			  inout_ext.stride(1),
			  inout_ext.stride(2),
			  inout_ext.size(0),
			  inout_ext.size(1),
			  inout_ext.size(2));
    }
    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      inout *= scale_;
  }

  T scale() const { return scale_; }

private:
  T scale_;
  aligned_array<std::complex<T> > input_buffer_;
  aligned_array<std::complex<T> > output_buffer_;
};

template <typename T>
class workspace<3, T, std::complex<T> >
{
public:
  template <typename BE>
  workspace(BE*, Domain<3> const &in, Domain<3> const &out, T scale)
    : scale_(scale),
      input_buffer_(in.size()),
      output_buffer_(out.size())
  {}
  
  template <typename BE, typename Block0, typename Block1>
  void out_of_place_blk(
    BE*           backend,
    Block0 const& in,
    Block1&       out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<3> rtl_in = block_layout<3>(in); 
    Rt_layout<3> rtl_out = block_layout<3>(out); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);
    // Check whether the input buffer will be destroyed.
    sync_action_type in_sync = backend->requires_copy(rtl_in)
      ? SYNC_IN_NOPRESERVE
      : SYNC_IN; 
    {
      // Create a 'direct data accessor', adjusting the block layout if necessary.
      Rt_ext_data<Block0> in_ext(in, rtl_in, in_sync,
				 input_buffer_.get());
      Rt_ext_data<Block1> out_ext(out, rtl_out, SYNC_OUT,
				  output_buffer_.get());
    
      // Call the backend.
      if (rtl_out.complex == cmplx_inter_fmt) 
	backend->out_of_place(in_ext.data().as_real(),
			      in_ext.stride(0),
			      in_ext.stride(1),
			      in_ext.stride(2),
			      out_ext.data().as_inter(),
			      out_ext.stride(0),
			      out_ext.stride(1),
			      out_ext.stride(2),
			      in_ext.size(0),
			      in_ext.size(1),
			      in_ext.size(2));
      else
	backend->out_of_place(in_ext.data().as_real(),
			      in_ext.stride(0),
			      in_ext.stride(1),
			      in_ext.stride(2),
			      out_ext.data().as_split(),
			      out_ext.stride(0),
			      out_ext.stride(1),
			      out_ext.stride(2),
			      in_ext.size(0),
			      in_ext.size(1),
			      in_ext.size(2));
    }
    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      scale_block(scale_, out);
  }

  template <typename BE, typename Block0, typename Block1>
  void out_of_place(BE *backend,
		    const_Tensor<T, Block0> in,
		    Tensor<std::complex<T>, Block1> out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<3> rtl_in = block_layout<3>(in.block()); 
    Rt_layout<3> rtl_out = block_layout<3>(out.block()); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);
    // Check whether the input buffer will be destroyed.
    sync_action_type in_sync = backend->requires_copy(rtl_in)
      ? SYNC_IN_NOPRESERVE
      : SYNC_IN; 
    {
      // Create a 'direct data accessor', adjusting the block layout if necessary.
      Rt_ext_data<Block0> in_ext(in.block(), rtl_in, in_sync,
				 input_buffer_.get());
      Rt_ext_data<Block1> out_ext(out.block(), rtl_out, SYNC_OUT,
				  output_buffer_.get());
    
      // Call the backend.
      if (rtl_out.complex == cmplx_inter_fmt) 
	backend->out_of_place(in_ext.data().as_real(),
			      in_ext.stride(0),
			      in_ext.stride(1),
			      in_ext.stride(2),
			      out_ext.data().as_inter(),
			      out_ext.stride(0),
			      out_ext.stride(1),
			      out_ext.stride(2),
			      in_ext.size(0),
			      in_ext.size(1),
			      in_ext.size(2));
      else
	backend->out_of_place(in_ext.data().as_real(),
			      in_ext.stride(0),
			      in_ext.stride(1),
			      in_ext.stride(2),
			      out_ext.data().as_split(),
			      out_ext.stride(0),
			      out_ext.stride(1),
			      out_ext.stride(2),
			      in_ext.size(0),
			      in_ext.size(1),
			      in_ext.size(2));
    }
    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      out *= scale_;
  }

  T scale() const { return scale_; }

private:
  T scale_;
  aligned_array<T> input_buffer_;
  aligned_array<std::complex<T> > output_buffer_;
};

template <typename T>
class workspace<3, std::complex<T>, T >
{
public:
  template <typename BE>
  workspace(BE*, Domain<3> const &in, Domain<3> const &out, T scale)
    : scale_(scale),
      input_buffer_(in.size()),
      output_buffer_(out.size())
  {}
  
  template <typename BE, typename Block0, typename Block1>
  void out_of_place_blk(
    BE*           backend,
    Block0 const& in,
    Block1&       out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<3> rtl_in = block_layout<3>(in); 
    Rt_layout<3> rtl_out = block_layout<3>(out); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);
    // Check whether the input buffer will be destroyed.
    sync_action_type in_sync = backend->requires_copy(rtl_in)
      ? SYNC_IN_NOPRESERVE
      : SYNC_IN;
    { 
      // Create a 'direct data accessor', adjusting the block layout if necessary.
      Rt_ext_data<Block0> in_ext(in, rtl_in, in_sync,
				 input_buffer_.get());
      Rt_ext_data<Block1> out_ext(out, rtl_out, SYNC_OUT,
				  output_buffer_.get());
    
      // Call the backend.
      if (rtl_in.complex == cmplx_inter_fmt) 
	backend->out_of_place(in_ext.data().as_inter(),
			      in_ext.stride(0),
			      in_ext.stride(1),
			      in_ext.stride(2),
			      out_ext.data().as_real(),
			      out_ext.stride(0),
			      out_ext.stride(1),
			      out_ext.stride(2),
			      out_ext.size(0),
			      out_ext.size(1),
			      out_ext.size(2));
      else
	backend->out_of_place(in_ext.data().as_split(),
			      in_ext.stride(0),
			      in_ext.stride(1),
			      in_ext.stride(2),
			      out_ext.data().as_real(),
			      out_ext.stride(0),
			      out_ext.stride(1),
			      out_ext.stride(2),
			      out_ext.size(0),
			      out_ext.size(1),
			      out_ext.size(2));
    }
    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      scale_block(scale_, out);
  }

  template <typename BE, typename Block0, typename Block1>
  void out_of_place(BE *backend,
		    const_Tensor<std::complex<T>, Block0> in,
		    Tensor<T, Block1> out)
    const
  {
    // Find out about the blocks's actual layout.
    Rt_layout<3> rtl_in = block_layout<3>(in.block()); 
    Rt_layout<3> rtl_out = block_layout<3>(out.block()); 
    
    // Find out about what layout is acceptable for this backend.
    backend->query_layout(rtl_in, rtl_out);
    // Check whether the input buffer will be destroyed.
    sync_action_type in_sync = backend->requires_copy(rtl_in)
      ? SYNC_IN_NOPRESERVE
      : SYNC_IN;
    { 
      // Create a 'direct data accessor', adjusting the block layout if necessary.
      Rt_ext_data<Block0> in_ext(in.block(), rtl_in, in_sync,
				 input_buffer_.get());
      Rt_ext_data<Block1> out_ext(out.block(), rtl_out, SYNC_OUT,
				  output_buffer_.get());
    
      // Call the backend.
      if (rtl_in.complex == cmplx_inter_fmt) 
	backend->out_of_place(in_ext.data().as_inter(),
			      in_ext.stride(0),
			      in_ext.stride(1),
			      in_ext.stride(2),
			      out_ext.data().as_real(),
			      out_ext.stride(0),
			      out_ext.stride(1),
			      out_ext.stride(2),
			      out_ext.size(0),
			      out_ext.size(1),
			      out_ext.size(2));
      else
	backend->out_of_place(in_ext.data().as_split(),
			      in_ext.stride(0),
			      in_ext.stride(1),
			      in_ext.stride(2),
			      out_ext.data().as_real(),
			      out_ext.stride(0),
			      out_ext.stride(1),
			      out_ext.stride(2),
			      out_ext.size(0),
			      out_ext.size(1),
			      out_ext.size(2));
    }
    // Scale the data if not already done by the backend.
    if (!backend->supports_scale() && scale_ != T(1.))
      out *= scale_;
  }

  T scale() const { return scale_; }

private:
  T scale_;
  aligned_array<std::complex<T> > input_buffer_;
  aligned_array<T> output_buffer_;
};

} // namespace vsip::impl::fft
} // namespace vsip::impl
} // namespace vsip

#endif
