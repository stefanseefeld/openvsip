/* Copyright (c) 2007, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/ppu/fft.cpp
    @author  Stefan Seefeld
    @date    2007-01-31
    @brief   VSIPL++ Library: FFT wrappers and traits to bridge with 
             the CBE SDK.
*/

#include <algorithm>
#include <vsip/support.hpp>
#include <vsip/core/allocation.hpp>
#include <vsip/core/fft/backend.hpp>
#include <vsip/core/fft/util.hpp>
#include <vsip/opt/cbe/overlay_params.h>
#include <vsip/opt/cbe/ppu/fft.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>

namespace vsip
{
namespace impl
{
namespace cbe
{

template <typename T>
class Fft_base
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;

public:
  Fft_base(length_type) 
  {}

  virtual ~Fft_base() 
  {}

  // Interleaved-complex FFT
  void 
  fft(std::complex<T> const* in, std::complex<T>* out, 
    length_type length, T scale, int exponent)
  {
    assert(is_dma_addr_ok(in));
    assert(is_dma_addr_ok(out));

    static char* code_ea = 0;
    static int   code_size;
    Fft_params  params;
    Fft_params* fftp = &params;
    length_type num_buffers = (length <= 4096) ? 2 : 1;

    if (code_ea == 0) lwp::load_plugin(code_ea, code_size, "plugins/chalfast_f.plg");

    Task_manager *mgr = Task_manager::instance();
    std::auto_ptr<lwp::Task> task =
      mgr->reserve_lwp_task(sizeof(complex<T>)*length, num_buffers,
			    (uintptr_t)code_ea, code_size, overlay_zfft_f);
    
    fftp->cmd            = overlay_cfft_f;
    fftp->direction      = (exponent == -1 ? fwd_fft : inv_fft);
    fftp->size           = length;
    fftp->scale          = scale;
    fftp->ea_input       = ea_from_ptr(in);
    fftp->ea_output      = ea_from_ptr(out);
    fftp->in_blk_stride  = 0;  // not applicable in the single FFT case
    fftp->out_blk_stride = 0;
    fftp->chunks_per_wb  = 1;
    fftp->chunks_per_spe = 1;

    lwp::Workblock block = task.get()->create_workblock(1);
    block.set_parameters(params);
    block.enqueue();

    task.get()->sync();
  }

  // Split-complex FFT
  void 
  fft(T const* in_re, T const* in_im, T* out_re, T* out_im, 
      length_type length, T scale, int exponent)
  {
    assert(is_dma_addr_ok(in_re));
    assert(is_dma_addr_ok(in_im));
    assert(is_dma_addr_ok(out_re));
    assert(is_dma_addr_ok(out_im));

    static char* code_ea = 0;
    static int   code_size;
    Fft_split_params params;
    Fft_split_params* fftp = &params;
    length_type num_buffers = (length <= 4096) ? 2 : 1;

    Task_manager *mgr = Task_manager::instance();

    if (code_ea == 0) lwp::load_plugin(code_ea, code_size, "plugins/zhalfast_f.plg");

    std::auto_ptr<lwp::Task> task =
      mgr->reserve_lwp_task(sizeof(complex<T>)*length, num_buffers,
			    (uintptr_t)code_ea, code_size,
			    overlay_zfft_f);

    fftp->cmd            = overlay_zfft_f;
    fftp->direction      = (exponent == -1 ? fwd_fft : inv_fft);
    fftp->size           = length;
    fftp->scale          = scale;
    fftp->ea_input_re    = ea_from_ptr(in_re);
    fftp->ea_input_im    = ea_from_ptr(in_im);
    fftp->ea_output_re   = ea_from_ptr(out_re);
    fftp->ea_output_im   = ea_from_ptr(out_im);
    fftp->in_blk_stride  = 0;  // not applicable in the single FFT case
    fftp->out_blk_stride = 0;
    fftp->chunks_per_wb  = 1;
    fftp->chunks_per_spe = 1;

    lwp::Workblock block = task.get()->create_workblock(1);
    block.set_parameters(params);
    block.enqueue();

    task.get()->sync();
  }

  // Interleaved-complex FFTM
  void 
  fftm(std::complex<T> const* in, std::complex<T>* out, 
       stride_type in_r_stride, stride_type in_c_stride,
       stride_type out_r_stride, stride_type out_c_stride,
       length_type rows, length_type cols, 
       T scale, int exponent, int axis)
  {
    assert(is_dma_addr_ok(in));
    assert(is_dma_addr_ok(out));
    assert(is_dma_addr_ok(in   + (axis == 1 ? in_r_stride  : in_c_stride)));
    assert(is_dma_addr_ok(out  + (axis == 1 ? out_r_stride : out_c_stride)));

    static char* code_ea = 0;
    static int   code_size;
    Fft_params params;
    Fft_params* fftp = &params;

    if (code_ea == 0) lwp::load_plugin(code_ea, code_size, "plugins/chalfast_f.plg");

    Task_manager *mgr = Task_manager::instance();


    length_type num_ffts;
    length_type in_stride;
    length_type out_stride;

    if (axis == 1)
    {
      num_ffts = rows;
      in_stride = in_r_stride;
      out_stride = out_r_stride;
      fftp->size = cols;
    }
    else
    {
      num_ffts = cols;
      in_stride = in_c_stride;
      out_stride = out_c_stride;
      fftp->size = rows;
    }

    length_type num_buffers = (fftp->size <= 4096) ? 2 : 1;

    fftp->cmd            = overlay_cfft_f;
    fftp->direction      = (exponent == -1 ? fwd_fft : inv_fft);
    fftp->scale          = scale;
    fftp->ea_input       = ea_from_ptr(in);
    fftp->ea_output      = ea_from_ptr(out);
    fftp->in_blk_stride  = in_stride;
    fftp->out_blk_stride = out_stride;

    length_type spes    = mgr->num_spes();
    length_type chunks_per_wb;

    // A chunk is the amount of data to perform 1 FFT.
    //
    // If the chunk size is less than 16 KB, send multiple chunks per
    // workblock to amortize transfer costs.

    if (fftp->size * sizeof(float) < 16384)
    {
      // Ideal case: send enough chunks to reack 16KB DMA xfer size.
      chunks_per_wb = 16384 / (fftp->size * sizeof(float));

      // Limit number of chunks to fit into available DMA list size.
      if (in_stride != fftp->size || out_stride != fftp->size)
      {
	chunks_per_wb = std::min<length_type>(chunks_per_wb,
					      VSIP_IMPL_OVERLAY_DTL_SIZE / 4);
	assert(2*chunks_per_wb <= VSIP_IMPL_OVERLAY_DTL_SIZE);
      }
    }
    else 
      chunks_per_wb = 1;

    std::auto_ptr<lwp::Task> task =
      mgr->reserve_lwp_task(chunks_per_wb*sizeof(complex<T>)*fftp->size,
			    num_buffers,
			    (uintptr_t)code_ea, code_size, overlay_cfft_f);

    length_type num_wb     = num_ffts / chunks_per_wb;
    length_type wb_per_spe = num_wb / spes;
    length_type extra_wb   = (num_ffts % chunks_per_wb) ? 1 : 0;


    for (length_type i = 0; i < spes && i < num_wb + extra_wb; ++i)
    {
      // If wbs don't divide evenly, give the first SPEs one extra.
      length_type spe_wb = (i < num_wb % spes) ? wb_per_spe + 1
                                               : wb_per_spe;
      length_type spe_ffts = spe_wb * chunks_per_wb;

      if (extra_wb && (i == spes-1 || i >= num_wb))
      {
	spe_wb   += 1;
	spe_ffts += num_ffts % chunks_per_wb;
      }

      fftp->chunks_per_wb  = chunks_per_wb;
      fftp->chunks_per_spe = spe_ffts;
      lwp::Workblock block = task.get()->create_workblock(spe_wb);
      block.set_parameters(params);
      block.enqueue();

      fftp->ea_input  += sizeof(ctype) * spe_ffts * in_stride;
      fftp->ea_output += sizeof(ctype) * spe_ffts * out_stride;
    }
    task.get()->sync();
  }

  // Split-complex FFTM
  void 
  fftm(T const* in_re, T const* in_im,
       T* out_re, T* out_im,
       stride_type in_r_stride, stride_type in_c_stride,
       stride_type out_r_stride, stride_type out_c_stride,
       length_type rows, length_type cols, 
       T scale, int exponent, int axis)
  {
    assert(is_dma_addr_ok(in_re));
    assert(is_dma_addr_ok(in_im));
    assert(is_dma_addr_ok(out_re));
    assert(is_dma_addr_ok(out_im));
    assert(is_dma_addr_ok(in_re  + (axis == 1 ? in_r_stride  : in_c_stride)));
    assert(is_dma_addr_ok(out_re + (axis == 1 ? out_r_stride : out_c_stride)));
    assert(is_dma_addr_ok(in_im  + (axis == 1 ? in_r_stride  : in_c_stride)));
    assert(is_dma_addr_ok(out_im + (axis == 1 ? out_r_stride : out_c_stride)));

    static char* code_ea = 0;
    static int   code_size;
    Fft_split_params params;
    Fft_split_params* fftp = &params;

    if (code_ea == 0) lwp::load_plugin(code_ea, code_size, "plugins/zhalfast_f.plg");

    Task_manager *mgr = Task_manager::instance();

    length_type num_ffts;
    length_type in_stride;
    length_type out_stride;

    if (axis == 1)
    {
      num_ffts = rows;
      in_stride = in_r_stride;
      out_stride = out_r_stride;
      fftp->size = cols;
    }
    else
    {
      num_ffts = cols;
      in_stride = in_c_stride;
      out_stride = out_c_stride;
      fftp->size = rows;
    }

    length_type num_buffers = (fftp->size <= 4096) ? 2 : 1;

    fftp->cmd            = overlay_zfft_f;
    fftp->direction      = (exponent == -1 ? fwd_fft : inv_fft);
    fftp->scale          = scale;
    fftp->ea_input_re    = ea_from_ptr(in_re);
    fftp->ea_input_im    = ea_from_ptr(in_im);
    fftp->ea_output_re   = ea_from_ptr(out_re);
    fftp->ea_output_im   = ea_from_ptr(out_im);
    fftp->in_blk_stride  = in_stride;
    fftp->out_blk_stride = out_stride;

    length_type spes          = mgr->num_spes();
    length_type chunks_per_wb;

    // A chunk is the amount of data to perform 1 FFT.
    //
    // If the chunk size is less than 16 KB, send multiple chunks per
    // workblock to amortize transfer costs.

    if (fftp->size * sizeof(float) < 16384)
    {
      // Ideal case: send enough chunks to reack 16KB DMA xfer size.
      chunks_per_wb = 16384 / (fftp->size * sizeof(float));

      // Limit number of chunks to fit into available DMA list size.
      if (in_stride != fftp->size || out_stride != fftp->size)
      {
	chunks_per_wb = std::min<length_type>(chunks_per_wb,
					      VSIP_IMPL_OVERLAY_DTL_SIZE / 4);
	assert(2*chunks_per_wb <= VSIP_IMPL_OVERLAY_DTL_SIZE);
      }
    }
    else 
      chunks_per_wb = 1;

    std::auto_ptr<lwp::Task> task =
      mgr->reserve_lwp_task(chunks_per_wb*sizeof(complex<T>)*fftp->size,
			    num_buffers,
			    (uintptr_t)code_ea, code_size, overlay_zfft_f);

    length_type num_wb     = num_ffts / chunks_per_wb;
    length_type wb_per_spe = num_wb / spes;
    length_type extra_wb   = (num_ffts % chunks_per_wb) ? 1 : 0;

    for (length_type i = 0; i < spes && i < num_wb + extra_wb; ++i)
    {
      // If wbs don't divide evenly, give the first SPEs one extra.
      length_type spe_wb = (i < num_wb % spes) ? wb_per_spe + 1
                                               : wb_per_spe;
      length_type spe_ffts = spe_wb * chunks_per_wb;

      if (extra_wb && (i == spes-1 || i >= num_wb))
      {
	spe_wb   += 1;
	spe_ffts += num_ffts % chunks_per_wb;
      }

      fftp->chunks_per_wb  = chunks_per_wb;
      fftp->chunks_per_spe = spe_ffts;
      lwp::Workblock block = task.get()->create_workblock(spe_wb);
      block.set_parameters(params);
      block.enqueue();

      fftp->ea_input_re  += sizeof(T) * spe_ffts * in_stride;
      fftp->ea_input_im  += sizeof(T) * spe_ffts * in_stride;
      fftp->ea_output_re += sizeof(T) * spe_ffts * out_stride;
      fftp->ea_output_im += sizeof(T) * spe_ffts * out_stride;
    }
    task.get()->sync();
  }
};


template <dimension_type D, //< Dimension
          typename I,       //< Input type
	  typename O,       //< Output type
	  int S>            //< Special Direction
class Fft_impl;

// 1D complex -> complex FFT

template <typename T, int S>
class Fft_impl<1, std::complex<T>, std::complex<T>, S>
    : public fft::Fft_backend<1, std::complex<T>, std::complex<T>, S>,
      private Fft_base<T>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const exponent = S == fft_fwd ? -1 : 1;

public:
  Fft_impl(Domain<1> const &dom, rtype scale)
    : Fft_base<T>(dom.size()),
      scale_(scale)
  {
  }
  virtual ~Fft_impl()
  {}

  virtual char const* name() { return "fft-cbe-1D-complex"; }
  virtual bool supports_scale() { return true;}
  virtual void query_layout(Rt_layout<1> &rtl_inout)
  {
    rtl_inout.packing = aligned;
    rtl_inout.order = tuple<0, 1, 2>();
    // Both split and interleaved supported.
    rtl_inout.alignment = 16;
  }
  virtual void query_layout(Rt_layout<1> &rtl_in, Rt_layout<1> &rtl_out)
  {
    rtl_in.packing = rtl_out.packing = aligned;
    rtl_in.order = rtl_out.order = tuple<0, 1, 2>();
    // Both split and interleaved supported, however in and out must match.
    rtl_in.storage_format = rtl_out.storage_format;
    rtl_in.alignment = rtl_out.alignment = 16;
  }
  virtual void in_place(ctype *inout, stride_type stride, length_type length)
  {
    assert(stride == 1);
    this->fft(inout, inout, length, this->scale_, exponent);
  }
  virtual void in_place(ztype inout, stride_type stride, length_type length)
  {
    assert(stride == 1);
    this->fft(inout.first, inout.second, inout.first, inout.second,
	      length, this->scale_, exponent);
  }
  virtual void out_of_place(ctype *in, stride_type in_stride,
			    ctype *out, stride_type out_stride,
			    length_type length)
  {
    assert(in_stride == 1);
    assert(out_stride == 1);
    this->fft(in, out, length, this->scale_, exponent);
  }
  virtual void out_of_place(ztype in, stride_type in_stride,
			    ztype out, stride_type out_stride,
			    length_type length)
  {
    assert(in_stride == 1);
    assert(out_stride == 1);
    this->fft(in.first, in.second, out.first, out.second,
	      length, this->scale_, exponent);
  }

private:
  rtype scale_;
};




template <typename I, //< Input type
	  typename O, //< Output type
	  int A,      //< Axis
	  int D>      //< Direction
class Fftm_impl;

// complex -> complex FFTM
template <typename T, int A, int D>
class Fftm_impl<std::complex<T>, std::complex<T>, A, D>
    : public fft::Fftm_backend<std::complex<T>, std::complex<T>, A, D>,
      private Fft_base<T>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const axis = A == vsip::col ? 0 : 1;
  static int const exponent = D == fft_fwd ? -1 : 1;

public:
  Fftm_impl(Domain<2> const &dom, rtype scale)
    : Fft_base<T>(dom[axis].size()),
      scale_(scale)    
  {
    this->size_[0] = dom[0].size();
    this->size_[1] = dom[1].size();
  }
  virtual ~Fftm_impl()
  {}

  virtual char const* name() { return "fftm-cbe-1D-complex"; }
  virtual bool supports_scale() { return true;}

  virtual void query_layout(Rt_layout<2> &rtl_inout)
  {
    // must have unit stride, but does not have to be dense
    if (axis == 1)
      rtl_inout.order = tuple<0, 1, 2>();
    else
      rtl_inout.order = tuple<1, 0, 2>();
    rtl_inout.packing = aligned;
    // Both split and interleaved supported.
    rtl_inout.alignment = 16;
  }
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    // must have unit stride, but does not have to be dense
    if (axis == 1)
      rtl_in.order = rtl_out.order = tuple<0, 1, 2>();
    else
      rtl_in.order = rtl_out.order = tuple<1, 0, 2>();
    rtl_in.packing = rtl_out.packing = aligned;
    // Both split and interleaved supported, however in and out must match.
    rtl_in.storage_format = rtl_out.storage_format;
    rtl_in.alignment = rtl_out.alignment = 16;
  }

  virtual void in_place(ctype *inout,
			stride_type r_stride, stride_type c_stride,
			length_type rows, length_type cols)
  {
    if (axis == 1)
    {
      if (rows == 0) return; // Handle empty local subblock.
      assert(rows <= this->size_[0]); // OK if rows are distributed.
      assert(cols == this->size_[1]); // Columns must be whole.
      assert(c_stride == 1);
    }
    else
    {
      if (cols == 0) return; // Handle empty local subblock.
      assert(rows == this->size_[0]); // Rows must be whole.
      assert(cols <= this->size_[1]); // OK if columns are distributed.
      assert(r_stride == 1);
    }
    this->fftm(inout, inout,
	       r_stride, c_stride,
	       r_stride, c_stride,
	       rows, cols,
	       this->scale_, exponent, axis);
  }

  virtual void in_place(ztype inout,
			stride_type r_stride, stride_type c_stride,
			length_type rows, length_type cols)
  {
    if (axis == 1)
    {
      if (rows == 0) return; // Handle empty local subblock.
      assert(rows <= this->size_[0]); // OK if rows are distributed.
      assert(cols == this->size_[1]); // Columns must be whole.
      assert(c_stride == 1);
    }
    else
    {
      if (cols == 0) return; // Handle empty local subblock.
      assert(rows == this->size_[0]); // Rows must be whole.
      assert(cols <= this->size_[1]); // OK if columns are distributed.
      assert(r_stride == 1);
    }
    this->fftm(inout.first, inout.second,
	       inout.first, inout.second,
	       r_stride, c_stride,
	       r_stride, c_stride,
	       rows, cols,
	       this->scale_, exponent, axis);
  }

  virtual void out_of_place(ctype *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    ctype *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    if (axis == 1)
    {
      if (rows == 0) return; // Handle empty local subblock.
      assert(rows <= this->size_[0]); // OK if rows are distributed.
      assert(cols == this->size_[1]); // Columns must be whole.
      assert(in_c_stride == 1 && out_c_stride == 1);
    }
    else
    {
      if (cols == 0) return; // Handle empty local subblock.
      assert(rows == this->size_[0]); // Rows must be whole.
      assert(cols <= this->size_[1]); // OK if columns are distributed.
      assert(in_r_stride == 1 && out_r_stride == 1);
    }
    this->fftm(in, out, 
	       in_r_stride, in_c_stride,
	       out_r_stride, out_c_stride,
	       rows, cols,
	       this->scale_, exponent, axis);
  }
  virtual void out_of_place(ztype in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    ztype out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    if (axis == 1)
    {
      if (rows == 0) return; // Handle empty local subblock.
      assert(rows <= this->size_[0]); // OK if rows are distributed.
      assert(cols == this->size_[1]); // Columns must be whole.
      assert(in_c_stride == 1 && out_c_stride == 1);
    }
    else
    {
      if (cols == 0) return; // Handle empty local subblock.
      assert(rows == this->size_[0]); // Rows must be whole.
      assert(cols <= this->size_[1]); // OK if columns are distributed.
      assert(in_r_stride == 1 && out_r_stride == 1);
    }
    this->fftm(in.first, in.second, out.first, out.second, 
	       in_r_stride, in_c_stride,
	       out_r_stride, out_c_stride,
	       rows, cols,
	       this->scale_, exponent, axis);
  }

private:
  rtype scale_;
  length_type fft_length_;

  length_type size_[2];
};

#define FFT_DEF(D, I, O, S)	                       \
template <>                                            \
std::auto_ptr<fft::Fft_backend<D, I, O, S> >	       \
create(Domain<D> const &dom, float scale)              \
{                                                      \
  return std::auto_ptr<fft::Fft_backend<D, I, O, S> >  \
    (new Fft_impl<D, I, O, S>(dom, scale));            \
}

FFT_DEF(1, std::complex<float>, std::complex<float>, fft_fwd)
FFT_DEF(1, std::complex<float>, std::complex<float>, fft_inv)

#undef FFT_DEF

#define FFTM_DEF(I, O, A, D)   		               \
template <>                                            \
std::auto_ptr<fft::Fftm_backend<I, O, A, D> >          \
create(Domain<2> const &dom, float scale)	       \
{                                                      \
  return std::auto_ptr<fft::Fftm_backend<I, O, A, D> > \
    (new Fftm_impl<I, O, A, D>(dom, scale));           \
}

//FFTM_DEF(float, std::complex<float>, 0, -1)
//FFTM_DEF(float, std::complex<float>, 1, -1)
//FFTM_DEF(std::complex<float>, float, 0, 1)
//FFTM_DEF(std::complex<float>, float, 1, 1)
FFTM_DEF(std::complex<float>, std::complex<float>, 0, fft_fwd)
FFTM_DEF(std::complex<float>, std::complex<float>, 1, fft_fwd)
FFTM_DEF(std::complex<float>, std::complex<float>, 0, fft_inv)
FFTM_DEF(std::complex<float>, std::complex<float>, 1, fft_inv)

#undef FFTM_DEF

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip

