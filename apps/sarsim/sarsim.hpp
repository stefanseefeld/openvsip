/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    sarsim.cpp
    @author  Jules Bergmann
    @date    03/02/2005
    @brief   VSIPL++ implementation of RASSP benchmark 0.
*/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <vsip/complex.hpp>
#include <vsip/signal.hpp>
#include <vsip/math.hpp>
#include <vsip/map.hpp>
#include <vsip/parallel.hpp>

#include "cast-block.hpp"

#define USE_SA PARALLEL
#define VERBOSE 0

/* Perform SAR processing.

   All input and output data is represented using single-precision
   (float) values.  Therefore, "float" is called the "I/O type".
   However, the precision used to perform computations is the template
   parameter T (the "value type").  Because the value type may be
   double-precision, the computation may be either uniform-precision
   or mixed-precision.  */

template <typename T>
class SarSim {
public:

  typedef vsip::index_type index_type;

  typedef float io_type;
  typedef vsip::Vector<float> io_vector_type;
  typedef vsip::complex<io_type> cio_type;
  typedef vsip::Matrix<cio_type> cio_matrix_type;

  typedef T value_type;
  typedef vsip::complex<value_type> cval_type;
  typedef vsip::Vector<cval_type> val_vector_type;
  typedef vsip::impl::Strided<1, cval_type,
				 vsip::Layout<1, vsip::row1_type,
					vsip::impl::packing::dense,
					vsip::impl::split_complex> >
					cval_split_block_type;
  typedef vsip::Vector<cval_type, cval_split_block_type>
		 cval_split_vector_type;

  enum polarity_type { 
    pt_first,
    pt_hh = pt_first, 
    pt_hv, 
    pt_vh, 
    pt_vv, 
    pt_npols
  };

  SarSim(index_type nrange,
	 index_type npulse,
	 index_type ncsamples,
	 index_type niq,
	 io_type swath,
	 cio_matrix_type w_eq,
	 io_vector_type rcs,
	 io_vector_type i_coef,
	 io_vector_type q_coef,
	 cio_matrix_type cphase,
#if PARALLEL
	 vsip::Map<> map_in,
	 vsip::Map<> map_rg,
	 vsip::Map<> map_az,
	 vsip::Map<> map_out,
#endif
         int /*pmode*/
         );
  virtual ~SarSim() {}

  void process(index_type nframe, bool pol_on[pt_npols]);
  void init_io(index_type nframe);
  void fini_io(index_type nframe);

  void report_performance() const;

protected:
  
  // Read the data corresponding to the Pth pulse.  Put the data read
  // in vec_iq_.
  virtual io_type read_pulse(int pol, index_type p) = 0;

  // Write a azimuth frame header to the output.
  virtual void write_output_header(int pol) = 0;

  // Write a single azimuth line to the output. The actual data to be
  // written is in the left half of azbuf_.
  virtual void write_output(int pol) = 0;

  index_type nrange_;
  index_type npulse_;
  index_type ncsamples_;
  index_type niq_;
  io_type swath_;
  
  cio_matrix_type w_eq_;
  io_vector_type rcs_;
  io_vector_type i_coef_;
  io_vector_type q_coef_;
  cio_matrix_type cphase_;

  cval_split_vector_type vec_iq_;
  val_vector_type azbuf_;
  val_vector_type azbuf2_;

private:

  typedef vsip::Convolution<
    vsip::const_Vector, 
    vsip::nonsym, 
    vsip::support_min, 
    T, 0, 
    vsip::alg_space> convolution_type;

  typedef vsip::Fft<
    vsip::const_Vector, 
    cval_type, 
    cval_type, 
    vsip::fft_fwd, 
    vsip::by_reference, 
    0, 
    vsip::alg_space> for_fft_type;

  typedef vsip::Fft<
    vsip::const_Vector, 
    cval_type, 
    cval_type, 
    vsip::fft_inv, 
    vsip::by_reference, 
    0, 
    vsip::alg_space> inv_fft_type;

  typedef vsip::whole_domain_type whole_domain_type;

#if PARALLEL
  typedef vsip::Replicated_map<1> map_global_type;
  typedef vsip::Map<> map_in_type;
  typedef vsip::Map<> map_rg_type;
  typedef vsip::Map<> map_az_type;
  typedef vsip::Map<> map_out_type;
#else
  typedef vsip::Local_map map_global_type;
  typedef vsip::Local_map map_in_type;
  typedef vsip::Local_map map_rg_type;
  typedef vsip::Local_map map_az_type;
  typedef vsip::Local_map map_out_type;
#endif

  typedef vsip::Dense<1, io_type, vsip::row1_type, map_in_type>
		 range_in_block_type;
  typedef vsip::Vector<io_type, range_in_block_type> range_in_type;

  typedef vsip::Dense<1, io_type, vsip::row1_type, map_global_type>
		 range_global_block_type;
  typedef vsip::Vector<io_type, range_global_block_type> range_global_type;


  typedef vsip::Dense<3, cval_type, vsip::tuple<0, 1, 2>, map_in_type>
		 cube_in_block_type;
  typedef vsip::Tensor<cval_type, cube_in_block_type> cube_in_type;

  typedef vsip::Dense<3, cval_type, vsip::tuple<0, 1, 2>, map_rg_type>
		 cube_rg_block_type;
  typedef vsip::Tensor<cval_type, cube_rg_block_type> cube_rg_type;

  typedef vsip::Dense<3, cval_type, vsip::tuple<0, 2, 1>, map_az_type>
		 cube_az_block_type;
  typedef vsip::Tensor<cval_type, cube_az_block_type> cube_az_type;

  typedef vsip::Dense<3, cval_type, vsip::tuple<0, 2, 1>, map_out_type>
		 cube_img_block_type;
  typedef vsip::Tensor<cval_type, cube_img_block_type> cube_img_type;

  typedef vsip::Dense<3, cval_type, vsip::tuple<0, 2, 1>, map_out_type>
		 cube_out_block_type;
  typedef vsip::Tensor<cval_type, cube_out_block_type> cube_out_type;

  typedef vsip::impl::profile::Acc_timer timer_type;

  template <typename Block1, typename Block2>
  void read_frame(vsip::Tensor<cval_type, Block1> cube_in,
		  vsip::Vector<io_type, Block2>  current_range,
		  index_type frame);

  template <typename Block>
  void write_frame(vsip::Tensor<cval_type, Block> cube_out);

  void io_process(index_type frame);
  void range_process();
  void azimuth_process(index_type frame, bool is_last);
 
  // The forward FFT applied during range processing.
  for_fft_type range_fft_;
  // The forward FFT applied during azimuth processing.
  for_fft_type az_for_fft_;
  // The inverse FFT applied during azimuth processing.
  inv_fft_type az_inv_fft_;
  // The in-phase convolution.
  convolution_type iconv_;
  // The quadrature convolution.
  convolution_type qconv_;

  // These variables are re-used during each all to process().
  io_type initial_range_;
  range_global_type current_range_;
  bool pol_on_[pt_npols];

  // Range processing
  val_vector_type line1_;
  val_vector_type line2_;

  map_in_type map_in_;
  map_rg_type map_rg_;
  map_az_type map_az_;
  map_out_type map_out_;

  // The input cube is a NPOLARITY x NPULSE x NRANGE tensor.
  //   The raw cube from disk is NPOLARIRY x NPULSE x NCSAMPLES,
  //   Range processing produces NPOLARIRY x NPULSE x NRANGE,
  // (Note: NCSAMPLES < NRANGE)
//  cube_in_block_type block_in_;
//  cube_in_type cube_in_;

  // The range cube is a NPOLARITY x NPULSE x NRANGE tensor.
  cube_rg_block_type block_rg_;
  cube_rg_type cube_rg_;

  // The az data cube is a NPOLARITY x 2 * NPULSE x NRANGE tensor.
  cube_az_type cube_az_;

  // The output cube is a NPOLARITY x NPULSE x NRANGE tensor.
  cube_img_block_type block_img_;
  cube_img_type cube_img_;

  cval_type** input_frame_buffer_;
  cval_type** output_frame_buffer_;

  timer_type rp_time_;	// range processing time
  timer_type ap_time_;	// azimuth processing time
  timer_type proc_time_;
  timer_type ct_time_;	// corner-turn time
  timer_type rvm1_time_;
  timer_type rvm2_time_;
  timer_type avm_time_;
  int        rg_line_cnt_;
  int        az_line_cnt_;
};

template <typename T>
SarSim<T>::SarSim(index_type nrange,
		  index_type npulse,
		  index_type ncsamples,
		  index_type niq,
		  io_type swath,
		  cio_matrix_type w_eq,
		  io_vector_type rcs,
		  io_vector_type i_coef,
		  io_vector_type q_coef,
		  cio_matrix_type cphase,
#if PARALLEL
		  vsip::Map<> map_in,
		  vsip::Map<> map_rg,
		  vsip::Map<> map_az,
		  vsip::Map<> map_out,
#endif
                  int /*pmode*/)
  : nrange_(nrange),
    npulse_(npulse),
    ncsamples_(ncsamples),
    niq_(niq),
    swath_(swath),
    w_eq_(w_eq),
    rcs_(rcs),
    i_coef_(i_coef),
    q_coef_(q_coef),
    cphase_(cphase),
    vec_iq_(ncsamples_),
    azbuf_(2 * npulse),
    azbuf2_(2 * npulse),
    // Because creating an FFT may require significant computation
    // (planning), we create the FFTs before beginning the main loop.
    range_fft_(vsip::Domain<1>(nrange_), 1.f),
    az_for_fft_(vsip::Domain<1>(2 * npulse_), 1.f),
    az_inv_fft_(vsip::Domain<1>(2 * npulse_), 1.f / (2 * npulse_)),
    iconv_(vsip::impl::cast_view<T>(i_coef_), 
	   vsip::Domain<1>(ncsamples_), 1),
    qconv_(vsip::impl::cast_view<T>(q_coef_), 
	   vsip::Domain<1>(ncsamples_), 1),
    current_range_(4),
    line1_(nrange_),
    line2_(nrange_),
#if PARALLEL
    map_in_(map_in),
    map_rg_(map_rg),
    map_az_(map_az),
    map_out_(map_out),
#endif
    block_rg_(vsip::Domain<3>(pt_npols, npulse_, nrange_), 
	      static_cast<cval_type*>(0), map_rg_),
    cube_rg_(block_rg_),
    cube_az_(pt_npols, 2 * npulse_, nrange_, 0.f, map_az_),
    block_img_(vsip::Domain<3>(pt_npols, npulse_, nrange_), 
	       static_cast<cval_type*>(0), map_az_),
    cube_img_(block_img_),
    rg_line_cnt_(0),
    az_line_cnt_(0)
{
}

template <typename T>
void
SarSim<T>::init_io(index_type nframe) {
  using vsip::impl::alloc_align;

  int align = 256;

  input_frame_buffer_  = new cval_type*[nframe];
  output_frame_buffer_ = new cval_type*[nframe];

  cube_in_type  cube_in(pt_npols, npulse_, ncsamples_, map_in_);
  range_in_type current_range(nframe);

  assert(cube_rg_.block().admitted() == false);

  size_t in_size  = subblock_domain(cube_rg_).size()  * sizeof(cval_type);
  size_t out_size = subblock_domain(cube_img_).size() * sizeof(cval_type);

  for (index_type frame = 0; frame < nframe; ++frame) 
  {
    input_frame_buffer_[frame] =
      alloc_align<cval_type>(align, in_size);

    if (map_out_.subblock() != vsip::no_subblock)
      read_frame(get_local_view(cube_in),
		 vsip::impl::get_local_view(current_range), frame);

    cube_rg_.block().rebind(input_frame_buffer_[frame]);
    cube_rg_.block().admit(false);
    cube_rg_(vsip::Domain<3>(pt_npols, npulse_, ncsamples_)) = cube_in;
    cube_rg_.block().release(true);

    output_frame_buffer_[frame] =
      alloc_align<cval_type>(align, out_size);
  }

  current_range_ = current_range;
}

template <typename T>
void
SarSim<T>::fini_io(index_type nframe) {
  cube_out_type cube_out(pt_npols, npulse_, nrange_, map_out_);

  // Release the last frame of output data.
  assert(cube_img_.block().admitted() == true);
  cube_img_.block().release(true);

  // Write each frame of data to disk.
  for (index_type frame = 0; frame < nframe; ++frame) {
    cube_img_.block().rebind(output_frame_buffer_[frame]);
    cube_img_.block().admit(true);
    cube_out = cube_img_;
    cube_img_.block().release(false);

    if (map_out_.subblock() != vsip::no_subblock)
      write_frame(get_local_view(cube_out));
  }

  // Free up resources allocated in init_io.
  for (index_type frame = 0; frame < nframe; ++frame) {
    vsip::impl::free_align((void*)input_frame_buffer_[frame]);
    vsip::impl::free_align((void*)output_frame_buffer_[frame]);
  }

  delete[] input_frame_buffer_;
  delete[] output_frame_buffer_;
}

template <typename T>
template <typename Block1,
	  typename Block2>
void 
SarSim<T>::read_frame(
  vsip::Tensor<cval_type, Block1> cube_in,
  vsip::Vector<io_type, Block2>  current_range,
  index_type frame)
{
  vsip::impl::profile::Scope<vsip::impl::profile::user> scope("read_frame");

  // Read a frame of pulses into the input buffer.
  for (index_type p=0; p < npulse_; p++) {
    for (int pol = pt_first; pol < pt_npols; ++pol) {
      // Normally, we only read pulses for the polarities that interest
      // us.  But, we always read the pt_first polarity at the center
      // of the frame so that we can get the range associated with the
      // frame.
      const bool center_frame = (pol == pt_first && p == npulse_ / 2 - 1); 
      if (!center_frame && !pol_on_[pol])
	continue;
      io_type range = read_pulse(pol, p);
      if (center_frame) {
	// We remember the range of the first frame; during azimuth
	// processing we use the difference from the current range to
	// the initial range.
	current_range(frame) = range;
	if (!pol_on_[pol]) 
	    continue;
      }
      cube_in(pol, p, vsip::Domain<1>(ncsamples_)) = vec_iq_;
    }
  }
}

template <typename T>
template <typename Block>
void 
SarSim<T>::write_frame(vsip::Tensor<cval_type, Block> cube_out)
{
  vsip::impl::profile::Scope<vsip::impl::profile::user> scope("write_frame");

  whole_domain_type whole = vsip::whole_domain;

  for (int pol = pt_first; pol < pt_npols; pol++) {
    write_output_header(pol);
    for (index_type i=0; i < nrange_; i++) {
      azbuf_(vsip::Domain<1>(npulse_, 1, npulse_)) = cube_out(pol, whole, i);
      write_output(pol);
    }
  }
}

template <typename T>
void
SarSim<T>::process(index_type nframe, bool pol_on[pt_npols]) {

  vsip::Domain<3> second_dom(pt_npols, vsip::Domain<1>(npulse_, 1, npulse_), 
			     nrange_);

  for (int pol = pt_first; pol < pt_npols; ++pol)
    pol_on_[pol] = pol_on[pol];

#if USE_SA
  vsip::Setup_assign corner_turn(cube_az_(second_dom), cube_rg_);
  std::cout << "corner_turn: " << corner_turn.impl_type() << std::endl;
#endif

  init_io(nframe);

#if PARALLEL
  {
    vsip::impl::profile::Scope<vsip::impl::profile::user> scope("start-barrier");
    map_rg_.impl_comm().barrier();
  }
#endif

  {
    vsip::impl::profile::Scope_timer time(proc_time_);
    vsip::impl::profile::Scope<vsip::impl::profile::user> scope("process");
    for (index_type frame = 0; frame < nframe; ++frame) {
      io_process(frame);
      range_process();
      // FIXME: remove timer.
      { vsip::impl::profile::Scope_timer time(ct_time_);
	vsip::impl::profile::Scope<vsip::impl::profile::user> scope("corner-turn");
#if USE_SA
	 corner_turn();
#else
	 cube_az_(second_dom) = cube_rg_;
#endif
      }
      azimuth_process(frame, frame == nframe - 1);
    }
  }

  fini_io(nframe);
}

template <typename T>
void 
SarSim<T>::io_process(index_type frame)
{
  vsip::impl::profile::Scope<vsip::impl::profile::user> scope("input-process");

  // On first iteration, block will initially be released.
  if (cube_rg_.block().admitted())
    cube_rg_.block().release(false);
  cube_rg_.block().rebind(input_frame_buffer_[frame]);
  cube_rg_.block().admit(true);

  // Save the last frame of output data, set up to collect next frame.
  if (cube_img_.block().admitted())
    cube_img_.block().release(true);
  cube_img_.block().rebind(output_frame_buffer_[frame]);
  cube_img_.block().admit(false);
}

template <typename T>
void 
SarSim<T>::range_process()
{
  vsip::impl::profile::Scope_timer time(rp_time_);
  vsip::impl::profile::Scope<vsip::impl::profile::user> scope("range-process");

  vsip::Domain<1> conv_dom(ncsamples_ - niq_ +1);
  vsip::Domain<1> keep_dom(ncsamples_ - niq_);
  vsip::Domain<1> zero_dom(ncsamples_ - niq_, 1, 
			   nrange_ - (ncsamples_ - niq_));

  whole_domain_type whole = vsip::whole_domain;

  vsip::Domain<3> g_dom = global_domain(cube_rg_);
  typename cube_rg_type::local_type l_cube_rg = get_local_view(cube_rg_);

  // Read a frame of pulses.  Perform range-processing on each pulse
  // as it is read.
  for (index_type lpol = 0; lpol<g_dom[0].size(); ++lpol) {
    int pol = g_dom[0].impl_nth(lpol);
#if VERBOSE
    std::cout << "(" << map_rg_.impl_rank() << ") rg pol " << pol << "  "
	      << get_local_view(cube_rg_(pol, 0, whole)).get(0)
	      << std::endl;
#endif
    for (index_type lp = 0; lp<g_dom[1].size(); ++lp) {
      rg_line_cnt_++;
      vec_iq_ = l_cube_rg(lpol, lp, vsip::Domain<1>(ncsamples_));

      // Implement the FIR which is an upper sideband filter, and
      // zero-pad to the end (implicit because arrays are initialized to
      // zero).  The current frame is stored in the second half of the
      // main processing array, 'cbuf', as the first half holds the
      // previous frame.  (During the first frame, the first half of the
      // main processing array is 0.)
      iconv_(vec_iq_.real(), line1_(conv_dom).real());
      qconv_(vec_iq_.imag(), line1_(conv_dom).imag());
      rvm1_time_.start();
      line1_(keep_dom) *= vsip::impl::cast_view<cval_type>(w_eq_.row(pol)(keep_dom));
      rvm1_time_.stop();
      line1_(zero_dom) = 0.f;
      // Perform range processing on these I/Q pairs by taking their DFT.
      range_fft_(line1_, line2_);
      // Apply RCS weighting.
      rvm2_time_.start();
      line2_ *= vsip::impl::cast_view<value_type>(rcs_);
      rvm2_time_.stop();
      l_cube_rg(lpol, lp, whole) = line2_;
    }
  }
}

template <typename T>
void
SarSim<T>::azimuth_process(index_type frame, bool is_last)
{
  vsip::impl::profile::Scope_timer time(ap_time_);
  vsip::impl::profile::Scope<vsip::impl::profile::user> scope("azimuth-process");

  whole_domain_type whole = vsip::whole_domain;

  vsip::Domain<1> first_dom(npulse_);
  vsip::Domain<1> second_dom(npulse_, 1, npulse_);
  
  // Find the initial convolution kernel, then enter the main
  // azimuth processing loop.  There are 31 total kernels, a set of
  // 16 kernels covers all range gates, and kernel0 specifies which
  // subset will be used for this frame.
  int k0 = 8 - int((get_local_view(current_range_)(0) -
		    get_local_view(current_range_)(frame))/(swath_ / 16));
  vsip::Domain<3> g_dom = global_domain(cube_az_);
  typename cube_az_type::local_type l_cube_az   = get_local_view(cube_az_);
  typename cube_img_type::local_type l_cube_img = get_local_view(cube_img_);
  // for (int pol = pt_first; pol < pt_npols; ++pol) {
  for (index_type lpol = 0; lpol<g_dom[0].size(); ++lpol) {
#if VERBOSE
    int pol = g_dom[0].impl_nth(lpol);
    std::cout << "(" << map_az_.impl_rank() << ") az pol: " << pol
	      << "  k0: " << k0
	      << "  " << get_local_view(cube_az_(pol, whole, 0)).get(0)
	      << std::endl;
#endif
    for (index_type li = 0; li<g_dom[2].size(); ++li) {
      az_line_cnt_++;
      int i = g_dom[2].impl_nth(li);
      // Perform DFT.
      az_for_fft_(l_cube_az(lpol, whole, li), azbuf_);
      // If this is not the last frame, make room for next PRI by
      // shifting the latest frame to the first half of the main
      // processing array.
      if (!is_last)
	l_cube_az(lpol, first_dom, li) = l_cube_az(lpol, second_dom, li);
      // Multiply DFT result by appropriate convolution
      // kernel. (Kernels were already transformed during
      // initialization.) 
      avm_time_.start();
      azbuf_ *= vsip::impl::cast_view<cval_type>
	(cphase_.row(k0 + i * 16 / nrange_));
      avm_time_.stop();
      // Perform IDFT.
      az_inv_fft_(azbuf_, azbuf2_);
      // Write the second half of this range cell to file.
      l_cube_img(lpol, whole, li) = azbuf2_(second_dom);
    }
  }
}

template <typename T>
void
SarSim<T>::report_performance() const
{
  // On each range processing frame, we do:
  //   foreach polariy (pt_npols)
  //     foreach pulse (npulse_)
  //       - 1 x FFT(nrange): 5 * nrange * log(nrange)
  //       - 2 x conv       : 2 * (ncsamples-niq) * niq
  //       - 1 x vmul       : 6 * ncsamples_ - niq_ (*)
  //       - 1 x vmul       : 6 * nrange

  float rp_ops = 
    rg_line_cnt_ * (
      1 * 5 * nrange_ * log((float)nrange_) / log(2.f) +
      2 * 2 * (ncsamples_ - niq_) * niq_ +
      1 * 6 * (nrange_ - niq_) +
      1 * 6 * nrange_  );

  float rvm1_ops = rg_line_cnt_ * ( 1 * 6 * (nrange_ - niq_) );
  float rvm2_ops = rg_line_cnt_ * ( 1 * 6 * nrange_ );

  // On each azimuth processing frame, we do:
  //   foreach polariy (pt_npols)
  //     foreach range cell (nrange_)
  //       - 2 x FFT(2*npulse): 5 * 2*npulse * log(2*npulse)
  //       - 1 x vmul         : 6 * 2*npulse

  float ap_ops =
    az_line_cnt_ * (
      2 * 5 * 2*npulse_ * log(2.f*npulse_) / log(2.f) +
      1 * 6 * 2*npulse_);

  float avm_ops = az_line_cnt_ * ( 1 * 6 * 2*npulse_ );

  // Compute mflops.
  // Timers rp_time_ and ap_time_ are triggered once per frame, so
  // rp_time_.count() == nframe.

  float rp_mflops = rp_ops /* * rp_time_.count() */ / (1e6 * rp_time_.total());
  float ap_mflops = ap_ops /* * ap_time_.count() */ / (1e6 * ap_time_.total());
  float proc_mflops = (rp_ops + ap_ops) / (1e6 * proc_time_.total());

  float rvm1_mflops = rvm1_ops / (1e6 * rvm1_time_.total());
  float rvm2_mflops = rvm2_ops / (1e6 * rvm2_time_.total());
  float avm_mflops  = avm_ops  / (1e6 * avm_time_.total());

  printf("Total Processing  : %7.2f mflops (%6.2f s)\n",
	 proc_mflops, proc_time_.total());
  printf("  corner-turn     :                (%6.2f s)\n",
	 ct_time_.total());

  printf("Range Processing  : %7.2f mflops (%6.2f s)\n",
	 rp_mflops, rp_time_.total());
  printf("   range fft      : %7.2f mflops (%6.2f s)\n",
	 range_fft_.impl_performance("mops"),
	 range_fft_.impl_performance("time"));
  printf("   iconv          : %7.2f mflops (%6.2f s)\n",
	 iconv_.impl_performance("mops"),
	 iconv_.impl_performance("time"));
  printf("   qconv          : %7.2f mflops (%6.2f s)\n",
	 qconv_.impl_performance("mops"),
	 qconv_.impl_performance("time"));

  printf("Azimuth Processing: %7.2f mflops (%6.2f s)\n",
	 ap_mflops, ap_time_.total());
  printf("   az for fft     : %7.2f mflops (%6.2f s)\n",
	 az_for_fft_.impl_performance("mops"),
	 az_for_fft_.impl_performance("time"));
  printf("   az inv fft     : %7.2f mflops (%6.2f s)\n",
	 az_inv_fft_.impl_performance("mops"),
	 az_inv_fft_.impl_performance("time"));

  printf("\n");
  printf("  rvm1            : %7.2f mflops (%6.2f s)\n", rvm1_mflops, rvm1_time_.total());
  printf("  rvm2            : %7.2f mflops (%6.2f s)\n", rvm2_mflops, rvm2_time_.total());
  printf("  avm             : %7.2f mflops (%6.2f s)\n", avm_mflops, avm_time_.total());
}
