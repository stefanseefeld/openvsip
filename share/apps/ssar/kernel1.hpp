/* Copyright (c) 2006-2011 CodeSourcery, Inc.  All rights reserved. */

/* Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
         copyright notice, this list of conditions and the following
	 disclaimer in the documentation and/or other materials
	 provided with the distribution.
       * Neither the name of CodeSourcery nor the names of its
         contributors may be used to endorse or promote products
	 derived from this software without specific prior written
	 permission.

   THIS SOFTWARE IS PROVIDED BY CODESOURCERY, INC. "AS IS" AND ANY
   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CODESOURCERY BE LIABLE FOR
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
   BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
   EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  */

/// Description
///   VSIPL++ implementation of SSCA #3: Kernel 1, Image Formation

#include <vsip/selgen.hpp>
#include <vsip/dense.hpp>
#include <vsip_csl/strided.hpp>
#include <vsip_csl/profile.hpp>
#include <vsip_csl/matlab_utils.hpp>
#include <vsip_csl/save_view.hpp>
#include <vsip_csl/load_view.hpp>
#include <vsip_csl/udk.hpp>
#include <stdint.h>
#include "ssar.hpp"

#if 0
#define VERBOSE
#define SAVE_VIEW(a, b, c)    \
  vsip_csl::save_view_as<complex<float> >((this->data_dir_ + a).c_str(), b, c)
#else
#define SAVE_VIEW(a, b, c)
#endif



// This compiler switch changes the way the digital spotlighting routine
// interacts with the cache.
// 
// A value of '1' will use 1-D FFTs instead of FFTMs (multiple-FFTs)
// and it will perform several operations at a time when it processes the
// rows, leading to more cache hits on some architectures (like x86).
// 
// A value of '0' will utilize FFTMs and likewise perform each step on the 
// entire image before proceeding to the next.  This can be more efficient
// on certain architectures (such as Cell/B.E.) where large computations
// can be distributed amongst several compute elements and run in parallel.
#if VSIP_IMPL_CBE_SDK || VSIP_IMPL_HAVE_CUDA
#  define DIGITAL_SPOTLIGHT_BY_ROW  0
#else
#  define DIGITAL_SPOTLIGHT_BY_ROW  1
#endif


// On Cell/B.E. platforms, this may be defined to utilize a user-defined
// kernel for part of the interpolation stage.
//
// Setting it to '1' will utilize the kernel for the range-loop portion
// of the computation (polar-to-rectangular interpolation) which will 
// distribute groups of columns of the image to the SPEs.  This processes
// data in parallel with a corresponding increase in performance.
//
// Setting it to '0' will perform the computation entirely on the PPE
// as it does on x86 processors.
#if VSIP_IMPL_CBE_SDK
#  define USE_CELL_UKERNEL 1
#else
#  define USE_CELL_UKERNEL 0
#endif

#if USE_CELL_UKERNEL
#  include <vsip_csl/ukernel/host/ukernel.hpp>
#  include <cbe/host/interp.hpp>
#endif


// On CUDA-enabled platforms, this may be defined to use the custom kernel
// for the range loop (as with Cell above).
#if VSIP_IMPL_HAVE_CUDA
# include <cuda/interp.hpp>
#endif

using namespace vsip;
namespace udk = vsip_csl::udk;

template <typename T>
class Kernel1_base
{
protected:
#if VSIP_IMPL_CBE_SDK
  typedef Layout<2, row2_type, aligned_128, split_complex> row_layout_type;
  typedef Layout<2, col2_type, aligned_128, split_complex> col_layout_type;
  typedef vsip_csl::Strided<2, T, row_layout_type> real_row_block_type;
  typedef vsip_csl::Strided<2, T, col_layout_type> real_col_block_type;
  typedef vsip_csl::Strided<2, complex<T>, row_layout_type> complex_row_block_type;
  typedef vsip_csl::Strided<2, complex<T>, col_layout_type> complex_col_block_type;
#else
  typedef Dense<2, complex<T>, col2_type> complex_col_block_type;
  typedef Dense<2, complex<T>, row2_type> complex_row_block_type;
  typedef Dense<2, T, col2_type> real_col_block_type;
  typedef Dense<2, T, row2_type> real_row_block_type;
#endif
  typedef Matrix<complex<T>, complex_col_block_type> complex_col_matrix_type;
  typedef Matrix<complex<T>, complex_row_block_type> complex_matrix_type;
  typedef Matrix<T, real_col_block_type> real_col_matrix_type;
  typedef Matrix<T, real_row_block_type> real_matrix_type;
  typedef Vector<complex<T> > complex_vector_type;
  typedef Vector<T> real_vector_type;

  Kernel1_base(ssar_options const &opt, Local_map const &huge_map);
  ~Kernel1_base() {}

  scalar_f scale_;
  length_type n_;
  length_type mc_;
  length_type m_;
  bool swap_bytes_;
  std::string data_dir_;
  length_type nx_;
  length_type interp_sidelobes_;
  length_type I_;
  T range_factor_;
  T aspect_ratio_;
  T L_;
  T Y0_;
  T X0_;
  T Xc_;
  T dkx_;
  T kx_min_;
  T kxs_;

  complex_vector_type fast_time_filter_;
  complex_matrix_type fs_ref_;
  complex_matrix_type fs_ref_preshift_;
  real_vector_type ks_;
  real_vector_type ucs_;
  real_vector_type us_;
  real_matrix_type kx_;
};

template <typename T>
Kernel1_base<T>::Kernel1_base(ssar_options const &opt, Local_map const &huge_map)
  : scale_(opt.scale), n_(opt.n), mc_(opt.mc), m_(opt.m), swap_bytes_(opt.swap_bytes),
    data_dir_(opt.data_dir),
    fast_time_filter_(n_),
    fs_ref_(n_, m_),
    fs_ref_preshift_(n_, m_, huge_map),
    ks_(n_),
    ucs_(mc_),
    us_(m_),
    kx_(n_, m_)
{
  using vsip_csl::matlab::fftshift;
  using vsip_csl::matlab::fd_fftshift;
  using vsip_csl::load_view_as;

  interp_sidelobes_ = 8;     // 2. (scalar, integer) number of 
                             //    neighboring sidelobes used in sinc interp.
                             //    WARNING: Changing 'nInterpSidelobes' 
                             //    changes the size of the Y dimension  
                             //    of the resulting image.

  // SPATIAL/GEOMETRIC PARAMETERS

  range_factor_ = T(10);     // 3. (scalar, real) ratio of swath's range
                             //    center point to the synthetic aperture's
                             //    half-length (unitless)

  aspect_ratio_ = T(0.4);    // 4. (scalar, real) ratio of swath's range
                             //    to cross-range (unitless)

  L_ = T(100) * scale_;      // 5. (scalar, real) half of the synthetic
                             //    aperture (in meters, synthetic aperterture 
                             //    is 2*L). 'L' is the only physical dimension
                             //    (since we have the L=Y0 simplification)
                             //    all other SAR parameters are keyed on
                             //    this value.

  Y0_ = L_;                  // 6. (scalar, real) target area's half
                             //    cross-range (within [Yc-Y0, Yc+Y0], meters)

  X0_ = aspect_ratio_ * Y0_; // 7. (scalar, real) target area's half range
                             //    (within [Xc-X0, Xc+X0], in meters)

  Xc_ = range_factor_ * Y0_; // 8. (scalar, real) swath's range
                             //    center point (m)

  // Load scale-dependent processing parameters.
  real_vector_type k(n_);
  real_vector_type uc(mc_);
  real_vector_type u(m_);
  real_vector_type ku0(m_);

  load_view_as<complex<float>, complex_vector_type>
    (opt.fast_time_filter.c_str(), fast_time_filter_, swap_bytes_);
  load_view_as<float, real_vector_type>(opt.slow_time_wavenumber.c_str(), k, swap_bytes_);
  load_view_as<float, real_vector_type>
    (opt.slow_time_compressed_aperture_position.c_str(), uc, swap_bytes_);
  load_view_as<float, real_vector_type>(opt.slow_time_aperture_position.c_str(), u, swap_bytes_);
  load_view_as<float, real_vector_type>(opt.slow_time_spatial_frequency.c_str(), ku0, swap_bytes_);

  // 60. (1 by n array of reals) fftshifted slow-time wavenumber
  ks_ = freqswap(k);

  // 61. (1 by mc array of reals) fftshifted slow-time synthetic aperture
  ucs_ = freqswap(uc);

  // 67. (1 by m array of reals) shifted u
  us_ = freqswap(u);

  // 70. (1 by m array of reals) ku0 is transformed into the intermediate 
  //     (n by m array of reals) kx0 (wn)
  real_matrix_type ku(n_, m_);
  ku = vmmul<row>(ku0, real_matrix_type(n_, m_, T(1)));

  real_matrix_type k1(n_, m_);
  k1 = vmmul<col>(k, real_matrix_type(n_, m_, T(1)));

  real_matrix_type kx0(n_, m_);
  kx0 = 4 * sq(k1) - sq(ku);

  // 71. (n by m array of reals) kx is the Doppler domain range 
  //     wavenumber (wn)    
  kx_ = sqrt(max(T(0), kx0));

  // 72. (scalar, real) minimum wavenum (wn)
  Index<2> idx;
  kx_min_ = minval(kx_, idx);

  // 73. (scalar, real) maximum wavenum (wn)
  T kx_max = maxval(kx_, idx);

  // 74. (scalar, real) Nyquist sample spacing in kx domain (wn)
  dkx_ = M_PI / X0_;

  // 75. (scalar, integer) nx0 is the min number of required kx samples 
  //     (pixels);  (later it will be increased slightly to avoid 
  //     negative array indexing)
  index_type nx0 = static_cast<index_type>
    (2 * ceil((kx_max - kx_min_) / (2 * dkx_)));   

  // generate the Doppler domain representation the reference signal's 
  // complex conjugate

  // 76. (n by m array of complex nums) reference signal's complex conjugate
  //
  // this is equivalent to the elementwise operation
  //    fs_ref_.put(i, j, (kx_.get(i, j) > 0) ? exp(...) : complex<T>(0));
  fs_ref_ = ite(kx_ > 0, exp(complex<T>(0, 1) * 
    (Xc_ * (kx_ - 2 * k1) + T(0.25 * M_PI) + ku)), complex<T>(0));
  fftshift<col>(fs_ref_, fs_ref_preshift_);
  fd_fftshift<row>(fs_ref_preshift_, fs_ref_preshift_);

  SAVE_VIEW("p76_fs_ref.view", fs_ref_, swap_bytes_);

  // 78. (scalar, int) interpolation processing sliver size
  I_ = 2 * interp_sidelobes_ + 1;
                            
  // 79. (scalar, real) +/- interpolation neighborhood size in KX domain
  kxs_ = interp_sidelobes_ * dkx_;  

  // 80. (scalar, int) total number of kx samples required in the SAR 
  //     image's col (in pixels; increased to avoid negative array 
  //     indexing in interpolation loop)
  nx_ = nx0 + 2 * interp_sidelobes_ + 4;


#ifdef VERBOSE
  std::cout << "kx_min = " << kx_min_ << std::endl;
  std::cout << "kx_max = " << kx_max << std::endl;
  std::cout << "dkx = " << dkx_ << std::endl;
  std::cout << "n = " << n_ << std::endl;
  std::cout << "mc = " << mc_ << std::endl;
  std::cout << "m = " << m_ << std::endl;
  std::cout << "nx0 = " << nx0 << std::endl;
  std::cout << "I = " << I_ << std::endl;
  std::cout << "kxs = " << kxs_ << std::endl;
  std::cout << "nx = " << nx_ << std::endl;
#endif
}






template <typename T>
class Kernel1 : public Kernel1_base<T>
{
public:
  typedef typename Kernel1_base<T>::complex_col_block_type complex_col_block_type;
  typedef typename Kernel1_base<T>::complex_col_matrix_type complex_col_matrix_type;
  typedef typename Kernel1_base<T>::complex_matrix_type complex_matrix_type;
  typedef typename Kernel1_base<T>::complex_vector_type complex_vector_type;
  typedef typename Kernel1_base<T>::real_col_matrix_type real_col_matrix_type;
  typedef typename Kernel1_base<T>::real_matrix_type real_matrix_type;
  typedef typename Kernel1_base<T>::real_vector_type real_vector_type;

  typedef Fft<const_Vector, complex<T>, complex<T>, fft_fwd, by_reference> col_fft_type;
  typedef Fft<const_Vector, complex<T>, complex<T>, fft_fwd, by_reference> row_fft_type;
  typedef Fft<const_Vector, complex<T>, complex<T>, fft_inv, by_reference> inv_fft_type;
  typedef Fftm<complex<T>, complex<T>, row, fft_fwd, by_reference> row_fftm_type;
  typedef Fftm<complex<T>, complex<T>, row, fft_fwd, by_value> val_row_fftm_type;
  typedef Fftm<complex<T>, complex<T>, col, fft_fwd, by_value> val_col_fftm_type;
  typedef Fftm<complex<T>, complex<T>, row, fft_inv, by_reference> inv_row_fftm_type;
  typedef Fftm<complex<T>, complex<T>, row, fft_inv, by_value> val_inv_row_fftm_type;
  typedef Fftm<complex<T>, complex<T>, col, fft_inv, by_reference> inv_col_fftm_type;

  Kernel1(ssar_options const &opt, Local_map huge_map);
  ~Kernel1() {}

  void process_image(complex_matrix_type const input, 
    real_matrix_type output);

  length_type output_size(dimension_type dim) 
    { 
      assert(dim < 2);
      return (dim ? this->nx_ : m_);
    }

private:
  void
  digital_spotlighting(complex_matrix_type s_raw);

  void
  interpolation(real_matrix_type image);

private:
  scalar_f scale_;
  length_type n_;
  length_type mc_;
  length_type m_;

  length_type fast_time_filter_ops_;
  length_type slow_time_compression_ops_;
  length_type slow_time_decompression_ops_;
  length_type matched_filter_ops_;
  length_type digital_spotlighting_ops_;

  length_type range_loop_ops_;
  length_type magnitude_ops_;
  length_type interp_fftm_ops_;
  length_type interpolation_ops_;

  length_type kernel1_total_ops_;

  complex_col_matrix_type s_filt_;
  complex_matrix_type s_filt_t_;
#if DIGITAL_SPOTLIGHT_BY_ROW
  complex_matrix_type s_compr_filt_;
#else
  complex_col_matrix_type s_compr_filt_;
  complex_col_matrix_type s_compr_filt_shift_;
#endif
  complex_matrix_type s_decompr_filt_;
#if !DIGITAL_SPOTLIGHT_BY_ROW
  complex_matrix_type s_decompr_filt_shift_;
#endif
  complex_matrix_type fsm_;
  complex_col_matrix_type fsm_t_;
  Matrix<uint32_t, Dense<2, uint32_t, col2_type> > icKX_;
  Tensor<T, Dense<3, T, tuple<1, 0, 2> > > SINC_HAM_;
  real_vector_type KX0_;
  complex_col_matrix_type F_;
  complex_matrix_type spatial_;

#if DIGITAL_SPOTLIGHT_BY_ROW
  Vector<complex<T> > fs_row_;
  Vector<complex<T> > fs_spotlit_row_;
  col_fft_type ft_fft_;
  row_fft_type st_fft_;
  row_fft_type compr_fft_;
  inv_fft_type decompr_fft_;
#else
  complex_matrix_type fs_spotlit_;
  complex_matrix_type s_compr_;
  complex_matrix_type fs_;
  complex_matrix_type fs_padded_;
  val_col_fftm_type ft_fftm_;
  val_row_fftm_type st_fftm_;
  row_fftm_type compr_fftm_;
  val_inv_row_fftm_type decompr_fftm_;
#endif
  inv_row_fftm_type ifftmr_;
  inv_col_fftm_type ifftmc_;
};


template <typename T>
Kernel1<T>::Kernel1(ssar_options const &opt, Local_map huge_map)
  : Kernel1_base<T>(opt, huge_map),
    scale_(opt.scale), n_(opt.n), mc_(opt.mc), m_(opt.m), 
    s_filt_(n_, mc_, huge_map),
    s_filt_t_(n_, mc_),
    s_compr_filt_(n_, mc_),
#if !DIGITAL_SPOTLIGHT_BY_ROW
    s_compr_filt_shift_(n_, mc_, huge_map),
#endif
    s_decompr_filt_(n_, m_),
#if !DIGITAL_SPOTLIGHT_BY_ROW
    s_decompr_filt_shift_(n_, m_, huge_map),
#endif
    fsm_(n_, m_, huge_map),
    fsm_t_(n_, m_, huge_map),
    icKX_(n_, m_, huge_map),
    SINC_HAM_(n_, m_, 20 /*this->I_*/, huge_map),
    KX0_(this->nx_),
    F_(this->nx_, m_, huge_map),
    spatial_(this->nx_, m_, huge_map),
#if DIGITAL_SPOTLIGHT_BY_ROW
    fs_row_(mc_),
    fs_spotlit_row_(m_),
    ft_fft_(Domain<1>(n_), T(1)),
    st_fft_(Domain<1>(m_), T(1)),
    compr_fft_(Domain<1>(mc_), static_cast<T>(m_) / mc_),
    decompr_fft_(Domain<1>(m_), T(1.f/m_)),
#else
    fs_spotlit_(n_, m_),
    s_compr_(n_, mc_),
    fs_(n_, mc_, huge_map),
    fs_padded_(n_, m_, huge_map),
    ft_fftm_(Domain<2>(n_, mc_), T(1)),
    st_fftm_(Domain<2>(n_, m_), T(1)),
    compr_fftm_(Domain<2>(n_, mc_), static_cast<T>(m_) / mc_),
    decompr_fftm_(Domain<2>(n_, m_), T(1.f/m_)),
#endif
    ifftmr_(Domain<2>(this->nx_, m_), T(1./m_)),
    ifftmc_(Domain<2>(this->nx_, m_), T(1./this->nx_))
{
  using vsip_csl::matlab::fftshift;
  using vsip_csl::matlab::fd_fftshift;

  // 83. (1 by nx array of reals) uniformly-spaced KX0 points where 
  //     interpolation is done  
  KX0_ = this->kx_min_ + 
    (vsip::ramp(T(0),T(1), this->nx_) - this->interp_sidelobes_ - 2) *
    this->dkx_;

  // Pre-computed values for eq. 62.
  real_matrix_type nmc_ones(n_, mc_, T(1));
  s_compr_filt_ = vmmul<col>(this->fast_time_filter_, 
    exp(complex<T>(0, 2) * vmmul<col>(this->ks_, nmc_ones) *
      (sqrt(sq(this->Xc_) + sq(vmmul<row>(this->ucs_, nmc_ones))) - this->Xc_)));
#if !DIGITAL_SPOTLIGHT_BY_ROW
  fftshift<row>(s_compr_filt_, s_compr_filt_shift_);
  fd_fftshift<col>(s_compr_filt_shift_, s_compr_filt_shift_);
#endif

  // Pre-computed values for eq. 68. 
  real_matrix_type nm_ones(n_, m_, T(1));
  s_decompr_filt_ = exp( complex<T>(0, 2) * vmmul<col>(this->ks_, nm_ones) *
    (this->Xc_ - sqrt(sq(this->Xc_) + sq(vmmul<row>(this->us_, nm_ones)))) );
#if !DIGITAL_SPOTLIGHT_BY_ROW
  fftshift<row>(s_decompr_filt_, s_decompr_filt_shift_);
  fd_fftshift<row>(s_decompr_filt_shift_, s_decompr_filt_shift_);
#endif

  // Pre-computed values for eq. 92.
  for (index_type i = 0; i < n_; ++i)
    for (index_type j = 0; j < m_; ++j)
    {
      // 87. (1 by m array of ints) icKX are the indices of the closest 
      //     cross-range sliver in the KX domain
      icKX_.put(i, j, static_cast<index_type>(
        ((this->kx_.get(i, j) - KX0_.get(0)) / this->dkx_) + 0.5f)
        - this->interp_sidelobes_);

      // 88. (I by m array of ints) ikx are the indices of the slice that 
      //     include the cross-range sliver at its center
      index_type ikxrows = icKX_.get(i, j);

      for (index_type h = 0; h < this->I_; ++h)
      {
        // 89. (I by m array of reals) nKX are the signal values 
        //     of the corresponding slice
        T nKX =  KX0_.get(ikxrows + h);

        // 90. (I by m array of reals) SINC is the interpolating window 
        //     (note not stand-alone sinc coefficients)
        T sx = M_PI * (nKX - this->kx_.get(i, j)) / this->dkx_;

        // reduce interpolation computational costs by using a tapered 
        // window
    
        // 91. (I by m array of reals) (not stand-alone Hamming 
        //     coefficients)
        SINC_HAM_.put(i, j, h, (sx ? sin(sx) / sx : 1) * 
          (0.54 + 0.46 * cos((M_PI / this->kxs_) * 
            (nKX - this->kx_.get(i, j)))) );
      }
    }

  // Calculate operation counts

  // Digital Spotlighting
  //   : Forward FFTs = 5 N log2(N) per row/column
  //   : Fast time filter = Forward FFT (by column) plus a vector multiply.
  //       There are 6 ops/point for vmul.
  //   : Slow time compression = Forward FFT (by row) of length mc followed
  //       by an Inverse FFT (by row) of length m (after bandwidth expansion).
  //   : Slow time decompression = Vector multiply, Forward FFT (by row)
  //   : 2-D Matched filter ops = Vector multiply across columns, for each row.
  float rows = static_cast<float>(n_);
  float cols = static_cast<float>(mc_);
  fast_time_filter_ops_ = static_cast<length_type>(
    5 * rows * log(rows)/log(2.f) * cols +
    6 * rows * cols);

  float bwx_cols = static_cast<float>(m_);
  slow_time_compression_ops_ = static_cast<length_type>(
    5 * cols * log(cols)/log(2.f) * rows +
    5 * bwx_cols * log(bwx_cols)/log(2.f) * rows);

  slow_time_decompression_ops_ = static_cast<length_type>(
    6 * rows * bwx_cols +
    5 * bwx_cols * log(bwx_cols)/log(2.f) * rows);

  matched_filter_ops_ = static_cast<length_type>(
    6 * rows * bwx_cols);

  digital_spotlighting_ops_ = fast_time_filter_ops_ + matched_filter_ops_ + 
    slow_time_compression_ops_ + slow_time_decompression_ops_;


  // Interpolation
  //   : Range loop = scalar/complex multiply + complex add = 4 ops/point
  //   : Complex mag = two multiplies, an add and a square root = 4 ops/point
  //   : Inverse FFTMs = 5 N log2(N) in each dimension, times the opposite
  //       dimension (there are two total, one in each direction)

  range_loop_ops_ = 4 * m_ * n_ * this->I_;

  magnitude_ops_ = 4 * m_ * this->nx_;

  rows = static_cast<float>(m_);
  cols = static_cast<float>(this->nx_);
  interp_fftm_ops_ = static_cast<length_type>(
      5 * rows * log(rows)/log(2.f) * cols +
      5 * cols * log(cols)/log(2.f) * rows);

  interpolation_ops_ = interp_fftm_ops_ + range_loop_ops_ + magnitude_ops_;


  // Grand Total
  
  kernel1_total_ops_ = digital_spotlighting_ops_ + interpolation_ops_;
}


template <typename T>
void
Kernel1<T>::process_image(complex_matrix_type const input, 
  real_matrix_type output)
{
  using vsip_csl::profile::Scope;
  using vsip_csl::profile::user;
  assert(input.size(0) == n_);
  assert(input.size(1) == mc_);
  assert(output.size(0) == m_);
  assert(output.size(1) == this->nx_);

  // Time the remainder of this function, provided profiling is enabled 
  // (pass '--vsip-profile-mode=[accum|trace]' on the command line).  
  // If profiling is not enabled, then this statement has no effect.
  Scope<user> scope("Kernel1 total", kernel1_total_ops_);


  // Digital spotlighting and bandwidth-expansion using slow-time 
  // compression and decompression.  
  this->digital_spotlighting(input);


  // Digital reconstruction via spatial frequency interpolation.
  this->interpolation(output);
}



#if DIGITAL_SPOTLIGHT_BY_ROW

template <typename T>
void
Kernel1<T>::digital_spotlighting(complex_matrix_type s_raw)
{
  using vsip_csl::profile::Scope;
  using vsip_csl::profile::user;
  Scope<user> scope("digital_spotlighting", digital_spotlighting_ops_);

  using vsip_csl::matlab::fftshift;
  assert(s_raw.size(0) == n_);
  assert(s_raw.size(1) == mc_);

  // 64. (scalar, int) number of zeros to be padded into the ku domain 
  //     for slow-time upsampling
  length_type mz = m_ - mc_;

  // Domains for bandwidth expansion.
  Domain<1> left(0, 1, mc_/2);
  Domain<1> center_dst(mc_/2, 1, mz);
  Domain<1> right_dst(mz + mc_/2, 1, mc_/2);
  Domain<1> right_src(mc_/2, 1, mc_/2);

  // left/right domains for emulating fftshift of fs_spotlit_.
  Domain<1> ldom(0, 1, m_/2);
  Domain<1> rdom(m_/2, 1, m_/2);

  // The baseband reference signal is first transformed into the Doppler 
  // (spatial frequency) domain.  

  // corner-turn: to col-major
  fftshift(s_raw, s_filt_); 

  // 59. (n by mc array of complex numbers) filtered echoed signal
  // 
  // Note that the fast-time filter is combined with the compression
  // along the slow-time axis below.  
  for (index_type j = 0; j < mc_; ++j)
  {
    ft_fft_(s_filt_.col(j));
  }

  // Digital spotlighting and bandwidth expansion in the ku domain 
  // via slow-time compression and decompression:

  // corner-turn: to row-major
  s_filt_t_ = s_filt_;

  for (index_type i = 0; i < n_; ++i)
  {
    // 62. (n by mc array of complex numbers) signal compressed along 
    //     slow-time (note that to view 'sCompr' it will need to be 
    //     fftshifted first.)
    fs_row_ = s_filt_t_.row(i) * s_compr_filt_.row(i);
    
    // 63. (n by mc array of complex numbers) narrow-bandwidth polar format
    //     reconstruction along slow-time
    compr_fft_(fs_row_);

    // 65. (n by m array of complex numbers) zero pad the spatial frequency 
    //     domain's compressed signal along its slow-time (note that to view 
    //     'fsPadded' it will need to be fftshifted first)
    fs_spotlit_row_(left)       = fs_row_(left);
    fs_spotlit_row_(center_dst) = T();
    fs_spotlit_row_(right_dst)  = fs_row_(right_src);

    // 66. (n by m array of complex numbers) transform-back the zero 
    //     padded spatial spectrum along its cross-range
    decompr_fft_(fs_spotlit_row_);

    // 68. (n by m array of complex numbers) slow-time decompression (note 
    //     that to view 'sDecompr' it will need to be fftshifted first.)
    fs_spotlit_row_ *= s_decompr_filt_.row(i);

    // 69. (n by m array of complex numbers) digitally-spotlighted SAR 
    //     signal spectrum
    st_fft_(fs_spotlit_row_);


    // Match filter the spotlighted signal 'fsSpotLit' with the reference's 
    // complex conjugate 'fsRef' along fast-time and slow-time, to remove 
    // the reference signal's spectral components.

    // 77. (n by m array of complex nums) Doppler domain matched-filtered
    //     signal

    // Merge fftshift and vmul:
    //
    //   fftshift(fs_spotlit_, fsm_);
    //   fsm_.row(xr) = fs_spotlit_ * this->fs_ref_.row(xr);
    //
    index_type xr = (i < n_/2) ? (n_/2 + i) : (i - n_/2);
    fsm_.row(xr)(ldom) = fs_spotlit_row_(rdom) * this->fs_ref_.row(xr)(ldom);
    fsm_.row(xr)(rdom) = fs_spotlit_row_(ldom) * this->fs_ref_.row(xr)(rdom);
  }

  SAVE_VIEW("p77_fsm_row.view", fsm_, this->swap_bytes_);
}


#else

template <typename T>
void
Kernel1<T>::digital_spotlighting(complex_matrix_type s_raw)
{
  using vsip_csl::profile::Scope;
  using vsip_csl::profile::user;
  Scope<user> scope("digital_spotlighting", digital_spotlighting_ops_);

  assert(s_raw.size(0) == n_);
  assert(s_raw.size(1) == mc_);

  // The baseband reference signal is first transformed into the Doppler 
  // (spatial frequency) domain.  

  {
    Scope<user> scope("corner-turn-1", n_ * mc_ * sizeof(complex<float>));
    s_filt_ = s_raw;
  }

  // 59. (n by mc array of complex numbers) filtered echoed signal
  //
  // Note that the fast-time filter is combined with the compression
  // along the slow-time axis below.  
  //
  // 62. (n by mc array of complex numbers) signal compressed along 
  //     slow-time (note that to view 'sCompr' it will need to be 
  //     fftshifted first.)
  {
    Scope<user> scope("ft-half-fc",  fast_time_filter_ops_);
    s_filt_ = s_compr_filt_shift_ * ft_fftm_(s_filt_);
  }
  SAVE_VIEW("p62_s_filt.view", s_filt_, this->swap_bytes_);
  {
    Scope<user> scope("corner-turn-2", n_ * mc_ * sizeof(complex<float>));
    fs_ = s_filt_;
  }

  // 63. (n by mc array of complex numbers) narrow-bandwidth polar format
  //     reconstruction along slow-time
  compr_fftm_(fs_);

  // 64. (scalar, int) number of zeros to be padded into the ku domain 
  //     for slow-time upsampling
  length_type mz = m_ - mc_;

  // 65. (n by m array of complex numbers) zero pad the spatial frequency 
  //     domain's compressed signal along its slow-time (note that to view 
  //     'fsPadded' it will need to be fftshifted first)
  Domain<2> left(Domain<1>(0, 1, n_), Domain<1>(0, 1, mc_/2));
  Domain<2> center_dst(Domain<1>(0, 1, n_), Domain<1>(mc_/2, 1, mz));
  Domain<2> right_dst(Domain<1>(0, 1, n_), Domain<1>(mz + mc_/2, 1, mc_/2));
  Domain<2> right_src(Domain<1>(0, 1, n_), Domain<1>(mc_/2, 1, mc_/2));

  {
    Scope<user> scope("expand", n_ * m_ * sizeof(complex<float>));
    fs_padded_(left) = fs_(left);
    fs_padded_(center_dst) = complex<T>();
    fs_padded_(right_dst) = fs_(right_src);
  }

  // 66. (n by m array of complex numbers) transform-back the zero 
  // padded spatial spectrum along its cross-range
  //
  // 68. (n by m array of complex numbers) slow-time decompression (note 
  //     that to view 'sDecompr' it will need to be fftshifted first.)
  {
    Scope<user> scope("decompr-half-fc",  slow_time_decompression_ops_);
    fs_padded_ = s_decompr_filt_shift_ * decompr_fftm_(fs_padded_);
  }

  // 69. (n by m array of complex numbers) digitally-spotlighted SAR 
  //     signal spectrum
  //
  // match filter the spotlighted signal 'fsSpotLit' with the reference's 
  // complex conjugate 'fsRef' along fast-time and slow-time, to remove 
  // the reference signal's spectral components.
  //
  // 77. (n by m array of complex nums) Doppler domain matched-filtered signal

  {
    Scope<user> scope("st-half-fc",  slow_time_decompression_ops_);
    fsm_ = this->fs_ref_preshift_ * st_fftm_(fs_padded_); // row
  }

  SAVE_VIEW("p77_fsm_half_fc.view", fsm_, this->swap_bytes_);
}
#endif // DIGITAL_SPOTLIGHT_BY_ROW



template <typename T>
void // Matrix<T>
Kernel1<T>::interpolation(real_matrix_type image)
{
  using vsip_csl::profile::Scope;
  using vsip_csl::profile::user;
  Scope<user> scope("interpolation", interpolation_ops_);

  assert(image.size(0) == m_);
  assert(image.size(1) == this->nx_);

  // Interpolate From Polar Coordinates to Rectangular Coordinates

  // corner-turn to col-major
  {
    Scope<user> scope("corner-turn-3", n_ * m_ * sizeof(complex<float>));
    fsm_t_ = fsm_;
  }


  // 86b. begin the range loop
  {
    Scope<user> scope("range loop", range_loop_ops_);
#if USE_CELL_UKERNEL
    // (86a. initialize the F(kx,ku) array) - ukernel does this
    Interp_proxy obj;
    vsip_csl::ukernel::Ukernel<Interp_proxy> uk(obj);
    uk(
      icKX_.transpose(), 
      SINC_HAM_.template transpose<1, 0, 2>(), 
      fsm_t_.transpose(), 
      F_.transpose());

#elif VSIP_IMPL_HAVE_CUDA
    udk::Task<udk::target::cuda, 
      udk::tuple<udk::in<Dense<2, uint32_t, col2_type> >,
      udk::in<Dense<3, T, tuple<1, 0, 2> > >,
      udk::in<complex_col_block_type>,
      udk::out<complex_col_block_type> > > 
    task(interpolate_with_shift<Dense<2, uint32_t, col2_type>,
	 Dense<3, T, tuple<1, 0, 2> >,
	 complex_col_block_type,
	 complex_col_block_type>);
    task.execute(icKX_, SINC_HAM_, fsm_t_, F_);
#else
    // (86a. initialize the F(kx,ku) array)
    {
      Scope<user> scope("zero", this->nx_ * m_ * sizeof(complex<float>));
      F_ = complex<T>(0);
    }

    for (index_type j = 0; j < m_; ++j)
    {
      for (index_type i = 0; i < n_; ++i)
      {
	// 88. (I by m array of ints) ikx are the indices of the slice that 
	//     include the cross-range sliver at its center
	index_type ikxrows = icKX_.get(i, j);
#if DIGITAL_SPOTLIGHT_BY_ROW
	index_type i_shift = i;
#else
	index_type i_shift = (i + n_/2) % n_;
#endif
	
	for (index_type h = 0; h < this->I_; ++h)
	{
	  // sinc convolution interpolation of the signal's Doppler 
	  // spectrum, from polar to rectangular coordinates 
	  
	  // 92. (nx by m array of complex nums) F is the rectangular signal 
	  //     spectrum
	  F_.put(ikxrows + h, j, F_.get(ikxrows + h, j) + 
		 (fsm_t_.get(i_shift, j) * SINC_HAM_.get(i, j, h)));
	}
      }
      F_.col(j)(Domain<1>(j%2, 2, this->nx_/2)) *= T(-1);
    } // 93. end the range loop
#endif
  }

  SAVE_VIEW("p92_F.view", F_, this->swap_bytes_);


  // transform from the Doppler domain image into a spatial domain image

  // 94. (nx by m array of complex nums) spatial image (complex pixel 
  //     intensities) 
  {
    Scope<user> scope("doppler to spatial transform", interp_fftm_ops_);
    ifftmc_(F_);	// col
    spatial_ = F_;	// row := col corner-turn-
    ifftmr_(spatial_);	// row

    // The final freq-domain fftshift can be skipped because mag() throws
    // away sign:
    // fd_fftshift(spatial_, spatial_);
  }

  {
    Scope<user> scope("corner-turn-4", m_ * this->nx_ *sizeof(complex<float>));
    F_ = spatial_;
  }

  // for viewing, transpose spatial's magnitude 
  // 95. (m by nx array of reals) image (pixel intensities)
  {
    Scope<user> scope("image-prep", magnitude_ops_);
    image = mag(F_.transpose());
  }
}
