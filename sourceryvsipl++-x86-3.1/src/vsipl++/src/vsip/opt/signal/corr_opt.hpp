/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/signal/corr-opt.hpp
    @author  Jules Bergmann
    @date    2005-10-05
    @brief   VSIPL++ Library: Correlation class implementation using 
			      FFT overlap and add algorithm.
*/

#ifndef VSIP_OPT_SIGNAL_CORR_OPT_HPP
#define VSIP_OPT_SIGNAL_CORR_OPT_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <algorithm>

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/selgen.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/signal/conv_common.hpp>
#include <vsip/core/signal/corr_common.hpp>
#include <vsip/opt/dispatch.hpp>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{

/// Compute next power of 2 after len.

inline length_type
next_power_of_2(length_type len)
{
  length_type pow2 = 1;
  while (pow2 < len)
    pow2 *= 2;
  return pow2;
}



// Choose FFT size for overlap and add.

inline length_type
choose_fft_size(length_type M, length_type N)
{
  if (M+N < 1024)
    return next_power_of_2(M+N);
  if (4*next_power_of_2(M) > 1024)
    return 4*next_power_of_2(M);
  return 1024;
}

template <dimension_type      D,
	  support_region_type Supp,
	  typename            T,
	  unsigned            n_times,
          alg_hint_type       a_hint>
class Correlation_fft
{
  typedef typename complex_of<T>::type storage_format;

  static dimension_type const dim = D;

  // Compile-time constants.
public:
  static support_region_type const supprt  = Supp;

  // Constructors, copies, assignments, and destructors.
public:
  Correlation_fft(
    Domain<dim> const&   ref_size,
    Domain<dim> const&   input_size)
    VSIP_THROW((std::bad_alloc));

  Correlation_fft(Correlation_fft const&) VSIP_NOTHROW;
  Correlation_fft& operator=(Correlation_fft const&) VSIP_NOTHROW;
  ~Correlation_fft() VSIP_NOTHROW {}

  // Accessors.
public:
  Domain<dim> const& reference_size() const VSIP_NOTHROW  { return ref_size_; }
  Domain<dim> const& input_size() const VSIP_NOTHROW   { return input_size_; }
  Domain<dim> const& output_size() const VSIP_NOTHROW  { return output_size_; }

  float impl_performance(char* what) const
  {
    if (!strcmp(what, "in_ext_cost")) return pm_in_ext_cost_;
    else if (!strcmp(what, "out_ext_cost")) return pm_out_ext_cost_;
    else if (!strcmp(what, "non-opt-calls")) return pm_non_opt_calls_;
    else return 0.f;
  }

  // Implementation functions.
public:
  template <typename Block0,
	    typename Block1,
	    typename Block2>
  void
  impl_correlate(bias_type               bias,
	    const_Vector<T, Block0> ref,
	    const_Vector<T, Block1> in,
	    Vector<T, Block2>       out)
    VSIP_NOTHROW;

  template <typename Block0,
	    typename Block1,
	    typename Block2>
  void
  impl_correlate(bias_type               bias,
	    const_Matrix<T, Block0> ref,
	    const_Matrix<T, Block1> in,
	    Matrix<T, Block2>       out)
    VSIP_NOTHROW;

  typedef Layout<1, row1_type, unit_stride, interleaved_complex> layout_type;
  typedef Vector<T> coeff_view_type;

  // Member data.
private:
  Domain<dim>     ref_size_;
  Domain<dim>     input_size_;
  Domain<dim>     output_size_;

  length_type	  n_fft_;	// length of fft to overlap
  length_type	  N2_;		// legnth of zero-pad in overlap
  length_type	  N1_;		// length of real-data in overlap

  Fft<const_Vector, T, storage_format,
      is_same<T, storage_format>::value ? fft_fwd : 0, by_reference>
		f_fft_;

  Fft<const_Vector, storage_format, T,
      is_same<T, storage_format>::value ? fft_inv : 0, by_reference>
		i_fft_;

  length_type	  fft_fd_size_;

  Vector<T>            t_in_;	// temporary input - time domain
  Vector<T>            t_ref_;	// temporary ref   - time domain
  Vector<storage_format> f_in_;	// temporary input - freq domain
  Vector<storage_format> f_ref_;	// temporary ref   - freq domain

  int             pm_non_opt_calls_;
  size_t          pm_ref_ext_cost_;
  size_t          pm_in_ext_cost_;
  size_t          pm_out_ext_cost_;
};



/***********************************************************************
  Definitions
***********************************************************************/

/// Construct a correlation object.

template <dimension_type      D,
	  support_region_type Supp,
	  typename            T,
	  unsigned            n_times,
          alg_hint_type       a_hint>
Correlation_fft<D, Supp, T, n_times, a_hint>::Correlation_fft(
  Domain<dim> const&   ref_size,
  Domain<dim> const&   input_size)
VSIP_THROW((std::bad_alloc))
  : ref_size_   (normalize(ref_size)),
    input_size_ (normalize(input_size)),
    output_size_(conv_output_size(Supp, ref_size_, input_size_, 1)),

    n_fft_      (choose_fft_size(ref_size_.size(), input_size_.size())),
    N2_         (ref_size_.size()),
    N1_	        (n_fft_-N2_),

    f_fft_      (n_fft_, 1.0),
    i_fft_      (n_fft_, 1.0/n_fft_),
    fft_fd_size_(f_fft_.output_size().size()),
    t_in_       (n_fft_),
    t_ref_      (n_fft_),
    f_in_       (fft_fd_size_),
    f_ref_      (fft_fd_size_),

    pm_non_opt_calls_ (0)
{
}

// Perform 1-D correlation.

template <dimension_type      D,
	  support_region_type Supp,
	  typename            T,
	  unsigned            n_times,
          alg_hint_type       a_hint>
template <typename Block0,
	  typename Block1,
	  typename Block2>
void
Correlation_fft<D, Supp, T, n_times, a_hint>::impl_correlate(
  bias_type               bias,
  const_Vector<T, Block0> ref,
  const_Vector<T, Block1> in,
  Vector<T, Block2>       out)
VSIP_NOTHROW
{
  length_type const M = this->ref_size_[0].size();
  length_type const N = this->input_size_[0].size();
  length_type const P = this->output_size_[0].size();

  assert(M == ref.size());
  assert(N == in.size());
  assert(P == out.size());


  // Transform the reference
  t_ref_(Domain<1>(0, 1, M))      = ref;
  t_ref_(Domain<1>(M, 1, n_fft_-M)) = T();
  f_fft_(t_ref_, f_ref_);


  // Determine how much to "shift" the output vector.
  length_type shift;

  if (Supp == support_full)
    shift = N2_-(M-1);
  else if (Supp == support_same)
    shift = N2_-(M/2);
  else
    shift = N2_-0;

  
  // Perform correlation using overlap and add

  for (index_type i=0; i*N1_ < N; ++i)
  {
    // Copy input
    if (N1_ > N - i*N1_)
    {
      length_type n_copy = N - i*N1_;
      t_in_(Domain<1>(0,  1, N2_))     = T();
      t_in_(Domain<1>(N2_, 1, n_copy)) = in(Domain<1>(i*N1_, 1, n_copy));
      t_in_(Domain<1>(N2_+n_copy, 1, N1_-n_copy)) = T();
    }
    else
    {
      t_in_(Domain<1>(0,  1, N2_))  = T();
      t_in_(Domain<1>(N2_, 1, N1_)) = in(Domain<1>(i*N1_, 1, N1_));
    }

    // Perform correlation
    f_fft_(t_in_,  f_in_);

    f_in_ = f_in_ * impl_conj(f_ref_);
    i_fft_(f_in_, t_in_);
    t_in_ = impl_conj(t_in_);

    // Copy output (with overlap-add)
    if (i == 0)
    {
      length_type len = std::min(N1_+N2_-shift, P);
      out(Domain<1>(0, 1, len)) = t_in_(Domain<1>(shift, 1, len));
    }
    else if (i*N1_+N1_+N2_-shift < P)
    {
      out(Domain<1>(i*N1_-shift,     1, N2_)) += t_in_(Domain<1>(0,   1, N2_));
      out(Domain<1>(i*N1_+N2_-shift, 1, N1_))  = t_in_(Domain<1>(N2_, 1, N1_));
    }
    else
    {
      length_type len1 = std::min(P - (i*N1_-shift), N2_);
      if (len1 > 0)
	out(Domain<1>(i*N1_-shift, 1, len1)) += t_in_(Domain<1>(0, 1, len1));

      length_type len2 = std::min(P - (i*N1_+N2_-shift), N1_);
      if (len2 > 0)
	out(Domain<1>(i*N1_+N2_-shift, 1, len2)) =
	  t_in_(Domain<1>(N2_, 1, len2));
    }
  }


  // Unbias the result (if requested).

  if (bias == unbiased)
  {
    if (Supp == support_full)
    {
      if (M > 1)
      {
	out(Domain<1>(0, 1, M-1))     /= ramp(T(1), T(1), M-1);
	out(Domain<1>(P-M+1, 1, M-1)) /= ramp(T(M-1), T(-1), M-1);
      }
      if (P+2 > 2*M)
	out(Domain<1>(M-1, 1, P-2*M+2)) /= T(M);
    }
    else if (Supp == support_same)
    {
      length_type edge  = M - (M/2);

      if (edge > 0)
      {
	out(Domain<1>(0, 1, edge))      /= ramp(T(M/2 + (M%2)), T(1), edge);
#if VSIP_IMPL_CORR_CORRECT_SAME_SUPPORT_SCALING
	out(Domain<1>(P-edge, 1, edge)) /= ramp(T(M), T(-1), edge);
#else
	out(Domain<1>(P-edge, 1, edge)) /= ramp(T(M-(1-M%2)), T(-1), edge);
#endif
      }
      if (P > 2*edge)
	out(Domain<1>(edge, 1, P - 2*edge)) /= T(M);
    }
    else // (Supp == support_min)
    {
      out /= T(M);
    }
  }
}



// Perform 2-D correlation.

template <dimension_type      D,
	  support_region_type Supp,
	  typename            T,
	  unsigned            n_times,
          alg_hint_type       a_hint>
template <typename Block0,
	  typename Block1,
	  typename Block2>
void
Correlation_fft<D, Supp, T, n_times, a_hint>::impl_correlate(
  bias_type               bias,
  const_Matrix<T, Block0> ref,
  const_Matrix<T, Block1> in,
  Matrix<T, Block2>       out)
VSIP_NOTHROW
{
  VSIP_IMPL_THROW(vsip::impl::unimplemented(
    "Correlation_fft: 2D correlation not implemented."));
}

} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <support_region_type R,
          typename            T,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<1, R, T, N, H>, be::opt>
{
  static bool const ct_valid = true;
  typedef vsip::impl::Correlation_fft<1, R, T, N, H> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_IMPL_SIGNAL_CORR_OPT_HPP
