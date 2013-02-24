/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/cml/conv.hpp
    @author  Mike LeBlanc
    @date    2008-05-14
    @brief   VSIPL++ Library: Convolution class implementation using CML.
*/

#ifndef VSIP_OPT_CBE_CML_CONV_HPP
#define VSIP_OPT_CBE_CML_CONV_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/signal/conv_common.hpp>
#include <vsip/opt/dispatch.hpp>
#include <cml.h>

namespace vsip
{
namespace impl
{
namespace cml
{

// Convolution

// Wrappers for CML functions.
inline
void
conv(float const* coeff, length_type c_size,
     float const* in,    length_type /*i_size*/, stride_type s_in,
     float*       out,   length_type o_size, stride_type s_out,
     length_type decimation)
{
  cml_conv1d_min_f(coeff,1,in,s_in,out,s_out,decimation,c_size,o_size);
}

inline
void
conv(std::complex<float> const* coeff, length_type c_size,
     std::complex<float> const* in,    length_type /*i_size*/, stride_type s_in,
     std::complex<float>*       out,   length_type o_size, stride_type s_out,
     length_type decimation)
{
  float const* fcoeff = reinterpret_cast<float const*>(coeff);
  float const* fin    = reinterpret_cast<float const*>(in);
  float*       fout   = reinterpret_cast<float*>(out);
  cml_cconv1d_min_f(fcoeff,1,fin,s_in,fout,s_out,decimation,c_size,o_size);
}

inline
void
conv(std::pair<float const *,float const *> coeff, length_type c_size,
     std::pair<float const *,float const *> in, length_type /*i_size*/, stride_type s_in,
     std::pair<float*,float*> out, length_type o_size, stride_type s_out,
     length_type decimation)
{
  cml_zconv1d_min_f(coeff.first,coeff.second,1,
		    in.first,in.second,s_in,
		    out.first,out.second,s_out,
		    decimation,c_size,o_size);
}

inline
void
conv(float const* coeff, stride_type s_coeff,
     float const* in,    stride_type s_in,
     float*       out,   stride_type s_out,
     length_type nkr,
     length_type nkc,
     length_type nr,
     length_type nc,
     length_type decimation)
{
  cml_conv2d_f(coeff,s_coeff,
	       in,s_in,
	       out,s_out,
	       decimation, decimation,
	       nkr, nkc,
	       nr, nc);
}

inline
void
conv(std::complex<float> const* coeff, stride_type s_coeff,
     std::complex<float> const* in,    stride_type s_in,
     std::complex<float>*       out,   stride_type s_out,
     length_type nkr,
     length_type nkc,
     length_type nr,
     length_type nc,
     length_type decimation)
{
  float const* fcoeff = reinterpret_cast<float const*>(coeff);
  float const* fin    = reinterpret_cast<float const*>(in);
  float*       fout   = reinterpret_cast<float*>(out);
  cml_cconv2d_f(fcoeff,s_coeff,
		fin,s_in,
		fout,s_out,
		decimation, decimation,
		nkr, nkc,
		nr, nc);
}

inline
void
conv(std::pair<float const *,float const *> coeff, stride_type s_coeff,
     std::pair<float const *,float const *> in,    stride_type s_in,
     std::pair<float*,float*> out,   stride_type s_out,
     length_type nkr,
     length_type nkc,
     length_type nr,
     length_type nc,
     length_type decimation)
{
  cml_zconv2d_f(coeff.first,coeff.second, s_coeff,
		in.first,   in.second,    s_in,
		out.first,  out.second,   s_out,
		decimation, decimation,
		nkr, nkc,
		nr, nc);
}

// Wrappers for "same" support.
//
template <typename T>
inline
void
conv_same(T const * coeff,
	  length_type nkr, length_type nkc,
	  stride_type s_coeff,
	  stride_type s_coeff_col,
	  T const * in,
	  length_type mr, length_type mc,
	  stride_type s_in,
	  stride_type s_in_col,
	  T* out,
	  length_type nr, length_type nc,
	  stride_type s_out,
	  stride_type s_out_col,
	  length_type decimation)
{
  // CML conv_same only supports unit-stride
  assert(s_coeff_col == 1 && s_in_col == 1 && s_out_col == 1);

  // Determine the first element computed by conv_min.
  index_type n0_r  = ( (nkr - 1) - (nkr/2) ) / decimation;
  index_type n0_c  = ( (nkc - 1) - (nkc/2) ) / decimation;
  index_type res_r = ( (nkr - 1) - (nkr/2) ) % decimation;
  index_type res_c = ( (nkc - 1) - (nkc/2) ) % decimation;
  if (res_r > 0) n0_r += 1;
  if (res_c > 0) n0_c += 1;

  // Determine the phase of the input given to conv_min.
  index_type phase_r = (res_r == 0) ? 0 : (decimation - res_r);
  index_type phase_c = (res_c == 0) ? 0 : (decimation - res_c);

  // Determine the last element + 1 computed by conv_min.
  index_type n1_r = (mr - (nkr/2)) / decimation;
  index_type n1_c = (mc - (nkc/2)) / decimation;
  if ((mr - (nkr/2)) % decimation > 0) n1_r++;
  if ((mc - (nkc/2)) % decimation > 0) n1_c++;

  T* out_adj = out + (n0_r)*s_out + (n0_c);
  T const *  in_adj = in  + (phase_r)*s_in + (phase_c);

  if (n1_r > n0_r && n1_c > n0_c)
    conv(coeff,   s_coeff,
	 in_adj,  s_in,
	 out_adj, s_out,
	 nkr, nkc,
	 n1_r-n0_r, n1_c-n0_c,
	 decimation);

  conv_same_edge(coeff, nkr, nkc, s_coeff, 1,
		 in,    mr,  mc,  s_in,    1,
		 out,   nr,  nc,  s_out,   1,
		 decimation);
}

template <typename T>
inline
void
conv_same(std::pair<T const *,T const *> coeff,
	  length_type nkr, length_type nkc,
	  stride_type s_coeff,
	  stride_type s_coeff_col,
	  std::pair<T const *,T const *> in,
	  length_type mr, length_type mc,
	  stride_type s_in,
	  stride_type s_in_col,
	  std::pair<T*,T*> out,
	  length_type nr, length_type nc,
	  stride_type s_out,
	  stride_type s_out_col,
	  length_type decimation)
{
  // Determine the first element computed by conv_min.
  index_type n0_r  = ( (nkr - 1) - (nkr/2) ) / decimation;
  index_type n0_c  = ( (nkc - 1) - (nkc/2) ) / decimation;
  index_type res_r = ( (nkr - 1) - (nkr/2) ) % decimation;
  index_type res_c = ( (nkc - 1) - (nkc/2) ) % decimation;
  if (res_r > 0) n0_r += 1;
  if (res_c > 0) n0_c += 1;

  // Determine the phase of the input given to conv_min.
  index_type phase_r = (res_r == 0) ? 0 : (decimation - res_r);
  index_type phase_c = (res_c == 0) ? 0 : (decimation - res_c);

  // Determine the last element + 1 computed by conv_min.
  index_type n1_r = (mr - (nkr/2)) / decimation;
  index_type n1_c = (mc - (nkc/2)) / decimation;
  if ((mr - (nkr/2)) % decimation > 0) n1_r++;
  if ((mc - (nkc/2)) % decimation > 0) n1_c++;

  if (n1_r > n0_r && n1_c > n0_c)
  {
    T* out_adj_re = out.first + (n0_r)*s_out + (n0_c);
    T* out_adj_im = out.second + (n0_r)*s_out + (n0_c);
    T const * in_adj_re = in.first + (phase_r)*s_in + (phase_c);
    T const * in_adj_im = in.second + (phase_r)*s_in + (phase_c);

    conv(coeff,   s_coeff,
	 std::pair<T const *,T const *>(in_adj_re,in_adj_im),  s_in,
	 std::pair<T*,T*>(out_adj_re,out_adj_im), s_out,
	 nkr, nkc,
	 n1_r-n0_r, n1_c-n0_c,
	 decimation);
  }

  conv_same_edge(coeff, nkr, nkc, s_coeff, 1,
		 in,    mr,  mc,  s_in,    1,
		 out,   nr,  nc,  s_out,   1,
		 decimation);
}


// End of Wrappers for "same" support.

template <template <typename, typename> class V,
	  symmetry_type       S,
	  support_region_type R,
	  typename            T,
	  unsigned            N,
          alg_hint_type       H>
class Convolution
{
  static dimension_type const dim = Dim_of_view<V>::dim;

  static storage_format_type const storage_format = dense_complex_format;
  typedef Storage<storage_format, T> storage_type;
  typedef typename storage_type::const_type ptr_type;

  // Compile-time constants.
public:
  static symmetry_type const       symmtry = S;
  static support_region_type const supprt  = R;

  // Constructors, copies, assignments, and destructors.
public:
  template <typename Block>
  Convolution(V<T, Block> filter_coeffs,
              Domain<dim> const& input_size,
              length_type decimation)
    VSIP_THROW((std::bad_alloc));

  Convolution(Convolution const&) VSIP_NOTHROW;
  Convolution& operator=(Convolution const&) VSIP_NOTHROW;
  ~Convolution() VSIP_NOTHROW;

  // Accessors.
public:
  Domain<dim> const& kernel_size() const VSIP_NOTHROW  { return kernel_size_; }
  Domain<dim> const& filter_order() const VSIP_NOTHROW { return kernel_size_; }
  Domain<dim> const& input_size() const VSIP_NOTHROW   { return input_size_; }
  Domain<dim> const& output_size() const VSIP_NOTHROW  { return output_size_; }
  symmetry_type symmetry() const VSIP_NOTHROW          { return symmtry; }
  support_region_type support() const VSIP_NOTHROW     { return supprt; }
  length_type decimation() const VSIP_NOTHROW          { return decimation_; }

  float impl_performance(char* what) const
  {
    if (!strcmp(what, "in_dda_cost")) return pm_in_dda_cost_;
    else if (!strcmp(what, "out_dda_cost")) return pm_out_dda_cost_;
    else if (!strcmp(what, "non-opt-calls")) return pm_non_opt_calls_;
    else return 0.f;
  }


  // Implementation functions.
protected:
  template <typename Block0,
	    typename Block1>
  void
  convolve(const_Vector<T, Block0>,
	   Vector<T, Block1>)
    VSIP_NOTHROW;

  template <typename Block0,
	    typename Block1>
  void
  convolve(const_Matrix<T, Block0>,
	   Matrix<T, Block1>)
    VSIP_NOTHROW;

  typedef Layout<dim,
                 typename Row_major<dim>::type,
		 unit_stride,
		 storage_format>
    layout_type;
  typedef typename view_of<Dense<dim, T> >::type coeff_view_type;
  typedef dda::Data<typename coeff_view_type::block_type, dda::in, layout_type>
    c_data_type;

public:

  coeff_view_type kernel() const { return coeff_;}

  // Member data.
private:
  coeff_view_type coeff_;
  c_data_type     coeff_data_;
  ptr_type  pcoeff_;

  Domain<dim>     kernel_size_;
  Domain<dim>     input_size_;
  Domain<dim>     output_size_;
  aligned_array<T>        in_buffer_;
  aligned_array<T>        out_buffer_;
  length_type     decimation_;

  int             pm_non_opt_calls_;
  size_t          pm_in_dda_cost_;
  size_t          pm_out_dda_cost_;
};


/// Construct a convolution object.

template <template <typename, typename> class ConstViewT,
	  symmetry_type                       Symm,
	  support_region_type                 Supp,
	  typename                            T,
	  unsigned                            n_times,
          alg_hint_type                       a_hint>
template <typename Block>
Convolution<ConstViewT, Symm, Supp, T, n_times, a_hint>::Convolution(
  ConstViewT<T, Block> filter_coeffs,
  Domain<dim> const&   input_size,
  length_type          decimation)
VSIP_THROW((std::bad_alloc))
  : coeff_      (conv_kernel<coeff_view_type>(Symm, filter_coeffs)),
    coeff_data_ (coeff_.block()),
    pcoeff_     (coeff_data_.ptr()),
    kernel_size_(impl::view_domain(coeff_)),
    input_size_ (input_size),
    output_size_(impl::conv_output_size(Supp, kernel_size_, input_size,
					decimation)),
    in_buffer_  (input_size_.size()),
    out_buffer_ (output_size_.size()),
    decimation_ (decimation),
    pm_non_opt_calls_ (0)
{
}

template <template <typename, typename> class ConstViewT,
	  symmetry_type                       Symm,
	  support_region_type                 Supp,
	  typename                            T,
	  unsigned                            n_times,
          alg_hint_type                       a_hint>
Convolution<ConstViewT, Symm, Supp, T, n_times, a_hint>::~Convolution()
  VSIP_NOTHROW
{
}



// Perform 1-D convolution.

template <template <typename, typename> class ConstViewT,
	  symmetry_type       Symm,
	  support_region_type Supp,
	  typename            T,
	  unsigned            n_times,
          alg_hint_type       a_hint>
template <typename Block0,
	  typename Block1>
void
Convolution<ConstViewT, Symm, Supp, T, n_times, a_hint>::convolve(
  const_Vector<T, Block0> in,
  Vector<T, Block1>       out)
VSIP_NOTHROW
{
  length_type const M = this->coeff_.size(0);
  length_type const N = this->input_size_[0].size();
  length_type const P = this->output_size_[0].size();

  assert(P == out.size());

  typedef vsip::dda::Data<Block0, dda::in> in_data_type;
  typedef vsip::dda::Data<Block1, dda::out> out_data_type;

  static storage_format_type const storage_format = get_block_layout<Block0>::storage_format;

  in_data_type in_data(in.block(),  array_cast<storage_format>(in_buffer_));
  out_data_type out_data(out.block(), array_cast<storage_format>(out_buffer_));

  VSIP_IMPL_PROFILE(pm_in_dda_cost_  += in_data.cost());
  VSIP_IMPL_PROFILE(pm_out_dda_cost_ += out_data.cost());

  stride_type s_in  = in_data.stride(0);
  stride_type s_out = out_data.stride(0);

  if (Supp == support_full)
  {
    VSIP_IMPL_PROFILE(pm_non_opt_calls_++);
    conv_full(pcoeff_, M, in_data.ptr(), N, s_in, out_data.ptr(), P, s_out, decimation_);
  }
  else if (Supp == support_same)
  {
    VSIP_IMPL_PROFILE(pm_non_opt_calls_++);
    vsip::impl::conv_same(pcoeff_, M, in_data.ptr(), N, s_in, out_data.ptr(), P, s_out, decimation_);
  }
  else // (Supp == support_min)
  {
    conv(pcoeff_, M, in_data.ptr(), N, s_in, out_data.ptr(), P, s_out, decimation_);
  }
}

// Perform 2-D convolution.

template <template <typename, typename> class ConstViewT,
	  symmetry_type                       Symm,
	  support_region_type                 Supp,
	  typename                            T,
	  unsigned                            n_times,
          alg_hint_type                       a_hint>
template <typename Block0,
	  typename Block1>
void
Convolution<ConstViewT, Symm, Supp, T, n_times, a_hint>::convolve(
  const_Matrix<T, Block0> in,
  Matrix<T, Block1>       out)
VSIP_NOTHROW
{
  length_type const Mr = this->coeff_.size(0);
  length_type const Mc = this->coeff_.size(1);

  length_type const Nr = this->input_size_[0].size();
  length_type const Nc = this->input_size_[1].size();

  length_type const Pr = this->output_size_[0].size();
  length_type const Pc = this->output_size_[1].size();

  assert(Pr == out.size(0) && Pc == out.size(1));

  typedef vsip::dda::Data<Block0, dda::in> in_data_type;
  typedef vsip::dda::Data<Block1, dda::out> out_data_type;

  in_data_type in_data(in.block(), array_cast<storage_format>( in_buffer_));
  out_data_type out_data(out.block(), array_cast<storage_format>(out_buffer_));

  VSIP_IMPL_PROFILE(pm_in_dda_cost_ += in_data.cost());
  VSIP_IMPL_PROFILE(pm_out_dda_cost_ += out_data.cost());

  stride_type coeff_row_stride = coeff_data_.stride(0);
  stride_type coeff_col_stride = coeff_data_.stride(1);
  stride_type in_row_stride    = in_data.stride(0);
  stride_type in_col_stride    = in_data.stride(1);
  stride_type out_row_stride   = out_data.stride(0);
  stride_type out_col_stride   = out_data.stride(1);

  if (
    coeff_col_stride == 1
      &&
    in_col_stride == 1
      &&
    out_col_stride == 1
    )
  {
    switch(Supp)
    {
      case support_min:
	vsip::impl::cml::conv(pcoeff_, coeff_row_stride,
			      in_data.ptr(), in_row_stride,
			      out_data.ptr(), out_row_stride,
			      Mr, Mc,
			      Pr, Pc,
			      decimation_);
	return;
      case support_same:
	vsip::impl::cml::conv_same(pcoeff_, Mr, Mc, coeff_row_stride, 1,
				   in_data.ptr(), Nr, Nc, in_row_stride, 1,
				   out_data.ptr(), Pr, Pc, out_row_stride, 1,
				   decimation_);
	return;
    }
  }
  VSIP_IMPL_PROFILE(pm_non_opt_calls_++);
  if (Supp == support_min)
    vsip::impl::conv_min(pcoeff_, Mr, Mc, coeff_row_stride, coeff_col_stride,
			 in_data.ptr(), Nr, Nc, in_row_stride, in_col_stride,
			 out_data.ptr(), Pr, Pc, out_row_stride, out_col_stride,
			 decimation_);
  else
    vsip::impl::conv_same(pcoeff_, Mr, Mc, coeff_row_stride, coeff_col_stride,
			  in_data.ptr(), Nr, Nc, in_row_stride, in_col_stride,
			  out_data.ptr(), Pr, Pc, out_row_stride, out_col_stride,
			  decimation_);
}

} // namespace vsip::impl::cml
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<1, S, R, float, N, H>, be::cml>
{
  static bool const ct_valid = R==support_min;
  typedef impl::cml::Convolution<const_Vector, S, R, float, N, H> backend_type;
};

template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<1, S, R, complex<float>, N, H>, be::cml>
{
  static bool const ct_valid = R==support_min;
  typedef impl::cml::Convolution<const_Vector, S, R, complex<float>, N, H> backend_type;
};

template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<2, S, R, float, N, H>, be::cml>
{
  static bool const ct_valid = R==support_min || R==support_same;
  typedef impl::cml::Convolution<const_Matrix, S, R, float, N, H> backend_type;
};

template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<2, S, R, complex<float>, N, H>, be::cml>
{
  static bool const ct_valid = R==support_min || R==support_same;
  typedef impl::cml::Convolution<const_Matrix, S, R, complex<float>, N, H> backend_type;
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
