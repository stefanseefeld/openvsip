/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/sal/conv.hpp
    @author  Don McCoy
    @date    2005-11-18
    @brief   VSIPL++ Library: Convolution class implementation using SAL.
*/

#ifndef VSIP_OPT_SAL_CONV_HPP
#define VSIP_OPT_SAL_CONV_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/signal/conv_common.hpp>
#include <vsip/core/extdata_dist.hpp>
#include <vsip/opt/sal/bindings.hpp>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace sal
{
template <template <typename, typename> class V,
	  symmetry_type       S,
	  support_region_type R,
	  typename            T,
	  unsigned            N,
          alg_hint_type       H>
class Convolution
{
  static dimension_type const dim = impl::Dim_of_view<V>::dim;

  typedef dense_complex_type complex_type;
  typedef Storage<complex_type, T> storage_type;
  typedef typename storage_type::type ptr_type;

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
    if (!strcmp(what, "in_ext_cost"))        return pm_in_ext_cost_;
    else if (!strcmp(what, "out_ext_cost"))  return pm_out_ext_cost_;
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

  typedef vsip::impl::Layout<dim,
			     typename Row_major<dim>::type,
			     vsip::impl::Stride_unit,
			     complex_type>
		layout_type;
  typedef typename View_of_dim<dim, T, Dense<dim, T> >::type coeff_view_type;
  typedef impl::Ext_data<typename coeff_view_type::block_type, layout_type>
		c_ext_type;

  // Member data.
private:
  coeff_view_type coeff_;
  c_ext_type      coeff_ext_;
  ptr_type        pcoeff_;

  coeff_view_type sal_coeff_;
  c_ext_type      sal_coeff_ext_;
  ptr_type        sal_pcoeff_;

  Domain<dim>     kernel_size_;
  Domain<dim>     input_size_;
  Domain<dim>     output_size_;
  ptr_type        in_buffer_;
  ptr_type        out_buffer_;
  length_type     decimation_;

  int             pm_non_opt_calls_;
  size_t          pm_in_ext_cost_;
  size_t          pm_out_ext_cost_;
};



/***********************************************************************
  Definitions
***********************************************************************/

// 080313: These kernel sizes represent cross-over points where
//   frequency domain convolution may be more efficient.  Currently
//   we ignore them because we don't use SAL's frequency domain
//   convolution, and SAL's time-domain convolution is faster than
//   a generic time-domain convolution.
template <typename T>
struct Max_kernel_length
{
  static length_type const value = 0;
};

template <>
struct Max_kernel_length<float>
{
  static length_type const value = 36;
};

template <>
struct Max_kernel_length<std::complex<float> >
{
  static length_type const value = 17;
};

template <typename T,
	  typename Block1,
	  typename Block2>
inline void
mirror(
  const_Vector<T, Block1> src,
  Vector<T, Block2>       dst)
{
  dst(Domain<1>(dst.size()-1, -1, dst.size())) = src;
}

template <typename T,
	  typename Block1,
	  typename Block2>
inline void
mirror(
  const_Matrix<T, Block1> src,
  Matrix<T, Block2>       dst)
{
  dst(Domain<2>(Domain<1>(dst.size(0)-1, -1, dst.size(0)),
		Domain<1>(dst.size(1)-1, -1, dst.size(1)))) = src;
}



/// Construct a convolution object.

template <template <typename, typename> class ConstViewT,
	  symmetry_type                       Symm,
	  support_region_type                 Supp,
	  typename                            T,
	  unsigned                            n_times,
          alg_hint_type                       a_hint>
template <typename Block>
Convolution<ConstViewT, Symm, Supp, T, n_times, a_hint>::
Convolution(
  ConstViewT<T, Block> filter_coeffs,
  Domain<dim> const&   input_size,
  length_type          decimation)
VSIP_THROW((std::bad_alloc))
  : coeff_      (conv_kernel<coeff_view_type>(Symm, filter_coeffs)),
    coeff_ext_  (coeff_.block(), impl::SYNC_IN),
    pcoeff_     (coeff_ext_.data()),
    sal_coeff_     (conv_kernel<coeff_view_type>(Symm, filter_coeffs)),
    sal_coeff_ext_ (sal_coeff_.block(), impl::SYNC_IN),
    sal_pcoeff_    (sal_coeff_ext_.data()),
    kernel_size_(impl::view_domain(coeff_)),
    input_size_ (input_size),
    output_size_(impl::conv_output_size(Supp, kernel_size_, input_size,
					decimation)),
    decimation_ (decimation),
    pm_non_opt_calls_ (0)
{
  mirror(coeff_, sal_coeff_);
  in_buffer_  = storage_type::allocate(input_size_.size());
  if (storage_type::is_null(in_buffer_))
    VSIP_IMPL_THROW(std::bad_alloc());

  out_buffer_ = storage_type::allocate(output_size_.size());
  if (storage_type::is_null(out_buffer_))
  {
    storage_type::deallocate(in_buffer_);
    VSIP_IMPL_THROW(std::bad_alloc());
  }
}



/// Destroy a SAL Convolution_impl object.

template <template <typename, typename> class ConstViewT,
	  symmetry_type                       Symm,
	  support_region_type                 Supp,
	  typename                            T,
	  unsigned                            n_times,
          alg_hint_type                       a_hint>
Convolution<ConstViewT, Symm, Supp, T, n_times, a_hint>::~Convolution()
  VSIP_NOTHROW
{
  storage_type::deallocate(out_buffer_);
  storage_type::deallocate(in_buffer_);
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
Convolution<ConstViewT, Symm, Supp, T, n_times, a_hint>::
convolve(
  const_Vector<T, Block0> in,
  Vector<T, Block1>       out)
VSIP_NOTHROW
{
  length_type const M = this->coeff_.size(0);
  length_type const N = this->input_size_[0].size();
  length_type const P = this->output_size_[0].size();

  assert(P == out.size());

  typedef typename Block_layout<Block0>::layout_type LP0;
  typedef typename Block_layout<Block1>::layout_type LP1;

  typedef Layout<1, Any_type, Any_type, complex_type> req_LP;

  typedef typename Adjust_layout<T, req_LP, LP0>::type use_LP0;
  typedef typename Adjust_layout<T, req_LP, LP1>::type use_LP1;

  typedef vsip::impl::Ext_data_dist<Block0, SYNC_IN,  use_LP0>  in_ext_type;
  typedef vsip::impl::Ext_data_dist<Block1, SYNC_OUT, use_LP1> out_ext_type;

  in_ext_type  in_ext (in.block(),  in_buffer_);
  out_ext_type out_ext(out.block(), out_buffer_);

  VSIP_IMPL_PROFILE(pm_in_ext_cost_  += in_ext.cost());
  VSIP_IMPL_PROFILE(pm_out_ext_cost_ += out_ext.cost());

  ptr_type pin    = in_ext.data();
  ptr_type pout   = out_ext.data();

  stride_type s_in  = in_ext.stride(0);
  stride_type s_out = out_ext.stride(0);
  stride_type s_coeff = coeff_.block().impl_stride(1, 0);

  assert( Max_kernel_length<T>::value != 0 );
  // See note above on Max_kernel_length defn.
  if ( /*(M <= Max_kernel_length<T>::value) &&*/ (decimation_ == 1) ) 
  {
    // SAL only does the minimum convolution
    if (Supp == support_full)
    {
      impl::sal::conv( pcoeff_, s_coeff, M, 
                       pin, s_in, N, 
                       storage_type::offset(pout, (M - 1) * s_out),
		       s_out );

      // fill in missing values
      for (index_type n = 0; n < M - 1; ++n )
      {
	T sum = T();
        for (index_type k = 0; k < M; ++k )
          if ( (n >= k) && (n - k < N) )
	    sum += storage_type::get(pcoeff_, k * s_coeff) *
	           storage_type::get(pin,     (n - k) * s_in);
	storage_type::put(pout, n * s_out, sum);
      }
      for (index_type n = N; n < N + M - 1; ++n )
      {
	T sum = T();
        for (index_type k = 0; k < M; ++k )
          if ( (n >= k) && (n - k < N) )
	    sum += storage_type::get(pcoeff_, k * s_coeff) *
	           storage_type::get(pin,     (n - k) * s_in);
	storage_type::put(pout, n * s_out, sum);
      }
    }
    else if (Supp == support_same)
    {
      impl::sal::conv( pcoeff_, s_coeff, M, 
                       pin, s_in, N - (M - M/2), 
                       storage_type::offset(pout, (M - M/2 - 1) * s_out), s_out );

      // fill in missing values
      for (index_type n = 0; n < M/2; ++n )
      {
	T sum = T();
        for (index_type k = 0; k < M; ++k )
          if ( (n + M/2 >= k) && (n + M/2 - k < N) )
	    sum += storage_type::get(pcoeff_, k * s_coeff) *
	           storage_type::get(pin,     (n + M/2 - k) * s_in);
	storage_type::put(pout, n * s_out, sum);
      }
      for (index_type n = N - (M - M/2); n < N; ++n )
      {
	T sum = T();
        for (index_type k = 0; k < M; ++k )
          if ( (n + M/2 >= k) && (n + M/2 - k < N) )
	    sum += storage_type::get(pcoeff_, k * s_coeff) *
	           storage_type::get(pin,     (n + M/2 - k) * s_in);
	storage_type::put(pout, n * s_out, sum);
      }
    }
    else // (Supp == support_min)
    {
      impl::sal::conv( pcoeff_, s_coeff, M, 
                       pin, s_in, N - (M - 1), 
                       pout, s_out );
    }
  }
  else // ( M > Max_kernel_length<T> )
  {
    if (Supp == support_full)
    {
      VSIP_IMPL_PROFILE(pm_non_opt_calls_++);
      conv_full(pcoeff_, M, pin, N, s_in, pout, P, s_out, decimation_);
    }
    else if (Supp == support_same)
    {
      VSIP_IMPL_PROFILE(pm_non_opt_calls_++);
      conv_same(pcoeff_, M, pin, N, s_in, pout, P, s_out, decimation_);
    }
    else // (Supp == support_min)
    {
      VSIP_IMPL_PROFILE(pm_non_opt_calls_++);
      conv_min(pcoeff_, M, pin, N, s_in, pout, P, s_out, decimation_);
    }
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
Convolution<ConstViewT, Symm, Supp, T, n_times, a_hint>::
convolve(
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

  typedef vsip::impl::Ext_data<Block0> in_ext_type;
  typedef vsip::impl::Ext_data<Block1> out_ext_type;

  in_ext_type  in_ext (in.block(),  vsip::impl::SYNC_IN,  in_buffer_);
  out_ext_type out_ext(out.block(), vsip::impl::SYNC_OUT, out_buffer_);

  VSIP_IMPL_PROFILE(pm_in_ext_cost_  += in_ext.cost());
  VSIP_IMPL_PROFILE(pm_out_ext_cost_ += out_ext.cost());

  T* pin    = in_ext.data();
  T* pout   = out_ext.data();

  stride_type coeff_row_stride = coeff_ext_.stride(0);
  stride_type coeff_col_stride = coeff_ext_.stride(1);
  stride_type in_row_stride    = in_ext.stride(0);
  stride_type in_col_stride    = in_ext.stride(1);
  stride_type out_row_stride   = out_ext.stride(0);
  stride_type out_col_stride   = out_ext.stride(1);

  if (Supp == support_full)
  {
#if 0
    // Full support not implemented yet.
    if (decimation_ == 1 && coeff_col_stride == 1 &&
	in_ext.stride(1) == 1 && out_ext.stride(1) == 1)
    {
      impl::sal::conv2d_full(
	pcoeff_, Mr, Mc, coeff_row_stride,
	pin, Nr, Nc, in_row_stride,
	pout, out_row_stride);
    }
    else
#endif
    {
      conv_full<T>(pcoeff_, Mr, Mc, coeff_row_stride, coeff_col_stride,
		   pin, Nr, Nc, in_row_stride, in_col_stride,
		   pout, Pr, Pc, out_row_stride, out_col_stride,
		   decimation_);
    }
  }
  else if (Supp == support_same)
  {
    if (decimation_ == 1 && coeff_col_stride == 1 &&
	in_ext.stride(1) == 1 && out_ext.stride(1) == 1)
    {
      if (Mr == 3 && Mc == 3 &&
	  coeff_row_stride == (stride_type)Mc &&
	  in_row_stride    == (stride_type)Nc &&
	  out_row_stride   == (stride_type)Nc)
      {
	impl::sal::conv2d_3x3(sal_pcoeff_, pin, pout, Nr, Nc);
      }
      else
      {
	index_type n0_r = (Mr - 1) - (Mr/2);
	index_type n0_c = (Mc - 1) - (Mc/2);
	index_type n1_r = Nr - (Mr/2);
	index_type n1_c = Nc - (Mc/2);

	T* pout_adj = pout + (n0_r)*out_row_stride
			   + (n0_c)*out_col_stride;

	if (n1_r > n0_r && n1_c > n0_c)
	  impl::sal::conv2d(
	    sal_pcoeff_, Mr, Mc, coeff_row_stride,
	    pin,         Nr, Nc, in_row_stride,
	    pout_adj,    Pr - (Mr-1), Pc - (Mc-1), out_row_stride,
	    decimation_, decimation_);
      }

      // SAL conv2d (conv2dx) is min-support, while SAL conv2d_3x3
      // (f3x3x) leaves a strip of 0 around the edge of the output.
      //
      // Implement same-support by filling out the edges.

      conv_same_edge<T>(pcoeff_, Mr, Mc, coeff_row_stride, coeff_col_stride,
			pin, Nr, Nc, in_row_stride, in_col_stride,
			pout, Pr, Pc, out_row_stride, out_col_stride,
			decimation_);
    }
    else
    {
      conv_same<T>(pcoeff_, Mr, Mc, coeff_row_stride, coeff_col_stride,
		   pin, Nr, Nc, in_row_stride, in_col_stride,
		   pout, Pr, Pc, out_row_stride, out_col_stride,
		   decimation_);
    }
  }
  else // (Supp == support_min)
  {
    if (decimation_ == 1 && coeff_col_stride == 1 &&
	in_ext.stride(1) == 1 && out_ext.stride(1) == 1)
    {
      impl::sal::conv2d(
	sal_pcoeff_, Mr, Mc, coeff_row_stride,
	pin,         Nr, Nc, in_row_stride,
	pout,        Pr, Pc, out_row_stride,
	decimation_, decimation_);
    }
    else
    {
      conv_min<T>(pcoeff_, Mr, Mc, coeff_row_stride, coeff_col_stride,
		  pin, Nr, Nc, in_row_stride, in_col_stride,
		  pout, Pr, Pc, out_row_stride, out_col_stride,
		  decimation_);
    }
  }
}
} // namespace vsip::impl::sal
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
struct Evaluator<op::conv<1, S, R, float, N, H>, be::mercury_sal>
{
  static bool const ct_valid = true;
  typedef impl::sal::Convolution<const_Vector, S, R, float, N, H> backend_type;
};
template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<1, S, R, std::complex<float>, N, H>, be::mercury_sal>
{
  static bool const ct_valid = true;
  typedef impl::sal::Convolution<const_Vector, S, R, std::complex<float>, N, H> backend_type;
};
template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<2, S, R, float, N, H>, be::mercury_sal>
{
  static bool const ct_valid = true;
  typedef impl::sal::Convolution<const_Matrix, S, R, float, N, H> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_SAL_CONV_HPP
