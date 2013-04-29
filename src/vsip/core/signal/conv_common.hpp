/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/signal/conv_common.hpp
    @author  Jules Bergmann
    @date    08/31/2005
    @brief   VSIPL++ Library: Common decls and functions for convolution.
*/

#ifndef VSIP_CORE_SIGNAL_CONV_COMMON_HPP
#define VSIP_CORE_SIGNAL_CONV_COMMON_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/signal/types.hpp>



// C-VSIPL defines the size of the support_min convolution such that
// it sometimes requires values outside of the input support for
// decimations not equal to 1.
//
// This define controls VSIPL++'s behavior for support_min convolutions:
//
// If unset or set to 0, the output size of a support_min
// convolution is as defined by VSIPL++ / C-VSIPL specifications.
//
//       size = floor( (N-1) / D ) - floor( (M-1) / D ) + 1
//
// If set to a value other than 0, the output size of a support_min
// convolution is defined to be:
//
//       size = ceil( (N-M+1) / D )
 
#define VSIP_IMPL_CONV_CORRECT_MIN_SUPPORT_SIZE 1



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip_csl
{
namespace dispatcher
{
namespace op
{
template <dimension_type D,
          symmetry_type S,
          support_region_type R,
          typename T = VSIP_DEFAULT_VALUE_TYPE,
          unsigned int N = 0,
          alg_hint_type H = alg_time>
struct conv;
} // namespace vsip_csl::dispatcher::op
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip
{
namespace impl
{

template <typename T>
struct Convolution_accum_trait
{
  typedef T sum_type;
};



/***********************************************************************
  Definitions
***********************************************************************/

/// Helper function to determine the kernel_size of a convolution, based
/// on the size of the coefficient view and its symmetry.
template <symmetry_type S, dimension_type D> 
Domain<D>
conv_kernel_size(Domain<D> const &coeff_size);

template <>
inline Domain<1>
conv_kernel_size<sym_even_len_odd, 1>(Domain<1> const &coeff_size) 
{ return 2 * coeff_size.length() - 1;}

template <>
inline Domain<1>
conv_kernel_size<sym_even_len_even, 1>(Domain<1> const &coeff_size) 
{ return 2 * coeff_size.length();}

template <>
inline Domain<1>
conv_kernel_size<nonsym, 1>(Domain<1> const &coeff_size) 
{ return coeff_size.length();}

template <>
inline Domain<2>
conv_kernel_size<sym_even_len_odd, 2>(Domain<2> const &coeff_size) 
{ return Domain<2>(2 * coeff_size[0].length() - 1, 2 * coeff_size[1].length() - 1);}

template <>
inline Domain<2>
conv_kernel_size<sym_even_len_even, 2>(Domain<2> const &coeff_size) 
{ return Domain<2>(2 * coeff_size[0].length(), 2 * coeff_size[1].length() - 1);}

template <>
inline Domain<2>
conv_kernel_size<nonsym, 2>(Domain<2> const &coeff_size) 
{ return Domain<2>(coeff_size[0].length(), coeff_size[1].length());}



/// Helper function to determine the output_size of a convolution
/// for a single dimension.

inline length_type
conv_dim_output_size(
  support_region_type supp,
  length_type         M,    // kernel length
  length_type         N,    // input  length
  length_type         D)    // decimation factor
{
  if      (supp == support_full)
    return ((N + M - 2)/D) + 1;
  else if (supp == support_same)
    return ((N - 1)/D) + 1;
  else //(supp == support_min)
  {
#if VSIP_IMPL_CONV_CORRECT_MIN_SUPPORT_SIZE
    return ((N - M + 1) / D) + ((N - M + 1) % D == 0 ? 0 : 1);
#else
    return ((N - 1)/D) - ((M-1)/D) + 1;
#endif
  }
}



/// Helper function to determine the output_size of a convolution.

template <dimension_type Dim>
Domain<Dim>
conv_output_size(
  support_region_type supp,
  Domain<Dim> const&  kernel,
  Domain<Dim> const&  input,
  length_type         dec)    // decimation factor
{
  Domain<1> dom[Dim];

  for (dimension_type d=0; d<Dim; ++d)
    dom[d] = Domain<1>(conv_dim_output_size(supp, kernel[d].size(),
					    input[d].size(),
					    dec));
  return construct_domain<Dim>(dom);
}



/// Helper function to determine a 1-D convolution's kernel from its
/// symmetry and coefficients.

template <typename CoeffViewT,
	  typename T,
	  typename BlockT>
CoeffViewT
conv_kernel(symmetry_type sym, const_Vector<T, BlockT> coeff)
{
  if (sym == sym_even_len_odd)
  {
    length_type M = coeff.size(0);
    CoeffViewT full_coeff(2*M-1);
    assign_local(full_coeff(Domain<1>(0, 1, M))  , coeff);
    assign_local(full_coeff(Domain<1>(M, 1, M-1)), coeff(Domain<1>(M-2, -1, M-1)));
    return full_coeff;
  }
  else if (sym == sym_even_len_even)
  {
    length_type M = coeff.size(0);
    CoeffViewT full_coeff(2*M);
    assign_local(full_coeff(Domain<1>(0, 1, M)), coeff);
    assign_local(full_coeff(Domain<1>(M, 1, M)), coeff(Domain<1>(M-1, -1, M)));
    return full_coeff;
  }
  else /* (sym == nonsym) */
  {
    length_type M = coeff.size(0);
    CoeffViewT full_coeff(M);
    assign_local(full_coeff, coeff);
    return full_coeff;
  }
}



/// Helper function to determine a 2-D convolution's kernel from its
/// symmetry and coefficients.

template <typename CoeffViewT,
	  typename T,
	  typename BlockT>
CoeffViewT
conv_kernel(symmetry_type sym, const_Matrix<T, BlockT> coeff)
{
  if (sym == sym_even_len_odd)
  {
    length_type Mr = coeff.size(0);
    length_type Mc = coeff.size(1);
    CoeffViewT full_coeff(2*Mr-1, 2*Mc-1);

    // fill upper-left
    full_coeff(Domain<2>(Domain<1>(0, 1, Mr), Domain<1>(0, 1, Mc))) = coeff;

    // fill upper-right
    full_coeff(Domain<2>(Domain<1>(0, 1, Mr), Domain<1>(Mc, 1, Mc-1))) =
      coeff(Domain<2>(Mr, Domain<1>(Mc-2, -1, Mc-1)));

    // fill lower-right (by folding over)
    full_coeff(Domain<2>(Domain<1>(Mr, 1, Mr-1), 2*Mc-1)) =
      full_coeff(Domain<2>(Domain<1>(Mr-2, -1, Mr-1), 2*Mc-1));

    return full_coeff;
  }
  else if (sym == sym_even_len_even)
  {
    length_type Mr = coeff.size(0);
    length_type Mc = coeff.size(1);
    CoeffViewT full_coeff(2*Mr, 2*Mc);

    // fill upper-left
    full_coeff(Domain<2>(Mr, Mc)) = coeff;

    // fill upper-right
    full_coeff(Domain<2>(Mr, Domain<1>(Mc, 1, Mc))) =
      coeff(Domain<2>(Mr, Domain<1>(Mc-1, -1, Mc)));

    // fill lower-right (by folding over)
    full_coeff(Domain<2>(Domain<1>(Mr, 1, Mr), 2*Mc)) =
      full_coeff(Domain<2>(Domain<1>(Mr-1, -1, Mr), 2*Mc));

    return full_coeff;
  }
  else /* (sym == nonsym) */
  {
    length_type Mr = coeff.size(0);
    length_type Mc = coeff.size(1);
    CoeffViewT full_coeff(Mr, Mc);
    full_coeff = coeff;
    return full_coeff;
  }
}



/***********************************************************************
  1-D Convolutions (interleaved)
***********************************************************************/

/// Perform 1-D convolution with full region of support.

template <typename T>
inline void
conv_full(T const *coeff,
	  length_type coeff_size,	// M
	  T const *in,
	  length_type in_size,		// N
	  stride_type in_stride,
	  T *out,
	  length_type out_size,		// P
	  stride_type out_stride,
	  length_type decimation)
{
  typedef typename Convolution_accum_trait<T>::sum_type sum_type;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum = sum_type();
      
    for (index_type k=0; k<coeff_size; ++k)
    {
      if (n*decimation >= k && n*decimation-k < in_size)
	sum += coeff[k] * in[(n*decimation-k) * in_stride];
    }
    out[n * out_stride] = sum;
  }
}



/// Perform 1-D convolution with same region of support.

template <typename T>
inline void
conv_same(T const *coeff,
	  length_type coeff_size,	// M
	  T const *in,
	  length_type in_size,		// N
	  stride_type in_stride,
	  T *out,
	  length_type out_size,		// P
	  stride_type out_stride,
	  length_type decimation)
{
  typedef typename Convolution_accum_trait<T>::sum_type sum_type;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum = sum_type();
      
    for (index_type k=0; k<coeff_size; ++k)
    {
      if (n*decimation + (coeff_size/2)   >= k &&
	  n*decimation + (coeff_size/2)-k <  in_size)
	sum += coeff[k] * in[(n*decimation+(coeff_size/2)-k) * in_stride];
    }
    out[n * out_stride] = sum;
  }
}



/// Perform 1-D convolution with minimal region of support.

template <typename T>
inline void
conv_min(T const *coeff,
	 length_type coeff_size,		// M
	 T const *in,
	 length_type in_size ATTRIBUTE_UNUSED,	// N
	 stride_type in_stride,
	 T *out,
	 length_type out_size,			// P
	 stride_type out_stride,
	 length_type decimation)
{
  typedef typename Convolution_accum_trait<T>::sum_type sum_type;

#if VSIP_IMPL_CONV_CORRECT_MIN_SUPPORT_SIZE
  assert((out_size-1)*decimation+(coeff_size-1) < in_size);

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum = sum_type();
      
    index_type offset = n*decimation+(coeff_size-1);
    for (index_type k=0; k<coeff_size; ++k)
    {
      sum += coeff[k] * in[(offset-k) * in_stride];
    }
    out[n * out_stride] = sum;
  }
#else
  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum = sum_type();
      
    index_type offset = n*decimation+(coeff_size-1);
    for (index_type k=0; k<coeff_size; ++k)
    {
      if (offset-k < in_size)
	sum += coeff[k] * in[(offset-k) * in_stride];
    }
    out[n * out_stride] = sum;
  }
#endif
}



/***********************************************************************
  1-D Convolutions (split)
***********************************************************************/

/// Perform 1-D convolution with full region of support.

template <typename T>
inline void
conv_full(std::pair<T const*, T const*> coeff,
	  length_type coeff_size,	// M
	  std::pair<T const*, T const*> in,
	  length_type in_size,		// N
	  stride_type in_stride,
	  std::pair<T*, T*> out,
	  length_type out_size,		// P
	  stride_type out_stride,
	  length_type decimation)
{
  typedef typename Convolution_accum_trait<complex<T> >::sum_type sum_type;
  typedef Storage<split_complex, complex<T> > storage_type;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum = sum_type();
      
    for (index_type k=0; k<coeff_size; ++k)
    {
      if (n*decimation >= k && n*decimation-k < in_size)
	sum += storage_type::get(coeff, k) *
	       storage_type::get(in,   (n*decimation-k) * in_stride);
    }
    storage_type::put(out, n * out_stride, sum);
  }
}



/// Perform 1-D convolution with same region of support.

template <typename T>
inline void
conv_same(std::pair<T const *, T const *> coeff,
	  length_type coeff_size,	// M
	  std::pair<T const *, T const *> in,
	  length_type       in_size,		// N
	  stride_type       in_stride,
	  std::pair<T*, T*> out,
	  length_type       out_size,		// P
	  stride_type       out_stride,
	  length_type       decimation)
{
  typedef typename Convolution_accum_trait<complex<T> >::sum_type sum_type;
  typedef Storage<split_complex, complex<T> > storage_type;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum = sum_type();
      
    for (index_type k=0; k<coeff_size; ++k)
    {
      if (n*decimation + (coeff_size/2)   >= k &&
	  n*decimation + (coeff_size/2)-k <  in_size)
	sum += storage_type::get(coeff, k) *
	       storage_type::get(in, (n*decimation+(coeff_size/2)-k) * in_stride);
    }
    storage_type::put(out, n * out_stride, sum);
  }
}



/// Perform 1-D convolution with minimal region of support.

template <typename T>
inline void
conv_min(std::pair<T const *, T const *> coeff,
	 length_type       coeff_size,			// M
	 std::pair<T const *, T const *> in,
	 length_type       in_size ATTRIBUTE_UNUSED,	// N
	 stride_type       in_stride,
	 std::pair<T*, T*> out,
	 length_type       out_size,			// P
	 stride_type       out_stride,
	 length_type       decimation)
{
  typedef typename Convolution_accum_trait<complex<T> >::sum_type sum_type;
  typedef Storage<split_complex, complex<T> > storage_type;

#if VSIP_IMPL_CONV_CORRECT_MIN_SUPPORT_SIZE
  assert((out_size-1)*decimation+(coeff_size-1) < in_size);

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum = sum_type();
      
    index_type offset = n*decimation+(coeff_size-1);
    for (index_type k=0; k<coeff_size; ++k)
    {
      sum += storage_type::get(coeff, k) *
	     storage_type::get(in,    (offset-k) * in_stride);
    }
    storage_type::put(out, n * out_stride, sum);
  }
#else
  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum = sum_type();
      
    index_type offset = n*decimation+(coeff_size-1);
    for (index_type k=0; k<coeff_size; ++k)
    {
      if (offset-k < in_size)
        sum += storage_type::get(coeff, k) *
	       storage_type::get(in,    (offset-k) * in_stride);
    }
    storage_type::put(out, n * out_stride, sum);
  }
#endif
}



/***********************************************************************
  2-D Convolutions (interleaved)
***********************************************************************/

/// Perform 2-D convolution with full region of support.

template <typename T>
inline void
conv_full(T const *coeff,
	  length_type coeff_rows,	// Mr
	  length_type coeff_cols,	// Mc
	  stride_type coeff_row_stride,
	  stride_type coeff_col_stride,
	  T const *in,
	  length_type in_rows,		// Nr
	  length_type in_cols,		// Nc
	  stride_type in_row_stride,
	  stride_type in_col_stride,
	  T *out,
	  length_type out_rows,		// Pr
	  length_type out_cols,		// Pc
	  stride_type out_row_stride,
	  stride_type out_col_stride,
	  length_type decimation)
{
  typedef typename Convolution_accum_trait<T>::sum_type sum_type;

  for (index_type r=0; r<out_rows; ++r)
  {
    for (index_type c=0; c<out_cols; ++c)
    {
      sum_type sum = sum_type();

      for (index_type rr=0; rr<coeff_rows; ++rr)
      {
	for (index_type cc=0; cc<coeff_cols; ++cc)
	{
	  if (r*decimation >= rr && r*decimation-rr < in_rows &&
	      c*decimation >= cc && c*decimation-cc < in_cols)
	  {
	    sum += coeff[rr*coeff_row_stride + cc*coeff_col_stride] *
                   in[(r*decimation-rr) * in_row_stride +
		      (c*decimation-cc) * in_col_stride];
	  }
	}
      }
      out[r * out_row_stride + c * out_col_stride] = sum;
    }
  }
}



/// Perform 2-D convolution with same region of support.

template <typename T>
inline void
conv_same(T const *coeff,
	  length_type coeff_rows,	// Mr
	  length_type coeff_cols,	// Mc
	  stride_type coeff_row_stride,
	  stride_type coeff_col_stride,
	  T const *in,
	  length_type in_rows,		// Nr
	  length_type in_cols,		// Nc
	  stride_type in_row_stride,
	  stride_type in_col_stride,
	  T *out,
	  length_type out_rows,		// Pr
	  length_type out_cols,		// Pc
	  stride_type out_row_stride,
	  stride_type out_col_stride,
	  length_type decimation)
{
  typedef typename Convolution_accum_trait<T>::sum_type sum_type;

  for (index_type r=0; r<out_rows; ++r)
  {
    index_type ir = r*decimation + (coeff_rows/2);

    for (index_type c=0; c<out_cols; ++c)
    {
      index_type ic = c*decimation + (coeff_cols/2);

      sum_type sum = sum_type();

      for (index_type rr=0; rr<coeff_rows; ++rr)
      {
	for (index_type cc=0; cc<coeff_cols; ++cc)
	{

	  if (ir >= rr && ir-rr < in_rows && ic >= cc && ic-cc < in_cols) 
	  {
	    sum += coeff[rr*coeff_row_stride + cc*coeff_col_stride] *
	           in[(ir-rr) * in_row_stride + (ic-cc) * in_col_stride];
	  }
	}
      }
      out[r * out_row_stride + c * out_col_stride] = sum;
    }
  }
}



/// Perform 2-D convolution with minimal region of support.

template <typename T>
inline void
conv_min(T const *coeff,
	 length_type coeff_rows,	// Mr
	 length_type coeff_cols,	// Mc
	 stride_type coeff_row_stride,
	 stride_type coeff_col_stride,
	 T const *in,
	 length_type in_rows,		// Nr
	 length_type in_cols,		// Nc
	 stride_type in_row_stride,
	 stride_type in_col_stride,
	 T *out,
	 length_type out_rows,		// Pr
	 length_type out_cols,		// Pc
	 stride_type out_row_stride,
	 stride_type out_col_stride,
	 length_type decimation)
{
  typedef typename Convolution_accum_trait<T>::sum_type sum_type;

#if VSIP_IMPL_CONV_CORRECT_MIN_SUPPORT_SIZE
  (void)in_rows;
  (void)in_cols;

  for (index_type r=0; r<out_rows; ++r)
  {
    index_type ir = r*decimation + (coeff_rows-1);
    for (index_type c=0; c<out_cols; ++c)
    {
      index_type ic = c*decimation + (coeff_cols-1);

      sum_type sum = sum_type();

      for (index_type rr=0; rr<coeff_rows; ++rr)
      {
	for (index_type cc=0; cc<coeff_cols; ++cc)
	{
	  sum += coeff[rr*coeff_row_stride + cc*coeff_col_stride] *
	         in[(ir-rr) * in_row_stride + (ic-cc) * in_col_stride];
	}
      }
      out[r * out_row_stride + c * out_col_stride] = sum;
    }
  }
#else
  for (index_type r=0; r<out_rows; ++r)
  {
    index_type ir = r*decimation + (coeff_rows-1);
    for (index_type c=0; c<out_cols; ++c)
    {
      index_type ic = c*decimation + (coeff_cols-1);

      sum_type sum = sum_type();

      for (index_type rr=0; rr<coeff_rows; ++rr)
      {
	for (index_type cc=0; cc<coeff_cols; ++cc)
	{
	  if (ir-rr < in_rows && ic-cc < in_cols)
	  {
	    sum += coeff[rr*coeff_row_stride + cc*coeff_col_stride] *
	           in[(ir-rr) * in_row_stride + (ic-cc) * in_col_stride];
	  }
	}
      }
      out[r * out_row_stride + c * out_col_stride] = sum;
    }
  }
#endif
}



/// Perform edge portion of 2-D convolution with same region of support.

/// conv_same = conv_min + conv_same_edge

template <typename T>
inline void
conv_same_edge(T const *coeff,
	       length_type coeff_rows,	// Mr
	       length_type coeff_cols,	// Mc
	       stride_type coeff_row_stride,
	       stride_type coeff_col_stride,
	       T const *in,
	       length_type in_rows,		// Nr
	       length_type in_cols,		// Nc
	       stride_type in_row_stride,
	       stride_type in_col_stride,
	       T *out,
	       length_type out_rows,		// Pr
	       length_type out_cols,		// Pc
	       stride_type out_row_stride,
	       stride_type out_col_stride,
	       length_type decimation)
{
  typedef typename Convolution_accum_trait<T>::sum_type sum_type;

  for (index_type r=0; r<out_rows; ++r)
  {
    index_type ir = r*decimation + (coeff_rows/2);

    for (index_type c=0; c<out_cols; ++c)
    {
      index_type ic = c*decimation + (coeff_cols/2);

      if ((r < coeff_rows/2 || r >= out_rows-(coeff_rows/2)) ||
	  (c < coeff_cols/2 || c >= out_cols-(coeff_cols/2)))
      {
	sum_type sum = sum_type();

	for (index_type rr=0; rr<coeff_rows; ++rr)
	{
	  for (index_type cc=0; cc<coeff_cols; ++cc)
	  {

	    if (ir >= rr && ir-rr < in_rows && ic >= cc && ic-cc < in_cols) 
	    {
	      sum += coeff[rr*coeff_row_stride + cc*coeff_col_stride] *
		     in[(ir-rr) * in_row_stride + (ic-cc) * in_col_stride];
	    }
	  }
	}
	out[r * out_row_stride + c * out_col_stride] = sum;
      }
    }
  }
}

/***********************************************************************
  2-D Convolutions (split)
***********************************************************************/

/// Perform 2-D convolution with full region of support.

template <typename T>
inline void
conv_full(std::pair<T const *, T const *> coeff,
	  length_type coeff_rows,	// Mr
	  length_type coeff_cols,	// Mc
	  stride_type coeff_row_stride,
	  stride_type coeff_col_stride,
	  std::pair<T const *, T const *> in,
	  length_type in_rows,		// Nr
	  length_type in_cols,		// Nc
	  stride_type in_row_stride,
	  stride_type in_col_stride,
	  std::pair<T*, T*> out,
	  length_type out_rows,		// Pr
	  length_type out_cols,		// Pc
	  stride_type out_row_stride,
	  stride_type out_col_stride,
	  length_type decimation)
{
  typedef typename Convolution_accum_trait<complex<T> >::sum_type sum_type;
  typedef Storage<split_complex, complex<T> > storage_type;

  for (index_type r=0; r<out_rows; ++r)
  {
    for (index_type c=0; c<out_cols; ++c)
    {
      sum_type sum = sum_type();

      for (index_type rr=0; rr<coeff_rows; ++rr)
      {
	for (index_type cc=0; cc<coeff_cols; ++cc)
	{
	  if (r*decimation >= rr && r*decimation-rr < in_rows &&
	      c*decimation >= cc && c*decimation-cc < in_cols)
	  {
	    sum = sum +
	           storage_type::get(coeff,rr*coeff_row_stride + cc*coeff_col_stride) *
                   storage_type::get(in,(r*decimation-rr) * in_row_stride +
		      (c*decimation-cc) * in_col_stride);
	  }
	}
      }
      storage_type::put(out,r * out_row_stride + c * out_col_stride,sum);
    }
  }
}



/// Perform 2-D convolution with same region of support.

template <typename T>
inline void
conv_same(std::pair<T const *, T const *> coeff,
	  length_type coeff_rows,	// Mr
	  length_type coeff_cols,	// Mc
	  stride_type coeff_row_stride,
	  stride_type coeff_col_stride,
	  std::pair<T const *, T const *> in,
	  length_type in_rows,		// Nr
	  length_type in_cols,		// Nc
	  stride_type in_row_stride,
	  stride_type in_col_stride,
	  std::pair<T*, T*> out,
	  length_type out_rows,		// Pr
	  length_type out_cols,		// Pc
	  stride_type out_row_stride,
	  stride_type out_col_stride,
	  length_type decimation)
{
  typedef typename Convolution_accum_trait<complex<T> >::sum_type sum_type;
  typedef Storage<split_complex, complex<T> > storage_type;

  for (index_type r=0; r<out_rows; ++r)
  {
    index_type ir = r*decimation + (coeff_rows/2);

    for (index_type c=0; c<out_cols; ++c)
    {
      index_type ic = c*decimation + (coeff_cols/2);

      sum_type sum = sum_type();

      for (index_type rr=0; rr<coeff_rows; ++rr)
      {
	for (index_type cc=0; cc<coeff_cols; ++cc)
	{

	  if (ir >= rr && ir-rr < in_rows && ic >= cc && ic-cc < in_cols) 
	  {
	    sum = sum +
	           storage_type::get(coeff,rr*coeff_row_stride + cc*coeff_col_stride) *
	           storage_type::get(in,(ir-rr) * in_row_stride + (ic-cc) * in_col_stride);
	  }
	}
      }
      storage_type::put(out,r * out_row_stride + c * out_col_stride,sum);
    }
  }
}



/// Perform 2-D convolution with minimal region of support.

template <typename T>
inline void
conv_min(std::pair<T const *, T const *> coeff,
	 length_type coeff_rows,	// Mr
	 length_type coeff_cols,	// Mc
	 stride_type coeff_row_stride,
	 stride_type coeff_col_stride,
	 std::pair<T const *, T const *> in,
	 length_type in_rows,		// Nr
	 length_type in_cols,		// Nc
	 stride_type in_row_stride,
	 stride_type in_col_stride,
	 std::pair<T*, T*> out,
	 length_type out_rows,		// Pr
	 length_type out_cols,		// Pc
	 stride_type out_row_stride,
	 stride_type out_col_stride,
	 length_type decimation)
{
  typedef typename Convolution_accum_trait<complex<T> >::sum_type sum_type;
  typedef Storage<split_complex, complex<T> > storage_type;

#if VSIP_IMPL_CONV_CORRECT_MIN_SUPPORT_SIZE
  (void)in_rows;
  (void)in_cols;

  for (index_type r=0; r<out_rows; ++r)
  {
    index_type ir = r*decimation + (coeff_rows-1);
    for (index_type c=0; c<out_cols; ++c)
    {
      index_type ic = c*decimation + (coeff_cols-1);

      sum_type sum = sum_type();

      for (index_type rr=0; rr<coeff_rows; ++rr)
      {
	for (index_type cc=0; cc<coeff_cols; ++cc)
	{
	  sum = sum +
	         storage_type::get(coeff,rr*coeff_row_stride + cc*coeff_col_stride) *
	         storage_type::get(in,(ir-rr) * in_row_stride + (ic-cc) * in_col_stride);
	}
      }
      storage_type::put(out,r * out_row_stride + c * out_col_stride,sum);
    }
  }
#else
  for (index_type r=0; r<out_rows; ++r)
  {
    index_type ir = r*decimation + (coeff_rows-1);
    for (index_type c=0; c<out_cols; ++c)
    {
      index_type ic = c*decimation + (coeff_cols-1);

      sum_type sum = sum_type();

      for (index_type rr=0; rr<coeff_rows; ++rr)
      {
	for (index_type cc=0; cc<coeff_cols; ++cc)
	{
	  if (ir-rr < in_rows && ic-cc < in_cols)
	  {
	    sum = sum +
	           storage_type::get(coeff,rr*coeff_row_stride + cc*coeff_col_stride) *
	           storage_type::get(in,(ir-rr) * in_row_stride + (ic-cc) * in_col_stride);
	  }
	}
      }
      storage_type::put(out,r * out_row_stride + c * out_col_stride,sum);
    }
  }
#endif
}



/// Perform edge portion of 2-D convolution with same region of support.

/// conv_same = conv_min + conv_same_edge

template <typename T>
inline void
conv_same_edge(std::pair<T const *, T const *> coeff,
	       length_type coeff_rows,	// Mr
	       length_type coeff_cols,	// Mc
	       stride_type coeff_row_stride,
	       stride_type coeff_col_stride,
	       std::pair<T const *, T const *> in,
	       length_type in_rows,		// Nr
	       length_type in_cols,		// Nc
	       stride_type in_row_stride,
	       stride_type in_col_stride,
	       std::pair<T*, T*> out,
	       length_type out_rows,		// Pr
	       length_type out_cols,		// Pc
	       stride_type out_row_stride,
	       stride_type out_col_stride,
	       length_type decimation)
{
  typedef typename Convolution_accum_trait<complex<T> >::sum_type sum_type;
  typedef Storage<split_complex, complex<T> > storage_type;

  for (index_type r=0; r<out_rows; ++r)
  {
    index_type ir = r*decimation + (coeff_rows/2);

    for (index_type c=0; c<out_cols; ++c)
    {
      index_type ic = c*decimation + (coeff_cols/2);

      if ((r < coeff_rows/2 || r >= out_rows-(coeff_rows/2)) ||
	  (c < coeff_cols/2 || c >= out_cols-(coeff_cols/2)))
      {
	sum_type sum = sum_type();

	for (index_type rr=0; rr<coeff_rows; ++rr)
	{
	  for (index_type cc=0; cc<coeff_cols; ++cc)
	  {

	    if (ir >= rr && ir-rr < in_rows && ic >= cc && ic-cc < in_cols) 
	    {
	      sum = sum +
	             storage_type::get(coeff,rr*coeff_row_stride + cc*coeff_col_stride) *
		     storage_type::get(in,(ir-rr) * in_row_stride + (ic-cc) * in_col_stride);
	    }
	  }
	}
	storage_type::put(out,r * out_row_stride + c * out_col_stride,sum);
      }
    }
  }
}


/// Example of how to combine conv_min and conv_same_edge to achieve
/// conv_same.

template <typename T>
inline void
conv_same_example(T const *coeff,
		  length_type coeff_rows,	// Mr
		  length_type coeff_cols,	// Mc
		  stride_type coeff_row_stride,
		  stride_type coeff_col_stride,
		  T const *in,
		  length_type in_rows,		// Nr
		  length_type in_cols,		// Nc
		  stride_type in_row_stride,
		  stride_type in_col_stride,
		  T *out,
		  length_type out_rows,		// Pr
		  length_type out_cols,		// Pc
		  stride_type out_row_stride,
		  stride_type out_col_stride,
		  length_type decimation)
{
  // Determine the first element computed by conv_min.
  index_type n0_r  = ( (coeff_rows - 1) - (coeff_rows/2) ) / decimation;
  index_type n0_c  = ( (coeff_cols - 1) - (coeff_cols/2) ) / decimation;
  index_type res_r = ( (coeff_rows - 1) - (coeff_rows/2) ) % decimation;
  index_type res_c = ( (coeff_cols - 1) - (coeff_cols/2) ) % decimation;
  if (res_r > 0) n0_r += 1;
  if (res_c > 0) n0_c += 1;

  // Determine the phase of the input given to conv_min.
  index_type phase_r = (res_r == 0) ? 0 : (decimation - res_r);
  index_type phase_c = (res_c == 0) ? 0 : (decimation - res_c);


  // Determine the last element + 1 computed by conv_min.
  index_type n1_r = (in_rows - (coeff_rows/2)) / decimation;
  index_type n1_c = (in_cols - (coeff_cols/2)) / decimation;
  if ((in_rows - (coeff_rows/2)) % decimation > 0) n1_r++;
  if ((in_cols - (coeff_cols/2)) % decimation > 0) n1_c++;


  T* out_adj = out + (n0_r)*out_row_stride
		   + (n0_c)*out_col_stride;
  T* in_adj  = in  + (phase_r)*in_row_stride
		   + (phase_c)*in_col_stride;

  if (n1_r > n0_r && n1_c > n0_c)
    conv_min<T>(coeff,
		coeff_rows, coeff_cols,
		coeff_row_stride, coeff_col_stride,

		in_adj,
		in_rows - phase_r, in_cols - phase_c,
		in_row_stride, in_col_stride,

		out_adj,
		n1_r - n0_r, n1_c - n0_c,
		out_row_stride, out_col_stride,
		decimation);

  conv_same_edge<T>(coeff, coeff_rows, coeff_cols, coeff_row_stride, coeff_col_stride,
		    in, in_rows, in_cols, in_row_stride, in_col_stride,
		    out, out_rows, out_cols, out_row_stride, out_col_stride,
		    decimation);
}



} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_CORE_SIGNAL_CONV_COMMON_HPP
