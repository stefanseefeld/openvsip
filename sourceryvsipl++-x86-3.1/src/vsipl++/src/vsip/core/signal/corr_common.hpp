/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/signal/corr-common.hpp
    @author  Jules Bergmann
    @date    2005-10-05
    @brief   VSIPL++ Library: Common decls and functions for correlation.
*/

#ifndef VSIP_CORE_SIGNAL_CORR_COMMON_HPP
#define VSIP_CORE_SIGNAL_CORR_COMMON_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/signal/types.hpp>


// C-VSIPL defines the scaling for support_same correlation such
// that the number of terms in the correlation product is different
// from the scaling factor.
//
// This occurs for the last elements in output.
//
// For N - floor(M/2) <= n < N, C-VSIPL defines the following scaling:
//                    1
//     scale = -----------------------
//              N - 1 + ceil(M/2) - n
//
// For the last element n=N-1 the scaling factor is:
//
//                    1                          1
//     scale = -------------------------- = -----------
//              N - 1 + cel(M/2) - (N-1)     ceil(M/2)
//
// The biased value y_^n is:
//
// y_^n = sum from k=0 to M-1 : r_k  *  x'_{n+k-floor(M/2)}
//
// For n = N-1:
//
//    y_^{N-1} = sum from k=0 to M-1 : r_k  *  x'_{{N-1}+k-floor(M/2)}
//
// since k > M/2 => that the index to x_j will be outside the support
// of x, in which case 0 will be used instead, we can adjust the loop
// bounds:
//
//    y_^{N-1} = sum from k=0 to floor(M/2) : r_k  *  x'_{{N-1}+k-floor(M/2)}
//
// The range of k is from 0 to floor(M/2) is *inclusive*, which means
// that there are floor(M/2) + 1 terms in the sum.  When M is odd,
// ceil(M/2) = floor(M/2)+1, so the scaling factor of 1/ceil(M/2) is correct.
//
// However, when M is even, ceil(M/2) == floor(M/2), resulting in a scaling
// factor that is off by one.
//
// This define controls Sourcery VSIPL++'s behavior for unbiased
// support_same correlations:
//
// If set to 0, the scaling defined by C-VSIPL is used.
//
// If set to non-0, the scaling 1/(floor(M/2)+1) is used.

#define VSIP_IMPL_CORR_CORRECT_SAME_SUPPORT_SCALING 0



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
          support_region_type R,
          typename T = VSIP_DEFAULT_VALUE_TYPE,
          unsigned int N = 0,
          alg_hint_type H = alg_time>
struct corr;
} // namespace vsip_csl::dispatcher::op
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip
{
namespace impl
{

template <typename T>
struct Correlation_accum_trait
{
  typedef T sum_type;
};



/***********************************************************************
  1-D Correlations (interleaved)
***********************************************************************/

/// Perform 1-D correlation with full region of support.

template <typename T>
inline void
corr_full(bias_type   bias,
	  T const *ref,
	  length_type ref_size,		// M
	  stride_type ref_stride,
	  T const *in,
	  length_type in_size,		// N
	  stride_type in_stride,
	  T *out,
	  length_type out_size,		// P
	  stride_type out_stride)
{
  assert(ref_size <= in_size);

  typedef typename Correlation_accum_trait<T>::sum_type sum_type;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum   = sum_type();
      
    for (index_type k=0; k<ref_size; ++k)
    {
      index_type pos = n + k - (ref_size-1);

      if (n+k >= (ref_size-1) && n+k < in_size+(ref_size-1))
      {
	sum += ref[k * ref_stride] * impl_conj(in[pos * in_stride]);
      }
    }

    if (bias == unbiased)
    {
      if (n < ref_size-1)
	sum /= sum_type(n+1);
      else if (n >= in_size)
	sum /= sum_type(in_size + ref_size - 1 - n);
      else
	sum /= sum_type(ref_size);
    }
      
    out[n * out_stride] = sum;
  }
}



/// Perform 1-D correlation with same region of support.

template <typename T>
inline void
corr_same(bias_type   bias,
	  T const *ref,
	  length_type ref_size,		// M
	  stride_type ref_stride,
	  T const *in,
	  length_type in_size,		// N
	  stride_type in_stride,
	  T *out,
	  length_type out_size,		// P
	  stride_type out_stride)
{
  typedef typename Correlation_accum_trait<T>::sum_type sum_type;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum   = sum_type();
      
    for (index_type k=0; k<ref_size; ++k)
    {
      index_type pos = n + k - (ref_size/2);

      if (n+k >= (ref_size/2) && n+k <  in_size + (ref_size/2))
      {
	sum += ref[k * ref_stride] * impl_conj(in[pos * in_stride]);
      }
    }
    if (bias == unbiased)
    {
      if (n < ref_size/2)
	sum /= sum_type(n + ((ref_size+1)/2));
      else if (n >= in_size - (ref_size/2))
      {
#if VSIP_IMPL_CORR_CORRECT_SAME_SUPPORT_SCALING
	sum /= sum_type(in_size     + (ref_size    /2) - n);
#else
	// Definition in C-VSIPL:
	sum /= sum_type(in_size - 1 + ((ref_size+1)/2) - n);
#endif
      }
      else
	sum /= sum_type(ref_size);
    }
    out[n * out_stride] = sum;
  }
}



/// Perform correlation with minimal region of support.

template <typename T>
inline void
corr_min(bias_type   bias,
	 T const *ref,
	 length_type ref_size,		// M
	 stride_type ref_stride,
	 T const *in,
	 length_type /*in_size*/,	// N
	 stride_type in_stride,
	 T *out,
	 length_type out_size,		// P
	 stride_type out_stride)
{
  typedef typename Correlation_accum_trait<T>::sum_type sum_type;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum = sum_type();
      
    for (index_type k=0; k<ref_size; ++k)
    {
      sum += ref[k*ref_stride] * impl_conj(in[(n+k) * in_stride]);
    }

    if (bias == unbiased)
      sum /= sum_type(ref_size);

    out[n * out_stride] = sum;
  }
}


/***********************************************************************
  1-D Correlations (split)
***********************************************************************/

/// Perform 1-D correlation with full region of support.

template <typename T>
inline void
corr_full(bias_type bias,
	  std::pair<T const *, T const *> ref,
	  length_type       ref_size,		// M
	  stride_type       ref_stride,
	  std::pair<T const *, T const *> in,
	  length_type       in_size,		// N
	  stride_type       in_stride,
	  std::pair<T*, T*> out,
	  length_type       out_size,		// P
	  stride_type       out_stride)
{
  assert(ref_size <= in_size);

  typedef typename Correlation_accum_trait<std::complex<T> >::sum_type sum_type;
  typedef Storage<split_complex, complex<T> > storage_type;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum   = sum_type();
      
    for (index_type k=0; k<ref_size; ++k)
    {
      index_type pos = n + k - (ref_size-1);

      if (n+k >= (ref_size-1) && n+k < in_size+(ref_size-1))
      {
	sum = sum +
	       storage_type::get(ref, k * ref_stride) *
	       impl_conj(storage_type::get(in, pos * in_stride));
      }
    }

    if (bias == unbiased)
    {
      if (n < ref_size-1)
	sum /= sum_type(n+1);
      else if (n >= in_size)
	sum /= sum_type(in_size + ref_size - 1 - n);
      else
	sum /= sum_type(ref_size);
    }
      
    storage_type::put(out, n * out_stride, sum);
  }
}


/// Perform 1-D correlation with same region of support.

template <typename T>
inline void
corr_same(bias_type         bias,
	  std::pair<T const *, T const *> ref,
	  length_type       ref_size,		// M
	  stride_type       ref_stride,
	  std::pair<T const *, T const *> in,
	  length_type       in_size,		// N
	  stride_type       in_stride,
	  std::pair<T*, T*> out,
	  length_type       out_size,		// P
	  stride_type       out_stride)
{
  typedef typename Correlation_accum_trait<std::complex<T> >::sum_type sum_type;
  typedef Storage<split_complex, complex<T> > storage_type;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum   = sum_type();
      
    for (index_type k=0; k<ref_size; ++k)
    {
      index_type pos = n + k - (ref_size/2);

      if (n+k >= (ref_size/2) && n+k <  in_size + (ref_size/2))
      {
	sum = sum +
	       storage_type::get(ref, k * ref_stride) *
	       impl_conj(storage_type::get(in, pos * in_stride));
      }
    }
    if (bias == unbiased)
    {
      if (n < ref_size/2)
	sum /= sum_type(n + ((ref_size+1)/2));
      else if (n >= in_size - (ref_size/2))
      {
#if VSIP_IMPL_CORR_CORRECT_SAME_SUPPORT_SCALING
	sum /= sum_type(in_size     + (ref_size    /2) - n);
#else
	// Definition in C-VSIPL:
	sum /= sum_type(in_size - 1 + ((ref_size+1)/2) - n);
#endif
      }
      else
	sum /= sum_type(ref_size);
    }
    storage_type::put(out, n * out_stride, sum);
  }
}



/// Perform correlation with minimal region of support.

template <typename T>
inline void
corr_min(bias_type         bias,
	 std::pair<T const *, T const *> ref,
	 length_type       ref_size,		// M
	 stride_type       ref_stride,
	 std::pair<T const *, T const *> in,
	 length_type       /*in_size*/,	// N
	 stride_type       in_stride,
	 std::pair<T*, T*> out,
	 length_type       out_size,		// P
	 stride_type       out_stride)
{
  typedef typename Correlation_accum_trait<std::complex<T> >::sum_type sum_type;
  typedef Storage<split_complex, complex<T> > storage_type;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum = sum_type();
      
    for (index_type k=0; k<ref_size; ++k)
    {
      sum = sum +
             storage_type::get(ref, k*ref_stride) *
	     impl_conj(storage_type::get(in, (n+k) * in_stride));
    }

    if (bias == unbiased)
      sum /= sum_type(ref_size);

    storage_type::put(out, n * out_stride, sum);
  }
}



/***********************************************************************
  2-D Definitions
***********************************************************************/

/// Perform 2-D correlation with full region of support.

template <typename T>
inline void
corr_base(bias_type   bias,
	  T const *ref,
	  length_type ref_rows,		// Mr
	  length_type ref_cols,		// Mc
	  stride_type ref_row_stride,
	  stride_type ref_col_stride,
	  length_type row_shift,
	  length_type col_shift,
	  length_type row_edge,
	  length_type col_edge,
	  T const *in,
	  length_type in_rows,		// Nr
	  length_type in_cols,		// Nc
	  stride_type in_row_stride,
	  stride_type in_col_stride,
	  T *out,
	  length_type out_rows,		// Pr
	  length_type out_cols,		// Pc
	  stride_type out_row_stride,
	  stride_type out_col_stride)
{
  assert(ref_rows <= in_rows);
  assert(ref_cols <= in_cols);

  typedef typename Correlation_accum_trait<T>::sum_type sum_type;

  for (index_type r=0; r<out_rows; ++r)
  {
    for (index_type c=0; c<out_cols; ++c)
    {
      sum_type sum   = sum_type();
      
      for (index_type rr=0; rr<ref_rows; ++rr)
      {
	for (index_type cc=0; cc<ref_cols; ++cc)
	{
	  index_type rpos = r + rr - row_shift;
	  index_type cpos = c + cc - col_shift;

	  if (r+rr >= row_shift && r+rr < in_rows+row_shift &&
	      c+cc >= col_shift && c+cc < in_cols+col_shift)
	  {
	    sum += ref[rr * ref_row_stride + cc * ref_col_stride] *
                   impl_conj(in[rpos * in_row_stride + cpos * in_col_stride]);
	  }
	}
      }

      if (bias == unbiased)
      {
	sum_type scale = sum_type(1);

	if (r < row_shift)     scale *= sum_type(r+ (ref_rows-row_shift));
	else if (r >= in_rows - row_edge)
                               scale *= sum_type(in_rows + row_shift - r);
	else                   scale *= sum_type(ref_rows);

	if (c < col_shift)     scale *= sum_type(c+ (ref_cols-col_shift));
	else if (c >= in_cols - col_edge)
                               scale *= sum_type(in_cols + col_shift - c);
	else                   scale *= sum_type(ref_cols);

	sum /= scale;
      }
      
      out[r * out_row_stride + c * out_col_stride] = sum;
    }
  }
}



/// Perform 2-D correlation with full region of support.

template <typename T>
inline void
corr_full(bias_type   bias,
	  T const *ref,
	  length_type ref_rows,		// Mr
	  length_type ref_cols,		// Mc
	  stride_type ref_row_stride,
	  stride_type ref_col_stride,
	  T const *in,
	  length_type in_rows,		// Nr
	  length_type in_cols,		// Nc
	  stride_type in_row_stride,
	  stride_type in_col_stride,
	  T *out,
	  length_type out_rows,		// Pr
	  length_type out_cols,		// Pc
	  stride_type out_row_stride,
	  stride_type out_col_stride)
{
  corr_base(bias,
	    ref, ref_rows, ref_cols, ref_row_stride, ref_col_stride,
	    ref_rows-1, ref_cols-1, 0, 0,
	    in, in_rows, in_cols, in_row_stride, in_col_stride,
	    out, out_rows, out_cols, out_row_stride, out_col_stride);
}



/// Perform 2-D correlation with same region of support.

template <typename T>
inline void
corr_same(bias_type   bias,
	  T const *ref,
	  length_type ref_rows,		// Mr
	  length_type ref_cols,		// Mc
	  stride_type ref_row_stride,
	  stride_type ref_col_stride,
	  T const *in,
	  length_type in_rows,		// Nr
	  length_type in_cols,		// Nc
	  stride_type in_row_stride,
	  stride_type in_col_stride,
	  T *out,
	  length_type out_rows,		// Pr
	  length_type out_cols,		// Pc
	  stride_type out_row_stride,
	  stride_type out_col_stride)
{
  corr_base(bias,
	    ref, ref_rows, ref_cols, ref_row_stride, ref_col_stride,
	    ref_rows/2, ref_cols/2, ref_rows/2, ref_cols/2,
	    in, in_rows, in_cols, in_row_stride, in_col_stride,
	    out, out_rows, out_cols, out_row_stride, out_col_stride);
}



/// Perform 2-D correlation with minimal region of support.

template <typename T>
inline void
corr_min(bias_type   bias,
	 T const *ref,
	 length_type ref_rows,		// Mr
	 length_type ref_cols,		// Mc
	 stride_type ref_row_stride,
	 stride_type ref_col_stride,
	 T const *in,
	 length_type in_rows,		// Nr
	 length_type in_cols,		// Nc
	 stride_type in_row_stride,
	 stride_type in_col_stride,
	 T *out,
	 length_type out_rows,		// Pr
	 length_type out_cols,		// Pc
	 stride_type out_row_stride,
	 stride_type out_col_stride)
{
  corr_base(bias,
	    ref, ref_rows, ref_cols, ref_row_stride, ref_col_stride,
	    0, 0, 0, 0,
	    in, in_rows, in_cols, in_row_stride, in_col_stride,
	    out, out_rows, out_cols, out_row_stride, out_col_stride);
}


/***********************************************************************
  2-D Definitions (split)
***********************************************************************/

/// Perform 2-D correlation with full region of support.

template <typename T>
inline void
corr_base(bias_type   bias,
	  std::pair<T const *, T const *> ref,
	  length_type ref_rows,		// Mr
	  length_type ref_cols,		// Mc
	  stride_type ref_row_stride,
	  stride_type ref_col_stride,
	  length_type row_shift,
	  length_type col_shift,
	  length_type row_edge,
	  length_type col_edge,
	  std::pair<T const *, T const *> in,
	  length_type in_rows,		// Nr
	  length_type in_cols,		// Nc
	  stride_type in_row_stride,
	  stride_type in_col_stride,
	  std::pair<T*, T*> out,
	  length_type out_rows,		// Pr
	  length_type out_cols,		// Pc
	  stride_type out_row_stride,
	  stride_type out_col_stride)
{
  assert(ref_rows <= in_rows);
  assert(ref_cols <= in_cols);

  typedef typename Correlation_accum_trait<complex<T> >::sum_type sum_type;
  typedef Storage<split_complex, complex<T> > storage_type;

  for (index_type r=0; r<out_rows; ++r)
  {
    for (index_type c=0; c<out_cols; ++c)
    {
      sum_type sum   = sum_type();
      
      for (index_type rr=0; rr<ref_rows; ++rr)
      {
	for (index_type cc=0; cc<ref_cols; ++cc)
	{
	  index_type rpos = r + rr - row_shift;
	  index_type cpos = c + cc - col_shift;

	  if (r+rr >= row_shift && r+rr < in_rows+row_shift &&
	      c+cc >= col_shift && c+cc < in_cols+col_shift)
	  {
	    sum += storage_type::get(ref,rr * ref_row_stride + cc * ref_col_stride) *
                   impl_conj(storage_type::get(in,rpos * in_row_stride + cpos * in_col_stride));
	  }
	}
      }

      if (bias == unbiased)
      {
	sum_type scale = sum_type(1);

	if (r < row_shift)     scale *= sum_type(r+ (ref_rows-row_shift));
	else if (r >= in_rows - row_edge)
                               scale *= sum_type(in_rows + row_shift - r);
	else                   scale *= sum_type(ref_rows);

	if (c < col_shift)     scale *= sum_type(c+ (ref_cols-col_shift));
	else if (c >= in_cols - col_edge)
                               scale *= sum_type(in_cols + col_shift - c);
	else                   scale *= sum_type(ref_cols);

	sum /= scale;
      }
      
      storage_type::put(out,r * out_row_stride + c * out_col_stride,sum);
    }
  }
}


/// Perform 2-D correlation with minimal region of support.

template <typename T>
inline void
corr_min(bias_type   bias,
	 std::pair<T const *, T const *> ref,
	 length_type ref_rows,		// Mr
	 length_type ref_cols,		// Mc
	 stride_type ref_row_stride,
	 stride_type ref_col_stride,
	 std::pair<T const *, T const *> in,
	 length_type in_rows,		// Nr
	 length_type in_cols,		// Nc
	 stride_type in_row_stride,
	 stride_type in_col_stride,
	 std::pair<T*, T*>          out,
	 length_type out_rows,		// Pr
	 length_type out_cols,		// Pc
	 stride_type out_row_stride,
	 stride_type out_col_stride)
{
  corr_base(bias,
	    ref, ref_rows, ref_cols, ref_row_stride, ref_col_stride,
	    0, 0, 0, 0,
	    in, in_rows, in_cols, in_row_stride, in_col_stride,
	    out, out_rows, out_cols, out_row_stride, out_col_stride);
}



} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_CORE_SIGNAL_CORR_COMMON_HPP
