//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_signal_corr_hpp_
#define ovxx_signal_corr_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <ovxx/domain_utils.hpp>
#include <ovxx/aligned_array.hpp>
#include <vsip/impl/signal/types.hpp>

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

namespace ovxx
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
} // namespace ovxx::dispatcher::op
} // namespace ovxx::dispatcher

namespace signal
{
template <typename T>
struct corr_accum_trait
{
  typedef T sum_type;
};

/// Perform 1-D correlation with full region of support.
template <typename T>
inline void
corr_full(bias_type bias,
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
  OVXX_PRECONDITION(ref_size <= in_size);

  typedef typename corr_accum_trait<T>::sum_type sum_type;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum   = sum_type();
      
    for (index_type k=0; k<ref_size; ++k)
    {
      index_type pos = n + k - (ref_size-1);

      if (n+k >= (ref_size-1) && n+k < in_size+(ref_size-1))
      {
	sum += ref[k * ref_stride] * math::impl_conj(in[pos * in_stride]);
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
corr_same(bias_type bias,
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
  typedef typename corr_accum_trait<T>::sum_type sum_type;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum   = sum_type();
      
    for (index_type k=0; k<ref_size; ++k)
    {
      index_type pos = n + k - (ref_size/2);

      if (n+k >= (ref_size/2) && n+k <  in_size + (ref_size/2))
      {
	sum += ref[k * ref_stride] * math::impl_conj(in[pos * in_stride]);
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
corr_min(bias_type bias,
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
  typedef typename corr_accum_trait<T>::sum_type sum_type;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum = sum_type();
      
    for (index_type k=0; k<ref_size; ++k)
    {
      sum += ref[k*ref_stride] * math::impl_conj(in[(n+k) * in_stride]);
    }

    if (bias == unbiased)
      sum /= sum_type(ref_size);

    out[n * out_stride] = sum;
  }
}

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
  OVXX_PRECONDITION(ref_size <= in_size);

  typedef typename corr_accum_trait<std::complex<T> >::sum_type sum_type;
  typedef storage_traits<complex<T>, split_complex> storage;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum   = sum_type();
      
    for (index_type k=0; k<ref_size; ++k)
    {
      index_type pos = n + k - (ref_size-1);

      if (n+k >= (ref_size-1) && n+k < in_size+(ref_size-1))
      {
	sum = sum +
	  storage::get(ref, k * ref_stride) *
	  math::impl_conj(storage::get(in, pos * in_stride));
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
      
    storage::put(out, n * out_stride, sum);
  }
}

/// Perform 1-D correlation with same region of support.
template <typename T>
inline void
corr_same(bias_type bias,
	  std::pair<T const *, T const *> ref,
	  length_type ref_size,		// M
	  stride_type ref_stride,
	  std::pair<T const *, T const *> in,
	  length_type in_size,		// N
	  stride_type in_stride,
	  std::pair<T*, T*> out,
	  length_type out_size,		// P
	  stride_type out_stride)
{
  typedef typename corr_accum_trait<std::complex<T> >::sum_type sum_type;
  typedef storage_traits<complex<T>, split_complex> storage;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum   = sum_type();
      
    for (index_type k=0; k<ref_size; ++k)
    {
      index_type pos = n + k - (ref_size/2);

      if (n+k >= (ref_size/2) && n+k <  in_size + (ref_size/2))
      {
	sum = sum +
	  storage::get(ref, k * ref_stride) *
	  math::impl_conj(storage::get(in, pos * in_stride));
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
    storage::put(out, n * out_stride, sum);
  }
}

/// Perform correlation with minimal region of support.
template <typename T>
inline void
corr_min(bias_type bias,
	 std::pair<T const *, T const *> ref,
	 length_type ref_size,		// M
	 stride_type ref_stride,
	 std::pair<T const *, T const *> in,
	 length_type /*in_size*/,	// N
	 stride_type in_stride,
	 std::pair<T*, T*> out,
	 length_type out_size,		// P
	 stride_type out_stride)
{
  typedef typename corr_accum_trait<std::complex<T> >::sum_type sum_type;
  typedef storage_traits<complex<T>, split_complex> storage;

  for (index_type n=0; n<out_size; ++n)
  {
    sum_type sum = sum_type();
      
    for (index_type k=0; k<ref_size; ++k)
    {
      sum = sum +
	storage::get(ref, k*ref_stride) *
	math::impl_conj(storage::get(in, (n+k) * in_stride));
    }

    if (bias == unbiased)
      sum /= sum_type(ref_size);

    storage::put(out, n * out_stride, sum);
  }
}

/// Perform 2-D correlation with full region of support.
template <typename T>
inline void
corr_base(bias_type bias,
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
  OVXX_PRECONDITION(ref_rows <= in_rows);
  OVXX_PRECONDITION(ref_cols <= in_cols);

  typedef typename corr_accum_trait<T>::sum_type sum_type;

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
	      math::impl_conj(in[rpos * in_row_stride + cpos * in_col_stride]);
	  }
	}
      }

      if (bias == unbiased)
      {
	sum_type scale = sum_type(1);

	if (r < row_shift)
	  scale *= sum_type(r+ (ref_rows-row_shift));
	else if (r >= in_rows - row_edge)
	  scale *= sum_type(in_rows + row_shift - r);
	else
	  scale *= sum_type(ref_rows);
	if (c < col_shift)     
	  scale *= sum_type(c+ (ref_cols-col_shift));
	else if (c >= in_cols - col_edge)
	  scale *= sum_type(in_cols + col_shift - c);
	else
	  scale *= sum_type(ref_cols);
	sum /= scale;
      }
      out[r * out_row_stride + c * out_col_stride] = sum;
    }
  }
}

/// Perform 2-D correlation with full region of support.
template <typename T>
inline void
corr_full(bias_type bias,
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
corr_same(bias_type bias,
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
corr_min(bias_type bias,
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

/// Perform 2-D correlation with full region of support.
template <typename T>
inline void
corr_base(bias_type bias,
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
  OVXX_PRECONDITION(ref_rows <= in_rows);
  OVXX_PRECONDITION(ref_cols <= in_cols);

  typedef typename corr_accum_trait<complex<T> >::sum_type sum_type;
  typedef storage_traits<complex<T>, split_complex> storage;

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
	    sum += storage::get(ref,rr * ref_row_stride + cc * ref_col_stride) *
	      math::impl_conj(storage::get(in,rpos * in_row_stride + cpos * in_col_stride));
	  }
	}
      }

      if (bias == unbiased)
      {
	sum_type scale = sum_type(1);

	if (r < row_shift)
	  scale *= sum_type(r+ (ref_rows-row_shift));
	else if (r >= in_rows - row_edge)
	  scale *= sum_type(in_rows + row_shift - r);
	else
	  scale *= sum_type(ref_rows);
	if (c < col_shift)
	  scale *= sum_type(c+ (ref_cols-col_shift));
	else if (c >= in_cols - col_edge)
	  scale *= sum_type(in_cols + col_shift - c);
	else
	  scale *= sum_type(ref_cols);
	sum /= scale;
      }
      
      storage::put(out,r * out_row_stride + c * out_col_stride,sum);
    }
  }
}

/// Perform 2-D correlation with minimal region of support.
template <typename T>
inline void
corr_min(bias_type bias,
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

template <dimension_type      D,
	  support_region_type R,
	  typename            T,
	  unsigned            N,
          alg_hint_type       H>
class Correlation
{
  static dimension_type const dim = D;

public:
  static support_region_type const supprt  = R;

  Correlation(Domain<dim> const &ref_size,
              Domain<dim> const &input_size)
    VSIP_THROW((std::bad_alloc))
  : ref_size_(normalize(ref_size)),
    input_size_(normalize(input_size)),
    output_size_(conv_output_size(R, ref_size_, input_size_, 1)),
    in_buffer_(input_size_.size()),
    out_buffer_(output_size_.size()),
    ref_buffer_(ref_size_.size())
  {}
  Correlation(Correlation const&) VSIP_NOTHROW;
  Correlation& operator=(Correlation const&) VSIP_NOTHROW;
  ~Correlation() VSIP_NOTHROW {}

  Domain<dim> const &reference_size() const VSIP_NOTHROW { return ref_size_;}
  Domain<dim> const &input_size() const VSIP_NOTHROW { return input_size_;}
  Domain<dim> const &output_size() const VSIP_NOTHROW { return output_size_;}

  template <typename B1, typename B2, typename B3>
  void
  correlate(bias_type bias, const_Vector<T, B1> ref,
	    const_Vector<T, B2> in, Vector<T, B3> out)
    VSIP_NOTHROW;

  template <typename B1, typename B2, typename B3>
  void
  correlate(bias_type bias, const_Matrix<T, B1> ref,
	    const_Matrix<T, B2> in, Matrix<T, B3> out)
    VSIP_NOTHROW;

private:
  Domain<dim> ref_size_;
  Domain<dim> input_size_;
  Domain<dim> output_size_;

  aligned_array<T> in_buffer_;
  aligned_array<T> out_buffer_;
  aligned_array<T> ref_buffer_;
};

template <dimension_type      D,
	  support_region_type R,
	  typename            T,
	  unsigned            Nu,
          alg_hint_type       H>
template <typename B1, typename B2, typename B3>
void
Correlation<D, R, T, Nu, H>::correlate
(bias_type bias, const_Vector<T, B1> ref,
 const_Vector<T, B2> in, Vector<T, B3> out)
  VSIP_NOTHROW
{
  length_type const M = this->ref_size_[0].size();
  length_type const N = this->input_size_[0].size();
  length_type const P = this->output_size_[0].size();

  OVXX_PRECONDITION(M == ref.size());
  OVXX_PRECONDITION(N == in.size());
  OVXX_PRECONDITION(P == out.size());

  typedef typename get_block_layout<B1>::type L1;
  typedef typename get_block_layout<B2>::type L2;
  typedef typename get_block_layout<B3>::type L3;

  typedef Layout<1, any_type, any_packing, array> req_layout;

  typedef typename adjust_layout<req_layout, L1>::type use_l1;
  typedef typename adjust_layout<req_layout, L2>::type use_l2;
  typedef typename adjust_layout<req_layout, L3>::type use_l3;

  typedef dda::Data<B1, dda::in, use_l1> ref_data_type;
  typedef dda::Data<B2, dda::in, use_l2> in_data_type;
  typedef dda::Data<B3, dda::out, use_l3> out_data_type;

  ref_data_type ref_data(ref.block(), ref_buffer_.get());
  in_data_type in_data(in.block(), in_buffer_.get());
  out_data_type out_data(out.block(), out_buffer_.get());

  if (R == support_full)
  {
    corr_full<T>(bias, ref_data.ptr(), M, ref_data.stride(0),
		 in_data.ptr(), N, in_data.stride(0),
		 out_data.ptr(), P, out_data.stride(0));
  }
  else if (R == support_same)
  {
    corr_same<T>(bias, ref_data.ptr(), M, ref_data.stride(0),
		 in_data.ptr(), N, in_data.stride(0),
		 out_data.ptr(), P, out_data.stride(0));
  }
  else // (R == support_min)
  {
    corr_min<T>(bias, ref_data.ptr(), M, ref_data.stride(0),
		in_data.ptr(), N, in_data.stride(0),
		out_data.ptr(), P, out_data.stride(0));
  }
}

template <dimension_type      D,
	  support_region_type R,
	  typename            T,
	  unsigned            N,
          alg_hint_type       H>
template <typename B1, typename B2, typename B3>
void
Correlation<D, R, T, N, H>::correlate
(bias_type bias, const_Matrix<T, B1> ref,
 const_Matrix<T, B2> in, Matrix<T, B3> out)
  VSIP_NOTHROW
{
  length_type const Mr = this->ref_size_[0].size();
  length_type const Mc = this->ref_size_[1].size();
  length_type const Nr = this->input_size_[0].size();
  length_type const Nc = this->input_size_[1].size();
  length_type const Pr = this->output_size_[0].size();
  length_type const Pc = this->output_size_[1].size();

  OVXX_PRECONDITION(Mr == ref.size(0));
  OVXX_PRECONDITION(Mc == ref.size(1));
  OVXX_PRECONDITION(Nr == in.size(0));
  OVXX_PRECONDITION(Nc == in.size(1));
  OVXX_PRECONDITION(Pr == out.size(0));
  OVXX_PRECONDITION(Pc == out.size(1));

  typedef typename get_block_layout<B1>::type L1;
  typedef typename get_block_layout<B2>::type L2;
  typedef typename get_block_layout<B3>::type L3;

  typedef Layout<2, any_type, any_packing, array> req_layout;

  typedef typename adjust_layout<req_layout, L1>::type use_l1;
  typedef typename adjust_layout<req_layout, L2>::type use_l2;
  typedef typename adjust_layout<req_layout, L3>::type use_l3;

  typedef dda::Data<B1, dda::in, use_l1> ref_data_type;
  typedef dda::Data<B2, dda::in, use_l2> in_data_type;
  typedef dda::Data<B3, dda::out, use_l3> out_data_type;

  ref_data_type ref_data(ref.block(), ref_buffer_.get());
  in_data_type in_data(in.block(), in_buffer_.get());
  out_data_type out_data(out.block(), out_buffer_.get());

  if (R == support_full)
  {
    corr_full<T>(bias,
		 ref_data.ptr(), Mr, Mc, ref_data.stride(0), ref_data.stride(1),
		 in_data.ptr(), Nr, Nc, in_data.stride(0), in_data.stride(1),
		 out_data.ptr(), Pr, Pc, out_data.stride(0), out_data.stride(1));
  }
  else if (R == support_same)
  {
    corr_same<T>(bias,
		 ref_data.ptr(), Mr, Mc, ref_data.stride(0), ref_data.stride(1),
		 in_data.ptr(), Nr, Nc, in_data.stride(0), in_data.stride(1),
		 out_data.ptr(), Pr, Pc, out_data.stride(0), out_data.stride(1));
  }
  else // (R == support_min)
  {
    corr_min<T>(bias,
		ref_data.ptr(), Mr, Mc, ref_data.stride(0), ref_data.stride(1),
		in_data.ptr(), Nr, Nc, in_data.stride(0), in_data.stride(1),
		out_data.ptr(), Pr, Pc, out_data.stride(0), out_data.stride(1));
  }
}

} // namespace ovxx::signal

namespace dispatcher
{
template <dimension_type      D,
          support_region_type R,
          typename            T,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<D, R, T, N, H>, be::generic>
{
  static bool const ct_valid = true;
  typedef signal::Correlation<D, R, T, N, H> backend_type;
};
} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
