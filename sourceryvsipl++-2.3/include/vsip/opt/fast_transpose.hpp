/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/fast-transpose.hpp
    @author  Jules Bergmann
    @date    2005-05-10
    @brief   VSIPL++ Library: Fast matrix tranpose algorithms.

*/

#ifndef VSIP_IMPL_FAST_TRANSPOSE_HPP
#define VSIP_IMPL_FAST_TRANSPOSE_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Includes & Macros
***********************************************************************/

#if   __3dNOW__
#  define VSIP_IMPL_TRANSPOSE_USE_3DNOW 1
#  include <mmintrin.h>
#elif __SSE2__
#  define VSIP_IMPL_TRANSPOSE_USE_SSE   1
#  include <xmmintrin.h>
#endif

#include <vsip/support.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace trans_detail
{

inline void
transpose_simd_start()
{
#if VSIP_IMPL_TRANSPOSE_USE_3DNOW
  __asm__ __volatile__ ("femms");
#endif
}

inline void
transpose_simd_stop()
{
#if VSIP_IMPL_TRANSPOSE_USE_3DNOW
  __asm__ __volatile__ ("femms");
#endif
}



/***********************************************************************
  General Unrolled_transpose definitions
***********************************************************************/

template <typename    T1,
	  typename    T2,
	  length_type Block,
	  bool        SimdAligned>
struct Unrolled_transpose
{
  static void exec(
    T1*               dst,
    T2 const*         src,
    stride_type const dst_col_stride,
    stride_type const src_row_stride)
  {
    typedef Unrolled_transpose<T1, T2, Block/2, SimdAligned> meta;

    meta::exec(dst, src,
	       dst_col_stride,
	       src_row_stride);
    meta::exec(dst + (Block/2)*dst_col_stride, src + (Block/2),
	       dst_col_stride,
	       src_row_stride);
    meta::exec(dst + (Block/2), src + (Block/2)*src_row_stride,
	       dst_col_stride,
	       src_row_stride);
    meta::exec(dst + (Block/2) + (Block/2)*dst_col_stride,
	       src + (Block/2)*src_row_stride + (Block/2),
	       dst_col_stride,
	       src_row_stride);
  }
};



template <typename T1,
	  typename T2,
	  bool     SimdAligned>
struct Unrolled_transpose<T1, T2, 1, SimdAligned>
{
  static void exec(
    T1*               dst,
    T2 const*         src,
    stride_type const /*dst_col_stride*/,
    stride_type const /*src_row_stride*/)
  {
    *dst = *src;
  }
};



/***********************************************************************
  SSE specific Unrolled_transpose definitions
***********************************************************************/

#if VSIP_IMPL_TRANSPOSE_USE_SSE
// 4x4 float unaligned fragment for SSE.

template <>
struct Unrolled_transpose<float, float, 4, false>
{
  static void exec(
    float*            dst,
    float const*      src,
    stride_type const dst_col_stride,
    stride_type const src_row_stride)
  {
    __m128 row0 = _mm_loadu_ps(src + 0*src_row_stride + 0);
    __m128 row1 = _mm_loadu_ps(src + 1*src_row_stride + 0);
    __m128 row2 = _mm_loadu_ps(src + 2*src_row_stride + 0);
    __m128 row3 = _mm_loadu_ps(src + 3*src_row_stride + 0);
    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
    _mm_storeu_ps(dst + 0 + 0*dst_col_stride, row0);
    _mm_storeu_ps(dst + 0 + 1*dst_col_stride, row1);
    _mm_storeu_ps(dst + 0 + 2*dst_col_stride, row2);
    _mm_storeu_ps(dst + 0 + 3*dst_col_stride, row3);
  }
};



// 4x4 float aligned fragment for SSE.

template <>
struct Unrolled_transpose<float, float, 4, true>
{
  static void exec(
    float*            dst,
    float const*      src,
    stride_type const dst_col_stride,
    stride_type const src_row_stride)
  {
    __m128 row0 = _mm_load_ps(src + 0*src_row_stride + 0);
    __m128 row1 = _mm_load_ps(src + 1*src_row_stride + 0);
    __m128 row2 = _mm_load_ps(src + 2*src_row_stride + 0);
    __m128 row3 = _mm_load_ps(src + 3*src_row_stride + 0);
    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
    _mm_store_ps(dst + 0 + 0*dst_col_stride, row0);
    _mm_store_ps(dst + 0 + 1*dst_col_stride, row1);
    _mm_store_ps(dst + 0 + 2*dst_col_stride, row2);
    _mm_store_ps(dst + 0 + 3*dst_col_stride, row3);
  }
};



// 2x2 complex<float> unaligned fragment for SSE.

template <>
struct Unrolled_transpose<complex<float>, complex<float>, 2, false>
{
  static void exec(
    complex<float>*       dst,
    complex<float>const * src,
    stride_type const     dst_col_stride,
    stride_type const     src_row_stride)
  {
    __m128 row0 = _mm_loadu_ps(
		reinterpret_cast<float const*>(src + 0*src_row_stride + 0));
    __m128 row1 = _mm_loadu_ps(
		reinterpret_cast<float const*>(src + 1*src_row_stride + 0));

    __m128 col0 = _mm_shuffle_ps(row0, row1, 0x44); // 10 00 01 00
    __m128 col1 = _mm_shuffle_ps(row0, row1, 0xEE); // 11 10 11 10

    _mm_storeu_ps(reinterpret_cast<float*>(dst + 0 + 0*dst_col_stride), col0);
    _mm_storeu_ps(reinterpret_cast<float*>(dst + 0 + 1*dst_col_stride), col1);
  }
};



// 2x2 complex<float> aligned fragment for SSE.

template <>
struct Unrolled_transpose<complex<float>, complex<float>, 2, true>
{
  static void exec(
    complex<float>*       dst,
    complex<float>const * src,
    stride_type const     dst_col_stride,
    stride_type const     src_row_stride)
  {
    __m128 row0 = _mm_load_ps(
		reinterpret_cast<float const*>(src + 0*src_row_stride + 0));
    __m128 row1 = _mm_load_ps(
		reinterpret_cast<float const*>(src + 1*src_row_stride + 0));

    __m128 col0 = _mm_shuffle_ps(row0, row1, 0x44); // 10 00 01 00
    __m128 col1 = _mm_shuffle_ps(row0, row1, 0xEE); // 11 10 11 10

    _mm_store_ps(reinterpret_cast<float*>(dst + 0 + 0*dst_col_stride), col0);
    _mm_store_ps(reinterpret_cast<float*>(dst + 0 + 1*dst_col_stride), col1);
  }
};
#endif



/***********************************************************************
  3DNow! specific Unrolled_transpose definitions
***********************************************************************/

#if VSIP_IMPL_TRANSPOSE_USE_3DNOW

typedef float __v2sf __attribute__ ((__mode__ (__V2SF__),__aligned__(8)));

template <bool     SimdAligned>
struct Unrolled_transpose<float, float, 2, SimdAligned>
{
  static void exec(
    float*            dst,
    float const*      src,
    stride_type const dst_col_stride,
    stride_type const src_row_stride)
  {
    __v2sf row0 = *(__v2sf*)(src  + 0*src_row_stride + 0);
    __v2sf row1 = *(__v2sf*)(src  + 1*src_row_stride + 0);

    __v2sf col0 = (__v2sf)_m_punpckldq((__m64)row0, (__m64)row1);
    __v2sf col1 = (__v2sf)_m_punpckhdq((__m64)row0, (__m64)row1);

    *(__v2sf*)(dst + 0 + 0*dst_col_stride) = col0;
    *(__v2sf*)(dst + 0 + 1*dst_col_stride) = col1;
  }
};

template <bool     SimdAligned>
struct Unrolled_transpose<complex<float>, complex<float>, 1, SimdAligned>
{
  static void exec(
    complex<float>*         dst,
    complex<float> const*   src,
    stride_type const       /*dst_col_stride*/,
    stride_type const       /*src_row_stride*/)
  {
    __v2sf row0 = *(__v2sf*)(src);

    *(__v2sf*)(dst) = row0;
  }
};
#endif



/***********************************************************************
  Definitions - transpose_unit
***********************************************************************/

// transpose_unit implementation tags.

struct Impl_loop {};
template <length_type Block, bool SimdAligned> struct Impl_block_iter {};
template <length_type Block, bool SimdAligned> struct Impl_block_recur {};
template <length_type Block, bool SimdAligned> struct Impl_block_recur_helper
{};
struct Impl_recur {};

template <typename T1,
	  typename T2>
void
transpose_unit(
  T1*               dst,
  T2 const*         src,
  length_type const rows,		// dst rows
  length_type const cols,		// dst cols
  stride_type const dst_col_stride,
  stride_type const src_row_stride,
  Impl_loop)
{
  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
      dst[r+c*dst_col_stride] = src[r*src_row_stride+c];
}



#if VSIP_IMPL_TRANSPOSE_USE_3DNOW
template <>
inline void
transpose_unit(
  complex<float>*       dst,
  complex<float> const* src,
  length_type const     rows,		// dst rows
  length_type const     cols,		// dst cols
  stride_type const     dst_col_stride,
  stride_type const     src_row_stride,
  Impl_loop)
{
  __asm__ __volatile__ ("femms");
  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
    {
      __v2sf row0 = *(__v2sf*)(src + r*src_row_stride+c);
      *(__v2sf*)(dst + r+c*dst_col_stride) = row0;
    }
  __asm__ __volatile__ ("femms");
}
#endif



/// Blocked transpose, using iteration over blocks.

template <typename    T1,
	  typename    T2,
	  length_type Block,
	  bool        SimdAligned>
void
transpose_unit(
  T1*               dst,
  T2 const*         src,
  length_type const rows,		// dst rows
  length_type const cols,		// dst cols
  stride_type const dst_col_stride,
  stride_type const src_row_stride,
  Impl_block_iter<Block, SimdAligned>)
{
  length_type full_cols = cols - cols%Block;
  length_type full_rows = rows - rows%Block;

  // Transpose core of matrix using Unrolled_transpose a block
  // at a time.

  for (index_type r=0; r<full_rows; r += Block)
    for (index_type c=0; c<full_cols; c += Block)
    {
      transpose_simd_start();
      Unrolled_transpose<T1, T2, Block, SimdAligned>::exec(
			dst + r + c*dst_col_stride,
			src + r*src_row_stride + c,
			dst_col_stride,
			src_row_stride);
      transpose_simd_stop();
    }


  // Cleanup edges of matrix using Impl_loop.

  if (full_cols != cols)
  {
    length_type extra_cols = cols - full_cols;
    for (index_type r=0; r<full_rows; r += Block)
      transpose_unit(dst + r + full_cols*dst_col_stride,
		     src + r*src_row_stride + full_cols,
		     Block, extra_cols,
		     dst_col_stride,
		     src_row_stride,
		     Impl_loop());
  }
  if (full_rows != rows)
  {
    length_type extra_rows = rows - full_rows;
    for (index_type c=0; c<full_cols; c += Block)
      transpose_unit(dst + full_rows + c*dst_col_stride,
		     src + full_rows*src_row_stride + c,
		     extra_rows, Block,
		     dst_col_stride,
		     src_row_stride,
		     Impl_loop());
    if (full_cols != cols)
    {
      transpose_unit(dst + full_rows + full_cols*dst_col_stride,
		     src + full_rows*src_row_stride + full_cols,
		     extra_rows, cols - full_cols,
		     dst_col_stride,
		     src_row_stride,
		     Impl_loop());
    }
  }
}



/// Recurcive blocked transpose helper function.

/// This routine performs the recursive sub-division for Impl_block_recur.

template <typename    T1,
	  typename    T2,
	  length_type Block,
	  bool        SimdAligned>
void
transpose_unit(
  T1*               dst,
  T2 const*         src,
  length_type const rows,		// dst rows
  length_type const cols,		// dst cols
  stride_type const dst_col_stride,
  stride_type const src_row_stride,
  Impl_block_recur_helper<Block, SimdAligned>)
{
  length_type const thresh = 4*Block;
  if (rows <= thresh && cols <= thresh)
  {
    for (index_type r=0; r<rows; r+=Block)
      for (index_type c=0; c<cols; c+=Block)
	Unrolled_transpose<T1, T2, Block, SimdAligned>::exec(
			dst + r + c*dst_col_stride,
			src + r*src_row_stride + c,
			dst_col_stride,
			src_row_stride);
  }
  else if (cols >= rows)
  {
    length_type cols1 = ((cols/Block)/2) * Block;
    length_type cols2 = cols - cols1;

    transpose_unit(dst,          src,
		   rows, cols1,
		   dst_col_stride, src_row_stride,
		   Impl_block_recur_helper<Block, SimdAligned>());

    transpose_unit(dst + (cols1)*dst_col_stride, src + (cols1),
		   rows, cols2,
		   dst_col_stride, src_row_stride,
		   Impl_block_recur_helper<Block, SimdAligned>());
  }
  else
  {
    length_type rows1 = ((rows/Block)/2) * Block;
    length_type rows2 = rows - rows1;

    transpose_unit(dst,          src,
		   rows1, cols,
		   dst_col_stride, src_row_stride,
		   Impl_block_recur_helper<Block, SimdAligned>());

    transpose_unit(dst + (rows1), src + (rows1)*src_row_stride,
		   rows2, cols,
		   dst_col_stride, src_row_stride,
		   Impl_block_recur_helper<Block, SimdAligned>());
  }
}



/// Blocked transpose, using recursion over blocks.

template <typename    T1,
	  typename    T2,
	  length_type Block,
	  bool        SimdAligned>
void
transpose_unit(
  T1*            dst,
  T2 const*      src,
  length_type const rows,		// dst rows
  length_type const cols,		// dst cols
  stride_type const dst_col_stride,
  stride_type const src_row_stride,
  Impl_block_recur<Block, SimdAligned>)
{
  length_type full_cols = cols - cols%Block;
  length_type full_rows = rows - rows%Block;

  // Transpose core of matrix using Unrolled_transpose a block
  // at a time.

  transpose_simd_start();
  transpose_unit(dst, src, full_rows, full_cols,
		 dst_col_stride,
		 src_row_stride,
		 Impl_block_recur_helper<Block, SimdAligned>());
  transpose_simd_stop();


  // Cleanup edges of matrix using Impl_loop.

  if (full_cols != cols)
  {
    length_type extra_cols = cols - full_cols;
    for (index_type r=0; r<full_rows; r += Block)
      transpose_unit(dst + r + full_cols*dst_col_stride,
		     src + r*src_row_stride + full_cols,
		     Block, extra_cols,
		     dst_col_stride,
		     src_row_stride,
		     Impl_loop());
  }
  if (full_rows != rows)
  {
    length_type extra_rows = rows - full_rows;
    for (index_type c=0; c<full_cols; c += Block)
      transpose_unit(dst + full_rows + c*dst_col_stride,
		     src + full_rows*src_row_stride + c,
		     extra_rows, Block,
		     dst_col_stride,
		     src_row_stride,
		     Impl_loop());
    if (full_cols != cols)
    {
      transpose_unit(dst + full_rows + full_cols*dst_col_stride,
		     src + full_rows*src_row_stride + full_cols,
		     extra_rows, cols - full_cols,
		     dst_col_stride,
		     src_row_stride,
		     Impl_loop());
    }
  }
}



// Recurively decomposed transposition for unit-strides.

// Algorithm based on "Cache-Oblivious Algorithms (Extended Abstract)"
// by M. Frigo, C. Leiseron, H. Prokop, S. Ramachandran.
// citeseer.csail.mit.edu/307799.html.

template <typename T1,
	  typename T2>
void
transpose_unit(
  T1*               dst,
  T2 const*         src,
  length_type const rows,		// dst rows
  length_type const cols,		// dst cols
  stride_type const dst_col_stride,
  stride_type const src_row_stride,
  Impl_recur)
{
  length_type const thresh = 16;
  if (rows <= thresh && cols <= thresh)
  {
    for (index_type r=0; r<rows; ++r)
      for (index_type c=0; c<cols; ++c)
	dst[r+c*dst_col_stride] = src[r*src_row_stride+c];
  }
  else if (cols >= rows)
  {
    transpose_unit(dst,          src,
		   rows, cols/2,
		   dst_col_stride, src_row_stride);

    transpose_unit(dst + (cols/2)*dst_col_stride, src + (cols/2),
		   rows, cols/2 + cols%2,
		   dst_col_stride, src_row_stride);
  }
  else
  {
    transpose_unit(dst,          src,
		   rows/2, cols,
		   dst_col_stride, src_row_stride);

    transpose_unit(dst + (rows/2), src + (rows/2)*src_row_stride,
		   rows/2 + rows%2, cols,
		   dst_col_stride, src_row_stride);
  }
}

} // namespace vsip::impl::trans_detail


// Unit-stride transpose dispatch function.

template <typename T1,
	  typename T2>
inline void
transpose_unit(
  T1*            dst,
  T2 const*      src,
  length_type const rows,		// dst rows
  length_type const cols,		// dst cols
  stride_type const dst_col_stride,
  stride_type const src_row_stride)
{
  // Intel C++ 9.1 for Windows has trouble with transpose_unit()
  // when T1 and T2 are complex<>.  This appears to be a compiler
  // bug (see icl-transpose-assign.cpp).
  //
  // Empircally, it appears we can avoid this bug by forgoing
  // dispatch on block size and ignoring SIMD alignment.
  // (060914)
#if _WIN32
  // #if 0
  trans_detail::transpose_unit(dst, src, rows, cols,
			       dst_col_stride, src_row_stride,
			       trans_detail::Impl_block_recur<4, false>()
			       );
#else
  // Check if data is aligned for SSE ISA.
  //  - pointers need to be 16-byte aligned
  //  - rows/cols need to start with 16-byte alignment.
  bool aligned = 
    (((unsigned long)dst                      ) & 0xf) == 0 &&
    (((unsigned long)src                      ) & 0xf) == 0 &&
    (((unsigned long)dst_col_stride*sizeof(T1)) & 0xf) == 0 &&
    (((unsigned long)src_row_stride*sizeof(T1)) & 0xf) == 0;

  if (rows < 16 || cols < 16)
  {
    if (aligned)
      trans_detail::transpose_unit(dst, src, rows, cols,
				   dst_col_stride, src_row_stride,
				   trans_detail::Impl_block_recur<4, true>()
				   );
    else
      trans_detail::transpose_unit(dst, src, rows, cols,
				   dst_col_stride, src_row_stride,
				   trans_detail::Impl_block_recur<4, false>()
				   );
  }
  else
  {
    if (aligned)
      trans_detail::transpose_unit(dst, src, rows, cols,
				   dst_col_stride, src_row_stride,
				   trans_detail::Impl_block_recur<16, true>()
				   );
    else
      trans_detail::transpose_unit(dst, src, rows, cols,
				   dst_col_stride, src_row_stride,
				   trans_detail::Impl_block_recur<16, false>()
				   );
  }
#endif
}



template <typename T1,
	  typename T2>
inline void
transpose_unit(
  std::pair<T1*, T1*> const& dst,
  std::pair<T2*, T2*> const& src,
  length_type const          rows,		// dst rows
  length_type const          cols,		// dst cols
  stride_type const          dst_col_stride,
  stride_type const          src_row_stride)
{
  transpose_unit(dst.first, src.first,
		 rows, cols, dst_col_stride, src_row_stride);
  transpose_unit(dst.second, src.second,
		 rows, cols, dst_col_stride, src_row_stride);
}



// Transpose for matrices with arbitrary strides.

// Algorithm based on "Cache-Oblivious Algorithms (Extended Abstract)"
// by M. Frigo, C. Leiseron, H. Prokop, S. Ramachandran.
// citeseer.csail.mit.edu/307799.html.
template <typename T1,
	  typename T2>
void
transpose(
  T1*               dst,
  T2 const*         src,
  length_type const rows,		// dst rows
  length_type const cols,		// dst cols
  stride_type const dst_stride0,
  stride_type const dst_stride1,	// eq. to dst_col_stride
  stride_type const src_stride0,	// eq. to src_row_stride
  stride_type const src_stride1)
{
  length_type const thresh = 16;
  if (rows <= thresh && cols <= thresh)
  {
    for (index_type r=0; r<rows; ++r)
      for (index_type c=0; c<cols; ++c)
	dst[r*dst_stride0+c*dst_stride1] = src[r*src_stride0+c*src_stride1];
  }
  else if (cols >= rows)
  {
    transpose(dst,          src,
	      rows, cols/2,
	      dst_stride0, dst_stride1,
	      src_stride0, src_stride1);

    transpose(dst + (cols/2)*dst_stride1, src + (cols/2)*src_stride1,
	      rows, cols/2 + cols%2,
	      dst_stride0, dst_stride1,
	      src_stride0, src_stride1);
  }
  else
  {
    transpose(dst,          src,
	      rows/2, cols,
	      dst_stride0, dst_stride1,
	      src_stride0, src_stride1);

    transpose(dst + (rows/2)*dst_stride0, src + (rows/2)*src_stride0,
	      rows/2 + rows%2, cols,
	      dst_stride0, dst_stride1,
	      src_stride0, src_stride1);
  }
}

template <typename T1,
	  typename T2>
void
transpose(
  std::pair<T1*, T1*> const& dst,
  std::pair<T2*, T2*> const& src,
  length_type const          rows,		// dst rows
  length_type const          cols,		// dst cols
  stride_type const          dst_stride0,
  stride_type const          dst_stride1,
  stride_type const          src_stride0,
  stride_type const          src_stride1)
{
  transpose(dst.first, src.first,
	    rows, cols,
	    dst_stride0, dst_stride1,
	    src_stride0, src_stride1);
  transpose(dst.second, src.second,
	    rows, cols,
	    dst_stride0, dst_stride1,
	    src_stride0, src_stride1);
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_FAST_TRANSPOSE_HPP
