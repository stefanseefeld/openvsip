/* Copyright (c) 2006, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    matlab_utils.hpp
    @author  Don McCoy
    @date    2006-10-31
    @brief   VSIPL++ CodeSourcery Library: Matlab-like utility functions.
*/

#ifndef VSIP_CSL_MATLAB_UTILS_HPP
#define VSIP_CSL_MATLAB_UTILS_HPP

#include <cassert>

#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/block_traits.hpp>

#ifndef VSIP_IMPL_REF_IMPL
# include <vsip/opt/dispatch.hpp>
# if VSIP_IMPL_HAVE_CUDA
#  include <vsip/opt/cuda/eval_freqswap.hpp>
# endif
#endif

namespace vsip_csl
{

namespace matlab
{
  
// Vector fftshift

template <typename T1,
          typename T2,
          typename Block1,
          typename Block2>
vsip::Vector<T2, Block2>
fftshift(
  vsip::const_Vector<T1, Block1> in, vsip::Vector<T2, Block2> out)
{
  using vsip::length_type;
  using vsip::Domain;

  // This function swaps halves of a vector (dimension
  // must be even).

  length_type nx = in.size(0);
  assert(!(nx & 1));
  assert(nx == out.size(0));

  Domain<1> left(0, 1, nx/2);
  Domain<1> right(nx/2, 1, nx/2);

  out(left) = in(right);
  out(right) = in(left);

  return out;
}



// Matrix fftshift

template <typename T1,
	  typename T2,
	  typename Block1,
	  typename Block2>
vsip::Matrix<T2, Block2>
fftshift(
  vsip::const_Matrix<T1, Block1> in, vsip::Matrix<T2, Block2> out)
{
#if VSIP_IMPL_HAVE_CUDA
  typedef typename dispatcher::Evaluator<
      dispatcher::op::freqswap,
      dispatcher::be::cuda,
      void(Block1 const&, Block2&)>  cuda_evaluator_type;

  if (cuda_evaluator_type::ct_valid &&
      cuda_evaluator_type::rt_valid(in.block(), out.block()))
  {
    cuda_evaluator_type::exec(in.block(), out.block());
  }
  else
#endif
  {
  using vsip::length_type;
  using vsip::Domain;

  // This function swaps quadrants of a matrix (both dimensions
  // must be even) as follows:
  //
  //  | 1  2 |            | 4  3 |
  //  | 3  4 |   becomes  | 1  2 |

  length_type nx = in.size(0);
  length_type ny = in.size(1);
  assert(!(nx & 1));
  assert(!(ny & 1));
  assert(nx == out.size(0));
  assert(ny == out.size(1));

  Domain<1> left(0, 1, nx/2);
  Domain<1> right(nx/2, 1, nx/2);
  Domain<1> upper(0, 1, ny/2);
  Domain<1> lower(ny/2, 1, ny/2);

  Domain<2> dom1(left, upper);
  Domain<2> dom2(right, upper);
  Domain<2> dom3(left, lower);
  Domain<2> dom4(right, lower);

  out(dom1) = in(dom4);
  out(dom2) = in(dom3);
  out(dom3) = in(dom2);
  out(dom4) = in(dom1);
  }

  return out;
}



// By-value vector fftshift.
//
// May not as efficient as by-reference due to the overhead of
// creating a new view.

template <typename T1,
          typename Block1>
vsip::Vector<T1>
fftshift(
  vsip::const_Vector<T1, Block1> in)
{
  vsip::Vector<T1> out(in.size(0));
  return fftshift(in, out);
}



// By-value matrix fftshift.
//
// May not as efficient as by-reference due to the overhead of
// creating a new view.

template <typename T1,
	  typename Block1>
vsip::Matrix<T1>
fftshift(
  vsip::const_Matrix<T1, Block1> in)
{
  vsip::Matrix<T1> out(in.size(0), in.size(1));
  return fftshift(in, out);
}



// Partial matrix fftshift across dimension 0 (along columns)
//
// This function swaps halves of a matrix (both dimensions
// must be even) as follows:
//
//  | 1  |            | 2 |
//  | 2  |   becomes  | 1 |

template <typename T1,
	  typename T2,
	  typename Block1,
	  typename Block2>
vsip::Matrix<T2, Block2>
fftshift_col(
  vsip::const_Matrix<T1, Block1> in, vsip::Matrix<T2, Block2> out)
{
  using vsip::length_type;
  using vsip::Domain;

  length_type nx = in.size(0);
  length_type ny = in.size(1);
  assert(!(nx & 1));
  assert(!(ny & 1));
  assert(nx == out.size(0));
  assert(ny == out.size(1));

  Domain<1> upper(0, 1, nx/2);
  Domain<1> lower(nx/2, 1, nx/2);
  Domain<1> all(0, 1, ny);

  Domain<2> dom1(upper, all);
  Domain<2> dom2(lower, all);

  out(dom1) = in(dom2);
  out(dom2) = in(dom1);

  return out;
}



// Partial matrix fftshift across dimension 1 (along rows)
//
// This function swaps halves of a matrix (both dimensions
// must be even) as follows:
//
//  | 1  2 |   becomes  | 1  2 |

template <typename T1,
	  typename T2,
	  typename Block1,
	  typename Block2>
vsip::Matrix<T2, Block2>
fftshift_row(
  vsip::const_Matrix<T1, Block1> in, vsip::Matrix<T2, Block2> out)
{
  using vsip::length_type;
  using vsip::Domain;

  length_type nx = in.size(0);
  length_type ny = in.size(1);
  assert(!(nx & 1));
  assert(!(ny & 1));
  assert(nx == out.size(0));
  assert(ny == out.size(1));

  Domain<1> all(0, 1, nx);

  Domain<1> left (0, 1, ny/2);
  Domain<1> right(ny/2, 1, ny/2);

  Domain<2> dom1(all,  left);
  Domain<2> dom2(all, right);

  out(dom1) = in(dom2);
  out(dom2) = in(dom1);

  return out;
}



template <vsip::dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
vsip::Matrix<T2, Block2>
fftshift(
  vsip::const_Matrix<T1, Block1> in, vsip::Matrix<T2, Block2> out)
{
  return (Dim == vsip::row) ? fftshift_row(in, out) : fftshift_col(in, out);
}



// Frequency domain matrix fftshift

template <typename Matrix1T,
	  typename Matrix2T>
void
fd_fftshift(Matrix1T in, Matrix2T out)
{
  using namespace vsip;

  typedef typename Matrix1T::value_type T;
  typedef typename Matrix2T::block_type block2_type;
  typedef typename Matrix1T::block_type::map_type map1_type;

  length_type rows = in.local().size(0);
  length_type cols = in.local().size(1);

  Matrix<T> w(rows, cols);

  index_type g_offset0 = global_from_local_index(out, 0, 0);
  index_type g_offset1 = global_from_local_index(out, 1, 0);
  
  w = T(+1);
  if (g_offset0 % 2 == 0 && g_offset1 % 2 == 0)
  {
    w(Domain<2>(Domain<1>(0, 2, rows/2), Domain<1>(1, 2, cols/2))) = T(-1);
    w(Domain<2>(Domain<1>(1, 2, rows/2), Domain<1>(0, 2, cols/2))) = T(-1);
  }
  else
  {
    w(Domain<2>(Domain<1>(0, 2, rows/2), Domain<1>(0, 2, cols/2))) = T(-1);
    w(Domain<2>(Domain<1>(1, 2, rows/2), Domain<1>(1, 2, cols/2))) = T(-1);
  }

  out.local() = in.local() * w;
}



// Partial frequency domain matrix fftshift across dimension 0 (along columns)

template <typename Matrix1T,
	  typename Matrix2T>
void
fd_fftshift_col(Matrix1T in, Matrix2T out)
{
  using namespace vsip;

  typedef typename Matrix1T::value_type T;
  typedef typename Matrix2T::block_type block2_type;
  typedef typename Matrix1T::block_type::map_type map1_type;

  assert((vsip::impl::Is_par_same_map<2, map1_type, block2_type>
	  ::value(in.block().map(), out.block())));

  length_type rows = in.local().size(0);
  length_type cols = in.local().size(1);

  Matrix<T> w(rows, cols);

  index_type g_offset = global_from_local_index(out, 0, 0);
  index_type start    = 1 - (g_offset % 2);
  
  w = T(+1);
  w(Domain<2>(Domain<1>(start, 2, rows/2), Domain<1>(0, 1, cols))) = T(-1);

  out.local() = in.local() * w;
}



// Partial frequency domain matrix fftshift across dimension 1 (along rows)
// 
// Input and output matrices must have same map.

template <typename Matrix1T,
	  typename Matrix2T>
void
fd_fftshift_row(Matrix1T in, Matrix2T out)
{
  using namespace vsip;

  typedef typename Matrix1T::value_type T;
  typedef typename Matrix2T::block_type block2_type;
  typedef typename Matrix1T::block_type::map_type map1_type;

  assert((vsip::impl::Is_par_same_map<2, map1_type, block2_type>
	  ::value(in.block().map(), out.block())));

  length_type rows = in.local().size(0);
  length_type cols = in.local().size(1);

  Matrix<T> w(rows, cols);

  index_type g_offset = global_from_local_index(out, 1, 0);
  index_type start    = 1 - (g_offset % 2);
  
  w = T(+1);
  w(Domain<2>(Domain<1>(0, 1, rows), Domain<1>(start, 2, cols/2))) = T(-1);

  out.local() = in.local() * w;
}



template <vsip::dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
fd_fftshift(
  vsip::const_Matrix<T1, Block1> in, vsip::Matrix<T2, Block2> out)
{
  return (Dim == vsip::row) ? fd_fftshift_row(in, out) : fd_fftshift_col(in, out);
}




} // namesapce matlab

} // namespace vsip_csl

#endif // VSIP_CSL_MATLAB_UTILS_HPP
