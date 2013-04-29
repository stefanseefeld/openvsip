/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <vsip/core/config.hpp>
#include <vsip/math.hpp>
#include <vsip/opt/cbe/reduction_params.h>
#include <vsip/opt/cbe/ppu/reductions.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>
#include <memory>

namespace vsip
{
namespace impl
{
namespace cbe
{

complex<float>
partial_sum(char const *code_ea,
	    int code_size,
	    std::pair<float const *, float const *> const& A,
	    length_type len,
	    bool sq,
	    length_type &leftover)
{
  length_type const chunk_size = 1024;
  // DMA granularity is 16 bytes for sizes 16 bytes or larger.
  length_type const granularity = VSIP_IMPL_CBE_DMA_GRANULARITY / sizeof(float);

  assert(is_dma_addr_ok(A.first) && is_dma_addr_ok(A.second));

  Task_manager *mgr = Task_manager::instance();

  std::auto_ptr<lwp::Task> task = 
    mgr->reserve_lwp_task(VSIP_IMPL_REDUCTION_BUFFER_SIZE, 2,
			  (uintptr_t)code_ea, code_size);

  assert(sizeof(float)*4*chunk_size <= VSIP_IMPL_REDUCTION_BUFFER_SIZE);
  assert(8 < VSIP_IMPL_REDUCTION_DTL_SIZE);

  Reduction_split_params params;
  params.cmd         = sq ? ZSUMSQ : ZSUM;
  params.length      = chunk_size;
  length_type chunks = len / chunk_size;
  length_type spes   = mgr->num_spes();
  length_type chunks_per_spe = chunks / spes;
  assert(chunks_per_spe * spes <= chunks);

  // Make room for all workblock iterations to store
  // results in.
  // DMA alignment restrictions require a stride of 2.
  aligned_array<complex<float> > R(2 * (chunks + 1));
  // Only initialize the last, the others are guaranteed
  // to be set via DMA.
  R[2* chunks] = 0.;
    
  float const *re_ptr = A.first;
  float const *im_ptr = A.second;
  complex<float> *r_ptr = R.get();

  for (index_type i=0; i<spes && i<chunks; ++i)
  {
    // If chunks don't divide evenly, give the first SPEs one extra.
    length_type my_chunks = (i < chunks % spes) ? chunks_per_spe + 1
      : chunks_per_spe;
      
    lwp::Workblock block = task->create_workblock(my_chunks);
    params.re_ptr = (uintptr_t)re_ptr;
    params.im_ptr = (uintptr_t)im_ptr;
    params.r_ptr = (uintptr_t)r_ptr;
    block.set_parameters(params);
    block.enqueue();

    re_ptr += my_chunks*chunk_size;
    im_ptr += my_chunks*chunk_size;
    // Advance by 16 bytes, which means stride=2 for 
    // complex<float> elements.
    r_ptr += my_chunks*2;
    len -= my_chunks * chunk_size;
  }

  // Cleanup leftover data that doesn't fit into a full chunk.

  // First, handle data that can be DMA'd to the SPEs.
  if (len >= granularity)
  {
    params.length = (len / granularity) * granularity;
    assert(is_dma_size_ok(params.length*sizeof(float)));
    lwp::Workblock block = task->create_workblock(1);
    params.re_ptr = (uintptr_t)re_ptr;
    params.im_ptr = (uintptr_t)im_ptr;
    params.r_ptr = (uintptr_t)r_ptr;
    block.set_parameters(params);
    block.enqueue();
    len -= params.length;
  }
  // Wait for all partial sums...
  task->sync();
  // ...and accumulate them.
  complex<float> result = 0;
  for (unsigned int i = 0; i != R.size(); i += 2)
    result += R[i];
  leftover = len;
  return result;
}

template <typename T> T const &extract(complex<float> const &);
template <> 
float const &extract<float>(complex<float> const &v) { return v.real();}
template <> 
complex<float> const &extract<complex<float> >(complex<float> const &v) { return v;}

template <typename T>
T partial_sum(char const *code_ea, int code_size,
	      T const *A, length_type len,
	      bool sq,
	      length_type &leftover)
{
  length_type const chunk_size = 1024;
  // DMA granularity is 16 bytes for sizes 16 bytes or larger.
  length_type const granularity = VSIP_IMPL_CBE_DMA_GRANULARITY / sizeof(T);

  assert(is_dma_addr_ok(A));

  Task_manager *mgr = Task_manager::instance();

  std::auto_ptr<lwp::Task> task = 
    mgr->reserve_lwp_task(VSIP_IMPL_REDUCTION_BUFFER_SIZE, 2,
			  (uintptr_t)code_ea, code_size);

  assert(sizeof(T)*2*chunk_size <= VSIP_IMPL_REDUCTION_BUFFER_SIZE);

  Reduction_params params;
  if (is_same<T, float>::value)
    params.cmd       = sq ? SUMSQ : SUM;
  else
    params.cmd       = sq ? CSUMSQ : CSUM;
  params.length      = chunk_size;
  length_type chunks = len / chunk_size;
  length_type spes   = mgr->num_spes();
  length_type chunks_per_spe = chunks / spes;
  assert(chunks_per_spe * spes <= chunks);

  // Make room for all workblock iterations to store
  // results in.
  // This is used for float as well as complex<float>,
  // but the DMA alignment requirements make it easier to
  // treat the return values as complex<float>, and simply ignore
  // the imaginary part in the float case.
  aligned_array<complex<float> > R(2 * (chunks + 1));
  // Only initialize the last, the others are guaranteed
  // to be set via DMA.
  R[2* chunks] = 0.;
    
  T const* a_ptr = A;
  complex<float> *r_ptr = R.get();

  for (index_type i=0; i<spes && i<chunks; ++i)
  {
    // If chunks don't divide evenly, give the first SPEs one extra.
    length_type my_chunks = (i < chunks % spes) ? chunks_per_spe + 1
      : chunks_per_spe;
      
    lwp::Workblock block = task->create_workblock(my_chunks);
    params.a_ptr = (uintptr_t)a_ptr;
    params.r_ptr = (uintptr_t)r_ptr;
    block.set_parameters(params);
    block.enqueue();

    a_ptr += my_chunks*chunk_size;
    // Advance by 16 bytes, which means stride=2 for 
    // complex<float> elements.
    r_ptr += my_chunks*2;
    len -= my_chunks * chunk_size;
  }

  // Cleanup leftover data that doesn't fit into a full chunk.

  // First, handle data that can be DMA'd to the SPEs.
    
  if (len >= granularity)
  {
    params.length = (len / granularity) * granularity;
    assert(is_dma_size_ok(params.length*sizeof(T)));
    lwp::Workblock block = task->create_workblock(1);
    params.a_ptr = (uintptr_t)a_ptr;
    params.r_ptr = (uintptr_t)r_ptr;
    block.set_parameters(params);
    block.enqueue();
    len -= params.length;
  }
  // Wait for all partial sums...
  task->sync();
  // ...and accumulate them.
  T result(0);
  for (unsigned int i = 0; i != R.size(); i += 2)
    result += extract<T>(R[i]);
  leftover = len;
  return result;
}

// PPU-side variant, for leftovers.
complex<float>
partial_sum(float const *A_re, float const *A_im, length_type len, bool sq)
{
  complex<float> result(0);
  if (sq)
    for (index_type i = 0; i < len; ++i)
      result += complex<float>(A_re[i]*A_re[i] - A_im[i]*A_im[i], 2*A_re[i]*A_im[i]);
  else
    for (index_type i = 0; i < len; ++i)
      result += complex<float>(A_re[i], A_im[i]);
  return result;
}

template <typename T>
T partial_sum(T const *A, length_type len, bool sq)
{
  T result(0);
  if (sq)
    for (index_type i = 0; i < len; ++i)
      result += A[i]*A[i];
  else
    for (index_type i = 0; i < len; ++i)
      result += A[i];
  return result;
}



/// This function support sum, sumsq, mean and meansq.
/// Valid values for T are float and complex<T>
float reduce(float const *A, length_type len, int cmd)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/sum_f.plg");
  bool sq = (cmd == sumsq) || (cmd == meansq);
  index_type leftover;
  float result = partial_sum(code, size, A, len, sq, leftover);
  if (leftover)
    result += partial_sum(A + len - leftover, leftover, sq);
  if ((cmd == mean) || (cmd == meansq))
    result /= len;
  return result;
}

complex<float> reduce(complex<float> const *A, length_type len, int cmd)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/sum_f.plg");
  bool sq = (cmd == sumsq) || (cmd == meansq);
  index_type leftover;
  complex<float> result = partial_sum(code, size, A, len, sq, leftover);
  if (leftover)
    result += partial_sum(A + len - leftover, leftover, sq);
  if ((cmd == mean) || (cmd == meansq))
    result /= len;
  return result;
}

complex<float> reduce(std::pair<float const *,float const *> const &A,
		      length_type len,
		      int cmd)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/sum_f.plg");
  bool sq = (cmd == sumsq) || (cmd == meansq);
  index_type leftover;
  complex<float> result = partial_sum(code, size, A, len, sq, leftover);
  if (leftover)
    result += partial_sum(A.first + len - leftover,
                          A.second + len - leftover,
                          leftover, sq);
  if ((cmd == mean) || (cmd == meansq))
    result /= len;
  return result;
}

/// Supports sum, sumsq, mean and meansq.
float
reduce_matrix(float const* A, length_type rows, index_type row_stride, length_type cols, int cmd)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/sum_f.plg");

  // Allocate an array in which to store partial results (the sum of
  // each row).  
  aligned_array<float> R(rows);

  // Perform the reduction for each row first
  bool sq = (cmd == sumsq) || (cmd == meansq);
  index_type leftover;
  for (length_type i = 0; i < rows; ++i)
  {
    float const* data = A + i * row_stride;
    R[i] = partial_sum(code, size, data, cols, sq, leftover);
    if (leftover)
      R[i] += partial_sum(data + cols - leftover, leftover, sq);
  }

  // Then further reduce those results to get the final answer.  No need
  // to square the partial results though -- they are already squared.
  sq = false;
  float result = partial_sum(code, size, R.get(), rows, sq, leftover);
  if (leftover)
    result += partial_sum(R.get() + rows - leftover, leftover, sq);
  
  if ((cmd == mean) || (cmd == meansq))
    result /= (rows * cols);
  return result;
}


/// Supports sum, sumsq, mean and meansq.
std::complex<float>
reduce_matrix(std::complex<float> const* A, length_type rows, index_type row_stride, length_type cols, int cmd)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/sum_f.plg");

  // Allocate an array in which to store partial results (the sum of
  // each row).  
  aligned_array<std::complex<float> > R(rows);

  // Perform the reduction for each row first
  bool sq = (cmd == sumsq) || (cmd == meansq);
  index_type leftover;
  for (length_type i = 0; i < rows; ++i)
  {
    std::complex<float> const* data = A + i * row_stride;
    R[i] = partial_sum(code, size, data, cols, sq, leftover);
    if (leftover)
      R[i] += partial_sum(data + cols - leftover, leftover, sq);
  }

  // Then further reduce those results to get the final answer.  No need
  // to square the partial results though -- they are already squared.
  sq = false;
  std::complex<float> result = partial_sum(code, size, R.get(), rows, sq, leftover);
  if (leftover)
    result += partial_sum(R.get() + rows - leftover, leftover, sq);

  if ((cmd == mean) || (cmd == meansq))
    result /= (rows * cols);
  return result;
}


/// Supports sum, sumsq, mean and meansq.
std::complex<float>
reduce_matrix(std::pair<float const *,float const *> const &A, length_type rows, index_type row_stride, length_type cols, int cmd)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/sum_f.plg");

  // Allocate an array in which to store partial results (the sum of
  // each row).  
  aligned_array<std::complex<float> > R(rows);

  // Perform the reduction for each row first
  bool sq = (cmd == sumsq) || (cmd == meansq);
  index_type leftover;
  for (length_type i = 0; i < rows; ++i)
  {
    typedef std::pair<float const*, float const*> ptr_type;
    ptr_type data = ptr_type(A.first + i * row_stride, A.second + i * row_stride);
    R[i] = partial_sum(code, size, data, cols, sq, leftover);
    if (leftover)
      R[i] += partial_sum(data.first + cols - leftover, 
                          data.second + cols - leftover, 
                          leftover, sq);
  }

  // Then further reduce those results to get the final answer.  No need
  // to square the partial results though -- they are already squared.
  sq = false;
  std::complex<float> result = partial_sum(code, size, R.get(), rows, sq, leftover);
  if (leftover)
    result += partial_sum(R.get() + rows - leftover, leftover, sq);

  if ((cmd == mean) || (cmd == meansq))
    result /= (rows * cols);
  return result;
}


} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
