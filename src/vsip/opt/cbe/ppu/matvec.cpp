/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <vsip/core/config.hpp>
#include <vsip/math.hpp>
#include <vsip/opt/cbe/dot_params.h>
#include <vsip/opt/cbe/ppu/matvec.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>
#include <memory>

namespace
{
using namespace vsip;
using namespace vsip::impl;
using namespace vsip::impl::cbe;

complex<float>
partial_dot(char const *code_ea,
	    int code_size,
	    std::pair<float const *, float const *> const& A,
	    std::pair<float const *, float const *> const& B,
	    length_type len,
	    bool conj,
	    length_type &leftover)
{
  length_type const chunk_size = 1024;
  // DMA granularity is 16 bytes for sizes 16 bytes or larger.
  length_type const granularity = VSIP_IMPL_CBE_DMA_GRANULARITY / sizeof(float);

  assert(is_dma_addr_ok(A.first) && is_dma_addr_ok(A.second));
  assert(is_dma_addr_ok(B.first) && is_dma_addr_ok(B.second));

  Task_manager *mgr = Task_manager::instance();

  std::auto_ptr<lwp::Task> task = 
    mgr->reserve_lwp_task(VSIP_IMPL_DOT_BUFFER_SIZE, 2,
			  (uintptr_t)code_ea, code_size);

  assert(sizeof(float)*4*chunk_size <= VSIP_IMPL_DOT_BUFFER_SIZE);
  assert(8 < VSIP_IMPL_DOT_DTL_SIZE);

  Dot_split_params params;
  params.conj        = conj;
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
    
  float const* a_re_ptr = A.first;
  float const* a_im_ptr = A.second;
  float const* b_re_ptr = B.first;
  float const* b_im_ptr = B.second;
  complex<float> *r_ptr = R.get();

  for (index_type i=0; i<spes && i<chunks; ++i)
  {
    // If chunks don't divide evenly, give the first SPEs one extra.
    length_type my_chunks = (i < chunks % spes) ? chunks_per_spe + 1
      : chunks_per_spe;
      
    lwp::Workblock block = task->create_workblock(my_chunks);
    params.a_re_ptr = (uintptr_t)a_re_ptr;
    params.a_im_ptr = (uintptr_t)a_im_ptr;
    params.b_re_ptr = (uintptr_t)b_re_ptr;
    params.b_im_ptr = (uintptr_t)b_im_ptr;
    params.r_ptr = (uintptr_t)r_ptr;
    block.set_parameters(params);
    block.enqueue();

    a_re_ptr += my_chunks*chunk_size;
    a_im_ptr += my_chunks*chunk_size;
    b_re_ptr += my_chunks*chunk_size;
    b_im_ptr += my_chunks*chunk_size;
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
    params.a_re_ptr = (uintptr_t)a_re_ptr;
    params.a_im_ptr = (uintptr_t)a_im_ptr;
    params.b_re_ptr = (uintptr_t)b_re_ptr;
    params.b_im_ptr = (uintptr_t)b_im_ptr;
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
T partial_dot(char const *code_ea, int code_size,
	      T const *A, T const *B, length_type len,
	      bool conj,
	      length_type &leftover)
{
  length_type const chunk_size = 1024;
  // DMA granularity is 16 bytes for sizes 16 bytes or larger.
  length_type const granularity = VSIP_IMPL_CBE_DMA_GRANULARITY / sizeof(T);

  assert(is_dma_addr_ok(A));
  assert(is_dma_addr_ok(B));

  Task_manager *mgr = Task_manager::instance();

  std::auto_ptr<lwp::Task> task = 
    mgr->reserve_lwp_task(VSIP_IMPL_DOT_BUFFER_SIZE, 2,
			  (uintptr_t)code_ea, code_size);

  assert(sizeof(T)*2*chunk_size <= VSIP_IMPL_DOT_BUFFER_SIZE);

  Dot_params params;
  params.conj        = conj;
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
  T const* b_ptr = B;
  complex<float> *r_ptr = R.get();

  for (index_type i=0; i<spes && i<chunks; ++i)
  {
    // If chunks don't divide evenly, give the first SPEs one extra.
    length_type my_chunks = (i < chunks % spes) ? chunks_per_spe + 1
      : chunks_per_spe;
      
    lwp::Workblock block = task->create_workblock(my_chunks);
    params.a_ptr = (uintptr_t)a_ptr;
    params.b_ptr = (uintptr_t)b_ptr;
    params.r_ptr = (uintptr_t)r_ptr;
    block.set_parameters(params);
    block.enqueue();

    a_ptr += my_chunks*chunk_size;
    b_ptr += my_chunks*chunk_size;
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
    params.b_ptr = (uintptr_t)b_ptr;
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
partial_dot(float const *A_re, float const *A_im,
	    float const *B_re, float const *B_im,
	    length_type len)
{
  complex<float> result(0);
  for (index_type i = 0; i < len; ++i)

    result += complex<float>(A_re[i] * B_re[i] - A_im[i] * B_im[i],
			     A_re[i] * B_im[i] + A_im[i] * B_re[i]);
  return result;
}

complex<float>
partial_jdot(float const *A_re, float const *A_im,
	     float const *B_re, float const *B_im,
	     length_type len)
{
  complex<float> result(0);
  for (index_type i = 0; i < len; ++i)

    result += complex<float>(A_re[i] * B_re[i] + A_im[i] * B_im[i],
			     A_im[i] * B_re[i] - A_re[i] * B_im[i]);
  return result;
}

complex<float>
partial_dot(complex<float> const *A,
	    complex<float> const *B,
	    length_type len)
{
  complex<float> result(0);
  for (index_type i = 0; i < len; ++i)
    result += A[i] * B[i];
  return result;
}

complex<float>
partial_jdot(complex<float> const *A,
	     complex<float> const *B,
	     length_type len)
{
  complex<float> result(0);
  for (index_type i = 0; i < len; ++i)
    result += A[i] * conj(B[i]);
  return result;
}

float
partial_dot(float const *A, float const *B, length_type len)
{
  float result(0);
  for (index_type i = 0; i < len; ++i)
    result += A[i] * B[i];
  return result;
}
}

namespace vsip
{
namespace impl
{
namespace cbe
{

float dot(float const *A, float const *B, length_type len, bool /* conj */)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/dot_f.plg");
  index_type leftover;
  float result = partial_dot(code, size, A, B, len, false, leftover);
  if (leftover)
    result += partial_dot(A + len - leftover, B + len - leftover, leftover);
  return result;
}

complex<float> dot(complex<float> const *A, complex<float> const *B, length_type len, bool conj)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/cdot_f.plg");
  index_type leftover;
  complex<float> result = partial_dot(code, size, A, B, len, conj, leftover);
  if (leftover)
  {
    if (conj)
      result += partial_jdot(A + len - leftover, B + len - leftover, leftover);
    else
      result += partial_dot(A + len - leftover, B + len - leftover, leftover);
  }
  return result;
}

complex<float> dot(std::pair<float const *,float const *> const &A,
		   std::pair<float const *,float const *> const &B,
		   length_type len,
		   bool conj)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/zdot_f.plg");
  index_type leftover;
  complex<float> result = partial_dot(code, size, A, B, len, conj, leftover);
  if (leftover)
  {
    if (conj)
      result += partial_jdot(A.first + len - leftover,
			     A.second + len - leftover,
			     B.first + len - leftover,
			     B.second + len - leftover,
			     leftover);
    else
      result += partial_dot(A.first + len - leftover,
			    A.second + len - leftover,
			    B.first + len - leftover,
			    B.second + len - leftover,
			    leftover);
  }
  return result;
}

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
