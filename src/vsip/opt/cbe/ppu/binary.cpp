/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#include <vsip/core/config.hpp>
#include <vsip/math.hpp>
#include <vsip/opt/cbe/ppu/binary.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>
#include <vsip/opt/cbe/binary_params.h>
#include <vsip/opt/cbe/overlay_params.h>
#include <vsip/opt/cbe/unary_params.h>

namespace
{
using namespace vsip;
using namespace vsip::impl;
using namespace vsip::impl::cbe;

struct Worker
{
  // This constructor is valid for T=float
  Worker(char const *code_ea,
	 int code_size,
	 std::pair<float const *, float const *> const& A,
	 std::pair<float const *, float const *> const& B,
	 std::pair<float*, float*> const& R,
	 length_type len,
	 int cmd = 0)
  {
    length_type const chunk_size = 1024;
  
    assert(is_dma_addr_ok(A.first) && is_dma_addr_ok(A.second));
    assert(is_dma_addr_ok(B.first) && is_dma_addr_ok(B.second));
    assert(is_dma_addr_ok(R.first) && is_dma_addr_ok(R.second));

    Task_manager *mgr = Task_manager::instance();

    task = mgr->reserve_lwp_task(VSIP_IMPL_BINARY_BUFFER_SIZE, 2,
				 (uintptr_t)code_ea, code_size, cmd);

    assert(sizeof(float)*4*chunk_size <= VSIP_IMPL_BINARY_BUFFER_SIZE);
    assert(8 < VSIP_IMPL_BINARY_DTL_SIZE);

    Binary_split_params params;

    params.cmd          = cmd;
    params.length       = chunk_size;

    length_type chunks = len / chunk_size;
    length_type spes   = mgr->num_spes();
    length_type chunks_per_spe = chunks / spes;
    assert(chunks_per_spe * spes <= chunks);
    
    float const* a_re_ptr = A.first;
    float const* a_im_ptr = A.second;
    float const* b_re_ptr = B.first;
    float const* b_im_ptr = B.second;
    float*       r_re_ptr = R.first;
    float*       r_im_ptr = R.second;

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
      params.r_re_ptr = (uintptr_t)r_re_ptr;
      params.r_im_ptr = (uintptr_t)r_im_ptr;
      block.set_parameters(params);
      block.enqueue();
      
      a_re_ptr += my_chunks*chunk_size;
      a_im_ptr += my_chunks*chunk_size;
      b_re_ptr += my_chunks*chunk_size;
      b_im_ptr += my_chunks*chunk_size;
      r_re_ptr += my_chunks*chunk_size;
      r_im_ptr += my_chunks*chunk_size;
      len -= my_chunks * chunk_size;
    }

    // Cleanup leftover data that doesn't fit into a full chunk.

    // First, handle data that can be DMA'd to the SPEs.  DMA granularity
    // is 16 bytes for sizes 16 bytes or larger.

    length_type const granularity = VSIP_IMPL_CBE_DMA_GRANULARITY / sizeof(float);
    
    if (len >= granularity)
    {
      params.length = (len / granularity) * granularity;
      assert(is_dma_size_ok(params.length*sizeof(float)));
      lwp::Workblock block = task->create_workblock(1);
      params.a_re_ptr = (uintptr_t)a_re_ptr;
      params.a_im_ptr = (uintptr_t)a_im_ptr;
      params.b_re_ptr = (uintptr_t)b_re_ptr;
      params.b_im_ptr = (uintptr_t)b_im_ptr;
      params.r_re_ptr = (uintptr_t)r_re_ptr;
      params.r_im_ptr = (uintptr_t)r_im_ptr;
      block.set_parameters(params);
      block.enqueue();
      len -= params.length;
    }
    leftover = len;
  }

  // Valid template arguments are float and complex<float>
  template <typename T>
  Worker(char const *code_ea,
	 int code_size,
	 T const *A,
	 T const *B,
	 T *R,
	 length_type len,
	 int cmd = 0)
  {
    length_type const chunk_size = 1024;

    assert(is_dma_addr_ok(A));
    assert(is_dma_addr_ok(B));
    assert(is_dma_addr_ok(R));

    Task_manager *mgr = Task_manager::instance();

    task = mgr->reserve_lwp_task(VSIP_IMPL_BINARY_BUFFER_SIZE, 2,
				 (uintptr_t)code_ea, code_size, cmd);

    assert(sizeof(T)*4*chunk_size <= VSIP_IMPL_BINARY_BUFFER_SIZE);
    assert(8 < VSIP_IMPL_BINARY_DTL_SIZE);

    Binary_params params;

    params.cmd          = cmd;
    params.length       = chunk_size;

    length_type chunks = len / chunk_size;
    length_type spes   = mgr->num_spes();
    length_type chunks_per_spe = chunks / spes;
    assert(chunks_per_spe * spes <= chunks);
    
    T const* a_ptr = A;
    T const* b_ptr = B;
    T*       r_ptr = R;

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
      r_ptr += my_chunks*chunk_size;
      len -= my_chunks * chunk_size;
    }

    // Cleanup leftover data that doesn't fit into a full chunk.

    // First, handle data that can be DMA'd to the SPEs.  DMA granularity
    // is 16 bytes for sizes 16 bytes or larger.

    length_type const granularity = VSIP_IMPL_CBE_DMA_GRANULARITY / sizeof(T);
    
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

    leftover = len;
  }

  ~Worker() { task->sync();}

  std::auto_ptr<lwp::Task> task;
  length_type              leftover;
};

}


namespace vsip
{
namespace impl
{
namespace cbe
{

void vmul(float const *A, float const *B, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/xvmul_f.plg");
  Worker worker(code, size, A, B, R, len, overlay_vmul_f);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = A[i] * B[i];
}

void vmul(complex<float> const *A, complex<float> const *B,
	  complex<float> *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/xvmul_f.plg");
  Worker worker(code, size, A, B, R, len, overlay_cvmul_f);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = A[i] * B[i];
}

void vmul(std::pair<float const *, float const *> const &A,
	  std::pair<float const *, float const *> const &B,
	  std::pair<float*, float*> const &R,
	  length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/xvmul_f.plg");
  Worker worker(code, size, A, B, R, len, overlay_zvmul_f);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
    {
      float tmp   = A.first[i] * B.first[i]  - A.second[i] * B.second[i];
      R.second[i] = A.first[i] * B.second[i] + A.second[i] * B.first[i];
      R.first[i]  = tmp;
    }
}

void vdiv(float const* A, float const* B, float* R, length_type len)
{
  static char* code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/xvdiv_f.plg");
  Worker worker(code, size, A, B, R, len, overlay_vdiv_f);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = A[i] / B[i];
}

void vdiv(complex<float> const *A,
	  complex<float> const *B,
	  complex<float> *R,
	  length_type len)
{
  // No plugin yet.
  for (index_type i = 0; i < len; ++i)
  {
    R[i] = A[i] / B[i];
  }
}

void vdiv(std::pair<float const *, float const *> const &A,
	  std::pair<float const *, float const *> const &B,
	  std::pair<float*, float*> const &R,
	  length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/xvdiv_f.plg");
  Worker worker(code, size, A, B, R, len, overlay_zvdiv_f);
  if (worker.leftover)
  {
    for (index_type i = len - worker.leftover; i < len; ++i)
    {
      float den = B.first[i] * B.first[i] + B.second[i]*B.second[i];
      float tmp   = (A.first[i]  * B.first[i] + A.second[i] * B.second[i]) / den;
      R.second[i] = (A.second[i] * B.first[i] - A.first[i]  * B.second[i]) / den;
      R.first[i]  = tmp;
    }
  }
}

void vadd(float const *A, float const *B, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/xvadd_f.plg");
  Worker worker(code, size, A, B, R, len, overlay_vadd_f);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = A[i] + B[i];
}

void vadd(complex<float> const *A, complex<float> const *B, complex<float> *R,
	  length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/xvadd_f.plg");
  Worker worker(code, size, A, B, R, len, overlay_cvadd_f);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = A[i] + B[i];
}

void vadd(std::pair<float const *, float const *> const &A,
	  std::pair<float const *, float const *> const &B,
	  std::pair<float*, float*> const &R,
	  length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/xvadd_f.plg");
  Worker worker(code, size, A, B, R, len, overlay_zvadd_f);
  if (worker.leftover)
  {
    for (index_type i = len - worker.leftover; i < len; ++i)
    {
      R.first[i]  = A.first[i] + B.first[i];
      R.second[i] = A.second[i] + B.second[i];
    }
  }
}

void vsub(float const *A, float const *B, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/xvsub_f.plg");
  Worker worker(code, size, A, B, R, len, overlay_vsub_f);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = A[i] - B[i];
}

void vsub(complex<float> const *A, complex<float> const *B, complex<float> *R,
	  length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/xvsub_f.plg");
  Worker worker(code, size, A, B, R, len, overlay_cvsub_f);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = A[i] - B[i];
}

void vsub(std::pair<float const *, float const *> const &A,
	  std::pair<float const *, float const *> const &B,
	  std::pair<float*, float*> const &R,
	  length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/xvsub_f.plg");
  Worker worker(code, size, A, B, R, len, overlay_zvsub_f);
  if (worker.leftover)
  {
    for (index_type i = len - worker.leftover; i < len; ++i)
    {
      R.first[i]  = A.first[i] - B.first[i];
      R.second[i] = A.second[i] - B.second[i];
    }
  }
}

void vatan2(float const *A, float const *B, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/binary_f.plg");
  Worker worker(code, size, A, B, R, len, ATAN2);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = atan2(A[i],B[i]);
}

void vhypot(float const *A, float const *B, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, B, R, len, ZMAG);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = hypot(A[i],B[i]);
}

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
