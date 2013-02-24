/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#include <vsip/core/config.hpp>
#include <vsip/math.hpp>
#include <vsip/opt/cbe/ppu/ternary.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>
#include <vsip/opt/cbe/ternary_params.h>

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
	 std::pair<float const *, float const *> const& C,
	 std::pair<float*, float*> const& R,
	 length_type len,
	 int cmd)
  {
    length_type const chunk_size = 1024;
  
    assert(is_dma_addr_ok(A.first) && is_dma_addr_ok(A.second));
    assert(is_dma_addr_ok(B.first) && is_dma_addr_ok(B.second));
    assert(is_dma_addr_ok(R.first) && is_dma_addr_ok(R.second));
    assert(is_dma_addr_ok(R.first) && is_dma_addr_ok(R.second));

    Task_manager *mgr = Task_manager::instance();

    task = mgr->reserve_lwp_task(VSIP_IMPL_TERNARY_BUFFER_SIZE, 2,
				 (uintptr_t)code_ea, code_size, cmd);

    assert(sizeof(float)*4*chunk_size <= VSIP_IMPL_TERNARY_BUFFER_SIZE);
    assert(8 < VSIP_IMPL_TERNARY_DTL_SIZE);

    Ternary_split_params params;

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
    float const* c_re_ptr = C.first;
    float const* c_im_ptr = C.second;
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
      params.c_re_ptr = (uintptr_t)c_re_ptr;
      params.c_im_ptr = (uintptr_t)c_im_ptr;
      params.r_re_ptr = (uintptr_t)r_re_ptr;
      params.r_im_ptr = (uintptr_t)r_im_ptr;
      block.set_parameters(params);
      block.enqueue();
      
      a_re_ptr += my_chunks*chunk_size;
      a_im_ptr += my_chunks*chunk_size;
      b_re_ptr += my_chunks*chunk_size;
      b_im_ptr += my_chunks*chunk_size;
      c_re_ptr += my_chunks*chunk_size;
      c_im_ptr += my_chunks*chunk_size;
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
      params.c_re_ptr = (uintptr_t)c_re_ptr;
      params.c_im_ptr = (uintptr_t)c_im_ptr;
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
	 T const *C,
	 T *R,
	 length_type len,
	 int cmd)
  {
    length_type const chunk_size = 1024;

    assert(is_dma_addr_ok(A));
    assert(is_dma_addr_ok(B));
    assert(is_dma_addr_ok(C));
    assert(is_dma_addr_ok(R));

    Task_manager *mgr = Task_manager::instance();

    task = mgr->reserve_lwp_task(VSIP_IMPL_TERNARY_BUFFER_SIZE, 2,
				 (uintptr_t)code_ea, code_size, cmd);

    assert(sizeof(T)*4*chunk_size <= VSIP_IMPL_TERNARY_BUFFER_SIZE);
    assert(8 < VSIP_IMPL_TERNARY_DTL_SIZE);

    Ternary_params params;

    params.cmd          = cmd;
    params.length       = chunk_size;

    length_type chunks = len / chunk_size;
    length_type spes   = mgr->num_spes();
    length_type chunks_per_spe = chunks / spes;
    assert(chunks_per_spe * spes <= chunks);
    
    T const* a_ptr = A;
    T const* b_ptr = B;
    T const* c_ptr = C;
    T*       r_ptr = R;

    for (index_type i=0; i<spes && i<chunks; ++i)
    {
      // If chunks don't divide evenly, give the first SPEs one extra.
      length_type my_chunks = (i < chunks % spes) ? chunks_per_spe + 1
                                                  : chunks_per_spe;
      
      lwp::Workblock block = task->create_workblock(my_chunks);
      params.a_ptr = (uintptr_t)a_ptr;
      params.b_ptr = (uintptr_t)b_ptr;
      params.c_ptr = (uintptr_t)c_ptr;
      params.r_ptr = (uintptr_t)r_ptr;
      block.set_parameters(params);
      block.enqueue();

      a_ptr += my_chunks*chunk_size;
      b_ptr += my_chunks*chunk_size;
      c_ptr += my_chunks*chunk_size;
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
      params.c_ptr = (uintptr_t)c_ptr;
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

void vam(float const *A, float const *B, float const *C, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/ternary_f.plg");
  Worker worker(code, size, A, B, C, R, len, AM);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = (A[i] + B[i]) * C[i];
}

void vam(complex<float> const *A, complex<float> const *B, complex<float> const *C,
	 complex<float> *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/ternary_f.plg");

  Worker worker(code, size, A, B, C, R, len, CAM);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = (A[i] + B[i]) * C[i];
}

void vam(std::pair<float const *, float const *> const &A,
	 std::pair<float const *, float const *> const &B,
	 std::pair<float const *, float const *> const &C,
	 std::pair<float*, float*> const &R,
	 length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/ternary_f.plg");
  Worker worker(code, size, A, B, C, R, len, ZAM);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
    {
      float tr = A.first[i] + B.first[i];
      float ti = A.second[i] + B.second[i];
      R.first[i] = tr * C.first[i] - ti * C.second[i];
      R.second[i] = tr * C.second[i] + ti * C.first[i];
    }
}

void vma(float const *A, float const *B, float const *C, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/ternary_f.plg");
  Worker worker(code, size, A, B, C, R, len, MA);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = A[i] * B[i] + C[i];
}

void vma(complex<float> const *A, complex<float> const *B, complex<float> const *C,
	 complex<float> *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/ternary_f.plg");

  Worker worker(code, size, A, B, C, R, len, CMA);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = A[i] * B[i] + C[i];
}

void vma(std::pair<float const *, float const *> const &A,
	 std::pair<float const *, float const *> const &B,
	 std::pair<float const *, float const *> const &C,
	 std::pair<float*, float*> const &R,
	 length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/ternary_f.plg");
  Worker worker(code, size, A, B, C, R, len, ZMA);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
    {
      float tmp   = A.first[i] * B.first[i] - A.second[i] * B.second[i] + C.first[i];
      R.second[i] = A.first[i] * B.second[i] + A.second[i] * B.first[i] + C.second[i];
      R.first[i]  = tmp;
    }
}

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
