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
#include <vsip/opt/cbe/unary_params.h>

namespace
{
using namespace vsip;
using namespace vsip::impl;
using namespace vsip::impl::cbe;

struct Worker
{
  Worker(char const *code_ea,
	 int code_size,
	 std::pair<float const *, float const *> const& A,
	 std::pair<float*, float*> const& R,
	 length_type len,
	 int cmd = 0)
  {
    length_type const chunk_size = 1024;
  
    assert(is_dma_addr_ok(A.first) && is_dma_addr_ok(A.second));
    assert(is_dma_addr_ok(R.first) && is_dma_addr_ok(R.second));

    Task_manager *mgr = Task_manager::instance();

    task = mgr->reserve_lwp_task(VSIP_IMPL_UNARY_BUFFER_SIZE, 2,
				 (uintptr_t)code_ea, code_size, cmd);

    assert(sizeof(float)*4*chunk_size <= VSIP_IMPL_UNARY_BUFFER_SIZE);
    assert(8 < VSIP_IMPL_UNARY_DTL_SIZE);

    Unary_split_params params;

    params.cmd          = cmd;
    params.length       = chunk_size;

    length_type chunks = len / chunk_size;
    length_type spes   = mgr->num_spes();
    length_type chunks_per_spe = chunks / spes;
    assert(chunks_per_spe * spes <= chunks);
    
    float const* a_re_ptr = A.first;
    float const* a_im_ptr = A.second;
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
      params.r_re_ptr = (uintptr_t)r_re_ptr;
      params.r_im_ptr = (uintptr_t)r_im_ptr;
      block.set_parameters(params);
      block.enqueue();
      
      a_re_ptr += my_chunks*chunk_size;
      a_im_ptr += my_chunks*chunk_size;
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
      params.r_re_ptr = (uintptr_t)r_re_ptr;
      params.r_im_ptr = (uintptr_t)r_im_ptr;
      block.set_parameters(params);
      block.enqueue();
      len -= params.length;
    }
    leftover = len;
  }
  Worker(char const *code_ea, int code_size,
	 std::pair<float const *, float const *> const& A, float *R, length_type len,
	 int cmd = 0)
  {
    length_type const chunk_size = 1024;
  
    assert(is_dma_addr_ok(A.first) && is_dma_addr_ok(A.second));
    assert(is_dma_addr_ok(R));

    Task_manager *mgr = Task_manager::instance();

    task = mgr->reserve_lwp_task(VSIP_IMPL_UNARY_BUFFER_SIZE, 2,
				 (uintptr_t)code_ea, code_size, cmd);

    assert(sizeof(float)*4*chunk_size <= VSIP_IMPL_UNARY_BUFFER_SIZE);
    assert(8 < VSIP_IMPL_UNARY_DTL_SIZE);

    Unary_split_params params;

    params.cmd          = cmd;
    params.length       = chunk_size;

    length_type chunks = len / chunk_size;
    length_type spes   = mgr->num_spes();
    length_type chunks_per_spe = chunks / spes;
    assert(chunks_per_spe * spes <= chunks);
    
    float const* a_re_ptr = A.first;
    float const* a_im_ptr = A.second;
    float*       r_ptr = R;

    for (index_type i=0; i<spes && i<chunks; ++i)
    {
      // If chunks don't divide evenly, give the first SPEs one extra.
      length_type my_chunks = (i < chunks % spes) ? chunks_per_spe + 1
	: chunks_per_spe;
      
      lwp::Workblock block = task->create_workblock(my_chunks);
      params.a_re_ptr = (uintptr_t)a_re_ptr;
      params.a_im_ptr = (uintptr_t)a_im_ptr;
      params.r_re_ptr = (uintptr_t)r_ptr;
      block.set_parameters(params);
      block.enqueue();
      
      a_re_ptr += my_chunks*chunk_size;
      a_im_ptr += my_chunks*chunk_size;
      r_ptr += my_chunks*chunk_size;
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
      params.r_re_ptr = (uintptr_t)r_ptr;
      block.set_parameters(params);
      block.enqueue();
      len -= params.length;
    }
    leftover = len;
  }

  // Valid template arguments are float and complex<float>
  template <typename T1, typename T2>
  Worker(char const *code_ea,
	 int code_size,
	 T1 const *A,
	 T2 *R,
	 length_type len,
	 int cmd = 0)
  {
    length_type const chunk_size = 1024;

    assert(is_dma_addr_ok(A));
    assert(is_dma_addr_ok(R));

    Task_manager *mgr = Task_manager::instance();

    task = mgr->reserve_lwp_task(VSIP_IMPL_UNARY_BUFFER_SIZE, 2,
				 (uintptr_t)code_ea, code_size, cmd);

    assert(sizeof(T1)*4*chunk_size <= VSIP_IMPL_UNARY_BUFFER_SIZE);
    assert(8 < VSIP_IMPL_UNARY_DTL_SIZE);

    Unary_params params;

    params.cmd          = cmd;
    params.length       = chunk_size;

    length_type chunks = len / chunk_size;
    length_type spes   = mgr->num_spes();
    length_type chunks_per_spe = chunks / spes;
    assert(chunks_per_spe * spes <= chunks);
    
    T1 const* a_ptr = A;
    T2*       r_ptr = R;

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
      r_ptr += my_chunks*chunk_size;
      len -= my_chunks * chunk_size;
    }

    // Cleanup leftover data that doesn't fit into a full chunk.

    // First, handle data that can be DMA'd to the SPEs.  DMA granularity
    // is 16 bytes for sizes 16 bytes or larger.

    // However, taking vectorization into account, it's simpler to always use
    // the greater granularity of 4.
    length_type const granularity = 4; //VSIP_IMPL_CBE_DMA_GRANULARITY / sizeof(T2);
    
    if (len >= granularity)
    {
      params.length = (len / granularity) * granularity;
      assert(is_dma_size_ok(params.length*sizeof(T1)));
      assert(is_dma_size_ok(params.length*sizeof(T2)));
      lwp::Workblock block = task->create_workblock(1);
      params.a_ptr = (uintptr_t)a_ptr;
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

void vsqrt(float const *A, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, SQRT);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = sqrt(A[i]);
}

void vatan(float const *A, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, ATAN);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = atan(A[i]);
}

void vlog(float const *A, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, LOG);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = log(A[i]);
}

void vlog(complex<float> const *A, complex<float> *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, CLOG);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = log(A[i]);
}

void vlog(std::pair<float const *,float const *> const &A, std::pair<float*,float*> const &R,
	 length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, ZLOG);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
    {
      float ar = A.first[i],
            ai = A.second[i];
      R.first[i]  = log(sqrt(ar*ar + ai*ai));
      R.second[i] = atan2(ai,ar);
    }
}

void vlog10(float const *A, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, LOG10);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = log10(A[i]);
}

void vlog10(complex<float> const *A, complex<float> *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, CLOG10);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = log10(A[i]);
}

void vlog10(std::pair<float const *,float const *> const &A, std::pair<float*,float*> const &R,
	 length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, ZLOG10);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
    {
      float ar = A.first[i],
            ai = A.second[i];
      R.first[i]  = log10(sqrt(ar*ar + ai*ai));
      R.second[i] = atan2(ai,ar)/2.30258509299404568402;
    }
}

void vcos(float const *A, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, COS);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = cos(A[i]);
}

void vcos(complex<float> const *A, complex<float> *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, CCOS);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = cos(A[i]);
}

void vcos(std::pair<float const *,float const *> const &A, std::pair<float*,float*> const &R,
	 length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, ZCOS);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
    {
      float ar = A.first[i],
            ai = A.second[i];
      R.first[i]  =  cos(ar) * cosh(ai);
      R.second[i] = -sin(ar) * sinh(ai);
    }
}

void vsin(float const *A, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, SIN);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = sin(A[i]);
}

void vsin(complex<float> const *A, complex<float> *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, ZSIN);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = sin(A[i]);
}

void vsin(std::pair<float const *,float const *> const &A, std::pair<float*,float*> const &R,
	 length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, ZSIN);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
    {
      float ar = A.first[i],
            ai = A.second[i];
      R.first[i]  =  sin(ar) * cosh(ai);
      R.second[i] =  cos(ar) * sinh(ai);
    }
}

void vminus(float const *A, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, MINUS);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = -A[i];
}

void vminus(complex<float> const *A, complex<float> *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, CMINUS);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = -A[i];
}

void vminus(std::pair<float const *,float const *> const &A, std::pair<float*,float*> const &R,
	 length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, ZMINUS);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
    {
      R.first[i] = -A.first[i];
      R.second[i] = -A.second[i];
    }
}

void vsq(float const *A, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, SQ);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = A[i]*A[i];
}

void vsq(complex<float> const *A, complex<float> *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, CSQ);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = A[i]*A[i];
}

void vsq(std::pair<float const *,float const *> const &A, std::pair<float*,float*> const &R,
	 length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, ZSQ);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
    {
      R.first[i] = A.first[i]*A.first[i] - A.second[i]*A.second[i];
      R.second[i] = 2*A.first[i]*A.second[i];
    }
}

void vmag(float const *A, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, MAG);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = fn::mag(A[i]);
}

void vmag(complex<float> const *A, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, CMAG);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = fn::mag(A[i]);
}

void vmag(std::pair<float const *,float const *> const &A, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, ZMAG);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = sqrt(A.first[i]*A.first[i] + A.second[i]*A.second[i]);
}

void vmagsq(float const *A, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, MAGSQ);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = A[i]*A[i];
}

void vmagsq(complex<float> const *A, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, CMAGSQ);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = fn::magsq(A[i]);
}

void vmagsq(std::pair<float const *,float const *> const &A, float *R, length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, ZMAGSQ);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = A.first[i]*A.first[i] + A.second[i]*A.second[i];
}

void vconj(complex<float> const* A, complex<float>* R,
	   length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, CCONJ);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = conj(A[i]);
}

void vconj(std::pair<float const *, float const *> const& A,
           std::pair<float*, float*> const& R,
           length_type len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/unary_f.plg");
  Worker worker(code, size, A, R, len, ZCONJ);
  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
    {
      R.first[i]  =  A.first[i];
      R.second[i] = -A.second[i];
    }
}

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
