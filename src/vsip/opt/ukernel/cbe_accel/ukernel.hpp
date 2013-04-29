/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_OPT_UKERNEL_CBE_ACCEL_UKERNEL_HPP
#define VSIP_OPT_UKERNEL_CBE_ACCEL_UKERNEL_HPP

#include <vsip/opt/ukernel/ukernel_params.hpp>
#include <vsip/opt/cbe/dma.h>
#include <lwp_kernel.h>
#include <alf_accel.h>
#include <utility>
#include <cassert>
#include <complex>

#define DEBUG_ALF_BASE 0

#if DEBUG_ALF_BASE
#  include <cstdio>
#endif

struct Pinfo
{
  unsigned int dim;            // dimensions in this sub-block
  unsigned int l_total_size;   // total elements for this (local) iteration
  unsigned int l_offset[3];    // offset to beginning of data (if overlap
                               //  is requested, or alignment is required 
                               //  for DMA)
  unsigned int l_size[3];      // elements per dimension for this iteration
  signed int   l_stride[3];    // next-element stride in each dimension
  signed int   g_offset[3];    // local chunk's offset in global view
  signed int   o_leading[3];   // leading overlap
  signed int   o_trailing[3];  // trailing overlap
};

namespace vsip_csl
{
namespace ukernel
{
namespace impl
{

template <typename T1, typename T2>
struct is_same
{
  static bool const value = false;
};

template <typename T> 
struct is_same<T, T>
{
  static bool const value = true;
  typedef T type;
};

template <typename T>
inline void
add_entry(lwp_functions* pf,
	  void*             entries,
	  unsigned int      size,
	  alf_data_addr64_t ea)
{
  while (size > 0)
  {
    unsigned int this_size = size*sizeof(T) > 16384 ? 16384/sizeof(T) : size;
    assert(VSIP_IMPL_IS_DMA_ALIGNED(ea));
    assert(VSIP_IMPL_IS_DMA_SIZE(this_size, T));
    (pf->f_dtl_entry_add)(entries, this_size, ALF_DATA_FLOAT, ea);
    size -= this_size;
    ea   += this_size * sizeof(T);
  }
}



/// Helper functor, converts void buffer pointer to appropriate type.
///
/// The 'off' parameter is a byte offset, while 'size' is in elements.
/// This is necessary because the first is calculated from the amount
/// data previously allocated, which may or may not have the same data
/// type.  Conversely, the second parameter refers to the amount of
/// data for the current segment and it is therefore easier to use
/// pointer arithmetic since the type is known.
///
template <typename T>
struct To_ptr
{
  static T offset(void* data, size_t off, size_t) 
    { return (T)((size_t)data + off); }
};

template <typename T>
struct To_ptr<std::pair<T*, T*> >
{
  static std::pair<T*, T*> offset(void* data, size_t off, size_t size)
    { return std::pair<T*, T*>((T*)((size_t)data + off), 
                               (T*)((size_t)data + off) + size); }
};


/// Converts a size in number of elements (or index value) into an offset 
/// based on the type referenced by the pointer.
///
template <typename T>
struct Byte_offset;

template <typename T>
struct Byte_offset<T*>
{
  static size_t index(size_t size) { return sizeof(T) * size; }
};

template <typename T>
struct Byte_offset<std::pair<T*, T*> >
{
  static size_t index(size_t size) { return 2 * sizeof(T) * size; }
};


void
stream_buffer_size(Uk_stream const &stream,
		   unsigned int iter, 
		   unsigned int /*iter_count*/,
		   unsigned int &num_lines,
		   unsigned int &line_size,
		   int &offset,
		   char /*ptype*/);

template <typename PtrT>
void
add_stream(lwp_functions* pf,
	   void *entries,
	   Uk_stream const &stream,
	   unsigned int iter, 
	   unsigned int iter_count)
{
  alf_data_addr64_t ea;
  int offset;
  unsigned int num_lines;
  unsigned int line_size;

  char ptype =
    is_same<PtrT, float*>::value ? 'S' :
      is_same<PtrT, std::complex<float>*>::value ? 'C' :
        is_same<PtrT, std::pair<float*, float*> >::value ? 'Z' :
          is_same<PtrT, unsigned int*>::value ? 'I' :
            '?';

  stream_buffer_size(stream, iter, iter_count, num_lines, line_size, offset,
		     ptype);

  if (is_same<PtrT, float*>::value)
  {
    ea = stream.addr + offset;

    for (unsigned int i=0; i<num_lines; ++i)
    {
      alf_data_addr64_t eax = ea + i*stream.stride0 * sizeof(float);
      add_entry<float>(pf, entries, line_size, eax);
    }
  }
  else if (is_same<PtrT, std::complex<float>*>::value)
  {
    ea = stream.addr + 2 * offset;

    for (unsigned int i=0; i<num_lines; ++i)
      add_entry<float>(pf, entries, 2*line_size,
		       ea + 2*i*stream.stride0*sizeof(float));
  }
  else if (is_same<PtrT, std::pair<float*, float*> >::value)
  {
    ea = stream.addr + offset;

    for (unsigned int i=0; i<num_lines; ++i)
      add_entry<float>(pf, entries, line_size,
		       ea + i*stream.stride0*sizeof(float));

    ea = stream.addr_split + offset;

    for (unsigned int i=0; i<num_lines; ++i)
      add_entry<float>(pf, entries, line_size,
		       ea + i*stream.stride0*sizeof(float));
  }
  else if (is_same<PtrT, unsigned int*>::value)
  {
    ea = stream.addr + offset;

    for (unsigned int i=0; i<num_lines; ++i)
    {
      alf_data_addr64_t eax = ea + i*stream.stride0 * sizeof(unsigned int);
      add_entry<unsigned int>(pf, entries, line_size, eax);
    }
  }
  else { assert(0); }
}



void
set_chunk_info(Uk_stream const &stream, Pinfo &pinfo, int iter);

/***********************************************************************
  Z = f(X)
***********************************************************************/

template <typename      KernelT,
	  unsigned int PreArgc = KernelT::pre_argc,
	  unsigned int InArgc  = KernelT::in_argc,
	  unsigned int OutArgc = KernelT::out_argc>
struct Kernel_helper;

template <typename KernelT>
struct Kernel_helper<KernelT, 0, 0, 0>
{
  typedef Ukernel_params<KernelT::pre_argc, KernelT::in_argc,
			 KernelT::out_argc, typename KernelT::param_type>
  param_type;

  static void
  input(lwp_functions* /*pf*/,
	param_type*  /*ukp*/,
	void*        /*entries*/,
	unsigned int /*iter*/, 
	unsigned int /*iter_count*/)
  {
  }

  static void
  kernel(KernelT&     ukobj,
	 param_type*  /*ukp*/,
	 void*        /*inout*/,
	 unsigned int /*iter*/, 
	 unsigned int /*iter_count*/)
  {
    ukobj.compute();
  }
};



template <typename KernelT>
struct Kernel_helper<KernelT, 0, 1, 1>
{
  typedef Ukernel_params<KernelT::pre_argc, KernelT::in_argc,
			 KernelT::out_argc, typename KernelT::param_type>
  param_type;

  static void
  input(lwp_functions* pf,
	param_type*       ukp,
	void*             entries,
	unsigned int      iter, 
	unsigned int      iter_count)
  {
    typedef typename KernelT::in0_type  in0_type;
    typedef typename KernelT::out0_type out0_type;

    Pinfo p_out;
    set_chunk_info(ukp->out_stream[0], p_out, iter);
    size_t off1 = Byte_offset<out0_type>::index(p_out.l_total_size);

    // Transfer input A.
    (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, off1);

    add_stream<in0_type>(pf, entries, ukp->in_stream[0], iter, iter_count);

    (pf->f_dtl_end)(entries);
  }

  static void
  kernel(KernelT&     ukobj,
	 param_type*  ukp,
	 void*        inout,
	 unsigned int iter, 
	 unsigned int /*iter_count*/)
  {
    typedef typename KernelT::in0_type  in0_type;
    typedef typename KernelT::out0_type out0_type;

    Pinfo p_in, p_out;

    set_chunk_info(ukp->in_stream[0], p_in,   iter);
    set_chunk_info(ukp->out_stream[0], p_out, iter);

    size_t off1 = Byte_offset<out0_type>::index(p_out.l_total_size);

    ukobj.compute(
      To_ptr<in0_type >::offset(inout, off1, p_in.l_total_size),
      To_ptr<out0_type>::offset(inout,    0, p_out.l_total_size),
      p_in, p_out);
  }
};

template <typename KernelT>
struct Kernel_helper<KernelT, 0, 2, 1>
{
  typedef Ukernel_params<KernelT::pre_argc, KernelT::in_argc,
			 KernelT::out_argc, typename KernelT::param_type>
  param_type;

  static void
  input(lwp_functions* pf,
	param_type*  ukp,
	void*        entries,
	unsigned int iter, 
	unsigned int iter_count)
  {
    typedef typename KernelT::in0_type  in0_type;
    typedef typename KernelT::in1_type  in1_type;
    typedef typename KernelT::out0_type out0_type;

    Pinfo p_out;
    set_chunk_info(ukp->out_stream[0], p_out, iter);
    size_t off1 = Byte_offset<out0_type>::index(p_out.l_total_size);

    (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, off1);

    add_stream<in0_type>(pf, entries, ukp->in_stream[0], iter, iter_count);
    add_stream<in1_type>(pf, entries, ukp->in_stream[1], iter, iter_count);

    (pf->f_dtl_end)(entries);
  }

  static void
  kernel(KernelT&     ukobj,
	 param_type*  ukp,
	 void*        inout,
	 unsigned int iter, 
	 unsigned int /*iter_count*/)
  {
    typedef typename KernelT::in0_type  in0_type;
    typedef typename KernelT::in1_type  in1_type;
    typedef typename KernelT::out0_type out0_type;

    Pinfo p_in0, p_in1, p_out;

    set_chunk_info(ukp->in_stream[0],  p_in0, iter);
    set_chunk_info(ukp->in_stream[1],  p_in1, iter);
    set_chunk_info(ukp->out_stream[0], p_out, iter);

    size_t off1 = Byte_offset<out0_type>::index(p_out.l_total_size);
    size_t off2 = Byte_offset<in0_type>::index(p_in0.l_total_size) + off1;

    ukobj.compute(To_ptr<in0_type >::offset(inout, off1, p_in0.l_total_size),
		  To_ptr<in1_type >::offset(inout, off2, p_in1.l_total_size),
		  To_ptr<out0_type>::offset(inout, 0,    p_out.l_total_size),
		  p_in0, p_in1, p_out);
  }
};



template <typename KernelT>
struct Kernel_helper<KernelT, 1, 1, 1>
{
  typedef Ukernel_params<KernelT::pre_argc, KernelT::in_argc,
			 KernelT::out_argc, typename KernelT::param_type>
  param_type;

  static void
  input(lwp_functions* pf,
	param_type*  ukp,
	void*        entries,
	unsigned int iter, 
	unsigned int iter_count)
  {
    typedef typename KernelT::in0_type  in0_type;
    typedef typename KernelT::in1_type  in1_type;
    typedef typename KernelT::out0_type out0_type;

    if (iter < ukp->pre_chunks)
    {
      (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);
      add_stream<in0_type>(pf, entries, ukp->in_stream[0], iter, iter_count);
    }
    else
    {
      Pinfo p_out;
      set_chunk_info(ukp->out_stream[0], p_out, iter);
      size_t off1 = Byte_offset<out0_type>::index(p_out.l_total_size);

      (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, off1);
      add_stream<in1_type>(pf, entries, ukp->in_stream[1],
			   iter - ukp->pre_chunks, 
			   iter_count - ukp->pre_chunks);
    }

    (pf->f_dtl_end)(entries);
  }

  static void
  kernel(KernelT&     ukobj,
	 param_type*  ukp,
	 void*        inout,
	 unsigned int iter, 
	 unsigned int /*iter_count*/)
  {
    typedef typename KernelT::in0_type  in0_type;
    typedef typename KernelT::in1_type  in1_type;
    typedef typename KernelT::out0_type out0_type;

    Pinfo p_in0, p_in1, p_out;

    if (iter < ukp->pre_chunks)
    {
      set_chunk_info(ukp->in_stream[0],  p_in0, iter);
      ukobj.pre_compute(
	To_ptr<in0_type >::offset(inout,  0, p_in0.l_total_size),
	p_in0);
    }
    else
    {
      // the iteration count must be adjusted to account for the
      // one used above
      set_chunk_info(ukp->in_stream[1],  p_in1, iter - 1);
      set_chunk_info(ukp->out_stream[0], p_out, iter - 1);
      size_t off1 = Byte_offset<out0_type>::index(p_out.l_total_size);
      ukobj.compute(
	To_ptr<in1_type >::offset(inout, off1, p_in1.l_total_size),
	To_ptr<out0_type>::offset(inout, 0,    p_out.l_total_size),
	p_in1, p_out);
    }
  }
};
  

template <typename KernelT>
struct Kernel_helper<KernelT, 0, 3, 1>
{
  typedef Ukernel_params<KernelT::pre_argc, KernelT::in_argc,
			 KernelT::out_argc, typename KernelT::param_type>
  param_type;

  static void
  input(lwp_functions* pf,
	param_type*  ukp,
	void*        entries,
	unsigned int iter, 
	unsigned int iter_count)
  {
    typedef typename KernelT::in0_type  in0_type;
    typedef typename KernelT::in1_type  in1_type;
    typedef typename KernelT::in2_type  in2_type;
    typedef typename KernelT::out0_type out0_type;

    Pinfo p_out;
    set_chunk_info(ukp->out_stream[0], p_out, iter);
    size_t off1 = Byte_offset<out0_type>::index(p_out.l_total_size);

    (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, off1);

    add_stream<in0_type>(pf, entries, ukp->in_stream[0], iter, iter_count);
    add_stream<in1_type>(pf, entries, ukp->in_stream[1], iter, iter_count);
    add_stream<in2_type>(pf, entries, ukp->in_stream[2], iter, iter_count);

    (pf->f_dtl_end)(entries);
  }

  static void
  kernel(KernelT&     ukobj,
	 param_type*  ukp,
	 void*        inout,
	 unsigned int iter, 
	 unsigned int /*iter_count*/)
  {
    typedef typename KernelT::in0_type  in0_type;
    typedef typename KernelT::in1_type  in1_type;
    typedef typename KernelT::in2_type  in2_type;
    typedef typename KernelT::out0_type out0_type;

    Pinfo p_in0, p_in1, p_in2, p_out;

    set_chunk_info(ukp->in_stream[0],  p_in0, iter);
    set_chunk_info(ukp->in_stream[1],  p_in1, iter);
    set_chunk_info(ukp->in_stream[2],  p_in2, iter);
    set_chunk_info(ukp->out_stream[0], p_out, iter);

    // Pointers must be extracted from knowledge of the stream sizes as ALF
    // transfers all the input data into one contiguous space.

    size_t off1 = Byte_offset<out0_type>::index(p_out.l_total_size);
    size_t off2 = Byte_offset<in0_type>::index(p_in0.l_total_size) + off1;
    size_t off3 = Byte_offset<in1_type>::index(p_in1.l_total_size) + off2;

    // The To_ptr<> struct calculates the correct offset for a given
    // pointer type (scalar, interleaved complex or split complex).  The 
    // first size passes refers to the previous data segments.  The second 
    // size pertains to the current segment and is only needed to calculate 
    // offsets in the case of split complex.

    ukobj.compute(
      To_ptr<in0_type >::offset(inout, off1, p_in0.l_total_size),
      To_ptr<in1_type >::offset(inout, off2, p_in1.l_total_size),
      To_ptr<in2_type >::offset(inout, off3, p_in2.l_total_size),
      To_ptr<out0_type>::offset(inout, 0,    p_out.l_total_size),
      p_in0, p_in1, p_in2, p_out);
  }
};



template <typename     KernelT,
	  unsigned int OutArgc = KernelT::out_argc>
struct Output_helper
{
  typedef Ukernel_params<KernelT::pre_argc, KernelT::in_argc,
			 KernelT::out_argc, typename KernelT::param_type>
  param_type;

  static void
  output(lwp_functions *,
	 param_type*,
	 void*, 
	 unsigned int, 
	 unsigned int)
  {}
};

template <typename KernelT>
struct Output_helper<KernelT, 1>
{
  typedef Ukernel_params<KernelT::pre_argc, KernelT::in_argc,
			 KernelT::out_argc, typename KernelT::param_type>
  param_type;

  static void
  output(lwp_functions* pf,
	 param_type*  ukp,
	 void*        entries, 
	 unsigned int iter, 
	 unsigned int iter_count)
  {
    typedef typename KernelT::out0_type out0_type;

    if (iter < ukp->pre_chunks)
      return;
    else
    {
      iter       -= ukp->pre_chunks;
      iter_count -= ukp->pre_chunks;
    }
    
    // Transfer output Z.
    (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);
    
    add_stream<out0_type>(pf, entries, ukp->out_stream[0], iter, iter_count);
    
    (pf->f_dtl_end)(entries);
  }
};
  
} // namespace vsip_csl::ukernel::impl
} // namespace vsip_csl::ukernel
} // namespace vsip_csl

#define DEFINE_KERNEL(KERNEL)			  \
KERNEL ukobj;				          \
extern "C"					  \
int						  \
input(lwp_functions *pf,			  \
      void *params,				  \
      void *entries,				  \
      unsigned int iter,			  \
      unsigned int iter_count)			  \
{						  \
  using namespace vsip_csl::ukernel::impl;	  \
  typedef Ukernel_params<KERNEL::pre_argc,	  \
                         KERNEL::in_argc,	  \
                         KERNEL::out_argc,	  \
                         KERNEL::param_type>	  \
  param_type;					  \
						  \
  param_type *ukp  = (param_type*)params;	  \
  Kernel_helper<KERNEL>::input(pf, ukp, entries,  \
			       iter, iter_count); \
  return 0;					  \
}						  \
						  \
extern "C"					  \
int						  \
output(lwp_functions *pf,			  \
       void *params,				  \
       void *entries,				  \
       unsigned int iter,			  \
       unsigned int iter_count)			  \
{						  \
  using namespace vsip_csl::ukernel::impl;	  \
  typedef Ukernel_params<KERNEL::pre_argc,	  \
                         KERNEL::in_argc,	  \
			 KERNEL::out_argc,	  \
			 KERNEL::param_type>	  \
  param_type;					  \
						  \
  param_type *ukp = (param_type*)params;	  \
  Output_helper<KERNEL>::output(pf, ukp, entries, \
				iter, iter_count);\
  return 0;					  \
}						  \
						  \
extern "C"					  \
int						  \
kernel(lwp_functions *pf,			  \
       void *params,				  \
       void *inout,				  \
       unsigned int iter,			  \
       unsigned int iter_count)			  \
{						  \
  using namespace vsip_csl::ukernel::impl;	  \
  typedef Ukernel_params<KERNEL::pre_argc,	  \
			 KERNEL::in_argc,	  \
			 KERNEL::out_argc,	  \
			 KERNEL::param_type>	  \
  param_type;					  \
						  \
  param_type *ukp  = (param_type *)params;	  \
  if (iter == 0)				  \
  {						  \
    ukobj.init_rank(ukp->rank, ukp->nspe);	  \
    ukobj.init(ukp->kernel_params);		  \
  }						  \
  						  \
  Kernel_helper<KERNEL>::kernel(ukobj, ukp, inout,\
				iter, iter_count);\
						  \
  if (iter == iter_count-1)			  \
    ukobj.fini();				  \
  return 0;					  \
}


/// Accelerator-side user kernel base class.
/// Deprecated.
struct Spu_kernel
{
  static unsigned int const in_argc  = 1;
  static unsigned int const out_argc = 1;
  static unsigned int const pre_argc = 0;
  typedef Empty_params param_type;

  static bool const in_place        = false;

  void init_rank(int /*rank*/, int /*nspe*/) {}

  template <typename ParamT>
  void init(ParamT const &) {}

  void fini() {}

  template <typename T>
  void pre_compute(T, Pinfo const&) {}
};

#endif // VSIP_OPT_UKERNEL_CBE_ACCEL_UKERNEL_HPP
