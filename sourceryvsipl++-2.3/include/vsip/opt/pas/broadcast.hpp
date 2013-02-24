/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/pas/broadcast.hpp
    @author  Jules Bergmann
    @date    2005-08-23
    @brief   VSIPL++ Library: PAS Broadcast.

*/

#ifndef VSIP_OPT_PAS_BROADCAST_HPP
#define VSIP_OPT_PAS_BROADCAST_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

extern "C" {
#include <pas.h>
}

#include <vsip/support.hpp>
#include <vsip/opt/pas/param.hpp>
#include <vsip/opt/pas/util.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace pas
{

// Forward Definition.
template <typename T>
struct Pas_datatype;



extern long             global_tag;

struct Broadcast
{
  typedef unsigned int data_type;
  static long          const num_elems = 4;

  Broadcast(PAS_id pset);
  ~Broadcast();

  template <typename T>
  void operator()(processor_type proc, T& value);

  long                    tag_;
  PAS_id                  pset_;
  PAS_data_spec           data_spec_;

  PAS_distribution_handle local_dist_;
  PAS_pbuf_handle         local_pbuf_;
  PAS_buffer*             local_buffer_;

  PAS_distribution_handle global_dist_;
  PAS_pbuf_handle         global_pbuf_;
  PAS_buffer*             global_buffer_;
};

inline
Broadcast::Broadcast(PAS_id pset)
  : pset_      (pset),
    data_spec_ (PAS_DATA_REAL_U32)
{
  long rc;

  tag_ = global_tag;
  global_tag += 2;

  // Create local buffer
  rc = pas_pbuf_create_1D(
      tag_,			// tag
      local_processor(),	// procs
      num_elems,		// num_elems
      PAS_WHOLE,		// vector_flag
      1,			// modulo
      0,			// overlap
      data_spec_,		// data_spec
      0,			// alignment
      PAS_ATOMIC,		// atomic/split flag
      0,			// memory clearing flag
      &local_dist_,		// dist handle
      &local_pbuf_,		// pbuf handle
      &local_buffer_);		// PAS buffer
  VSIP_IMPL_CHECK_RC(rc,"pas_pbuf_create_1D");

  // Create replicated buffer
  rc = pas_pbuf_create_1D(
      tag_+1,			// tag
      pset_,			// procs
      num_elems,		// num_elems
      PAS_WHOLE,		// vector_flag
      1,			// modulo
      0,			// overlap
      data_spec_,		// data_spec
      0,			// alignment
      PAS_ATOMIC,		// atomic/split flag
      0,			// memory clearing flag
      &global_dist_,		// dist handle
      &global_pbuf_,		// pbuf handle
      &global_buffer_);		// PAS buffer

  VSIP_IMPL_CHECK_RC(rc,"pas_pbuf_create_1D");
}



inline
Broadcast::~Broadcast()
{
  long rc;

  rc = pas_distribution_destroy(local_dist_);  assert(rc == CE_SUCCESS);
  rc = pas_pbuf_destroy(local_pbuf_, 0);       assert(rc == CE_SUCCESS);
  rc = pas_buffer_destroy(local_buffer_);      assert(rc == CE_SUCCESS);
  rc = pas_distribution_destroy(global_dist_); assert(rc == CE_SUCCESS);
  rc = pas_pbuf_destroy(global_pbuf_, 0);      assert(rc == CE_SUCCESS);
  rc = pas_buffer_destroy(global_buffer_);     assert(rc == CE_SUCCESS);
}



template <typename T>
inline
void
Broadcast::operator()(
  processor_type proc,
  T&             value)
{
  long rc;
  long sem_index = 0;
  long pull_flags = 0;

  assert(sizeof(T) < num_elems*sizeof(data_type));

  pas::semaphore_give(proc, sem_index);

  if (local_processor() == proc)
  {
    *((T*)local_buffer_->virt_addr_list[0]) = value;

    pas::semaphore_take(pset_, sem_index);
    rc = pas_push(NULL, NULL,
		  local_pbuf_,
		  local_dist_,
		  global_pbuf_,
		  global_dist_,
		  data_spec_,
		  sem_index,
		  pull_flags | VSIP_IMPL_PAS_XFER_ENGINE |
		  VSIP_IMPL_PAS_SEM_GIVE_AFTER,
		  NULL); 
    assert(rc == CE_SUCCESS);
  }
  pas::semaphore_take(proc, sem_index);
  fflush(stdout);

  value = *((T*)global_buffer_->virt_addr_list[0]);
}

} // namespace vsip::impl::pas
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_PAS_BROADCAST_HPP
