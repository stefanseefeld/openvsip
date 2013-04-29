/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/pas/services.hpp
    @author  Jules Bergmann
    @date    2006-06-21
    @brief   VSIPL++ Library: Parallel Services: PAS

*/

#ifndef VSIP_OPT_PAS_SERVICES_HPP
#define VSIP_OPT_PAS_SERVICES_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

// Only par-services-xxx.hpp header should be included
#ifdef VSIP_IMPL_PAR_SERVICES_UNIQUE
#  error "Only one par-services-xxx.hpp should be included"
#endif
#define VSIP_IMPL_PAR_SERVICES_UNIQUE



/***********************************************************************
  Included Files
***********************************************************************/

#include <deque>
#include <vector>
#include <complex>
#include <memory>
#include <sstream>
#include <iostream>

extern "C" {
#include <pas.h>
}

#include <vsip/core/refcount.hpp>
#include <vsip/core/noncopyable.hpp>
#include <vsip/core/parallel/copy_chain.hpp>
#include <vsip/core/reductions/types.hpp>
#include <vsip/core/parallel/assign_fwd.hpp>
#include <vsip/core/argv_utils.hpp>
#include <vsip/opt/pas/param.hpp>
#include <vsip/opt/pas/broadcast.hpp>



/***********************************************************************
  Macros
***********************************************************************/

#define NET_TAG 0



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

typedef PAS_id          par_ll_pset_type;
typedef PAS_pbuf_handle par_ll_pbuf_type;

namespace pas
{

/// Traits class to determine MPI_DATATYPE from a C++ datatype

template <typename T>
struct Pas_datatype;

#define VSIP_IMPL_PASDATATYPE(CTYPE, PASTYPE)				\
template <>								\
struct Pas_datatype<CTYPE>						\
{									\
  static PAS_data_spec value() { return PASTYPE; }			\
};

VSIP_IMPL_PASDATATYPE(bool,                 PAS_DATA_REAL_U8);
VSIP_IMPL_PASDATATYPE(char,                 PAS_DATA_REAL_S8);
VSIP_IMPL_PASDATATYPE(unsigned char,        PAS_DATA_REAL_U8);
VSIP_IMPL_PASDATATYPE(signed char,          PAS_DATA_REAL_S8);
VSIP_IMPL_PASDATATYPE(short,                PAS_DATA_REAL_S16);
VSIP_IMPL_PASDATATYPE(unsigned short,       PAS_DATA_REAL_U16);
VSIP_IMPL_PASDATATYPE(int,                  PAS_DATA_REAL_S32);
VSIP_IMPL_PASDATATYPE(unsigned int,         PAS_DATA_REAL_U32);
VSIP_IMPL_PASDATATYPE(long,                 PAS_DATA_REAL_S32);
VSIP_IMPL_PASDATATYPE(unsigned long,        PAS_DATA_REAL_U32);
VSIP_IMPL_PASDATATYPE(float,                PAS_DATA_REAL_F32);
VSIP_IMPL_PASDATATYPE(double,               PAS_DATA_REAL_F64);
VSIP_IMPL_PASDATATYPE(std::complex<float>,  PAS_DATA_COMPLEX_F32);
VSIP_IMPL_PASDATATYPE(std::complex<double>, PAS_DATA_COMPLEX_F64);

#undef VSIP_IMPL_PASDATATYPE


/// DMA Chain builder.

class Chain_builder
{
public:
  Chain_builder()
    : chain_ ()
  {}

  ~Chain_builder()
  {}

  template <typename T>
  void add(ptrdiff_t start, int stride, unsigned length)
  {
    chain_.add(reinterpret_cast<void*>(start), sizeof(T), stride, length);
  }

  template <typename T>
  void add(ptrdiff_t start,
	   int stride0, unsigned length0,
	   int stride1, unsigned length1)
  {
    for (unsigned i=0; i<length1; ++i)
      chain_.add(reinterpret_cast<void*>(start + sizeof(T)*i*stride1),
		 sizeof(T), stride0, length0);
  }

  void* base() { return 0; }

  Copy_chain get_chain()
  { return chain_; }

  void stitch(void* base, Copy_chain chain)
  { chain_.append_offset(base, chain); }

  void stitch(std::pair<void*, void*> base, Copy_chain chain)
  {
    chain_.append_offset(base.first,  chain);
    chain_.append_offset(base.second, chain);
  }

  bool is_empty() const { return (chain_.size() == 0); }

  // Private member data.
private:
  Copy_chain                    chain_;
};



/// Communicator class.

/// A VSIPL++ Communicator is essentially just an MPI Communicator at
/// the moment.

class Communicator : Non_copyable
{

  struct Req_entry : public impl::Ref_count<Req_entry>
  {
    bool       done;

    Req_entry() : done(false) {}
  };
  class Req;
  friend class Req;
  class Req
  {
  public:
    Req() : entry_(new Req_entry, impl::noincrement) {}

    void set(bool val) { (*entry_).done = val; }
    bool get() { return (*entry_).done; }

  private:
    impl::Ref_counted_ptr<Req_entry> entry_;
  };
  struct Msg;
  friend struct Msg;
  struct Msg
  {
    Copy_chain  chain_;
    Req         req_;
    void*       memory_;

    Msg(Copy_chain chain, Req req, void* memory = 0)
      : chain_  (chain),
	req_    (req),
	memory_ (memory)
      {}
  };

  typedef std::deque<Msg> msg_list_type;
  struct Msg_list;
  friend struct Msg_list;
  struct Msg_list : public impl::Ref_count<Msg_list>
  {
    msg_list_type list_;
  };

public:
  typedef Req                         request_type;
  typedef Copy_chain                  chain_type;
  typedef std::vector<processor_type> pvec_type;

public:
  Communicator() : rank_(0), size_(0),
      msgs_ (new Msg_list, impl::noincrement), pvec_(0), bcast_(0) {}

  Communicator(long rank, long size)
    : rank_ (rank),
      size_ (size),
      msgs_ (new Msg_list, impl::noincrement),
      pvec_ (size_),
      bcast_(0)
  {
    for (index_type i=0; i<size_; ++i)
      pvec_[i] = static_cast<processor_type>(i);

    long* pnums;
    pnums = new long[size_+1];
    for (index_type i=0; i<size_; ++i)
      pnums[i] = pvec_[i];
    pnums[size_] = PAS_PNUMS_TERM;
    long rc = pas_pset_create(pnums, 0, &pset_);
    VSIP_IMPL_CHECK_RC(rc, "pas_pset_create");
    delete[] pnums;

    bcast_.reset(new pas::Broadcast(impl_ll_pset()));
  }

  void initialize(long rank, long size)
  {
    rank_ = rank;
    size_ = size;
    // msgs_ = new Msg_list, impl::noincrement;
    pvec_.resize(size_);

    for (index_type i=0; i<size_; ++i)
      pvec_[i] = static_cast<processor_type>(i);

    long* pnums;
    pnums = new long[size_+1];
    for (index_type i=0; i<size_; ++i)
      pnums[i] = pvec_[i];
    pnums[size_] = PAS_PNUMS_TERM;
    long rc = pas_pset_create(pnums, 0, &pset_);
    VSIP_IMPL_CHECK_RC(rc, "pas_pset_create");
    delete[] pnums;

    bcast_.reset(new pas::Broadcast(impl_ll_pset()));
  }

  void cleanup()
  {
    bcast_.reset(0);
    long rc = pas_pset_close(pset_, 0);
    VSIP_IMPL_CHECK_RC(rc, "pas_pset_close");
  }

  ~Communicator()
  {
  }

  processor_type   rank() const { return rank_; }
  length_type      size() const { return size_; }
  pvec_type const& pvec() const { return pvec_; }

  void barrier() const
  {
    long rc = pas_barrier_sync(pset_, 0, PAS_YIELD | PAS_DMA);
    VSIP_IMPL_CHECK_RC(rc, "pas_barrier_sync");
  }

  template <typename T>
  void buf_send(processor_type dest_proc, T* data, length_type size);

  template <typename T>
  void send(processor_type dest_proc, T* data, length_type size,
	    request_type& req);

  void send(processor_type dest, chain_type const& chain, request_type& req);

  template <typename T>
  void recv(processor_type src_proc, T* data, length_type size);

  void recv(processor_type dest, chain_type const& chain);

  void wait(request_type& req);

  template <typename T>
  void broadcast(processor_type root_proc, T* data, length_type size);

  template <typename T>
  T allreduce(reduction_type rdx, T value);

  friend bool operator==(Communicator const&, Communicator const&);

  par_ll_pset_type impl_ll_pset() const VSIP_NOTHROW { return pset_; }

private:
  processor_type	        rank_;
  length_type		        size_;
  Ref_counted_ptr<Msg_list>     msgs_;
  pvec_type                     pvec_;
  par_ll_pset_type              pset_;
  std::auto_ptr<pas::Broadcast> bcast_;
};

} // namespace vsip::impl::pas



typedef pas::Communicator     Communicator;
typedef pas::Chain_builder    Chain_builder;
typedef Direct_pas_assign     par_assign_impl_type;


inline void
create_ll_pset(
  std::vector<processor_type> const& pvec,
  par_ll_pset_type&                  pset)
{
  std::vector<processor_type> copy(pvec.size() + 1);

  length_type size = pvec.size();
  for (index_type i=0; i<size; ++i)
    copy[i] = pvec[i];
  copy[pvec.size()] = PAS_PNUMS_TERM;

  long rc = pas_pset_create(&copy[0], 0, &pset);
  VSIP_IMPL_CHECK_RC(rc, "pas_pset_create");
}



inline void
destroy_ll_pset(par_ll_pset_type& pset)
{
  long rc = pas_pset_close(pset, 0);
  VSIP_IMPL_CHECK_RC(rc, "destroy_ll_pset");
}



inline void
free_chain(Copy_chain const& /*chain*/)
{
}



/// Par_service class for when using PAS parallel services.

class Par_service
{
  // Compile-time values and typedefs.
public:
  typedef pas::Communicator communicator_type;

  // Constructors.
public:
  Par_service(int& argc, char**& argv)
    : valid_(1)
    {
      long rc;
      assert(valid_);

      long size = -1;
      long rank = -1;
      long heap = VSIP_IMPL_PAS_HEAP_SIZE;
      int i = 1;
      while (i < argc)
      {
	if (!strcmp(argv[i], "-pas_size"))
	{
	  size = atoi(argv[i+1]);
	  shift_argv(argc, argv, i, 2);
	}
	else if (!strcmp(argv[i], "-pas_rank"))
	{
	  rank = atoi(argv[i+1]);
	  shift_argv(argc, argv, i, 2);
	}
	else if (!strcmp(argv[i], "-pas_heap"))
	{
          std::istringstream iss(argv[i+1]);
          iss >> std::hex >> heap;
          if (!iss)
          {
            heap = -1;
            std::cerr << "Error reading hex value for -pas_heap" << std::endl;
          }
	  shift_argv(argc, argv, i, 2);
	}
	else
	  i += 1;
      }

      if (rank == -1 || size == -1 || heap == -1)
      {
	printf("Usage: runmc <ceid> %s -pas_size <size> -pas_rank <rank>"
               " [-pas_heap <heap-size>]\n", argv[0]);
	exit(1);
      }

      printf("INIT: PAS rank/size %ld %ld\n", rank, size);
      fflush(stdout);

      rc = pas_net_create(NET_TAG, rank, size, &net_handle_);
      VSIP_IMPL_CHECK_RC(rc,"pas_net_create");

#if VSIP_IMPL_PAS_XR
#  if VSIP_IMPL_PAS_USE_INTERRUPT()
      rc = pas_net_enable_semaphore_interrupts (net_handle_);
      VSIP_IMPL_CHECK_RC(rc, "pas_net_enable_semaphore_interrupts");
#  endif
#endif

      /* override some PAS defaults */
      rc = pas_net_set_heap_size(net_handle_, heap);
      VSIP_IMPL_CHECK_RC(rc,"pas_net_set_heap_size");
      rc = pas_net_set_num_semaphores(net_handle_, 2);
      VSIP_IMPL_CHECK_RC(rc,"pas_net_setnum_semaphores");


#if VSIP_IMPL_PAS_XR
#  if VSIP_IMPL_PAS_XR_SET_PORTNUM
      rc = pas_net_set_tr_base_portnum(net_handle_, 3939);
      VSIP_IMPL_CHECK_RC(rc, "pas_net_set_tr_base_portnum");
#  endif

#  if VSIP_IMPL_PAS_XR_SET_ADAPTERNAME
      rc = pas_net_set_tr_adapter_name(net_handle_,
				       VSIP_IMPL_PAS_XR_ADAPTERNAME);
      VSIP_IMPL_CHECK_RC(rc, "pas_net_set_tr_adapter_name");
#  endif

#  if VSIP_IMPL_PAS_XR_SET_SHMKEY
      rc = pas_net_set_tr_nodedb_shmkey (net_handle_, VSIP_IMPL_PAS_XR_SHMKEY);
      VSIP_IMPL_CHECK_RC(rc, "pas_net_set_tr_nodedb_shmkey");
#  endif

#  if VSIP_IMPL_PAS_XR_SET_PIR
      rc = pas_net_set_tr_num_prepost_intr_recvs (net_handle_, 32);
      VSIP_IMPL_CHECK_RC(rc, "pas_net_set_tr_num_prepost_intr_recvs");
#  endif

#  if VSIP_IMPL_PAS_XR_SET_RMD
      rc = pas_net_set_tr_rdma_multibuffer_depth (net_handle_, 32);
      VSIP_IMPL_CHECK_RC(rc, "pas_net_set_tr_rdma_multibuffer_depth");
#  endif

#  if VSIP_IMPL_PAS_XR_SET_UDAPL_MAX_RECVS
      rc = pas_net_set_tr_udapl_max_recvs (net_handle_, VSIP_IMPL_PAS_XR_PAS_UDAPL_MAX_RECVS);
      VSIP_IMPL_CHECK_RC(rc, "pas_net_set_tr_udapl_max_recvs");
#  endif

#  if VSIP_IMPL_PAS_XR_SET_UDAPL_MAX_REQUESTS
      rc = pas_net_set_tr_udapl_max_requests (net_handle_, 
					      VSIP_IMPL_PAS_XR_PAS_UDAPL_MAX_REQUESTS);
      VSIP_IMPL_CHECK_RC(rc, "pas_net_set_tr_udapl_max_requests");
#  endif
#endif

#if 0 // Enable this to print debug info about the PAS net.
      char  msg[80];
      sprintf(msg, "CE%ld ", /*ce_getid(),*/ rank);
      rc =  pas_net_print_info(net_handle_, msg, NULL, 0);
      VSIP_IMPL_CHECK_RC(rc,"pas_net_print_info");
#endif

      /* now open the PAS network */
      rc = pas_net_open(net_handle_);
      VSIP_IMPL_CHECK_RC(rc,"pas_net_open");

#if VSIP_IMPL_PAS_SHARE_DYNAMIC_XFER == 1
      rc = pas_dynamic_xfer_create(size, 3, 0, &dynamic_xfer_);
      VSIP_IMPL_CHECK_RC(rc, "pas_dynamic_xfer_create");
#endif

      default_communicator_.initialize(rank, size);
    }

  ~Par_service()
    {
      long rc;
#if VSIP_IMPL_PAS_SHARE_DYNAMIC_XFER == 1
      rc = pas_dynamic_xfer_destroy(dynamic_xfer_, 0);
      VSIP_IMPL_CHECK_RC(rc, "pas_dynamic_xfer_destroy");
#endif
      default_communicator_.cleanup();
      rc = pas_net_close(net_handle_);
      VSIP_IMPL_CHECK_RC(rc, "pas_net_close");
      valid_ = 0;
    }

  static communicator_type& default_communicator()
    {
      return default_communicator_;
    }

#if VSIP_IMPL_PAS_SHARE_DYNAMIC_XFER == 1
  PAS_dynamic_xfer_handle dynamic_xfer() const
  { return dynamic_xfer_; }
#endif

private:
  static communicator_type default_communicator_;

  int			   valid_;
  PAS_net_handle           net_handle_;
#if VSIP_IMPL_PAS_SHARE_DYNAMIC_XFER == 1
  PAS_dynamic_xfer_handle dynamic_xfer_;
#endif
};



template <reduction_type rtype,
	  typename       T>
struct Reduction_supported
{ static bool const value = false; };



/***********************************************************************
  Definitions
***********************************************************************/

namespace pas
{

inline bool
operator==(Communicator const& comm1, Communicator const& comm2)
{
  return comm1.msgs_.get() == comm2.msgs_.get();
}



inline bool
operator!=(Communicator const& comm1, Communicator const& comm2)
{
  return !operator==(comm1, comm2);
}



template <typename T>
inline void
Communicator::buf_send(
  processor_type dest_proc,
  T*             data,
  length_type    size)
{
  assert(0);
  assert(dest_proc == 0);

  T* raw = new T[size];

  for (index_type i=0; i<size; ++i)
    raw[i] = data[i];

  // printf("buf_send %d (val %d) %08x\n", size, *raw, (int)raw);
  Copy_chain chain;
  chain.add(reinterpret_cast<void*>(raw), sizeof(T), 1, size);

  msgs_->list_.push_back(Msg(chain, Req(), reinterpret_cast<void*>(raw)));
}



template <typename T>
inline void
Communicator::send(
  processor_type dest_proc,
  T*             data,
  length_type    size,
  request_type&  req)
{
  assert(0);
  assert(dest_proc == 0);

  // printf("send %d (val %d)\n", size, *data);
  Copy_chain chain;
  chain.add(reinterpret_cast<void*>(data), sizeof(T), 1, size);

  req.set(false);
  msgs_->list_.push_back(Msg(chain, req));
}



inline void
Communicator::send(
  processor_type    dest_proc,
  chain_type const& chain,
  request_type&     req)
{
  assert(0);
  assert(dest_proc == 0);

  // printf("send chain\n");
  req.set(false);
  msgs_->list_.push_back(Msg(chain, req));
}



template <typename T>
inline void
Communicator::recv(
  processor_type   src_proc,
  T*               data,
  length_type      size)
{
  assert(0);
  assert(src_proc == 0);
  assert(msgs_->list_.size() > 0);

  Msg msg = msgs_->list_.front();
  msgs_->list_.pop_front();

  // assert(msg.type_size_ == sizeof(T));
  assert(msg.chain_.data_size() == size * sizeof(T));

  msg.chain_.copy_into(data, size * sizeof(T));

  msg.req_.set(true);

  if (msg.memory_) delete[] static_cast<char*>(msg.memory_);
}



inline void
Communicator::recv(
  processor_type    src_proc,
  chain_type const& chain)
{
  assert(0);
  assert(src_proc == 0);
  assert(msgs_->list_.size() > 0);

  Msg msg = msgs_->list_.front();
  msgs_->list_.pop_front();

  // assert(msg.type_size_ == sizeof(T));
  assert(msg.chain_.data_size() == chain.data_size());

  msg.chain_.copy_into(chain);

  msg.req_.set(true);

  if (msg.memory_) delete[] static_cast<char*>(msg.memory_);
}



/// Wait for a previous communication (send or receive) to complete.

inline void
Communicator::wait(
  request_type& req)
{
  assert(0);
  // Since there is only one processor, we really can't wait for the
  // receive to post.  Either it has, or it hasn't, in which case
  // we are deadlocked.
  assert(req.get() == true);
}



/// Broadcast a value from root processor to other processors.

template <typename T>
inline void
Communicator::broadcast(processor_type root_proc, T* value, length_type len)
{
  assert(bcast_.get() != 0);
  for (index_type i=0; i<len; ++i)
    bcast_->operator()(root_proc, value[i]);
}



/// Reduce a value from all processors to all processors.

template <typename T>
inline T
Communicator::allreduce(reduction_type, T value)
{
  assert(0);
  return value;
}






} // namespace vsip::impl::pas
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_PAS_SERVICES_HPP
