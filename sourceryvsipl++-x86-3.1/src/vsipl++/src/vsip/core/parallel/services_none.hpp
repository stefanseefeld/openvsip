/* Copyright (c) 2005, 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/parallel/none.hpp
    @author  Jules Bergmann
    @date    2005-03-30
    @brief   VSIPL++ Library: Parallel Services: no services

*/

#ifndef VSIP_IMPL_PAR_SERVICES_NONE_HPP
#define VSIP_IMPL_PAR_SERVICES_NONE_HPP

// Only one parallel/xxx.hpp header should be included
#ifdef VSIP_IMPL_PAR_SERVICES_UNIQUE
#  error "Only one parallel/xxx.hpp should be included"
#endif
#define VSIP_IMPL_PAR_SERVICES_UNIQUE



/***********************************************************************
  Included Files
***********************************************************************/

#include <deque>
#include <vector>

#include <vsip/core/refcount.hpp>
#include <vsip/core/parallel/copy_chain.hpp>
#include <vsip/core/reductions/types.hpp>
#include <vsip/core/parallel/assign_fwd.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

typedef int par_ll_pbuf_type;
typedef int par_ll_pset_type;

namespace par_services_none
{


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

class Communicator
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
  Communicator()
    : rank_ (0),
      size_ (1),
      msgs_ (new Msg_list, impl::noincrement),
      pvec_ (size_)
  {
    for (index_type i=0; i<size_; ++i)
      pvec_[i] = static_cast<processor_type>(i);
  }

  processor_type   rank() const { return rank_; }
  length_type      size() const { return size_; }
  pvec_type const& pvec() const { return pvec_; }

  // barrier is no-op for serial execution.
  void barrier() const {}

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

  par_ll_pset_type impl_ll_pset() const VSIP_NOTHROW
  { return par_ll_pset_type(); }

  friend bool operator==(Communicator const&, Communicator const&);

private:
  processor_type	    rank_;
  length_type		    size_;
  Ref_counted_ptr<Msg_list> msgs_;
  pvec_type                 pvec_;
};

} // namespace vsip::impl::par_services



typedef par_services_none::Communicator Communicator;
typedef par_services_none::Chain_builder Chain_builder;
typedef Chained_assign par_assign_impl_type;

inline void
create_ll_pset(
  std::vector<processor_type> const&,
  par_ll_pset_type&)
{}

inline void
destroy_ll_pset(par_ll_pset_type&)
{}

inline void
free_chain(Copy_chain const& /*chain*/)
{
}


/// Par_service class for when no services are available.

class Par_service
{
  // Compile-time values and typedefs.
public:
  typedef par_services_none::Communicator communicator_type;

  // Constructors.
public:
  Par_service(int&, char**&)
    : valid_(1)
    {
      assert(valid_);

      default_communicator_ = communicator_type();
    }

  ~Par_service()
    {
      valid_ = 0;
    }

  static communicator_type& default_communicator()
  {
    return default_communicator_;
  }

private:
  static communicator_type default_communicator_;

  int			   valid_;
};



template <reduction_type rtype,
	  typename       T>
struct Reduction_supported
{ static bool const value = true; };



/***********************************************************************
  Definitions
***********************************************************************/

namespace par_services_none
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
  processor_type dest_proc ATTRIBUTE_UNUSED,
  T*             data,
  length_type    size,
  request_type&  req)
{
  assert(dest_proc == 0);

  // printf("send %d (val %d)\n", size, *data);
  Copy_chain chain;
  chain.add(reinterpret_cast<void*>(data), sizeof(T), 1, size);

  req.set(false);
  msgs_->list_.push_back(Msg(chain, req));
}



inline void
Communicator::send(
  processor_type    dest_proc ATTRIBUTE_UNUSED,
  chain_type const& chain,
  request_type&     req)
{
  assert(dest_proc == 0);

  // printf("send chain\n");
  req.set(false);
  msgs_->list_.push_back(Msg(chain, req));
}



template <typename T>
inline void
Communicator::recv(
  processor_type   src_proc ATTRIBUTE_UNUSED,
  T*               data,
  length_type      size)
{
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
  processor_type    src_proc ATTRIBUTE_UNUSED,
  chain_type const& chain)
{
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
  request_type& req ATTRIBUTE_UNUSED)
{
  // Since there is only one processor, we really can't wait for the
  // receive to post.  Either it has, or it hasn't, in which case
  // we are deadlocked.
  assert(req.get() == true);
}



/// Broadcast a value from root processor to other processors.

template <typename T>
inline void
Communicator::broadcast(processor_type root_proc ATTRIBUTE_UNUSED, T*, length_type)
{
  assert(root_proc == 0);
  // No-op: no need to broadcast w/one processor.
}



/// Reduce a value from all processors to all processors.

template <typename T>
inline T
Communicator::allreduce(reduction_type, T value)
{
  return value;
}

} // namespace vsip::impl::par_services_none
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_PAR_SERVICES_NONE_HPP
