//
// Copyright (c) 2005-2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_distributed_block_hpp_
#define ovxx_parallel_distributed_block_hpp_

#include <vsip/support.hpp>
#include <vsip/impl/map_fwd.hpp>
#include <ovxx/refcounted.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/domain_utils.hpp>
#include <ovxx/storage.hpp>
#include <ovxx/parallel/proxy_local_block.hpp>

namespace ovxx
{
namespace parallel
{

template <typename B, typename M>
class distributed_block : public ovxx::refcounted<distributed_block<B, M> >
{
protected:
  enum private_type {};
  typedef typename 
  detail::complex_value_type<typename B::value_type, private_type>::type uT;
public:
  static dimension_type const dim = B::dim;

  typedef typename B::value_type           value_type;
  typedef typename B::reference_type       reference_type;
  typedef typename B::const_reference_type const_reference_type;

  static storage_format_type const storage_format = get_block_layout<B>::storage_format;
  typedef storage_traits<value_type, storage_format> storage;
  typedef typename storage::ptr_type ptr_type;
  typedef typename storage::const_ptr_type const_ptr_type;
  typedef M map_type;

  typedef B local_block_type;
  typedef typename get_block_layout<local_block_type>::type local_layout;
  typedef proxy_local_block<dim, value_type, local_layout>  proxy_local_block_type;

  distributed_block(Domain<dim> const& dom, map_type const &map = map_type())
    : map_(map),
      proc_(local_processor()),
      sb_(map_.subblock(proc_)),
      subblock_(0)
  {
    map_.impl_apply(dom);
    for (dimension_type d=0; d<dim; ++d)
      size_[d] = dom[d].length();

    Domain<dim> sb_dom = 
      (sb_ != no_subblock) ? map_.template impl_subblock_domain<dim>(sb_)
                           : empty_domain<dim>();

    subblock_ = new local_block_type(sb_dom);
  }

  distributed_block(Domain<dim> const& dom, value_type value,
		    map_type const &map = map_type())
    : map_(map),
      proc_(local_processor()),
      sb_(map_.subblock(proc_)),
      subblock_(0)
  {
    map_.impl_apply(dom);
    for (dimension_type d=0; d<dim; ++d)
      size_[d] = dom[d].length();

    Domain<dim> sb_dom = 
      (sb_ != no_subblock) ? map_.template impl_subblock_domain<dim>(sb_)
                           : empty_domain<dim>();

    subblock_ = new local_block_type(sb_dom, value);
  }

  distributed_block(Domain<dim> const &dom, value_type *ptr,
		    map_type const &map = map_type())
    : map_(map),
      proc_(local_processor()),
      sb_(map_.subblock()),
      subblock_(0)
  {
    map_.impl_apply(dom);
    for (dimension_type d=0; d<dim; ++d)
      size_[d] = dom[d].length();

    Domain<dim> sb_dom = 
      (sb_ != no_subblock) ? map_.template impl_subblock_domain<dim>(sb_)
                           : empty_domain<dim>();

    subblock_ = new local_block_type(sb_dom, ptr);
  }

  distributed_block(Domain<dim> const &dom, uT *ptr,
		    map_type const &map = map_type())
    : map_(map),
      proc_(local_processor()),
      sb_(map_.subblock()),
      subblock_(0)
  {
    map_.impl_apply(dom);
    for (dimension_type d=0; d<dim; ++d)
      size_[d] = dom[d].length();

    Domain<dim> sb_dom = 
      (sb_ != no_subblock) ? map_.template impl_subblock_domain<dim>(sb_)
                           : empty_domain<dim>();

    subblock_ = new local_block_type(sb_dom, ptr);
  }

  distributed_block(Domain<dim> const &dom, uT *real_ptr, uT *imag_ptr,
		    map_type const &map = map_type())
    : map_(map),
      proc_(local_processor()),
      sb_(map_.subblock()),
      subblock_(0)
  {
    map_.impl_apply(dom);
    for (dimension_type d=0; d<dim; ++d)
      size_[d] = dom[d].length();

    Domain<dim> sb_dom = 
      (sb_ != no_subblock) ? map_.template impl_subblock_domain<dim>(sb_)
                           : empty_domain<dim>();

    subblock_ = new local_block_type(sb_dom, real_ptr, imag_ptr);
  }


  ~distributed_block()
  {
    if (subblock_)
    {
      subblock_->decrement_count();
    }
  }
    
  // get() on a distributed_block is a broadcast.  The processor
  // owning the index broadcasts the value to the other processors in
  // the data parallel group.
  value_type get(index_type idx) const VSIP_NOTHROW
  {
    // Optimize the no-communication case.
    if (is_same<map_type, Replicated_map<1> >::value &&
	map_.num_processors() == map_.impl_comm().size())
      return subblock_->get(idx);

    index_type     sb = map_.impl_subblock_from_global_index(Index<1>(idx));
    processor_type pr = *(map_.processor_begin(sb));
    value_type     val = value_type(); // avoid -Wall 'may not be initialized'

    if (pr == proc_)
    {
      OVXX_INVARIANT(map_.subblock() == sb);
      index_type lidx = map_.impl_local_from_global_index(0, idx);
      val = subblock_->get(lidx);
    }

    map_.impl_comm().broadcast(pr, &val, 1);

    return val;
  }

  value_type get(index_type idx0, index_type idx1) const VSIP_NOTHROW
  {
    // Optimize the no-communication case.
    if (is_same<map_type, Replicated_map<2> >::value &&
	map_.num_processors() == map_.impl_comm().size())
      return subblock_->get(idx0, idx1);

    index_type     sb = map_.impl_subblock_from_global_index(
				Index<2>(idx0, idx1));
    processor_type pr = *(map_.processor_begin(sb));
    value_type     val = value_type(); // avoid -Wall 'may not be initialized'

    if (pr == proc_)
    {
      OVXX_INVARIANT(map_.subblock() == sb);
      index_type l_idx0 = map_.impl_local_from_global_index(0, idx0);
      index_type l_idx1 = map_.impl_local_from_global_index(1, idx1);
      val = subblock_->get(l_idx0, l_idx1);
    }

    map_.impl_comm().broadcast(pr, &val, 1);

    return val;
  }

  value_type get(index_type idx0, index_type idx1, index_type idx2)
    const VSIP_NOTHROW
  {
    // Optimize the no-communication case.
    if (is_same<map_type, Replicated_map<3> >::value &&
	map_.num_processors() == map_.impl_comm().size())
      return subblock_->get(idx0, idx1, idx2);

    index_type     sb = map_.impl_subblock_from_global_index(
				Index<3>(idx0, idx1, idx2));
    processor_type pr = *(map_.processor_begin(sb));
    value_type     val = value_type(); // avoid -Wall 'may not be initialized'

    if (pr == proc_)
    {
      OVXX_INVARIANT(map_.subblock() == sb);
      index_type l_idx0 = map_.impl_local_from_global_index(0, idx0);
      index_type l_idx1 = map_.impl_local_from_global_index(1, idx1);
      index_type l_idx2 = map_.impl_local_from_global_index(2, idx2);
      val = subblock_->get(l_idx0, l_idx1, l_idx2);
    }

    map_.impl_comm().broadcast(pr, &val, 1);

    return val;
  }


  // put() on a distributed_block is executed only on the processor
  // owning the index.
  void put(index_type idx, value_type val) VSIP_NOTHROW
  {
    index_type     sb = map_.impl_subblock_from_global_index(Index<1>(idx));

    if (map_.subblock() == sb)
    {
      index_type lidx = map_.impl_local_from_global_index(0, idx);
      subblock_->put(lidx, val);
    }
  }

  void put(index_type idx0, index_type idx1, value_type val) VSIP_NOTHROW
  {
    index_type sb = map_.impl_subblock_from_global_index(Index<2>(idx0, idx1));

    if (map_.subblock() == sb)
    {
      index_type l_idx0 = map_.impl_local_from_global_index(0, idx0);
      index_type l_idx1 = map_.impl_local_from_global_index(1, idx1);
      subblock_->put(l_idx0, l_idx1, val);
    }
  }

  void put(index_type idx0, index_type idx1, index_type idx2, value_type val)
    VSIP_NOTHROW
  {
    index_type     sb = map_.impl_subblock_from_global_index(
				Index<3>(idx0, idx1, idx2));

    if (map_.subblock() == sb)
    {
      index_type l_idx0 = map_.impl_local_from_global_index(0, idx0);
      index_type l_idx1 = map_.impl_local_from_global_index(1, idx1);
      index_type l_idx2 = map_.impl_local_from_global_index(2, idx2);
      subblock_->put(l_idx0, l_idx1, l_idx2, val);
    }
  }


  ptr_type ptr() VSIP_NOTHROW
  { return subblock_->ptr();}

  const_ptr_type ptr() const VSIP_NOTHROW
  { return subblock_->ptr();}

  stride_type stride(dimension_type D, dimension_type d)
    const VSIP_NOTHROW
  { return subblock_->stride(D, d);}


  length_type size() const VSIP_NOTHROW
  {
    length_type size = 1;
    for (dimension_type d = 0; d < dim; ++d)
      size *= size_[d];
    return size;
  }
  length_type size(dimension_type block_dim OVXX_UNUSED, dimension_type d)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(block_dim == dim);
    OVXX_PRECONDITION(d < dim);
    return size_[d];
  }

  map_type const &map() const VSIP_NOTHROW 
  { return map_;}

  // length_type num_local_blocks() const { return num_subblocks_; }

  B &get_local_block() const
  {
    OVXX_PRECONDITION(subblock_ != 0);
    return *subblock_;
  }

  index_type subblock() const { return sb_;}

  void assert_local(index_type sb OVXX_UNUSED) const
  { OVXX_INVARIANT(sb == sb_ && subblock_ != 0);}

  void admit(bool update = true) VSIP_NOTHROW
  { subblock_->admit(update);}

  void release(bool update = true) VSIP_NOTHROW
  { subblock_->release(update);}

  void release(bool update, value_type *&pointer) VSIP_NOTHROW
  { subblock_->release(update, pointer);}
  void release(bool update, uT *&pointer) VSIP_NOTHROW
  { subblock_->release(update, pointer);}
  void release(bool update, uT *&real_pointer, uT *&imag_pointer) VSIP_NOTHROW
  { subblock_->release(update, real_pointer, imag_pointer);}

  void find(value_type*& pointer) VSIP_NOTHROW
  { subblock_->find(pointer);}
  void find(uT*& pointer) VSIP_NOTHROW
  { subblock_->find(pointer);}
  void find(uT*& real_pointer, uT*& imag_pointer) VSIP_NOTHROW
  { subblock_->find(real_pointer, imag_pointer);}

  void rebind(value_type* pointer) VSIP_NOTHROW
  { subblock_->rebind(pointer);}
  void rebind(uT* pointer) VSIP_NOTHROW
  { subblock_->rebind(pointer);}
  void rebind(uT* real_pointer, uT* imag_pointer) VSIP_NOTHROW
  { subblock_->rebind(real_pointer, imag_pointer);}

  user_storage_type user_storage() const VSIP_NOTHROW
  { return subblock_->user_storage();}

  bool admitted() const VSIP_NOTHROW
  { return subblock_->admitted();}

private:
  map_type map_;
  processor_type proc_;			// This processor in comm.
  index_type sb_;
  local_block_type *subblock_;
  length_type size_[dim];
};

template <typename B, typename M>
B &get_local_block(distributed_block<B, M> const &block)
{
  return block.get_local_block();
}

template <typename B, typename M>
void assert_local(distributed_block<B, M> const &block, index_type sb)
{
  block.assert_local(sb);
}

} // namespace ovxx::parallel

template <typename B, typename M>
struct distributed_local_block<parallel::distributed_block<B, M> >
{
  typedef typename parallel::distributed_block<B, M>::local_block_type type;
  typedef typename parallel::distributed_block<B, M>::proxy_local_block_type proxy_type;
};

template <typename B, typename M>
struct is_simple_distributed_block<parallel::distributed_block<B, M> >
{
  static bool const value = true;
};

} // namespace ovxx

namespace vsip
{

template <typename B, typename M>
struct get_block_layout<ovxx::parallel::distributed_block<B, M> > : get_block_layout<B> {};

template <typename B, typename M>
struct supports_dda<ovxx::parallel::distributed_block<B, M> > : supports_dda<B> {};

}

#endif
