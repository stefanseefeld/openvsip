//
// Copyright (c) 2005-2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_core_mpi_distributed_block_hpp_
#define vsip_core_mpi_distributed_block_hpp_

#include <vsip/support.hpp>
#include <vsip/core/map_fwd.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/user_storage.hpp>
#include <vsip/core/parallel/get_local_view.hpp>
#include <vsip/core/parallel/proxy_local_block.hpp>

namespace vsip
{
namespace impl
{
namespace mpi
{

template <typename Block,
	  typename Map>
class Distributed_block
  : public impl::Ref_count<Distributed_block<Block, Map> >
{
  // Compile-time values and types.
public:
  static dimension_type const dim = Block::dim;

  typedef typename Block::value_type           value_type;
  typedef typename Block::reference_type       reference_type;
  typedef typename Block::const_reference_type const_reference_type;

  static storage_format_type const storage_format = get_block_layout<Block>::storage_format;
  typedef Storage<storage_format, value_type>     impl_storage_type;
  typedef typename impl_storage_type::type           ptr_type;
  typedef typename impl_storage_type::const_type     const_ptr_type;

  typedef Map                                  map_type;

  // Non-standard typedefs:
  typedef Block                                local_block_type;
  typedef typename get_block_layout<local_block_type>::type local_LP;
  typedef Proxy_local_block<dim, value_type, local_LP>  proxy_local_block_type;

  // Private compile-time values and types.
private:
  enum private_type {};
  typedef typename impl::Complex_value_type<value_type, private_type>::ptr_type
          uT_ptr;

  // Constructors and destructor.
public:
  typedef typename impl::Complex_value_type<value_type, private_type>::type uT;

  Distributed_block(Domain<dim> const& dom, Map const& map = Map())
    : map_           (map),
      proc_          (local_processor()),
      sb_            (map_.subblock(proc_)),
      subblock_      (NULL)
  {
    map_.impl_apply(dom);
    for (dimension_type d=0; d<dim; ++d)
      size_[d] = dom[d].length();

    Domain<dim> sb_dom = 
      (sb_ != no_subblock) ? map_.template impl_subblock_domain<dim>(sb_)
                           : empty_domain<dim>();

    subblock_ = new Block(sb_dom);
  }

  Distributed_block(Domain<dim> const& dom, value_type value,
		    Map const& map = Map())
    : map_           (map),
      proc_          (local_processor()),
      sb_            (map_.subblock(proc_)),
      subblock_      (NULL)
  {
    map_.impl_apply(dom);
    for (dimension_type d=0; d<dim; ++d)
      size_[d] = dom[d].length();

    Domain<dim> sb_dom = 
      (sb_ != no_subblock) ? map_.template impl_subblock_domain<dim>(sb_)
                           : empty_domain<dim>();

    subblock_ = new Block(sb_dom, value);
  }

  Distributed_block(
    Domain<dim> const& dom, 
    value_type* const  ptr,
    Map const&         map = Map())
    : map_           (map),
      proc_          (local_processor()),
      sb_            (map_.subblock()),
      subblock_      (NULL)
  {
    map_.impl_apply(dom);
    for (dimension_type d=0; d<dim; ++d)
      size_[d] = dom[d].length();

    Domain<dim> sb_dom = 
      (sb_ != no_subblock) ? map_.template impl_subblock_domain<dim>(sb_)
                           : empty_domain<dim>();

    subblock_ = new Block(sb_dom, ptr);
  }

  Distributed_block(
    Domain<dim> const& dom, 
    uT_ptr const       pointer,
    Map const&         map = Map())
    : map_           (map),
      proc_          (local_processor()),
      sb_            (map_.subblock()),
      subblock_      (NULL)
  {
    map_.impl_apply(dom);
    for (dimension_type d=0; d<dim; ++d)
      size_[d] = dom[d].length();

    Domain<dim> sb_dom = 
      (sb_ != no_subblock) ? map_.template impl_subblock_domain<dim>(sb_)
                           : empty_domain<dim>();

    subblock_ = new Block(sb_dom, pointer);
  }

  Distributed_block(
    Domain<dim> const& dom, 
    uT_ptr const       real_pointer,
    uT_ptr const       imag_pointer,
    Map const&         map = Map())
    : map_           (map),
      proc_          (local_processor()),
      sb_            (map_.subblock()),
      subblock_      (NULL)
  {
    map_.impl_apply(dom);
    for (dimension_type d=0; d<dim; ++d)
      size_[d] = dom[d].length();

    Domain<dim> sb_dom = 
      (sb_ != no_subblock) ? map_.template impl_subblock_domain<dim>(sb_)
                           : empty_domain<dim>();

    subblock_ = new Block(sb_dom, real_pointer, imag_pointer);
  }


  ~Distributed_block()
  {
    if (subblock_)
    {
      // PROFILE: issue a warning if subblock is captured.
      subblock_->decrement_count();
    }
  }
    
  // Data accessors.
public:
  // get() on a distributed_block is a broadcast.  The processor
  // owning the index broadcasts the value to the other processors in
  // the data parallel group.
  value_type get(index_type idx) const VSIP_NOTHROW
  {
    // Optimize uni-processor and replicated cases.
    if (    is_same<Map, Global_map<1> >::value
        ||  vsip::num_processors() == 1
	|| (   is_same<Map, Replicated_map<1> >::value
	    && map_.subblock() != no_subblock))
      return subblock_->get(idx);

    index_type     sb = map_.impl_subblock_from_global_index(Index<1>(idx));
    processor_type pr = *(map_.processor_begin(sb));
    value_type     val = value_type(); // avoid -Wall 'may not be initialized'

    if (pr == proc_)
    {
      assert(map_.subblock() == sb);
      index_type lidx = map_.impl_local_from_global_index(0, idx);
      val = subblock_->get(lidx);
    }

    map_.impl_comm().broadcast(pr, &val, 1);

    return val;
  }

  value_type get(index_type idx0, index_type idx1) const VSIP_NOTHROW
  {
    // Optimize uni-processor and replicated cases.
    if (    is_same<Map, Global_map<2> >::value
        ||  vsip::num_processors() == 1
	|| (   is_same<Map, Replicated_map<2> >::value
	    && map_.subblock() != no_subblock))
      return subblock_->get(idx0, idx1);

    index_type     sb = map_.impl_subblock_from_global_index(
				Index<2>(idx0, idx1));
    processor_type pr = *(map_.processor_begin(sb));
    value_type     val = value_type(); // avoid -Wall 'may not be initialized'

    if (pr == proc_)
    {
      assert(map_.subblock() == sb);
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
    // Optimize uni-processor and replicated cases.
    if (    is_same<Map, Global_map<3> >::value
        ||  vsip::num_processors() == 1
	|| (   is_same<Map, Replicated_map<3> >::value
	    && map_.subblock() != no_subblock))
      return subblock_->get(idx0, idx1, idx2);

    index_type     sb = map_.impl_subblock_from_global_index(
				Index<3>(idx0, idx1, idx2));
    processor_type pr = *(map_.processor_begin(sb));
    value_type     val = value_type(); // avoid -Wall 'may not be initialized'

    if (pr == proc_)
    {
      assert(map_.subblock() == sb);
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


  // Support Direct_data interface.
public:
  ptr_type ptr() VSIP_NOTHROW
  { return subblock_->ptr();}

  const_ptr_type ptr() const VSIP_NOTHROW
  { return subblock_->ptr();}

  stride_type stride(dimension_type D, dimension_type d)
    const VSIP_NOTHROW
  { return subblock_->stride(D, d);}


  // Accessors.
public:
  length_type size() const VSIP_NOTHROW
  {
    length_type size = 1;
    for (dimension_type d = 0; d < dim; ++d)
      size *= size_[d];
    return size;
  }
  length_type size(dimension_type block_dim ATTRIBUTE_UNUSED, dimension_type d)
    const VSIP_NOTHROW
  {
    assert(block_dim == dim);
    assert(d < dim);
    return size_[d];
  }

  map_type const& map() const VSIP_NOTHROW 
    { return map_; }

  // length_type num_local_blocks() const { return num_subblocks_; }

  Block& get_local_block() const
  {
    assert(subblock_ != NULL);
    return *subblock_;
  }

  index_type subblock() const { return sb_; }

  void assert_local(index_type sb ATTRIBUTE_UNUSED) const
    { assert(sb == sb_ && subblock_ != NULL); }

  // User storage functions.
public:
  void admit(bool update = true) VSIP_NOTHROW
    { subblock_->admit(update); }

  void release(bool update = true) VSIP_NOTHROW
    { subblock_->release(update); }

  void release(bool update, value_type*& pointer) VSIP_NOTHROW
    { subblock_->release(update, pointer); }
  void release(bool update, uT*& pointer) VSIP_NOTHROW
    { subblock_->release(update, pointer); }
  void release(bool update, uT*& real_pointer, uT*& imag_pointer) VSIP_NOTHROW
    { subblock_->release(update, real_pointer, imag_pointer); }

  void find(value_type*& pointer) VSIP_NOTHROW
    { subblock_->find(pointer); }
  void find(uT*& pointer) VSIP_NOTHROW
    { subblock_->find(pointer); }
  void find(uT*& real_pointer, uT*& imag_pointer) VSIP_NOTHROW
    { subblock_->find(real_pointer, imag_pointer); }

  void rebind(value_type* pointer) VSIP_NOTHROW
    { subblock_->rebind(pointer); }
  void rebind(uT* pointer) VSIP_NOTHROW
    { subblock_->rebind(pointer); }
  void rebind(uT* real_pointer, uT* imag_pointer) VSIP_NOTHROW
    { subblock_->rebind(real_pointer, imag_pointer); }

  enum user_storage_type user_storage() const VSIP_NOTHROW
  { return subblock_->user_storage(); }

  bool admitted() const VSIP_NOTHROW
    { return subblock_->admitted(); }

  // Member data.
public:
  map_type              map_;
  processor_type	proc_;			// This processor in comm.
  index_type   		sb_;
  Block*		subblock_;
  length_type	        size_[dim];
};

} // namespace vsip::impl::mpi
} // namespace vsip::impl
} // namespace vsip


#endif
