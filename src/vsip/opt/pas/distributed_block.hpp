/* Copyright (c) 2006-2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef vsip_opt_pas_distributed_block_hpp_
#define vsip_opt_pas_distributed_block_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

extern "C" {
#include <pas.h>
}

#include <vsip/support.hpp>
#include <vsip/core/order_traits.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/user_storage.hpp>
#include <vsip/core/parallel/dist.hpp>
#include <vsip/core/parallel/get_local_view.hpp>
#include <vsip/core/parallel/proxy_local_block.hpp>
#include <vsip/opt/pas/offset.hpp>

#define PAS_BLOCK_VERBOSE 0



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{

namespace pas
{




template <typename       T,
	  typename       OrderT,
	  typename       ComplexFmt,
	  dimension_type Dim,
	  typename       MapT>
void
pbuf_create(
  long                     tag,
  Domain<Dim> const&       dom,
  MapT const&              map,
  PAS_global_data_handle&  gdo_handle,
  PAS_distribution_handle& dist_handle,
  PAS_pbuf_handle&         pbuf_handle,
  PAS_buffer**             buffer)
{
  PAS_data_spec        data_spec = pas::Pas_datatype<T>::value();
  unsigned long const  alignment = VSIP_IMPL_PAS_ALIGNMENT;
  long                 rc;

  long const           no_flag = 0;

  // gdo_dim can either be Dim, the actual dimension of the block,
  // or VSIP_MAX_DIMENSION.  When gdo_dim is VSIP_MAX_DIMENSION,
  // dimensions beyound Dim are given size 1.
  const dimension_type gdo_dim = Dim;

  long                 dim_sizes[gdo_dim];
  long                 layout[gdo_dim];
  long                 group_dims[gdo_dim];
  long                 prod_group_dims = 1;
  PAS_layout_handle    layout_handle[gdo_dim];
  PAS_partition_handle partition[gdo_dim];
  PAS_overlap_handle   zero_overlap;
  long                 distribution_flag;
  long                 components;

  if (is_same<ComplexFmt, split_complex>::value &&
      is_complex<T>::value)
  {
    components = 2;
    distribution_flag = PAS_SPLIT;
  }
  else
  {
    components = 1;
    distribution_flag = PAS_ATOMIC;
  }

  pas_overlap_create(
    PAS_OVERLAP_PAD_ZEROS, 
    0,				// num_positions
    0,				// flags: reserved, set to 0
    &zero_overlap);

  for (dimension_type d=0; d<gdo_dim; ++d)
  {
    dimension_type dim_order = (d == 0) ? Dim_of<OrderT, 0>::value :
                               (d == 1) ? Dim_of<OrderT, 1>::value
                                        : Dim_of<OrderT, 2>::value;
    if (d < Dim)
    {
      dim_sizes[d]   = static_cast<long>(dom[d].length());
      layout[d]      = Dim-dim_order-1;
      group_dims[d]  = map.num_subblocks(d);
    }
    else
    {
      dim_sizes[d]   = 1;
      layout[d]      = d;
      group_dims[d]  = 1;
    }

    if (d >= Dim || map.distribution(d) == whole)
    {
      assert(group_dims[d] == 1);
      rc = pas_partition_whole_create(
	0,			// flags: reserved, set to 0
	&partition[d]);		// returned partition handle
      VSIP_IMPL_CHECK_RC(rc,"pas_partition_block_create");
    }
    else if (map.distribution(d) == block)
    {
      rc = pas_partition_block_create(
	1,			// minimum number of elements in partition
	1,			// modulo
	zero_overlap,		// before:
	zero_overlap,		// after :
	0,			// flags: reserved, set to 0
	&partition[d]);		// returned partition handle
      VSIP_IMPL_CHECK_RC(rc,"pas_partition_block_create");

      // Adjust group_dims if the last block would be size 0.
      // PAS wants to replicate the first block onto the last
      // processor in this case.
      if (dim_sizes[d] < group_dims[d])
	group_dims[d] = dim_sizes[d];
      else if (segment_size(dim_sizes[d], group_dims[d], group_dims[d]-1) == 0)
	group_dims[d] -= 1;


      assert(group_dims[d] > 0);
    }
    else
      VSIP_IMPL_THROW(unimplemented("block-cyclic not implemented for PAS"));

    rc = pas_layout_packed_create(
      layout[d], // number of dimensions more packed than this one
      0, 0, 0, 0, no_flag,
      &layout_handle[d]);

    prod_group_dims *= group_dims[d];
  }

  long real_num_procs;
  rc = pas_pset_get_npnums(map.impl_ll_pset(), &real_num_procs);
  VSIP_IMPL_CHECK_RC(rc, "pas_pset_get_npnums");

  // Check that we've dropped the same number of processors
  // from both the pset and the group dims.
  assert(real_num_procs == prod_group_dims ||
	 prod_group_dims == 1 && map.num_subblocks() == 1
	 && map.num_processors() == static_cast<length_type>(real_num_procs));

  rc = pas_global_data_create(
    static_cast<long>(gdo_dim),	// number of dimensions
    dim_sizes,			// array of dimension sizes
    0,				// flags: reserved, set to 0
    &gdo_handle);		// returnd GDO handle
  VSIP_IMPL_CHECK_RC(rc,"pas_global_data_create");

  rc = pas_distribution_create(
    gdo_handle,			// global data object handle
    real_num_procs,		// number of processors
    group_dims,			// array of num of procs per dim
    partition,			// array of partition spec per dim
    layout_handle,		// layout spec
    0,				// buffer_offset
    distribution_flag,		// ATOMIC or SPLIT
    &dist_handle);		// returned distribution handle
  VSIP_IMPL_CHECK_RC(rc,"pas_distribution_create");

  long local_nbytes;
  rc = pas_distribution_calc_local_nbytes(
    dist_handle,		// distribution handle
    data_spec,			// data type spec
    no_flag,			// flags: reserved, set to 0
    &local_nbytes);		// returned number of bytes for buffer
  VSIP_IMPL_CHECK_RC(rc,"pas_distribution_calc_local_nbytes");

  rc = pas_pbuf_create(
    tag,			// buffer tag
    map.impl_ll_pset(),		// process set
    local_nbytes,		// allocation size
    alignment,			// alignment
    components,			// max split buffer components
    PAS_ZERO,			// flags
    &pbuf_handle);		// returned pbuf handle
  VSIP_IMPL_CHECK_RC(rc,"pas_pbuf_create");


  // Allocate the buffer.  If the local processor is not part of
  // the pset, don't allocate a buffer.
  if (pas_pset_is_member(map.impl_ll_pset()))
  {
    assert(map.subblock() != no_subblock);
    rc = pas_buffer_alloc(
      pbuf_handle,		// pbuf (partitioned buffer) handle
      map.impl_ll_pset(),	// process set
      dist_handle,		// distribution handle
      data_spec,		// data type spec
      PAS_MY_RANK,		// rank in pset
      no_flag,			// flags: reserved, set to 0
      0,			// channel for mapping
      buffer);			// allocated buffer ptr
    VSIP_IMPL_CHECK_RC(rc,"pas_buffer_alloc");

    if (distribution_flag == PAS_ATOMIC)
      assert((*buffer)->num_virt_addrs == 1);
    else
      assert((*buffer)->num_virt_addrs == 2);
  }
  else
    *buffer = NULL;


  // Cleanup temporary handles. 
  rc = pas_overlap_destroy(zero_overlap);
  VSIP_IMPL_CHECK_RC(rc,"overlap_destroy");
  for (dimension_type d=0; d<gdo_dim; ++d)
  {
    rc = pas_partition_destroy(partition[d]);
    VSIP_IMPL_CHECK_RC(rc,"partition_destroy");
    rc = pas_layout_destroy(layout_handle[d]);
    VSIP_IMPL_CHECK_RC(rc,"layout_destroy");
  }
}

extern long             global_tag;

template <typename L>
struct Distributed_block_unimplemented_for;

template <typename Block,
	  typename Map,
	  typename T = typename Block::value_type,
	  typename L = get_block_layout<Block>::type>
class Distributed_block : Distributed_block_unimplemented_for<L> {};

template <typename B, typename M, typename O>
class Distributed_block<B, M, typename B::value_type,
  Layout<B::dim, O, dense, dense_complex_format> >
  : public Ref_count<Distributed_block<B, M, typename B::value_type,
  Layout<B::dim, O, dense, dense_complex_format> > >
{
  // Compile-time values and types.
public:
  static dimension_type const dim = B::dim;

  typedef typename B::value_type T;
  typedef T value_type;
  typedef T &reference_type;
  typedef T const &const_reference_type;

  typedef O order_type;
  typedef M map_type;

  // Non-standard typedefs:
  typedef B<dim, value_type, O, Local_map>           local_block_type;

  typedef typename get_block_layout<local_block_type>::storage_format
                                                     impl_storage_format;
  typedef Storage<impl_storage_format, value_type>     impl_storage_type;
  typedef typename impl_storage_type::type           ptr_type;
  typedef typename impl_storage_type::const_type     const_ptr_type;


  typedef typename get_block_layout<local_block_type>::type local_LP;
  typedef Proxy_local_block<dim, value_type, local_LP>        proxy_local_block_type;

  static long const components =
    (is_same<impl_storage_format, split_complex>::value &&
     is_complex<T>::value)
    ? 2 : 1;

  // Private compile-time values and types.
private:
  enum private_type {};

  void init(Domain<dim> const& dom)
  {
    Domain<dim> sb_dom = 
      (sb_ != no_subblock) ? map_.template impl_subblock_domain<dim>(sb_)
                           : empty_domain<dim>();
    Domain<dim> sb_dom_0 = map_.template impl_subblock_domain<dim>(0);

    pas::pbuf_create<T, order_type, impl_storage_format>(
      tag_,
      dom,
      map_,
      gdo_handle_,
      dist_handle_,
      pbuf_handle_,
      &pas_buffer_);

    if (pas_buffer_)
    {
      for (dimension_type d=0; d<dim; ++d)
      {
	dimension_type l_d = pas_buffer_->local_part->layout_order[d];
#if PAS_BLOCK_VERBOSE
	std::cout
	  << "[" << local_processor() << "] - "
	  << d << " (" << l_d << "):" 
	  << " ssl=["
	  << pas_buffer_->local_part->block_dim[l_d].global_start_index << ", "
	  << pas_buffer_->local_part->block_dim[l_d].stride << ", "
	  << pas_buffer_->local_part->block_dim[l_d].length << "]"
	  << "  map:" << sb_dom[d].size()
	  << "  total:" << dom[d].size()
	  << "  dim_subblocks:" << map.num_subblocks(d)
	  << "  addr: " << (ptrdiff_t)pas_buffer_->virt_addr_list[0]
	  << std::endl;
#endif
	// Check that PAS and VSIPL++ agree.
	assert(pas_buffer_->local_part->block_dim[l_d].length ==
	       static_cast<int>(sb_dom[d].size()));
      }

      // Check that PAS allocation is Dense
      if (dim == 1)
      {
	dimension_type d   = order_type::impl_dim0;
	dimension_type l_d = pas_buffer_->local_part->layout_order[d];
	assert(pas_buffer_->local_part->block_dim[l_d].stride == 1);
      }
      if (dim == 2)
      {
	dimension_type d0   = order_type::impl_dim0;
	dimension_type d1   = order_type::impl_dim1;
	dimension_type l_d0 = pas_buffer_->local_part->layout_order[d0];
	dimension_type l_d1 = pas_buffer_->local_part->layout_order[d1];

	assert(pas_buffer_->local_part->block_dim[l_d0].stride == 
	       static_cast<int>(sb_dom[d1].size()));
	assert(pas_buffer_->local_part->block_dim[l_d1].stride == 1);
      }
      if (dim >= 3)
      {
	dimension_type d0   = order_type::impl_dim0;
	dimension_type d1   = order_type::impl_dim1;
	dimension_type d2   = order_type::impl_dim2;
	dimension_type l_d0 = pas_buffer_->local_part->layout_order[d0];
	dimension_type l_d1 = pas_buffer_->local_part->layout_order[d1];
	dimension_type l_d2 = pas_buffer_->local_part->layout_order[d2];

	assert(pas_buffer_->local_part->block_dim[l_d0].stride == 
	       static_cast<int>(sb_dom[d1].size() * sb_dom[d2].size()));
	assert(pas_buffer_->local_part->block_dim[l_d1].stride == 
	       static_cast<int>(sb_dom[d2].size()));
	assert(pas_buffer_->local_part->block_dim[l_d2].stride == 1);
      }


      assert(pas_buffer_->num_virt_addrs == components);

      if (is_same<impl_storage_format, split_complex>::value &&
	  is_complex<T>::value)
      {
	typedef typename scalar_of<T>::type scalar_type;
	assert(pas_buffer_->elem_nbytes           == sizeof(T));
	assert(pas_buffer_->elem_component_nbytes == sizeof(scalar_type));

#if PAS_BLOCK_VERBOSE
	// Check that the offset of the imaginary data (PAS' second
	// component) is consistent.

	// First, make sure that a scalar_type evenly divides the
	// alignment.
	assert(VSIP_IMPL_PAS_ALIGNMENT % sizeof(scalar_type) == 0);

	// Second, compute the padding and expected offset.
	size_t t_alignment = (VSIP_IMPL_PAS_ALIGNMENT / sizeof(scalar_type));
	size_t offset      = sb_dom_0.size();
	size_t extra       = offset % t_alignment;

	// If not naturally aligned (extra != 0), pad by t_alignment - extra.
	if (extra) offset += (t_alignment - extra);

	std::cout << "offset " << tag_ << ":"
		  << " dom: " << sb_dom
		  << " size: " << sb_dom.size()
		  << " real: "
		  << (scalar_type*)pas_buffer_->virt_addr_list[1] -
	             (scalar_type*)pas_buffer_->virt_addr
		  << "  expected: " << offset 
		  << "  (extra: " << extra << ")"
		  << std::endl;

	assert(offset = ( (scalar_type*)pas_buffer_->virt_addr_list[1] -
			  (scalar_type*)pas_buffer_->virt_addr ));
#endif

	Offset<impl_storage_format, T>::check_imag_offset(
			sb_dom_0.size(),
			(scalar_type*)pas_buffer_->virt_addr_list[1] -
			(scalar_type*)pas_buffer_->virt_addr);
	       
	subblock_ = new local_block_type(sb_dom,
			impl::User_storage<T>(split_format, 
				(scalar_type*)pas_buffer_->virt_addr_list[0],
				(scalar_type*)pas_buffer_->virt_addr_list[1]));
      }
      else
      {
	subblock_ = new local_block_type(sb_dom,
			impl::User_storage<T>(array_format, 
				(T*)pas_buffer_->virt_addr_list[0]));
      }
      subblock_->admit(false);
    }
    else
    {
#if PAS_BLOCK_VERBOSE
      std::cout << "[" << local_processor() << "] no subblock" << std::endl;
#endif
      subblock_ = new local_block_type(sb_dom);
    }
  }

  // Constructors and destructor.
public:
  typedef typename impl::Complex_value_type<value_type, private_type>::type uT;

  Distributed_block(Domain<dim> const& dom,
		    map_type const& map = map_type())
  : map_           (map),
    proc_          (local_processor()),
    sb_            (map_.subblock(proc_)),
    subblock_      (NULL),
    admitted_      (true),
    tag_           (pas::global_tag++)
  {
    map_.impl_apply(dom);
    for (dimension_type d=0; d<dim; ++d)
      size_[d] = dom[d].length();

    this->init(dom);
  }

  Distributed_block(Domain<dim> const& dom,
		    value_type         value,
		    map_type const&        map = map_type())
    : map_           (map),
      proc_          (local_processor()),
      sb_            (map_.subblock(proc_)),
      subblock_      (NULL),
      admitted_      (true),
      tag_           (pas::global_tag++)
  {
    map_.impl_apply(dom);
    for (dimension_type d=0; d<dim; ++d)
      size_[d] = dom[d].length();

    this->init(dom);

    for (index_type i=0; i<subblock_->size(); ++i)
      subblock_->put(i, value);
  }

  Distributed_block(Domain<dim> const& dom, 
		    value_type* const  ptr,
		    map_type const&        map = map_type())
    : map_           (map),
      proc_          (local_processor()),
      sb_            (map_.subblock(proc_)),
      subblock_      (NULL),
      user_data_  (array_format, ptr),
      admitted_      (false),
      tag_           (pas::global_tag++)
  {
    map_.impl_apply(dom);
    for (dimension_type d=0; d<dim; ++d)
      size_[d] = dom[d].length();

    this->init(dom);
  }


  ~Distributed_block()
  {
    long rc;
    
    rc = pas_global_data_destroy(gdo_handle_);
    VSIP_IMPL_CHECK_RC(rc,"global_data_destroy");
    rc = pas_distribution_destroy(dist_handle_);
    VSIP_IMPL_CHECK_RC(rc,"distribution_destroy");
    rc = pas_pbuf_destroy(pbuf_handle_, 0);
    VSIP_IMPL_CHECK_RC(rc,"pbuf_destroy");
    if (pas_buffer_)
    {
      rc = pas_buffer_destroy(pas_buffer_);
      VSIP_IMPL_CHECK_RC(rc,"buffer_destroy");
    }

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
    if (    is_same<map_type, Global_map<1> >::value
        ||  vsip::num_processors() == 1
	|| (   is_same<map_type, Replicated_map<1> >::value
	    && map_.subblock() != no_subblock))
      return subblock_->get(idx);

    index_type     sb  = map_.impl_subblock_from_global_index(Index<1>(idx));
    processor_type pr  = *(map_.processor_begin(sb));
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
    if (    is_same<map_type, Global_map<1> >::value
        ||  vsip::num_processors() == 1
	|| (   is_same<map_type, Replicated_map<1> >::value
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
    if (    is_same<map_type, Global_map<1> >::value
        ||  vsip::num_processors() == 1
	|| (   is_same<map_type, Replicated_map<1> >::value
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
  ptr_type ptr() VSIP_NOTHROW { return subblock_->ptr();}

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
  length_type size(dimension_type block_dim, dimension_type d)
    const VSIP_NOTHROW
  {
    assert(block_dim == dim);
    assert(d < dim);
    return size_[d];
  }

  map_type const& map() const VSIP_NOTHROW 
    { return map_; }

  // length_type num_local_blocks() const { return num_subblocks_; }

  local_block_type& get_local_block() const
  {
    assert(subblock_ != NULL);
    return *subblock_;
  }


  proxy_local_block_type impl_proxy_block(index_type sb) const
  {
    return proxy_local_block_type(
		map_.template impl_subblock_extent<dim>(sb));
  }

  index_type subblock() const { return sb_; }

  void assert_local(index_type sb) const
    { assert(sb == sb_ && subblock_ != NULL); }

  // User storage functions.
public:
  void admit(bool update = true) VSIP_NOTHROW
  {
    if (update && this->user_storage() != no_user_format)
    {
      for (index_type i=0; i<subblock_->size(); ++i)
	subblock_->put(i, user_data_.get(i));
    }
    admitted_ = true;
  }

  void release(bool update = true) VSIP_NOTHROW
  {
    if (update && this->user_storage() != no_user_format)
    {
      for (index_type i=0; i<subblock_->size(); ++i)
	user_data_.put(i, subblock_->get(i));
    }
    admitted_ = false;
  }

  void release(bool update, value_type*& pointer) VSIP_NOTHROW
  {
    assert(this->user_storage() == no_user_format ||
	   this->user_storage() == array_format);
    this->release(update);
    this->user_data_.find(pointer);
  }

  void release(bool update, uT*& pointer) VSIP_NOTHROW
  {
    assert(this->user_storage() == no_user_format ||
	   this->user_storage() == interleaved_format);
    this->release(update);
    this->user_data_.find(pointer);
  }

  void release(bool update, uT*& real_pointer, uT*& imag_pointer) VSIP_NOTHROW
  {
    assert(this->user_storage() == no_user_format ||
	   this->user_storage() == split_format);
    this->release(update);
    this->user_data_.find(real_pointer, imag_pointer);
  }

  void find(value_type*& pointer) VSIP_NOTHROW
  {
    assert(this->user_storage() == no_user_format ||
	   this->user_storage() == array_format);
    this->user_data_.find(pointer);
  }

  void find(uT*& pointer) VSIP_NOTHROW
  {
    assert(this->user_storage() == no_user_format ||
	   this->user_storage() == interleaved_format);
    this->user_data_.find(pointer);
  }

  void find(uT*& real_pointer, uT*& imag_pointer) VSIP_NOTHROW
  {
    assert(this->user_storage() == no_user_format ||
	   this->user_storage() == split_format);
    this->user_data_.find(real_pointer, imag_pointer);
  }

  void rebind(value_type* pointer) VSIP_NOTHROW
  {
    assert(!this->admitted() && this->user_storage() == array_format);
    this->user_data_.rebind(pointer);
  }

  void rebind(uT* pointer) VSIP_NOTHROW
  {
    assert(!this->admitted() &&
	   (this->user_storage() == split_format ||
	    this->user_storage() == interleaved_format));
    this->user_data_.rebind(pointer);
  }

  void rebind(uT* real_pointer, uT* imag_pointer) VSIP_NOTHROW
  {
    assert(!this->admitted() &&
	   (this->user_storage() == split_format ||
	    this->user_storage() == interleaved_format));
    this->user_data_.rebind(real_pointer, imag_pointer);
  }

  enum user_storage_type user_storage() const VSIP_NOTHROW
    { return this->user_data_.format(); }

  bool admitted() const VSIP_NOTHROW
    { return admitted_; }

  PAS_distribution_handle impl_ll_dist() VSIP_NOTHROW
  { return dist_handle_; }

  PAS_pbuf_handle impl_ll_pbuf() VSIP_NOTHROW
  { return pbuf_handle_; }

  // Provide component offset (real-component, imaginary-component)
  // Used by Direct_pas_assign algorithm when using split-complex.
public:
  typedef Offset<impl_storage_format, T> offset_traits;
  typedef typename offset_traits::type offset_type;

  offset_type impl_component_offset() const VSIP_NOTHROW
  {
    Domain<dim> sb_dom_0 = map_.template impl_subblock_domain<dim>(0);
    return offset_traits::create(sb_dom_0.size());
  }

  // Member data.
private:
  map_type                 map_;
  processor_type	   proc_;		// This processor in comm.
  index_type   		   sb_;
  local_block_type*	   subblock_;
  length_type	           size_[dim];
  User_storage<T>          user_data_;
  bool                     admitted_;
  
  long                    tag_;
  PAS_global_data_handle  gdo_handle_;
  PAS_distribution_handle dist_handle_;
  PAS_pbuf_handle         pbuf_handle_;
  PAS_buffer*             pas_buffer_;
};

} // namespace vsip::impl::pas

template <typename B, typename M>
struct Is_pas_block<pas::Distributed_block<B, M> >
{
  static bool const value = true;
};

} // namespace vsip::impl
} // namespace vsip

#undef PAS_BLOCK_VERBOSE

#endif
