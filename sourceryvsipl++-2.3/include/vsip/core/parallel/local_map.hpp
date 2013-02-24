/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/parallel/local_map.hpp
    @author  Jules Bergmann
    @date    2005-06-08
    @brief   VSIPL++ Library: Local_map class.

*/

#ifndef VSIP_CORE_PARALLEL_LOCAL_MAP_HPP
#define VSIP_CORE_PARALLEL_LOCAL_MAP_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/block_traits.hpp>
#include <vsip/core/value_iterator.hpp>
#include <vsip/core/parallel/services.hpp>
#include <vsip/core/parallel/map_traits.hpp>
#include <vsip/core/memory_pool.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

template <dimension_type Dim>
class Local_or_global_map;

namespace impl
{

template <dimension_type Dim>
class Scalar_block_map;

} // namespace vsip::impl;



class Local_map
{
  // Compile-time typedefs.
public:
  typedef impl::Value_iterator<processor_type, unsigned> processor_iterator;
  typedef impl::Communicator::pvec_type impl_pvec_type;

  static bool const impl_local_only  = true;
  static bool const impl_global_only = false;

  // Constructor.
public:
  Local_map() : pool_(vsip::impl::default_pool) {}

  template <dimension_type Dim>
  Local_map(Local_or_global_map<Dim> const&)
    : pool_(vsip::impl::default_pool)
  {}

  template <dimension_type Dim>
  Local_map(impl::Scalar_block_map<Dim> const&)
    : pool_(vsip::impl::default_pool)
  {}

  // Accessors.
public:
  distribution_type distribution     (dimension_type) const VSIP_NOTHROW
    { return whole; }
  length_type       num_subblocks    (dimension_type) const VSIP_NOTHROW
    { return 1; }
  length_type       cyclic_contiguity(dimension_type) const VSIP_NOTHROW
    { return 0; }

  length_type num_subblocks() const VSIP_NOTHROW { return 1; }

  index_type subblock(processor_type pr) const VSIP_NOTHROW
    { return (pr == local_processor()) ? 0 : no_subblock; }
  index_type subblock() const VSIP_NOTHROW
    { return 0; }

  processor_iterator processor_begin(index_type /*sb*/) const VSIP_NOTHROW
    { VSIP_IMPL_THROW(impl::unimplemented("Local_map::processor_begin()")); }
  processor_iterator processor_end  (index_type /*sb*/) const VSIP_NOTHROW
    { VSIP_IMPL_THROW(impl::unimplemented("Local_map::processor_end()")); }

  // Applied map functions.
public:
  length_type impl_num_patches(index_type sb) const VSIP_NOTHROW
    { assert(sb == 0); return 1; }

  template <dimension_type Dim>
  void impl_apply(Domain<Dim> const& /*dom*/) VSIP_NOTHROW
    {}

  template <dimension_type Dim2>
  Domain<Dim2> subblock_domain(index_type sb) const VSIP_NOTHROW
    { assert(0); }

  template <dimension_type Dim2>
  Domain<Dim2> global_domain(index_type /*sb*/, index_type /*patch*/)
    const VSIP_NOTHROW
    { assert(0); }

  template <dimension_type Dim2>
  Domain<Dim2> local_domain (index_type /*sb*/, index_type /*patch*/)
    const VSIP_NOTHROW
    { assert(0); }

  index_type impl_global_from_local_index(dimension_type /*d*/, index_type sb,
				     index_type idx)
    const VSIP_NOTHROW
  { assert(sb == 0); return idx; }

  // Extensions.
public:
  impl::Communicator& impl_comm() const { return impl::default_communicator();}
  impl_pvec_type const&  impl_pvec() const
    { return impl::default_communicator().pvec(); }

  length_type        impl_working_size() const
    { return 1; }

  processor_type impl_proc_from_rank(index_type idx) const
    { assert(idx == 0); return local_processor(); }

  impl::Memory_pool* impl_pool() const { return pool_; }
  void impl_set_pool(impl::Memory_pool* pool) { pool_ = pool; }

  // Member data.
private:
  impl::Memory_pool* pool_;
};

namespace impl
{

/// Specialize local/global traits for Local_map.

template <>
struct Is_local_map<Local_map>
{ static bool const value = true; };



template <dimension_type Dim>
struct Map_equal<Dim, Local_map, Local_map>
{
  static bool value(Local_map const&, Local_map const&)
    { return true; }
};

} // namespace impl

} // namespace vsip

#endif // VSIP_CORE_PARALLEL_LOCAL_MAP_HPP
