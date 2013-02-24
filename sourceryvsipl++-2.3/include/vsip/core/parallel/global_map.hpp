/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/parallel/global_map.hpp
    @author  Jules Bergmann
    @date    2005-06-08
    @brief   VSIPL++ Library: Global_map class.

*/

#ifndef VSIP_CORE_PARALLEL_GLOBAL_MAP_HPP
#define VSIP_CORE_PARALLEL_GLOBAL_MAP_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/vector_iterator.hpp>
#include <vsip/core/parallel/services.hpp>
#include <vsip/core/parallel/map_traits.hpp>
#include <vsip/core/parallel/support.hpp>
#include <vsip/core/parallel/scalar_block_map.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/map_fwd.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

template <dimension_type Dim>
class Global_map
{
  // Compile-time typedefs.
public:
  typedef impl::Vector_iterator<Vector<processor_type> > processor_iterator;
  typedef impl::Communicator::pvec_type impl_pvec_type;

  // Constructor.
public:
  Global_map()
    : dom_()
  {}

  ~Global_map()
  {}

  // Accessors.
public:
  // Information on individual distributions.
  distribution_type distribution     (dimension_type) const VSIP_NOTHROW
    { return whole; }
  length_type       num_subblocks    (dimension_type) const VSIP_NOTHROW
    { return 1; }
  length_type       cyclic_contiguity(dimension_type) const VSIP_NOTHROW
    { return 0; }

  length_type num_subblocks()  const VSIP_NOTHROW { return 1; }
  length_type num_processors() const VSIP_NOTHROW
    { return vsip::num_processors(); }

  index_type subblock(processor_type /*pr*/) const VSIP_NOTHROW
    { return 0; }
  index_type subblock() const VSIP_NOTHROW
    { return 0; }

  processor_iterator processor_begin(index_type sb) const VSIP_NOTHROW
  {
    assert(sb == 0);
    return processor_iterator(vsip::processor_set(), 0);
  }

  processor_iterator processor_end  (index_type sb) const VSIP_NOTHROW
  {
    assert(sb == 0);
    return processor_iterator(vsip::processor_set(), vsip::num_processors());
  }

  const_Vector<processor_type> processor_set() const
    { return vsip::processor_set(); }

  // Applied map functions.
public:
  length_type impl_num_patches(index_type sb) const VSIP_NOTHROW
    { assert(sb == 0); return 1; }

  void impl_apply(Domain<Dim> const& dom) VSIP_NOTHROW
    { dom_ = dom; }

  template <dimension_type Dim2>
  Domain<Dim2> impl_subblock_domain(index_type sb) const VSIP_NOTHROW
    { assert(sb == 0); return dom_; }

  template <dimension_type Dim2>
  impl::Length<Dim2> impl_subblock_extent(index_type sb) const VSIP_NOTHROW
    { assert(sb == 0); return impl::extent(dom_); }

  template <dimension_type Dim2>
  Domain<Dim2> impl_global_domain(index_type sb, index_type patch)
    const VSIP_NOTHROW
    { assert(sb == 0 && patch == 0); return dom_; }

  template <dimension_type Dim2>
  Domain<Dim2> impl_local_domain (index_type sb, index_type patch)
    const VSIP_NOTHROW
    { assert(sb == 0 && patch == 0); return dom_; }

  index_type impl_global_from_local_index(dimension_type /*d*/, index_type sb,
				     index_type idx)
    const VSIP_NOTHROW
  { assert(sb == 0); return idx; }

  index_type impl_local_from_global_index(dimension_type /*d*/, index_type idx)
    const VSIP_NOTHROW
  { return idx; }

  template <dimension_type Dim2>
  index_type impl_subblock_from_global_index(Index<Dim2> const& /*idx*/)
    const VSIP_NOTHROW
  { return 0; }

  template <dimension_type Dim2>
  Domain<Dim> impl_local_from_global_domain(index_type /*sb*/,
					    Domain<Dim2> const& dom)
    const VSIP_NOTHROW
  {
    VSIP_IMPL_STATIC_ASSERT(Dim == Dim2);
    return dom;
  }

  // Extensions.
public:
  impl::par_ll_pset_type impl_ll_pset() const VSIP_NOTHROW
    { return impl::default_communicator().impl_ll_pset(); }
  impl::Communicator&    impl_comm() const
    { return impl::default_communicator(); }
  impl_pvec_type const&  impl_pvec() const
    { return impl::default_communicator().pvec(); }

  length_type            impl_working_size() const
    { return this->num_processors(); }

  processor_type         impl_proc_from_rank(index_type idx) const
    { return this->impl_pvec()[idx]; }

  index_type impl_rank_from_proc(processor_type pr) const
  {
    for (index_type i=0; i<this->num_processors(); ++i)
      if (this->impl_pvec()[i] == pr) return i;
    return no_rank;
  }

  // Member data.
private:
  Domain<Dim> dom_;
};



template <dimension_type Dim>
class Local_or_global_map : public Global_map<Dim>
{
public:
  static bool const impl_local_only  = false;
  static bool const impl_global_only = false;

  // Constructor.
public:
  Local_or_global_map() {}
};



namespace impl
{

/// Specialize global traits for Local_or_global_map.

template <dimension_type Dim>
struct Is_global_map<Global_map<Dim> >
{ static bool const value = true; };

template <dimension_type Dim>
struct Is_local_map<Local_or_global_map<Dim> >
{ static bool const value = true; };

template <dimension_type Dim>
struct Is_global_map<Local_or_global_map<Dim> >
{ static bool const value = true; };



template <dimension_type Dim>
struct Map_equal<Dim, Global_map<Dim>, Global_map<Dim> >
{
  static bool value(Global_map<Dim> const&,
		    Global_map<Dim> const&)
    { return true; }
};

} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_CORE_PARALLEL_GLOBAL_MAP_HPP
