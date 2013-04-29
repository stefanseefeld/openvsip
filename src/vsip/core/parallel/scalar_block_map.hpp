//
// Copyright (c) 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_PARALLEL_SCALAR_BLOCK_MAP_HPP
#define VSIP_CORE_PARALLEL_SCALAR_BLOCK_MAP_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/vector_iterator.hpp>
#include <vsip/core/parallel/map_traits.hpp>
#include <vsip/core/map_fwd.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

/// Scalar_block_map class for Scalar_block.

/// Similar to Local_or_global_map, but with fixed size.

template <dimension_type Dim>
class Scalar_block_map
{
  // Compile-time typedefs.
public:
  typedef impl::Vector_iterator<Vector<processor_type> > processor_iterator;
  typedef impl::Communicator::pvec_type impl_pvec_type;

  // Constructor.
public:
  Scalar_block_map() {}

  ~Scalar_block_map() {}

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

  void impl_apply(Domain<Dim> const& /*dom*/) VSIP_NOTHROW
    {}

  template <dimension_type Dim2>
  Domain<Dim2> impl_subblock_domain(index_type sb) const VSIP_NOTHROW
    { assert(sb == 0); return Domain<Dim>(); }

  template <dimension_type Dim2>
  impl::Length<Dim2> impl_subblock_extent(index_type sb) const VSIP_NOTHROW
    { assert(sb == 0); return impl::extent(Domain<Dim>()); }

  template <dimension_type Dim2>
  Domain<Dim2> impl_global_domain(index_type sb, index_type patch)
    const VSIP_NOTHROW
    { assert(sb == 0 && patch == 0); return Domain<Dim>(); }

  template <dimension_type Dim2>
  Domain<Dim2> impl_local_domain (index_type sb, index_type patch)
    const VSIP_NOTHROW
    { assert(sb == 0 && patch == 0); return Domain<Dim>(); }

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

  // No member data.
};



/***********************************************************************
  Map traits
***********************************************************************/

template <dimension_type Dim>
struct Is_local_map<Scalar_block_map<Dim> >
{ static bool const value = true; };

template <dimension_type Dim>
struct is_global_map<Scalar_block_map<Dim> >
{ static bool const value = true; };



} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_PARALLEL_SCALAR_BLOCK_MAP_HPP
