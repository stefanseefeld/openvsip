//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_scalar_map_hpp_
#define ovxx_parallel_scalar_map_hpp_

#include <ovxx/support.hpp>
#include <ovxx/vector_iterator.hpp>
#include <vsip/impl/vector.hpp>
#include <ovxx/parallel/map_traits.hpp>
#include <vsip/impl/map_fwd.hpp>
#include <vsip/parallel.hpp>

namespace ovxx
{
namespace parallel
{
template <dimension_type D>
class scalar_map
{
public:
  typedef ovxx::vector_iterator<Vector<processor_type> > processor_iterator;
  typedef Communicator::pvec_type impl_pvec_type;

  scalar_map() {}
  ~scalar_map() {}

  // Information on individual distributions.
  distribution_type distribution(dimension_type) const VSIP_NOTHROW
  { return whole;}
  length_type num_subblocks(dimension_type) const VSIP_NOTHROW
  { return 1;}
  length_type cyclic_contiguity(dimension_type) const VSIP_NOTHROW
  { return 0;}

  length_type num_subblocks() const VSIP_NOTHROW { return 1;}
  length_type num_processors() const VSIP_NOTHROW
  { return vsip::num_processors();}

  index_type subblock(processor_type /*pr*/) const VSIP_NOTHROW
  { return 0;}
  index_type subblock() const VSIP_NOTHROW
  { return 0;}

  processor_iterator processor_begin(index_type sb) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb == 0);
    return processor_iterator(vsip::processor_set(), 0);
  }

  processor_iterator processor_end(index_type sb) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb == 0);
    return processor_iterator(vsip::processor_set(), vsip::num_processors());
  }

  const_Vector<processor_type> processor_set() const
  { return vsip::processor_set();}

  length_type impl_num_patches(index_type sb) const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb == 0); return 1;}

  void impl_apply(Domain<D> const& /*dom*/) VSIP_NOTHROW
  {}

  template <dimension_type D1>
  Domain<D1> impl_subblock_domain(index_type sb) const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb == 0); return Domain<D>();}

  template <dimension_type D1>
  Length<D1> impl_subblock_extent(index_type sb) const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb == 0); return extent(Domain<D>());}

  template <dimension_type D1>
  Domain<D1> impl_global_domain(index_type sb, index_type patch)
    const VSIP_NOTHROW
    { OVXX_PRECONDITION(sb == 0 && patch == 0); return Domain<D>(); }

  template <dimension_type D1>
  Domain<D1> impl_local_domain (index_type sb, index_type patch)
    const VSIP_NOTHROW
    { OVXX_PRECONDITION(sb == 0 && patch == 0); return Domain<D>(); }

  index_type impl_global_from_local_index(dimension_type /*d*/, index_type sb,
				     index_type idx)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb == 0); return idx; }

  index_type impl_local_from_global_index(dimension_type /*d*/, index_type idx)
    const VSIP_NOTHROW
  { return idx;}

  template <dimension_type D1>
  index_type impl_subblock_from_global_index(Index<D1> const& /*idx*/)
    const VSIP_NOTHROW
  { return 0;}

  template <dimension_type D1>
  Domain<D> impl_local_from_global_domain(index_type /*sb*/,
					  Domain<D1> const& dom)
    const VSIP_NOTHROW
  {
    OVXX_CT_ASSERT(D == D1);
    return dom;
  }

  par_ll_pset_type impl_ll_pset() const VSIP_NOTHROW
  { return default_communicator().impl_ll_pset();}
  Communicator &impl_comm() const
  { return default_communicator();}
  impl_pvec_type const &impl_pvec() const
  { return default_communicator().pvec();}

  length_type impl_working_size() const
  { return this->num_processors();}

  processor_type impl_proc_from_rank(index_type idx) const
  { return this->impl_pvec()[idx];}

  index_type impl_rank_from_proc(processor_type pr) const
  {
    for (index_type i=0; i<this->num_processors(); ++i)
      if (this->impl_pvec()[i] == pr) return i;
    return no_rank;
  }
};

template <dimension_type D, typename M>
struct map_equal<D, M, scalar_map<D> >
{
  static bool value(M const&, scalar_map<D> const&) { return true;}
};

template <dimension_type D>
struct is_local_map<scalar_map<D> >
{ static bool const value = true;};

template <dimension_type D>
struct is_global_map<scalar_map<D> >
{ static bool const value = true;};

} // namespace ovxx::parallel
} // namespace ovxx

#endif
