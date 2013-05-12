//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_local_map_hpp_
#define vsip_impl_local_map_hpp_

#include <vsip/support.hpp>
#include <vsip/impl/map_fwd.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/value_iterator.hpp>
#include <ovxx/parallel/map_traits.hpp>
#include <ovxx/allocator.hpp>

namespace vsip
{

class Local_map
{
public:
  typedef ovxx::value_iterator<processor_type, unsigned> processor_iterator;

  Local_map() : allocator_(ovxx::allocator::get_default()) {}

  template <dimension_type D>
  Local_map(ovxx::parallel::local_or_global_map<D> const&)
    : allocator_(ovxx::allocator::get_default())
  {}

  template <dimension_type D>
  Local_map(ovxx::parallel::scalar_map<D> const&)
    : allocator_(ovxx::allocator::get_default())
  {}

  distribution_type distribution(dimension_type) const VSIP_NOTHROW
  { return whole;}
  length_type num_subblocks(dimension_type) const VSIP_NOTHROW
  { return 1;}
  length_type cyclic_contiguity(dimension_type) const VSIP_NOTHROW
  { return 0;}

  length_type num_subblocks() const VSIP_NOTHROW { return 1;}
  index_type subblock(processor_type pr) const VSIP_NOTHROW
  { return (pr == local_processor()) ? 0 : no_subblock;}
  index_type subblock() const VSIP_NOTHROW { return 0;}

  processor_iterator processor_begin(index_type) const
  { return processor_iterator(local_processor(), 1);}
  processor_iterator processor_end(index_type) const
  { return processor_iterator(local_processor() + 1, 1);}

  length_type impl_num_patches(index_type sb OVXX_UNUSED) const VSIP_NOTHROW
  { assert(sb == 0); return 1;}

  template <dimension_type D>
  void impl_apply(Domain<D> const &) VSIP_NOTHROW {}

  template <dimension_type D>
  Domain<D> subblock_domain(index_type) const VSIP_NOTHROW
  { assert(0);}

  template <dimension_type D>
  Domain<D> global_domain(index_type /*sb*/, index_type /*patch*/)
    const VSIP_NOTHROW
  { assert(0);}

  template <dimension_type D>
  Domain<D> local_domain(index_type /*sb*/, index_type /*patch*/)
    const VSIP_NOTHROW
  { assert(0);}

  index_type impl_global_from_local_index(dimension_type /*d*/, index_type sb OVXX_UNUSED,
					  index_type idx)
    const VSIP_NOTHROW
  { assert(sb == 0); return idx;}

  ovxx::allocator *impl_allocator() const { return allocator_;}
  void impl_set_allocator(ovxx::allocator *a) { allocator_ = a;}

private:
  ovxx::allocator *allocator_;
};

} // namespace vsip

namespace ovxx
{
namespace parallel
{
template <>
struct is_local_map<Local_map> { static bool const value = true;};

template <dimension_type D>
struct map_equal<D, Local_map, Local_map>
{
  static bool value(Local_map const&, Local_map const&)
  { return true;}
};

} // namespace ovxx::parallel
} // namespace ovxx

#endif
