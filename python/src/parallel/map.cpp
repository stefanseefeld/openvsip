//
// Copyright (c) 2015 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/python/block.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/parallel.hpp>
#include "../unique_ptr.hpp"

namespace bpl = boost::python;
using namespace vsip;

namespace
{
Vector<processor_type> convert(bpl::list procs)
{
  Vector<processor_type> result(bpl::len(procs));
  for (index_type i = 0; i != result.size(); ++i)
    result.put(i, bpl::extract<processor_type>(procs[i]));
  return result;
}

typedef std::unique_ptr<Map<> > map_ptr;

map_ptr make_map_0(bpl::list procs) { return map_ptr(new Map<>(convert(procs)));}
map_ptr make_map_1(bpl::list procs, length_type d0)
{ return map_ptr(new Map<>(convert(procs), d0));}
map_ptr make_map_2(bpl::list procs, length_type d0, length_type d1)
{ return map_ptr(new Map<>(convert(procs), d0, d1));}
map_ptr make_map_3(bpl::list procs, length_type d0, length_type d1, length_type d2)
{ return map_ptr(new Map<>(convert(procs), d0, d1, d2));}

void apply(Map<> &map, bpl::tuple shape)
{
  typedef bpl::extract<Domain<1> > e;
  switch (bpl::len(shape))
  {
    case 1:
      return map.impl_apply(Domain<1>(e(shape[0])));
    case 2:
      return map.impl_apply(Domain<2>(e(shape[0]), e(shape[1])));
    case 3:
      return map.impl_apply(Domain<3>(e(shape[0]), e(shape[1]), e(shape[2])));
    default:
      throw std::invalid_argument("Invalid index");
  }
}
  
index_type _subblock_from_global_index(Map<> const &map, bpl::tuple index)
{
  typedef bpl::extract<index_type> e;
  switch (bpl::len(index))
  {
    case 1:
      return map.impl_subblock_from_global_index(Index<1>(e(index[0])));
    case 2:
      return map.impl_subblock_from_global_index(Index<2>(e(index[0]),
							  e(index[1])));
    case 3:
      return map.impl_subblock_from_global_index(Index<3>(e(index[0]),
							  e(index[1]),
							  e(index[2])));
    default:
      throw std::invalid_argument("Invalid index");
  }
}

bpl::object _subblock_domain(Map<> const &map, index_type sb, dimension_type d)
{
  switch (d)
  {
    case 1:
      return bpl::object(map.impl_subblock_domain<1>(sb));
    case 2:
      return bpl::object(map.impl_subblock_domain<2>(sb));
    case 3:
      return bpl::object(map.impl_subblock_domain<3>(sb));
    default:
      throw std::invalid_argument("Invalid dimension");
  }
}
  
// quick hack:
// return the (first) processor containing (owning) the given index
int processor_from_global_index(Map<> const &map, bpl::tuple index)
{
  index_type sb = _subblock_from_global_index(map, index);
  return *map.processor_begin(sb);
}
  
void barrier(Map<> const &map)
{
  map.impl_comm().barrier();
}

}

BOOST_PYTHON_MODULE(map)
{
  bpl::class_<Map<> > map("map");
  map.def(bpl::init<length_type>());
  map.def(bpl::init<length_type, length_type>());
  map.def(bpl::init<length_type, length_type, length_type>());
  map.def("__init__", bpl::make_constructor(make_map_0));
  map.def("__init__", bpl::make_constructor(make_map_1));
  map.def("__init__", bpl::make_constructor(make_map_2));
  map.def("__init__", bpl::make_constructor(make_map_3));
  map.def("apply", apply);
  typedef length_type (Map<>::*num_subblocks_t)() const;
  map.def("num_subblocks", (num_subblocks_t)&Map<>::num_subblocks);
  map.def("num_processors", &Map<>::num_processors);
  typedef length_type (Map<>::*subblock_t)() const;
  map.def("subblock", (subblock_t)&Map<>::subblock);
  map.def("subblock_from_global_index", _subblock_from_global_index);
  map.def("local_from_global_index", &Map<>::impl_local_from_global_index);
  map.def("subblock_domain", _subblock_domain);
  map.def("owner", processor_from_global_index);
  map.def("communicator", &Map<>::impl_comm, bpl::return_internal_reference<>());
}
