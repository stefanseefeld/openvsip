//
// Copyright (c) 2015 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef parallel_block_api_hpp_
#define parallel_block_api_hpp_

#include "../block_api.hpp"


namespace pyvsip
{
using namespace ovxx;
using namespace ovxx::python;

template <dimension_type D, typename T, typename M>
void define_distributed_block(char const *type_name)
{
  typedef Block<D, T, M> block_type;

  bpl::class_<block_type, boost::shared_ptr<block_type>, boost::noncopyable> 
    block(type_name, bpl::no_init);
  block.setattr("dtype", get_dtype<T>());
  block.def("assign", assign<D, T, M>);
  block.def("assign", assign_scalar<D, T, M>);

  block.def("copy", copy<D, T, M>);

  block.def("size", total_size<D, T, M>);
  block.def("size", size<D, T, M>);
  block.add_property("shape", shape<D, T, M>);
  // FIXME: Why can't this become a property ?
  block.def("map", &block_type::map, bpl::return_internal_reference<>());
  block.def("get", &traits<D, T, M>::get);
  block.def("put", &traits<D, T, M>::put);

  // block.def("__eq__", eq<D, T, M>);
  // block.def("__neg__", neg<D, T, M>);

  //  define_compound_assignment(block, T());
  // define_complex_subblocks<D>(block, T());

  /// Construction from shape
  bpl::def("block", traits<D, T, M>::construct);
  bpl::def("block", traits<D, T, M>::construct_init);
  // /// Conversion from array.
  // bpl::def("block", construct<T, M>);
  // /// Construct subblock
  // bpl::def("subblock", subblock1<T, M>);
  if (D == 2)
  {
    //    bpl::def("subblock", subblock2<T, M>);
    // block.def("row", get_row<T, M>);
    // block.def("col", get_col<T, M>);
    // block.def("diag", diag<T, M>);
  }

  //  typedef Block<D, T> &(*get_local_block_type)(block_type const &);
  block.def("local", &block_type::get_local_block, bpl::return_internal_reference<>());

}

}

#endif
