//
// Copyright (c) 2015 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <boost/python.hpp>
#include <ovxx/mpi/communicator.hpp>

namespace bpl = boost::python;
using namespace ovxx::mpi;

namespace
{
}

BOOST_PYTHON_MODULE(mpi)
{
  bpl::class_<Group> group("group");
  group.def("rank", &Group::rank);
  group.def("size", &Group::size);

  bpl::class_<Communicator> com("communicator");
  com.def("rank", &Communicator::rank);
  com.def("size", &Communicator::size);
  com.def("barrier", &Communicator::barrier);
  com.def("group", &Communicator::group);

}
