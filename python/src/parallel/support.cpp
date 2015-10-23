//
// Copyright (c) 2015 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <boost/python.hpp>
#include <ovxx/config.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/parallel.hpp>

namespace bpl = boost::python;

namespace
{
bpl::list get_processor_set()
{
  bpl::list result;
  vsip::Vector<vsip::processor_type> procs = vsip::processor_set();
  for (vsip::index_type i = 0; i != procs.size(); ++i)
    result.append(procs.get(i));
  return result;
}

}

BOOST_PYTHON_MODULE(support)
{
  bpl::def("num_processors", vsip::num_processors);
  bpl::def("local_processor", vsip::local_processor);
  bpl::def("processor_set", get_processor_set);
}
