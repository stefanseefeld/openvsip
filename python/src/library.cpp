//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/python/block.hpp>
#include <boost/python.hpp>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <stdexcept>
#include <cstdarg>

namespace bpl = boost::python;

BOOST_PYTHON_MODULE(library)
{
  bpl::class_<vsip::vsipl, boost::noncopyable> vsipl("library");
}
