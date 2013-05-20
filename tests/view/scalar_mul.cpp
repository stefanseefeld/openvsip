//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include "scalar.hpp"

int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0
  test_lite<op_mul>();
#else
  test_op<op_mul>();
#endif
}
