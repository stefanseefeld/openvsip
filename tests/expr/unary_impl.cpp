//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include "common.hpp"

using namespace vsip;

TEST_UNARY(is_nan,    is_nan,    impl::fn::is_nan,    anyval)
// These C99 functions are unavailable on Windows.
#if !defined(_MSC_VER)
TEST_UNARY(is_finite, is_finite, impl::fn::is_finite, anyval)
TEST_UNARY(is_normal, is_normal, impl::fn::is_normal, anyval)
#endif


int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  vector_cases2_rt<Test_is_nan,    float,  bool>();
#if !defined(_MSC_VER)
  vector_cases2_rt<Test_is_finite, float,  bool>();
  vector_cases2_rt<Test_is_normal, float,  bool>();
#endif
}
