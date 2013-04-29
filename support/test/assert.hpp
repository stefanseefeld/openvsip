//
// Copyright (c) 2005, 2006 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under theLicense.
// license contained in the accompanying LICENSE.GPL file.

#ifndef test_assert_hpp_
#define test_assert_hpp_

#include <iostream>
#include <cstdlib>

namespace test
{
void inline fail(char const *assertion,
		 char const *file, unsigned int line,
		 char const *function)
{
  std::cerr << "FAIL: assertion '" << assertion << "'\n";
  if (function)
    std::cerr << "  in " << function << '\n';
  std::cerr << "  at " << file << ':' << line;
  std::cerr << std::endl;
  abort();
}

} // namespace test

#if defined(__GNUC__)
# define OVXX_TEST_FUNCTION __PRETTY_FUNCTION__
#elif defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
# define OVXX_TEST_FUNCTION __func__
#else
# define OVXX_TEST_FUNCTION ((__const char *) 0)
#endif

#define test_assert(expr)      					\
(static_cast<void>((expr) ? 0 :					\
		   (test::fail(#expr, __FILE__, __LINE__,	\
			       OVXX_TEST_FUNCTION), 0)))

#endif
