//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/library.hpp>
#include <test.hpp>
#include "util.hpp"

template <typename T, storage_format_type F>
struct accessor
{
  static void ramp(host_storage<T, F> &s)
  {
    for (index_type i = 0; i != s.size(); ++i)
      s.put(i, T(i));
  }
  static void check(host_storage<T, F> const &s)
  {
    for (index_type i = 0; i != s.size(); ++i)
      test_assert(i == s.get(i));
  }
};

template <typename T, storage_format_type F>
void
test_storage(length_type size)
{
  typedef host_storage<T, F> storage_type;
  storage_type st(size);
  accessor<T, F>::ramp(st);
  accessor<T, F>::check(st);
}


int
main(int argc, char** argv)
{
  library lib(argc, argv);

  test_storage<int, array>(32);
}
