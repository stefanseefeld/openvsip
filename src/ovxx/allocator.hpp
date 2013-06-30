//
// Copyright (c) 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_allocator_hpp_
#define ovxx_allocator_hpp_

#include <ovxx/support.hpp>
#include <ovxx/c++11.hpp>
#include <limits>
#include <cstdlib>
#include <stdexcept>

namespace ovxx
{

class allocator
{
public:
  allocator() {}
  virtual ~allocator() {}

  template <typename T>
  T *allocate(length_type size)
  {
    return static_cast<T*>(allocate(size * sizeof(T)));
  }

  template <typename T>
  void deallocate(T *ptr, length_type size)
  {
    deallocate((void*)ptr, size * sizeof(T));
  }

  static void initialize(int &argc, char **&argv);
  static void finalize();
  static allocator *get_default()
  {
    OVXX_INVARIANT(default_ && "OpenVSIP not properly initialized !");
    return default_;
  }
  static void set_default(allocator *a) { default_ = a;}

private:
  virtual void *allocate(size_t size) = 0;
  virtual void deallocate(void *ptr, size_t size) = 0;

#if OVXX_ENABLE_THREADING
  static thread_local allocator *default_;
#else
  static allocator *default_;
#endif
};

} // namespace ovxx

#endif
