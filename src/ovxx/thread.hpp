//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_thread_hpp_
#define ovxx_thread_hpp_

#include <ovxx/config.hpp>
#include <thread>
#include <mutex>

#if OVXX_ENABLE_THREADING
# if __GNUC__
#  define thread_local __thread
# else
#  error "No support for threading with this compiler."
# endif
#else
# define thread_local
#endif

#endif
