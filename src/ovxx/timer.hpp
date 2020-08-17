//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_timer_hpp_
#define ovxx_timer_hpp_

#include <ovxx/chrono.hpp>

namespace ovxx
{

class timer
{
public:
  timer() : start_(clock::now()) {}
  clock::time_point restart() 
  { start_ = clock::now(); return start_;}
  double elapsed() 
  {
#ifdef OVXX_TIMER_SYSTEM
    using namespace cxx11::chrono;
    return duration_cast<nanoseconds>(clock::now() - start_).count()/1000000000;
#else
    return static_cast<double>(clock::now() - start_)/1000000000;
#endif
  }
private:
  clock::time_point start_;
};

} // namespace ovxx

#endif
