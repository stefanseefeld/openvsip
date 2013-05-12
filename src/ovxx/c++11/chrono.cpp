//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/c++11/chrono.hpp>
#include <fstream>

namespace
{
float read_clock_rate()
{
  char line[1024];
  float rate = 1000;
  std::ifstream file("/proc/cpuinfo");

  while(!file.eof())
  {
    file.getline(line, sizeof(line));
    if (std::sscanf(line, "cpu MHz : %f", &rate))
      break;
  }
  return rate;
}

float get_cpu_speed()
{
#if OVXX_CPU_SPEED
   return OVXX_CPU_SPEED;
#else
   return read_clock_rate();
#endif
}

}

#ifndef OVXX_TIMER_SYSTEM
namespace ovxx
{
namespace cxx11
{
namespace chrono
{

float high_resolution_clock::tics_per_nanosecond;

void high_resolution_clock::init()
{
  float mhz = get_cpu_speed();
  tics_per_nanosecond = mhz/1000;
}

} // namespace ovxx::cxx11::chrono
} // namespace ovxx::cxx11
} // namespace ovxx
#endif // OVXX_TIMER_SYSTEM
