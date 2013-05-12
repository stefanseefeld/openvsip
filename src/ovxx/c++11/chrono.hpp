//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

// C++11 compatibility

#ifndef ovxx_cxx11_chrono_hpp_
#define ovxx_cxx11_chrono_hpp_

// Provide just enough C++11 API so we can
// implement a C++11-compatible timer.

#include <ovxx/inttypes.hpp>
#if OVXX_TIMER_SYSTEM
# include <chrono>
#elif defined(OVXX_TIMER_POSIX)
# include <ctime>
#endif

namespace ovxx
{
namespace cxx11
{
#ifndef OVXX_TIMER_SYSTEM
namespace chrono
{

template<typename T>
struct duration
{
  typedef T rep;

  duration() {}
  duration(T v) : value_(v) {}

  rep count() const { return value_;}

  duration &operator+=(duration const &d)
  { value_ += d.value_; return *this;}
  duration &operator-=(duration const &d)
  { value_ -= d.value_; return *this;}

  static duration zero() { return duration();}

private:
  rep value_;
};

typedef duration<int64_type> nanoseconds;

template<typename C, typename D>
struct time_point
{
  typedef C clock;
  typedef D duration;

  time_point() : value_(duration::zero()) {}
  explicit time_point(duration const &d) : value_(d) {}

  duration time_since_epoch() const { return value_;}
  operator double() const { return value_.count();}

  time_point &operator+=(duration const &d)
  { value_ += d; return *this;}
  time_point &operator-=(duration const &d)
  { value_ -= d; return *this;}

private:
  duration value_;
};

template<typename C, typename D>
time_point<C,D>
operator- (time_point<C,D> const &a, time_point<C,D> const &b)
{
  time_point<C,D> result(a);
  result -= b.time_since_epoch();
  return result;
}

class high_resolution_clock
{
  static float tics_per_nanosecond;
public:
  typedef chrono::nanoseconds duration;
  typedef duration::rep rep;
  typedef chrono::time_point<high_resolution_clock, duration> time_point;

  static void init();
  static time_point now()
  {
    rep time;
#if defined(OVXX_TIMER_POSIX)
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    time = ts.tv_sec * 1000000000 + ts.tv_nsec;
#elif defined(OVXX_TIMER_IA32_TSC)
    long long tsc;
    __asm__ __volatile__("rdtsc": "=A"(tsc));
    time = static_cast<float>(tsc) / tics_per_nanosecond;
#elif defined(OVXX_TIMER_X64_TSC)
    unsigned a, d;
    __asm__ __volatile__("rdtsc": "=a"(a), "=d"(d));
    time = static_cast<float>((rep)d << 32 | a)/ tics_per_nanosecond;
#elif OVXX_TIMER_POWER
    unsigned int tbl, tbu0, tbu1;
    do
    {
      __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
      __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
      __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    }
    while (tbu0 != tbu1);
    time = static_cast<float>((rep)tbu0 << 32 | tbl)/ tics_per_nanoseconds;
#endif
    return time_point(time);
  }
};

} // namespace ovxx::cxx11::chrono
#else
namespace chrono = std::chrono;
#endif

} // namespace ovxx::cxx11

typedef cxx11::chrono::high_resolution_clock clock;

} // namespace ovxx

#endif
