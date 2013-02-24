/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/timer.hpp
    @author  Stefan Seefeld
    @date    2005-04-22
    @brief   VSIPL++ Library: Timer class to measure performance.

*/

#ifndef VSIP_CORE_TIMER_HPP
#define VSIP_CORE_TIMER_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <ctime>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{

class Timer
{
public:
  Timer() : start_(std::clock()) {}
  double elapsed() const 
  { return  double(std::clock() - start_) / CLOCKS_PER_SEC;}
private:
  std::clock_t start_;
};

} // namespace vsip::impl
} // namespace vsip

#endif
