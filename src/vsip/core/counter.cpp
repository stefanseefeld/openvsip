/* Copyright (c) 2005 CodeSourcery, LLC.  All rights reserved.  */

/** @file    vsip/core/counter.cpp
    @author  Zack Weinberg
    @date    2005-01-22
    @brief   VSIPL++ Library: Checked counter classes (implementation).

    This file defines the class functions called when overflow and
    underflow are detected.  */

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/counter.hpp>

#include <sstream>
#include <stdexcept>


/***********************************************************************
  Definitions
************************************************************************/

using vsip::impl::Checked_counter;

#define DIAGNOSTIC(word, op, a, b)		\
  "Checked_counter " word ": " << a << " " op " " << b << std::endl

void
Checked_counter::overflow(value_type a, value_type b)
{
  std::ostringstream msgbuf;
  msgbuf << "Checked_counter overflow: " << a << " + " << b;
  VSIP_IMPL_THROW(std::overflow_error(msgbuf.str()));
}

void
Checked_counter::underflow(value_type a, value_type b)
{
  std::ostringstream msgbuf;
  msgbuf << "Checked_counter underflow: " << a << " - " << b;
  VSIP_IMPL_THROW(std::underflow_error(msgbuf.str()));
}
