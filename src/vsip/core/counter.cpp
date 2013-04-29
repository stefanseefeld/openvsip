//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

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
