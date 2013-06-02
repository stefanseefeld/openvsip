//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_signal_window_hpp_
#define vsip_impl_signal_window_hpp_

#include <ovxx/signal/window.hpp>

namespace vsip
{

// Generates Blackman window
const_Vector<float>
blackman(length_type len) VSIP_THROW((std::bad_alloc))
{
  OVXX_PRECONDITION(len > 1);
  return ovxx::signal::blackman<float>(len);
}

// Generates Chebyshev window
const_Vector<float>
cheby(length_type len, float ripple) VSIP_THROW((std::bad_alloc))
{
  OVXX_PRECONDITION(len > 1);
  return ovxx::signal::cheby<float>(len, ripple);
}

// Generates Hanning window
const_Vector<float>
hanning(length_type len) VSIP_THROW((std::bad_alloc))
{
  OVXX_PRECONDITION(len > 1);
  return ovxx::signal::hanning<float>(len);
}

// Generates Kaiser window
const_Vector<float>
kaiser(length_type len, float beta) VSIP_THROW((std::bad_alloc))
{
  OVXX_PRECONDITION(len > 1);
  return ovxx::signal::kaiser<float>(len, beta);
}

} // namespace vsip

#endif
