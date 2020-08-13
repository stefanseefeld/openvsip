//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_signal_window_hpp_
#define ovxx_signal_window_hpp_


#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>

namespace ovxx
{
namespace signal
{
template <typename T>
const_Vector<T>
blackman(length_type len) VSIP_THROW((std::bad_alloc));

template <typename T>
const_Vector<T>
cheby(length_type len, T ripple) VSIP_THROW((std::bad_alloc));

template <typename T>
const_Vector<T>
hanning(length_type len) VSIP_THROW((std::bad_alloc));

template <typename T>
const_Vector<T>
kaiser(length_type len, T beta) VSIP_THROW((std::bad_alloc));

} // namespace ovxx::signal
} // namespace ovxx

#endif
