//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_python_config_hpp_
#define ovxx_python_config_hpp_

#include <stdexcept>

// Allow throwing exceptions everywhere...
#define VSIP_NOTHROW
// ...and throw an exception when a precondition isn't met.
#define OVXX_PRECONDITION(c) \
if (!(c)) OVXX_DO_THROW(std::runtime_error(#c " evaluates to false"))

// These bindings use the NumPy 1.7 API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#endif
