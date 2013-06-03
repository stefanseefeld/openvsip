//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_lapack_lapack_hpp_
#define ovxx_lapack_lapack_hpp_

#include <ovxx/config.hpp>

#ifdef OVXX_HAVE_LAPACKE
# include <ovxx/lapack/lapacke.hpp>
#else
# include <ovxx/lapack/flapack.hpp>
#endif

#endif
