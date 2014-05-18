//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/opencl/library.hpp>
#include <ovxx/opencl/context.hpp>
#ifdef OVXX_HAVE_CLMATH
# include <clAmdBlas.h>
#endif

namespace ovxx
{
namespace opencl
{

void initialize()
{
#ifdef OVXX_HAVE_CLMATH
  OVXX_OPENCL_CHECK_RESULT(clAmdBlasSetup, ());
#endif
}
void finalize()
{
#ifdef OVXX_HAVE_CLMATH
  clAmdBlasTeardown();
#endif
}

} // namespace ovxx::opencl
} // namespace ovxx
