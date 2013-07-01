//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/config.hpp>
#include <ovxx/support.hpp>
#include <ovxx/layout.hpp>
#include <ovxx/aligned_array.hpp>
#include <vsip/dense.hpp>
#include <fftw3.h>

namespace ovxx
{
namespace fftw
{

// FFTW needs to know the complex storage-format during planning.
// Since the actual storage-format is only known at execution time,
// we need to use a best guess, and then require that via query_layout().
// As best guess we use the library default complex storage-format.
storage_format_type const complex_storage_format = Dense<1, complex<float> >::storage_format;

// Turn a dimension to make 'A' the major axis
Domain<1> turn(Domain<1> const &dom, int) { return dom;}
Domain<2> turn(Domain<2> const &dom, int A)
{
  return A == 1 ? dom : Domain<2>(dom[1], dom[0]);
}
Domain<3> turn(Domain<3> const &dom, int A)
{
  switch (A)
  {
    case 2: return dom;
    case 1: return Domain<3>(dom[0], dom[2], dom[1]);
    default: return Domain<3>(dom[2], dom[1], dom[0]);
  }
}

// Use a rigor flag dependent on the number-of-times
// argument given to FFTs.
inline int
rigor(unsigned int number)
{
  // a number value of '0' means 'infinity', and so is captured
  // by a wrap-around.
  if (number - 1 > 30) return FFTW_PATIENT;
  if (number - 1 > 10) return FFTW_MEASURE;
  return FFTW_ESTIMATE;
}

template <typename T>
int make_flags(Domain<1> const &dom, unsigned NoT, bool preserve_input = true)
{
  int flags = rigor(NoT);
  // preserving input is the default for all but c2r, where it
  // incures some overhead. The alternative would be to clone
  // the input in the workspace...
  if (preserve_input)
    flags |= FFTW_PRESERVE_INPUT;
  // Don't require aligned input if FFTW can't take advantage of it
  // anyhow.
  if ((OVXX_ALLOC_ALIGNMENT % sizeof(T)) ||
      ((sizeof(T) * dom.length()) % OVXX_ALLOC_ALIGNMENT))
    flags |= FFTW_UNALIGNED;
  return flags;
}

int make_flags(unsigned NoT, bool preserve_input = true)
{
  int flags = rigor(NoT);
  if (preserve_input)
    flags |= FFTW_PRESERVE_INPUT;
  flags |= FFTW_UNALIGNED;
  return flags;
}

template <dimension_type D, typename I, typename O> struct planner;
template <dimension_type D, typename I, typename O, int S> class fft;
template <typename I, typename O, int A, int D> class fftm;

} // namespace ovxx::fftw
} // namespace ovxx

#ifdef OVXX_FFTW_HAVE_FLOAT
#  define FFTW(fun) fftwf_##fun
#  define SCALAR_TYPE float
#  include "fftw.cpp"
#  undef SCALAR_TYPE
#  undef FFTW
#endif
#ifdef OVXX_FFTW_HAVE_DOUBLE
#  define FFTW(fun) fftw_##fun
#  define SCALAR_TYPE double
#  include "fftw.cpp"
#  undef SCALAR_TYPE
#  undef FFTW
#endif
