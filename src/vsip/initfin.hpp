//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_INITFIN_HPP
#define VSIP_INITFIN_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/counter.hpp>

/***********************************************************************
  Declarations
***********************************************************************/

/// All VSIPL++ interfaces live in namespace \c vsip.
namespace vsip
{

/// Class for management of library private data structures.
///
/// While the VSIPL++ library is in use, at least one \c vsipl
/// object must exist.  Creation of more \c vsipl objects has
/// no additional effect.  The library remains usable until the
/// last vsipl object is destroyed.
///
/// Optionally, one may pass the program's command line arguments to
/// the vsipl constructor; they are then scanned for
/// implementation-defined options which modify the library's behavior.
/// (This only happens the first time a \c vsipl object is created.)
///
/// vsipl objects may not be copied.
class vsipl
{
public:
  /// Constructor requesting default library behavior.
  vsipl();

  /// Constructor requesting command-line-dependent library behavior.
  vsipl(int& argc, char**& argv);

  /// Destructor.
  ~vsipl() VSIP_NOTHROW;

private:
  // These are declared to prevent the compiler from synthesizing
  // default definitions; they are deliberately left undefined.
  vsipl(const vsipl&);            ///< Stub to block copy-initialization.
  vsipl& operator=(const vsipl&); ///< Stub to block assignment.

  // Internal variables:

  /// Count of active \c vsipl objects.
  static impl::Checked_counter use_count;

  /// Constructor worker function.
  static void initialize_library(int& argc, char**& argv);

  /// Destructor worker function.
  static void finalize_library();
};

} // namespace vsip

#endif // initfin.hpp
