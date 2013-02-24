/* Copyright (c) 2005 CodeSourcery, LLC.  All rights reserved.  */

/** @file    initfin.hpp
    @author  Zack Weinberg
    @date    2005-01-19
    @brief   VSIPL++ Library: [initfin] Initialization and finalization.
 
   This file declares the mechanism for initialization and
   finalization of the library's private data structures.  Use of any
   library routines while no \c vsipl object exists provokes
   undefined behavior.  */

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

namespace impl
{
// Forward Declaration
class Par_service;
namespace profile
{
class Profiler_options;
} // namespace profile
} // namespace impl
  
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

  static impl::Par_service* impl_par_service()
  { return par_service_; }

private:
  // These are declared to prevent the compiler from synthesizing
  // default definitions; they are deliberately left undefined.
  vsipl(const vsipl&);            ///< Stub to block copy-initialization.
  vsipl& operator=(const vsipl&); ///< Stub to block assignment.

  // Internal variables:

  /// Count of active \c vsipl objects.
  static impl::Checked_counter use_count;

  /// Parallel Service.
  static impl::Par_service* par_service_;

  /// Profiler.
  static impl::profile::Profiler_options* profiler_opts_;

  // Internal functions:
private:

  /// Constructor worker function.
  static void initialize_library(int& argc, char**& argv);

  /// Destructor worker function.
  static void finalize_library();
};

} // namespace vsip

#endif // initfin.hpp
