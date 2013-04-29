//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_initfin_hpp_
#define vsip_initfin_hpp_

#include <vsip/support.hpp>
#include <ovxx/library.hpp>

namespace vsip
{

/// Class for management of library private data structures.
///
/// While the VSIPL++ library is in use, at least one vsipl
/// object must exist.  Creation of more vsipl objects has
/// no additional effect.  The library remains usable until
/// the last vsipl object is destroyed.
///
/// Optionally, one may pass the program's command line 
/// arguments to the vsipl constructor; they are then scanned 
/// for implementation-defined options which modify the library's behavior.
/// (This only happens the first time a \c vsipl object is created.)
///
/// vsipl objects may not be copied.
class vsipl : ovxx::detail::noncopyable
{
public:
  /// Constructor requesting default library behavior.
  vsipl() : impl_() {}
  /// Constructor requesting command-line-dependent library behavior.
  vsipl(int& argc, char**& argv) : impl_(argc, argv) {}
  ~vsipl() VSIP_NOTHROW {}

private:
  ovxx::library impl_;
};

} // namespace vsip

#endif
