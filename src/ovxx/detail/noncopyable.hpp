//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_detail_noncopyable_hpp_
#define ovxx_detail_noncopyable_hpp_

namespace ovxx
{
namespace detail
{
//.  Explicitely disallow copying instances of classes derived from
//.  noncopyable.
class noncopyable
{
  // Constructor and Destructor are protected to emphasize that
  // noncopyable should only be used as a base class.
protected:
  noncopyable() {}
  ~noncopyable() {}
private:
  noncopyable(noncopyable const&);
  noncopyable const &operator=(noncopyable const&);
};

class nonassignable
{
protected:
  nonassignable() {}
  ~nonassignable() {}
private:
  nonassignable const &operator=(nonassignable const&);
};

} // namespace ovxx::detail
} // namespace ovxx

#endif
