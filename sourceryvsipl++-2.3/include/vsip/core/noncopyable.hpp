/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/noncopyable.hpp
    @author  Stefan Seefeld
    @date    2005-03-01
    @brief   VSIPL++ Library: Non_copyable base class.

    This file defines the Non_copyable base class.
*/

#ifndef VSIP_CORE_NONCOPYABLE_HPP
#define VSIP_CORE_NONCOPYABLE_HPP

/***********************************************************************
  Included Files
***********************************************************************/

namespace vsip
{
namespace impl
{

/***********************************************************************
  Declarations
***********************************************************************/

///  Explicitely disallow copying instances of classes derived from
///  Non_copyable.
class Non_copyable
{
  // Constructor and Destructor are protected to emphasize that
  // Non_copyable should only be used as a base class.
protected:
  Non_copyable() {}
  ~Non_copyable() {}
private:
  Non_copyable(Non_copyable const&);
  Non_copyable const& operator=(Non_copyable const&);
};



///  Explicitely disallow assignment to instances of classes derived from
///  Non_assignable.
class Non_assignable
{
  // Ideally, Constructor and Destructor are protected to emphasize
  // that Non_assignable should only be used as a base class.  However,
  // GCC issues '-W -Wall' warnings if classes derived from
  // Non_assignable do not initialize base class during copy
  // constructors.  We avoid this warning by not defining using the
  // default constructor.
protected:
  // Non_assignable() {}
  ~Non_assignable() {}
private:
  Non_assignable const& operator=(Non_assignable const&);
};

} // namespace vsip::impl
} // namespace vsip

#endif
