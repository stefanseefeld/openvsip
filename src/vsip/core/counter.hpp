/* Copyright (c) 2005, 2006 CodeSourcery.  All rights reserved.  */

/** @file    vsip/core/counter.hpp
    @author  Zack Weinberg
    @date    2005-01-21
    @brief   VSIPL++ Library: Checked counter classes.
 
    This file defines classes useful in use and reference counting.  */

#ifndef VSIP_CORE_COUNTER_HPP
#define VSIP_CORE_COUNTER_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
/// All VSIPL++ internal interfaces live in namespace \c vsip::impl.
namespace impl
{

/// A Checked_counter wraps an unsigned int with operator functions that
/// notice over- or underflow and throw appropriate exceptions.
/// Multiplication, division, and bitwise operations are not supported.
struct Checked_counter
{
  /// This typedef centralizes the choice of underlying type.  It is
  /// public because it may be useful to users of this class: for instance,
  /// one may wish to refer to std::numeric_limits<Checked_counter::value_type>.
  typedef unsigned int value_type;
  
private:
  /// Underlying value.
  value_type val;

  /// Out-of-line function called when overflow is detected; does not return.
  static void overflow(value_type a, value_type b) VSIP_IMPL_NORETURN;

  /// Out-of-line function called when underflow is detected; does not return.
  static void underflow(value_type a, value_type b) VSIP_IMPL_NORETURN;

public:
  /// Default constructor, initializes a counter to 0.
  Checked_counter() VSIP_NOTHROW : val(0) {}
  
  /// Constructor from a value.
  Checked_counter(value_type v) VSIP_NOTHROW : val(v) {}

  /// Extract-value function.
  /// There is no operator value_type because we don't want
  /// implicit conversions losing the checkedness.
  value_type value() const VSIP_NOTHROW { return val; }

  // The implicit C++ copy constructor and operator= are correct for
  // this class.

  // Primitive arithmetic operations:
  /// Destructive add.
  inline Checked_counter operator+=(const Checked_counter x);

  /// Destructive subtract.
  inline Checked_counter operator-=(const Checked_counter x);
  
  // Arithmetic operations defined in terms of the primitives:
  /// Prefix increment.
  Checked_counter operator++() { return *this += 1; }

  /// Prefix decrement.
  Checked_counter operator--() { return *this -= 1; }

  /// Postfix increment.
  Checked_counter operator++(int)
    { Checked_counter old = *this; *this += 1; return old; }

  /// Postfix decrement.
  Checked_counter operator--(int)
    { Checked_counter old = *this; *this -= 1; return old; }

  /// Nondestructive add.
  Checked_counter operator+(Checked_counter x) const
    { Checked_counter r = *this; r += x; return r; }

  /// Nondestructive subtract.
  Checked_counter operator-(Checked_counter x) const
    { Checked_counter r = *this; r -= x; return r; }

#define NC const VSIP_NOTHROW // shorthand
  
  // Logical operations:

  // There is no operator bool, because that could be used to convert
  // to value_type as well, resulting in truncation of the actual value.
  bool operator==(const Checked_counter x) NC { return val == x.val; }
  bool operator!=(const Checked_counter x) NC { return val != x.val; }
  bool operator> (const Checked_counter x) NC { return val >  x.val; }
  bool operator< (const Checked_counter x) NC { return val <  x.val; }
  bool operator<=(const Checked_counter x) NC { return val <= x.val; }
  bool operator>=(const Checked_counter x) NC { return val >= x.val; }

#undef NC
};

// The nondestructive arithmetic and logical operators have to be
// repeated as non-member functions with a Checked_counter:value_type left
// argument.
inline Checked_counter
operator+(Checked_counter::value_type x, const Checked_counter y)
{ Checked_counter r = x; r += y; return r; }

inline Checked_counter
operator-(Checked_counter::value_type x, const Checked_counter y)
{ Checked_counter r = x; r -= y; return r; }

inline bool
operator==(Checked_counter::value_type x, const Checked_counter y) VSIP_NOTHROW
{ return x == y.value(); }

inline bool
operator!=(Checked_counter::value_type x, const Checked_counter y) VSIP_NOTHROW
{ return x != y.value(); }

inline bool
operator< (Checked_counter::value_type x, const Checked_counter y) VSIP_NOTHROW
{ return x <  y.value(); }

inline bool
operator> (Checked_counter::value_type x, const Checked_counter y) VSIP_NOTHROW
{ return x >  y.value(); }

inline bool
operator<=(Checked_counter::value_type x, const Checked_counter y) VSIP_NOTHROW
{ return x <= y.value(); }

inline bool
operator>=(Checked_counter::value_type x, const Checked_counter y) VSIP_NOTHROW
{ return x >= y.value(); }

inline Checked_counter
Checked_counter::operator+=(Checked_counter x)
{
  value_type sum = val + x.val;

  if (sum < val)
    overflow(val, x.val);  // does not return

  val = sum;
  return *this;
}

inline Checked_counter
Checked_counter::operator-=(Checked_counter x)
{
  value_type diff = val - x.val;

  if (diff > val)
    underflow(val, x.val);  // does not return

  val = diff;
  return *this;
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_COUNTER_HPP
