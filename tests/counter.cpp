//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>
#include <stdexcept>
#include <typeinfo>
#include <limits>
#include <vsip/core/counter.hpp>

#include <vsip_csl/test.hpp>

using vsip::impl::Checked_counter;

/// Exercise all relational operators on Checked_counters.
/// This is done first so that the relational operators can
/// be relied on below.

static void
test_relational()
{
  Checked_counter a = 0;         test_assert(a.value() == 0);
  Checked_counter b = 27;        test_assert(b.value() == 27);
  Checked_counter c = 27;        test_assert(c.value() == 27);
  Checked_counter d = 40;        test_assert(d.value() == 40);

  // Equality.
  test_assert(b == c);
  test_assert(c == 27);
  test_assert(40 == d);
  test_assert(!(a == b));

  // Inequality.
  test_assert(a != b);
  test_assert(0 != c);
  test_assert(d != 27);
  test_assert(!(b != c));

  // Less than.
  test_assert(a < b);
  test_assert(10 < b);
  test_assert(d < 99);
  test_assert(!(b < c));

  // Greater than.
  test_assert(d > c);
  test_assert(90 > c);
  test_assert(c > 10);
  test_assert(!(b > c));
  
  // Less or equal.
  test_assert(a <= b);
  test_assert(b <= c);
  test_assert(d <= 41);
  test_assert(40 <= d);
  test_assert(!(b <= a));

  // Greater or equal.
  test_assert(b >= a);
  test_assert(b >= c);
  test_assert(d >= 39);
  test_assert(40 >= d);
  test_assert(!(a >= c));
}

/// Exercise all forms of non-over/underflowing addition.
static void
test_addition()
{
  Checked_counter a;      // default initialized to 0
  Checked_counter b = 23;

  a += 1; test_assert(a == 1);

  test_assert((a += 2) == 3); test_assert(a == 3);

  test_assert(a + 4 == 7); test_assert(a == 3);
  test_assert(4 + a == 7); test_assert(a == 3);

  test_assert(++a == 4); test_assert(a == 4);

  test_assert(a++ == 4); test_assert(a == 5);

  test_assert(b + a == 28); test_assert(b == 23); test_assert(a == 5);

  a += b; test_assert(a == 28); test_assert(b == 23);
}

/// Exercise all forms of non-over/underflowing subtraction.
static void
test_subtraction()
{
  Checked_counter a = 99;
  Checked_counter b = 23;

  a -= 1; test_assert(a == 98);

  test_assert((a -= 2) == 96); test_assert(a == 96);

  test_assert(a - 4 == 92); test_assert(a == 96);
  test_assert(100 - a == 4); test_assert(a == 96);

  test_assert(--a == 95); test_assert(a == 95);

  test_assert(a-- == 95); test_assert(a == 94);

  test_assert(a - b == 71); test_assert(b == 23); test_assert(a == 94);

  a -= b; test_assert(a == 71); test_assert(b == 23);
}

/// Underflow - minimal test.
static void
test_under()
{
#if VSIP_HAS_EXCEPTIONS
  try
  {
    Checked_counter a = 0;
    a--;
  }
  catch (std::underflow_error)
  {
    return;
  }
  test_assert (!"0-- failed to throw std::underflow_error\n");
#endif
}

/// Overflow - minimal test.
static void
test_over()
{
#if VSIP_HAS_EXCEPTIONS
  try
  {
    Checked_counter a = std::numeric_limits<Checked_counter::value_type>::max();
    a++;
  }
  catch (std::overflow_error)
  {
    return;
  }
  test_assert(!"UINT_MAX++ failed to throw std::overflow_error\n");
#endif
}

int
main(void)
{
#if VSIP_HAS_EXCEPTIONS
  try
  {
#endif
    test_relational();
    test_addition();
    test_subtraction();
    test_under();
    test_over();
#if VSIP_HAS_EXCEPTIONS
  }
  catch (std::exception& E)
  {
    std::cerr << "unexpected exception " << typeid(E).name()
              << ": " << E.what() << std::endl;
    test_assert(0);
  }
#endif
}
