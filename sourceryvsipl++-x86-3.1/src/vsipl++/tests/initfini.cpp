/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    initfini.cpp
    @author  Zack Weinberg
    @date    2005-01-20
    @brief   VSIPL++ Library: Unit tests for [initfin] items.

    This file has unit tests for functionality defined in the [initfin]
    section of the VSIPL++ specification.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <vsip/initfin.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

static void
use_the_library ()
{
  // This routine should attempt to use the library in some
  // relatively simple way (say, create a couple of vectors and add
  // them) to check that it is properly initialized at this point.
  // We cannot do this until more of the library is written.
}

static void
test_basic ()
{
  vsipl v;

  use_the_library ();
}

static void
test_two_objects ()
{
  vsipl v1;
  use_the_library ();

  {
    vsipl v2;
    use_the_library ();
  }

  use_the_library ();
}

static void
test_overlapping_lifetimes ()
{
  vsipl *v1 = new vsipl;

  use_the_library ();

  vsipl *v2 = new vsipl;

  use_the_library ();

  delete v1;

  use_the_library ();

  delete v2;
}

static void
test_cmdline_options ()
{
  // As at present no VSIPL++ command line options are defined,
  // we just test that the constructor is callable with some random
  // strings in argv.
  const char *argv_const[] = { "foo", "bar", "baz", 0 };
  char **argv = const_cast<char **>(argv_const);
  int argc = 3;

  vsipl v(argc, argv);

  use_the_library();
}

int
main (int argc, char** argv)
{
  int test = 0;

  for (int i=0; i<argc; ++i)
  {
    if (!strcmp(argv[i], "-test"))
      test = atoi(argv[++i]);
  }

  switch(test)
  {
  case 0:
    test_basic ();
    break;
  case 1:
    test_two_objects ();
    break;
  case 2:
    test_overlapping_lifetimes ();
    break;
  case 3:
    test_cmdline_options ();
    break;
  default:
    test_assert(0);
  }
}
