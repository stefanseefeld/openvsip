/* Copyright (c) 2007 by CodeSourcery.  All rights reserved. */

/** @file    tests/ea_from_ptr.cpp
    @author  Jules Bergmann
    @date    2007-04-16
    @brief   VSIPL++ Library: Tests for ea_from_ptr.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#if VSIP_IMPL_HAVE_CBE_SDK
#  include <vsip/opt/cbe/ppu/util.hpp>
#endif // VSIP_IMPL_HAVE_CBE_SDK

#include <vsip_csl/test.hpp>

using namespace vsip;



/***********************************************************************
  Definitions
***********************************************************************/

// Check conversion from ptr to EA using cast.

void
test_cast(char* ptr, unsigned long long expected_ea)
{
  unsigned long long ea;

  ea = reinterpret_cast<unsigned long long>(ptr);

  test_assert(ea == expected_ea);
}



// Check conversion from ptr to EA using ea_from_ptr.

void
test_conv(char* ptr, unsigned long long expected_ea)
{
#if VSIP_IMPL_HAVE_CBE_SDK
  unsigned long long ea;

  ea = vsip::impl::cbe::ea_from_ptr(ptr);

  test_assert(ea == expected_ea);
#else
  (void)ptr;
  (void)expected_ea;
#endif // VSIP_IMPL_HAVE_CBE_SDK
}



void
test()
{
// 1. Test using cast.
  if (sizeof(char*) == 4)
  {
    test_assert(sizeof(unsigned long long) == 8);

    test_cast((char*) 0x7fffffff,
	      0x000000007fffffffLL);

    // This is what we would like
    // test_cast((char*) 0x80000000,
    //	       0x0000000080000000LL);

    // However, this is what we get:
    test_cast((char*) 0x80000000,
	      0xffffffff80000000LL);
  }
  else
  {
    test_assert(sizeof(char*) == 8);
    test_assert(sizeof(unsigned long long) == 8);

    test_cast((char*) 0x7fffffff,
	      0x000000007fffffffLL);

    test_cast((char*) 0x80000000,
	      0x0000000080000000LL);
  }


// 2. Test using ea_from_ptr: 
  test_conv((char*) 0x7fffffff,
	    0x000000007fffffffLL);

  test_conv((char*) 0x80000000,
	    0x0000000080000000LL);
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test();
}
