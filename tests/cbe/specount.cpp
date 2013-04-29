/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.
*/

/** @file    tests/plugin/alf_init.cpp
    @author  Brooks Moses
    @date    2009-11-27
    @brief   VSIPL++ Library: Test ALF initialization and SPE counts.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <algorithm>

#include <vsip/initfin.hpp>
#include <vsip_csl/cbe.hpp>
#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;


/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsip_csl::cbe::set_num_spes(2);

  // Before initialization, this is -1.
  test_assert(vsip_csl::cbe::get_num_spes() == -1);

  {
  vsipl init(argc, argv);

  // On any Cell hardware, we should be able to get at least two SPEs.
  test_assert(vsip_csl::cbe::get_num_spes() == 2);

  // Similarly, we should be able to get at least three SPEs.
  vsip_csl::cbe::set_num_spes(3);
  test_assert(vsip_csl::cbe::get_num_spes() == 3);

  // And we should get something sane if we ask for too many.
  vsip_csl::cbe::set_num_spes(256);
  test_assert(vsip_csl::cbe::get_num_spes() > 0);
  test_assert(vsip_csl::cbe::get_num_spes() < 256);

  // We can also test this with validation.
  vsip_csl::cbe::set_num_spes(3, true);
  test_assert(vsip_csl::cbe::get_num_spes() == 3);
  
  // With validation, we should get an error if we ask for too many.
  bool err_caught = false;
  try { vsip_csl::cbe::set_num_spes(256, true); }
  catch(std::bad_alloc) { err_caught = true; }
  test_assert(err_caught);

  // Finally, after library finalization, we should have -1 again.
  }
  test_assert(vsip_csl::cbe::get_num_spes() == -1);
  
  return 0;
}
