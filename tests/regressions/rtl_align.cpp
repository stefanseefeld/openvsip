/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/regressions/rtl_align.cpp
    @author  Jules Bergmann
    @date    2007-06-15
    @brief   VSIPL++ Library: Regression test for aligned Rt_layout.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/core/layout.hpp>

#include <vsip_csl/test.hpp>

using namespace vsip;



/***********************************************************************
  Definitions
***********************************************************************/

void
test_aligned_rtl_2()
{
  dimension_type const dim = 2;

  impl::Rt_layout<dim> rtl;

  length_type rows = 2;
  length_type cols = 5;
  length_type align = 32;
  length_type elem_size = 8;

  rtl.packing = aligned;
  rtl.order = impl::Rt_tuple(0, 1, 2);
  rtl.storage_format = interleaved_complex;
  rtl.alignment = align;

  impl::Length<dim> ext(rows, cols);

  impl::Applied_layout<impl::Rt_layout<dim> > layout(rtl, ext, elem_size);

  test_assert(layout.size(0) == rows);
  test_assert(layout.size(1) == cols);

  // Check that alignment was achieved.
  test_assert((layout.stride(0)*elem_size) % align == 0);

  // Check that bounds of matrix didn't shrink while trying to
  // achieve alignement.  (This was not being handled correctly).
  test_assert(layout.stride(0) >= static_cast<stride_type>(cols));
  test_assert(layout.stride(1) == 1);
}



void
test_aligned_rtl_3()
{
  dimension_type const dim = 3;

  impl::Rt_layout<dim> rtl;

  length_type dim0      = 2;
  length_type dim1      = 2;
  length_type dim2      = 5;
  length_type align     = 32;
  length_type elem_size = 8;

  rtl.packing = aligned;
  rtl.order = impl::Rt_tuple(0, 1, 2);
  rtl.storage_format = interleaved_complex;
  rtl.alignment = align;

  impl::Length<dim> ext(dim0, dim1, dim2);

  impl::Applied_layout<impl::Rt_layout<dim> > layout(rtl, ext, elem_size);

  test_assert(layout.size(0) == dim0);
  test_assert(layout.size(1) == dim1);
  test_assert(layout.size(2) == dim2);

  // Check that alignment was achieved.
  test_assert((layout.stride(0)*elem_size) % align == 0);
  test_assert((layout.stride(1)*elem_size) % align == 0);

  // Check that bounds of matrix didn't shrink while trying to
  // achieve alignement.  (This was not being handled correctly).
  test_assert(layout.stride(0) >= static_cast<stride_type>(dim1*dim2));
  test_assert(layout.stride(1) >= static_cast<stride_type>(dim2));
  test_assert(layout.stride(2) == 1);
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_aligned_rtl_2();
  test_aligned_rtl_3();
}
