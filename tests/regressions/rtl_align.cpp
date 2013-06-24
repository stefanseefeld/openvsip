//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <ovxx/layout.hpp>
#include <test.hpp>

using namespace ovxx;

void
test_aligned_rtl_2()
{
  dimension_type const dim = 2;

  Rt_layout<dim> rtl;

  length_type rows = 2;
  length_type cols = 5;
  length_type align = 32;
  length_type elem_size = 8;

  rtl.packing = aligned;
  rtl.order = Rt_tuple(0, 1, 2);
  rtl.storage_format = interleaved_complex;
  rtl.alignment = align;

  Length<dim> ext(rows, cols);

  Applied_layout<Rt_layout<dim> > layout(rtl, ext, elem_size);

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

  Rt_layout<dim> rtl;

  length_type dim0      = 2;
  length_type dim1      = 2;
  length_type dim2      = 5;
  length_type align     = 32;
  length_type elem_size = 8;

  rtl.packing = aligned;
  rtl.order = Rt_tuple(0, 1, 2);
  rtl.storage_format = interleaved_complex;
  rtl.alignment = align;

  Length<dim> ext(dim0, dim1, dim2);

  Applied_layout<Rt_layout<dim> > layout(rtl, ext, elem_size);

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
