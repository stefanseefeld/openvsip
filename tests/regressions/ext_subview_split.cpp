//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <ovxx/strided.hpp>
#include <test.hpp>

using namespace ovxx;

template <storage_format_type C, dimension_type Dim>
void
run_test(stride_type        component_stride,
     Domain<Dim> const& dom)
{
  typedef complex<float> T;

  typedef Layout<Dim, row1_type, dense, C> layout_type;
  typedef Strided<Dim, T, layout_type> block_type;
  typedef typename view_of<block_type>::type view_type;

  block_type block(dom);
  view_type  view(block);


  vsip::dda::Data<block_type, vsip::dda::inout> ext(view.block());
  test_assert(ext.cost()        == 0);
  test_assert(ext.stride(Dim-1) == 1);

  stride_type str = 1;
  for (dimension_type d=0; d<Dim; ++d)
  {
    test_assert(ext.stride(Dim-d-1) == str);
    str *= dom[Dim-d-1].size();
  }


  typename view_type::realview_type real = view.real();

  vsip::dda::Data<typename view_type::realview_type::block_type, vsip::dda::inout>
    extr(real.block());

  test_assert(extr.cost()    == 0);
  test_assert(extr.stride(Dim-1) == component_stride);

  str = component_stride;
  for (dimension_type d=0; d<Dim; ++d)
  {
    test_assert(extr.stride(Dim-d-1) == str);
    str *= dom[Dim-d-1].size();
  }



  typename view_type::imagview_type imag = view.imag();

  vsip::dda::Data<typename view_type::imagview_type::block_type, vsip::dda::inout>
    exti(imag.block());

  test_assert(exti.cost()    == 0);
  test_assert(exti.stride(Dim-1) == component_stride);

  str = component_stride;
  for (dimension_type d=0; d<Dim; ++d)
  {
    test_assert(exti.stride(Dim-d-1) == str);
    str *= dom[Dim-d-1].size();
  }
}



int
main()
{
  vsip::vsipl init;
  run_test<interleaved_complex>(2, Domain<1>(5));
  run_test<split_complex>(1, Domain<1>(5));

  run_test<interleaved_complex>(2, Domain<2>(5, 7));
  run_test<split_complex>(1, Domain<2>(5, 7));

  run_test<interleaved_complex>(2, Domain<3>(5, 7, 9));
  run_test<split_complex>(1, Domain<3>(5, 7, 9));
}
