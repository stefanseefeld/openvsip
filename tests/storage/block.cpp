//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/library.hpp>
#include <test.hpp>
#include "util.hpp"

template <typename T, typename O, dimension_type D>
void
test_array_format(Domain<D> const &dom)
{
  typedef Layout<D, O, dense, array> layout_type;

  stored_block<T, layout_type> block(dom);
  fill_block<O>(block, dom, Filler<T>(3, 1));
  test_assert(check_block<O>(block, dom, Filler<T>(3, 1)));
  test_assert(check_array_storage<O>(block.ptr(), dom, Filler<T>(3, 1)));
}

template <typename T, typename O, dimension_type D>
void
test_interleaved_format(Domain<D> const &dom)
{
  typedef Layout<D, O, dense, interleaved_complex> layout_type;

  stored_block<complex<T>, layout_type> block(dom);
  fill_block<O>(block, dom, CFiller<T>(3, 2, 1));
  test_assert(check_block<O>(block, dom, CFiller<T>(3, 2, 1)));
  test_assert(check_interleaved_storage<O>(block.ptr(), dom, CFiller<T>(3, 2, 1)));
}

template <typename T, typename O, dimension_type D>
void
test_split_format(Domain<D> const &dom)
{
  typedef Layout<D, O, dense, split_complex> layout_type;
  stored_block<complex<T>, layout_type> block(dom);

  fill_block<O>(block, dom, CFiller<T>(3, 2, 1));
  test_assert(check_block<O>(block, dom, CFiller<T>(3, 2, 1)));
  test_assert(check_split_storage<O>(block.ptr(), dom, CFiller<T>(3, 2, 1)));
}

int
main(int argc, char** argv)
{
  library lib(argc, argv);

  test_array_format<float,          row1_type>(Domain<1>(50));
  test_array_format<complex<float>, row1_type>(Domain<1>(50));

  test_array_format<float,          row2_type>(Domain<2>(25, 50));
  test_array_format<complex<float>, row2_type>(Domain<2>(50, 25));
  test_array_format<float,          col2_type>(Domain<2>(25, 50));
  test_array_format<complex<float>, col2_type>(Domain<2>(50, 25));

  test_array_format<float, tuple<0, 1, 2> >(Domain<3>(25, 50, 13));
  test_array_format<float, tuple<0, 2, 1> >(Domain<3>(25, 50, 13));
  test_array_format<float, tuple<1, 0, 2> >(Domain<3>(25, 50, 13));
  test_array_format<float, tuple<2, 0, 1> >(Domain<3>(25, 50, 13));
  test_array_format<float, tuple<1, 2, 0> >(Domain<3>(25, 50, 13));
  test_array_format<float, tuple<2, 1, 0> >(Domain<3>(25, 50, 13));

  test_split_format<float, row1_type>(Domain<1>(50));
  test_split_format<float, row2_type>(Domain<2>(25, 35));
  test_split_format<float, col2_type>(Domain<2>(45, 15));
  test_split_format<float, tuple<0, 1, 2> >(Domain<3>(25, 50, 13));
  test_split_format<float, tuple<0, 2, 1> >(Domain<3>(25, 50, 13));
  test_split_format<float, tuple<1, 0, 2> >(Domain<3>(25, 50, 13));
  test_split_format<float, tuple<2, 0, 1> >(Domain<3>(25, 50, 13));
  test_split_format<float, tuple<1, 2, 0> >(Domain<3>(25, 50, 13));
  test_split_format<float, tuple<2, 1, 0> >(Domain<3>(25, 50, 13));

  test_interleaved_format<float, row1_type>(Domain<1>(50));
  test_interleaved_format<float, row2_type>(Domain<2>(25, 37));
  test_interleaved_format<float, col2_type>(Domain<2>(91, 13));
  test_interleaved_format<float, tuple<0, 1, 2> >(Domain<3>(25, 50, 13));
  test_interleaved_format<float, tuple<0, 2, 1> >(Domain<3>(25, 50, 13));
  test_interleaved_format<float, tuple<1, 0, 2> >(Domain<3>(25, 50, 13));
  test_interleaved_format<float, tuple<2, 0, 1> >(Domain<3>(25, 50, 13));
  test_interleaved_format<float, tuple<1, 2, 0> >(Domain<3>(25, 50, 13));
  test_interleaved_format<float, tuple<2, 1, 0> >(Domain<3>(25, 50, 13));
}
