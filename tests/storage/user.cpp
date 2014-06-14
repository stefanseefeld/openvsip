//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/library.hpp>
#include <test.hpp>
#include "util.hpp"

template <typename       T,
	  typename       Order,
	  dimension_type Dim,
	  typename       BlockT>
void
rebind_array(Domain<Dim> const& dom, BlockT &block, int k)
{
  length_type const size = block.size();
  T* data = new T[size];
  T* ptr;

  fill_array_storage<Order>(data, dom, Filler<T>(k, 1));

  block.rebind(data);

  test_assert(block.admitted()     == false);
  test_assert(block.user_storage() == array_format); // rebind could change format

  block.find(ptr);
  test_assert(ptr == data);

  block.admit(true);
  test_assert(block.admitted() == true);

  test_assert(check_block<Order>(block, dom, Filler<T>(k, 1)));

  fill_block<Order>(block, dom, Filler<T>(k+1, 1));

  block.release(true);
  test_assert(block.admitted() == false);

  test_assert(check_array_storage<Order>(data, dom, Filler<T>(k+1, 1)));

  delete[] data;
}


template <typename       T,
	  typename       Order,
	  dimension_type Dim,
	  typename       BlockT>
void
rebind_split(Domain<Dim> const& dom, BlockT &block, int k)
{
  length_type const size = block.size();

  T* real = new T[size];
  T* imag = new T[size];

  T* real_ptr;
  T* imag_ptr;

  fill_split_storage<Order>(std::make_pair(real, imag), dom, CFiller<T>(k, 0, 1));

  block.rebind(real, imag);

  test_assert(block.admitted()     == false);
  test_assert(block.user_storage() == split_format); // rebind could change format

  block.find(real_ptr, imag_ptr);
  test_assert(real_ptr == real);
  test_assert(imag_ptr == imag);

  block.admit(true);
  test_assert(block.admitted() == true);

  test_assert(check_block<Order>(block, dom, CFiller<T>(k, 0, 1)));
  fill_block<Order>(block, dom, CFiller<T>(k+1, 0, 1));

  block.release(true);
  test_assert(block.admitted() == false);

  test_assert(check_split_storage<Order>(std::make_pair(real, imag), dom, CFiller<T>(k+1, 0, 1)));

  delete[] real;
  delete[] imag;
}


template <typename       T,
	  typename       Order,
	  dimension_type Dim,
	  typename       BlockT>
void
rebind_interleaved(Domain<Dim> const& dom, BlockT &block, int k)
{
  length_type const size = block.size();

  T* data = new T[2*size];
  T* ptr;

  fill_interleaved_storage<Order>(data, dom, CFiller<T>(k, 0, 1));

  block.rebind(data);
  
  test_assert(block.admitted()     == false);
  test_assert(block.user_storage() == interleaved_format);

  block.find(ptr);
  test_assert(ptr == data);

  block.admit(true);
  test_assert(block.admitted() == true);

  test_assert(check_block<Order>(block, dom, CFiller<T>(k, 0, 1)));
  fill_block<Order>(block, dom, CFiller<T>(k+1, 0, 1));

  block.release(true);
  test_assert(block.admitted() == false);

  test_assert(check_interleaved_storage<Order>(data, dom, CFiller<T>(k+1, 0, 1)));

  delete[] data;
}



// Test admit/release of array-format data.
//
// Requires (Template):
//   T is a block value type,
//   ORDER is a dimension-order tuple,
//   DIM is the block dimensionality (inferred from DOM)
//
// Requires (Arguments):
//   DOM is a domain indicating the block dimensions.

template <typename       T,
	  typename       O,
	  dimension_type D>
void
test_array_format(Domain<D> const& dom)
{
  typedef Layout<D, O, dense, array> layout_type;
  T* data = new T[dom.size()];
  T* ptr;

  stored_block<T, layout_type> block(dom, data);

  test_assert(block.admitted()     == false);
  test_assert(block.user_storage() == array_format);

  // Check find()
  block.find(ptr);
  test_assert(ptr == data);

  fill_array_storage<O>(data, dom, Filler<T>(3, 0));

  block.admit(true);
  test_assert(block.admitted() == true);

  test_assert(check_block<O>(block, dom, Filler<T>(3, 0)));

  fill_block<O>(block, dom, Filler<T>(3, 1));

  block.release(true);
  test_assert(block.admitted() == false);

  test_assert(check_array_storage<O>(data, dom, Filler<T>(3, 1)));

  // Check release with pointer
  block.admit(true);
  block.release(true, ptr);

  test_assert(ptr == data);

  delete[] data;

  for (int i=0; i<5; ++i)
  {
    rebind_array<T, O>(dom, block, 5);
    rebind_array<T, O>(dom, block, 6);
    rebind_array<T, O>(dom, block, 7);
  }
}



template <typename       T,
	  typename       O,
	  dimension_type D>
void
test_interleaved_format(Domain<D> const& dom)
{
  typedef Layout<D, O, dense, array> layout_type;
  T* data = new T[2*dom.size()];
  T* ptr;

  stored_block<complex<T>, layout_type> block(dom, data);

  test_assert(block.admitted()     == false);
  test_assert(block.user_storage() == interleaved_format);

  block.find(ptr);
  test_assert(ptr == data);

  fill_interleaved_storage<O>(data, dom, CFiller<T>(3, 0, 2));

  block.admit(true);
  test_assert(block.admitted() == true);

  test_assert(check_block<O>(block, dom, CFiller<T>(3, 0, 2)));
  fill_block<O>(block, dom, CFiller<T>(3, 2, 1));

  block.release(true);
  test_assert(block.admitted() == false);

  test_assert(check_interleaved_storage<O>(data, dom, CFiller<T>(3, 2, 1)));

  // Check release with pointer
  block.admit(true);
  block.release(true, ptr);

  test_assert(ptr == data);

  delete[] data;

  for (int i=0; i<5; ++i)
  {
    rebind_interleaved<T, O>(dom, block, 4);
    rebind_split<T, O>      (dom, block, 5);
    rebind_split<T, O>      (dom, block, 6);
    rebind_interleaved<T, O>(dom, block, 7);
    rebind_interleaved<T, O>(dom, block, 8);
  }
}


template <typename       T,
	  typename       O,
	  dimension_type D>
void
test_split_format(Domain<D> const& dom)
{
  typedef Layout<D, O, dense, array> layout_type;
  T* real = new T[dom.size()];
  T* imag = new T[dom.size()];

  T* real_ptr;
  T* imag_ptr;

  stored_block<complex<T>, layout_type> block(dom, std::make_pair(real, imag));

  test_assert(block.admitted()     == false);
  test_assert(block.user_storage() == split_format);

  block.find(real_ptr, imag_ptr);
  test_assert(real_ptr == real);
  test_assert(imag_ptr == imag);

  fill_split_storage<O>(std::make_pair(real, imag), dom, CFiller<T>(3, 0, 2));

  block.admit(true);
  test_assert(block.admitted() == true);

  test_assert(check_block<O>(block, dom, CFiller<T>(3, 0, 2)));
  fill_block<O>(block, dom, CFiller<T>(3, 2, 1));

  block.release(true);
  test_assert(block.admitted() == false);

  test_assert(check_split_storage<O>(std::make_pair(real, imag), dom, CFiller<T>(3, 2, 1)));

  // Check release with pointer
  block.admit(true);
  block.release(true, real_ptr, imag_ptr);

  test_assert(real_ptr == real);
  test_assert(imag_ptr == imag);

  delete[] real;
  delete[] imag;

  for (int i=0; i<5; ++i)
  {
    rebind_split<T, O>      (dom, block, 4);
    rebind_interleaved<T, O>(dom, block, 5);
    rebind_interleaved<T, O>(dom, block, 6);
    rebind_split<T, O>      (dom, block, 7);
    rebind_split<T, O>      (dom, block, 8);
  }
}

template <typename       T,
	  typename       O,
	  dimension_type D>
void
test_no_user_format(Domain<D> const& dom)
{
  typedef Layout<D, O, dense, array> layout_type;
  T* data = new T[dom.size()];
  T* ptr;

  // initially the block doesn't have user storage
  stored_block<T, layout_type> block(dom);

  // test_assert(block.admitted()     == true);
  test_assert(block.user_storage() == no_user_format);

  // Check find()
  block.find(ptr);
  test_assert(ptr == 0);

  fill_array_storage<O>(data, dom, Filler<T>(3, 0));

  block.release(false);
  block.rebind(data);
  block.admit(true);
  test_assert(block.admitted() == true);

  test_assert(check_block<O>(block, dom, Filler<T>(3, 0)));

  // Check release with pointer
  block.admit(true);
  block.release(true, ptr);

  test_assert(ptr == data);

  delete[] data;
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

  test_no_user_format<float,          row1_type>(Domain<1>(50));
}
