//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>
#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/dense.hpp>
#include <test.hpp>

using namespace ovxx;

// The purpose of this class is to allow overloading the ramp and
// verify_ramp functions on the block's dimension.
template <typename Block, dimension_type Dim = Block::dim>
struct B;

template <typename Block>
struct B<Block, 1>
{
  typedef typename Block::value_type T;
  static void ramp(Block& block)
    {
      for (index_type i = 0; i < block.size(1, 0); i++)
        block.put(i, T(i));
    }
  static void verify_ramp(Block const& block, Domain<1> dom)
    {
      for (index_type i = 0; i < dom[0].length(); i++)
        test_assert(equal(block.get(i), T(dom[0].impl_nth(i))));
    }
};

template <typename Block>
struct B<Block, 2>
{
  typedef typename Block::value_type T;
  static void ramp(Block& block)
    {
      for (index_type i = 0; i < block.size(2, 0); i++)
        for (index_type j = 0; j < block.size(2, 1); i++)
          block.put(i, j, T(100*i + j));
    }
  static void verify_ramp(Block const& block, Domain<2> dom)
    {
      for (index_type i = 0; i < dom[0].length(); i++)
        for (index_type j = 0; j < dom[1].length(); j++)
          test_assert(equal(block.get(i, j),
                       T(dom[0].impl_nth(i)*100 + dom[1].impl_nth(j))));
    }

};

template <typename T>
void test_slices_1d(void)
{
  typedef Dense<1,T> D1T;
  typedef expr::Subset<D1T> SD1T;
  
  D1T block(Domain<1>(2*3*5*7));  block.increment_count();

  B<D1T>::ramp(block);

  SD1T s2(Domain<1>(3*5*7)*2, block);   s2.increment_count();
  SD1T s3(Domain<1>(2*5*7)*3, block);   s3.increment_count();
  SD1T s5(Domain<1>(2*3*7)*5, block);   s5.increment_count();
  SD1T s7(Domain<1>(2*3*5)*7, block);   s7.increment_count();

  B<SD1T>::verify_ramp(s2, Domain<1>(3*5*7)*2);
  B<SD1T>::verify_ramp(s3, Domain<1>(2*5*7)*3);
  B<SD1T>::verify_ramp(s5, Domain<1>(2*3*7)*5);
  B<SD1T>::verify_ramp(s7, Domain<1>(2*3*5)*7);
}

template <typename T>
void test_subset_write_1d(void)
{
  typedef Dense<1,T> D1T;
  typedef expr::Subset<D1T> SD1T;

  D1T block(Domain<1>(50)); block.increment_count();

  for (index_type i = 0; i < block.size(); i++)
    block.put(i, 1);

  SD1T s2(Domain<1>(25)*2, block); s2.increment_count(); // every other element
  B<SD1T>::ramp(s2);

  for (index_type i = 0; i < block.size(); i++)
    test_assert(equal(block.get(i), T((i % 2) ? 1 : i/2)));
}
  
int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  test_slices_1d<int>();
  test_slices_1d<float>();

  test_subset_write_1d<int>();
  test_subset_write_1d<float>();
}
