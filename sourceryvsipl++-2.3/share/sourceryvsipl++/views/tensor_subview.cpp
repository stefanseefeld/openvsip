/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   Matrix subview of a Tensor example.

#include <iostream>
#include <vsip/initfin.hpp>
#include <vsip/tensor.hpp>
#include <vsip_csl/output.hpp>

using namespace vsip;

int 
main(int argc, char **argv)
{
  vsipl init(argc, argv);

  // Create a tensor and clear all the elements
  typedef scalar_f T;
  Tensor<T> view(2, 3, 5, T());

  // Set all the slices to a different value
  for (index_type z = 0; z < 5; ++z)
    view(whole_domain, whole_domain, z) = T(z+1);

  // Look through all the slices
  for (index_type z = 0; z < 5; ++z)
    std::cout << view(whole_domain, whole_domain, z) << std::endl;

  std::cout << "--------------" << std::endl;

  // Create a reference view aliasing the same data in memory.  
  Tensor<T>::submatrix<2>::type subview = view(Domain<1>(2), Domain<1>(3), 2);

  // Use it to clear the middle slice
  subview = T();

  for (index_type z = 0; z < 5; ++z)
    std::cout << view(whole_domain, whole_domain, z) << std::endl;
}

