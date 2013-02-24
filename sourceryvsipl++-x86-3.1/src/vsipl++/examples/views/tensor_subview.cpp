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

  // Create a tensor and clear all the elements.
  typedef scalar_f T;
  Tensor<T> view(3, 4, 5, T());

  // Assign each plane a different value.
  for (index_type x = 0; x < 3; ++x)
    view(x, whole_domain, whole_domain) = T(x+1);

  // Print the tensor.
  std::cout << view << std::endl;

  std::cout << "--------------" << std::endl;

  // Create a reference view aliasing the middle plane of the tensor...
  Tensor<T>::submatrix<0>::type subview = view(1, whole_domain, whole_domain);

  // ...and reset it to 0.
  subview = 0.;

  std::cout << view << std::endl;
}

