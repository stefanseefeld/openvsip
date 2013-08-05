/* Copyright (c) 2009, 2011 CodeSourcery, Inc.  All rights reserved. */

/* Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
         copyright notice, this list of conditions and the following
         disclaimer in the documentation and/or other materials
         provided with the distribution.
       * Neither the name of CodeSourcery nor the names of its
         contributors may be used to endorse or promote products
         derived from this software without specific prior written
         permission.

   THIS SOFTWARE IS PROVIDED BY CODESOURCERY, INC. "AS IS" AND ANY
   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CODESOURCERY BE LIABLE FOR
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
   BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
   EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  */

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

