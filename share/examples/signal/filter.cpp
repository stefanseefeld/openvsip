/* Copyright (c) 2010, 2011 by CodeSourcery, Inc.  All rights reserved. */

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
///   Simple filter example utilizing smart wrappers around SV++ signal 
///   processing objects.

#include <assert.h>
#include <iostream>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/math.hpp>

using vsip::vsipl;
using vsip::Fft;
using vsip::cscalar_f;
using vsip::Domain;
using vsip::Vector;
using vsip::const_Vector;
using vsip::fft_fwd;
using vsip::fft_inv;
using vsip::length_type;


// This example shows how one can wrap VSIPL++ signal processing objects
// to conveniently allow them to be shared between modules processing data
// in a similar fashion.  This sharing saves the cost of initializing
// multiple FFT objects, which can be a significant savings in not only
// time, but also memory.  See simple_filter() below.
//
// This example goes on to show how a modified wrapper can allow
// resizing of the underlying vsip::Fft objects if needed.  In between
// calls that change the size, these objects are reused.  The same
// performance advantages are realized by avoiding costly FFT planning
// operations unless they are absolutely needed.  See resizeable_filter().
//



// This class encapsulates two FFT objects used to implement a filter.  
// The input signal, filter coefficients and output buffer are passed in
// each time, but the FFT objects themselves persist across calls.  This
// provides a portable, encapsulated way to share these objects amongst
// different functions or classes, provided they are operating on input
// signals of the same length.
//
// This version allows the caller to provide the filter coefficients, 
// which should already be transformed into the frequency domain.
//
// Use this when the size is known at instantiation time and/or when
// dynamic memory allocation is not desired for whatever reason.
class Filter
{
  typedef Fft<Vector, cscalar_f, cscalar_f, fft_fwd> f_fft_type;
  typedef Fft<Vector, cscalar_f, cscalar_f, fft_inv> i_fft_type;

public:
  Filter(length_type signal_length)
  : size_(signal_length),
    f_fft_(Domain<1>(size_), 1.0f),
    i_fft_(Domain<1>(size_), 1.0f/size_)
    {}

  length_type size() { return size_; }

  // Run data through the pipeline.
  template <typename Block1, 
            typename Block2, 
            typename Block3>
  void 
  operator()(
    const_Vector<cscalar_f, Block1> in, 
    const_Vector<cscalar_f, Block2> coefficients,
    Vector<cscalar_f, Block3> out)
  {
    if((size() != in.size()) ||
       (size() != coefficients.size()) ||
       (size() != out.size()))
    {
      std::cout << "Error (class Filter): "
                << "all input and output views must be element-conformant" << std::endl;
      return;
    }

    out = i_fft_(coefficients * f_fft_(in));
  }

private:
  length_type size_;
  f_fft_type f_fft_;
  i_fft_type i_fft_;
};
  

// This more flexible wrapper is useful if the size of the operation is not
// known at initialization time.  An instance of the class is not created
// until the reconfigure() function is called.  Calling reconfigure() a second 
// time to adjust the size is permitted.  From a performance perspective 
// though, it is generally better to have two or more objects -- one of each
// needed size -- instead of resizing, due to the cost associated with planning
// the FFTs.  This cost can be an order of magnitude greater than the
// cost of filtering a dataset of a given size.
//
class Dynamic_filter
{
public:
  // Two-stage construction: call Dynamic_filter foo(); then foo.reconfigure(N);
  Dynamic_filter()
  {
    filter_.release();
  }

  void reconfigure(length_type signal_length)
  {
    filter_.reset(new Filter(signal_length));
  }

  // One-stage construction: call `Dynamic_filter foo(N)`.  This acts
  // almost identically as a call to `Filter foo(N)` would.
  Dynamic_filter(length_type signal_length)
  {
    reconfigure(signal_length);
  }

  // Run data through the pipeline.
  template <typename Block1, 
            typename Block2, 
            typename Block3>
  void 
  operator()(
    const_Vector<cscalar_f, Block1> in, 
    const_Vector<cscalar_f, Block2> coefficients,
    Vector<cscalar_f, Block3> out)
  {
    if (filter_.get() == 0)
    {
      std::cout << "Error (class Dynamic_filter): "
                << "reconfigure() not called prior to processing input data" << std::endl;
    }
    else if((filter_->size() != in.size()) ||
       (filter_->size() != coefficients.size()) ||
       (filter_->size() != out.size()))
    {
      std::cout << "Error (class Dynamic_filter): "
                << "all input and output views must be element-conformant" << std::endl;
    }
    else
      (*filter_)(in, coefficients, out);
  }

private:
  std::unique_ptr<Filter> filter_;
};
    



// Simple filter example: Uses static allocation
void
simple_filter(length_type const N)
{
  // Create an object given a fixed input signal length, known
  // at initialization time.  May be allocated as a static or
  // global and persist throughout the lifetime of the program.
  Filter filter(N);

  Vector<cscalar_f> in(N, cscalar_f(1.f));
  Vector<cscalar_f> k(N, cscalar_f(1.f));
  Vector<cscalar_f> out(N);

  filter(in, k, out);
}  


// Resizable filter example: Uses dynamic allocation
void
resizable_filter(length_type const N)
{
  // Create an object and defer allocation of the wrapped object
  // until reconfigure() is called, usually at a time after the system
  // is up and running and the size information has been obtained
  // via some sort of I/O (a file is read, or a message is received).
  Dynamic_filter filter;
  filter.reconfigure(N);

  Vector<cscalar_f> in(N, cscalar_f(1.f));
  Vector<cscalar_f> k(N, cscalar_f(1.f));
  Vector<cscalar_f> out(N);

  filter(in, k, out);

  // Change the input signal length and only operate on the first
  // half of the data, just to demonstrate how this is done.
  filter.reconfigure(N/2);

  Domain<1> n2(N/2);
  filter(in(n2), k(n2), out(n2));
}  


int
main(int argc, char **argv)
{
  vsipl init(argc, argv);

  length_type const N = 256;

  simple_filter(N);
  resizable_filter(N);
}


