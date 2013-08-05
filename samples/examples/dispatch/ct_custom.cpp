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

/// Description: 
///   Compile-time dispatch for a custom operation.

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip_csl/diagnostics.hpp>
#include <iostream>

namespace example
{
/// Define an operation tag for the new operation.
struct Op;

/// Define backend tags for all available backends.
struct A;
struct B;
struct C;
struct D;

// Define backends:

// B_impl is a backend for 'Op', available for type 'float'.
struct B_impl
{
  static void process(float *, size_t)
  { std::cout << "B_impl::process called" << std::endl;}
};

// C_impl is a backend for 'Op', whose availability is expressed
// via is_c_supported. Specifically, C_impl is available for complex types.

template <typename T> struct is_c_supported 
{ static bool const value = false;};
template <typename T> struct is_c_supported<vsip::complex<T> >
{ static bool const value = true;};

template <typename T>
struct C_impl
{
  static void process(T *, size_t)
  { std::cout << "C_impl::process called" << std::endl;}
};

// D_impl is a backend for 'Op', available for any type 'T'.
template <typename T>
struct D_impl
{
  static void process(T *, size_t)
  { std::cout << "D_impl::process called" << std::endl;}
};

} // namespace example

namespace vsip_csl
{
namespace dispatcher
{

// Define the list of backends to be tried during the dispatch
// of operation 'Op'.
//
// - 'A' does never match (there is no Evaluator specialization for it).
// - 'B' will match for type float.
// - 'C' will match depending on an is_c_supported predicate meta-function.
// - 'D' is the generic fallback.
template <>
struct List<example::Op>
{
  typedef Make_type_list<example::A,
			 example::B,
			 example::C,
			 example::D>::type type;
};

// This is a backend that is available for float only.
template <>
struct Evaluator<example::Op, example::B, float>
{
  static bool const ct_valid = true;
  typedef example::B_impl backend_type;
};

// This is a generic backend, whose availability is reported
// by example::is_c_supported. 
template <typename T>
struct Evaluator<example::Op, example::C, T>
{
  static bool const ct_valid = example::is_c_supported<T>::value;
  typedef example::C_impl<T> backend_type;
};

// This is a generic backend, acting as fallback, if everything else fails.
template <typename T>
struct Evaluator<example::Op, example::D, T>
{
  static bool const ct_valid = true;
  typedef example::D_impl<T> backend_type;
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace example
{
template <typename T>
struct Op_frontend
{
  // Select the appropriate backend for the frontend.
  typedef typename vsip_csl::dispatcher::Dispatcher<Op, T>::type backend_type;
};
} // namespace example

template <typename T>
void run_op(T *data, size_t size)
{
  typedef typename example::Op_frontend<T>::backend_type backend_type;
  std::cout << "backend for " << vsip_csl::type_name<float>() 
	    << " is " << vsip_csl::type_name<backend_type>() << std::endl;
  backend_type::process(data, size);
}

int main(int, char **)
{
  run_op<float>(0, 0);
  run_op<double>(0, 0);
  run_op<vsip::complex<float> >(0, 0);
}
