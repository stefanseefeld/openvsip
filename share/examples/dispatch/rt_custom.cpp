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
///   Runtime dispatch for a custom operation.

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <ovxx/dispatch.hpp>
#include <ovxx/type_name.hpp>
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

// Support float data of arbitrary size
void b_process(float *, size_t)
{ std::cout << "b_process called" << std::endl;}

// Support arbitrary types, but only power-of-two sizes.
template <typename T>
void c_process(T *, size_t)
{ std::cout << "c_process called" << std::endl;}

// Support arbitrary types, with arbitrary sizes.
template <typename T>
void d_process(T *, size_t)
{ std::cout << "d_process called" << std::endl;}

// Helper function
bool is_power_of_two(unsigned size)
{
  return (size & (size - 1)) == 0;
}

} // namespace example

namespace ovxx
{
namespace dispatcher
{

// Define the list of backends to be tried during the dispatch
// of operation 'Op'.
//
// - 'A' does never match (there is no Evaluator specialization for it).
// - 'B' will match for type float.
// - 'C' will match for all types, but only for sizes that are a power of two.
// - 'D' is the generic fallback.
template <>
struct List<example::Op>
{
  typedef make_type_list<example::A,
			 example::B,
			 example::C,
			 example::D>::type type;
};

// This is a backend that is available for float only.
template <>
struct Evaluator<example::Op, example::B, void(float*,size_t)>
{
  static char const *name() { return "B";}
  static bool const ct_valid = true;
  static bool rt_valid(float*, size_t) {return true;}
  static void exec(float *data, size_t size)
  { example::b_process(data, size);}
};

// This is a backend that is available for input sizes that are a power of two.
template <typename T>
struct Evaluator<example::Op, example::C, void(T*,size_t)>
{
  static char const *name() { return "C";}
  static bool const ct_valid = true;
  static bool rt_valid(T*, size_t size) {return example::is_power_of_two(size);}
  static void exec(T *data, size_t size)
  { example::c_process(data, size);}
};

// This is a generic backend, acting as fallback, if everything else fails.
template <typename T>
struct Evaluator<example::Op, example::D, void(T*,size_t)>
{
  static char const *name() { return "D";}
  static bool const ct_valid = true;
  static bool rt_valid(T*, size_t) {return true;}
  static void exec(T *data, size_t size)
  { example::d_process(data, size);}
};

} // namespace ovxx::dispatcher
} // namespace ovxx

namespace example
{
template <typename T>
void op_frontend(T *data, size_t size)
{
  std::cout << "dispatch op_frontend<" << ovxx::type_name<T>() << "> :\n";
  // TODO: reimplement
  // std::cout << "dispatch trace :\n";
  // ovxx::dispatch_diagnostics<Op, void>(data, size);
  std::cout << "dispatch :\n";
  ovxx::dispatch<Op,void>(data, size);
  std::cout << std::endl;
};
} // namespace example

int main(int argc, char **argv)
{
  vsip::vsipl library(argc, argv);

  example::op_frontend<float>(0, 1);
  example::op_frontend<double>(0, 8);
  example::op_frontend<std::complex<float> >(0, 5);
}
