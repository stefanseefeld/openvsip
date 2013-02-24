/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description: 
///   Runtime dispatch for a custom operation.

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

namespace vsip_csl
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
  typedef Make_type_list<example::A,
			 example::B,
			 example::C,
			 example::D>::type type;
};

// This is a backend that is available for float only.
template <>
struct Evaluator<example::Op, example::B, void(float*,size_t)>
{
  static bool const ct_valid = true;
  static bool rt_valid(float*, size_t) {return true;}
  static void exec(float *data, size_t size)
  { example::b_process(data, size);}
};

// This is a backend that is available for input sizes that are a power of two.
template <typename T>
struct Evaluator<example::Op, example::C, void(T*,size_t)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T*, size_t size) {return example::is_power_of_two(size);}
  static void exec(T *data, size_t size)
  { example::c_process(data, size);}
};

// This is a generic backend, acting as fallback, if everything else fails.
template <typename T>
struct Evaluator<example::Op, example::D, void(T*,size_t)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T*, size_t) {return true;}
  static void exec(T *data, size_t size)
  { example::d_process(data, size);}
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace example
{
template <typename T>
void op_frontend(T *data, size_t size)
{
  vsip_csl::dispatch<Op,void>(data, size);
};
} // namespace example

int main(int, char **)
{
  example::op_frontend<float>(0, 1);
  example::op_frontend<double>(0, 8);
  example::op_frontend<std::complex<float> >(0, 5);
}
