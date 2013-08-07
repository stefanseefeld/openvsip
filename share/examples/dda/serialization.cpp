/* Copyright (c) 2010, 2011 CodeSourcery, Inc.  All rights reserved. */

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

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/serialization.hpp>
#include <ovxx/type_name.hpp>
#include <algorithm>
#include <iostream>

using namespace vsip;
using ovxx::type_name;
namespace s = serialization;

namespace example
{
// Define a custom value-type...
struct Pixel 
{
  char r, g, b;
};
}

namespace vsip
{
namespace serialization
{
// ...and provide a serialization encoding for it.
//
// The "value" field is a unique numeric indicator for the data type.  Numbers
// below 0x10000 are reserved for implementation-defined serialization types,
// so we may pick any number equal to or greater than 0x10000 for our type.
template <>
struct type_info<example::Pixel> { static s::uint64_type const value = 0x10000;};
} // namespace vsip::serialization
} // namespace vsip

template <typename T>
T *clone_data(T const *data, length_type size)
{
  T *clone = new T[size];
  std::copy(data, data + size, clone);
  return clone;
}

template <typename T>
void delete_data(T *data) { delete [] data;}

template <typename T>
std::pair<T*,T*> clone_data(std::pair<T*,T*> data, length_type size)
{
  std::pair<T*,T*> clone;
  clone.first = new T[size];
  clone.second = new T[size];
  std::copy(data.first, data.first + size, clone.first);
  std::copy(data.second, data.second + size, clone.second);
  return clone;
}

template <typename T>
std::pair<T*,T*> clone_data(std::pair<T const *,T const *> data, length_type size)
{
  return clone_data(std::pair<T*,T*>(data.first, data.second), size);
}

template <typename T>
void delete_data(std::pair<T*,T*> data) 
{
  delete [] data.first;
  delete [] data.second;
}

// Receive marshaled data together with a descriptor.
// Wrap it in an appropriate user-storage block and process
// it further.
template <typename T>
void process_remote(T const *data, s::Descriptor const &info)
{
  // Construct a user-storage block,...
  Dense<2, T> block(Domain<2>(0, 0), static_cast<T*>(0));
  Matrix<T> matrix(block);
  // ...(making sure it is compatible with the incoming data)...
  if (!s::is_compatible<Dense<2, T> >(info)) 
    throw std::runtime_error("input data type or format not supported");
  // ...and assign the marshaled data to it.
  block.rebind(const_cast<T*>(data), Domain<2>(info.size[0], info.size[1]));
  block.admit();
  std::cout << "processing marshaled " << type_name<T>() << " data" << std::endl;
}

// Receive marshaled split-complex data together with a descriptor.
// Wrap it in an appropriate user-storage block and process
// it further.
void process_remote(std::pair<float*,float*> data, s::Descriptor const &info)
{
  // Construct a user-storage block,...
  Dense<2, complex<float> > block(Domain<2>(0, 0), static_cast<float*>(0));
  Matrix<complex<float> > matrix(block);
  // ...(making sure it is compatible with the incoming data)...
  if (!s::is_compatible<Dense<2, complex<float> > >(info)) 
    throw std::runtime_error("input data type or format not supported");
  // ...and assign the marshaled data to it.
  block.rebind(data, Domain<2>(info.size[0], info.size[1]));
  block.admit();
  std::cout << "processing marshaled split-complex data" << std::endl;
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  {
    // Create a 7x9 matrix,...
    Matrix<float> input(7, 9);
    // ...obtain a raw pointer to the data,...
    dda::Data<Dense<2, float>, dda::in> data(input.block());
    // ...as well as some metadata,...
    s::Descriptor info;
    s::describe_data(data, info);
    // ...simulate a network transfer,...
    dda::Data<Dense<2, float>, dda::inout>::ptr_type clone = 
      clone_data(data.ptr(), info.storage_size);
    // ...and process it "remotely".
    process_remote(clone, info);
    delete_data(clone);
  }

  {
    // Create a 7x9 matrix with a custom value-type,...
    Matrix<example::Pixel> input(7, 9);
    // ...obtain a raw pointer to the data,...
    dda::Data<Dense<2, example::Pixel>, dda::in> data(input.block());
    // ...as well as some metadata,...
    s::Descriptor info;
    s::describe_data(data, info);
    // ...simulate a network transfer,...
    dda::Data<Dense<2, example::Pixel>, dda::inout>::ptr_type clone = 
      clone_data(data.ptr(), info.storage_size);
    // ...and process it "remotely".
    process_remote(clone, info);
    delete_data(clone);
  }

  {
    // Create user storage,...
    float *user_storage = new float[2 * 7 * 9];
    // ...and bind it to a block & view.
    Dense<2, complex<float> > block(Domain<2>(7, 9), user_storage, user_storage + 7 * 9);
    block.admit();
    Matrix<complex<float> > input(block);    
    // Obtain a raw pointer to the storage,...
    std::pair<float*, float*> data;
    block.release(true, data);
    // ...as well as some metadata,...
    s::Descriptor info;
    s::describe_user_storage(input.block(), info);
    // ...simulate a network transfer,...
    std::pair<float*,float*> clone = clone_data(data, info.storage_size);
    // ...and process it "remotely".
    process_remote(data, info);
    delete_data(clone);

    // Make the metadata invalid...
    info.dimensions = 3;
    // ...and catch the exception as the peer component can't process such data.
    try { process_remote(data, info);}
    catch (std::runtime_error const &e) { std::cout << e.what() << " (expected)" << std::endl;}

    delete [] user_storage;
  }
}
