//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_copy_chain_hpp_
#define ovxx_parallel_copy_chain_hpp_

#include <iterator>
#include <vector>
#include <algorithm>
#include <cassert>

namespace ovxx
{
namespace parallel
{

class Copy_chain
{
  // Record for a single copy.
  struct Record
  {
    void*    start_;		// starting address
    size_t   elem_size_;	// size of each element (bytes)
    int      stride_;		// stride between successive elements
    unsigned length_;		// number of elements

    Record(void* start, size_t elem_size, int stride, unsigned length)
      : start_(start), elem_size_(elem_size), stride_(stride), length_(length)
    {}
  };

  typedef std::vector<Record>::const_iterator iterator;

public:
  Copy_chain()
    : data_size_ (0),
      chain_     () 
  {}

  // Add a new copy record.
  void add(void* start, size_t elem_size, int stride, unsigned length)
  {
    this->chain_.push_back(Record(start, elem_size, stride, length));
    this->data_size_ += elem_size * length;
  }

  // Append records from an existing chain.
  void append(Copy_chain chain)
  {
    std::copy(chain.chain_.begin(), chain.chain_.end(),
	      back_inserter(this->chain_));
    this->data_size_ += chain.data_size_;
  }

  // Append records from an existing chain, w/offset.
  void append_offset(void* offset, Copy_chain chain);

  // Copy data into buffer.
  void copy_into(void* dest, size_t size) const;

  // Copy data into memory specified by another chain.
  void copy_into(Copy_chain dst_chain) const;

  // Return number of record in chain.
  unsigned size() const
  { return this->chain_.size(); }

  // Return number of data bytes in chain.
  size_t data_size() const
  { return data_size_; }

private:
  size_t              data_size_;
  std::vector<Record> chain_;
};

} // namespace ovxx::parallel
} // namespace ovxx

#endif
