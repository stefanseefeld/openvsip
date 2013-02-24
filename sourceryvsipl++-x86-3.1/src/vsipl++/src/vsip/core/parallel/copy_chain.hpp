/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/parallel/copy_chain.hpp
    @author  Jules Bergmann
    @date    2005-07-29
    @brief   VSIPL++ Library: Pseudo-DMA Chain for par-services-none.

*/

#ifndef VSIP_IMPL_COPY_CHAIN_HPP
#define VSIP_IMPL_COPY_CHAIN_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <iterator>
#include <vector>
#include <algorithm>
#include <cassert>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
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

  typedef std::vector<Record>      rec_list;
  typedef rec_list::const_iterator rec_iterator;

  // Constructors.
public:
  Copy_chain()
    : data_size_ (0),
      chain_     () 
  {}

  // Accessors.
public:

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

  // Member data.
private:
  size_t              data_size_;
  std::vector<Record> chain_;
};


} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_COPY_CHAIN_HPP
