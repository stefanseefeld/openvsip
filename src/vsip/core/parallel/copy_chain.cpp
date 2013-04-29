//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <cstring>
#include <cassert>
#include <vsip/core/static_assert.hpp>
#include <vsip/core/parallel/copy_chain.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{

namespace impl
{

void inline
incr_ptr(void*& ptr, size_t offset)
{
  VSIP_IMPL_STATIC_ASSERT(sizeof(void*) == sizeof(size_t));
  ptr = reinterpret_cast<void*>(reinterpret_cast<size_t>(ptr) + offset);
}

inline void*
add_ptr(void* ptr1, void* ptr2)
{
  VSIP_IMPL_STATIC_ASSERT(sizeof(void*) == sizeof(size_t));
  return reinterpret_cast<void*>(reinterpret_cast<size_t>(ptr1) +
				 reinterpret_cast<size_t>(ptr2));
}


  // Append records from an existing chain, w/offset.
void
Copy_chain::append_offset(void* offset, Copy_chain chain)
{
  rec_iterator cur = chain.chain_.begin();
  rec_iterator end = chain.chain_.end();

  for (; cur != end; ++cur)
  {
    this->chain_.push_back(Record(add_ptr(cur->start_, offset),
				  cur->elem_size_, 
				  cur->stride_, 
				  cur->length_));
  }
  this->data_size_ += chain.data_size_;
}



// Requires:
//   SIZE to be number of bytes to into buffer dest, which must be
//     equal to chain's data_size();
void
Copy_chain::copy_into(void* dest, size_t size) const
{
  rec_iterator cur = chain_.begin();
  rec_iterator end = chain_.end();

  assert(size == this->data_size());
  
  for (; cur != end; ++cur)
  {
    void* src        = (*cur).start_;
    size_t elem_size = (*cur).elem_size_;

    if ((*cur).stride_ == 1)
    {
      assert(size >= (*cur).length_ * elem_size);
      memcpy(dest, src, (*cur).length_ * elem_size);
      incr_ptr(dest, (*cur).length_ * elem_size);
      size -= (*cur).length_ * elem_size;
    }
    else
    {
	for (unsigned i=0; i<(*cur).length_; ++i)
	{
	  assert(size >= elem_size);
	  memcpy(dest, src, elem_size);
	  incr_ptr(dest, elem_size);
	  incr_ptr(src,  (*cur).stride_ * elem_size);
	  size -= elem_size;
	}
    }
  }

  assert(size == 0);
}



// Requires:
//   CHAIN to be copy chain with the same number of bytes and same
//     element sizes as this chain.
//   In addition, Source and destination must have same element size.
void
Copy_chain::copy_into(Copy_chain dst_chain) const
{
  rec_iterator src     = this->chain_.begin();
  rec_iterator src_end = this->chain_.end();
  
  rec_iterator dst     = dst_chain.chain_.begin();
  rec_iterator dst_end = dst_chain.chain_.end();
  
  void*  src_ptr    = (*src).start_;
  size_t src_i      = 0;
  void*  dst_ptr    = (*dst).start_;
  size_t dst_i      = 0;
  
  size_t elem_size  = (*src).elem_size_;
  assert((*dst).elem_size_ == elem_size);
  
  while (src != src_end && dst != dst_end)
  {
    unsigned elements;
    
    if ((*src).stride_ == 1 && (*dst).stride_ == 1)
    {
      elements = std::min((*src).length_ - src_i,
			  (*dst).length_ - dst_i);
    }
    else
      elements = 1;
    
    memcpy(dst_ptr, src_ptr, elements * elem_size);
    
    incr_ptr(dst_ptr, elements * elem_size * (*dst).stride_);
    incr_ptr(src_ptr, elements * elem_size * (*src).stride_);
    dst_i += elements;
    src_i += elements;
    
    if (src_i == (*src).length_ && ++src != src_end)
    {
      assert((*src).elem_size_ == elem_size);
      src_ptr    = (*src).start_;
      src_i      = 0;
    }
    
    if (dst_i == (*dst).length_ && ++dst != dst_end)
    {
      assert((*dst).elem_size_ == elem_size);
      dst_ptr    = (*dst).start_;
      dst_i      = 0;
    }
  }

  // If both chains have same data_size, then both should be exhausted.
  assert(src == src_end && dst == dst_end);
}

} // namespace vsip::impl
} // namespace vsip
