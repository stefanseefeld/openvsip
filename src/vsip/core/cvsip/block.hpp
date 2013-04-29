/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/cvsip/block.hpp
    @author  Stefan Seefeld
    @date    2006-10-17
    @brief   VSIPL++ Library: Bindings to create a C-VSIPL block from
             VSIPL++ blocks.
*/

#ifndef VSIP_CORE_CVSIP_BLOCK_HPP
#define VSIP_CORE_CVSIP_BLOCK_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/config.hpp>
#include <vsip/support.hpp>
#include <vsip/domain.hpp>
extern "C" {
#include <vsip.h>
}
/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace cvsip
{

template <typename T> struct Block_traits;

template <>
struct Block_traits<bool>
{
  typedef bool value_type;
  typedef vsip_block_bl block_type;

  static block_type* create(size_t s)
  {
    block_type* b = vsip_blockcreate_bl(s, VSIP_MEM_NONE);
    if (!b) VSIP_THROW(std::bad_alloc());
    return b;
  }
  static void destroy(block_type* b)
  { vsip_blockdestroy_bl(b);}

  // C-VSIPL and C++ do not agree on the bool type, so we're forced
  // to manually copy data.  Hence, we don't need bind, admit, or release.
};

template <>
struct Block_traits<int>
{
  typedef int value_type;
  typedef vsip_block_i block_type;

  static block_type* create(size_t s)
  {
    block_type* b = vsip_blockcreate_i(s, VSIP_MEM_NONE);
    if (!b) VSIP_THROW(std::bad_alloc());
    return b;
  }
  static void destroy(block_type* b)
  { vsip_blockdestroy_i(b);}
  static block_type* bind(value_type* d, size_t s)
  {
    block_type* b = vsip_blockbind_i(d, s, VSIP_MEM_NONE);
    if (!b) VSIP_THROW(std::bad_alloc());
    return b;
  }
  static void rebind(block_type* b, value_type* d)
  { (void) vsip_blockrebind_i(b, d);}
  static void admit(block_type* b, vsip_scalar_bl s)
  { (void) vsip_blockadmit_i(b, s);}
  static void release(block_type* b, vsip_scalar_bl s)
  { (void) vsip_blockrelease_i(b, s);}  
};

#if VSIP_IMPL_CVSIP_HAVE_FLOAT
template <>
struct Block_traits<float>
{
  typedef float value_type;
  typedef vsip_block_f block_type;

  static block_type *create(size_t s)
  {
    block_type *b= vsip_blockcreate_f(s, VSIP_MEM_NONE);
    if (!b) VSIP_THROW(std::bad_alloc());
    return b;
  }
  static void destroy(block_type *b)
  { vsip_blockdestroy_f(b);}
  static block_type *bind(value_type *d, size_t s)
  {
    block_type *b = vsip_blockbind_f(d, s, VSIP_MEM_NONE);
    if (!b) VSIP_THROW(std::bad_alloc());
    return b;
  }
  static void rebind(block_type *b, value_type *d)
  { (void) vsip_blockrebind_f(b, d);}
  static void admit(block_type *b, vsip_scalar_bl s)
  { (void) vsip_blockadmit_f(b, s);}
  static void release(block_type *b, vsip_scalar_bl s)
  { (void) vsip_blockrelease_f(b, s);}  
};

template <>
struct Block_traits<std::complex<float> >
{
  typedef float value_type;
  typedef vsip_cblock_f block_type;

  static block_type *create(size_t s)
  {
    block_type *b= vsip_cblockcreate_f(s, VSIP_MEM_NONE);
    if (!b) VSIP_THROW(std::bad_alloc());
    return b;
  }
  static void destroy(block_type *b)
  { vsip_cblockdestroy_f(b);}
  static block_type *bind(std::complex<value_type> *d, size_t s)
  {
    block_type *b = vsip_cblockbind_f(reinterpret_cast<value_type *>(d),
                                      0, s, VSIP_MEM_NONE);
    if (!b) VSIP_THROW(std::bad_alloc());
    return b;
  }
  static block_type *bind(std::pair<value_type *, value_type *> d, size_t s)
  {
    block_type *b = vsip_cblockbind_f(d.first, d.second, s, VSIP_MEM_NONE);
    if (!b) VSIP_THROW(std::bad_alloc());
    return b;
  }
  static void rebind(block_type *b, std::complex<value_type> *d)
  {
    value_type *old_r, *old_i;
    return vsip_cblockrebind_f(b, reinterpret_cast<value_type *>(d), 0,
                               &old_r, &old_i);
  }
  static void rebind(block_type *b, std::pair<value_type *, value_type *> d)
  {
    value_type *old_r, *old_i;
    return vsip_cblockrebind_f(b, d.first, d.second, &old_r, &old_i);
  }
  static void admit(block_type *b, vsip_scalar_bl s)
  { (void) vsip_cblockadmit_f(b, s);}
  static void release(block_type *b, vsip_scalar_bl s)
  {
    value_type *r, *i;
    vsip_cblockrelease_f(b, s, &r, &i);
  }
};

#endif
#if VSIP_IMPL_CVSIP_HAVE_DOUBLE

template <>
struct Block_traits<double>
{
  typedef double value_type;
  typedef vsip_block_d block_type;

  static block_type *create(size_t s)
  {
    block_type *b = vsip_blockcreate_d(s, VSIP_MEM_NONE);
    if (!b) VSIP_THROW(std::bad_alloc());
    return b;
  }
  static void destroy(block_type *b)
  { vsip_blockdestroy_d(b);}
  static block_type *bind(value_type *d, size_t s)
  {
    block_type *b = vsip_blockbind_d(d, s, VSIP_MEM_NONE);
    if (!b) VSIP_THROW(std::bad_alloc());
    return b;
  }
  static void rebind(block_type *b, value_type *d)
  { (void) vsip_blockrebind_d(b, d);}
  static void admit(block_type *b, vsip_scalar_bl s)
  { (void) vsip_blockadmit_d(b, s);}
  static void release(block_type *b, vsip_scalar_bl s)
  { (void) vsip_blockrelease_d(b, s);}  
};

template <>
struct Block_traits<std::complex<double> >
{
  typedef double value_type;
  typedef vsip_cblock_d block_type;

  static block_type *create(size_t s)
  {
    block_type *b = vsip_cblockcreate_d(s, VSIP_MEM_NONE);
    if (!b) VSIP_THROW(std::bad_alloc());
    return b;
  }
  static void destroy(block_type *b)
  { vsip_cblockdestroy_d(b);}
  static block_type *bind(std::complex<value_type> *d, size_t s)
  {
    block_type *b = vsip_cblockbind_d(reinterpret_cast<value_type *>(d),
                                      0, s, VSIP_MEM_NONE);
    if (!b) VSIP_THROW(std::bad_alloc());
    return b;
  }
  static block_type *bind(std::pair<value_type *, value_type *> d, size_t s)
  {
    block_type *b = vsip_cblockbind_d(d.first, d.second, s, VSIP_MEM_NONE);
    if (!b) VSIP_THROW(std::bad_alloc());
    return b;
  }
  static void rebind(block_type *b, std::complex<value_type> *d)
  {
    value_type *old_r, *old_i;
    return vsip_cblockrebind_d(b, reinterpret_cast<value_type *>(d), 0,
                               &old_r, &old_i);
  }
  static void rebind(block_type *b, std::pair<value_type *, value_type *> d)
  {
    value_type *old_r, *old_i;
    return vsip_cblockrebind_d(b, d.first, d.second, &old_r, &old_i);
  }
  static void admit(block_type *b, vsip_scalar_bl s)
  { (void) vsip_cblockadmit_d(b, s);}
  static void release(block_type *b, vsip_scalar_bl s)
  {
    value_type *r, *i;
    vsip_cblockrelease_d(b, s, &r, &i);
  }
};

#endif

// Wrapper class for C-VSIPL block with C-VSIPL allocated storage.
template <typename T>
class Block
{
public:
  typedef Block_traits<T> traits;

  Block(length_type size) VSIP_THROW((std::bad_alloc))
    : impl_(traits::create(size)), owner_(true) {}
  Block(typename traits::block_type *b) : impl_(b), owner_(false) {}
  Block(Block<T> const &b)
    : impl_(b.impl_), owner_(b.owner_) { b.owner_ = false;}
  ~Block() { if (owner_) traits::destroy(impl_);}
  typename traits::block_type *ptr() { return impl_;}
  void detach() { owner_ = false;}
private:
  typename traits::block_type *impl_;
  mutable bool owner_;
};

// Wrapper class for C-VSIPL block with user allocated storage.
template <typename T> 
class Us_block
{
public:
  typedef Block_traits<T> traits;
  
  Us_block(T *data, length_type size) VSIP_THROW((std::bad_alloc))
    : impl_(traits::bind(data, size)), owner_(true) {}
  Us_block(typename traits::block_type *b) : impl_(b), owner_(false) {}
  Us_block(Us_block<T> const &b)
    : impl_(b.impl_), owner_(b.owner_) { b.owner_ = false;}
  ~Us_block() { if (owner_) traits::destroy(impl_);}
  void rebind(T *data) { traits::rebind(impl_, data);}
  void admit(bool sync) { traits::admit(impl_, sync);}
  void release(bool sync) { traits::release(impl_, sync);}
  typename traits::block_type *ptr() { return impl_;}
  void detach() { owner_ = false;}
private:
  typename traits::block_type *impl_;
  mutable bool owner_;
};

template <typename T>
class Us_block<std::complex<T> >
{
public:
  typedef Block_traits<std::complex<T> > traits;
  
  Us_block(std::complex<T> *data, length_type size) VSIP_THROW((std::bad_alloc))
    : impl_(traits::bind(data, size)), owner_(true)
  {}
  Us_block(typename traits::block_type *b) : impl_(b), owner_(false) {}
  Us_block(std::pair<T *, T *> data, length_type size) VSIP_THROW((std::bad_alloc))
    : impl_(traits::bind(data, size)), owner_(true)
  {}
  Us_block(Us_block<std::complex<T> > const &b)
    : impl_(b.impl_), owner_(b.owner_) { b.owner_ = false;}
  ~Us_block() { if (owner_) traits::destroy(impl_);}
  void rebind(std::complex<T> *data) { traits::rebind(impl_, data);}
  void rebind(std::pair<T *, T *> data) { traits::rebind(impl_, data);}
  void admit(bool sync) { traits::admit(impl_, sync);}
  void release(bool sync) { traits::release(impl_, sync);}
  typename traits::block_type *ptr() { return impl_;}
  void detach() { owner_ = false;}
private:
  typename traits::block_type *impl_;
  mutable bool owner_;
};

} // namespace vsip::impl::cvsip
} // namespace vsip::impl
} // namespace vsip

#endif
