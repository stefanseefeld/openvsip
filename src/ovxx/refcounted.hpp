//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_refcounted_hpp_
#define ovxx_refcounted_hpp_

#include <ovxx/config.hpp>
#include <ovxx/detail/noncopyable.hpp>
#ifdef __GXX_EXPERIMENTAL_CXX0X__
# include <atomic>
#endif
#include <cassert>

namespace ovxx
{

template <typename T>
class refcounted : detail::noncopyable
{
public:
  refcounted() : count_(1) {}

  void increment_count() const
  {
#ifdef __GXX_EXPERIMENTAL_CXX0X__
    std::atomic_fetch_add_explicit(&count_, 1u, 
				   std::memory_order_relaxed);
#elif __GNUC__
  __sync_fetch_and_add(&count_, 1);
#else
  ++count_;
#endif
  }
  void decrement_count() const
  {
#ifdef __GXX_EXPERIMENTAL_CXX0X__
    if (std::atomic_fetch_sub_explicit(&count_, 1u, 
				       std::memory_order_release) == 1) 
    {
      std::atomic_thread_fence(std::memory_order_acquire);
      delete static_cast<T*>(const_cast<refcounted*>(this));
    }
#elif __GNUC__
    if (!__sync_sub_and_fetch(&count_, 1))
      delete static_cast<T*>(const_cast<refcounted*>(this));
#else
    if (!--count_)
      delete static_cast<T*>(const_cast<refcounted*>(this));
#endif
  }

private:
#ifdef __GXX_EXPERIMENTAL_CXX0X__
  mutable std::atomic<unsigned int> count_;
#else
  mutable unsigned int count_;
#endif
};

template <typename T>
class refcounted_ptr
{
public:
  refcounted_ptr() : ptr_(0) {}
  refcounted_ptr(T *p, bool add_ref = true) : ptr_(p) 
  { if (add_ref && ptr_) ptr_->increment_count();}
  refcounted_ptr(refcounted_ptr const &p) : ptr_(p.ptr_)
  { if (ptr_) ptr_->increment_count();}

  ~refcounted_ptr() { if (ptr_) ptr_->decrement_count();}

  refcounted_ptr&
  operator=(refcounted_ptr const &from)
  {
    if (ptr_ != from.ptr_)
    {
      if (ptr_) ptr_->decrement_count();
      ptr_ = from.ptr_;
      ptr_->increment_count();
    }
    return *this;
  }

  void reset(T *p, bool increment = false)
  {
    if (ptr_) ptr_->decrement_count();
    ptr_ = p;
    OVXX_PRECONDITION(ptr_ || !increment);
    if (increment) ptr_->increment_count();
  }

  T &operator*() const { return *ptr_;}
  T *operator->() const { return ptr_;}
  T *get() const { return ptr_;}

private:
  T *ptr_;
};

} // namespace ovxx

#endif
