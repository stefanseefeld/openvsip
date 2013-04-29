//
// Copyright (c) 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

// C++11 compatibility

#ifndef oxx_cxx11_shared_ptr_hpp_
#define oxx_cxx11_shared_ptr_hpp_

#include <ovxx/config.hpp>
#include <algorithm> // for std::swap
#include <cassert>
#include <memory>

namespace ovxx
{
namespace cxx11
{
#ifndef __GXX_EXPERIMENTAL_CXX0X__

template<typename T>
class shared_ptr
{
  class counter
  {
  public:
    counter() : count_(1) {}
    virtual ~counter() {}
    void increment()
    {
      // Only make this thread-safe if specifically asked to
#if OVXX_ENABLE_THREADING && __GNUC__
      __sync_fetch_and_add(&count_, 1);
#else
      ++count_;
#endif
    }
    void decrement()
    {
      // Only make this thread-safe if specifically asked to
#if OVXX_ENABLE_THREADING && __GNUC__
      __sync_fetch_and_sub(&count_, 1);
#else
      --count_;
#endif
      if (!count_) dispose();
    }

  private:
    counter(counter const &);
    counter &operator=(counter const &);
    virtual void dispose() = 0;

    unsigned int count_;
  };

  template <typename T1>
  class counted_ptr : public counter
  {
  public:
    counted_ptr(T1 *p) : ptr_(p) {}
    virtual void dispose() { delete ptr_;}

  protected:
    T1 *ptr_;
  };

  template <typename T1, typename D>
  class counted_deleter : public counted_ptr<T1>
  {
  public:
    counted_deleter(T1 *p, D d) : counted_ptr<T1>(p), deleter_(d) {}
    virtual void dispose() { deleter_(this->ptr_);}

  private:
    D deleter_;
  };

public:
  shared_ptr() : ptr_(0), counter_(0) {}

  template<typename T1>
  explicit
  shared_ptr(T1 *p) : ptr_(p), counter_(0)
  {
#if OVXX_HAS_EXCEPTIONS
    try 
    {
      counter_ = new counted_ptr<T1>(p);
    }
    catch(...)
    {
      delete p;
      throw;
    }
#else
    counter_ = new counted_ptr<T1>(p);
#endif
  }

  template<typename T1, typename Deleter>
  shared_ptr(T1 *p, Deleter d) : ptr_(p), counter_(0)
  {
#if OVXX_HAS_EXCEPTIONS
    try
    {
      counter_ = new counted_deleter<T1, Deleter> (p, d);
    }
    catch(...)
    {
      d(p);
      throw;
    }
#else
    counter_ = new counted_ptr<T1>(p);
#endif
  }

  shared_ptr(shared_ptr const &r) : ptr_(r.ptr_), counter_(r.counter_)
  {
    if (counter_) counter_->increment();
  }
  template<typename T1>
  shared_ptr(shared_ptr<T1> const &r) : ptr_(r.ptr_), counter_(r.counter_)
  {
    if (counter_) counter_->increment();
  }
  ~shared_ptr() { if (counter_) counter_->decrement();}

  shared_ptr&
  operator=(shared_ptr const &r)
  {
    shared_ptr(r).swap(*this);
    return *this;
  }
  template<typename T1>
  shared_ptr&
  operator=(shared_ptr<T1> const &r)
  {
    shared_ptr(r).swap(*this);
    return *this;
  }

  void swap(shared_ptr<T> & other)
  {
    std::swap(ptr_, other.ptr_);
    std::swap(counter_, other.counter_);
  }

  void reset() { shared_ptr().swap(*this);}
  template <typename T1>
  void reset(T1 *p) 
  {
    assert(!p || p != ptr_);
    shared_ptr(p).swap(*this);
  }
  template <typename T1, typename D>
  void reset(T1 *p, D d) 
  {
    shared_ptr(p, d).swap(*this);
  }

  T & operator* () const
  {
    assert(ptr_);
    return *ptr_;
  }

  T * operator-> () const
  {
    assert(ptr_);
    return ptr_;
  }

  T * get() const { return ptr_;}


private:
  T *ptr_;
  counter *counter_;
};
#else
using std::shared_ptr;
#endif

} // namespace ovxx::c++11
} // namespace ovxx

#endif
