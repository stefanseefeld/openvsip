/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   C++0x bits that we can't use from libstdc++ just yet.

#ifndef vsip_core_shared_ptr_hpp_
#define vsip_core_shared_ptr_hpp_

#include <vsip/core/config.hpp>
#include <algorithm> // for std::swap

namespace vsip
{
namespace impl
{
namespace cxx0x
{
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
#if VSIP_IMPL_ENABLE_THREADING && __GNUC__
      __sync_fetch_and_add(&count_, 1);
#else
      ++count_;
#endif
    }
    void decrement()
    {
      // Only make this thread-safe if specifically asked to
#if VSIP_IMPL_ENABLE_THREADING && __GNUC__
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
#if VSIP_HAS_EXCEPTIONS
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
#if VSIP_HAS_EXCEPTIONS
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


} // namespace vsip::impl::cxx0x
} // namespace vsip::impl
} // namespace vsip

#endif
