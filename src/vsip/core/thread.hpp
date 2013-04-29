/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Provide a c++0x-compatible thread API

#ifndef vsip_core_thread_hpp_
#define vsip_core_thread_hpp_

#include <memory>
#include <pthread.h>

namespace vsip
{
namespace impl
{
class mutex
{
public:
  mutex()
  {
    if (pthread_mutex_init(&mutex_, 0))
      throw std::bad_alloc();
  }
  ~mutex() { pthread_mutex_destroy(&mutex_);}
  void lock() { pthread_mutex_lock(&mutex_);}
  void unlock() { pthread_mutex_unlock(&mutex_);}


private:
  pthread_mutex_t mutex_;
};

template <typename L>
class lock_guard
{
public:
  lock_guard(L &l) : l_(l) { l_.lock();}
  ~lock_guard() { l_.unlock();}

private:
  lock_guard(lock_guard const &);
  lock_guard& operator=(lock_guard const &);

  L &l_;
};

class thread
{
  struct callable_base
  {
    virtual ~callable_base() {}
    virtual void run() = 0;
  };
  template <typename Callable>
  struct callable_wrapper : callable_base
  {
    callable_wrapper(Callable f) : func(f) {}
    void run() { func();}
    Callable func;
  };

public:
  template <typename Callable>
  thread(Callable c) : callable_(new callable_wrapper<Callable>(c))
  {
    if (pthread_create(&thread_, 0, &thread::start, this) != 0)
      throw std::bad_alloc();
  }
  ~thread()
  {
    detach();
  }
  void join(void **status = 0)
  {
    void *s;
    pthread_join(thread_, &s);
    if (status) *status = s;    
  }
private:
  thread(thread const &);
  thread &operator=(thread const &);

  void detach()
  { pthread_detach(thread_);}

  static void *start(void *self)
  {
    static_cast<thread*>(self)->callable_->run();
    return 0;
  }

  std::auto_ptr<callable_base> callable_;
  pthread_t  thread_;
};

} // namespace vsip::impl
} // namespace vsip

#endif // vsip_core_thread_hpp_
