//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

// C++11 compatibility

#ifndef ovxx_cxx11_thread_hpp_
#define ovxx_cxx11_thread_hpp_

#include <memory>
#include <pthread.h>

namespace ovxx
{
namespace cxx11
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
  class id
  {
  public:
    id() : thread_() {}
    explicit id(pthread_t t) : thread_(t) {}

  private:
    friend class thread;

    pthread_t thread_;

    friend bool operator==(id x, id y)
    { return pthread_equal(x.thread_, y.thread_);}
    friend bool operator<(id x, id y) { return x.thread_ < y.thread_;}
    friend std::ostream &
    operator<<(std::ostream &os, id const &x)
    {
      if(x.thread_) return os << x.thread_;
      else return os << "Not-a-thread";
    }
  };


  template <typename Callable>
  thread(Callable c) : callable_(new callable_wrapper<Callable>(c))
  {
    if (pthread_create(&id_.thread_, 0, &thread::start, this) != 0)
      throw std::bad_alloc();
  }
  ~thread()
  {
    detach();
  }
  void join(void **status = 0)
  {
    void *s;
    pthread_join(id_.thread_, &s);
    if (status) *status = s;    
  }
  id get_id() const { return id_;}

private:
  thread(thread const &);
  thread &operator=(thread const &);

  void detach()
  { pthread_detach(id_.thread_);}

  static void *start(void *self)
  {
    static_cast<thread*>(self)->callable_->run();
    return 0;
  }

  std::auto_ptr<callable_base> callable_;
  id id_;
};

namespace this_thread
{

inline thread::id get_id() { return thread::id(pthread_self());}
inline void yield() { sched_yield();}

} // namespace ovxx::cxx11::this_thread
} // namespace ovxx::cxx11
} // namespace ovxx

#endif
