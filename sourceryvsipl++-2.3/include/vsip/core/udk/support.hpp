/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef vsip_core_udk_support_hpp_
#define vsip_core_udk_support_hpp_

#include <vsip/core/udk/tuple.hpp>

namespace vsip_csl
{
namespace udk
{
/// Marker for input arguments.
template <typename T> struct in;
/// Marker for output arguments.
template <typename T> struct out;
/// Marker for input / output arguments.
template <typename T> struct inout;

namespace target
{
/// Tag for testing purposes.
struct test;
/// Tag for CUDA tasks.
struct cuda;
}

namespace impl
{
template <typename Target, typename Args> struct Policy;
}

/// A Task...
/// Template parameters:
///   :Target: A target tag
///   :Args: An argument tuple
template <typename Target, typename Args>
struct Task;

/// One-argument Task
template <typename T, typename A>
struct Task<T, tuple<A> >
{
  typedef impl::Policy<T, tuple<A> > policy;
  typedef typename policy::function_type function_type;

public:
  Task(function_type const &f) : f_(f) {}

  void execute(typename policy::template arg<0>::type a)
  {
    start(a);
    wait();
  }

  void start(typename policy::template arg<0>::type a)
  {
    policy::start();
    typedef typename policy::template channel<0>::type c;
    typename c::type p = c::create(a);
    f_(*p);
  }
  void wait() { policy::wait();}

private:
  function_type f_;
};

/// Two-arguments task.
template <typename T, typename A0, typename A1>
struct Task<T, tuple<A0, A1> >
{
  typedef impl::Policy<T, tuple<A0, A1> > policy;
  typedef typename policy::function_type function_type;

public:
  Task(function_type const &f) : f_(f) {}

  void execute(typename policy::template arg<0>::type a0,
	       typename policy::template arg<1>::type a1)
  {
    start(a0, a1);
    wait();
  }

  void start(typename policy::template arg<0>::type a0,
	     typename policy::template arg<1>::type a1)
  {
    policy::start();
    typedef typename policy::template channel<0>::type c0;
    typedef typename policy::template channel<1>::type c1;
    typename c0::type p0 = c0::create(a0);
    typename c1::type p1 = c1::create(a1);
    f_(*p0, *p1);
  }
  void wait() { policy::wait();}

private:
  function_type f_;
};

/// Three-arguments task.
template <typename T, typename A0, typename A1, typename A2>
struct Task<T, tuple<A0, A1, A2> >
{
  typedef impl::Policy<T, tuple<A0, A1, A2> > policy;
  typedef typename policy::function_type function_type;

public:
  Task(function_type const &f) : f_(f) {}

  void execute(typename policy::template arg<0>::type a0,
	       typename policy::template arg<1>::type a1,
	       typename policy::template arg<2>::type a2)
  {
    start(a0, a1, a2);
    wait();
  }

  void start(typename policy::template arg<0>::type a0,
	     typename policy::template arg<1>::type a1,
	     typename policy::template arg<2>::type a2)
  {
    policy::start();
    typedef typename policy::template channel<0>::type c0;
    typedef typename policy::template channel<1>::type c1;
    typedef typename policy::template channel<2>::type c2;
    typename c0::type p0 = c0::create(a0);
    typename c1::type p1 = c1::create(a1);
    typename c1::type p2 = c2::create(a2);
    f_(*p0, *p1, *p2);
  }
  void wait() { policy::wait();}

private:
  function_type f_;
};

/// Four-arguments task.
template <typename T, typename A0, typename A1, typename A2, typename A3>
struct Task<T, tuple<A0, A1, A2, A3> >
{
  typedef impl::Policy<T, tuple<A0, A1, A2, A3> > policy;
  typedef typename policy::function_type function_type;

public:
  Task(function_type const &f) : f_(f) {}

  void execute(typename policy::template arg<0>::type a0,
	       typename policy::template arg<1>::type a1,
	       typename policy::template arg<2>::type a2,
	       typename policy::template arg<3>::type a3)
  {
    start(a0, a1, a2, a3);
    wait();
  }

  void start(typename policy::template arg<0>::type a0,
	     typename policy::template arg<1>::type a1,
	     typename policy::template arg<2>::type a2,
	     typename policy::template arg<3>::type a3)
  {
    policy::start();
    typedef typename policy::template channel<0>::type c0;
    typedef typename policy::template channel<1>::type c1;
    typedef typename policy::template channel<2>::type c2;
    typedef typename policy::template channel<3>::type c3;
    typename c0::type p0 = c0::create(a0);
    typename c1::type p1 = c1::create(a1);
    typename c2::type p2 = c2::create(a2);
    typename c3::type p3 = c3::create(a3);
    f_(*p0, *p1, *p2, *p3);
  }
  void wait() { policy::wait();}

private:
  function_type f_;
};

template <typename T, typename A0, typename A1, typename A2, typename A3, typename A4>
struct Task<T, tuple<A0, A1, A2, A3, A4> >
{
  typedef impl::Policy<T, tuple<A0, A1, A2, A3, A4> > policy;
  typedef typename policy::function_type function_type;

public:
  Task(function_type const &f) : f_(f) {}

  void execute(typename policy::template arg<0>::type a0,
	       typename policy::template arg<1>::type a1,
	       typename policy::template arg<2>::type a2,
	       typename policy::template arg<3>::type a3,
	       typename policy::template arg<4>::type a4)
  {
    start(a0, a1, a2, a3, a4);
    wait();
  }

  void start(typename policy::template arg<0>::type a0,
	     typename policy::template arg<1>::type a1,
	     typename policy::template arg<2>::type a2,
	     typename policy::template arg<3>::type a3,
	     typename policy::template arg<4>::type a4)
  {
    policy::start();
    typedef typename policy::template channel<0>::type c0;
    typedef typename policy::template channel<1>::type c1;
    typedef typename policy::template channel<2>::type c2;
    typedef typename policy::template channel<3>::type c3;
    typedef typename policy::template channel<4>::type c4;
    typename c0::type p0 = c0::create(a0);
    typename c1::type p1 = c1::create(a1);
    typename c2::type p2 = c2::create(a2);
    typename c3::type p3 = c3::create(a3);
    typename c4::type p4 = c4::create(a4);
    f_(*p0, *p1, *p2, *p3, *p4);
  }
  void wait() { policy::wait();}

private:
  function_type f_;
};

} // namespace vsip_csl::udk
} // namespace vsip_csl

#endif
