/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/profile.hpp
    @author  Stefan Seefeld
    @date    2006-11-24
    @brief   VSIPL++ Library: Profiling routines & classes.
*/

#ifndef VSIP_CORE_PROFILE_HPP
#define VSIP_CORE_PROFILE_HPP

#include <vsip/core/config.hpp>
#include <vsip/core/noncopyable.hpp>
#include <vsip/core/ops_info.hpp>
#ifndef VSIP_IMPL_REF_IMPL
# include <vsip/opt/profile.hpp>
#endif

// First define individual flags mapping to features...

#define VSIP_IMPL_PROFILE_MASK_MEMORY    1 << 0 // 0xff is low-level data management
#define VSIP_IMPL_PROFILE_MASK_COPY      1 << 1
#define VSIP_IMPL_PROFILE_MASK_DISPATCH  1 << 2
#define VSIP_IMPL_PROFILE_MASK_PARALLEL  1 << 3

#define VSIP_IMPL_PROFILE_MASK_FUNC      1 << 8 // 0xff00 is high-level operations
#define VSIP_IMPL_PROFILE_MASK_SIGNAL    1 << 9
#define VSIP_IMPL_PROFILE_MASK_MATVEC    1 << 10
#define VSIP_IMPL_PROFILE_MASK_SOLVER    1 << 11

#define VSIP_IMPL_PROFILE_MASK_USER      1 << 16 // 0xff0000 is user data

#define VSIP_IMPL_PROFILE_MASK_ALL       0xffffff

// ...then default-define macros, so users can override them.

#ifndef VSIP_PROFILE_MEMORY
# define VSIP_PROFILE_MEMORY 0
#endif
#ifndef VSIP_PROFILE_COPY
# define VSIP_PROFILE_COPY 0
#endif
#ifndef VSIP_PROFILE_DISPATCH
# define VSIP_PROFILE_DISPATCH 0
#endif
#ifndef VSIP_PROFILE_PARALLEL
# define VSIP_PROFILE_PARALLEL 0
#endif
#ifndef VSIP_PROFILE_FUNC
# define VSIP_PROFILE_FUNC 0
#endif
#ifndef VSIP_PROFILE_SIGNAL
# define VSIP_PROFILE_SIGNAL 0
#endif
#ifndef VSIP_PROFILE_MATVEC
# define VSIP_PROFILE_MATVEC 0
#endif
#ifndef VSIP_PROFILE_SOLVER
# define VSIP_PROFILE_SOLVER 0
#endif
#ifndef VSIP_PROFILE_USER
# define VSIP_PROFILE_USER 0
#endif
#ifndef VSIP_PROFILE_ALL
# define VSIP_PROFILE_ALL 0
#endif

// Finally accumulate all enabled profile feature flags into a single mask.

#ifndef VSIP_PROFILE_MASK
# define VSIP_PROFILE_MASK \
  ((VSIP_PROFILE_MEMORY ? VSIP_IMPL_PROFILE_MASK_MEMORY : 0) |	   \
   (VSIP_PROFILE_COPY ? VSIP_IMPL_PROFILE_MASK_COPY : 0) |	   \
   (VSIP_PROFILE_DISPATCH ? VSIP_IMPL_PROFILE_MASK_DISPATCH : 0) | \
   (VSIP_PROFILE_PARALLEL ? VSIP_IMPL_PROFILE_MASK_PARALLEL : 0) | \
   (VSIP_PROFILE_FUNC ? VSIP_IMPL_PROFILE_MASK_FUNC : 0) |	   \
   (VSIP_PROFILE_SIGNAL ? VSIP_IMPL_PROFILE_MASK_SIGNAL : 0) |	   \
   (VSIP_PROFILE_MATVEC ? VSIP_IMPL_PROFILE_MASK_MATVEC : 0) |	   \
   (VSIP_PROFILE_SOLVER ? VSIP_IMPL_PROFILE_MASK_SOLVER : 0) |	   \
   (VSIP_PROFILE_USER ? VSIP_IMPL_PROFILE_MASK_USER : 0) |         \
   (VSIP_PROFILE_ALL ?  VSIP_IMPL_PROFILE_MASK_ALL : 0))
#endif

namespace vsip
{
namespace impl
{
namespace profile
{
unsigned long const mask = VSIP_PROFILE_MASK;

#ifndef VSIP_IMPL_PROFILE
# define VSIP_IMPL_PROFILE(X)
#endif

/// Different operations that may be profiled, each is referred to
/// as a 'feature'.
enum Feature
{
  none,
  memory  = VSIP_IMPL_PROFILE_MASK_MEMORY,  // memory (de-)allocation
  copy    = VSIP_IMPL_PROFILE_MASK_COPY,    // data copies
  dispatch= VSIP_IMPL_PROFILE_MASK_DISPATCH,// operation dispatch
  par     = VSIP_IMPL_PROFILE_MASK_PARALLEL,// parallel comms

  func    = VSIP_IMPL_PROFILE_MASK_FUNC,    // elementwise dispatch (+, -, etc)
  signal  = VSIP_IMPL_PROFILE_MASK_SIGNAL,  // signal processing (FFT, FIR, etc)
  matvec  = VSIP_IMPL_PROFILE_MASK_MATVEC,  // matrix-vector (prod, dot, etc)
  solver  = VSIP_IMPL_PROFILE_MASK_SOLVER,  // solvers (qr, svd, etc)

  user    = VSIP_IMPL_PROFILE_MASK_USER,    // user defined tag

  all     = VSIP_IMPL_PROFILE_MASK_ALL
};

template <unsigned int Feature>
inline void event(std::string const &, int = 0) {}

#if defined(VSIP_IMPL_REF_IMPL)
template <bool>
class Accumulator_base
{
public:
  struct Scope
  {
    Scope(Accumulator_base &) {}
  };
  Accumulator_base(std::string const &, unsigned int = 0) {}
  unsigned int ops() const { return 0;}
  float total() const { return 0.;}
  int   count() const { return 0;}
  float  mops() const { return 0.;}
};

template <bool>
class Scope_base : Non_copyable
{
public:
  Scope_base(std::string const &, int=0) {}
};

enum profiler_mode
{
  pm_trace,
  pm_accum,
  pm_none
};

class Profile
{
public:
  Profile(std::string const &, profiler_mode = pm_accum) {}
};

#else
//For all enabled features, overload event<>()

#if VSIP_PROFILE_MASK & VSIP_IMPL_PROFILE_MASK_MEMORY
template <>
inline void event<memory>(std::string const &e, int value)
{ prof->event(e, value);}
#endif
#if VSIP_PROFILE_MASK & VSIP_IMPL_PROFILE_MASK_COPY
template <>
inline void event<copy>(std::string const &e, int value)
{ prof->event(e, value);}
#endif
#if VSIP_PROFILE_MASK & VSIP_IMPL_PROFILE_MASK_DISPATCH
template <>
inline void event<dispatch>(std::string const &e, int value)
{ prof->event(e, value);}
#endif
#if VSIP_PROFILE_MASK & VSIP_IMPL_PROFILE_MASK_PARALLEL
template <>
inline void event<par>(std::string const &e, int value)
{ prof->event(e, value);}
#endif
#if VSIP_PROFILE_MASK & VSIP_IMPL_PROFILE_MASK_FUNC
template <>
inline void event<func>(std::string const &e, int value)
{ prof->event(e, value);}
#endif
#if VSIP_PROFILE_MASK & VSIP_IMPL_PROFILE_MASK_SIGNAL
template <>
inline void event<signal>(std::string const &e, int value)
{ prof->event(e, value);}
#endif
#if VSIP_PROFILE_MASK & VSIP_IMPL_PROFILE_MASK_MATVEC
template <>
inline void event<matvec>(std::string const &e, int value)
{ prof->event(e, value);}
#endif
#if VSIP_PROFILE_MASK & VSIP_IMPL_PROFILE_MASK_USER
template <>
inline void event<user>(std::string const &e, int value)
{ prof->event(e, value);}
#endif

#endif

template <unsigned int Feature>
class Accumulator : public Accumulator_base<Feature & mask>
{
  typedef Accumulator_base<Feature & mask> base;
public:
  Accumulator(std::string const &n, unsigned int c = 0) : base(n, c) {}
};

template <unsigned int Feature>
class Scope : public Scope_base<Feature & mask>
{
  typedef Scope_base<Feature & mask> base;
public:
  Scope(std::string const &n, int id = 0) : base(n, id) {}
};

} // namespace vsip::impl::profile
} // namespace vsip::impl
} // namespace vsip

#endif
