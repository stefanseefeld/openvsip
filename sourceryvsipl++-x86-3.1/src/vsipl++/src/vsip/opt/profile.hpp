/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Profiling routines & classes.

#ifndef VSIP_OPT_PROFILE_HPP
#define VSIP_OPT_PROFILE_HPP

#include <vsip/core/config.hpp>
#include <vsip/core/noncopyable.hpp>
#include <vector>
#include <map>
#include <string>
#include <iosfwd>
#include <memory>


#if (VSIP_IMPL_PROFILE_TIMER == 1)
// Posix_time uses clock_gettime()
#  include <time.h>
#endif

#if (VSIP_IMPL_PROFILE_TIMER == 4)
#  include <mcos.h>
#endif

/// These macros are used to conditionally compile profiling code
/// so that it can be easily added or removed from the library 
/// based on the mask defined above.
///
///   VSIP_IMPL_PROFILE_FEATURE     is used on profiling statements for
///                                 a given operation in a specific module
///                                 (see fft.hpp for an example).
///
///   VSIP_IMPL_PROFILE             is used on statements that are not
///                                 feature-specific (see this file).

// Enable (or not) for a single statement
#define VSIP_IMPL_PROFILE_EN_0(X) 
#define VSIP_IMPL_PROFILE_EN_1(X) X

// Join two names together (allowing for expansion of macros)
#define VSIP_IMPL_JOIN(A, B) VSIP_IMPL_JOIN_1(A, B)
#define VSIP_IMPL_JOIN_1(A, B) A ## B

/// The profiling of specific areas of the library is a feature which can be 
/// enabled in each module using this macro.  A second  macro, defined in 
/// the module itself, must be set prior to using this.
#define VSIP_IMPL_PROFILE_FEATURE(STMT)			\
     VSIP_IMPL_JOIN(VSIP_IMPL_PROFILE_EN_,		\
                    VSIP_IMPL_PROFILING_FEATURE_ENABLED) 	(STMT)

// This macro may be used to disable statements that apply to profiling
// in general.
#if (VSIP_PROFILE_MASK)
#define VSIP_IMPL_PROFILE(STMT)		STMT
#else
#define VSIP_IMPL_PROFILE(STMT)
#endif


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace profile
{

// Declaration of thread synchronization function.  For accurate timing
//  of asynchronous operations this must be called before recording the time to
//  make sure that all queued threads executing on the device have finished.
extern void synchronize(void);

/// Timing policies.  The following classes standardize access to
/// several different system timers.  These classes are used to
/// specialize P_timer and P_acc_timer.
///
/// Each provides the following:
///  - stamp_type       - the type used to represent a time sample
///                       and also a time delta.
///  - sample()         - sample the current time.
///  - add()            - sum of  two timestamps.
///  - sub()            - different between two timestamps.
///  - seconds()        - convert a timestamp into seconds.
///
///  - clocks_per_sec   - number of samples per second.
///
/// The following timers are provided:
///  - No_time
///  - Posix_time       - Posix clock()
///  - Posix_real_time  - Posix clock_gettime(CLOCK_REALTIME, ...)
///  - Pentium_tsc_time - pentium timestamp counter (ia32 asm)
///  - X86_64_tsc_time  - pentium timestamp counter (x86_64 asm)
///  - Mcoe_tmr_time    - MCOE tmr timer.
///  - Power_tb_time    - PowerPC timebase timer

struct No_time
{
  static bool const valid = false; 
  static char const* name() { return "No_time"; }
  static void init() {}

  typedef int stamp_type;
  typedef unsigned long long tick_type;
  static void sample(stamp_type& time) { time = 0; }
  static stamp_type zero() { return stamp_type(); }
  static stamp_type f_clocks_per_sec() { return 1; }
  static stamp_type add(stamp_type , stamp_type) { return 0; }
  static stamp_type sub(stamp_type , stamp_type) { return 0; }
  static float seconds(stamp_type) { return 0.f; }
  static tick_type ticks(stamp_type time) 
    { return (tick_type)time; }
  static bool is_zero(stamp_type const& stamp)
    { return stamp == 0; }

  static stamp_type clocks_per_sec;
};



#if (VSIP_IMPL_PROFILE_TIMER == 1)
struct Posix_time
{
  static bool const valid = true; 
  static char const* name() { return "Posix_time"; }
  static void init() { clocks_per_sec.tv_sec = 1; clocks_per_sec.tv_nsec = 0; }

  static clockid_t const clock = CLOCK_MONOTONIC;
  typedef struct timespec stamp_type;
  typedef unsigned long long tick_type;
  static void sample(stamp_type& time) { clock_gettime(clock, &time); }
  static stamp_type zero() { return stamp_type(); }

  static stamp_type add(stamp_type A, stamp_type B)
  {
    stamp_type res;
    res.tv_nsec = A.tv_nsec + B.tv_nsec;
    res.tv_sec  = A.tv_sec  + B.tv_sec;
    if (res.tv_nsec >= 1000000000LL)
    {
      res.tv_nsec -= 1000000000LL;
      res.tv_sec  += 1;
    }
    return res;
  }

  static stamp_type sub(stamp_type A, stamp_type B)
  {
    stamp_type res;
    if (A.tv_nsec >= B.tv_nsec)
    {
      res.tv_nsec = A.tv_nsec - B.tv_nsec;
      res.tv_sec  = A.tv_sec  - B.tv_sec;
    }
    else
    {
      res.tv_nsec = 1000000000LL - (B.tv_nsec - A.tv_nsec);
      res.tv_sec  = A.tv_sec  - B.tv_sec - 1;
    }
    return res;
  }

  static float seconds(stamp_type time)
    { return (float)(time.tv_sec) + (float)(time.tv_nsec) / 1e9; }

  static tick_type ticks(stamp_type time)
  { 
    return (tick_type)(time.tv_sec * 1e9) + 
      (tick_type)time.tv_nsec; 
  }
  static bool is_zero(stamp_type const& stamp)
    { return stamp.tv_nsec == 0 && stamp.tv_sec == 0; }

  static stamp_type clocks_per_sec;
};
#endif // (VSIP_IMPL_PROFILE_TIMER == 1)



#if (VSIP_IMPL_PROFILE_TIMER == 2)
struct Pentium_tsc_time
{
  static bool const valid = true; 
  static char const* name() { return "Pentium_tsc_time"; }
  static void init();

  typedef long long stamp_type;
  typedef unsigned long long tick_type;
  static void sample(stamp_type& time)
    { __asm__ __volatile__("rdtsc": "=A" (time)); }
  static stamp_type zero() { return stamp_type(); }
  static stamp_type add(stamp_type A, stamp_type B) { return A + B; }
  static stamp_type sub(stamp_type A, stamp_type B) { return A - B; }
  static float seconds(stamp_type time) { return (float)time / (float)clocks_per_sec; }
  static tick_type ticks(stamp_type time) { return (tick_type)time; }
  static bool is_zero(stamp_type const& stamp)
    { return stamp == stamp_type(); }

  static stamp_type clocks_per_sec;
};
#endif // (VSIP_IMPL_PROFILE_TIMER == 2)



#if (VSIP_IMPL_PROFILE_TIMER == 3)
struct X86_64_tsc_time
{
  static bool const valid = true; 
  static char const* name() { return "x86_64_tsc_time"; }
  static void init();

  typedef unsigned long long stamp_type;
  typedef unsigned long long tick_type;
  static void sample(stamp_type& time)
    { unsigned a, d; __asm__ __volatile__("rdtsc": "=a" (a), "=d" (d));
      time = ((stamp_type)a) | (((stamp_type)d) << 32); }
  static stamp_type zero() { return stamp_type(); }
  static stamp_type add(stamp_type A, stamp_type B) { return A + B; }
  static stamp_type sub(stamp_type A, stamp_type B) { return A - B; }
  static float seconds(stamp_type time) { return (float)time / (float)clocks_per_sec; }
  static tick_type ticks(stamp_type time) { return (tick_type)time; }
  static bool is_zero(stamp_type const& stamp)
    { return stamp == stamp_type(); }

  static stamp_type clocks_per_sec;
};
#endif // (VSIP_IMPL_PROFILE_TIMER == 3)

#if (VSIP_IMPL_PROFILE_TIMER == 4)
struct Mcoe_tmr_time
{
  typedef TMR_timespec stamp_type;
  typedef unsigned long long tick_type;
  static bool const valid = true; 
  static char const* name() { return "Mcoe_tmr_time"; }
  static void init()
  {
    tmr_timestamp(&time0); 
    clocks_per_sec.tv_sec = 1;
    clocks_per_sec.tv_nsec = 0;
  }

  static void sample(stamp_type& time)
  {
    TMR_ts tmp;
    tmr_timestamp(&tmp); 
    tmr_diff(time0, tmp, 0L, &time);
  }

  static stamp_type zero() { return stamp_type(); }

  static stamp_type add(stamp_type A, stamp_type B)
  {
    stamp_type res;
    res.tv_nsec = A.tv_nsec + B.tv_nsec;
    res.tv_sec  = A.tv_sec  + B.tv_sec;
    if (res.tv_nsec >= 1000000000LL)
    {
      res.tv_nsec -= 1000000000LL;
      res.tv_sec  += 1;
    }
    return res;
  }

  static stamp_type sub(stamp_type A, stamp_type B)
  {
    stamp_type res;
    if (A.tv_nsec >= B.tv_nsec)
    {
      res.tv_nsec = A.tv_nsec - B.tv_nsec;
      res.tv_sec  = A.tv_sec  - B.tv_sec;
    }
    else
    {
      res.tv_nsec = 1000000000LL - (B.tv_nsec - A.tv_nsec);
      res.tv_sec  = A.tv_sec  - B.tv_sec - 1;
    }
    return res;
  }

  static float seconds(stamp_type time)
    { return (float)(time.tv_sec) + (float)(time.tv_nsec) / 1e9; }

  static tick_type ticks(stamp_type time)
    { return (tick_type)(time.tv_sec * 1e9) + (tick_type)time.tv_nsec; }
  static bool is_zero(stamp_type const& stamp)
    { return stamp.tv_nsec == 0 && stamp.tv_sec == 0; }

  static stamp_type clocks_per_sec;
  static TMR_ts time0;
};
#endif // (VSIP_IMPL_PROFILE_TIMER == 4)

#if (VSIP_IMPL_PROFILE_TIMER == 5)
struct Power_tb_time
{
  static bool const valid = true; 
  static char const* name() { return "Power_tb_time"; }
  static void init();

  typedef long long stamp_type;
  typedef unsigned long long tick_type;
  static void sample(stamp_type& time)
  {
    unsigned int tbl, tbu0, tbu1;

    // Make sure that the upper 32 bits aren't incremented while
    // reading the lower 32.  Mixing a pre-increment lower 32 value
    // (FFFF) with a post-increment upper 32 bit value, or a
    // post-increment lower 32 value with a pre-increment upper 32
    // would introduce a large measurement error and might result in
    // non-sensical time deltas.
    do
    {
      __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
      __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
      __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    }
    while (tbu0 != tbu1);

    time = (((unsigned long long)tbu0) << 32) | tbl;
  }
  static stamp_type zero() { return stamp_type(); }
  static stamp_type add(stamp_type A, stamp_type B) { return A + B; }
  static stamp_type sub(stamp_type A, stamp_type B) { return A - B; }
  static float seconds(stamp_type time) { return (float)time / (float)clocks_per_sec; }
  static tick_type ticks(stamp_type time) { return (tick_type)time; }
  static bool is_zero(stamp_type const& stamp)
    { return stamp == stamp_type(); }

  static stamp_type clocks_per_sec;
};
#endif // (VSIP_IMPL_PROFILE_TIMER == 5)




#if   (VSIP_IMPL_PROFILE_TIMER == 1)
typedef Posix_time       DefaultTime;
#elif (VSIP_IMPL_PROFILE_TIMER == 2)
typedef Pentium_tsc_time DefaultTime;
#elif (VSIP_IMPL_PROFILE_TIMER == 3)
typedef X86_64_tsc_time DefaultTime;
#elif (VSIP_IMPL_PROFILE_TIMER == 4)
typedef Mcoe_tmr_time DefaultTime;
#elif (VSIP_IMPL_PROFILE_TIMER == 5)
typedef Power_tb_time DefaultTime;
#else // default choice if undefined or zero
typedef No_time        DefaultTime;
#endif



/// Timer class that keeps start/stop times.
///
/// Template parameters:
///   :TP: is a timer policy.
template <typename TP>
class P_timer 
{
private:
  typedef typename TP::stamp_type stamp_type;

  stamp_type	start_;
  stamp_type	stop_;

public:
  P_timer() {}

  void start() { synchronize(); TP::sample(start_); }

  void stop() { synchronize(); TP::sample(stop_); }

  stamp_type raw_delta() { return TP::sub(stop_, start_); }
  float delta() { return TP::seconds(TP::sub(stop_, start_)); }
};



/// Timer class that accumulates across multiple start/stop times.
///
/// Template parameters:
///   :TP: is a timer policy.
template <typename TP>
class P_acc_timer 
{
public:
  typedef typename TP::stamp_type stamp_type;

  P_acc_timer() { this->reset(); }

  stamp_type start() 
  { 
    synchronize();
    TP::sample(start_); return start_; 
  }
  stamp_type stop()
  {
    synchronize();
    TP::sample(stop_);
    total_ = TP::add(total_, TP::sub(stop_, start_));
    count_ += 1;
    return stop_;
  }

  void reset()
  { total_ = stamp_type(); count_ = 0; }

  stamp_type raw_delta() const { return TP::sub(stop_, start_); }
  float delta() const { return TP::seconds(TP::sub(stop_, start_)); }
  float total() const { return TP::seconds(total_); }
  int   count() const { return count_; }

private:
  stamp_type	total_;
  stamp_type	start_;
  stamp_type	stop_;
  unsigned	count_;
};
   


typedef P_timer<DefaultTime>     Timer;
typedef P_acc_timer<DefaultTime> Acc_timer;


enum profiler_mode
{
  pm_log,
  pm_trace,
  pm_accum,
  pm_none
};

class Profiler
{
public:

  typedef DefaultTime    TP;
  typedef TP::stamp_type stamp_type;

  struct Trace_entry
  {
    int         idx;
    std::string name;
    stamp_type  stamp;
    int         end;
    int         value;
    
    Trace_entry(int i, std::string const &n, stamp_type s, int e, int v)
      : idx(i), name(n), stamp(s), end(e), value(v) {}
  };

  struct Accum_entry
  {
    stamp_type total;  // total time spent
    size_t     count;  // # times called
    int        value;  // op count per call
    
    Accum_entry(stamp_type t, size_t c, int v)
      : total(t), count(c), value(v) {}
  };

  typedef std::map<std::string, Accum_entry> accum_type;
  typedef std::vector<Trace_entry> trace_type;

  Profiler();
  ~Profiler();

  void clear()
  {
    accum_.clear();
    data_.clear();
  }

  /// Create a profiler event.
  ///
  /// :Arguments:
  ///   :name: the event name.
  ///   :value: a value associated with the event (such as number of
  ///           operations, number of bytes, etc).
  ///   :id:    a scope id. If `0`, this is a scope enter event. If `>0`,
  ///           this is a scope leave event. If `<0`, this is a standalone
  ///           event.
  ///   :stamp:
  ///
  /// :Returns:
  ///   A new id. This is only valid if this was a scope enter event, i.e.
  ///   for `id == 0`.
  ///
  int event(std::string const &name, int value = 0, int id = -1, 
            stamp_type stamp = stamp_type());
  void dump(std::string const &filename, char mode='w');
  void set_mode(profiler_mode mode) { mode_ = mode;}
  void set_output(std::string const &filename);

  accum_type::iterator begin_accum() { return accum_.begin();}
  accum_type::iterator end_accum() { return accum_.end();}

  trace_type::iterator begin_trace() { return data_.begin();}
  trace_type::iterator end_trace() { return data_.end();}

private:
  /// Log the given message to the log stream.
  /// Arguments:
  ///
  ///   :msg: The message to be written
  ///   :id:  The id. If it is `0`, this is a scope enter event.
  ///         If it is `> 0`, it is a scope leave event. Otherwise
  ///         it is a standalone event.
  int log(std::string const &msg, int id);

  profiler_mode              mode_;
  int                        count_;
  std::auto_ptr<std::filebuf> log_buf_;
  std::ostream               log_;
  trace_type                 data_;
  accum_type                 accum_;

#if VSIP_IMPL_PROFILE_NESTING
  typedef std::vector<std::string> event_stack_type;
  event_stack_type           event_stack_;
#endif
};


extern Profiler* prof;

class Profile
{
public:
  Profile(std::string const &filename, profiler_mode mode = pm_accum)
    : filename_(filename), mode_(mode)
  {
    if (mode == pm_log) prof->set_output(filename_);
    prof->set_mode(mode);
  }
  ~Profile() 
  {
    if (mode_ != pm_log)
      prof->dump(this->filename_);
  }

private:
  std::string filename_;
  profiler_mode mode_;
};

class Profiler_options
{
  // Constructors.
public:
  Profiler_options(int& argc, char**& argv);
  ~Profiler_options();

private:
  void strip_args(int& argc, char**& argv);

  Profile* profile_;
};

template <bool enabled> class Accumulator_base;

template <>
class Accumulator_base<false>
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

template <>
class Accumulator_base<true>
{
public:
  class Scope
  {
  public:
    Scope(Accumulator_base &);
    ~Scope();

  private:
    Accumulator_base & scope_;
    int id_;
  };
  friend struct Scope;

  Accumulator_base(std::string const &name, unsigned int ops)
    : name_(name), ops_(ops)
  {}

  void ops(unsigned int ops_count) { this->ops_ = ops_count;}
  unsigned int ops() const { return this->ops_;}

  std::string const &name() const { return this->name_;}

  float total() const { return this->timer_.total();}
  int   count() const { return this->timer_.count();}
  float  mops() const { return (count() * ops()) / (1e6 * total());}

  Acc_timer::stamp_type enter() { return this->timer_.start();}
  Acc_timer::stamp_type  leave() { return this->timer_.stop();}

private:
  std::string name_;
  unsigned int ops_;
  Acc_timer timer_;
};

inline Accumulator_base<true>::Scope::Scope(Accumulator_base<true> &a)
  : scope_(a),
    id_(prof->event(scope_.name(), scope_.ops(), 0, scope_.enter()))
{}

inline Accumulator_base<true>::Scope::~Scope()
{
  prof->event(scope_.name(), 0, id_, scope_.leave());
}

template <bool enabled> class Scope_base;

template <>
class Scope_base<false> : Non_copyable
{
public:
  Scope_base(std::string const &, int) {}
};

template <>
class Scope_base<true> : Non_copyable
{
public:
  Scope_base(std::string const &name, int value=0)
    : name_(name), id_  (prof->event(name, value, 0)) {}
  ~Scope_base() { prof->event(name_, 0, id_, DefaultTime::stamp_type());}

private:
  std::string name_;
  int   id_;
};



/// Start/stop an Acc_timer while object is in scope.
class Scope_timer
{
public:
  Scope_timer(Acc_timer& timer)
    : timer_(timer)
  { this->timer_.start(); }

  ~Scope_timer()
  { this->timer_.stop(); }

private:
  Acc_timer& timer_;
};

typedef Scope_timer Time_in_scope;

} // namespace vsip::impl::profile
} // namespace vsip::impl
} // namespace vsip

#undef VSIP_IMPL_PROFILE

#endif // VSIP_OPT_PROFILE_HPP
