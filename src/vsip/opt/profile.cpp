/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Profile harness implementation

#include <fstream>
#include <iostream>
#include <cstring>

// Profiling should be enabled when compiling this module so that
// these functions are available if the user enables profiling.
// Setting this to a non-zero value before including the header 
// is sufficient.
#undef VSIP_PROFILE_MASK
#define VSIP_PROFILE_MASK   1
#include <vsip/opt/profile.hpp>
#include <vsip/core/c++0x.hpp>

# if VSIP_IMPL_HAVE_CUDA
#  include <cuda_runtime.h>
# endif

namespace vsip
{
namespace impl
{
namespace profile
{

/// Synchronize threads

void synchronize(void)
{
  # if VSIP_IMPL_HAVE_CUDA
  cudaThreadSynchronize();
  # endif
}

/// Returns CPU speed from /proc/cpuinfo, in MHz

float
read_cpu_info()
{
  char     buffer[1024];
  float    mhz = 1000;
  std::ifstream file;

  file.open("/proc/cpuinfo");

  while(!file.eof()) 
  {
    file.getline(buffer, sizeof(buffer));
    if (sscanf(buffer, "cpu MHz : %f", &mhz))
      break;
  }

  file.close();

  return mhz;
}



long long
read_timebase()
{
  char      buffer[1024];
  long long timebase = 1000;
  std::ifstream file;

  file.open("/proc/cpuinfo");

  while(!file.eof()) 
  {
    file.getline(buffer, sizeof(buffer));
    if (sscanf(buffer, "timebase : %lld", &timebase) == 1)
      break;
  }

  file.close();

  return timebase;
}


float
get_cpu_speed()
{
#if VSIP_IMPL_PROFILE_HARDCODE_CPU_SPEED
   return VSIP_IMPL_PROFILE_HARDCODE_CPU_SPEED;
#else
   return read_cpu_info();
#endif
}


#if (VSIP_IMPL_PROFILE_TIMER == 0)
No_time::stamp_type No_time::clocks_per_sec;
#endif

#if (VSIP_IMPL_PROFILE_TIMER == 1)
Posix_time::stamp_type Posix_time::clocks_per_sec;
#endif // (VSIP_IMPL_PROFILE_TIMER == 1)

#if (VSIP_IMPL_PROFILE_TIMER == 2)
Pentium_tsc_time::stamp_type Pentium_tsc_time::clocks_per_sec;

void
Pentium_tsc_time::init()
{
  float mhz = get_cpu_speed();
  clocks_per_sec = static_cast<Pentium_tsc_time::stamp_type>(mhz * 1000000);
}
#endif // (VSIP_IMPL_PROFILE_TIMER == 2)

#if (VSIP_IMPL_PROFILE_TIMER == 3)
X86_64_tsc_time::stamp_type X86_64_tsc_time::clocks_per_sec;

void
X86_64_tsc_time::init()
{
  float mhz = get_cpu_speed();
  clocks_per_sec = static_cast<X86_64_tsc_time::stamp_type>(mhz * 1000000);
}
#endif // (VSIP_IMPL_PROFILE_TIMER == 3)

#if (VSIP_IMPL_PROFILE_TIMER == 4)
Mcoe_tmr_time::stamp_type Mcoe_tmr_time::clocks_per_sec;
TMR_ts Mcoe_tmr_time::time0;
#endif // (VSIP_IMPL_PROFILE_TIMER == 4)

#if (VSIP_IMPL_PROFILE_TIMER == 5)
Power_tb_time::stamp_type Power_tb_time::clocks_per_sec;

void
Power_tb_time::init()
{
  float timebase = read_timebase();
  clocks_per_sec = static_cast<Power_tb_time::stamp_type>(timebase);
}
#endif // (VSIP_IMPL_PROFILE_TIMER == 5)

Profiler* prof;

class SetupProf
{
public:
  SetupProf()
  {
    prof = new Profiler;
    DefaultTime::init();
  }

  ~SetupProf()
  { delete prof; }
};

SetupProf obj;

Profiler::Profiler()
  : mode_(pm_none),
    log_buf_(new std::filebuf),
    log_(std::cout.rdbuf())
{
}

Profiler::~Profiler()
{
}

int
Profiler::event(std::string const &name, int value, int open_id, stamp_type stamp)
{
  if (mode_ == pm_log) return log(name, open_id);
  else if (mode_ == pm_trace || open_id < 0)
  {
    // Obtain a stamp if one is not provided.
    if (TP::is_zero(stamp))
    {
      synchronize();
      TP::sample(stamp);
    }

    count_++;
    data_.push_back(Trace_entry(count_, name, stamp, open_id, value));
    return count_;
  }
  else if (mode_ == pm_accum)
  {
#if VSIP_IMPL_PROFILE_NESTING
    // Push event onto stack if this is start of event.
    if (open_id == 0)
      event_stack_.push_back(name);

    // Build nested event name from stack.
    std::string event_name(event_stack_[0]);
    for (unsigned i=1; i<event_stack_.size(); ++i)
    {
      event_name += "\\,";
      event_name += event_stack_[i];
    }

    // Pop event from stack if this is end of event.
    if (open_id > 0)
      event_stack_.pop_back();
#else
    std::string event_name(name);
#endif
    // Obtain a stamp if one is not provided.
    if (TP::is_zero(stamp))
    {
      synchronize();
      TP::sample(stamp);
    }

    accum_type::iterator pos = accum_.find(event_name);
    if (pos == accum_.end())
    {
      accum_.insert(std::make_pair(event_name, 
                      Accum_entry(TP::zero(), 0, value)));
      pos = accum_.find(event_name);
    }

    // The value of 'open_id' determines if it is entering scope or exiting 
    // scope.  This allows it to work with the Scope classes, which call 
    // it with 0 in it's constructor (usually placed at the beginning of a 
    // function body) and with a non-zero value in it's destructor.  The net 
    // result is that it accumulates time when it is alive.
    if (open_id == 0)
      pos->second.total = TP::sub(pos->second.total, stamp);
    else if (open_id > 0)
    {
      pos->second.total = TP::add(pos->second.total, stamp);
      pos->second.count++;
    }
    // A non-zero value is returned for the benefit of the Scope classes,
    // which turns around and passes it back as the 'open_id' parameter in 
    // order to indicate the event is going out of scope.
    return 1;
  }
  return 0;
}


void
Profiler::dump(std::string const &filename, char /*mode*/)
{
  if (mode_ == pm_log) return;

  std::ofstream os;
  char const *delim = " : ";
  if (filename != "" && filename != "-") os.open(filename.c_str());
  else
  {
    os.copyfmt(std::cout);
    static_cast<std::basic_ios<char> &>(os).rdbuf(std::cout.rdbuf());
  }
  if (mode_ == pm_trace)
  {
    os << "# mode: pm_trace\n"
       << "# timer: " << TP::name() << '\n'
       << "# clocks_per_sec: " << TP::ticks(TP::clocks_per_sec) << '\n'
       << "# \n"
       << "# index" << delim << "tag" << delim << "ticks" << delim << "open id" 
       << delim << "op count" << std::endl;

    typedef trace_type::iterator iterator;

    for (iterator cur = data_.begin(); cur != data_.end(); ++cur)
    {
      os << (*cur).idx
         << delim << (*cur).name
         << delim << TP::ticks((*cur).stamp)
         << delim << (*cur).end
         << delim << (*cur).value 
         << std::endl;
    }
    data_.clear();
  }
  else if (mode_ == pm_accum)
  {
    os << "# mode: pm_accum\n"
       << "# timer: " << TP::name() << '\n'
       << "# clocks_per_sec: " << TP::ticks(TP::clocks_per_sec) << '\n'
       << "# \n"
       << "# tag" << delim << "total ticks" << delim << "num calls" 
       << delim << "op count" << delim << "mops" << std::endl;
    
    typedef accum_type::iterator iterator;
  
    for (iterator cur = accum_.begin(); cur != accum_.end(); ++cur)
    {
      float mops = (*cur).second.count * (*cur).second.value /
	(1e6 * TP::seconds((*cur).second.total));
      os << (*cur).first
	 << delim << TP::ticks((*cur).second.total)
	 << delim << (*cur).second.count
	 << delim << (*cur).second.value
	 << delim << mops
	 << std::endl;
    }
    accum_.clear();
  }
}

void Profiler::set_output(std::string const &filename)
{
  if (filename == "-")
  {
    log_.rdbuf(std::cout.rdbuf());
  }
  else
  {
    log_buf_->open(filename.c_str(), std::ios::out);
    log_.rdbuf(log_buf_.get());
  }
}

int Profiler::log(std::string const &msg, int id)
{
  static thread_local size_t indent;
  if (id > 0) --indent; // a scope is left
  size_t i = indent;
  while (i--) log_.put(' ');
  log_ << msg;
  if (id == 0) // a scope is entered
  {
    log_ << " (entered)" << std::endl;
    ++indent;
    ++count_;
  }
  else if (id > 0) log_ << " (left)" << std::endl;
  else log_ << std::endl;
  return count_;
}

char const *mode_option = "--vsip-profile-mode";
unsigned int const mode_length = sizeof("--vsip-profile-mode") - 1;
char const *output_option = "--vsip-profile-output";
unsigned int const output_length = sizeof("--vsip-profile-output") - 1;

Profiler_options::Profiler_options(int& argc, char**& argv)
    : profile_(0)
{
  int count = argc;
  char** value = argv;
  profiler_mode mode = pm_none;
  std::string filename = "-"; // default is stdout
  while (--count)
  {
    ++value;
    if (!strncmp(*value, mode_option, mode_length))
    {
      if (strlen(*value) > mode_length + 1)
      {
        char const * mode_str = &(*value)[mode_length + 1];
        if (!strcmp(mode_str, "accum"))
          mode = pm_accum;
        else if (!strcmp(mode_str, "trace"))
          mode = pm_trace;
	else if (!strcmp(mode_str, "log"))
          mode = pm_log;
      }
    }
    else if (!strncmp(*value, output_option, output_length))
    {
      if (strlen(*value) > output_length + 1)
        filename = &(*value)[output_length + 1];
    }
  }
  if (mode != pm_none)
    this->profile_ = new Profile(filename, mode);

  this->strip_args(argc, argv);
}

Profiler_options::~Profiler_options()
{
  delete this->profile_;
}

void
Profiler_options::strip_args(int& argc, char**& argv)
{
  for (int i = 1; i < argc;)
  {
    if ( !strncmp(argv[i], mode_option, mode_length) ||
         !strncmp(argv[i], output_option, output_length) )
    {
      for (int j = i; j < argc; ++j)
        argv[j] = argv[j + 1];
      --argc;
    }
    else
      ++i;
  }
}

} // namespace vsip::impl::profile
} // namespace vsip::impl
} // namespace vsip
