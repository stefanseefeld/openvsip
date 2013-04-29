//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark outer loop.

#ifndef LOOP_HPP
#define LOOP_HPP

#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <map>
#include <string>

#include <vsip/vector.hpp>
#include <vsip/math.hpp>
#include <vsip/parallel.hpp>
#include <vsip/core/memory_pool.hpp>
#include <vsip_csl/profile.hpp>

#ifdef VSIP_IMPL_SOURCERY_VPP
#  define PARALLEL_LOOP 1
#else
#  define PARALLEL_LOOP 0
#endif

#if PARALLEL_LOOP
#  include <vsip/map.hpp>
#  include <vsip/parallel.hpp>
#  define COMMUNICATOR_TYPE      vsip::impl::Communicator
#  define PROCESSOR_TYPE         vsip::processor_type
#  define DEFAULT_COMMUNICATOR() vsip::impl::default_communicator()
#  define RANK(comm)             comm.rank()
#  define BARRIER(comm)          comm.barrier()
#  define NUM_PROCESSORS()       vsip::num_processors()
#  define LOCAL(view)            view.local()
#else
#  define COMMUNICATOR_TYPE      int
#  define PROCESSOR_TYPE         int
#  define DEFAULT_COMMUNICATOR() 0
#  define RANK(comm)             0
#  define BARRIER(comm)
#  define NUM_PROCESSORS()       1
#  define LOCAL(view)            view
#endif

enum output_metric
{
  pts_per_sec,
  ops_per_sec,
  iob_per_sec,
  wiob_per_sec,
  riob_per_sec,
  all_per_sec,
  data_per_sec,
  secs_per_pt
};


enum lhs_metric
{
  lhs_pts,
  lhs_mem
};

enum bench_mode
{
  steady_mode,
  sweep_mode,
  single_mode,
  diag_mode
};

enum range_type
{
  natural_range,
  centered_range
};
  
enum prog_type
{
  geometric,
  linear,
  userfile
};



// 1 Parameter Loop
class Loop1P
{
public:

  // typedef void (TimingFunc)(int M, int loop, float* time, int calib);
  // typedef boost::function<void(unsigned, unsigned, float*)>	TimingFunctor;

  Loop1P() :
    start_	 (2),
    stop_	 (21),
    cal_	 (4),
    do_cal_      (true),
    fix_loop_    (false),
    loop_start_	 (1),
    samples_	 (1),
    goal_sec_	 (1.0),
    metric_      (pts_per_sec),
    lhs_         (lhs_pts),
    range_       (natural_range),
    progression_ (geometric),
    prog_scale_  (10),
    center_      (1000),
    note_        (0),
    user_param_  (0),
    do_prof_     (false),
    what_        (0),
    show_loop_   (false),
    show_time_   (false),
    mode_        (sweep_mode),
    m_array_     (),
    param_       (),
    pool_        (0)
  {}

  template <typename Functor>
  void sweep(Functor func);

  template <typename Functor>
  void steady(Functor func);

  template <typename Functor>
  void single(Functor func);

  template <typename Functor>
  void diag(Functor func);

  template <typename Functor>
  void operator()(Functor func);

  template <typename Functor>
  float metric(Functor& fcn, size_t M, size_t loop, float time,
	       output_metric m);

  unsigned
  m_value(unsigned i);

  // Member data.
public:
  unsigned	start_;		// loop start "i-value"
  unsigned	stop_;		// loop stop "i-value"
  int	 	cal_;		// calibration power-of-two
  bool          do_cal_;	// perform calibration
  bool          fix_loop_;	// use fixed loop count for each size.
  int	 	loop_start_;
  unsigned	samples_;
  double        goal_sec_;	// measurement goal (in seconds)
  output_metric metric_;
  lhs_metric    lhs_;
  range_type    range_;
  prog_type     progression_;
  int           prog_scale_;
  unsigned      center_;	// center value if range_ == centered
  char*         note_;
  int           user_param_;
  bool          do_prof_;
  int           what_;
  bool          show_loop_;
  bool          show_time_;
  bench_mode    mode_;
  std::vector<unsigned> m_array_;
  std::map<std::string, std::string> param_;
  vsip::impl::Memory_pool*           pool_;
};



struct Benchmark_base
{
  char const* what() { return "*unknown*"; }
  int ops_per_point(vsip::length_type) { return -1; }
  int riob_per_point(vsip::length_type) { return -1; }
  int wiob_per_point(vsip::length_type) { return -1; }
  int mem_per_point(vsip::length_type) { return -1; }

  void diag() { std::cout << "no diag\n"; }
};

template <typename Functor>
inline float
Loop1P::metric(
  Functor&      fcn,
  size_t        M,
  size_t        loop,
  float         time,
  output_metric m)
{
  if (m == pts_per_sec)
  {
    double pts = (double)M * loop;
    return pts / (time * 1e6);
  }
  else if (m == ops_per_sec)
  {
    double ops = (double)M * fcn.ops_per_point(M) * loop;
    return ops / (time * 1e6);
  }
  else if (m == iob_per_sec)
  {
    double ops = (double)M * (fcn.riob_per_point(M) + fcn.wiob_per_point(M))
                           * loop;
    return ops / (time * 1e6);
  }
  else if (m == wiob_per_sec)
  {
    double ops = (double)M * fcn.wiob_per_point(M) * loop;
    return ops / (time * 1e6);
  }
  else if (m == riob_per_sec)
  {
    double ops = (double)M * fcn.riob_per_point(M) * loop;
    return ops / (time * 1e6);
  }
  else if (m == secs_per_pt)
  {
    double pts = (double)M * loop;
    return (time * 1e6) / pts;
  }
  else
    return 0.f;
}


inline unsigned
Loop1P::m_value(unsigned i)
{
  unsigned mid = (start_ + stop_) / 2;
  switch ( progression_ )
  {
  case userfile:
    return m_array_[i];

  case linear:
    if (range_ == centered_range)
      return center_ + prog_scale_ * (i-mid);
    else 
      return prog_scale_ * i;

  case geometric:
  default:
    if (range_ == centered_range)
    {
      if (i < mid) return center_ / (1 << (mid-i));
      else        return center_ * (1 << (i-mid));
    }
    else return (1 << i);
  }
}


template <typename Functor>
inline void
Loop1P::sweep(Functor fcn)
{
  using namespace vsip;

  size_t   loop, M;
  float    time;
  double   growth;
  unsigned const n_time = samples_;

  COMMUNICATOR_TYPE& comm  = DEFAULT_COMMUNICATOR();
  PROCESSOR_TYPE     rank  = RANK(comm);
  PROCESSOR_TYPE     nproc = NUM_PROCESSORS();

  std::vector<float> mtime(n_time);

#if PARALLEL_LOOP
  Vector<float, Dense<1, float, row1_type, Map<> > >
      dist_time(nproc, Map<>(nproc));
  Vector<float, Dense<1, float, row1_type, Replicated_map<1> > > glob_time(nproc);
#else
  Vector<float, Dense<1, float, row1_type> > dist_time(nproc);
  Vector<float, Dense<1, float, row1_type> > glob_time(nproc);
#endif

  loop = loop_start_;
  M    = this->m_value(cal_);

  // calibrate --------------------------------------------------------
  if (do_cal_ && !fix_loop_)
  {
    float factor;
    float factor_thresh = 1.05;
    
    size_t old_loop;
    do 
    {
      old_loop = loop;
      BARRIER(comm);
      {
	vsip::impl::Memory_pool* cur_pool = vsip::impl::default_pool;
	if (pool_) vsip::impl::default_pool = pool_;
	fcn(M, loop, time);
	vsip::impl::default_pool = cur_pool;
      }
      BARRIER(comm);

      LOCAL(dist_time).put(0, time);
      glob_time = dist_time;

      Index<1> idx;

      time = maxval(LOCAL(glob_time), idx);

      if (time <= 0.01) time = 0.01;

      factor = goal_sec_ / time;
      loop = (size_t)(factor * loop);
      // printf("%d: time: %f  factor: %f  loop: %d\n", rank,time,factor,loop);
      if ( loop == 0 ) 
        loop = 1; 
    } while (factor >= factor_thresh && loop > old_loop);
  }

  if (rank == 0)
  {
    if (metric_ == data_per_sec)
    {
      std::cout << "what," << fcn.what() << "," << what_ << std::endl;
      std::cout << "nproc," << nproc << std::endl;
      std::cout << "size,med,min,max,mem/pt,ops/pt,riob/pt,wiob/pt,loop,time"
		<< std::endl;
    }
    else
    {
    std::cout << "# what             : " << fcn.what() << " (" << what_ << ")\n";
    std::cout << "# nproc            : " << nproc << "\n";
    std::cout << "# ops_per_point(1) : " << fcn.ops_per_point(1) << '\n';
    std::cout << "# riob_per_point(1): " << fcn.riob_per_point(1) << '\n';
    std::cout << "# wiob_per_point(1): " << fcn.wiob_per_point(1) << '\n';
    std::cout << "# metric           : " <<
      (metric_ == pts_per_sec  ? "pts_per_sec" :
       metric_ == ops_per_sec  ? "ops_per_sec" :
       metric_ == iob_per_sec  ? "iob_per_sec" :
       metric_ == riob_per_sec ? "riob_per_sec" :
       metric_ == wiob_per_sec ? "wiob_per_sec" :
       metric_ == data_per_sec ? "data_per_sec" :
       metric_ == secs_per_pt  ? "usecs_per_pt" :
       "*unknown*") << '\n';
    if (this->note_)
      std::cout << "# note: " << this->note_ << '\n';
    std::cout << "# start_loop       : " << static_cast<unsigned long>(loop) << '\n';
    }
  }

#if PARALLEL_LOOP
  if (this->do_prof_)
    vsip_csl::profile::prof->set_mode(vsip_csl::profile::pm_accum);
#endif

  // for real ---------------------------------------------------------
  for (unsigned i=start_; i<=stop_; i++)
  {
    M = this->m_value(i);

    for (unsigned j=0; j<n_time; ++j)
    {
      BARRIER(comm);
      {
	vsip::impl::Memory_pool* cur_pool = vsip::impl::default_pool;
	if (pool_) vsip::impl::default_pool = pool_;
	fcn(M, loop, time);
	vsip::impl::default_pool = cur_pool;
      }
      BARRIER(comm);

      LOCAL(dist_time).put(0, time);
      glob_time = dist_time;

      Index<1> idx;

      mtime[j] = maxval(LOCAL(glob_time), idx);
    }

#if PARALLEL_LOOP
    if (this->do_prof_)
    {
      char     filename[256];
      sprintf(filename, "vprof.%lu-%lu.out", (unsigned long) M,
	      (unsigned long)vsip::local_processor());
      vsip_csl::profile::prof->dump(filename);
    }
#endif

    std::sort(mtime.begin(), mtime.end());


    if (rank == 0)
    {
      size_t L;
      
      if (this->lhs_ == lhs_mem)
	L = M * fcn.mem_per_point(M);
      else // (this->lhs_ == lhs_pts)
	L = M;

      if (this->metric_ == all_per_sec)
	printf("%7ld %f %f %f", (unsigned long) L,
	       this->metric(fcn, M, loop, mtime[(n_time-1)/2], pts_per_sec),
	       this->metric(fcn, M, loop, mtime[(n_time-1)/2], ops_per_sec),
	       this->metric(fcn, M, loop, mtime[(n_time-1)/2], iob_per_sec));
      else if (this->metric_ == data_per_sec)
	printf("%ld,%f,%f,%f,%f,%f,%f,%f,%ld,%f",
	       (unsigned long) L,
	       this->metric(fcn, M, loop, mtime[(n_time-1)/2], pts_per_sec),
	       this->metric(fcn, M, loop, mtime[n_time-1],     pts_per_sec),
	       this->metric(fcn, M, loop, mtime[0],            pts_per_sec),
	       (float)fcn.mem_per_point(M),
	       (float)fcn.ops_per_point(M),
	       (float)fcn.riob_per_point(M),
	       (float)fcn.wiob_per_point(M),
	       (unsigned long)loop,
	       mtime[(n_time-1)/2]);
      else if (n_time > 1)
	// Note: max time is min op/s, and min time is max op/s
	printf("%7lu %f %f %f", (unsigned long) L,
	       this->metric(fcn, M, loop, mtime[(n_time-1)/2], metric_),
	       this->metric(fcn, M, loop, mtime[n_time-1],     metric_),
	       this->metric(fcn, M, loop, mtime[0],            metric_));
      else
	printf("%7lu %f", (unsigned long) L,
	       this->metric(fcn, M, loop, mtime[0], metric_));
      if (this->show_loop_)
	printf("  %8lu", (unsigned long)loop);
      if (this->show_time_)
	printf("  %f", mtime[(n_time-1)/2]);
      printf("\n");
      fflush(stdout);
    }

    time = mtime[(n_time-1)/2];

    if (!fix_loop_)
    {
      growth = 2.0 * fcn.ops_per_point(2*M) / fcn.ops_per_point(M);
      time = time * growth;

      float factor = goal_sec_ / time;
      if (factor < 1.0) factor += 0.1 * (1.0 - factor);
      loop = (int)(factor * loop);

      if (loop < 1) loop = 1;
    }
  }
}



template <typename Functor>
void
Loop1P::steady(Functor fcn)
{
  using namespace vsip;
  using vsip::impl::Memory_pool;

  size_t   loop, M;
  float    time;

  COMMUNICATOR_TYPE& comm  = DEFAULT_COMMUNICATOR();
  PROCESSOR_TYPE     rank  = RANK(comm);
  PROCESSOR_TYPE     nproc = NUM_PROCESSORS();

#if PARALLEL_LOOP
  Vector<float, Dense<1, float, row1_type, Map<> > >
    dist_time(nproc, Map<>(nproc));
  Vector<float, Dense<1, float, row1_type, Replicated_map<1> > > glob_time(nproc);
#else
  Vector<float, Dense<1, float, row1_type> > dist_time(nproc);
  Vector<float, Dense<1, float, row1_type> > glob_time(nproc);
#endif

  loop = loop_start_;

  if (rank == 0)
  {
    std::cout << "# what             : " << fcn.what() << " (" << what_ << ")\n";
    std::cout << "# nproc            : " << nproc << "\n";
    std::cout << "# ops_per_point(1) : " << fcn.ops_per_point(1) << '\n';
    std::cout << "# riob_per_point(1): " << fcn.riob_per_point(1) << '\n';
    std::cout << "# wiob_per_point(1): " << fcn.wiob_per_point(1) << '\n';
    std::cout << "# metric           : " <<
      (metric_ == pts_per_sec  ? "pts_per_sec" :
       metric_ == ops_per_sec  ? "ops_per_sec" :
       metric_ == iob_per_sec  ? "iob_per_sec" :
       metric_ == riob_per_sec ? "riob_per_sec" :
       metric_ == wiob_per_sec ? "wiob_per_sec" :
       metric_ == data_per_sec ? "data_per_sec" :
       metric_ == secs_per_pt  ? "usecs_per_pt" :
       "*unknown*") << '\n';
    if (this->note_)
      std::cout << "# note: " << this->note_ << '\n';
    std::cout << "# start_loop       : " << static_cast<unsigned long>(loop) << '\n';
  }

#if PARALLEL_LOOP
  if (this->do_prof_)
    vsip_csl::profile::prof->set_mode(vsip_csl::profile::pm_accum);
#endif

  // for real ---------------------------------------------------------
  while (1)
  {
    M = (1 << start_);

    BARRIER(comm);
    {
      Memory_pool* cur_pool = vsip::impl::default_pool;
      if (pool_) vsip::impl::default_pool = pool_;
      fcn(M, loop, time);
      vsip::impl::default_pool = cur_pool;
    }
    BARRIER(comm);

    LOCAL(dist_time).put(0, time);
    glob_time = dist_time;

    Index<1> idx;

    time = maxval(LOCAL(glob_time), idx);

#if 0
    if (this->do_prof_)
    {
      std::ostringstream oss;
      oss << "vprof." << static_cast<unsigned long>(M) << ".out";
      vsip_csl::profile::prof->dump(oss.str());
    }
#endif

    if (rank == 0)
    {
      if (this->metric_ == all_per_sec)
	printf("%7ld %f %f %f", (unsigned long) M,
	       this->metric(fcn, M, loop, time, pts_per_sec),
	       this->metric(fcn, M, loop, time, ops_per_sec),
	       this->metric(fcn, M, loop, time, iob_per_sec));
      else
	printf("%7lu %f", (unsigned long) M,
	       this->metric(fcn, M, loop, time, metric_));
      if (this->show_loop_)
	printf("  %8lu", (unsigned long)loop);
      if (this->show_time_)
	printf("  %f", time);
      printf("\n");
      fflush(stdout);
    }

    float factor = goal_sec_ / time;
    if (factor < 1.0) factor += 0.1 * (1.0 - factor);
    loop = (int)(factor * loop);

    if (loop < 1) loop = 1;
  }
}



template <typename Functor>
inline void
Loop1P::single(Functor fcn)
{
  float  time;
  size_t loop = 1;
  size_t M = this->m_value(cal_);

  COMMUNICATOR_TYPE& comm  = DEFAULT_COMMUNICATOR();

  BARRIER(comm);
  {
    vsip::impl::Memory_pool* cur_pool = vsip::impl::default_pool;
    if (pool_) vsip::impl::default_pool = pool_;
    fcn(M, loop, time);
    vsip::impl::default_pool = cur_pool;
  }
  BARRIER(comm);
}



template <typename Functor>
inline void
Loop1P::diag(Functor fcn)
{
  COMMUNICATOR_TYPE& comm  = DEFAULT_COMMUNICATOR();

  BARRIER(comm);
  fcn.diag();
  BARRIER(comm);
}



template <typename Functor>
inline void
Loop1P::operator()(
  Functor fcn)
{
  if (mode_ == steady_mode)
    this->steady(fcn);
  else if (mode_ == single_mode)
    this->single(fcn);
  else if (mode_ == diag_mode)
    this->diag(fcn);
  else
    this->sweep(fcn);
}

#endif
