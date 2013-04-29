/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for task switching

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/error_db.hpp>

#include "benchmarks.hpp"
#include "fastconv.hpp"

using namespace vsip;

struct Impl1;		// Alternating fftm/vmmul
struct Impl2;		// Alternate */+

bool check = true;
bool show_overhead = true;



/***********************************************************************
  Impl1: alternating fftm/vmmul
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl1> : fastconv_ops
{
  static length_type const num_args = 2;

  float ops(vsip::length_type npulse, vsip::length_type nrange) 
  {
    float fft_ops = 5 * nrange * std::log((float)nrange) / std::log(2.f);
    float tot_ops = 1 * npulse * fft_ops + 6 * npulse * nrange;
    return tot_ops;
  }

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
#if PARALLEL_FASTCONV
    typedef Map<Block_dist, Whole_dist>      map_type;
    typedef Dense<2, T, row2_type, map_type> block_type;
    typedef Matrix<T, block_type>            view_type;

    typedef Dense<1, T, row1_type, Replicated_map<1> > replica_block_type;
    typedef Vector<T, replica_block_type>          replica_view_type;

    processor_type np = num_processors();
    map_type map = map_type(Block_dist(np), Whole_dist());

    // Create the data cube.
    view_type data(npulse, nrange, map);
    view_type tmp(npulse, nrange, map);
#else
    typedef Matrix<T>  view_type;
    typedef Vector<T>  replica_view_type;

    view_type data(npulse, nrange);
    view_type tmp(npulse, nrange);
#endif
    
    // Create the pulse replica
    replica_view_type replica(nrange);

    // int const no_times = 0; // FFTW_PATIENT
    int const no_times = 15; // not > 12 = FFT_MEASURE
    
    typedef Fftm<T, T, row, fft_fwd, by_reference, no_times>
	  	for_fftm_type;

    // Create the FFT objects.
    for_fftm_type for_fftm(Domain<2>(npulse, nrange), 1.0);

    // Initialize
    data    = T();
    replica = T();


    // Before fast convolution, convert the replica into the
    // frequency domain
    // for_fft(replica);
    
    vsip_csl::profile::Timer t1, t2, t3;
    
    for_fftm(data, tmp);
    tmp = vmmul<0>(replica, tmp);
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      // Perform half-fast convolution:
      for_fftm(data, tmp);
      tmp = vmmul<0>(replica, tmp);
    }
    t1.stop();

    for_fftm(data, tmp);
    t2.start();
    for (index_type l=0; l<loop; ++l)
    {
      // Perform fast convolution:
      for_fftm(data, tmp);
    }
    t2.stop();

    tmp = vmmul<0>(replica, tmp);
    t3.start();
    for (index_type l=0; l<loop; ++l)
    {
      // Perform fast convolution:
      tmp = vmmul<0>(replica, tmp);
    }
    t3.stop();

    if (check)
    {
      data = T(1);
      replica = T(2);

      for_fftm(data, tmp);
      for (index_type p=0; p<npulse; ++p)
      {
	if (!equal(tmp.get(p, 0), T(nrange * 1)))
	{
	  std::cout << "miscompare-1: pulse " << p << std::endl
		    << "          : got " << tmp.get(p, 0) << std::endl
		    << "          : expected " << T(nrange * 1) << std::endl
	    ;
	}
	test_assert(equal(tmp.get(p, 0), T(nrange * 1)));
      }
      tmp = vmmul<0>(replica, tmp);

      for (index_type p=0; p<npulse; ++p)
      {
	if (!equal(tmp.get(p, 0), T(nrange * 2)))
	{
	  std::cout << "miscompare-2: pulse " << p << std::endl
		    << "          : got " << tmp.get(p, 0) << std::endl
		    << "          : expected " << T(nrange * 2) << std::endl
	    ;
	}
	test_assert(equal(tmp.get(p, 0), T(nrange * 2)));
      }
    }

    float overhead = (t1.delta() - (t2.delta() + t3.delta()) ) / loop;
    printf("overhead: %f s (fft/vmmul: %7.4f s)\n",
	   overhead, t1.delta());

    // CHECK RESULT
    time = t1.delta();
  }
};



/***********************************************************************
  Impl2: alternating vadd/vmul
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl2> : fastconv_ops
{
  static length_type const num_args = 2;

  float ops(vsip::length_type npulse, vsip::length_type nrange) 
  {
    return (vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add)
      * npulse * nrange;
  }

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    typedef Map<Block_dist, Whole_dist>      map_type;
    typedef Dense<2, T, row2_type, map_type> block_type;
    typedef Matrix<T, block_type>            view_type;

    typedef Dense<1, T, row1_type, Replicated_map<1> > replica_block_type;
    typedef Vector<T, replica_block_type>          replica_view_type;

    processor_type np = num_processors();
    map_type map = map_type(Block_dist(np), Whole_dist());

    // Create the data cube.
    view_type A1(npulse, nrange, map);
    view_type A2(npulse, nrange, map);
    view_type B (npulse, nrange, map);
    view_type C (npulse, nrange, map);

    Matrix<float> Ax(npulse, nrange);
    Matrix<float> Bx(npulse, nrange, 3.f);
    Matrix<float> Cx(npulse, nrange, 4.f);
    
    // Initialize
    B = T(3);
    C = T(4);


    // Step 1: measure alternating */+
    vsip_csl::profile::Timer t1, t2, t3;
    
    Ax = Bx * Cx;
    A1 = B * C;
    A2 = B + C;
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      A1 = B * C;
      A2 = B + C;
    }
    t1.stop();

    // Step 2: measure * only
    A1 = B * C;
    t2.start();
    for (index_type l=0; l<loop; ++l)
    {
      A1 = B * C;
    }
    t2.stop();

    // Step 2: measure + only
    A2 = B + C;
    t3.start();
    for (index_type l=0; l<loop; ++l)
    {
      A2 = B + C;
    }
    t3.stop();

    if (check)
    {
      A1 = B * C;
      A2 = B + C;

      for (index_type p=0; p<npulse; ++p)
	for (index_type r=0; r<nrange; ++r)
	{
	  test_assert(equal(A1.get(p, r), B.get(p, r) * C.get(p, r)));
	  test_assert(equal(A2.get(p, r), B.get(p, r) + C.get(p, r)));
	}
    }

    float overhead = (t1.delta() - (t2.delta() + t3.delta()) ) / loop;
    if (show_overhead)
      printf("overhead: %f s (vadd/vmul: %7.4f s)\n", overhead, t1.delta());

    // CHECK RESULT
    time = t1.delta();
  }
};



/***********************************************************************
  Benchmark Driver
***********************************************************************/

void
defaults(Loop1P& loop)
{
  loop.cal_        = 4;
  loop.start_      = 4;
  loop.stop_       = 12;
  loop.loop_start_ = 1;

  loop.param_["rows"]  = "64";
  loop.param_["size"]  = "2048";
  loop.param_["check"] = "1";
  loop.param_["over"]  = "1";
}



int
test(Loop1P& loop, int what)
{
  length_type rows  = atoi(loop.param_["rows"].c_str());
  length_type size  = atoi(loop.param_["size"].c_str());
  check             = (loop.param_["check"] == "1" ||
		       loop.param_["check"] == "y");
  show_overhead     = (loop.param_["over"] == "1" ||
		       loop.param_["over"] == "y");

  std::cout << "rows: " << rows << "  size: " << size 
	    << "  check: " << (check ? "yes" : "no")
	    << std::endl;

  switch (what)
  {
  case  1: loop(t_fastconv_pf<complex<float>, Impl1>(rows)); break;
  case  2: loop(t_fastconv_pf<complex<float>, Impl2>(rows)); break;

  case 11: loop(t_fastconv_rf<complex<float>, Impl1>(size)); break;
  case 12: loop(t_fastconv_rf<complex<float>, Impl2>(size)); break;


  // case 101: loop(t_fastconv_pf<complex<float>, Impl3>(param1)); break;

  case 0:
    std::cout
      << "task_switch -- Task switch overhead benchmark\n"
      << " Recommended:\n"
      << "   task_switch -11 -p:size 2048 -steady 10 -lat -show_loop -ms 200\n"
      << "\n"
      << " Sweeping pulse size:\n"
      << "   -1 -- Alternate Fftm/vmmul kernels\n"
      << "   -2 -- Alternate */+ kernels\n"
      << "\n"
      << " Parameters (for sweeping convolution size, cases 1 through 10)\n"
      << "  -p:rows ROWS -- set number of pulses (default 64)\n"
      << "\n"
      << " Sweeping number of pulses:\n"
      << "  -11 -- Alternate Fftm/vmmul kernels\n"
      << "  -12 -- Alternate */+ kernels\n"
      << "\n"
      << " Parameters (for sweeping number of convolutions, cases 11 through 20)\n"
      << "  -p:size SIZE -- size of pulse (default 2048)\n"
      << "\n"
      << " Common Parameters\n"
      << "  -p:check {0,n}|{1,y} -- check results (default 'y')\n"
      ;

  default: return 0;
  }
  return 1;
}
