/* Copyright (c) 2007, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for Fast Convolution (Cbe specific).

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/diagnostics.hpp>
#include <vsip/opt/cbe/ppu/fastconv.hpp>

#include <vsip_csl/error_db.hpp>

#include "benchmarks.hpp"
#include "alloc_block.hpp"
#include "fastconv.hpp"

#if !VSIP_IMPL_CBE_SDK
#  error VSIP_IMPL_CBE_SDK not set
#endif

using namespace vsip;
using vsip_csl::error_db;

/***********************************************************************
  Common definitions
***********************************************************************/

struct ImplCbe;		// interleaved fast-convolution on Cell
template <storage_format_type S, bool single_fc>
struct ImplCbe_ip;	// interleaved fast-convolution on Cell, in-place
template <storage_format_type S>
struct ImplCbe_op;	// interleaved fast-convolution on Cell, out-of-place
struct Impl4;		// Single-line fast-convolution
template <bool transform_replica>
struct ImplCbe_multi;	// interleaved fast-convolution on Cell, multiple



/***********************************************************************
  ImplCbe: interleaved fast-convolution on Cell

  Three versions of the benchmark case are provided:

  ImplCbe: in-place, distributed, split/interleaved format fixed
           to be library's preferred format.

  ImplCbe_ip: in-place, non-distributed, split/interleaved controllable.

  ImplCbe_op: out-of-place, non-distributed, split/interleaved
           controllable.

  ImplCbe_multi: in-place, non-distributed, split/interleaved
           as configured, multiple coefficient vectors (i.e. a matrix),
           pre-transforming coeffs to frequency space is controllable.
***********************************************************************/
bool        use_huge_pages_ = true;

template <typename T>
struct t_fastconv_base<T, ImplCbe> : fastconv_ops
{
  static length_type const num_args = 1;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
#if PARALLEL_FASTCONV
    typedef Replicated_map<1>               map1_type;
    typedef Map<Block_dist, Whole_dist> map2_type;

    processor_type np = num_processors();
    map1_type map1;
    map2_type map2 = map2_type(Block_dist(np), Whole_dist());
#else
    typedef Local_map  map1_type;
    typedef Local_map  map2_type;

    map1_type map1;
    map2_type map2;
#endif
    static storage_format_type const storage_format = impl::dense_complex_format;

    typedef typename Alloc_block<1, T, storage_format, map1_type>::block_type
	    block1_type;
    typedef typename Alloc_block<2, T, storage_format, map2_type>::block_type
	    block2_type;

    typedef Vector<T, block1_type> view1_type;
    typedef Matrix<T, block2_type> view2_type;

    typedef impl::cbe::Fastconv<1, T, storage_format>   fconv_type;

    block1_type* repl_block;
    block2_type* data_block;

    repl_block = alloc_block<1, T, storage_format>(nrange, mem_addr_, 0x0000000,
					       map1);
    data_block = alloc_block<2, T, storage_format>(Domain<2>(npulse, nrange),
					       mem_addr_, nrange*sizeof(T),
					       map2);

    { // Use scope to control lifetime of view.

    // Create the data cube.
    view2_type data(*data_block);
    
    // Create the pulse replica
    view1_type replica(*repl_block);

    // Create Fast Convolution object
    fconv_type fconv(replica, nrange);

    vsip_csl::profile::Timer t1;

    t1.start();
    for (index_type l=0; l<loop; ++l)
      fconv(data, data);
    t1.stop();

    time = t1.delta();
    }

    // Delete blocks after view has gone out of scope.  If we delete
    // the blocks while the views are still live, they will corrupt
    // memory when they try to decrement the blocks' reference counts.

    delete repl_block;
    delete data_block;

  }

  t_fastconv_base()
    : mem_addr_(0),
      pages_   (9)
  {
    char const* mem_file = "/huge/fastconv.bin";

    if (use_huge_pages_)
      mem_addr_ = open_huge_pages(mem_file, pages_);
    else
      mem_addr_ = 0;
  }

  char*        mem_addr_;
  unsigned int pages_;
};




template <typename T, storage_format_type S, bool single_fc>
struct t_fastconv_base<T, ImplCbe_ip<S, single_fc> > : fastconv_ops
{

  static length_type const num_args = 1;

  typedef impl::cbe::Fastconv<1, T, S>   fconv_type;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    typedef Local_map map_type;
    map_type map;
    typedef typename Alloc_block<1, T, S, map_type>::block_type
	    block1_type;
    typedef typename Alloc_block<2, T, S, map_type>::block_type
	    block2_type;

    block2_type* data_block;
    block1_type* repl_block;

    repl_block = alloc_block<1, T, S>(nrange, mem_addr_, 0x0000000, map);
    data_block = alloc_block<2, T, S>(Domain<2>(npulse, nrange),
				      mem_addr_, nrange*sizeof(T),
				      map);

    typedef Matrix<T, block2_type> view_type;
    typedef Vector<T, block1_type> replica_view_type;

    {
    // Create the data cube.
    view_type         data(*data_block);
    // Create the pulse replica
    replica_view_type replica(*repl_block);
    
    vsip_csl::profile::Timer t1;

    if (single_fc)
    {
      // Reuse single fastconv object

      // Create Fast Convolution object
      fconv_type fconv(replica, nrange);

      t1.start();
      for (index_type l=0; l<loop; ++l)
	fconv(data, data);
      t1.stop();

      time = t1.delta();
    }
    else
    {
      // Use multiple fastconv objects

      t1.start();
      for (index_type l=0; l<loop; ++l)
      {
	// Create Fast Convolution object
	fconv_type fconv(replica, nrange);
	fconv(data, data);
      }
      t1.stop();
    }

      time = t1.delta();
    }

    delete repl_block;
    delete data_block;
  }

  t_fastconv_base()
    : mem_addr_ (0)
    , pages_    (9)
  {
    char const* mem_file = "/huge/fastconv.bin";

    if (use_huge_pages_)
      mem_addr_ = open_huge_pages(mem_file, pages_);
    else
      mem_addr_ = 0;
  }

// Member data.
  char*        mem_addr_;
  unsigned int pages_;
};



template <typename T, storage_format_type S>
struct t_fastconv_base<T, ImplCbe_op<S> > : fastconv_ops
{
  typedef Layout<1, row1_type, vsip::dense, S> LP1;
  typedef Layout<2, row2_type, vsip::dense, S> LP2;
  typedef impl::Strided<1, T, LP1, Local_map> block1_type;
  typedef impl::Strided<2, T, LP2, Local_map> block2_type;

  static length_type const num_args = 2;

  typedef impl::cbe::Fastconv<1, T, S>   fconv_type;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    Rand<T> gen(0, 0);

    // Create the data cube.
    Matrix<T, block2_type> in (npulse, nrange, T());
    Matrix<T, block2_type> out(npulse, nrange, T());
    in = gen.randu(npulse, nrange);

    // Create the pulse replica
    Vector<T, block1_type> replica(nrange, T());
    replica.put(0, T(1));


    // Create Fast Convolution object
    fconv_type fconv(replica, nrange);

    vsip_csl::profile::Timer t1;
    t1.start();
    for (index_type l=0; l<loop; ++l)
      fconv(in, out);
    t1.stop();

    time = t1.delta();


    // Check result.
#if 0
    // Ideally we would do a full check, using FFT and vmul.
    // However, those also use the SPE and will bump the fastconv
    // kernel out.
    typedef Fft<const_Vector, T, T, fft_fwd, by_reference> for_fft_type;
    typedef Fft<const_Vector, T, T, fft_inv, by_reference> inv_fft_type;

    Vector<T> f_replica(nrange, T());
    Vector<T> chk      (nrange, T());

    for_fft_type for_fft(Domain<1>(nrange), 1.0);
    inv_fft_type inv_fft(Domain<1>(nrange), 1.0/(nrange));

    for_fft(replica, f_replica);
    for (index_type p=0; p<npulse; ++p)
    {
      for_fft(in.row(p), chk);
      chk *= f_replica;
      inv_fft(chk);
      test_assert(error_db(chk, out.row(p)) < -100);
    }
#else
    // Instead, we use a simple identity kernel and check that
    // in == out.
    test_assert(error_db(in, out) < -100);
#endif
  }
};





template <typename T,
          bool     transform_replica>
struct t_fastconv_base<T, ImplCbe_multi<transform_replica> > : fastconv_ops
{
  static length_type const num_args = 1;

  static storage_format_type const storage_format = impl::dense_complex_format;
  typedef vsip::impl::cbe::Fastconv<2, T, storage_format>   fconvm_type;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    typedef typename Alloc_block<2, T, storage_format, Local_map>::block_type  block_type;
    block_type* data_block;
    block_type* repl_block;

    Local_map map;
    repl_block = alloc_block<2, T, storage_format>(Domain<2>(npulse, nrange),
		                                 mem_addr_, 0x0000000,
						 map);
    data_block = alloc_block<2, T, storage_format>(Domain<2>(npulse, nrange),
						 mem_addr_, nrange*sizeof(T),
						 map);
    {
      typedef Matrix<T, block_type> view_type;
      // Create the data cube.
      view_type data(*data_block);
      // Create the pulse replicas
      view_type replica(*repl_block);
      // Note: we treat the replica as if it were in the frequency
      // domain already.  Actually performing an FFT would push
      // the compute kernel out of SPE memory.


      // Create Fast Convolution object
      fconvm_type fconvm(replica, nrange, transform_replica);

      vsip_csl::profile::Timer t1;
      t1.start();
      for (index_type l=0; l<loop; ++l)
        fconvm(data, data);
      t1.stop();

      time = t1.delta();
    }

    delete repl_block;
    delete data_block;
  }

  t_fastconv_base()
    : mem_addr_ (0)
    , pages_    (9)
  {
    char const* mem_file = "/huge/fastconv.bin";

    if (use_huge_pages_)
      mem_addr_ = open_huge_pages(mem_file, pages_);
    else
      mem_addr_ = 0;
  }

// Member data.
  char*        mem_addr_;
  unsigned int pages_;
};


/***********************************************************************
  Impl4: Single expression fast-convolution.
         (Identical to fastconv.cpp, but with huge page support).
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl4> : fastconv_ops
{
  static length_type const num_args = 1;

#if PARALLEL_FASTCONV
  typedef Replicated_map<1>                    map1_type;
  typedef Map<Block_dist, Whole_dist>      map2_type;
#else
  typedef Local_map  map1_type;
  typedef Local_map  map2_type;
#endif
  static storage_format_type const storage_format = impl::dense_complex_format;

  typedef typename Alloc_block<1, T, storage_format, map1_type>::block_type
	  block1_type;
  typedef typename Alloc_block<2, T, storage_format, map2_type>::block_type
	  block2_type;

  typedef Vector<T, block1_type> view1_type;
  typedef Matrix<T, block2_type> view2_type;

  // static int const no_times = 0; // FFTW_PATIENT
  static int const no_times = 15; // not > 12 = FFT_MEASURE
    
  typedef Fftm<T, T, row, fft_fwd, by_value, no_times>
               for_fftm_type;
  typedef Fftm<T, T, row, fft_inv, by_value, no_times>
	       inv_fftm_type;


  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
#if PARALLEL_FASTCONV
    processor_type np = num_processors();
    map1_type map1;
    map2_type map2 = map2_type(np, 1);
#else
    map1_type map1;
    map2_type map2;
#endif

    block1_type* repl_block;
    block2_type* data_block;

    repl_block = alloc_block<1, T, storage_format>(nrange, mem_addr_, 0x0000000,
					       map1);
    data_block = alloc_block<2, T, storage_format>(Domain<2>(npulse, nrange),
					       mem_addr_, nrange*sizeof(T),
					       map2);

    { // Use scope to control lifetime of view.

    // Create the data cube.
    view2_type data(*data_block);
    view2_type chk(npulse, nrange, map2);
    
    // Create the pulse replica
    view1_type replica(*repl_block);

    // Create the FFT objects.
    for_fftm_type for_fftm(Domain<2>(npulse, nrange), 1.0);
    inv_fftm_type inv_fftm(Domain<2>(npulse, nrange), 1.0/nrange);

    // Initialize
    data    = T();
    replica = T();


    // Before fast convolution, convert the replica into the
    // frequency domain
    // for_fft(replica);
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      data = inv_fftm(vmmul<0>(replica, for_fftm(data)));
    }
    t1.stop();

    // CHECK RESULT
#if 0
    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times>
	  	for_fft_type;

    Rand<T> gen(0, 0);
    for_fft_type for_fft(Domain<1>(nrange), 1.0);

    data = gen.randu(npulse, nrange);
    replica.put(0, T(1));
    for_fft(replica);

    chk = inv_fftm(vmmul<0>(replica, for_fftm(data)));

    double error = error_db(data, chk);

    test_assert(error < -100);
#endif

    time = t1.delta();
    }

    // Delete blocks after view has gone out of scope.  If we delete
    // the blocks while the views are still live, they will corrupt
    // memory when they try to decrement the blocks' reference counts.

    delete repl_block;
    delete data_block;
  }

  void diag()
  {
#if PARALLEL_FASTCONV
    processor_type np = num_processors();
    map2_type map = map2_type(np, 1);
#else
    map2_type map;
#endif

    length_type npulse = 16;
    length_type nrange = 2048;

    // Create the data cube.
    view2_type data(npulse, nrange, map);

    // Create the pulse replica
    view1_type replica(nrange);

    // Create the FFT objects.
    for_fftm_type for_fftm(Domain<2>(npulse, nrange), 1.0);
    inv_fftm_type inv_fftm(Domain<2>(npulse, nrange), 1.0/nrange);

    vsip_csl::assign_diagnostics(data, inv_fftm(vmmul<0>(replica, for_fftm(data))));
  }

  t_fastconv_base()
    : mem_addr_(0),
      pages_   (9)
  {
    char const* mem_file = "/huge/fastconv.bin";

    if (use_huge_pages_)
      mem_addr_ = open_huge_pages(mem_file, pages_);
    else
      mem_addr_ = 0;
  }

  char*        mem_addr_;
  unsigned int pages_;
};



/***********************************************************************
  Benchmark Driver
***********************************************************************/

void
defaults(Loop1P& loop)
{
  loop.cal_        = 0;
  loop.start_      = 0;
  loop.stop_       = 12;
  loop.loop_start_ = 10;
  loop.user_param_ = 2048;
}



int
test(Loop1P& loop, int what)
{
  using namespace vsip;
  typedef complex<float> T;

  length_type param1 = loop.user_param_;
  switch (what)
  {
  case  1: loop(t_fastconv_rf<T, Impl4>(param1)); break;
  case  2: loop(t_fastconv_rf<T, ImplCbe>(param1)); break;

  case 11: loop(t_fastconv_rf<T, ImplCbe_op<interleaved_complex> >(param1));break;
  case 12: loop(t_fastconv_rf<T, ImplCbe_ip<interleaved_complex, true> >(param1));break;
  case 13: loop(t_fastconv_rf<T, ImplCbe_ip<interleaved_complex, false> >(param1));break;

  case 21: loop(t_fastconv_rf<T, ImplCbe_op<split_complex> >(param1));break;
  case 22: loop(t_fastconv_rf<T, ImplCbe_ip<split_complex, true> >(param1));break;
  case 23: loop(t_fastconv_rf<T, ImplCbe_ip<split_complex, false> >(param1));break;

  case 32: loop(t_fastconv_rf<T, ImplCbe_multi<true> >(param1));break;
  case 42: loop(t_fastconv_rf<T, ImplCbe_multi<false> >(param1));break;

  case 0:
    std::cout
      << "fastconv -- fast convolution benchmark for Cell BE\n"
      << " Sweeping pulse size:\n"
      << "    -1 -- IP, native complex, distributed, single-expr\n"
      << "    -2 -- IP, native complex, distributed, Fastconv object\n"
      << "\n"
      << "   -11 -- OP, inter complex,  non-dist\n"
      << "   -12 -- IP, inter complex,  non-dist, single FC\n"
      << "   -13 -- IP, inter complex,  non-dist, multi FC\n"
      << "\n"
      << "   -21 -- OP, split complex,  non-dist\n"
      << "   -22 -- IP, split complex,  non-dist\n"
      << "   -23 -- IP, split complex,  non-dist, multi FC\n"
      << "\n"
      << "   -32 -- Multiple coeff vectors in time domain, IP, native complex, non-dist, single FC\n"
      << "   -42 -- Multiple coeff vectors in freq domain, IP, native complex, non-dist, single FC\n"
      ;

  default: return 0;
  }
  return 1;
}
