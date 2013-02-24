/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    firbank.cpp
    @author  Don McCoy
    @date    2006-01-26
    @brief   VSIPL++ Library: FIR Filter Bank - High Performance 
             Embedded Computing (HPEC) Kernel-Level Benchmarks

    This benchmark demonstrates one of the fundamental operations used
    in signal processing applications, the finite impulse response (FIR)
    filter.  Two algorithms, one that works in time-domain and one that
    uses FFT's, are implemented here.  The FFT-based fast convolution
    is usually more efficient for larger filters.  
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <string>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/map.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip_csl/load_view.hpp>
#include <vsip_csl/test.hpp>
#include "benchmarks.hpp"

using namespace vsip;
using vsip_csl::Load_view;
using vsip_csl::view_equal;

#ifdef VSIP_IMPL_SOURCERY_VPP
#  define PARALLEL_FIRBANK 1
#else
#  define PARALLEL_FIRBANK 0
#endif


/***********************************************************************
  Common definitions
***********************************************************************/

template <typename T,
	  typename ImplTag>
struct t_firbank_base;

struct ImplFull;	   // Time-domain convolution using Fir class
struct ImplFast;	   // Fast convolution using FFTs
struct ImplExpr;	   // Fast convolution using FFTMs


template <typename T>
struct t_local_view
{
  template <
    typename Block1,
    typename Block2,
    typename Block3,
    typename Block4
    >
  void verify_views(
    Matrix<T, Block1> inputs,
    Matrix<T, Block2> filters,
    Matrix<T, Block3> outputs,
    Matrix<T, Block4> expected )
  {
#if PARALLEL_FIRBANK
    // Check that dim-1 is not distributed
    typename Block1::map_type const& map1 = inputs.block().map();
    typename Block2::map_type const& map2 = filters.block().map();
    typename Block3::map_type const& map3 = outputs.block().map();
    typename Block4::map_type const& map4 = outputs.block().map();

    assert(map1.num_subblocks(1) == 1);
    assert(map2.num_subblocks(1) == 1);
    assert(map3.num_subblocks(1) == 1);
    assert(map4.num_subblocks(1) == 1);

    // Check that mappings are the same
    assert(map1.distribution(0) == map2.distribution(0));
    assert(map2.distribution(0) == map3.distribution(0));
    assert(map3.distribution(0) == map4.distribution(0));
    assert(map4.distribution(0) == map1.distribution(0));
#endif

    // Also check that local views are the same size
    assert(LOCAL(inputs).size(0) == LOCAL(filters).size(0)); 
    assert(LOCAL(filters).size(0) == LOCAL(outputs).size(0)); 
    assert(LOCAL(outputs).size(0) == LOCAL(expected).size(0)); 
    assert(LOCAL(expected).size(0) == LOCAL(inputs).size(0)); 
  }
};


/***********************************************************************
  ImplFull: built-in FIR 
***********************************************************************/

// This helper class holds an array of Fir objects 

template <typename T>
struct fir_vector : std::vector<Fir<T, nonsym, state_no_save, 1>*>
{
  typedef Fir<T, nonsym, state_no_save, 1> fir_type;
  typedef std::vector<fir_type*> base_type;
  typedef typename base_type::size_type size_type;
  typedef typename base_type::iterator iterator;

  fir_vector (size_type n) : base_type (n) 
  {
    for (iterator i = base_type::begin(); i != base_type::end(); ++i)
      *i = NULL;
  }

  ~fir_vector () 
  {
    iterator i = base_type::end();
    do 
    {
      delete *--i;
    } while (i != base_type::begin());
  }
};


template <typename T>
struct t_firbank_base<T, ImplFull> : t_local_view<T>,  Benchmark_base
{
  float ops(length_type filters, length_type points, length_type coeffs)
  {
    float total_ops = filters * points * coeffs *
                  (vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add); 
    return total_ops;
  }

  template <
    typename Block1,
    typename Block2,
    typename Block3,
    typename Block4
    >
  void firbank(
    Matrix<T, Block1> inputs,
    Matrix<T, Block2> filters,
    Matrix<T, Block3> outputs,
    Matrix<T, Block4> expected,
    length_type       loop,
    float&            time)
  {
    this->verify_views(inputs, filters, outputs, expected);

    // Create fir filters
    length_type local_M = LOCAL(inputs).size(0);
    length_type N = inputs.row(0).size();

    typedef Fir<T, nonsym, state_no_save, 1> fir_type;
    fir_vector<T> fir(local_M);
    for ( length_type i = 0; i < local_M; ++i )
      fir[i] = new fir_type(LOCAL(filters).row(i), N, 1);


    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      // Perform FIR convolutions
      for ( length_type i = 0; i < local_M; ++i )
        (*fir[i])( LOCAL(inputs).row(i), LOCAL(outputs).row(i) );
    }
    t1.stop();
    time = t1.delta();

    // Verify data
    assert( view_equal(LOCAL(outputs), LOCAL(expected)) );
  }

  t_firbank_base(length_type filters, length_type coeffs)
   : m_(filters), k_(coeffs) {}

public:
  // Member data
  length_type const m_;
  length_type const k_;
};


/***********************************************************************
  ImplFast: fast convolution using FFTs
***********************************************************************/

template <typename T>
struct t_firbank_base<T, ImplFast> : t_local_view<T>, Benchmark_base
{
  float fft_ops(length_type len)
  {
    return float(5 * len * std::log(float(len)) / std::log(float(2)));
  }

  float ops(length_type filters, length_type points, length_type /*coeffs*/)
  {
    // 'coeffs' is not used because the coefficients are zero-padded to the 
    // length of the inputs.

    return float(
      filters * ( 
        2 * fft_ops(points) +                   // one forward, one reverse FFT
        vsip::impl::Ops_info<T>::mul * points   // element-wise vector multiply
      )
    );
  }

  template <
    typename Block1,
    typename Block2,
    typename Block3,
    typename Block4
    >
  void firbank(
    Matrix<T, Block1> inputs,
    Matrix<T, Block2> filters,
    Matrix<T, Block3> outputs,
    Matrix<T, Block4> expected,
    length_type       loop,
    float&            time)
  {
    this->verify_views(inputs, filters, outputs, expected);

    // Create FFT objects
    length_type N = inputs.row(0).size();
    length_type scale = 1;

    typedef Fft<const_Vector, T, T, fft_fwd, by_reference> fwd_fft_type;
    typedef Fft<const_Vector, T, T, fft_inv, by_reference> inv_fft_type;

    fwd_fft_type fwd_fft(Domain<1>(N), scale);
    inv_fft_type inv_fft(Domain<1>(N), scale / float(N));

    // Copy the filters and zero pad to same length as inputs
    length_type K = this->k_;
    length_type local_M = LOCAL(inputs).size(0);

    Matrix<T> response(local_M, N, T());
    response(Domain<2>(local_M, K)) = LOCAL(filters); 

    // Pre-compute the FFT on the filters
    for ( length_type i = 0; i < local_M; ++i )
      fwd_fft(response.row(i));

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      // Perform FIR convolutions
      Vector<T> tmp(N);
      for ( length_type i = 0; i < local_M; ++i )
      {
        fwd_fft(LOCAL(inputs).row(i), tmp);
        tmp *= response.row(0);    // assume fft already done on response
        inv_fft(tmp, LOCAL(outputs).row(i)); 
      }
    }
    t1.stop();
    time = t1.delta();

    // Verify data - ignore values that overlap due to circular convolution. 
    // This means 'k-1' values at either end of each vector.
    if ( N > 2*(K-1) )
    {
      vsip::Domain<2> middle( Domain<1>(local_M), Domain<1>(K-1, 1, N-2*(K-1)) );
      assert( view_equal(LOCAL(outputs)(middle), LOCAL(expected)(middle)) );
    }
  }

  t_firbank_base(length_type filters, length_type coeffs)
   : m_(filters), k_(coeffs) {}

public:
  // Member data
  length_type const m_;
  length_type const k_;
};


/***********************************************************************
  ImplExpr: fast convolution using a single expression with FFTMs
***********************************************************************/

template <typename T>
struct t_firbank_base<T, ImplExpr> : t_local_view<T>, Benchmark_base
{
  float fft_ops(length_type len)
  {
    return float(5 * len * std::log(float(len)) / std::log(float(2)));
  }

  float ops(length_type filters, length_type points, length_type /*coeffs*/)
  {
    // 'coeffs' is not used because the coefficients are zero-padded to the 
    // length of the inputs.

    return float(
      filters * ( 
        2 * fft_ops(points) +                   // one forward, one reverse FFT
        vsip::impl::Ops_info<T>::mul * points   // element-wise vector multiply
      )
    );
  }
  
  static int const no_times = 15; // not > 12 = FFT_MEASURE
  
  typedef Fftm<T, T, row, fft_fwd, by_value, no_times>    for_fftm_type;
  typedef Fftm<T, T, row, fft_inv, by_value, no_times>    inv_fftm_type;

  template <
    typename Block1,
    typename Block2,
    typename Block3,
    typename Block4
    >
  void firbank(
    Matrix<T, Block1> inputs,
    Matrix<T, Block2> filters,
    Matrix<T, Block3> outputs,
    Matrix<T, Block4> expected,
    length_type       loop,
    float&            time)
  {
    this->verify_views(inputs, filters, outputs, expected);
    assert(inputs.size(0) <= this->m_);

    // Create FFT objects
    length_type local_M = LOCAL(inputs).size(0);
    length_type M = inputs.size(0);
    length_type N = inputs.size(1);
    length_type K = this->k_;
    length_type scale = 1;
    
    for_fftm_type for_fftm(Domain<2>(M, N), scale);
    inv_fftm_type inv_fftm(Domain<2>(M, N), scale / float(N));

    // Copy the filters and zero pad to same length as inputs
    Matrix<T, Block2> response(M, N, T(), filters.block().map());
    LOCAL(response)(Domain<2>(local_M, K)) = LOCAL(filters); 

    // Pre-compute the FFT on the filters
    response = for_fftm(response);

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      outputs = inv_fftm(response * for_fftm(inputs));
    }
    t1.stop();
    time = t1.delta();

    
    // Verify data - ignore values that overlap due to circular convolution. 
    // This means 'k-1' values at either end of each vector.
    if ( N > 2*(K-1) )
    {
      vsip::Domain<2> middle( Domain<1>(local_M), Domain<1>(K-1, 1, N-2*(K-1)) );
      assert( view_equal(LOCAL(outputs)(middle), LOCAL(expected)(middle)) );
    }
  }

  t_firbank_base(length_type filters, length_type coeffs)
   : m_(filters), k_(coeffs) {}

public:
  // Member data
  length_type const m_;
  length_type const k_;
};



/***********************************************************************
  Generic front-end for varying input vector lengths 
***********************************************************************/

template <typename T, typename ImplTag>
struct t_firbank_sweep_n : t_firbank_base<T, ImplTag>
{
  char const *what() { return "t_firbank_sweep_n"; }
  int ops_per_point(length_type size)
    { return (int)(this->ops(this->m_, size, this->k_) / size); }
  int riob_per_point(length_type) { return -1; }
  int wiob_per_point(length_type) { return -1; }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
#if PARALLEL_FIRBANK
    typedef Map<Block_dist, Whole_dist>      map_type;
    typedef Dense<2, T, row2_type, map_type> block_type;
    typedef Matrix<T, block_type>            view_type;

    processor_type np = num_processors();
    map_type map = map_type(Block_dist(np), Whole_dist());

    view_type inputs(this->m_, size, map);
    view_type filters(this->m_, this->k_, map);
    view_type outputs(this->m_, size, map);
    view_type expected(this->m_, size, map);
#else
    typedef Matrix<T>  view_type;

    view_type inputs(this->m_, size);
    view_type filters(this->m_, this->k_);
    view_type outputs(this->m_, size);
    view_type expected(this->m_, size);
#endif

    // Initialize
    inputs = T();
    filters = T();
    outputs = T();
    expected = T();

    // Create some test data
    inputs.row(0).put(0, 1);       filters.row(0).put(0, 1);
    inputs.row(0).put(1, 2);       filters.row(0).put(1, 1);
    inputs.row(0).put(2, 3);
    inputs.row(0).put(3, 4);

    expected.row(0).put(0, 1);
    expected.row(0).put(1, 3);
    expected.row(0).put(2, 5);
    expected.row(0).put(3, 7);
    expected.row(0).put(4, 4);

    // Run the test and time it
    this->firbank(inputs, filters, outputs, expected, loop, time);
  }

  t_firbank_sweep_n(length_type filters, length_type coeffs)
   : t_firbank_base<T, ImplTag>(filters, coeffs) {}
};


#ifdef VSIP_IMPL_SOURCERY_VPP
/***********************************************************************
  Generic front-end for using external data

  Note: This option is supported using Sourcery VSIPL++ extensions.
***********************************************************************/

template <typename T, typename ImplTag>
struct t_firbank_from_file : t_firbank_base<T, ImplTag>
{
  char const *what() { return "t_firbank_from_file"; }
  int ops_per_point(length_type size)
    { return (int)(this->ops(this->m_, size, this->k_) / size); }
  int riob_per_point(length_type) { return -1; }
  int wiob_per_point(length_type) { return -1; }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    // Perform file I/O to obtain input data, the filter (one copy 
    // is replicated for each row) and the output data.  Output
    // data is compared against the calculated convolution data
    // after the benchmark has been run.

    // Create a "root" view for each that will give the first
    // processor access to all of the data.  
    typedef Map<Block_dist, Block_dist>           root_map_type;
    typedef Dense<2, T, row2_type, root_map_type> root_block_type;
    typedef Matrix<T, root_block_type>            root_view_type;

    // A processor set with one processor in it is used to create 
    // a map with 1 subblock
    Vector<processor_type> pset0(1);
    pset0(0) = processor_set()(0);
    root_map_type root_map(pset0, 1, 1);

    root_view_type inputs_root (this->m_, size,     root_map);
    root_view_type filters_root(this->m_, this->k_, root_map);
    root_view_type expected_root(this->m_, size,     root_map);


    // Only the root processor need perform file I/O
    if (root_map.subblock() != no_subblock)
    {
      // Initialize
      LOCAL(inputs_root) = T();
      LOCAL(filters_root) = T();
      LOCAL(expected_root) = T();
      
      // read in inputs, filters and outputs
      std::ostringstream input_file;
      std::ostringstream filter_file;
      std::ostringstream output_file;

      length_type log2_size = 1;
      while ( ++log2_size < 32 )
        if ( static_cast<length_type>(1 << log2_size) == size )
          break;

      input_file << this->directory_ << "/inputs_" << log2_size << ".matrix";
      filter_file << this->directory_ << "/filters.matrix";
      output_file << this->directory_ << "/outputs_" << log2_size << ".matrix";
    
      Load_view<2, T> load_inputs (input_file.str().c_str(), 
        Domain<2>(this->m_, size));
      Load_view<2, T> load_filters(filter_file.str().c_str(), 
        Domain<2>(this->m_, this->k_));
      Load_view<2, T> load_outputs(output_file.str().c_str(), 
        Domain<2>(this->m_, size));

      LOCAL(inputs_root) = load_inputs.view();
      LOCAL(filters_root) = load_filters.view();
      LOCAL(expected_root) = load_outputs.view();
    }


    // Create the distributed views that will give each processor a 
    // subset of the data
    typedef Map<Block_dist, Whole_dist>      map_type;
    typedef Dense<2, T, row2_type, map_type> block_type;
    typedef Matrix<T, block_type>            view_type;

    processor_type np = num_processors();
    map_type map = map_type(Block_dist(np), Whole_dist());

    view_type inputs  (this->m_, size,     map);
    view_type filters (this->m_, this->k_, map);
    view_type outputs (this->m_, size,     map);
    view_type expected(this->m_, size,     map);

    inputs = inputs_root;
    filters = filters_root;
    outputs = T();
    expected = expected_root; 


    // Run the test and time it
    this->firbank(inputs, filters, outputs, expected, loop, time);
  }

  t_firbank_from_file(length_type m, length_type k, char const *directory)
   : t_firbank_base<T, ImplTag>(m, k),
     directory_(directory)
    {}

  // data
private:
  char const *directory_;
};
#endif // #ifdef VSIP_IMPL_SOURCERY_VPP



void
defaults(Loop1P& loop)
{
  loop.cal_        = 7;
  loop.start_      = 7;
  loop.stop_       = 15;
  loop.user_param_ = 64;
}


int
test(Loop1P& loop, int what)
{
  /* From PCA Kernel-Level Benchmarks Project Report:

              FIR ﬁlter bank input parameters.
  Parameter                                       Values
    Name               Description             Set 1 Set 2
     M      Number of ﬁlters                     64     20
     N      Length of input and output vectors 4096   1024
     K      Number of ﬁlter coefﬁcients         128     12
     W      Workload (Mﬂop)                      34   1.97

  Note: The workload calculations are given using the fast convolution
  algorithm for Set 1 and using the time-domain algorithm for Set 2.
  */

  switch (what)
  {
  case  1: loop(
    t_firbank_sweep_n<complex<float>, ImplFull>(64, 128));
    break;
  case  2: loop(
    t_firbank_sweep_n<complex<float>, ImplFull>(20,  12));
    break;
  case  11: loop(
    t_firbank_sweep_n<complex<float>, ImplFast>(64, 128));
    break;
  case  12: loop(
    t_firbank_sweep_n<complex<float>, ImplFast>(20,  12));
    break;
  case  21: loop(
    t_firbank_sweep_n<complex<float>, ImplExpr>(64, 128));
    break;
  case  22: loop(
    t_firbank_sweep_n<complex<float>, ImplExpr>(20,  12));
    break;

#ifdef VSIP_IMPL_SOURCERY_VPP
  case  51: loop(
    t_firbank_from_file<complex<float>, ImplFull> (64, 128, "data/set1"));
    break;
  case  52: loop(
    t_firbank_from_file<complex<float>, ImplFull> (20,  12, "data/set2"));
    break;
  case  61: loop(
    t_firbank_from_file<complex<float>, ImplFast> (64, 128, "data/set1"));
    break;
  case  62: loop(
    t_firbank_from_file<complex<float>, ImplFast> (20,  12, "data/set2"));
    break;
  case  71: loop(
    t_firbank_from_file<complex<float>, ImplExpr> (64, 128, "data/set1"));
    break;
  case  72: loop(
    t_firbank_from_file<complex<float>, ImplExpr> (20,  12, "data/set2"));
    break;
#endif

  case 0:
    std::cout
      << "firbank -- FIR Filter Bank\n"
      << "  #   Set    Method   Data\n"
      << "  -1   1      Time     generated\n"
      << "  -2   2      Time     generated\n"
      << " -11   1    Freq/FFT   generated\n"
      << " -12   2    Freq/FFT   generated\n"
      << " -21   1    Freq/FFTM  generated\n"
      << " -22   2    Freq/FFTM  generated\n"
      << " ---\n"
      << " -51   1      Time     external\n"
      << " -52   2      Time     external\n"
      << " -61   1    Freq/FFT   external\n"
      << " -62   2    Freq/FFT   external\n"
      << " -71   1    Freq/FFTM  external\n"
      << " -72   2    Freq/FFTM  external\n"
      << std::endl;

  default: 
    return 0;
  }
  return 1;
}
 
