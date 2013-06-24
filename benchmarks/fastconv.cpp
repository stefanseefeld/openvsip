//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for Fast Convolution.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include "benchmark.hpp"
#include "fastconv.hpp"

using namespace ovxx;

struct Impl1op;		// out-of-place, phased fast-convolution
struct Impl1ip;		// in-place, phased fast-convolution
struct Impl1pip1;	// psuedo in-place (using in-place Fft), phased
struct Impl1pip2;	// psuedo in-place (using out-of-place Fft), phased
struct Impl2op;		// out-of-place (tmp), interleaved fast-convolution
struct Impl2ip;		// in-place, interleaved fast-convolution
struct Impl2ip_tmp;	// in-place (w/tmp), interleaved fast-convolution
struct Impl3;		// Mixed fast-convolution
struct Impl4vc;		// Single-line fast-convolution, vector of coeffs
struct Impl4mc;		// Single-line fast-convolution, matrix of coeffs

struct Impl1pip2_nopar;

bool check = true;



/***********************************************************************
  Impl1op: out-of-place, phased fast-convolution
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl1op> : fastconv_ops
{
  static length_type const num_args = 2;

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
    typedef Fftm<T, T, row, fft_inv, by_reference, no_times>
	  	inv_fftm_type;

    // Create the FFT objects.
    for_fftm_type for_fftm(Domain<2>(npulse, nrange), 1.0);
    inv_fftm_type inv_fftm(Domain<2>(npulse, nrange), 1.0/nrange);

    // Initialize
    data    = T();
    replica = T();


    // Before fast convolution, convert the replica into the
    // frequency domain
    // for_fft(replica);
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      // Perform fast convolution:
      for_fftm(data, tmp);
      tmp = vmmul<0>(replica, tmp);
      inv_fftm(tmp, data);
    }
    time = t1.elapsed();
  }
};



/***********************************************************************
  Impl1ip: in-place, phased fast-convolution
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl1ip> : fastconv_ops
{
  static length_type const num_args = 1;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    // Create the data cube.
    Matrix<T> data(npulse, nrange);
    
    // Create the pulse replica
    Vector<T> replica(nrange);

    // int const no_times = 0; // FFTW_PATIENT
    int const no_times = 15; // not > 12 = FFT_MEASURE
    
    typedef Fftm<T, T, row, fft_fwd, by_reference, no_times>
	  	for_fftm_type;
    typedef Fftm<T, T, row, fft_inv, by_reference, no_times>
	  	inv_fftm_type;

    // Create the FFT objects.
    for_fftm_type for_fftm(Domain<2>(npulse, nrange), 1.0);
    inv_fftm_type inv_fftm(Domain<2>(npulse, nrange), 1.0/nrange);

    // Initialize
    data    = T();
    replica = T();


    // Before fast convolution, convert the replica into the
    // frequency domain
    // for_fft(replica);
    
    // Impl1 ip
    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      // Perform fast convolution:
      for_fftm(data);
      data = vmmul<0>(replica, data);
      inv_fftm(data);
    }
    time = t1.elapsed();
  }
};



/***********************************************************************
  Impl1pip1: psuedo in-place (using in-place Fft), phased fast-convolution
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl1pip1> : fastconv_ops
{
  static length_type const num_args = 1;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    // Create the data cube.
    Matrix<T> data(npulse, nrange);
    
    // Create the pulse replica
    Vector<T> tmp(nrange);
    Vector<T> replica(nrange);

    // int const no_times = 0; // FFTW_PATIENT
    int const no_times = 15; // not > 12 = FFT_MEASURE
    
    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times>
	  	for_fft_type;
    typedef Fft<const_Vector, T, T, fft_inv, by_reference, no_times>
	  	inv_fft_type;

    // Create the FFT objects.
    for_fft_type for_fft(Domain<1>(nrange), 1.0);
    inv_fft_type inv_fft(Domain<1>(nrange), 1.0/(nrange));

    // Initialize
    data    = T();
    replica = T();


    // Before fast convolution, convert the replica into the
    // frequency domain
    // for_fft(replica);
    
    // Impl1 pip
    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      // Perform fast convolution:
      for (index_type p=0; p<npulse; ++p)
	for_fft(data.row(p));
      data = vmmul<0>(replica, data);
      for (index_type p=0; p<npulse; ++p)
	inv_fft(data.row(p));
    }
    time = t1.elapsed();
  }
};



/***********************************************************************
  Impl1pip2: psuedo in-place (using out-of-place Fft), phased fast-convolution
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl1pip2> : fastconv_ops
{
  static length_type const num_args = 1;

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
#else
    typedef Matrix<T>  view_type;
    typedef Vector<T>  replica_view_type;

    view_type data(npulse, nrange);
#endif

    // Create the pulse replica
    Vector<T> tmp(nrange);
    replica_view_type replica(nrange);

    // int const no_times = 0; // FFTW_PATIENT
    int const no_times = 15; // not > 12 = FFT_MEASURE
    
    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times>
	  	for_fft_type;
    typedef Fft<const_Vector, T, T, fft_inv, by_reference, no_times>
	  	inv_fft_type;

    // Create the FFT objects.
    for_fft_type for_fft(Domain<1>(nrange), 1.0);
    inv_fft_type inv_fft(Domain<1>(nrange), 1.0/(nrange));

    // Initialize
    data    = T();
    replica = T();


    // Before fast convolution, convert the replica into the
    // frequency domain
    // for_fft(replica);
    
    // Impl1 pip2
    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      length_type l_npulse  = data.local().size(0);
      for (index_type p=0; p<l_npulse; ++p)
      {
	for_fft(data.local().row(p), tmp);
	data.local().row(p) = tmp;
      }
      data.local() = vmmul<0>(replica.local(), data.local());
      for (index_type p=0; p<l_npulse; ++p)
      {
	inv_fft(data.local().row(p), tmp);
	data.local().row(p) = tmp;
      }
    }
    time = t1.elapsed();
  }
};

template <typename T>
struct t_fastconv_base<T, Impl1pip2_nopar> : fastconv_ops
{
  static length_type const num_args = 1;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    // Create the data cube.
    Matrix<T> data(npulse, nrange);
    
    // Create the pulse replica
    Vector<T> tmp(nrange);
    Vector<T> replica(nrange);

    // int const no_times = 0; // FFTW_PATIENT
    int const no_times = 15; // not > 12 = FFT_MEASURE
    
    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times>
	  	for_fft_type;
    typedef Fft<const_Vector, T, T, fft_inv, by_reference, no_times>
	  	inv_fft_type;

    // Create the FFT objects.
    for_fft_type for_fft(Domain<1>(nrange), 1.0);
    inv_fft_type inv_fft(Domain<1>(nrange), 1.0/(nrange));

    // Initialize
    data    = T();
    replica = T();


    // Before fast convolution, convert the replica into the
    // frequency domain
    // for_fft(replica);
    
    // Impl1 pip2
    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      for (index_type p=0; p<npulse; ++p)
      {
	for_fft(data.row(p), tmp);
	data.row(p) = tmp;
      }
      data = vmmul<0>(replica, data);
      for (index_type p=0; p<npulse; ++p)
      {
	inv_fft(data.row(p), tmp);
	data.row(p) = tmp;
      }
    }
    time = t1.elapsed();
  }
};



/***********************************************************************
  Impl2op: out-of-place (tmp), interleaved fast-convolution
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl2op> : fastconv_ops
{
  static length_type const num_args = 2;

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
    view_type chk(npulse, nrange, map);
#else
    typedef Matrix<T>  view_type;
    typedef Vector<T>  replica_view_type;

    view_type data(npulse, nrange);
    view_type chk(npulse, nrange);
#endif
    Vector<T> tmp(nrange);
    
    // Create the pulse replica
    replica_view_type replica(nrange);

    // int const no_times = 0; // FFTW_PATIENT
    int const no_times = 15; // not > 12 = FFT_MEASURE
    
    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times>
	  	for_fft_type;
    typedef Fft<const_Vector, T, T, fft_inv, by_reference, no_times>
	  	inv_fft_type;

    // Create the FFT objects.
    for_fft_type for_fft(Domain<1>(nrange), 1.0);
    inv_fft_type inv_fft(Domain<1>(nrange), 1.0/(nrange));

    // Initialize
    data    = T();
    replica = T();


    // Before fast convolution, convert the replica into the
    // frequency domain
    // for_fft(replica);
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      length_type l_npulse  = data.local().size(0);
      for (index_type p=0; p<l_npulse; ++p)
      {
	for_fft(data.local().row(p), tmp);
	tmp *= replica.local();
	inv_fft(tmp, data.local().row(p));
      }
    }
    time = t1.elapsed();

    // CHECK RESULT
    Rand<T> gen(0, 0);

    data = gen.randu(npulse, nrange);
    replica.put(0, T(1));
    for_fft(replica);

    length_type l_npulse  = data.local().size(0);
    for (index_type p=0; p<l_npulse; ++p)
    {
      for_fft(data.local().row(p), tmp);
      tmp *= replica.local();
      inv_fft(tmp, chk.local().row(p));
    }

    double error = test::diff(data.local(), chk.local());

    test_assert(error < -100);
  }
};



/***********************************************************************
  Impl2ip: in-place, interleaved fast-convolution
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl2ip> : fastconv_ops
{
  static length_type const num_args = 1;

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
    view_type chk(npulse, nrange, map);
    
#else
    typedef Matrix<T>  view_type;
    typedef Vector<T>  replica_view_type;

    view_type data(npulse, nrange);
    view_type chk(npulse, nrange);
#endif

    // Create the pulse replica
    replica_view_type replica(nrange);

    // int const no_times = 0; // FFTW_PATIENT
    int const no_times = 15; // not > 12 = FFT_MEASURE
    
    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times>
	  	for_fft_type;
    typedef Fft<const_Vector, T, T, fft_inv, by_reference, no_times>
	  	inv_fft_type;

    // Create the FFT objects.
    for_fft_type for_fft(Domain<1>(nrange), 1.0);
    inv_fft_type inv_fft(Domain<1>(nrange), 1.0/(nrange));

    // Initialize
    data    = T();
    replica = T();


    // Before fast convolution, convert the replica into the
    // frequency domain
    // for_fft(replica);
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      length_type l_npulse  = data.local().size(0);
      for (index_type p=0; p<l_npulse; ++p)
      {
	for_fft(data.local().row(p));
	data.local().row(p) *= replica.local();
	inv_fft(data.local().row(p));
      }
    }
    time = t1.elapsed();

    // CHECK RESULT
    Rand<T> gen(0, 0);

    data = gen.randu(npulse, nrange);
    chk  = data;
    replica.put(0, T(1));
    for_fft(replica);

    length_type l_npulse  = data.local().size(0);
    for (index_type p=0; p<l_npulse; ++p)
    {
      for_fft(data.local().row(p));
      data.local().row(p) *= replica.local();
      inv_fft(data.local().row(p));
    }

    double error = test::diff(data.local(), chk.local());

    test_assert(error < -100);
  }
};



/***********************************************************************
  Impl2ip_tmp: in-place, interleaved fast-convolution
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl2ip_tmp> : fastconv_ops
{
  static length_type const num_args = 1;

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

#else
    typedef Matrix<T>  view_type;
    typedef Vector<T>  replica_view_type;

    view_type data(npulse, nrange);
#endif
    Vector<T> tmp(nrange);
    
    // Create the pulse replica
    replica_view_type replica(nrange);

    // int const no_times = 0; // FFTW_PATIENT
    int const no_times = 15; // not > 12 = FFT_MEASURE
    
    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times>
	  	for_fft_type;
    typedef Fft<const_Vector, T, T, fft_inv, by_reference, no_times>
	  	inv_fft_type;

    // Create the FFT objects.
    for_fft_type for_fft(Domain<1>(nrange), 1.0);
    inv_fft_type inv_fft(Domain<1>(nrange), 1.0/(nrange));

    // Initialize
    data    = T();
    replica = T();


    // Before fast convolution, convert the replica into the
    // frequency domain
    // for_fft(replica);
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      length_type l_npulse  = data.local().size(0);
      for (index_type p=0; p<l_npulse; ++p)
      {
	tmp = data.local().row(p);
	for_fft(tmp);
	tmp *= replica.local();
	inv_fft(tmp);
	data.local().row(p) = tmp;
      }
    }
    time = t1.elapsed();
  }
};

template <typename T>
class Fast_convolution
{
  // static int const no_times = 0; // FFTW_PATIENT
  static int const no_times = 15; // not > 12 = FFT_MEASURE

  typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times>
		for_fft_type;
  typedef Fft<const_Vector, T, T, fft_inv, by_reference, no_times>
		inv_fft_type;

public:
  template <typename Block>
  Fast_convolution(
    Vector<T, Block> replica)
    : replica_(replica.size()),
      tmp_    (replica.size()),
      for_fft_(Domain<1>(replica.size()), 1.0),
      inv_fft_(Domain<1>(replica.size()), 1.0/replica.size())
  {
    replica_ = replica;
  }

  template <typename       Block1,
	    typename       Block2,
	    dimension_type Dim>
  void operator()(
    Vector<T, Block1> in,
    Vector<T, Block2> out,
    Index<Dim>        /*idx*/)
  {
    for_fft_(in, tmp_);
    tmp_ *= replica_;
    inv_fft_(tmp_, out);
  }

  // Member data.
private:
  Vector<T>    replica_;
  Vector<T>    tmp_;
  for_fft_type for_fft_;
  inv_fft_type inv_fft_;
};

/***********************************************************************
  Impl3: Mixed phase/interleave fast-convolution
***********************************************************************/

#if PARALLEL_FASTCONV
template <typename T>
struct t_fastconv_base<T, Impl3> : fastconv_ops
{
  static length_type const num_args = 1;

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
    view_type data(npulse, nrange, map);
    view_type tmp(npulse, nrange, map);
    
    // Create the pulse replica
    replica_view_type replica(nrange);

    // int const no_times = 0; // FFTW_PATIENT
    int const no_times = 15; // not > 12 = FFT_MEASURE
    
    typedef Fftm<T, T, row, fft_fwd, by_reference, no_times>
	  	for_fftm_type;
    typedef Fftm<T, T, row, fft_inv, by_reference, no_times>
	  	inv_fftm_type;

    // Create the FFT objects.
    for_fftm_type for_fftm(Domain<2>(npulse, nrange), 1.0);
    inv_fftm_type inv_fftm(Domain<2>(npulse, nrange), 1.0/nrange);

    // Initialize
    data    = T();
    replica = T();


    // Before fast convolution, convert the replica into the
    // frequency domain
    // for_fft(replica);
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      // Perform fast convolution:
      for_fftm(data, tmp);
      tmp = vmmul<0>(replica, tmp);
      inv_fftm(tmp, data);
    }
    time = t1.elapsed();
  }
};
#endif // PARALLEL_FASTCONV



/***********************************************************************
  Impl4vc: Single expression fast-convolution, vector of coefficients.
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl4vc> : fastconv_ops
{
  static length_type const num_args = 1;

#if PARALLEL_FASTCONV
  typedef Map<Block_dist, Whole_dist>      map_type;
  typedef Dense<2, T, row2_type, map_type> block_type;
  typedef Matrix<T, block_type>            view_type;

  typedef Dense<1, T, row1_type, Replicated_map<1> > replica_block_type;
  typedef Vector<T, replica_block_type>          replica_view_type;
#else
  typedef Local_map  map_type;
  typedef Matrix<T>  view_type;
  typedef Vector<T>  replica_view_type;
#endif

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
    map_type map = map_type(Block_dist(np), Whole_dist());
#else
    map_type map;
#endif

    // Create the data cube.
    view_type data(npulse, nrange, map);
    view_type chk(npulse, nrange, map);
    
    // Create the pulse replica
    replica_view_type replica(nrange);

    // Create the FFT objects.
    for_fftm_type for_fftm(Domain<2>(npulse, nrange), 1.0);
    inv_fftm_type inv_fftm(Domain<2>(npulse, nrange), 1.0/nrange);

    // Initialize
    data    = T();
    replica = T();


    // Before fast convolution, convert the replica into the
    // frequency domain
    // for_fft(replica);
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      data = inv_fftm(vmmul<0>(replica, for_fftm(data)));
    }
    time = t1.elapsed();

    // CHECK RESULT
    if (check)
    {
      typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times>
	  	for_fft_type;

      Rand<T> gen(0, 0);
      for_fft_type for_fft(Domain<1>(nrange), 1.0);

      data = gen.randu(npulse, nrange);
      replica.put(0, T(1));
      for_fft(replica);
      
      chk = inv_fftm(vmmul<0>(replica, for_fftm(data)));

      double error = test::diff(data.local(), chk.local());

      test_assert(error < -100);
    }
  }

  void diag()
  {
#if PARALLEL_FASTCONV
    processor_type np = num_processors();
    map_type map = map_type(Block_dist(np), Whole_dist());
#else
    map_type map;
#endif

    length_type npulse = 16;
    length_type nrange = 2048;

    // Create the data cube.
    view_type data(npulse, nrange, map);

    // Create the pulse replica
    replica_view_type replica(nrange);

    // Create the FFT objects.
    for_fftm_type for_fftm(Domain<2>(npulse, nrange), 1.0);
    inv_fftm_type inv_fftm(Domain<2>(npulse, nrange), 1.0/nrange);

    std::cout << assignment::diagnostics(data, inv_fftm(vmmul<0>(replica, for_fftm(data)))) << std::endl;
  }
};



/***********************************************************************
  Impl4mc: Single expression fast-convolution, matrix of coefficients.
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl4mc> : fastconv_ops
{
  // This operation is in-place, however the coefficients are unique 
  // for each row, therefore they constitute a second argument.
  static length_type const num_args = 2;

#if PARALLEL_FASTCONV
  typedef Map<Block_dist, Whole_dist>      map_type;
  typedef Dense<2, T, row2_type, map_type> block_type;
  typedef Matrix<T, block_type>            view_type;
  typedef view_type                        replica_view_type;
#else
  typedef Local_map  map_type;
  typedef Matrix<T>  view_type;
  typedef view_type  replica_view_type;
#endif

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
    map_type map = map_type(Block_dist(np), Whole_dist());
#else
    map_type map;
#endif

    // Create the data cube.
    view_type data(npulse, nrange, map);
    view_type chk(npulse, nrange, map);
    
    // Create the pulse replica
    replica_view_type replica(npulse, nrange, map);

    // Create the FFT objects.
    for_fftm_type for_fftm(Domain<2>(npulse, nrange), 1.0);
    inv_fftm_type inv_fftm(Domain<2>(npulse, nrange), 1.0/nrange);

    // Initialize
    data    = T();
    replica = T();


    // Before fast convolution, convert the replica into the
    // frequency domain
    // for_fft(replica);
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      data = inv_fftm(replica * for_fftm(data));
    }
    time = t1.elapsed();

    // CHECK RESULT
    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times>
	  	for_fft_type;

    Rand<T> gen(0, 0);
    for_fft_type for_fft(Domain<1>(nrange), 1.0);
    Vector<T> tmp(nrange);

    data = gen.randu(npulse, nrange);
    length_type l_npulse  = data.local().size(0);
    for (index_type p = 0; p < l_npulse; ++p)
    {
      replica.put(p, 0, T(1));
      for_fft(replica.local().row(p), tmp);
      replica.local().row(p) = tmp;
    }

    chk = inv_fftm(replica * for_fftm(data));

    double error = test::diff(data.local(), chk.local());

    test_assert(error < -100);
  }

  void diag()
  {
#if PARALLEL_FASTCONV
    processor_type np = num_processors();
    map_type map = map_type(Block_dist(np), Whole_dist());
#else
    map_type map;
#endif

    length_type npulse = 16;
    length_type nrange = 2048;

    // Create the data cube.
    view_type data(npulse, nrange, map);

    // Create the pulse replica
    replica_view_type replica(npulse, nrange, map);

    // Create the FFT objects.
    for_fftm_type for_fftm(Domain<2>(npulse, nrange), 1.0);
    inv_fftm_type inv_fftm(Domain<2>(npulse, nrange), 1.0/nrange);

    std::cout << assignment::diagnostics(data, inv_fftm(replica * for_fftm(data))) << std::endl;
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
  loop.stop_       = 16;
  loop.loop_start_ = 10;

  loop.param_["rows"] = "64";
  loop.param_["size"] = "2048";
  loop.param_["check"] = "1";
}



int
benchmark(Loop1P& loop, int what)
{
  length_type rows  = atoi(loop.param_["rows"].c_str());
  length_type size  = atoi(loop.param_["size"].c_str());
  check             = (loop.param_["check"] == "1" ||
		       loop.param_["check"] == "y");

  std::cout << "rows: " << rows << "  size: " << size 
	    << "  check: " << (check ? "yes" : "no")
	    << std::endl;

  switch (what)
  {
  case  1: loop(t_fastconv_pf<complex<float>, Impl1op>(rows)); break;
  case  2: loop(t_fastconv_pf<complex<float>, Impl1ip>(rows)); break;
  case  3: loop(t_fastconv_pf<complex<float>, Impl1pip1>(rows)); break;
  case  4: loop(t_fastconv_pf<complex<float>, Impl1pip2>(rows)); break;
  case  5: loop(t_fastconv_pf<complex<float>, Impl2op>(rows)); break;
  case  6: loop(t_fastconv_pf<complex<float>, Impl2ip>(rows)); break;
  case  7: loop(t_fastconv_pf<complex<float>, Impl2ip_tmp>(rows)); break;
  case  9: loop(t_fastconv_pf<complex<float>, Impl4vc>(rows)); break;
  case 10: loop(t_fastconv_pf<complex<float>, Impl4mc>(rows)); break;

  case 11: loop(t_fastconv_rf<complex<float>, Impl1op>(size)); break;
  case 12: loop(t_fastconv_rf<complex<float>, Impl1ip>(size)); break;
  case 13: loop(t_fastconv_rf<complex<float>, Impl1pip1>(size)); break;
  case 14: loop(t_fastconv_rf<complex<float>, Impl1pip2>(size)); break;
  case 15: loop(t_fastconv_rf<complex<float>, Impl2op>(size)); break;
  case 16: loop(t_fastconv_rf<complex<float>, Impl2ip>(size)); break;
  case 17: loop(t_fastconv_rf<complex<float>, Impl2ip_tmp>(size)); break;
  case 19: loop(t_fastconv_rf<complex<float>, Impl4vc>(size)); break;
  case 20: loop(t_fastconv_rf<complex<float>, Impl4mc>(size)); break;

  case 101: loop(t_fastconv_pf<complex<float>, Impl1pip2_nopar>(rows)); break;


  // case 101: loop(t_fastconv_pf<complex<float>, Impl3>(param1)); break;

  case 0:
    std::cout
      << "fastconv -- fast convolution benchmark\n"
      << " Sweeping pulse size:\n"
      << "   -1 -- Out-of-place, phased\n"
      << "   -2 -- In-place, phased\n"
      << "   -3 -- Psuedo in-place Fftm (in-place Fft), phased\n"
      << "   -4 -- Psuedo in-place Fftm (out-of-place Fft), phased\n"
      << "   -5 -- Out-of-place, interleaved\n"
      << "   -6 -- In-place, interleaved\n"
      << "   -7 -- In-place (w/tmp), interleaved\n"
      << "   -8 -- Foreach_vector, interleaved (2fv)\n"
      << "   -9 -- Fused expression, vector of coefficients (4vc)\n"
      << "  -10 -- Fused expression, matrix of coefficients (4mc)\n"
      << "\n"
      << " Parameters (for sweeping convolution size, cases 1 through 10)\n"
      << "  -p:rows ROWS -- set number of pulses (default 64)\n"
      << "\n"
      << " Sweeping number of pulses:\n"
      << "  -11 -- Out-of-place, phased\n"
      << "  -12 -- In-place, phased\n"
      << "  -13 -- Psuedo in-place Fftm (in-place Fft), phased\n"
      << "  -14 -- Psuedo in-place Fftm (out-of-place Fft), phased\n"
      << "  -15 -- Out-of-place, interleaved\n"
      << "  -16 -- In-place, interleaved\n"
      << "  -17 -- In-place (w/tmp), interleaved\n"
      << "  -18 -- Foreach_vector, interleaved (2fv)\n"
      << "  -19 -- Fused expression, vector of coefficients (4vc)\n"
      << "  -20 -- Fused expression, matrix of coefficients (4mc)\n"
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
