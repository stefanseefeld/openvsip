/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for Fast Convolution.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip_csl/profile.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"
#include "benchmarks.hpp"
#include "fastconv.hpp"

using namespace vsip;

inline unsigned long
ilog2(length_type size)    // assume size = 2^n, != 0, return n.
{
  unsigned int n = 0;
  while (size >>= 1) ++n;
  return n;
}



template <storage_format_type C>
struct Impl1ip;		// in-place, phased fast-convolution (fscm)
template <storage_format_type C>
struct Impl2ip;		// in-place, interleaved fast-convolution (fcs loop)



/***********************************************************************
  Impl1ip: in-place, phased fast-convolution
***********************************************************************/

template <>
struct t_fastconv_base<complex<float>, Impl1ip<interleaved_complex> >
  : fastconv_ops
{
  static length_type const num_args = 1;

  typedef complex<float> T;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP1;
    typedef impl::Strided<1, T, LP1, Local_map> block1_type;

    typedef Layout<2, row2_type, dense, interleaved_complex> LP2;
    typedef impl::Strided<2, T, LP2, Local_map> block2_type;
    
    // Create the data cube.
    Matrix<T, block2_type> data(npulse, nrange);
    
    // Create the pulse replica and temporary buffer
    Vector<T, block1_type> replica(nrange);
    Vector<T, block1_type> tmp(nrange);

    // Initialize
    data    = T();
    replica = T();

    // setup direct access to data buffers
    dda::Data<block1_type, dda::in> data_replica(replica.block());
    dda::Data<block1_type, dda::in> data_tmp(tmp.block());

    // Create weights array
    unsigned long log2N = ilog2(nrange);
    long flag = FFT_FAST_CONVOLUTION;
    FFT_setup setup;
    unsigned long nbytes = 0;
    fft_setup( log2N, flag, &setup, &nbytes );

    // Create filter
    COMPLEX const* filter = reinterpret_cast<COMPLEX const*>(data_replica.ptr());
    COMPLEX const* t = reinterpret_cast<COMPLEX const*>(data_tmp.ptr());
    float scale = 1;
    long f_conv = FFT_INVERSE;  // does not indicate direction, but rather 
                                // convolution as opposed to correlation
    long eflag = 0;  // no caching hints

    fcf_ciptx(&setup, const_cast<COMPLEX*>(filter), const_cast<COMPLEX*>(t),
	      &scale, log2N, f_conv, eflag);

    
    // Set up convolution 
    dda::Data<block2_type, dda::inout> data_data(data.block());
    COMPLEX *msignal = reinterpret_cast<COMPLEX *>(data_data.ptr());
    long jr = data_data.stride(1);
    long jc = data_data.stride(0);
    unsigned long M = npulse;
    
    vsip_csl::profile::Timer t1;
    
    // Impl1 ip
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      // Perform fast convolution
      fcsm_ciptx(&setup, const_cast<COMPLEX*>(filter), msignal, jr, jc, const_cast<COMPLEX*>(t), 
		 log2N, M, f_conv, eflag);
    }
    t1.stop();

    // CHECK RESULT
    time = t1.delta();
    fft_free(&setup);
  }
};



/***********************************************************************
  Impl1ip: SPLIT in-place, phased fast-convolution
***********************************************************************/

template <>
struct t_fastconv_base<complex<float>, Impl1ip<split_complex> >
  : fastconv_ops
{
  static length_type const num_args = 1;

  typedef complex<float> T;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    typedef Layout<1, row1_type, dense, split_complex> LP1;
    typedef impl::Strided<1, T, LP1, Local_map> block1_type;

    typedef Layout<2, row2_type, dense, split_complex> LP2;
    typedef impl::Strided<2, T, LP2, Local_map> block2_type;

    typedef dda::Data<block1_type, dda::inout>::ptr_type ptr_type;
    
    // Create the data cube.
    Matrix<T, block2_type> data(npulse, nrange);
    
    // Create the pulse replica and temporary buffer
    Vector<T, block1_type> replica(nrange);
    Vector<T, block1_type> tmp(nrange);

    // Initialize
    data    = T();
    replica = T();

    // setup direct access to data buffers
    dda::Data<block1_type, dda::inout> data_replica(replica.block());
    dda::Data<block1_type, dda::inout> data_tmp(tmp.block());

    // Create weights array
    unsigned long log2N = ilog2(nrange);
    long flag = FFT_FAST_CONVOLUTION;
    FFT_setup setup;
    unsigned long nbytes = 0;
    fft_setup( log2N, flag, &setup, &nbytes );

    // Create filter
    // COMPLEX* filter = reinterpret_cast<COMPLEX *>(ext_replica.ptr());
    // COMPLEX* t = reinterpret_cast<COMPLEX *>(ext_tmp.ptr());
    ptr_type p_replica = data_replica.ptr();
    ptr_type p_tmp     = data_tmp.ptr();

    COMPLEX_SPLIT filter;
    COMPLEX_SPLIT tmpbuf;

    filter.realp = p_replica.first;
    filter.imagp = p_replica.second;
    tmpbuf.realp = p_tmp.first;
    tmpbuf.imagp = p_tmp.second;

    float scale = 1;
    long f_conv = FFT_INVERSE;  // does not indicate direction, but rather 
                                // convolution as opposed to correlation
    long eflag = 0;  // no caching hints

    fcf_ziptx( &setup, &filter, &tmpbuf, &scale, log2N, f_conv, eflag );

    
    // Set up convolution 
    dda::Data<block2_type, dda::inout> data_data(data.block());
    // COMPLEX* msignal = reinterpret_cast<COMPLEX *>(ext_data.ptr());

    ptr_type p_data = data_data.ptr();
    COMPLEX_SPLIT msignal;
    msignal.realp = p_data.first;
    msignal.imagp = p_data.second;

    long jr = data_data.stride(1);
    long jc = data_data.stride(0);
    unsigned long M = npulse;
    
    vsip_csl::profile::Timer t1;
    
    // Impl1 ip
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      // Perform fast convolution
      fcsm_ziptx(&setup, &filter, &msignal, jr, jc, &tmpbuf, 
		 log2N, M, f_conv, eflag );
    }
    t1.stop();

    // CHECK RESULT
    time = t1.delta();
    fft_free(&setup);
  }
};





/***********************************************************************
  Impl2ip: out-of-place (tmp), interleaved fast-convolution
***********************************************************************/

template <>
struct t_fastconv_base<complex<float>, Impl2ip<interleaved_complex> >
  : fastconv_ops
{
  static length_type const num_args = 1;

  typedef complex<float> T;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    typedef Layout<2, row2_type, dense, interleaved_complex> LP2;
    typedef impl::Strided<2, T, LP2, Local_map> block2_type;
    typedef Matrix<T, block2_type>            view_type;

    typedef Layout<1, row1_type, dense, interleaved_complex> LP1;
    typedef impl::Strided<1, T, LP1, Local_map> block1_type;
    typedef Vector<T, block1_type>          replica_view_type;

    // Create the data cube.
    view_type data(npulse, nrange);
    
    // Create the pulse replica
    replica_view_type replica(nrange);
    replica_view_type tmp(nrange);

    // Initialize
    data    = T();
    replica = T();

    // setup direct access to data buffers
    dda::Data<block1_type, dda::inout> data_replica(replica.block());
    dda::Data<block1_type, dda::inout> data_tmp(tmp.block());

    // Create weights array
    unsigned long log2N = ilog2(nrange);
    long flag = FFT_FAST_CONVOLUTION;
    FFT_setup setup = 0;
    unsigned long nbytes = 0;
    fft_setup( log2N, flag, &setup, &nbytes );

    // Create filter
    COMPLEX *filter = reinterpret_cast<COMPLEX *>(data_replica.ptr());
    COMPLEX *t     = reinterpret_cast<COMPLEX *>(data_tmp.ptr());
    float scale = 1;
    long f_conv = FFT_INVERSE;  // does not indicate direction, but rather 
                                // convolution as opposed to correlation
    long eflag = 0;  // no caching hints

    fcf_ciptx( &setup, filter, t, &scale, log2N, f_conv, eflag );

    // Set up convolution 
    dda::Data<block2_type, dda::out> data_data(data.block());
    COMPLEX* signal    = reinterpret_cast<COMPLEX *>(data_data.ptr());
    long signal_stride = 2*data_data.stride(1);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      for (index_type p=0; p<npulse; ++p)
      {
	// Perform fast convolution
	fcs_ciptx( &setup, filter, signal, signal_stride, t, log2N, f_conv, eflag );
      }
    }
    t1.stop();

    // CHECK RESULT
    time = t1.delta();
  }
};



/***********************************************************************
  Impl2ip: SPLIT out-of-place (tmp), interleaved fast-convolution
***********************************************************************/

template <>
struct t_fastconv_base<complex<float>, Impl2ip<split_complex> >
  : fastconv_ops
{
  static length_type const num_args = 1;

  typedef complex<float> T;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    typedef Layout<2, row2_type, dense, split_complex> LP2;
    typedef impl::Strided<2, T, LP2, Local_map> block2_type;
    typedef Matrix<T, block2_type>            view_type;

    typedef Layout<1, row1_type, dense, split_complex> LP1;
    typedef impl::Strided<1, T, LP1, Local_map> block1_type;
    typedef Vector<T, block1_type>          replica_view_type;

    typedef dda::Data<block1_type, dda::inout>::ptr_type ptr_type;

    // Create the data cube.
    view_type data(npulse, nrange);
    
    // Create the pulse replica
    replica_view_type replica(nrange);
    replica_view_type tmp(nrange);

    // Initialize
    data    = T();
    replica = T();

    // setup direct access to data buffers
    dda::Data<block1_type, dda::inout> data_replica(replica.block());
    dda::Data<block1_type, dda::out> data_tmp(tmp.block());

    // Create weights array
    unsigned long log2N = ilog2(nrange);
    long flag = FFT_FAST_CONVOLUTION;
    FFT_setup setup = 0;
    unsigned long nbytes = 0;
    fft_setup( log2N, flag, &setup, &nbytes );

    // Create filter
    ptr_type p_replica = data_replica.ptr();
    ptr_type p_tmp     = data_tmp.ptr();

    COMPLEX_SPLIT filter;
    COMPLEX_SPLIT tmpbuf;

    filter.realp = p_replica.first;
    filter.imagp = p_replica.second;
    tmpbuf.realp = p_tmp.first;
    tmpbuf.imagp = p_tmp.second;


    float scale = 1;
    long f_conv = FFT_INVERSE;  // does not indicate direction, but rather 
                                // convolution as opposed to correlation
    long eflag = 0;  // no caching hints

    fcf_ziptx(&setup, &filter, &tmpbuf, &scale, log2N, f_conv, eflag );

    // Set up convolution 
    dda::Data<block2_type, dda::out> data_data(data.block());

    ptr_type p_data = data_data.ptr();
    COMPLEX_SPLIT signal;
    signal.realp = p_data.first;
    signal.imagp = p_data.second;

    long signal_stride = 1*data_data.stride(1);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      for (index_type p=0; p<npulse; ++p)
      {
	// Perform fast convolution
	fcs_ziptx(&setup, &filter, &signal, signal_stride, &tmpbuf,
		  log2N, f_conv, eflag );
      }
    }
    t1.stop();

    // CHECK RESULT
    time = t1.delta();
  }
};



void
defaults(Loop1P& loop)
{
  loop.cal_        = 4;
  loop.start_      = 4;
  loop.stop_       = 16;
  loop.loop_start_ = 10;
  loop.user_param_ = 64;
}



int
test(Loop1P& loop, int what)
{
  typedef complex<float> C;

  length_type param1 = loop.user_param_;
  switch (what)
  {
  case   2: loop(t_fastconv_pf<C, Impl1ip<interleaved_complex> >(param1)); break;
  case   6: loop(t_fastconv_pf<C, Impl2ip<interleaved_complex> >(param1)); break;

  case  12: loop(t_fastconv_rf<C, Impl1ip<interleaved_complex> >(param1)); break;
  case  16: loop(t_fastconv_rf<C, Impl2ip<interleaved_complex> >(param1)); break;

  case 102: loop(t_fastconv_pf<C, Impl1ip<split_complex> >(param1)); break;
  case 106: loop(t_fastconv_pf<C, Impl2ip<split_complex> >(param1)); break;
   
  case 112: loop(t_fastconv_rf<C, Impl1ip<split_complex> >(param1)); break;
  case 116: loop(t_fastconv_rf<C, Impl2ip<split_complex> >(param1)); break;

  case 0:
    std::cout
      << "fastconv -- Fast Convolution\n"
      << "\n"
      << "    -2: pulse fixed, phased, interleaved complex\n"
      << "    -6: pulse fixed, interleaved, interleaved complex\n"
      << "\n"
      << "   -12: range cells fixed, phased, interleaved complex\n"
      << "   -16: range cells fixed, interleaved, interleaved complex\n"
      << "\n"
      << "  -102: pulse fixed, phased, split complex\n"
      << "  -106: pulse fixed, interleaved, split complex\n"
      << "   \n"
      << "  -112: range cells fixed, phased, split complex\n"
      << "  -116: range cells fixed, interleaved, split complex\n"
      << "\n"
      << "  Parameters: default\n"
      << "    -cal         4  calibrate with size 2^4 \n"
      << "    -start       4  start with size 2^4 \n"
      << "    -stop       16  stop  with size 2^16\n"
      << "    -loop_start 10  run problem 10 times for calibration\n"
      << "    -param      64  number of pulses / size of range\n"
      ;
  default: return 0;
  }
  return 1;
}
