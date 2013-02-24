/* Copyright (c) 2006, 2007 by CodeSourcery.  All rights reserved. */

/** @file    benchmarks/fftw3/fft.cpp
    @author  Jules Bergmann
    @date    2006-10-19
    @brief   VSIPL++ Library: Benchmark for FFTW3 FFT.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <fftw3.h>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>

#include "benchmarks.hpp"



/***********************************************************************
  Definitions
***********************************************************************/

using namespace vsip;

using vsip::impl::Ext_data;
using vsip::impl::Cmplx_inter_fmt;
using vsip::impl::Cmplx_split_fmt;
using vsip::impl::Stride_unit_dense;
using vsip::impl::SYNC_IN;
using vsip::impl::SYNC_OUT;
using vsip::impl::SYNC_INOUT;



inline unsigned long
ilog2(length_type size)    // assume size = 2^n, != 0, return n.
{
  unsigned int n = 0;
  while (size >>= 1) ++n;
  return n;
}


float
fft_ops(length_type len)
{
  return 5.0 * std::log((double)len) / std::log(2.0);
}



/***********************************************************************
  FFTW3 Traits
***********************************************************************/

template <typename T>
struct Fftw_traits;

#if VSIP_IMPL_PROVIDE_FFT_FLOAT
template <>
struct Fftw_traits<complex<float> >
{
  typedef fftwf_plan plan_type;

  // Plan 1-D interleaved-complex FFT
  static plan_type
  plan_dft_1d(
    length_type           size,
    complex<float> const* in,
    complex<float>*       out,
    int                   dir,
    int                   flags)
  {
    return fftwf_plan_dft_1d(size,
			   (fftwf_complex*)in,
			   (fftwf_complex*)out,
			   dir, flags);
  }

  // Plan N-D interleaved-complex FFT
  static plan_type
  plan_dft(
    dimension_type        D,
    int const*            size,
    complex<float> const* in,
    complex<float>*       out,
    int                   dir,
    int                   flags)
  {
    return fftwf_plan_dft(D, size,
			   (fftwf_complex*)in,
			   (fftwf_complex*)out,
			   dir, flags);
  }

  // Plan 1-D split-complex FFT
  static plan_type
  plan_dft_1d(
    length_type                size,
    std::pair<float*, float*> const& in,
    std::pair<float*, float*> const& out,
    int                        dir,
    int                        flags)
  {
    assert(dir == FFTW_FORWARD);
    fftw_iodim dims[1];

    dims[0].n = size;
    dims[0].is = 1;
    dims[0].os = 1;

    return fftwf_plan_guru_split_dft(
		1,		// rank
		dims,
		0,		// howmany_rank
		0,		// howmany_dims
		in.first, in.second,
		out.first, out.second,
		flags);
  }

  static void
  execute(plan_type p)
  {
    fftwf_execute(p);
  }

  static void
  destroy_plan(plan_type p)
  {
    fftwf_destroy_plan(p);
  }

  static void
  forget_wisdom()
  {
    fftwf_forget_wisdom();
  }

  static void
  export_wisdom_to_file(FILE* fd)
  {
    fftwf_export_wisdom_to_file(fd);
  }
};
#endif // VSIP_IMPL_PROVIDE_FFT_FLOAT


#if VSIP_IMPL_PROVIDE_FFT_DOUBLE
template <>
struct Fftw_traits<complex<double> >
{
  typedef fftw_plan plan_type;

  static plan_type
  plan_dft_1d(
    length_type            size,
    complex<double> const* in,
    complex<double>*       out,
    int                    dir,
    int                    flags)
  {
    return fftw_plan_dft_1d(size,
			   (fftw_complex*)in,
			   (fftw_complex*)out,
			   dir, flags);
  }

  static plan_type
  plan_dft_1d(
    length_type                size,
    std::pair<double*, double*> const& in,
    std::pair<double*, double*> const& out,
    int                        dir,
    int                        flags)
  {
    assert(dir == FFTW_FORWARD);
    fftw_iodim dims[1];

    dims[0].n = size;
    dims[0].is = 1;
    dims[0].os = 1;

    return fftw_plan_guru_split_dft(
		1,		// rank
		dims,
		0,		// howmany_rank
		0,		// howmany_dims
		in.first, in.second,
		out.first, out.second,
		flags);
  }

  static void
  execute(plan_type p)
  {
    fftw_execute(p);
  }

  static void
  destroy_plan(plan_type p)
  {
    fftw_destroy_plan(p);
  }

  static void
  forget_wisdom()
  {
    fftw_forget_wisdom();
  }

  static void
  export_wisdom_to_file(FILE* fd)
  {
    fftw_export_wisdom_to_file(fd);
  }
};
#endif // VSIP_IMPL_PROVIDE_FFT_DOUBLE


template <typename T>
struct fft_base
{
  fft_base(bool save_wisdom)
    : save_wisdom_(save_wisdom)
  {}

  void reset()
  {
    if (save_wisdom_)
      Fftw_traits<T>::forget_wisdom();
  }

  void save(length_type size)
  {
    if (save_wisdom_)
    {
      char file[80];
      sprintf(file, "wisdom.%d", (int)size);
      FILE* fd = fopen(file, "w");
      Fftw_traits<T>::export_wisdom_to_file(fd);
      fclose(fd);
    }
  }

  // Member data.
  bool save_wisdom_;
};


/***********************************************************************
  Out-of-place Benchmark driver
***********************************************************************/

template <typename T,
	  typename ComplexFmt>
struct t_fft_op;

template <typename T>
struct t_fft_op<T, Cmplx_inter_fmt> : Benchmark_base, fft_base<T>
{
  typedef Cmplx_inter_fmt ComplexFmt;

  typedef Fftw_traits<T> traits;

  char const* what() { return "t_fft_op"; }
  float ops_per_point(length_type len)  { return fft_ops(len); }
  int riob_per_point(length_type) { return 1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return 1*(int)sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef impl::Layout<1, row1_type, Stride_unit_dense, ComplexFmt> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A  (size, T());
    Vector<T, block_type>   Z  (size);

    
    vsip::impl::profile::Timer t1;

    this->reset();
    {
      Ext_data<block_type> ext_A(A.block(), SYNC_IN);
      Ext_data<block_type> ext_Z(Z.block(), SYNC_OUT);

      typename traits::plan_type p;

      p = traits::plan_dft_1d(size,
			      ext_A.data(),
			      ext_Z.data(),
			      FFTW_FORWARD, flags_);

      // FFTW3 may scribble

      A = T(1);
    
      t1.start();
      for (index_type l=0; l<loop; ++l)
	traits::execute(p);
      t1.stop();

      traits::destroy_plan(p);
    }
    this->save(size);
    
    if (!equal(Z.get(0), T(scale_ ? 1 : size)))
    {
      std::cout << "t_fft_op: ERROR" << std::endl;
      std::cout << "  got     : " << Z.get(0) << std::endl;
      std::cout << "  expected: " << T(scale_ ? 1 : size) << std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  t_fft_op(bool scale, int flags, bool save_wisdom)
    : fft_base<T>(save_wisdom),
      scale_(scale),
      flags_(flags)
  {}

  // Member data
  bool scale_;
  int  flags_;
};



template <typename T>
struct t_fft_op<T, Cmplx_split_fmt> : Benchmark_base, fft_base<T>
{
  typedef Cmplx_split_fmt ComplexFmt;

  typedef Fftw_traits<T> traits;

  char const* what() { return "t_fft_op"; }
  float ops_per_point(length_type len)  { return fft_ops(len); }
  int riob_per_point(length_type) { return 1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return 1*(int)sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef impl::Layout<1, row1_type, Stride_unit_dense, ComplexFmt> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A  (size, T());
    Vector<T, block_type>   Z  (size);

    
    vsip::impl::profile::Timer t1;

    this->reset();
    {
      Ext_data<block_type> ext_A(A.block(), SYNC_IN);
      Ext_data<block_type> ext_Z(Z.block(), SYNC_OUT);

      typename traits::plan_type p;

      p = traits::plan_dft_1d(size,
			      ext_A.data(),
			      ext_Z.data(),
			      FFTW_FORWARD, flags_);

      // FFTW3 may scribble

      A = T(1);
    
      t1.start();
      for (index_type l=0; l<loop; ++l)
	traits::execute(p);
      t1.stop();

      traits::destroy_plan(p);
    }
    this->save(size);
    
    if (!equal(Z.get(0), T(scale_ ? 1 : size)))
    {
      std::cout << "t_fft_op: ERROR" << std::endl;
      std::cout << "  got     : " << Z.get(0) << std::endl;
      std::cout << "  expected: " << T(scale_ ? 1 : size) << std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  t_fft_op(bool scale, int flags, bool save_wisdom)
    : fft_base<T>(save_wisdom),
      scale_(scale),
      flags_(flags)
  {}

  // Member data
  bool scale_;
  int  flags_;
};



/***********************************************************************
  Out-of-place Benchmark driver (variant -- use N-D planner)
***********************************************************************/

template <typename T,
	  typename ComplexFmt>
struct t_fft_op_nd;

template <typename T>
struct t_fft_op_nd<T, Cmplx_inter_fmt> : Benchmark_base, fft_base<T>
{
  typedef Cmplx_inter_fmt ComplexFmt;

  typedef Fftw_traits<T> traits;

  char const* what() { return "t_fft_op_nd"; }
  float ops_per_point(length_type len)  { return fft_ops(len); }
  int riob_per_point(length_type) { return 1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return 1*(int)sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef impl::Layout<1, row1_type, Stride_unit_dense, ComplexFmt> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A  (size, T());
    Vector<T, block_type>   Z  (size);

    
    vsip::impl::profile::Timer t1;

    this->reset();
    {
      Ext_data<block_type> ext_A(A.block(), SYNC_IN);
      Ext_data<block_type> ext_Z(Z.block(), SYNC_OUT);

      typename traits::plan_type p;

      int size_array[1];
      size_array[0] = size;

      p = traits::plan_dft(1, size_array,
			   ext_A.data(),
			   ext_Z.data(),
			   FFTW_FORWARD, flags_);

      // FFTW3 may scribble

      A = T(1);
    
      t1.start();
      for (index_type l=0; l<loop; ++l)
	traits::execute(p);
      t1.stop();

      traits::destroy_plan(p);
    }
    this->save(size);
    
    if (!equal(Z.get(0), T(scale_ ? 1 : size)))
    {
      std::cout << "t_fft_op_nd: ERROR" << std::endl;
      std::cout << "  got      : " << Z.get(0) << std::endl;
      std::cout << "  expected : " << T(scale_ ? 1 : size) << std::endl;
    }
    test_assert(equal(Z.get(0), T(scale_ ? 1 : size)));
    
    time = t1.delta();
  }

  t_fft_op_nd(bool scale, int flags, bool save_wisdom)
    : fft_base<T>(save_wisdom),
      scale_(scale),
      flags_(flags)
  {}

  // Member data
  bool scale_;
  int  flags_;
};



/***********************************************************************
  In-place Benchmark driver
***********************************************************************/

template <typename T,
	  typename ComplexFmt>
struct t_fft_ip;

template <typename T>
struct t_fft_ip<T, Cmplx_inter_fmt> : Benchmark_base, fft_base<T>
{
  typedef Cmplx_inter_fmt ComplexFmt;

  typedef Fftw_traits<T> traits;

  char const* what() { return "t_fft_ip"; }
  float ops_per_point(length_type len)  { return fft_ops(len); }
  int riob_per_point(length_type) { return 1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return 1*(int)sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef impl::Layout<1, row1_type, Stride_unit_dense, ComplexFmt> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A  (size, T());
    
    vsip::impl::profile::Timer t1;

    this->reset();
    {
      Ext_data<block_type> ext_A(A.block(), SYNC_INOUT);

      typename traits::plan_type p;

      p = traits::plan_dft_1d(size,
			      ext_A.data(),
			      ext_A.data(),
			      FFTW_FORWARD, flags_);

      // FFTW3 may scribble

      A = T(0);
    
      t1.start();
      for (index_type l=0; l<loop; ++l)
	traits::execute(p);
      t1.stop();

      A = T(1);
      traits::execute(p);

      traits::destroy_plan(p);
    }
    this->save(size);
    
    if (!equal(A.get(0), T(scale_ ? 1 : size)))
    {
      std::cout << "t_fft_ip: ERROR" << std::endl;
      std::cout << "  got     : " << A.get(0) << std::endl;
      std::cout << "  expected: " << T(scale_ ? 1 : size) << std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  t_fft_ip(bool scale, int flags, bool save_wisdom)
    : fft_base<T>(save_wisdom),
      scale_(scale),
      flags_(flags)
  {}

  // Member data
  bool scale_;
  int  flags_;
};



template <typename T>
struct t_fft_ip<T, Cmplx_split_fmt> : Benchmark_base, fft_base<T>
{
  typedef Cmplx_split_fmt ComplexFmt;

  typedef Fftw_traits<T> traits;

  char const* what() { return "t_fft_ip"; }
  float ops_per_point(length_type len)  { return fft_ops(len); }
  int riob_per_point(length_type) { return 1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return 1*(int)sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef impl::Layout<1, row1_type, Stride_unit_dense, ComplexFmt> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A  (size, T());
    
    vsip::impl::profile::Timer t1;

    this->reset();
    {
      Ext_data<block_type> ext_A(A.block(), SYNC_INOUT);

      typename traits::plan_type p;

      p = traits::plan_dft_1d(size,
			      ext_A.data(),
			      ext_A.data(),
			      FFTW_FORWARD, flags_);

      // FFTW3 may scribble

      A = T(0);
    
      t1.start();
      for (index_type l=0; l<loop; ++l)
	traits::execute(p);
      t1.stop();

      A = T(1);
      traits::execute(p);

      traits::destroy_plan(p);
    }
    this->save(size);

    if (!equal(A.get(0), T(scale_ ? 1 : size)))
    {
      std::cout << "t_fft_ip: ERROR" << std::endl;
      std::cout << "  got     : " << A.get(0) << std::endl;
      std::cout << "  expected: " << T(scale_ ? 1 : size) << std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  t_fft_ip(bool scale, int flags, bool save_wisdom)
    : fft_base<T>(save_wisdom),
      scale_(scale),
      flags_(flags)
  {}

  // Member data
  bool scale_;
  int  flags_;
};





void
defaults(Loop1P& loop)
{
  loop.start_ = 4;
  loop.user_param_ = 0;
}



int
test(Loop1P& loop, int what)
{
  typedef complex<float>  Cf;
  typedef complex<double> Cd;
  typedef Cmplx_inter_fmt Cif;
  typedef Cmplx_split_fmt Csf;

  int est = FFTW_ESTIMATE;
  int msr = FFTW_MEASURE;
  int pnt = FFTW_PATIENT;
  int exh = FFTW_EXHAUSTIVE;

  bool sw = loop.user_param_ == 1; // save wisdom

  switch (what)
  {
#if VSIP_IMPL_PROVIDE_FFT_FLOAT
  case   1: loop(t_fft_op<Cf, Cif>(false, est, sw)); break;
  case   2: loop(t_fft_ip<Cf, Cif>(false, est, sw)); break;
  case   4: loop(t_fft_op<Cf, Cif>(false, est | FFTW_UNALIGNED, sw)); break;
  // case  5: loop(t_fft_op<complex<float>, Cmplx_inter_fmt>(true)); break;
  // case  6: loop(t_fft_ip<complex<float>, Cmplx_inter_fmt>(true)); break;

  case  11: loop(t_fft_op<Cf, Cif>(false, msr, sw)); break;
  case  12: loop(t_fft_ip<Cf, Cif>(false, msr, sw)); break;
  case  14: loop(t_fft_op<Cf, Cif>(false, msr | FFTW_UNALIGNED, sw)); break;
  case  15: loop(t_fft_op<Cf, Cif>(false, msr | FFTW_PRESERVE_INPUT, sw)); break;
  case  16: loop(t_fft_op_nd<Cf, Cif>(false, msr | FFTW_PRESERVE_INPUT, sw)); break;

  case  21: loop(t_fft_op<Cf, Cif>(false, pnt, sw)); break;
  case  22: loop(t_fft_ip<Cf, Cif>(false, pnt, sw)); break;
  case  24: loop(t_fft_op<Cf, Cif>(false, pnt | FFTW_UNALIGNED, sw)); break;

  case  31: loop(t_fft_op<Cf, Cif>(false, exh, sw)); break;
  case  32: loop(t_fft_ip<Cf, Cif>(false, exh, sw)); break;
  case  34: loop(t_fft_op<Cf, Cif>(false, exh | FFTW_UNALIGNED, sw)); break;

  case  51: loop(t_fft_op<Cf, Csf>(false, est, sw)); break;
  case  52: loop(t_fft_ip<Cf, Csf>(false, est, sw)); break;
  case  61: loop(t_fft_op<Cf, Csf>(false, msr, sw)); break;
  case  62: loop(t_fft_ip<Cf, Csf>(false, msr, sw)); break;
  case  71: loop(t_fft_op<Cf, Csf>(false, pnt, sw)); break;
  case  72: loop(t_fft_ip<Cf, Csf>(false, pnt, sw)); break;
#endif // VSIP_IMPL_PROVIDE_FFT_FLOAT

#if VSIP_IMPL_PROVIDE_FFT_DOUBLE
  case 101: loop(t_fft_op<Cd, Cif>(false, est, sw)); break;
  case 102: loop(t_fft_ip<Cd, Cif>(false, est, sw)); break;
  case 104: loop(t_fft_op<Cd, Cif>(false, est | FFTW_UNALIGNED, sw)); break;
  case 111: loop(t_fft_op<Cd, Cif>(false, msr, sw)); break;
  case 112: loop(t_fft_ip<Cd, Cif>(false, msr, sw)); break;
  case 114: loop(t_fft_op<Cd, Cif>(false, msr | FFTW_UNALIGNED, sw)); break;
  case 121: loop(t_fft_op<Cd, Cif>(false, pnt, sw)); break;
  case 122: loop(t_fft_ip<Cd, Cif>(false, pnt, sw)); break;
  case 124: loop(t_fft_op<Cd, Cif>(false, pnt | FFTW_UNALIGNED, sw)); break;
  case 131: loop(t_fft_op<Cd, Cif>(false, exh, sw)); break;
  case 132: loop(t_fft_ip<Cd, Cif>(false, exh, sw)); break;
  case 134: loop(t_fft_op<Cd, Cif>(false, exh | FFTW_UNALIGNED, sw)); break;
#endif // VSIP_IMPL_PROVIDE_FFT_DOUBLE

  case 0:
    std::cout
      << "fftw3/fft -- FFTW3 Fft (fast fourier transform)\n"
      << "Single precision, Interleaved complex\n"
      << " Planning effor: estimate:\n"
      << "   -1 -- op: out-of-place CC fwd fft\n"
      << "   -2 -- ip: in-place     CC fwd fft\n"
      << "   -4 -- op: out-of-place CC fwd fft + UNALIGNED\n"
      << " Planning effor: measure:\n"
      << "  -11 -- op: out-of-place CC fwd fft\n"
      << "  -12 -- ip: in-place     CC fwd fft\n"
      << "  -14 -- op: out-of-place CC fwd fft + UNALIGNED\n"
      << "  -15 -- op: out-of-place CC fwd fft + PRESERVE_INPUT\n"
      << "  -16 -- op: out-of-place CC fwd fft + PRESERVE_INPUT + ND\n"
      << " Planning effor: patient:\n"
      << "  -21 -- op: out-of-place CC fwd fft\n"
      << "  -22 -- ip: in-place     CC fwd fft\n"
      << "  -24 -- op: out-of-place CC fwd fft + UNALIGNED\n"
      << " Planning effor: exhaustive:\n"
      << "  -31 -- op: out-of-place CC fwd fft\n"
      << "  -32 -- ip: in-place     CC fwd fft\n"
      << "  -34 -- op: out-of-place CC fwd fft + UNALIGNED\n"
      << "\n"
      << "Single precision, Split complex\n"
      << " Planning effor: estimate:\n"
      << "  -51 -- op: out-of-place CC fwd fft\n"
      << "  -52 -- ip: in-place     CC fwd fft\n"
      << " Planning effor: measure:\n"
      << "  -61 -- op: out-of-place CC fwd fft\n"
      << "  -62 -- ip: in-place     CC fwd fft\n"
      << " Planning effor: patient:\n"
      << "  -71 -- op: out-of-place CC fwd fft\n"
      << "  -72 -- ip: in-place     CC fwd fft\n"
      ;

  default: return 0;
  }

  return 1;
}
