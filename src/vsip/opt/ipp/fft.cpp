/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/ipp/fft.cpp
    @author  Stefan Seefeld, Nathan Myers
    @date    2006-05-05
    @brief   VSIPL++ Library: FFT wrappers and traits to bridge with 
             Intel's IPP.
*/

#include <vsip/core/config.hpp>
#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/fft/backend.hpp>
#include <vsip/core/fft/util.hpp>
#include <vsip/opt/ipp/fft.hpp>
#include <vsip/core/fns_scalar.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <ipps.h>
#include <ippi.h>
#include <complex>

namespace vsip
{
namespace impl
{
namespace ipp
{
namespace
{
inline int
int_log2(unsigned size)    // assume size = 2^n, != 0, return n.
{
  int n = 0;
  while (size >>= 1) ++n;
  return n;
}
}

// Intel's type naming convention:
// suffix:  C type:         typedef:
//
// 32f      float           Ipp32f
// 32fc     complex float   Ipp32fc
// 64f      double          Ipp64f
// 64fc     complex double  Ipp64fc

// flag values
//  IPP_FFT_NODIV_BY_ANY, IPP_FFT_DIV_INV_BY_N, IPP_FFT_DIV_FWD_BY_N

template <typename T, //< Type used in the API.
	  typename I, //< IPP's corresponding type.
	  typename P, //< Plan type.
	  IppStatus (VSIP_IMPL_IPP_CALL *Plan)(P**, int, int, IppHintAlgorithm),
	  IppStatus (VSIP_IMPL_IPP_CALL *Dispose)(P*),
	  IppStatus (VSIP_IMPL_IPP_CALL *Bufsize)(P const*, int*),
	  IppStatus (VSIP_IMPL_IPP_CALL *Forward)(I const*, I*, P const*, Ipp8u*),
	  IppStatus (VSIP_IMPL_IPP_CALL *Inverse)(I const*, I*, P const*, Ipp8u*)>
struct Driver_base_1d
{
  Driver_base_1d() : plan_(0), buffer_(0) {}
  void init(int x, int flags) VSIP_THROW((std::bad_alloc))
  {
    IppStatus result = (*Plan)(&plan_, x, flags, ippAlgHintFast);
    if (result != ippStsNoErr) VSIP_THROW(std::bad_alloc());
    buffer_ = alloc_align<Ipp8u>(VSIP_IMPL_ALLOC_ALIGNMENT, bufsize());
    if (!buffer_)
    {
      IppStatus result = (*Dispose)(plan_);
      assert(result == ippStsNoErr);
      VSIP_THROW(std::bad_alloc());
    }
  }
  void fini() VSIP_NOTHROW
  {
    free_align(buffer_);
    IppStatus result = (*Dispose)(plan_);
    assert(result == ippStsNoErr);
  }
  int bufsize() VSIP_NOTHROW
  {
    int size;
    IppStatus result = (*Bufsize)(plan_, &size);
    assert(result == ippStsNoErr);
    return size;
  }
  void forward(T const* in, T* out)
      VSIP_NOTHROW
  {
    IppStatus result = (*Forward)(reinterpret_cast<I const*>(in),
				  reinterpret_cast<I*>(out),
				  plan_, buffer_);
    assert(result == ippStsNoErr);
  }
  void inverse(T const* in, T* out)
    VSIP_NOTHROW
  {
    IppStatus result = (*Inverse)(reinterpret_cast<I const*>(in),
				  reinterpret_cast<I*>(out),
				  plan_, buffer_);
    assert(result == ippStsNoErr);
  }

  P *plan_;
  Ipp8u *buffer_;
};

template <typename T, //< Type used in the API.
	  typename I, //< IPP's corresponding type.
	  typename P, //< Plan type.
	  IppStatus (VSIP_IMPL_IPP_CALL *Plan)(P**, int, int, int, IppHintAlgorithm),
	  IppStatus (VSIP_IMPL_IPP_CALL *Dispose)(P*),
	  IppStatus (VSIP_IMPL_IPP_CALL *Bufsize)(P const*, int*),
	  IppStatus (VSIP_IMPL_IPP_CALL *Forward)(I const*, int, I*, int, P const*, Ipp8u*),
	  IppStatus (VSIP_IMPL_IPP_CALL *Inverse)(I const*, int, I*, int, P const*, Ipp8u*)>
struct Driver_base_2d
{
  Driver_base_2d() : plan_(0), buffer_(0) {}
  void init(int x, int y, int flags) VSIP_THROW((std::bad_alloc))
  {
    // Attention: IPP uses the opposite axis order, compared to VSIPL++:
    IppStatus result = (*Plan)(&plan_, y, x, flags, ippAlgHintFast);
    if (result != ippStsNoErr) VSIP_THROW(std::bad_alloc());
    buffer_ = alloc_align<Ipp8u>(VSIP_IMPL_ALLOC_ALIGNMENT, bufsize());
    if (!buffer_)
    {
      IppStatus result = (*Dispose)(plan_);
      assert(result == ippStsNoErr);
      VSIP_THROW(std::bad_alloc());
    }
  }
  void fini() VSIP_NOTHROW
  {
    free_align(buffer_);
    IppStatus result = (*Dispose)(plan_);
    assert(result == ippStsNoErr);
  }
  int bufsize() VSIP_NOTHROW
  {
    int size;
    IppStatus result = (*Bufsize)(plan_, &size);
    assert(result == ippStsNoErr);
    return size;
  }
  void forward(T const* in, unsigned in_stride,
	       T* out, unsigned out_stride)
    VSIP_NOTHROW
  {
    IppStatus result = (*Forward)(reinterpret_cast<I const*>(in), 
				  sizeof(I) * in_stride,
				  reinterpret_cast<I*>(out),
				  sizeof(I) * out_stride,
				  plan_, buffer_);
    assert(result == ippStsNoErr);
  }
  void inverse(T const* in, unsigned in_stride,
	       T* out, unsigned out_stride)
    VSIP_NOTHROW
  {
    IppStatus result = (*Inverse)(reinterpret_cast<I const*>(in),
				  sizeof(T) * in_stride,
				  reinterpret_cast<I*>(out),
				  sizeof(T) * out_stride,
				  plan_, buffer_);
    assert(result == ippStsNoErr);
  }

  P *plan_;
  Ipp8u *buffer_;
};

template <dimension_type D, //< Dimension
	  typename T,       //< Type
	  bool F>           //< Fast (Use FFT if true, DFT otherwise).
struct Driver;

// 1D, complex -> complex, float
template <bool F>
struct Driver<1, std::complex<float>, F>
  : conditional<F,
	     Driver_base_1d<std::complex<float>,
			    Ipp32fc,
			    IppsFFTSpec_C_32fc,
			    ippsFFTInitAlloc_C_32fc,
			    ippsFFTFree_C_32fc,
			    ippsFFTGetBufSize_C_32fc,
			    ippsFFTFwd_CToC_32fc,
			    ippsFFTInv_CToC_32fc>,
	     Driver_base_1d<std::complex<float>,
			    Ipp32fc,
			    IppsDFTSpec_C_32fc,
			    ippsDFTInitAlloc_C_32fc,
			    ippsDFTFree_C_32fc,
			    ippsDFTGetBufSize_C_32fc,
			    ippsDFTFwd_CToC_32fc,
			    ippsDFTInv_CToC_32fc> >::type
{
  Driver(Domain<1> const &dom) 
  {
    int size = dom.size();
    // For FFTs we actually pass the 2's exponent of the size.
    if (F) size = int_log2(size);
    this->init(size, IPP_FFT_NODIV_BY_ANY);
  }
  ~Driver() { this->fini();}
};

// Provide function wrapper to adjust to desired uniform signature.
IppStatus 
VSIP_IMPL_IPP_CALL ippiDFTInitAlloc_C_32fc(IppiDFTSpec_C_32fc**plan, int x, int y,
                                           int flag, IppHintAlgorithm hint)
{
  IppiSize roi = {x, y};
  return ippiDFTInitAlloc_C_32fc(plan, roi, flag, hint);
}


// 2D, complex -> complex, float
template <bool F>
struct Driver<2, std::complex<float>, F>
  : conditional<F,
		Driver_base_2d<std::complex<float>,
			       Ipp32fc,
			       IppiFFTSpec_C_32fc,
			       ippiFFTInitAlloc_C_32fc,
			       ippiFFTFree_C_32fc,
			       ippiFFTGetBufSize_C_32fc,
			       ippiFFTFwd_CToC_32fc_C1R,
			       ippiFFTInv_CToC_32fc_C1R>,
		Driver_base_2d<std::complex<float>,
			       Ipp32fc,
			       IppiDFTSpec_C_32fc,
			       ippiDFTInitAlloc_C_32fc,
			       ippiDFTFree_C_32fc,
			       ippiDFTGetBufSize_C_32fc,
			       ippiDFTFwd_CToC_32fc_C1R,
			       ippiDFTInv_CToC_32fc_C1R> >::type
{
  Driver(Domain<2> const &dom) 
  {
    int x = dom[0].size();
    int y = dom[1].size();
    // For FFTs we actually pass the 2's exponent of the size.
    if (F)
    {
      x = int_log2(x);
      y = int_log2(y);
    }
    this->init(x, y, IPP_FFT_NODIV_BY_ANY);
  }
  ~Driver() { this->fini();}
};

// 1D, complex -> complex, double
template <bool F>
struct Driver<1, std::complex<double>, F>
  : conditional<F,
		Driver_base_1d<std::complex<double>,
			       Ipp64fc,
			       IppsFFTSpec_C_64fc,
			       ippsFFTInitAlloc_C_64fc,
			       ippsFFTFree_C_64fc,
			       ippsFFTGetBufSize_C_64fc,
			       ippsFFTFwd_CToC_64fc,
			       ippsFFTInv_CToC_64fc>,
		Driver_base_1d<std::complex<double>,
			       Ipp64fc,
			       IppsDFTSpec_C_64fc,
			       ippsDFTInitAlloc_C_64fc,
			       ippsDFTFree_C_64fc,
			       ippsDFTGetBufSize_C_64fc,
			       ippsDFTFwd_CToC_64fc,
			       ippsDFTInv_CToC_64fc> >::type
{
  Driver(Domain<1> const &dom) 
  {
    int size = dom.size();
    // For FFTs we actually pass the 2's exponent of the size.
    if (F) size = int_log2(size);
    this->init(size, IPP_FFT_NODIV_BY_ANY);
  }
  ~Driver() { this->fini();}
};

// 1D, complex -> real, float
// 1D, real -> complex, float
template <bool F>
struct Driver<1, float, F>
  : conditional<F,
		Driver_base_1d<float,
			       Ipp32f,
			       IppsFFTSpec_R_32f,
			       ippsFFTInitAlloc_R_32f,
			       ippsFFTFree_R_32f,
			       ippsFFTGetBufSize_R_32f,
			       ippsFFTFwd_RToCCS_32f,
			       ippsFFTInv_CCSToR_32f>,
		Driver_base_1d<float,
			       Ipp32f,
			       IppsDFTSpec_R_32f,
			       ippsDFTInitAlloc_R_32f,
			       ippsDFTFree_R_32f,
			       ippsDFTGetBufSize_R_32f,
			       ippsDFTFwd_RToCCS_32f,
			       ippsDFTInv_CCSToR_32f> >::type
{
  Driver(Domain<1> const &dom) 
  {
    int size = dom.size();
    // For FFTs we actually pass the 2's exponent of the size.
    if (F) size = int_log2(size);
    this->init(size, IPP_FFT_NODIV_BY_ANY);
  }
  ~Driver() { this->fini();}
};

// 2D, complex -> real, float
// 2D, real -> complex, float
// 
// Not implemented yet. Right now we will typically fall back to another backend,
// such as fftw.

// 1D, complex -> real, double
// 1D, real -> complex, double
template <bool F>
struct Driver<1, double, F>
  : conditional<F,
		Driver_base_1d<double,
			       Ipp64f,
			       IppsFFTSpec_R_64f,
			       ippsFFTInitAlloc_R_64f,
			       ippsFFTFree_R_64f,
			       ippsFFTGetBufSize_R_64f,
			       ippsFFTFwd_RToCCS_64f,
			       ippsFFTInv_CCSToR_64f>,
		Driver_base_1d<double,
			       Ipp64f,
			       IppsDFTSpec_R_64f,
			       ippsDFTInitAlloc_R_64f,
			       ippsDFTFree_R_64f,
			       ippsDFTGetBufSize_R_64f,
			       ippsDFTFwd_RToCCS_64f,
			       ippsDFTInv_CCSToR_64f> >::type
{
  Driver(Domain<1> const &dom) 
  {
    int size = dom.size();
    // For FFTs we actually pass the 2's exponent of the size.
    if (F) size = int_log2(size);
    this->init(size, IPP_FFT_NODIV_BY_ANY);
  }
  ~Driver() { this->fini();}
};

template <dimension_type D, //< Dimension
	  typename I,       //< Input type
	  typename O,       //< Output type
	  int S,            //< Special dimension
	  bool F>           //< Fast (FFT, as opposed to DFT)
class Fft_impl;

// 1D complex -> complex FFT
template <typename T, int S, bool F>
class Fft_impl<1, std::complex<T>, std::complex<T>, S, F>
  : public fft::Fft_backend<1, std::complex<T>, std::complex<T>, S>,
    private Driver<1, std::complex<T>, F>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<1> const &dom, rtype /*scale*/)
    : Driver<1, std::complex<T>, F>(dom)
  {
  }
  virtual void in_place(ctype *inout, stride_type s, length_type /*l*/)
  {
    assert(s == 1);
    if (S == fft_fwd) this->forward(inout, inout);
    else this->inverse(inout, inout);
  }
  virtual void out_of_place(ctype *in, stride_type in_s,
			    ctype *out, stride_type out_s,
			    length_type /*l*/)
  {
    assert(in_s == 1 && out_s == 1);
    if (S == fft_fwd) this->forward(in, out);
    else this->inverse(in, out);
  }
};

// 1D real -> complex FFT
template <typename T, bool F>
class Fft_impl<1, T, std::complex<T>, 0, F>
  : public fft::Fft_backend<1, T, std::complex<T>, 0>,
    private Driver<1, T, F>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<1> const &dom, rtype /*scale*/)
    : Driver<1, T, F>(dom)
  {
  }
  virtual void out_of_place(rtype *in, stride_type in_s,
			    ctype *out, stride_type out_s,
			    length_type /*l*/)
  {
    assert(in_s == 1 && out_s == 1);
    this->forward(in, reinterpret_cast<rtype*>(out));
  }
};

// 1D complex -> real FFT
template <typename T, bool F>
class Fft_impl<1, std::complex<T>, T, 0, F>
  : public fft::Fft_backend<1, std::complex<T>, T, 0>,
    private Driver<1, T, F>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<1> const &dom, rtype /*scale*/)
    : Driver<1, T, F>(dom)
  {
  }
  virtual void out_of_place(ctype *in, stride_type in_s,
			    rtype *out, stride_type out_s,
			    length_type /*l*/)
  {
    assert(in_s == 1 && out_s == 1);
    this->inverse(reinterpret_cast<rtype*>(in), out);
  }
};

// 2D complex -> complex FFT
template <typename T, int S, bool F>
class Fft_impl<2, std::complex<T>, std::complex<T>, S, F>
  : public fft::Fft_backend<2, std::complex<T>, std::complex<T>, S>,
    private Driver<2, std::complex<T>, F>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<2> const &dom, rtype /*scale*/)
    : Driver<2, std::complex<T>, F>(dom)
  {
  }
  virtual void in_place(ctype *inout,
			stride_type r_stride, stride_type c_stride,
			length_type /*rows*/, length_type /*cols*/)
  {
    if (c_stride == 1)
    {
      if (S == fft_fwd) this->forward(inout, r_stride, inout, r_stride);
      else this->inverse(inout, r_stride, inout, r_stride);
    }
    else
    {
      if (S == fft_fwd) this->forward(inout, c_stride, inout, c_stride);
      else this->inverse(inout, c_stride, inout, c_stride);
    }
  }
  virtual void out_of_place(ctype *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    ctype *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type /*rows*/, length_type /*cols*/)
  {
    if (in_c_stride == 1)
    {
      assert(out_c_stride == 1);
      if (S == fft_fwd) this->forward(in, in_r_stride, out, out_r_stride);
      else this->inverse(in, in_r_stride, out, out_r_stride);
    }
    else
    {
      assert(in_r_stride == 1 && out_r_stride == 1);
      if (S == fft_fwd) this->forward(in, in_c_stride, out, out_c_stride);
      else this->inverse(in, in_c_stride, out, out_c_stride);
    }
  }
};

template <typename I, //< Input type
	  typename O, //< Output type
	  int A,      //< Axis
	  int D,      //< Direction
	  bool F>     //< Fast (FFT as opposed to DFT)
class Fftm_impl;

// real -> complex FFTM
template <typename T, int A, bool F>
class Fftm_impl<T, std::complex<T>, A, fft_fwd, F>
  : public fft::Fftm_backend<T, std::complex<T>, A, fft_fwd>,
    private Driver<1, T, F>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fftm_impl(Domain<2> const &dom, rtype /*scalar*/)
    : Driver<1, T, F>(dom[1 - A]),
      mult_(dom[A].size())
  {
  }
  virtual void out_of_place(rtype *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    ctype *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    if (A == vsip::col)
    {
      assert(cols == mult_);
      for (length_type m = 0; m != mult_;
	   ++m, in += in_c_stride, out += out_c_stride)
	this->forward(in, reinterpret_cast<rtype*>(out));
    }
    else
    {
      assert(rows == mult_);
      for (length_type m = 0; m != mult_;
	   ++m, in += in_r_stride, out += out_r_stride)
	this->forward(in, reinterpret_cast<rtype*>(out));
    }
  }

  length_type mult_;
};

// complex -> real FFTM
template <typename T, int A, bool F>
class Fftm_impl<std::complex<T>, T, A, fft_inv, F>
  : public fft::Fftm_backend<std::complex<T>, T, A, fft_inv>,
    private Driver<1, T, F>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fftm_impl(Domain<2> const &dom, rtype /*scalar*/)
    : Driver<1, T, F>(dom[1 - A]),
      mult_(dom[A].size())
  {
  }
  virtual void out_of_place(ctype *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    rtype *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    if (A == vsip::col)
    {
      assert(cols == mult_);
      for (length_type m = 0; m != mult_;
	   ++m, in += in_c_stride, out += out_c_stride)
	this->inverse(reinterpret_cast<rtype*>(in), out);
    }
    else
    {
      assert(rows == mult_);
      for (length_type m = 0; m != mult_;
	   ++m, in += in_r_stride, out += out_r_stride)
	this->inverse(reinterpret_cast<rtype*>(in), out);
    }
  }

  length_type mult_;
};

// complex -> complex FFTM
template <typename T, int A, int D, bool F>
class Fftm_impl<std::complex<T>, std::complex<T>, A, D, F>
  : public fft::Fftm_backend<std::complex<T>, std::complex<T>, A, D>,
    private Driver<1, std::complex<T>, F>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fftm_impl(Domain<2> const &dom, rtype /*scale*/)
    : Driver<1, std::complex<T>, F>(dom[1 - A]),
      mult_(dom[A].size())
  {
  }

  virtual void in_place(ctype *inout,
			stride_type r_stride, stride_type c_stride,
			length_type rows, length_type cols)
  {
    if (A == vsip::col)
    {
      assert(cols <= mult_);
      for (length_type m = 0; m != cols; ++m, inout += c_stride)
	if (D == fft_fwd) this->forward(inout, inout);
	else this->inverse(inout, inout);
    }
    else
    {
      assert(rows <= mult_);
      for (length_type m = 0; m != rows; ++m, inout += r_stride)
	if (D == fft_fwd) this->forward(inout, inout);
	else this->inverse(inout, inout);
    }
  }

  virtual void out_of_place(ctype *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    ctype *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    if (A == vsip::col)
    {
      assert(cols <= mult_);
      for (length_type m = 0; m != cols;
	   ++m, in += in_c_stride, out += out_c_stride)
	if (D == fft_fwd) this->forward(in, out);
	else this->inverse(in, out);
    }
    else
    {
      assert(rows <= mult_);
      for (length_type m = 0; m != rows;
	   ++m, in += in_r_stride, out += out_r_stride)
	if (D == fft_fwd) this->forward(in, out);
	else this->inverse(in, out);
    }
  }

  length_type mult_;
};

#define FFT_DEF(D, I, O, S)		                \
template <>                                             \
std::auto_ptr<fft::Fft_backend<D, I, O, S> >		\
create(Domain<D> const &dom,                            \
       fft::Fft_backend<D, I, O, S>::scalar_type scale, \
       bool fast)                                       \
{                                                       \
  if (fast)						\
    return std::auto_ptr<fft::Fft_backend<D, I, O, S> >	\
      (new Fft_impl<D, I, O, S, true>(dom, scale));	\
  else                                                  \
    return std::auto_ptr<fft::Fft_backend<D, I, O, S> >	\
      (new Fft_impl<D, I, O, S, false>(dom, scale));	\
}

FFT_DEF(1, float, std::complex<float>, 0)
FFT_DEF(1, std::complex<float>, float, 0)
FFT_DEF(1, std::complex<float>, std::complex<float>, fft_fwd)
FFT_DEF(1, std::complex<float>, std::complex<float>, fft_inv)
FFT_DEF(1, double, std::complex<double>, 0)
FFT_DEF(1, std::complex<double>, double, 0)
FFT_DEF(1, std::complex<double>, std::complex<double>, fft_fwd)
FFT_DEF(1, std::complex<double>, std::complex<double>, fft_inv)

// TODO:
// FFT_DEF(2, float, std::complex<float>, 0)
// FFT_DEF(2, float, std::complex<float>, 1)
FFT_DEF(2, std::complex<float>, std::complex<float>, fft_fwd)
FFT_DEF(2, std::complex<float>, std::complex<float>, fft_inv)

// Not supported by IPP:
// FFT_DEF(2, double, std::complex<double>, 0)
// FFT_DEF(2, double, std::complex<double>, 1)
// FFT_DEF(2, std::complex<double>, double, 0)
// FFT_DEF(2, std::complex<double>, double, 1)
// FFT_DEF(2, std::complex<double>, std::complex<double>, fft_fwd)
// FFT_DEF(2, std::complex<double>, std::complex<double>, fft_inv)

#undef FFT_DEF

#define FFTM_DEF(I, O, A, D)			        \
template <>                                             \
std::auto_ptr<fft::Fftm_backend<I, O, A, D> >		\
create(Domain<2> const &dom,                            \
       vsip::impl::scalar_of<I>::type scale,            \
       bool fast)					\
{                                                       \
  if (fast)                                             \
    return std::auto_ptr<fft::Fftm_backend<I, O, A, D> >\
      (new Fftm_impl<I, O, A, D, true>(dom, scale));	\
  else                                                  \
    return std::auto_ptr<fft::Fftm_backend<I, O, A, D> >\
      (new Fftm_impl<I, O, A, D, false>(dom, scale));   \
}

FFTM_DEF(float, std::complex<float>, 0, fft_fwd)
FFTM_DEF(float, std::complex<float>, 1, fft_fwd)
FFTM_DEF(std::complex<float>, float, 0, fft_inv)
FFTM_DEF(std::complex<float>, float, 1, fft_inv)
FFTM_DEF(std::complex<float>, std::complex<float>, 0, fft_fwd)
FFTM_DEF(std::complex<float>, std::complex<float>, 1, fft_fwd)
FFTM_DEF(std::complex<float>, std::complex<float>, 0, fft_inv)
FFTM_DEF(std::complex<float>, std::complex<float>, 1, fft_inv)

FFTM_DEF(double, std::complex<double>, 0, fft_fwd)
FFTM_DEF(double, std::complex<double>, 1, fft_fwd)
FFTM_DEF(std::complex<double>, double, 0, fft_inv)
FFTM_DEF(std::complex<double>, double, 1, fft_inv)
FFTM_DEF(std::complex<double>, std::complex<double>, 0, fft_fwd)
FFTM_DEF(std::complex<double>, std::complex<double>, 1, fft_fwd)
FFTM_DEF(std::complex<double>, std::complex<double>, 0, fft_inv)
FFTM_DEF(std::complex<double>, std::complex<double>, 1, fft_inv)

#undef FFTM_DEF

} // namespace vsip::impl::ipp
} // namespace vsip::impl
} // namespace vsip
