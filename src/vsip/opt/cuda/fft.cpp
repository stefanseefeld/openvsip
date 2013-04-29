/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cuda/fft.cpp
    @author  Don McCoy
    @date    2009-02-26
    @brief   VSIPL++ Library: FFT wrappers and traits to bridge with 
             NVidia's CUDA FFT library.
*/

#include <complex>
#include <cufft.h>
#include <cuda_runtime.h>

#include <vsip/core/config.hpp>
#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/equal.hpp>
#include <vsip/core/fft/backend.hpp>
#include <vsip/core/fft/util.hpp>
#include <vsip/core/fns_scalar.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/opt/cuda/bindings.hpp>
#include <vsip/opt/cuda/fft.hpp>
#include <vsip/opt/cuda/vmmul.hpp>

#ifndef NDEBUG
#include <iostream>

#define ASSERT_CUFFT_OK(result)					\
{								\
  if (result != 0)						\
    std::cerr << "CUFFT problem encountered (error "		\
	      << result << ")" << std::endl;			\
  assert(result == 0);						\
}

#else
#define ASSERT_CUFFT_OK(r)

#endif // CUDA_DEBUG



namespace vsip
{
namespace impl
{
namespace cuda
{


// CUDA's type naming convention:
//   C type:         typedef:
//
//   float           cufftReal
//   complex<float>  cufftComplex


/// The base class mini 'driver' for both 1-D FFTs and FFTMs
template <typename  T>
struct Driver_base_1d
{
  Driver_base_1d() {}

  typedef typename scalar_of<T>::type scalar_type;

  // 1-D size threshold is 512MB as of CUDA Tookit 3.1.
  //  See CUDA Toolkit 3.1 release-notes dated 6/2010.
  static size_t const max_limit_bytes = 536870912;

  /// 1-D Driver initialization function
  ///   fft_type   Specifies real or complex FFT
  ///   n          Number of columns (FFT length)
  ///   m          Number of rows (defaults to one for single FFTs)
  void init(cufftType fft_type, size_t n, size_t m = 1, 
    scalar_type scale = 1.f)  VSIP_THROW((std::bad_alloc))
  {
    // Determine buffer memory requirements and batching parameters
    size_t n2 = n / 2 + 1;

    batch_size_main_ = m;

    if (fft_type == CUFFT_R2C)
    {
      insize_ = m * n * sizeof(cufftReal);
      outsize_ = m * n2 * sizeof(cufftComplex);

      if (insize_ > max_limit_bytes)
        batch_size_main_ = max_limit_bytes / (n * sizeof(cufftReal));
    }
    else if (fft_type == CUFFT_C2R)
    {
      insize_ = m * n2 * sizeof(cufftComplex);
      outsize_ = m * n * sizeof(cufftReal);

      if (insize_ > max_limit_bytes)
        batch_size_main_ = max_limit_bytes / (n2 * sizeof(cufftComplex));
    }
    else if (fft_type == CUFFT_C2C)
    {
      insize_ = m * n * sizeof(cufftComplex);
      outsize_ = m * n * sizeof(cufftComplex);

      if (insize_ > max_limit_bytes)
        batch_size_main_ = max_limit_bytes / (n * sizeof(cufftComplex));
    }

    num_batches_ = m / batch_size_main_;
    batch_size_remainder_ = m - num_batches_ * batch_size_main_;

    // Create a plan
    cufftResult result = 
      cufftPlan1d(&plan_main_, n, fft_type, batch_size_main_);
    ASSERT_CUFFT_OK(result);

    // Create a plan for the clean-up FFT on the remaining batch if applicable
    if (batch_size_remainder_)
    {
      cufftResult result =
        cufftPlan1d(&plan_remainder_, n, fft_type, batch_size_remainder_);
      ASSERT_CUFFT_OK(result);
    }

    // Save the sizes and scale factor
    rows_ = m;
    cols_ = n;
    scale_ = scale;
  }

  void fini()  VSIP_NOTHROW
  {
    cufftResult result = cufftDestroy(plan_main_);
    ASSERT_CUFFT_OK(result);

    if (batch_size_remainder_)
    {
      cufftResult result = cufftDestroy(plan_remainder_);
      ASSERT_CUFFT_OK(result);
    }
  }


  /// real->complex
  void forward(T const* in, std::complex<T>* out)  VSIP_NOTHROW
  {

    for (size_t i = 0; i < num_batches_; ++i)
    {
      cufftResult result = cufftExecR2C(plan_main_, 
        reinterpret_cast<cufftReal*>(const_cast<T*>(in + i * batch_size_main_ * cols_)),
        reinterpret_cast<cufftComplex*>(out + i * batch_size_main_ * (cols_ / 2 + 1)));
      ASSERT_CUFFT_OK(result);
    }

    if (batch_size_remainder_)
    {
      cufftResult result = cufftExecR2C(plan_remainder_, 
        reinterpret_cast<cufftReal*>(const_cast<T*>(in + num_batches_ * batch_size_main_ * cols_)),
        reinterpret_cast<cufftComplex*>(out + num_batches_ * batch_size_main_ * (cols_ / 2 + 1)));
      ASSERT_CUFFT_OK(result);
    }

    if(scale_ != scalar_type(1.))
      smmul(scale_, out, out, rows_, cols_);
  }

  /// complex->real
  void forward(std::complex<T> const* in, T* out)  VSIP_NOTHROW
  {
    for (size_t i = 0; i < num_batches_; ++i)
    {
      cufftResult result = cufftExecC2R(plan_main_, 
        reinterpret_cast<cufftComplex*>(const_cast<std::complex<T>*>(in + i * batch_size_main_ * (cols_ / 2 + 1))),
        reinterpret_cast<cufftReal*>(out + i * batch_size_main_ * cols_));
      ASSERT_CUFFT_OK(result);
    }

    if (batch_size_remainder_)
    {
      cufftResult result = cufftExecC2R(plan_remainder_, 
        reinterpret_cast<cufftComplex*>(const_cast<std::complex<T>*>(in + num_batches_ * batch_size_main_ * (cols_ / 2 + 1))),
        reinterpret_cast<cufftReal*>(out + num_batches_ * batch_size_main_ * cols_));
      ASSERT_CUFFT_OK(result);
    }

    if(scale_ != scalar_type(1.))
      smmul(scale_, out, out, rows_, cols_);
  }

  /// complex->complex
  void forward(T const* in, T* out)  VSIP_NOTHROW
  {
    for (size_t i = 0; i < num_batches_; ++i)
    {
      cufftResult result = cufftExecC2C(plan_main_, 
        reinterpret_cast<cufftComplex*>(const_cast<T*>(in + i * batch_size_main_ * cols_)),
        reinterpret_cast<cufftComplex*>(out + i * batch_size_main_ * cols_), CUFFT_FORWARD);
      ASSERT_CUFFT_OK(result);
    }

    if (batch_size_remainder_)
    {
      cufftResult result = cufftExecC2C(plan_remainder_, 
        reinterpret_cast<cufftComplex*>(const_cast<T*>(in + num_batches_ * batch_size_main_ * cols_)),
        reinterpret_cast<cufftComplex*>(out + num_batches_ * batch_size_main_ * cols_), CUFFT_FORWARD);
      ASSERT_CUFFT_OK(result);
    }

    if(scale_ != scalar_type(1.))
      smmul(scale_, out, out, rows_, cols_);
  }

  /// complex->complex
  void inverse(T const* in, T* out)  VSIP_NOTHROW
  {
    for (size_t i = 0; i < num_batches_; ++i)
    {
      cufftResult result = cufftExecC2C(plan_main_, 
        reinterpret_cast<cufftComplex*>(const_cast<T*>(in + i * batch_size_main_ * cols_)),
        reinterpret_cast<cufftComplex*>(out + i * batch_size_main_ * cols_), CUFFT_INVERSE);
      ASSERT_CUFFT_OK(result);
    }

    if (batch_size_remainder_)
    {
      cufftResult result = cufftExecC2C(plan_remainder_, 
        reinterpret_cast<cufftComplex*>(const_cast<T*>(in + num_batches_ * batch_size_main_ * cols_)),
        reinterpret_cast<cufftComplex*>(out + num_batches_ * batch_size_main_ * cols_), CUFFT_INVERSE);
      ASSERT_CUFFT_OK(result);
    }

    if(scale_ != scalar_type(1.))
      smmul(scale_, out, out, rows_, cols_);
  }

  cufftHandle plan_main_;
  cufftHandle plan_remainder_;
  void* dev_in_;
  void* dev_out_;
  size_t insize_;
  size_t outsize_;
  size_t rows_;
  size_t cols_;
  size_t batch_size_main_;
  size_t batch_size_remainder_;
  size_t num_batches_;
  scalar_type scale_;
};




template <dimension_type D, //< Dimension
          typename       T> //< Type
struct Driver;


template <typename  T>
struct Driver<1, T>
  : Driver_base_1d<T>
{
  // CUDA FFT type is R2C, C2R or C2C
  Driver(cufftType fft_type, Domain<1> const &dom, length_type mult = 1, 
    typename scalar_of<T>::type scale = 1.f) 
  {
    this->init(fft_type, dom.size(), mult, scale);
  }
  ~Driver() { this->fini();}
};




template <dimension_type D, //< Dimension
	  typename I,       //< Input type
	  typename O,       //< Output type
	  int S>            //< Special Dimension
class Fft_impl;

// 1D real -> complex FFT
template <typename T>
class Fft_impl<1, T, std::complex<T>, 0>
  : public fft::Fft_backend<1, T, std::complex<T>, 0>,
    private Driver<1, T>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<1> const &dom, rtype scale)
    : Driver<1, T>(CUFFT_R2C, dom, 1, scale)
  {}
  virtual char const* name() { return "fft-cuda-1D-real-forward"; }
  virtual bool supports_scale() { return true;}
  virtual bool supports_cuda_memory() { return true;}
  virtual void out_of_place(rtype *in, stride_type in_s,
			    ctype *out, stride_type out_s,
			    length_type /*l*/)
  {
    assert(in_s == 1 && out_s == 1);
    this->forward(in, out);
  }
};

// 1D complex -> real FFT
template <typename T>
class Fft_impl<1, std::complex<T>, T, 0>
  : public fft::Fft_backend<1, std::complex<T>, T, 0>,
    private Driver<1, T>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<1> const &dom, rtype scale)
    : Driver<1, T>(CUFFT_C2R, dom, 1, scale)
  {}
  virtual char const* name() { return "fft-cuda-1D-real-inverse"; }
  virtual bool supports_scale() { return true;}
  virtual bool supports_cuda_memory() { return true;}
  virtual void out_of_place(ctype *in, stride_type in_s,
			    rtype *out, stride_type out_s,
			    length_type /*l*/)
  {
    assert(in_s == 1 && out_s == 1);
    this->forward(in, out);
  }
};

// 1D complex -> complex FFT
template <typename T, int S>
class Fft_impl<1, std::complex<T>, std::complex<T>, S>
  : public fft::Fft_backend<1, std::complex<T>, std::complex<T>, S>,
    private Driver<1, std::complex<T> >
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<1> const &dom, rtype scale)
    : Driver<1, std::complex<T> >(CUFFT_C2C, dom, 1, scale)
  {}
  virtual char const* name() { return "fft-cuda-1D-complex"; }
  virtual bool supports_scale() { return true;}
  virtual bool supports_cuda_memory() { return true;}
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




template <typename I, //< Input type
	  typename O, //< Output type
	  int A,      //< Axis
          int D>      //< Exponent
class Fftm_impl;

// real -> complex FFTM
template <typename T, int A>
class Fftm_impl<T, std::complex<T>, A, fft_fwd>
  : public fft::Fftm_backend<T, std::complex<T>, A, fft_fwd>,
    private Driver<1, T>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const axis = A == vsip::col ? 0 : 1;

public:
  Fftm_impl(Domain<2> const &dom, rtype scale)
    : Driver<1, T>(CUFFT_R2C, dom[axis], dom[1 - axis].size(), scale),
      mult_(dom[1 - axis].size())
  {}

  virtual char const* name() { return "fftm-cuda-real-forward"; }
  virtual bool supports_scale() { return true;}
  virtual bool supports_cuda_memory() { return true;}
  virtual void out_of_place(rtype *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    ctype *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    if (axis == 0)
    {
      assert(cols <= mult_);
      assert(in_r_stride == 1 && out_r_stride == 1);
      assert((in_c_stride > 0) && (static_cast<length_type>(in_c_stride) >= rows));
      assert((out_c_stride > 0) && (static_cast<length_type>(out_c_stride) >= rows/2 + 1));
    }
    else
    {   
      assert(rows <= mult_);
      assert(in_c_stride == 1 && out_c_stride == 1);
      assert((in_r_stride > 0) && (static_cast<length_type>(in_r_stride) >= cols));
      assert((out_r_stride > 0) && (static_cast<length_type>(out_r_stride) >= cols/2 + 1));
    }
    this->forward(in, out);
  }

  length_type mult_;
};

// complex -> real FFTM
template <typename T, int A>
class Fftm_impl<std::complex<T>, T, A, fft_inv>
  : public fft::Fftm_backend<std::complex<T>, T, A, fft_inv>,
    private Driver<1, T>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const axis = A == vsip::col ? 0 : 1;

public:
  Fftm_impl(Domain<2> const &dom, rtype scale)
    : Driver<1, T>(CUFFT_C2R, dom[axis], dom[1 - axis].size(), scale),
      mult_(dom[1 - axis].size())
  {}

  virtual char const* name() { return "fftm-cuda-real-inverse"; }
  virtual bool supports_scale() { return true;}
  virtual bool supports_cuda_memory() { return true;}
  virtual void out_of_place(ctype *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    rtype *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    if (axis == 0)
    {
      assert(cols <= mult_);
      assert(in_r_stride == 1 && out_r_stride == 1);
      assert((in_c_stride > 0) && (static_cast<length_type>(in_c_stride) >= rows/2 + 1));
      assert((out_c_stride > 0) && (static_cast<length_type>(out_c_stride) >= rows));
    }
    else
    {   
      assert(rows <= mult_);
      assert(in_c_stride == 1 && out_c_stride == 1);
      assert((in_r_stride > 0) && (static_cast<length_type>(in_r_stride) >= cols/2 + 1));
      assert((out_r_stride > 0) && (static_cast<length_type>(out_r_stride) >= cols));
    }

    this->forward(in, out);
  }

  length_type mult_;
};


// complex -> complex FFTM
template <typename T, int A, int D>
class Fftm_impl<std::complex<T>, std::complex<T>, A, D>
  : public fft::Fftm_backend<std::complex<T>, std::complex<T>, A, D>,
    private Driver<1, std::complex<T> >
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const axis = A == vsip::col ? 0 : 1;

public:
  Fftm_impl(Domain<2> const &dom, rtype scale)
    : Driver<1, std::complex<T> >(CUFFT_C2C, dom[axis], dom[1 - axis].size(), scale),
      mult_(dom[1 - axis].size())
  {}

  virtual char const* name() { return "fftm-cuda-complex"; }
  virtual bool supports_scale() { return true;}
  virtual bool supports_cuda_memory() { return true;}
  virtual void in_place(ctype *inout,
			stride_type r_stride, stride_type c_stride,
			length_type rows, length_type cols)
  {
    if (axis == 0)
    {
      assert(cols <= mult_);
      assert(r_stride == 1);
      assert((c_stride > 0) && (static_cast<length_type>(c_stride) >= rows));
    }
    else
    {   
      assert(rows <= mult_);
      assert(c_stride == 1);
      assert((r_stride > 0) && (static_cast<length_type>(r_stride) >= cols));
    }

    if (D == fft_fwd) this->forward(inout, inout);
    else this->inverse(inout, inout);
  }

  virtual void out_of_place(ctype *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    ctype *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    if (axis == 0)
    {
      assert(cols <= mult_);
      assert(in_r_stride == 1 && out_r_stride == 1);
      assert((in_c_stride > 0) && (static_cast<length_type>(in_c_stride) >= rows));
      assert((out_c_stride > 0) && (static_cast<length_type>(out_c_stride) >= rows));
    }
    else
    {   
      assert(rows <= mult_);
      assert(in_c_stride == 1 && out_c_stride == 1);
      assert((in_r_stride > 0) && (static_cast<length_type>(in_r_stride) >= cols));
      assert((out_r_stride > 0) && (static_cast<length_type>(out_r_stride) >= cols));
    }

    if (D == fft_fwd) this->forward(in, out);
    else this->inverse(in, out);
  }

  length_type mult_;
};



// FFT create() functions

#define FFT_DEF(D, I, O, S)				\
template <>                                             \
std::auto_ptr<fft::Fft_backend<D, I, O, S> >		\
create(Domain<D> const &dom, float scale)		\
{                                                       \
  return std::auto_ptr<fft::Fft_backend<D, I, O, S> >	\
    (new Fft_impl<D, I, O, S>(dom, scale));             \
}

// Note: Double precision is not supported by CUDA 2.1
FFT_DEF(1, float, std::complex<float>, 0)
FFT_DEF(1, std::complex<float>, float, 0)
FFT_DEF(1, std::complex<float>, std::complex<float>, fft_fwd)
FFT_DEF(1, std::complex<float>, std::complex<float>, fft_inv)

#undef FFT_DEF

// FFTM create() functions

#define FFTM_DEF(I, O, A, D)				\
template <>                                             \
std::auto_ptr<fft::Fftm_backend<I, O, A, D> >		\
create(Domain<2> const &dom, float scale)		\
{                                                       \
  return std::auto_ptr<fft::Fftm_backend<I, O, A, D> >	\
    (new Fftm_impl<I, O, A, D>(dom, scale));		\
}

FFTM_DEF(float, std::complex<float>, 0, fft_fwd)
FFTM_DEF(float, std::complex<float>, 1, fft_fwd)
FFTM_DEF(std::complex<float>, float, 0, fft_inv)
FFTM_DEF(std::complex<float>, float, 1, fft_inv)
FFTM_DEF(std::complex<float>, std::complex<float>, 0, fft_fwd)
FFTM_DEF(std::complex<float>, std::complex<float>, 1, fft_fwd)
FFTM_DEF(std::complex<float>, std::complex<float>, 0, fft_inv)
FFTM_DEF(std::complex<float>, std::complex<float>, 1, fft_inv)

#undef FFTM_DEF


} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip
