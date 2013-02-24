/* Copyright (c) 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/fft/no_fft.hpp
    @author  Stefan Seefeld
    @date    2006-05-01
    @brief   VSIPL++ Library: FFT backend.
*/

#ifndef VSIP_CORE_FFT_NO_FFT_HPP
#define VSIP_CORE_FFT_NO_FFT_HPP

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/opt/dispatch.hpp>
#include <vsip/core/fft/util.hpp>

namespace vsip
{
namespace impl
{
namespace fft
{
/// Dummy backend, for testing purposes.
/// As the Fft_backend base class throws "unimplemented" in all its
/// out_of_place and in_place functions, we merely override those here
/// as no-ops, since this makes testing easier.
template <dimension_type D, typename I, typename O, int S> class No_fft;

// 1D complex -> complex FFT
template <typename T, int S>
class No_fft<1, std::complex<T>, std::complex<T>, S>
  : public fft::Fft_backend<1, std::complex<T>, std::complex<T>, S>
{
public:
  void in_place(std::complex<T> *, stride_type, length_type) {}
  void in_place(std::pair<T *, T *>, stride_type, length_type) {}
  void out_of_place(std::complex<T> *, stride_type,
  		    std::complex<T> *, stride_type,
  		    length_type)
  {}
  void out_of_place(std::pair<T *, T *>, stride_type,
		    std::pair<T *, T *>, stride_type,
		    length_type)
  {}
};

// 1D real -> complex FFT
template <typename T>
class No_fft<1, T, std::complex<T>, 0>
  : public fft::Fft_backend<1, T, std::complex<T>, 0>
{
public:
  void out_of_place(T *, stride_type,
  		    std::complex<T> *, stride_type,
  		    length_type)
  {}
  void out_of_place(T *, stride_type,
		    std::pair<T *, T *>, stride_type,
		    length_type)
  {}
};

// 1D complex -> real FFT
template <typename T>
class No_fft<1, std::complex<T>, T, 0>
  : public fft::Fft_backend<1, std::complex<T>, T, 0>
{
public:
  void out_of_place(std::complex<T> *, stride_type,
  		    T *, stride_type,
  		    length_type)
  {}
  void out_of_place(std::pair<T *, T *>, stride_type,
		    T *, stride_type,
		    length_type)
  {}
};

// 2D complex -> complex FFT
template <typename T, int S>
class No_fft<2, std::complex<T>, std::complex<T>, S>
  : public fft::Fft_backend<2, std::complex<T>, std::complex<T>, S>
{
public:
  virtual void in_place(std::complex<T> *,
			stride_type, stride_type,
			length_type, length_type)
  {}
  virtual void in_place(std::pair<T *, T *>,
			stride_type, stride_type,
			length_type, length_type)
  {}
  virtual void out_of_place(std::complex<T> *,
			    stride_type, stride_type,
			    std::complex<T> *,
			    stride_type, stride_type,
			    length_type, length_type)
  {}
  virtual void out_of_place(std::pair<T *, T *>,
			    stride_type, stride_type,
			    std::pair<T *, T *>,
			    stride_type, stride_type,
			    length_type, length_type)
  {}
};

// 2D real -> complex FFT
template <typename T, int S>
class No_fft<2, T, std::complex<T>, S>
  : public fft::Fft_backend<2, T, std::complex<T>, S>
{
public:
  void out_of_place(T *,
  		    stride_type, stride_type,
  		    std::complex<T> *,
  		    stride_type, stride_type,
  		    length_type, length_type)
  {}
  void out_of_place(T *,
		    stride_type, stride_type,
		    std::pair<T *, T *>,
		    stride_type, stride_type,
		    length_type, length_type)
  {}
};

// 2D complex -> real FFT
template <typename T, int S>
class No_fft<2, std::complex<T>, T, S>
  : public fft::Fft_backend<2, std::complex<T>, T, S>
{
public:
  void out_of_place(std::complex<T> *,
  		    stride_type, stride_type,
  		    T *,
  		    stride_type, stride_type,
  		    length_type, length_type)
  {}
  void out_of_place(std::pair<T *, T *>,
		    stride_type, stride_type,
		    T *,
		    stride_type, stride_type,
		    length_type, length_type)
  {}
};

// 3D complex -> complex FFT
template <typename T, int S>
class No_fft<3, std::complex<T>, std::complex<T>, S>
  : public fft::Fft_backend<3, std::complex<T>, std::complex<T>, S>
{
public:
  void in_place(std::complex<T> *,
  		stride_type,
  		stride_type,
  		stride_type,
  		length_type,
  		length_type,
  		length_type)
  {}
  void in_place(std::pair<T *, T *>,
  		stride_type,
  		stride_type,
  		stride_type,
  		length_type,
  		length_type,
  		length_type)
  {}
  void out_of_place(std::complex<T> *,
  		    stride_type,
  		    stride_type,
  		    stride_type,
  		    std::complex<T> *,
  		    stride_type,
  		    stride_type,
  		    stride_type,
  		    length_type,
  		    length_type,
  		    length_type)
  {}
  void out_of_place(std::pair<T *, T *>,
		    stride_type,
		    stride_type,
		    stride_type,
		    std::pair<T *, T *>,
		    stride_type,
		    stride_type,
		    stride_type,
		    length_type,
		    length_type,
		    length_type)
  {}
};

// 3D real -> complex FFT
template <typename T, int S>
class No_fft<3, T, std::complex<T>, S>
  : public fft::Fft_backend<3, T, std::complex<T>, S>
{
public:
  void out_of_place(T *,
  		    stride_type,
  		    stride_type,
  		    stride_type,
  		    std::complex<T> *,
  		    stride_type,
  		    stride_type,
  		    stride_type,
  		    length_type,
  		    length_type,
  		    length_type)
  {}
  void out_of_place(T *,
		    stride_type,
		    stride_type,
		    stride_type,
		    std::pair<T *, T *>,
		    stride_type,
		    stride_type,
		    stride_type,
		    length_type,
		    length_type,
		    length_type)
  {}
};

// 3D complex -> real FFT
template <typename T, int S>
class No_fft<3, std::complex<T>, T, S>
  : public fft::Fft_backend<3, std::complex<T>, T, S>
{
public:
  void out_of_place(std::complex<T> *,
  		    stride_type,
  		    stride_type,
  		    stride_type,
  		    T *,
  		    stride_type,
  		    stride_type,
  		    stride_type,
  		    length_type,
  		    length_type,
  		    length_type)
  {}
  void out_of_place(std::pair<T *, T *>,
		    stride_type,
		    stride_type,
		    stride_type,
		    T *,
		    stride_type,
		    stride_type,
		    stride_type,
		    length_type,
		    length_type,
		    length_type)
  {}
};

template <typename I, typename O, int A, int D> class No_fftm;

// real -> complex FFTM
template <typename T, int A>
class No_fftm<T, std::complex<T>, A, fft_fwd>
  : public fft::Fftm_backend<T, std::complex<T>, A, fft_fwd>
{
public:
  void out_of_place(T *,
  		    stride_type, stride_type,
  		    std::complex<T> *,
  		    stride_type, stride_type,
  		    length_type, length_type)
  {}
  void out_of_place(T *,
		    stride_type, stride_type,
		    std::pair<T *, T *>,
		    stride_type, stride_type,
		    length_type, length_type)
  {}
};

// complex -> real FFTM
template <typename T, int A>
class No_fftm<std::complex<T>, T, A, fft_inv>
  : public fft::Fftm_backend<std::complex<T>, T, A, fft_inv>
{
public:
  void out_of_place(std::complex<T> *,
  		    stride_type, stride_type,
  		    T *,
  		    stride_type, stride_type,
  		    length_type, length_type)
  {}
  void out_of_place(std::pair<T *, T *>,
		    stride_type, stride_type,
		    T *,
		    stride_type, stride_type,
		    length_type, length_type)
  {}
};

// complex -> complex FFTM
template <typename T, int A, int D>
class No_fftm<std::complex<T>, std::complex<T>, A, D>
  : public fft::Fftm_backend<std::complex<T>, std::complex<T>, A, D>
{
public:
  void in_place(std::complex<T> *,
  		stride_type, stride_type,
  		length_type, length_type)
  {}
  void in_place(std::pair<T *, T *>,
  		stride_type, stride_type,
  		length_type, length_type)
  {}
  void out_of_place(std::complex<T> *,
  		    stride_type, stride_type,
  		    std::complex<T> *,
  		    stride_type, stride_type,
  		    length_type, length_type)
  {}
  void out_of_place(std::pair<T *, T *>,
		    stride_type, stride_type,
		    std::pair<T *, T *>,
		    stride_type, stride_type,
		    length_type, length_type)
  {}
};

} // namespace vsip::impl::fft
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <dimension_type D,
	  typename I,
	  typename O,
	  int S,
	  return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fft<D, I, O, S, R, N>, be::no_fft,
  std::auto_ptr<impl::fft::Fft_backend<D, I, O, S> >
  (Domain<D> const &, typename impl::Scalar_of<I>::type)>
{
  typedef typename impl::Scalar_of<I>::type scalar_type;
  static bool const ct_valid = true;
  static bool rt_valid(Domain<D> const &, scalar_type) { return true;}
  static std::auto_ptr<impl::fft::Fft_backend<D, I, O, S> >
  exec(Domain<D> const &, scalar_type)
  {
    return std::auto_ptr<impl::fft::Fft_backend<D, I, O, S> >
      (new impl::fft::No_fft<D, I, O, S>());
  }
};

template <typename I,
	  typename O,
	  int A,
	  int D,
	  return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fftm<I, O, A, D, R, N>, be::no_fft,
  std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> > 
  (Domain<2> const &, typename impl::Scalar_of<I>::type)>
{
  typedef typename impl::Scalar_of<I>::type scalar_type;
  static bool const ct_valid = true;
  static bool rt_valid(Domain<2> const &, scalar_type) { return true;}
  static std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> > 
  exec(Domain<2> const &, scalar_type)
  {
    return std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> > 
      (new impl::fft::No_fftm<I, O, A, D>());
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif

