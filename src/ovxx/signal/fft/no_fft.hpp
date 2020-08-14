//
// Copyright (c) 2006 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_signal_fft_no_fft_hpp_
#define ovxx_signal_fft_no_fft_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <ovxx/dispatch.hpp>
#include <ovxx/signal/fft/util.hpp>

namespace ovxx
{
namespace signal
{
namespace fft
{
/// Dummy backend, for testing purposes.
/// As the fft_backend base class throws "unimplemented" in all its
/// out_of_place and in_place functions, we merely override those here
/// as no-ops, since this makes testing easier.
template <dimension_type D, typename I, typename O, int S> class no_fft;

// 1D complex -> complex FFT
template <typename T, int S>
class no_fft<1, complex<T>, complex<T>, S>
  : public fft_backend<1, complex<T>, complex<T>, S>
{
public:
  void in_place(complex<T> *, stride_type, length_type) {}
  void in_place(std::pair<T *, T *>, stride_type, length_type) {}
  void out_of_place(complex<T> *, stride_type,
  		    complex<T> *, stride_type,
  		    length_type)
  {}
  void out_of_place(std::pair<T *, T *>, stride_type,
		    std::pair<T *, T *>, stride_type,
		    length_type)
  {}
};

// 1D real -> complex FFT
template <typename T>
class no_fft<1, T, complex<T>, 0> : public fft_backend<1, T, complex<T>, 0>
{
public:
  void out_of_place(T *, stride_type,
		    complex<T> *, stride_type,
		    length_type)
  {}
  void out_of_place(T *, stride_type,
		    std::pair<T *, T *>, stride_type,
		    length_type)
  {}
};

// 1D complex -> real FFT
template <typename T>
class no_fft<1, complex<T>, T, 0>
  : public fft_backend<1, complex<T>, T, 0>
{
public:
  void out_of_place(complex<T> *, stride_type,
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
class no_fft<2, complex<T>, complex<T>, S>
  : public fft_backend<2, complex<T>, complex<T>, S>
{
public:
  virtual void in_place(complex<T> *,
			stride_type, stride_type,
			length_type, length_type)
  {}
  virtual void in_place(std::pair<T *, T *>,
			stride_type, stride_type,
			length_type, length_type)
  {}
  virtual void out_of_place(complex<T> *,
			    stride_type, stride_type,
			    complex<T> *,
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
class no_fft<2, T, complex<T>, S>
  : public fft_backend<2, T, complex<T>, S>
{
public:
  void out_of_place(T *,
  		    stride_type, stride_type,
  		    complex<T> *,
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
class no_fft<2, complex<T>, T, S>
  : public fft_backend<2, complex<T>, T, S>
{
public:
  void out_of_place(complex<T> *,
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
class no_fft<3, complex<T>, complex<T>, S>
  : public fft_backend<3, complex<T>, complex<T>, S>
{
public:
  void in_place(complex<T> *,
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
  void out_of_place(complex<T> *,
  		    stride_type,
  		    stride_type,
  		    stride_type,
  		    complex<T> *,
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
class no_fft<3, T, complex<T>, S>
  : public fft_backend<3, T, complex<T>, S>
{
public:
  void out_of_place(T *,
  		    stride_type,
  		    stride_type,
  		    stride_type,
  		    complex<T> *,
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
class no_fft<3, complex<T>, T, S>
  : public fft_backend<3, complex<T>, T, S>
{
public:
  void out_of_place(complex<T> *,
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

template <typename I, typename O, int A, int D> class no_fftm;

// real -> complex FFTM
template <typename T, int A>
class no_fftm<T, complex<T>, A, fft_fwd>
  : public fftm_backend<T, complex<T>, A, fft_fwd>
{
public:
  void out_of_place(T *,
  		    stride_type, stride_type,
  		    complex<T> *,
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
class no_fftm<complex<T>, T, A, fft_inv>
  : public fftm_backend<complex<T>, T, A, fft_inv>
{
public:
  void out_of_place(complex<T> *,
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
class no_fftm<complex<T>, complex<T>, A, D>
  : public fftm_backend<complex<T>, complex<T>, A, D>
{
public:
  void in_place(complex<T> *,
  		stride_type, stride_type,
  		length_type, length_type)
  {}
  void in_place(std::pair<T *, T *>,
  		stride_type, stride_type,
  		length_type, length_type)
  {}
  void out_of_place(complex<T> *,
  		    stride_type, stride_type,
  		    complex<T> *,
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

} // namespace ovxx::signal::fft
} // namespace ovxx::signal

namespace dispatcher
{
template <dimension_type D,
	  typename I,
	  typename O,
	  int S,
	  vsip::return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fft<D, I, O, S, R, N>, be::no_fft,
  std::unique_ptr<signal::fft::fft_backend<D, I, O, S> >
  (Domain<D> const &, typename scalar_of<I>::type)>
{
  typedef typename scalar_of<I>::type scalar_type;
  static bool const ct_valid = true;
  static bool rt_valid(Domain<D> const &, scalar_type) { return true;}
  static std::unique_ptr<signal::fft::fft_backend<D, I, O, S> >
  exec(Domain<D> const &, scalar_type)
  {
    return std::unique_ptr<signal::fft::fft_backend<D, I, O, S> >
      (new signal::fft::no_fft<D, I, O, S>());
  }
};

template <typename I,
	  typename O,
	  int A,
	  int D,
	  return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fftm<I, O, A, D, R, N>, be::no_fft,
  std::unique_ptr<signal::fft::fftm_backend<I, O, A, D> > 
  (Domain<2> const &, typename scalar_of<I>::type)>
{
  typedef typename scalar_of<I>::type scalar_type;
  static bool const ct_valid = true;
  static bool rt_valid(Domain<2> const &, scalar_type) { return true;}
  static std::unique_ptr<signal::fft::fftm_backend<I, O, A, D> > 
  exec(Domain<2> const &, scalar_type)
  {
    return std::unique_ptr<signal::fft::fftm_backend<I, O, A, D> > 
      (new signal::fft::no_fftm<I, O, A, D>());
  }
};

} // namespace ovxx::dispatcher
} // namespace ovxx

#endif

