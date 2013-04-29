//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_FFT_BACKEND_HPP
#define VSIP_CORE_FFT_BACKEND_HPP

#include <vsip/support.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/core/metaprogramming.hpp>

namespace vsip
{

/// Perform a forward FFT.
int const fft_fwd = -2;
/// Perform an inverse FFT.
int const fft_inv = -1;

namespace impl
{
namespace fft
{

bool const forward = true;
bool const inverse = false;
  
template <typename B>
struct is_fft_backend { static bool const value = false;};
template <typename B>
struct is_fftm_backend { static bool const value = false;};

template <dimension_type D, typename T, int A, int E>
struct Backend
{
  typedef T scalar_type;
  static dimension_type const dim = D;
  static int const axis = A;
  static int const exponent = E;
  
};

template <dimension_type D, typename I, typename O, int S> 
class Fft_backend;

template <dimension_type D, typename I, typename O, int S> 
struct is_fft_backend<Fft_backend<D, I, O, S> >
{ static bool const value = true;};

/// 1D real forward FFT
template <typename T>
class Fft_backend<1, T, complex<T>, 0>
  : public Backend<1, T, 0, -1>
{
public:
  typedef T input_value_type;
  typedef complex<T> output_value_type;

  virtual ~Fft_backend() {}
  virtual char const* name() { return "fft-backend-1D-real-forward"; }
  virtual bool supports_scale() { return false;}
  virtual bool supports_cuda_memory() { return false;}
  virtual void query_layout(Rt_layout<1> &rtl_in, Rt_layout<1> &rtl_out)
  {
    // By default use unit_stride, tuple<0, 1, 2>, interleaved_complex
    rtl_in.packing = rtl_out.packing = dense;
    rtl_in.order = rtl_out.order = tuple<0, 1, 2>();
    rtl_out.storage_format = interleaved_complex;
  }
  virtual bool requires_copy(Rt_layout<1> &) { return false;}
  /// real -> complex (interleaved)
  virtual void out_of_place(T *, stride_type,
			    complex<T> *, stride_type,
			    length_type)
  {
    VSIP_IMPL_THROW(unimplemented("Fft_backend<1, real, fft_fwd>::out_of_place "
				  "unimplemented"));
  }
  /// real -> complex (split)
  virtual void out_of_place(T *, stride_type,
			    std::pair<T *, T *>, stride_type,
			    length_type)
  {
    VSIP_IMPL_THROW(unimplemented("Fft_backend<1, real, fft_fwd>::out_of_place "
				  "unimplemented"));
  }
};

/// 1D real inverse FFT
template <typename T>
class Fft_backend<1, complex<T>, T, 0>
  : public Backend<1, T, 0, 1>
{
public:
  typedef complex<T> input_value_type;
  typedef T output_value_type;

  virtual ~Fft_backend() {}
  virtual char const* name() { return "fft-backend-1D-real-inverse"; }
  virtual bool supports_scale() { return false;}
  virtual bool supports_cuda_memory() { return false;}
  virtual void query_layout(Rt_layout<1> &rtl_in, Rt_layout<1> &rtl_out)
  {
    // By default use unit_stride, tuple<0, 1, 2>, interleaved_complex
    rtl_in.packing = rtl_out.packing = dense;
    rtl_in.order = rtl_out.order = tuple<0, 1, 2>();
    rtl_in.storage_format = interleaved_complex;
  }
  virtual bool requires_copy(Rt_layout<1> &) { return false;}
  /// complex (interleaved) -> real
  virtual void out_of_place(complex<T> *, stride_type,
			    T *, stride_type,
			    length_type)
  {
    VSIP_IMPL_THROW(unimplemented("Fft_backend<1, real, fft_inv>::out_of_place "
				  "unimplemented"));
  }
  /// real -> complex (split)
  virtual void out_of_place(std::pair<T *, T *>, stride_type,
			    T *, stride_type,
			    length_type)
  {
    VSIP_IMPL_THROW(unimplemented("Fft_backend<1, real, fft_inv>::out_of_place "
				  "unimplemented"));
  }
};

/// 1D complex FFT
template <typename T, int S>
class Fft_backend<1, complex<T>, complex<T>, S>
  : public Backend<1, T, 0, S == fft_fwd ? -1 : 1>
{
public:
  typedef complex<T> input_value_type;
  typedef complex<T> output_value_type;

  virtual ~Fft_backend() {}
  virtual char const* name() { return "fft-backend-1D-complex"; }
  virtual bool supports_scale() { return false;}
  virtual bool supports_cuda_memory() { return false;}
  virtual void query_layout(Rt_layout<1> &rtl_inout)
  {
    // By default use unit_stride, tuple<0, 1, 2>, interleaved_complex
    rtl_inout.packing = dense;
    rtl_inout.order = tuple<0, 1, 2>();
    rtl_inout.storage_format = interleaved_complex;
  }
  virtual void query_layout(Rt_layout<1> &rtl_in, Rt_layout<1> &rtl_out)
  {
    // By default use unit_stride, tuple<0, 1, 2>, interleaved_complex
    rtl_in.packing = rtl_out.packing = dense;
    rtl_in.order = rtl_out.order = tuple<0, 1, 2>();
    rtl_in.storage_format = rtl_out.storage_format = interleaved_complex;
  }
  virtual bool requires_copy(Rt_layout<1> &) { return false;}
  /// complex (interleaved) in-place
  virtual void in_place(complex<T> *, stride_type, length_type)
  {
    if (S == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fft_backend<1, complex, fft_fwd>::in_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fft_backend<1, complex, fft_inv>::in_place "
				    "unimplemented"));
  }
  /// complex (split) in-place
  virtual void in_place(std::pair<T *, T *>, stride_type, length_type)
  {
    if (S == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fft_backend<1, complex, fft_fwd>::in_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fft_backend<1, complex, fft_inv>::in_place "
				    "unimplemented"));
  }
  /// complex (interleaved) by-reference
  virtual void out_of_place(complex<T> *, stride_type,
			    complex<T> *, stride_type,
			    length_type)
  {
    if (S == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fft_backend<1, complex, fft_fwd>::out_of_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fft_backend<1, complex, fft_inv>::out_of_place "
				    "unimplemented"));
  }
  /// complex (split) by-reference
  virtual void out_of_place(std::pair<T *, T *>, stride_type,
			    std::pair<T *, T *>, stride_type,
			    length_type)
  {
    if (S == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fft_backend<1, complex, fft_fwd>::out_of_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fft_backend<1, complex, fft_inv>::out_of_place "
				    "unimplemented"));
  }
};

/// 2D real forward FFT
template <typename T, int S>
class Fft_backend<2, T, complex<T>, S>
  : public Backend<2, T, S, -1>
{
public:
  typedef T input_value_type;
  typedef complex<T> output_value_type;

  virtual ~Fft_backend() {}
  virtual char const* name() { return "fft-backend-2D-real-forward"; }
  virtual bool supports_scale() { return false;}
  virtual bool supports_cuda_memory() { return false;}
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    // By default use unit_stride, tuple<0, 1, 2>, interleaved_complex
    rtl_in.packing = rtl_out.packing = dense;
    rtl_in.order = rtl_out.order = tuple<0, 1, 2>();
    rtl_out.storage_format = interleaved_complex;
  }
  virtual bool requires_copy(Rt_layout<2> &) { return false;}
  /// real -> complex (interleaved) by-reference
  virtual void out_of_place(T *,
			    stride_type, stride_type,
			    std::complex<T> *,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    VSIP_IMPL_THROW(unimplemented("Fft_backend<2, real, fft_fwd>::out_of_place "
				  "unimplemented"));
  }
  /// real -> complex (split) by-reference
  virtual void out_of_place(T *,
			    stride_type, stride_type,
			    std::pair<T *, T *>,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    VSIP_IMPL_THROW(unimplemented("Fft_backend<2, real, fft_fwd>::out_of_place "
				  "unimplemented"));
  }
};

/// 2D real inverse FFT
template <typename T, int S>
class Fft_backend<2, complex<T>, T, S>
  : public Backend<2, T, S, 1>
{
public:
  typedef complex<T> input_value_type;
  typedef T output_value_type;

  virtual ~Fft_backend() {}
  virtual char const* name() { return "fft-backend-2D-real-inverse"; }
  virtual bool supports_scale() { return false;}
  virtual bool supports_cuda_memory() { return false;}
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    // By default use unit_stride, tuple<0, 1, 2>, interleaved_complex
    rtl_in.packing = rtl_out.packing = dense;
    rtl_in.order = rtl_out.order = tuple<0, 1, 2>();
    rtl_in.storage_format = interleaved_complex;
  }
  virtual bool requires_copy(Rt_layout<2> &) { return false;}
  /// complex (interleaved) -> real by-reference
  virtual void out_of_place(std::complex<T> *,
			    stride_type, stride_type,
			    T *,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    VSIP_IMPL_THROW(unimplemented("Fft_backend<1, real, fft_inv>::out_of_place "
				  "unimplemented"));
  }
  /// complex (split) -> real by-reference
  virtual void out_of_place(std::pair<T *, T *>,
			    stride_type, stride_type,
			    T *,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    VSIP_IMPL_THROW(unimplemented("Fft_backend<1, real, fft_inv>::out_of_place "
				  "unimplemented"));
  }
};

/// 2D complex FFT
template <typename T, int S>
class Fft_backend<2, std::complex<T>, std::complex<T>, S>
  : public Backend<2, T, S, S == fft_fwd ? -1 : 1>
{
public:
  typedef std::complex<T> input_value_type;
  typedef std::complex<T> output_value_type;

  virtual ~Fft_backend() {}
  virtual char const* name() { return "fft-backend-2D-complex"; }
  virtual bool supports_scale() { return false;}
  virtual bool supports_cuda_memory() { return false;}
  virtual void query_layout(Rt_layout<2> &rtl_inout)
  {
    // By default use unit_stride, tuple<0, 1, 2>, interleaved_complex
    rtl_inout.packing = dense;
    rtl_inout.order = tuple<0, 1, 2>();
    rtl_inout.storage_format = interleaved_complex;
  }
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    // By default use unit_stride, tuple<0, 1, 2>, interleaved_complex
    rtl_in.packing = rtl_out.packing = dense;
    rtl_in.order = rtl_out.order = tuple<0, 1, 2>();
    rtl_in.storage_format = rtl_out.storage_format = interleaved_complex;
  }
  virtual bool requires_copy(Rt_layout<2> &) { return false;}
  /// complex (interleaved) in-place
  virtual void in_place(std::complex<T> *,
			stride_type, stride_type,
			length_type, length_type)
  {
    if (S == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fft_backend<2, complex, fft_fwd>::in_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fft_backend<2, complex, fft_inv>::in_place "
				    "unimplemented"));
  }
  /// complex (split) in-place
  virtual void in_place(std::pair<T *, T *>,
			stride_type, stride_type,
			length_type, length_type)
  {
    if (S == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fft_backend<2, complex, fft_fwd>::in_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fft_backend<2, complex, fft_inv>::in_place "
				    "unimplemented"));
  }
  /// complex (interleaved) by-reference
  virtual void out_of_place(std::complex<T> *,
			    stride_type, stride_type,
			    std::complex<T> *,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    if (S == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fft_backend<2, complex, fft_fwd>::out_of_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fft_backend<2, complex, fft_inv>::out_of_place "
				    "unimplemented"));
  }
  /// complex (split) by-reference
  virtual void out_of_place(std::pair<T *, T *>,
			    stride_type, stride_type,
			    std::pair<T *, T *>,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    if (S == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fft_backend<2, complex, fft_fwd>::out_of_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fft_backend<2, complex, fft_inv>::out_of_place "
				    "unimplemented"));
  }
};

/// 3D real forward FFT
template <typename T, int S>
class Fft_backend<3, T, std::complex<T>, S>
  : public Backend<3, T, S, -1>
{
public:
  typedef T input_value_type;
  typedef std::complex<T> output_value_type;

  virtual ~Fft_backend() {}
  virtual char const* name() { return "fft-backend-2D-real-forward"; }
  virtual bool supports_scale() { return false;}
  virtual bool supports_cuda_memory() { return false;}
  virtual void query_layout(Rt_layout<3> &rtl_in, Rt_layout<3> &rtl_out)
  {
    // By default use unit_stride, tuple<0, 1, 2>, interleaved_complex
    rtl_in.packing = rtl_out.packing = dense;
    rtl_in.order = rtl_out.order = tuple<0, 1, 2>();
    rtl_out.storage_format = interleaved_complex;
  }
  virtual bool requires_copy(Rt_layout<3> &) { return false;}
  /// real -> complex (interleaved) by-reference
  virtual void out_of_place(T *,
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
  {
    VSIP_IMPL_THROW(unimplemented("Fft_backend<3, real, fft_fwd>::out_of_place "
				  "unimplemented"));
  }
  /// real -> complex (split) by-reference
  virtual void out_of_place(T *,
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
  {
    VSIP_IMPL_THROW(unimplemented("Fft_backend<3, real, fft_fwd>::out_of_place "
				  "unimplemented"));
  }
};

/// 3D real inverse FFT
template <typename T, int S>
class Fft_backend<3, std::complex<T>, T, S>
  : public Backend<3, T, S, 1>
{
public:
  typedef std::complex<T> input_value_type;
  typedef T output_value_type;

  virtual ~Fft_backend() {}
  virtual char const* name() { return "fft-backend-2D-real-inverse"; }
  virtual bool supports_scale() { return false;}
  virtual bool supports_cuda_memory() { return false;}
  virtual void query_layout(Rt_layout<3> &rtl_in, Rt_layout<3> &rtl_out)
  {
    // By default use unit_stride, tuple<0, 1, 2>, interleaved_complex
    rtl_in.packing = rtl_out.packing = dense;
    rtl_in.order = rtl_out.order = tuple<0, 1, 2>();
    rtl_in.storage_format = interleaved_complex;
  }
  virtual bool requires_copy(Rt_layout<3> &) { return false;}
  /// complex (interleaved) -> real by-reference
  virtual void out_of_place(std::complex<T> *,
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
  {
    VSIP_IMPL_THROW(unimplemented("Fft_backend<3, real, fft_inv>::out_of_place "
				  "unimplemented"));
  }
  /// complex (split) -> real by-reference
  virtual void out_of_place(std::pair<T *, T *>,
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
  {
    VSIP_IMPL_THROW(unimplemented("Fft_backend<3, real, fft_inv>::out_of_place "
				  "unimplemented"));
  }
};

/// 3D complex FFT
template <typename T, int S>
class Fft_backend<3, std::complex<T>, std::complex<T>, S>
  : public Backend<3, T, 0, S == fft_fwd ? -1 : 1>
{
public:
  typedef std::complex<T> input_value_type;
  typedef std::complex<T> output_value_type;

  virtual ~Fft_backend() {}
  virtual char const* name() { return "fft-backend-2D-complex"; }
  virtual bool supports_scale() { return false;}
  virtual bool supports_cuda_memory() { return false;}
  virtual void query_layout(Rt_layout<3> &rtl_inout)
  {
    // By default use unit_stride, tuple<0, 1, 2>, interleaved_complex
    rtl_inout.packing = dense;
    rtl_inout.order = tuple<0, 1, 2>();
    rtl_inout.storage_format = interleaved_complex;
  }
  virtual void query_layout(Rt_layout<3> &rtl_in, Rt_layout<3> &rtl_out)
  {
    // By default use unit_stride, tuple<0, 1, 2>, interleaved_complex
    rtl_in.packing = rtl_out.packing = dense;
    rtl_in.order = rtl_out.order = tuple<0, 1, 2>();
    rtl_in.storage_format = rtl_out.storage_format = interleaved_complex;
  }
  virtual bool requires_copy(Rt_layout<3> &) { return false;}
  /// complex (interleaved) in-place
  virtual void in_place(std::complex<T> *,
			stride_type,
			stride_type,
			stride_type,
			length_type,
			length_type,
			length_type)
  {
    if (S == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fft_backend<3, complex, fft_fwd>::in_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fft_backend<3, complex, fft_inv>::in_place "
				    "unimplemented"));
  }
  /// complex (split) in-place
  virtual void in_place(std::pair<T *, T *>,
			stride_type,
			stride_type,
			stride_type,
			length_type,
			length_type,
			length_type)
  {
    if (S == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fft_backend<3, complex, fft_fwd>::in_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fft_backend<3, complex, fft_inv>::in_place "
				    "unimplemented"));
  }
  /// complex (interleaved) by-reference
  virtual void out_of_place(std::complex<T> *,
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
  {
    if (S == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fft_backend<3, complex, fft_fwd>::out_of_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fft_backend<3, complex, fft_inv>::out_of_place "
				    "unimplemented"));
  }
  /// complex (split) by-reference
  virtual void out_of_place(std::pair<T *, T *>,
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
  {
    if (S == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fft_backend<3, complex, fft_fwd>::out_of_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fft_backend<3, complex, fft_inv>::out_of_place "
				    "unimplemented"));
  }
};

/// FFTM
template <typename I, typename O, int A, int D>
class Fftm_backend;

template <typename I, typename O, int A, int D>
struct is_fftm_backend<Fftm_backend<I, O, A, D> >
{ static bool const value = true;};

/// real forward FFTM
template <typename T, int A>
class Fftm_backend<T, std::complex<T>, A, fft_fwd>
  : public Backend<2, T, 1 - A, -1>
{
public:
  typedef T input_value_type;
  typedef std::complex<T> output_value_type;

  virtual ~Fftm_backend() {}
  virtual char const* name() { return "fftm-backend-real-forward"; }
  virtual bool supports_scale() { return false;}
  virtual bool supports_cuda_memory() { return false;}
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    // By default use unit_stride,
    rtl_in.packing = rtl_out.packing = dense;
    // an order that gives unit strides on the axis perpendicular to A,
    if (A == vsip::col) rtl_in.order = tuple<1, 0, 2>();
    else rtl_in.order = tuple<0, 1, 2>();
    rtl_out.order = rtl_in.order;
    // and interleaved complex.
    rtl_out.storage_format = interleaved_complex;
  }
  virtual bool requires_copy(Rt_layout<2> &) { return false;}
  /// real -> complex (interleaved) by-reference
  virtual void out_of_place(T *,
			    stride_type, stride_type,
			    std::complex<T> *,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    VSIP_IMPL_THROW(unimplemented("Fftm_backend<real, fft_fwd>::out_of_place "
				  "unimplemented"));
  }
  /// real -> complex (split) by-reference
  virtual void out_of_place(T *,
			    stride_type, stride_type,
			    std::pair<T *, T *>,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    VSIP_IMPL_THROW(unimplemented("Fftm_backend<real, fft_fwd>::out_of_place "
				  "unimplemented"));
  }
};

/// real inverse FFTM
template <typename T, int A>
class Fftm_backend<std::complex<T>, T, A, fft_inv>
  : public Backend<2, T, 1 - A, 1>
{
public:
  typedef std::complex<T> input_value_type;
  typedef T output_value_type;

  virtual ~Fftm_backend() {}
  virtual char const* name() { return "fftm-backend-real-inverse"; }
  virtual bool supports_scale() { return false;}
  virtual bool supports_cuda_memory() { return false;}
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    // By default use unit_stride,
    rtl_in.packing = rtl_out.packing = dense;
    // an order that gives unit strides on the axis perpendicular to A,
    if (A == vsip::col) rtl_in.order = tuple<1, 0, 2>();
    else rtl_in.order = tuple<0, 1, 2>();
    rtl_out.order = rtl_in.order;
    // and interleaved complex.
    rtl_in.storage_format = interleaved_complex;
  }
  virtual bool requires_copy(Rt_layout<2> &) { return false;}
  /// complex (interleaved) -> real by-reference
  virtual void out_of_place(std::complex<T> *,
			    stride_type, stride_type,
			    T *,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    VSIP_IMPL_THROW(unimplemented("Fftm_backend<real, fft_inv>::out_of_place "
				  "unimplemented"));
  }
  /// complex (split) -> real by-reference
  virtual void out_of_place(std::pair<T *, T *>,
			    stride_type, stride_type,
			    T *,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    VSIP_IMPL_THROW(unimplemented("Fftm_backend<real, fft_inv>::out_of_place "
				  "unimplemented"));
  }
};

/// complex FFTM
template <typename T, int A, int D>
class Fftm_backend<std::complex<T>, std::complex<T>, A, D>
  : public Backend<2, T, 1 - A, D == fft_fwd ? -1 : 1>
{
public:
  typedef std::complex<T> input_value_type;
  typedef std::complex<T> output_value_type;

  virtual ~Fftm_backend() {}
  virtual char const* name() { return "fftm-backend-complex"; }
  virtual bool supports_scale() { return false;}
  virtual bool supports_cuda_memory() { return false;}
  virtual void query_layout(Rt_layout<2> &rtl_inout)
  {
    // By default use unit_stride,
    rtl_inout.packing = dense;
    // an order that gives unit strides on the axis perpendicular to A,
    if (A == vsip::col) rtl_inout.order = tuple<1, 0, 2>();
    else rtl_inout.order = tuple<0, 1, 2>();
    // and interleaved complex.
    rtl_inout.storage_format = interleaved_complex;
  }
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    // By default use unit_stride,
    rtl_in.packing = rtl_out.packing = dense;
    // an order that gives unit strides on the axis perpendicular to A,
    if (A == vsip::col) rtl_in.order = tuple<1, 0, 2>();
    else rtl_in.order = tuple<0, 1, 2>();
    rtl_out.order = rtl_in.order;
    // and interleaved complex.
    rtl_in.storage_format = rtl_out.storage_format = interleaved_complex;
  }
  virtual bool requires_copy(Rt_layout<2> &) { return false;}
  /// complex (interleaved) in-place
  virtual void in_place(std::complex<T> *,
			stride_type, stride_type,
			length_type, length_type)
  {
    if (D == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fftm_backend<complex, fft_fwd>::in_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fftm_backend<complex, fft_inv>::in_place "
				    "unimplemented"));
  }
  /// complex (split) in-place
  virtual void in_place(std::pair<T *, T *>,
			stride_type, stride_type,
			length_type, length_type)
  {
    if (D == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fftm_backend<complex, fft_fwd>::in_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fftm_backend<complex, fft_inv>::in_place "
				    "unimplemented"));
  }
  /// complex (interleaved) by-reference
  virtual void out_of_place(std::complex<T> *,
			    stride_type, stride_type,
			    std::complex<T> *,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    if (D == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fftm_backend<complex, fft_fwd>::out_of_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fftm_backend<complex, fft_inv>::out_of_place "
				    "unimplemented"));
  }
  /// complex (split) by-reference
  virtual void out_of_place(std::pair<T *, T *>,
			    stride_type, stride_type,
			    std::pair<T *, T *>,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    if (D == fft_fwd)
      VSIP_IMPL_THROW(unimplemented("Fftm_backend<complex, fft_fwd>::out_of_place "
				    "unimplemented"));
    else
      VSIP_IMPL_THROW(unimplemented("Fftm_backend<complex, fft_inv>::out_of_place "
				    "unimplemented"));
  }
};

} // namespace vsip::impl::fft
} // namespace vsip::impl
} // namespace vsip

#endif
