//
// Copyright (c) 2006 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_signal_fft_workspace_hpp_
#define ovxx_signal_fft_workspace_hpp_

#include <ovxx/support.hpp>
#include <ovxx/layout.hpp>
#include <ovxx/signal/fft/backend.hpp>
#include <ovxx/view/traits.hpp>
#include <ovxx/adjust_layout.hpp>
#include <ovxx/equal.hpp>
#include <ovxx/dda.hpp>

namespace ovxx
{
namespace signal
{
namespace fft
{

/// This provides the temporary data as well as the
/// conversion logic from blocks to arrays as expected
/// by fft backends.
template <dimension_type D, typename I, typename O>
class workspace;

namespace detail
{
template <typename I, typename O>
struct fft_size
{
  static length_type exec(length_type /*in_size*/, length_type out_size)
  { return out_size;}
};

template <typename T>
struct fft_size<T, complex<T> >
{
  static length_type exec(length_type in_size, length_type /*out_size*/)
  { return in_size;}
};

template <typename T, int S, typename B>
void in_place(fft_backend<1, T, T, S> &backend,
	      storage_format_type storage_format,
	      dda::Rt_data<B, dda::inout> &data)
{
  if (storage_format == split_complex) 
    backend.in_place(data.ptr().template as<split_complex>(), data.stride(0), data.size(0));
  else
    backend.in_place(data.ptr().template as<array>(), data.stride(0), data.size(0));
} 

template <typename T, int S, typename B1, typename B2>
void out_of_place(fft_backend<1, T, T, S> &backend,
		  storage_format_type storage_format,
		  dda::Rt_data<B1, dda::in> &in_data,
		  storage_format_type,
		  dda::Rt_data<B2, dda::out> &out_data)
{
  if (storage_format == split_complex)
    backend.out_of_place(in_data.non_const_ptr().template as<split_complex>(), in_data.stride(0),
			 out_data.ptr().template as<split_complex>(), out_data.stride(0),
			 in_data.size(0));
  else
    backend.out_of_place(in_data.non_const_ptr().template as<array>(), in_data.stride(0),
			 out_data.ptr().template as<array>(), out_data.stride(0),
			 in_data.size(0));
} 

template <typename T, int S, typename B1, typename B2>
void out_of_place(fft_backend<1, T, complex<T>, S>  &backend,
		  storage_format_type,
		  dda::Rt_data<B1, dda::in> &in_data,
		  storage_format_type storage_format,
		  dda::Rt_data<B2, dda::out> &out_data)
{
  if (storage_format == split_complex)
    backend.out_of_place(in_data.non_const_ptr(), in_data.stride(0),
			 out_data.ptr().template as<split_complex>(), out_data.stride(0),
			 in_data.size(0));
  else
    backend.out_of_place(in_data.non_const_ptr(), in_data.stride(0),
			 out_data.ptr().template as<array>(), out_data.stride(0),
			 in_data.size(0));
} 

template <typename T, int S, typename B1, typename B2>
void out_of_place(fft_backend<1, complex<T>, T, S>  &backend,
		  storage_format_type storage_format,
		  dda::Rt_data<B1, dda::in> &in_data,
		  storage_format_type,
		  dda::Rt_data<B2, dda::out> &out_data)
{
  if (storage_format == split_complex) 
    backend.out_of_place(in_data.non_const_ptr().template as<split_complex>(), in_data.stride(0),
			 out_data.ptr(), out_data.stride(0),
			 out_data.size(0));
  else
    backend.out_of_place(in_data.non_const_ptr().template as<array>(), in_data.stride(0),
			 out_data.ptr(), out_data.stride(0),
			 out_data.size(0));
} 

template <typename T, int S, typename B>
void in_place(fft_backend<2, T, T, S>  &backend,
	      storage_format_type storage_format,
	      dda::Rt_data<B, dda::inout> &data)
{
  if (storage_format == split_complex) 
    backend.in_place(data.ptr().template as<split_complex>(),
		     data.stride(0), data.stride(1),
		     data.size(0), data.size(1));
  else
    backend.in_place(data.ptr().template as<array>(),
		     data.stride(0), data.stride(1), 
		     data.size(0), data.size(1));
} 

template <typename T, int S, typename B1, typename B2>
void out_of_place(fft_backend<2, T, T, S>  &backend,
		  storage_format_type storage_format,
		  dda::Rt_data<B1, dda::in> &in_data,
		  storage_format_type,
		  dda::Rt_data<B2, dda::out> &out_data)
{
  if (storage_format == split_complex)
    backend.out_of_place(in_data.non_const_ptr().template as<split_complex>(),
			 in_data.stride(0), in_data.stride(1),
			 out_data.ptr().template as<split_complex>(),
			 out_data.stride(0), out_data.stride(1),
			 in_data.size(0), in_data.size(1));
  else
    backend.out_of_place(in_data.non_const_ptr().template as<array>(),
			 in_data.stride(0), in_data.stride(1),
			 out_data.ptr().template as<array>(),
			 out_data.stride(0), out_data.stride(1),
			 in_data.size(0), in_data.size(1));
} 

template <typename T, int S, typename B1, typename B2>
void out_of_place(fft_backend<2, T, complex<T>, S>  &backend,
		  storage_format_type,
		  dda::Rt_data<B1, dda::in> &in_data,
		  storage_format_type storage_format,
		  dda::Rt_data<B2, dda::out> &out_data)
{
  if (storage_format == split_complex)
    backend.out_of_place(in_data.non_const_ptr(),
			 in_data.stride(0), in_data.stride(1),
			 out_data.ptr().template as<split_complex>(),
			 out_data.stride(0), out_data.stride(1),
			 in_data.size(0), in_data.size(1));
  else
    backend.out_of_place(in_data.non_const_ptr(),
			 in_data.stride(0), in_data.stride(1),
			 out_data.ptr().template as<array>(),
			 out_data.stride(0), out_data.stride(1),
			 in_data.size(0), in_data.size(1));
} 

template <typename T, int S, typename B1, typename B2>
void out_of_place(fft_backend<2, complex<T>, T, S>  &backend,
		  storage_format_type storage_format,
		  dda::Rt_data<B1, dda::in> &in_data,
		  storage_format_type,
		  dda::Rt_data<B2, dda::out> &out_data)
{
  if (storage_format == split_complex)
    backend.out_of_place(in_data.non_const_ptr().template as<split_complex>(),
			 in_data.stride(0), in_data.stride(1),
			 out_data.ptr(),
			 out_data.stride(0), out_data.stride(1),
			 out_data.size(0), out_data.size(1));
  else
    backend.out_of_place(in_data.non_const_ptr().template as<array>(),
			 in_data.stride(0), in_data.stride(1),
			 out_data.ptr(),
			 out_data.stride(0), out_data.stride(1),
			 out_data.size(0), out_data.size(1));
} 

template <typename T, int S, typename B>
void in_place(fft_backend<3, T, T, S>  &backend,
	      storage_format_type storage_format,
	      dda::Rt_data<B, dda::inout> &data)
{
  if (storage_format == split_complex) 
    backend.in_place(data.ptr().template as<split_complex>(),
		     data.stride(0),
		     data.stride(1),
		     data.stride(2),
		     data.size(0),
		     data.size(1),
		     data.size(2));
  else
    backend.in_place(data.ptr().template as<array>(),
		     data.stride(0),
		     data.stride(1),
		     data.stride(2), 
		     data.size(0),
		     data.size(1),
		     data.size(2));
} 

template <typename T, int S, typename B1, typename B2>
void out_of_place(fft_backend<3, T, T, S>  &backend,
		  storage_format_type storage_format,
		  dda::Rt_data<B1, dda::in> &in_data,
		  storage_format_type,
		  dda::Rt_data<B2, dda::out> &out_data)
{
  if (storage_format == split_complex) 
    backend.out_of_place(in_data.non_const_ptr().template as<split_complex>(),
			 in_data.stride(0),
			 in_data.stride(1),
			 in_data.stride(2),
			 out_data.ptr().template as<split_complex>(),
			 out_data.stride(0),
			 out_data.stride(1),
			 out_data.stride(2),
			 in_data.size(0),
			 in_data.size(1),
			 in_data.size(2));
  else
    backend.out_of_place(in_data.non_const_ptr().template as<array>(),
			 in_data.stride(0),
			 in_data.stride(1),
			 in_data.stride(2),
			 out_data.ptr().template as<array>(),
			 out_data.stride(0),
			 out_data.stride(1),
			 out_data.stride(2),
			 in_data.size(0),
			 in_data.size(1),
			 in_data.size(2));
} 

template <typename T, int S, typename B1, typename B2>
void out_of_place(fft_backend<3, T, complex<T>, S>  &backend,
		  storage_format_type,
		  dda::Rt_data<B1, dda::in> &in_data,
		  storage_format_type storage_format,
		  dda::Rt_data<B2, dda::out> &out_data)
{
  if (storage_format == split_complex) 
    backend.out_of_place(in_data.non_const_ptr(),
			 in_data.stride(0),
			 in_data.stride(1),
			 in_data.stride(2),
			 out_data.ptr().template as<split_complex>(),
			 out_data.stride(0),
			 out_data.stride(1),
			 out_data.stride(2),
			 in_data.size(0),
			 in_data.size(1),
			 in_data.size(2));
  else
    backend.out_of_place(in_data.non_const_ptr(),
			 in_data.stride(0),
			 in_data.stride(1),
			 in_data.stride(2),
			 out_data.ptr().template as<array>(),
			 out_data.stride(0),
			 out_data.stride(1),
			 out_data.stride(2),
			 in_data.size(0),
			 in_data.size(1),
			 in_data.size(2));
} 

template <typename T, int S, typename B1, typename B2>
void out_of_place(fft_backend<3, complex<T>, T, S>  &backend,
		  storage_format_type storage_format,
		  dda::Rt_data<B1, dda::in> &in_data,
		  storage_format_type,
		  dda::Rt_data<B2, dda::out> &out_data)
{
  if (storage_format == split_complex) 
    backend.out_of_place(in_data.non_const_ptr().template as<split_complex>(),
			 in_data.stride(0),
			 in_data.stride(1),
			 in_data.stride(2),
			 out_data.ptr(),
			 out_data.stride(0),
			 out_data.stride(1),
			 out_data.stride(2),
			 out_data.size(0),
			 out_data.size(1),
			 out_data.size(2));
  else
    backend.out_of_place(in_data.non_const_ptr().template as<array>(),
			 in_data.stride(0),
			 in_data.stride(1),
			 in_data.stride(2),
			 out_data.ptr(),
			 out_data.stride(0),
			 out_data.stride(1),
			 out_data.stride(2),
			 out_data.size(0),
			 out_data.size(1),
			 out_data.size(2));
} 

template <typename T, int A, int D, typename B>
void in_place(fftm_backend<T, T, A, D>  &backend,
	      storage_format_type storage_format,
	      dda::Rt_data<B, dda::inout> &data)
{
  if (storage_format == split_complex) 
    backend.in_place(data.ptr().template as<split_complex>(),
		     data.stride(0), data.stride(1),
		     data.size(0), data.size(1));
  else
    backend.in_place(data.ptr().template as<array>(),
		     data.stride(0), data.stride(1), 
		     data.size(0), data.size(1));
} 

template <typename T, int A, int D, typename B1, typename B2>
void out_of_place(fftm_backend<T, T, A, D>  &backend,
		  storage_format_type storage_format,
		  dda::Rt_data<B1, dda::in> &in_data,
		  storage_format_type,
		  dda::Rt_data<B2, dda::out> &out_data)
{
  if (storage_format == split_complex) 
    backend.out_of_place(in_data.non_const_ptr().template as<split_complex>(),
			 in_data.stride(0), in_data.stride(1),
			 out_data.ptr().template as<split_complex>(),
			 out_data.stride(0), out_data.stride(1),
			 in_data.size(0), in_data.size(1));
  else
    backend.out_of_place(in_data.non_const_ptr().template as<array>(),
			 in_data.stride(0), in_data.stride(1),
			 out_data.ptr().template as<array>(),
			 out_data.stride(0), out_data.stride(1),
			 in_data.size(0), in_data.size(1));
} 

template <typename T, int A, int D, typename B1, typename B2>
void out_of_place(fftm_backend<T, complex<T>, A, D>  &backend,
		  storage_format_type,
		  dda::Rt_data<B1, dda::in> &in_data,
		  storage_format_type storage_format,
		  dda::Rt_data<B2, dda::out> &out_data)
{
  if (storage_format == split_complex) 
    backend.out_of_place(in_data.non_const_ptr(),
			 in_data.stride(0), in_data.stride(1),
			 out_data.ptr().template as<split_complex>(),
			 out_data.stride(0), out_data.stride(1),
			 in_data.size(0), in_data.size(1));
  else
    backend.out_of_place(in_data.non_const_ptr(),
			 in_data.stride(0), in_data.stride(1),
			 out_data.ptr().template as<array>(),
			 out_data.stride(0), out_data.stride(1),
			 in_data.size(0), in_data.size(1));
} 

template <typename T, int A, int D, typename B1, typename B2>
void out_of_place(fftm_backend<complex<T>, T, A, D>  &backend,
		  storage_format_type storage_format,
		  dda::Rt_data<B1, dda::in> &in_data,
		  storage_format_type,
		  dda::Rt_data<B2, dda::out> &out_data)
{
  if (storage_format == split_complex) 
    backend.out_of_place(in_data.non_const_ptr().template as<split_complex>(),
			 in_data.stride(0), in_data.stride(1),
			 out_data.ptr(),
			 out_data.stride(0), out_data.stride(1),
			 out_data.size(0), out_data.size(1));
  else
    backend.out_of_place(in_data.non_const_ptr().template as<array>(),
			 in_data.stride(0), in_data.stride(1),
			 out_data.ptr(),
			 out_data.stride(0), out_data.stride(1),
			 out_data.size(0), out_data.size(1));
} 

} // namespace ovxx::fft::detail
template <typename I, typename O>
inline length_type
select_fft_size(length_type in_size, length_type out_size)
{
  return detail::fft_size<I, O>::exec(in_size, out_size);
}

template <typename I, typename O>
class workspace<1, I, O>
{
  typedef typename scalar_of<O>::type scalar_type;

public:
  template <typename BE>
  workspace(BE *backend, Domain<1> const &in, Domain<1> const &out,
	    scalar_type scale)
    : scale_(scale)
  {
  }
  
  template <typename BE, typename B1, typename B2>
  void out_of_place(BE &backend, B1 const &in, B2 &out)
  {
    Rt_layout<1> rtl_in = block_layout<1>(in); 
    Rt_layout<1> rtl_out = block_layout<1>(out); 
    backend.query_layout(rtl_in, rtl_out);
    bool force_copy = backend.requires_copy(rtl_in);
    {
      dda::Rt_data<B1, dda::in> in_data(in, force_copy, rtl_in, backend.input_buffer());
      dda::Rt_data<B2, dda::out> out_data(out, rtl_out, backend.output_buffer());
      detail::out_of_place(backend, rtl_in.storage_format, in_data, rtl_out.storage_format, out_data);
    }
    if (!backend.supports_scale() && scale_ != scalar_type(1.))
    {
      typename view_of<B2>::type view(out);
      view *= scale_;
    }
  }

  template <typename BE, typename B>
  void in_place(BE &backend, B &inout)
  {
    Rt_layout<1> rtl_inout = block_layout<1>(inout); 
    backend.query_layout(rtl_inout);
    {
      dda::Rt_data<B, dda::inout> data(inout, rtl_inout, backend.input_buffer());
      detail::in_place(backend, rtl_inout.storage_format, data);
    }
    if (!backend.supports_scale() && scale_ != scalar_type(1.))
    {
      typename view_of<B>::type view(inout);
      view *= scale_;
    }
  }

private:
  scalar_type scale_;
};

template <typename I, typename O>
class workspace<2, I, O>
{
  typedef typename scalar_of<O>::type scalar_type;

public:
  template <typename BE>
  workspace(BE* /*backend*/, Domain<2> const &in, Domain<2> const &out,
	    scalar_type scale)
    : scale_(scale)
  {
  }
  
  template <typename BE, typename B1, typename B2>
  void out_of_place(BE &backend, B1 const &in, B2 &out)
  {
    Rt_layout<2> rtl_in = block_layout<2>(in); 
    Rt_layout<2> rtl_out = block_layout<2>(out); 
    backend.query_layout(rtl_in, rtl_out);
    bool force_copy = backend.requires_copy(rtl_in);
    {
      dda::Rt_data<B1, dda::in> in_data(in, force_copy, rtl_in, backend.input_buffer());
      dda::Rt_data<B2, dda::out> out_data(out, rtl_out, backend.output_buffer());
      detail::out_of_place(backend, rtl_in.storage_format, in_data, rtl_out.storage_format, out_data);
    }
    if (!backend.supports_scale() && scale_ != scalar_type(1.))
    {
      typename view_of<B2>::type view(out);
      view *= scale_;
    }
  }

  template <typename BE, typename B>
  void in_place(BE &backend, B &inout)
  {
    Rt_layout<2> rtl_inout = block_layout<2>(inout); 
    backend.query_layout(rtl_inout);
    {
      dda::Rt_data<B, dda::inout> inout_data(inout, rtl_inout, backend.input_buffer());
      detail::in_place(backend, rtl_inout.storage_format, inout_data);
    }
    if (!backend.supports_scale() && scale_ != scalar_type(1.))
    {
      typename view_of<B>::type view(inout);
      view *= scale_;
    }
  }

private:
  scalar_type scale_;
};

template <typename I, typename O>
class workspace<3, I, O>
{
  typedef typename scalar_of<O>::type scalar_type;

public:
  template <typename BE>
  workspace(BE* /*backend*/, Domain<3> const &in, Domain<3> const &out,
	    scalar_type scale)
    : scale_(scale)
  {
  }
  
  template <typename BE, typename B1, typename B2>
  void out_of_place(BE &backend, B1 const &in, B2 &out)
  {
    Rt_layout<3> rtl_in = block_layout<3>(in); 
    Rt_layout<3> rtl_out = block_layout<3>(out); 
    backend.query_layout(rtl_in, rtl_out);
    bool force_copy = backend.requires_copy(rtl_in);
    {
      dda::Rt_data<B1, dda::in> in_data(in, force_copy, rtl_in, backend.input_buffer());
      dda::Rt_data<B2, dda::out> out_data(out, rtl_out, backend.output_buffer());
      detail::out_of_place(backend, rtl_in.storage_format, in_data, rtl_out.storage_format, out_data);
    }
    if (!backend.supports_scale() && scale_ != scalar_type(1.))
    {
      typename view_of<B2>::type view(out);
      view *= scale_;
    }
  }

  template <typename BE, typename B>
  void in_place(BE &backend, B &inout)
  {
    Rt_layout<3> rtl_inout = block_layout<3>(inout); 
    backend.query_layout(rtl_inout);
    {
      dda::Rt_data<B, dda::inout> inout_data(inout, rtl_inout, backend.input_buffer());
      detail::in_place(backend, rtl_inout.storage_format, inout_data);
    }
    if (!backend.supports_scale() && scale_ != scalar_type(1.))
    {
      typename view_of<B>::type view(inout);
      view *= scale_;
    }
  }

private:
  scalar_type scale_;
};

} // namespace ovxx::signal::fft
} // namespace ovxx::signal
} // namespace ovxx

#endif
