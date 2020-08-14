//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.


// This file is included multiple times into the same
// translation unit. It is parametrized using the
// following macros:
//   SCALAR_TYPE (float, double)
//   FFTW(x) (prefixes 'x' with 'fftwf_', 'fftw_')

#include <cassert>

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <ovxx/signal/fft/backend.hpp>
#include <ovxx/signal/fft/util.hpp>
#include <ovxx/fftw/fft.hpp>
#include <vsip/dense.hpp>
#include <fftw3.h>
#include <cstring> // for memcpy

namespace ovxx
{
namespace fftw
{
using ovxx::signal::fft::fft_backend;
using ovxx::signal::fft::fftm_backend;
using ovxx::signal::fft::exponent;
using ovxx::signal::fft::io_size;

template <dimension_type D>
Domain<D> FFTW(iosize)(Domain<D> const &dom)
{ return io_size<D, complex<SCALAR_TYPE>, SCALAR_TYPE, D-1>::size(dom);}

template <dimension_type D>
struct planner<D, complex<SCALAR_TYPE>, complex<SCALAR_TYPE> >
{
  planner(Domain<D> const &dom, int exp, int flags, length_type mult = 1)
    : aligned_(!(flags & FFTW_UNALIGNED)),
      mult_(mult)
  {
    Applied_layout<Rt_layout<D> > layout
      (Rt_layout<D>(aligned_ ? aligned : dense, tuple<0,1,2>(), complex_storage_format, OVXX_ALLOC_ALIGNMENT),
       dom, sizeof(SCALAR_TYPE));
    length_type total_size = layout.total_size();
    if (mult_ > 1)
    {
      // For fftm we need more buffer...
      Applied_layout<Rt_layout<2> > multi_layout
       	(Rt_layout<2>(aligned_ ? aligned : dense, tuple<0,1,2>(), complex_storage_format, OVXX_ALLOC_ALIGNMENT),
       	 Domain<2>(mult_, dom[0]), sizeof(SCALAR_TYPE));
      total_size = multi_layout.total_size();
    }
    in_buffer_ = aligned_array<complex<SCALAR_TYPE> >(32, total_size);
    out_buffer_ = aligned_array<complex<SCALAR_TYPE> >(32, total_size);

    FFTW(iodim) dims[D];
    for (index_type i = 0; i != D; ++i) 
    { 
      dims[i].n = layout.size(i);
      dims[i].is = dims[i].os = layout.stride(i);
    }
    if (complex_storage_format == split_complex)
    {
      std::pair<SCALAR_TYPE*,SCALAR_TYPE*> in = 
	array_cast<split_complex>(in_buffer_);
      std::pair<SCALAR_TYPE*,SCALAR_TYPE*> out = 
	array_cast<split_complex>(out_buffer_);
      plan_ip_ = FFTW(plan_guru_split_dft)(D, dims, 0, 0,
					   in.first, in.second,
					   in.first, in.second,
					   flags);
      plan_op_ = FFTW(plan_guru_split_dft)(D, dims, 0, 0,
					   in.first, in.second,
					   out.first, out.second,
					   flags);
    }
    else
    {
      FFTW(complex) *in =
	reinterpret_cast<FFTW(complex)*>(in_buffer_.get());
      FFTW(complex) *out =
	reinterpret_cast<FFTW(complex)*>(out_buffer_.get());
      plan_ip_ = FFTW(plan_guru_dft)(D, dims, 0, 0,
				     in,
				     in,
				     exp, flags);
      plan_op_ = FFTW(plan_guru_dft)(D, dims, 0, 0,
				     in,
				     out,
				     exp, flags);
    }
    if (!plan_ip_) OVXX_DO_THROW(std::bad_alloc());
    if (!plan_op_)
    {
      FFTW(destroy_plan)(plan_ip_);
      OVXX_DO_THROW(std::bad_alloc());
    }
  }
  ~planner() VSIP_NOTHROW
  {
    FFTW(destroy_plan)(plan_op_);
    FFTW(destroy_plan)(plan_ip_);
  }

  aligned_array<complex<SCALAR_TYPE> > in_buffer_;
  aligned_array<complex<SCALAR_TYPE> > out_buffer_;
  FFTW(plan) plan_ip_;
  FFTW(plan) plan_op_;
  bool aligned_;
  length_type mult_;
};

template <dimension_type D>
struct planner<D, SCALAR_TYPE, complex<SCALAR_TYPE> >
{
  planner(Domain<D> dom, int A, int flags, length_type mult = 1)
    VSIP_THROW((std::bad_alloc))
  : aligned_(!(flags & FFTW_UNALIGNED)),
    mult_(mult)
  {
    // FFTW requires the 'special dimension' to be the major one.
    // Thus in some cases this means FFTW will operate on the
    // transpose of the argument. See query_layout(), where this
    // requirement is communicated to the workspace, and the corner-turn
    // is handled.
    dom = turn(dom, A);
    Applied_layout<Rt_layout<D> >
      in_layout(Rt_layout<D>(aligned_ ? aligned : dense,
			     tuple<0,1,2>(),
			     complex_storage_format,
			     OVXX_ALLOC_ALIGNMENT),
		dom, sizeof(SCALAR_TYPE));
    Applied_layout<Rt_layout<D> >
      out_layout(Rt_layout<D>(aligned_ ? aligned : dense,
			      tuple<0,1,2>(),
			      complex_storage_format,
			      OVXX_ALLOC_ALIGNMENT),
		 FFTW(iosize)(dom), sizeof(SCALAR_TYPE));
    length_type in_total_size = in_layout.total_size();
    length_type out_total_size = out_layout.total_size();
    if (mult_ > 1)
    {
      // For fftm we need more buffer...
      Applied_layout<Rt_layout<2> > 
	in_multi_layout(Rt_layout<2>(aligned_ ? aligned : dense,
				     tuple<0,1,2>(),
				     complex_storage_format,
				     OVXX_ALLOC_ALIGNMENT),
			Domain<2>(mult_, dom[0]), sizeof(SCALAR_TYPE));
      Applied_layout<Rt_layout<2> > 
	out_multi_layout(Rt_layout<2>(aligned_ ? aligned : dense,
				      tuple<0,1,2>(),
				      complex_storage_format,
				      OVXX_ALLOC_ALIGNMENT),
			 Domain<2>(mult_, FFTW(iosize)(dom[0])), sizeof(SCALAR_TYPE));
      in_total_size = in_multi_layout.total_size();
      out_total_size = out_multi_layout.total_size();
    }
    in_buffer_ = aligned_array<SCALAR_TYPE>(32, in_total_size);
    out_buffer_ = aligned_array<complex<SCALAR_TYPE> >(32, out_total_size);

    FFTW(iodim) dims[D];
    for (index_type i = 0; i != D; ++i) 
    {
      dims[i].n = in_layout.size(i);
      dims[i].is = in_layout.stride(i); 
      dims[i].os = out_layout.stride(i); 
    }
    if (complex_storage_format == split_complex)
    {
      SCALAR_TYPE *in = in_buffer_.get();
      std::pair<SCALAR_TYPE*,SCALAR_TYPE*> out = 
	array_cast<split_complex>(out_buffer_);
      plan_ = FFTW(plan_guru_split_dft_r2c)(D, dims, 0, 0,
					    in, out.first, out.second,
					    flags);
    }
    else
    {
      SCALAR_TYPE *in = in_buffer_.get();
      FFTW(complex) *out = reinterpret_cast<FFTW(complex)*>(out_buffer_.get());
      plan_ = FFTW(plan_guru_dft_r2c)(D, dims, 0, 0, in, out, flags);
    }
    if (!plan_) OVXX_DO_THROW(std::bad_alloc());
  }
  ~planner() VSIP_NOTHROW { FFTW(destroy_plan)(plan_);}

  aligned_array<SCALAR_TYPE> in_buffer_;
  aligned_array<complex<SCALAR_TYPE> > out_buffer_;
  FFTW(plan) plan_;
  bool aligned_;
  length_type mult_;
};

template <vsip::dimension_type D>
struct planner<D, complex<SCALAR_TYPE>, SCALAR_TYPE>
{
  planner(Domain<D> dom, int A, int flags, length_type mult = 1)
    VSIP_THROW((std::bad_alloc))
  : aligned_(!(flags & FFTW_UNALIGNED)),
    mult_(mult)
  {
    // FFTW requires the 'special dimension' to be the major one.
    // Thus in some cases this means FFTW will operate on the
    // transpose of the argument. See query_layout(), where this
    // requirement is communicated to the workspace, and the corner-turn
    // is handled.
    dom = turn(dom, A);
    Applied_layout<Rt_layout<D> > 
      in_layout(Rt_layout<D>(aligned_ ? aligned : dense,
			     tuple<0,1,2>(),
			     complex_storage_format,
			     OVXX_ALLOC_ALIGNMENT),
		FFTW(iosize)(dom), sizeof(SCALAR_TYPE));
    Applied_layout<Rt_layout<D> > 
      out_layout(Rt_layout<D>(aligned_ ? aligned : dense,
			      tuple<0,1,2>(),
			      complex_storage_format,
			      OVXX_ALLOC_ALIGNMENT),
		 dom, sizeof(SCALAR_TYPE));
    length_type in_total_size = in_layout.total_size();
    length_type out_total_size = out_layout.total_size();
    if (mult_ > 1)
    {
      // For fftm we need more buffer...
      Applied_layout<Rt_layout<2> > 
	in_multi_layout(Rt_layout<2>(aligned_ ? aligned : dense,
				     tuple<0,1,2>(),
				     complex_storage_format,
				     OVXX_ALLOC_ALIGNMENT),
			Domain<2>(mult_, FFTW(iosize)(dom[0])), sizeof(SCALAR_TYPE));
      Applied_layout<Rt_layout<2> > 
	out_multi_layout(Rt_layout<2>(aligned_ ? aligned : dense,
				      tuple<0,1,2>(),
				      complex_storage_format,
				      OVXX_ALLOC_ALIGNMENT),
			 Domain<2>(mult_, dom[0]), sizeof(SCALAR_TYPE));
      in_total_size = in_multi_layout.total_size();
      out_total_size = out_multi_layout.total_size();
    }
    in_buffer_ = aligned_array<complex<SCALAR_TYPE> >(32, in_total_size);
    out_buffer_ = aligned_array<SCALAR_TYPE>(32, out_total_size);
    
    FFTW(iodim) dims[D];    
    for (index_type i = 0; i != D; ++i) 
    {
      dims[i].n = out_layout.size(i);
      dims[i].is = in_layout.stride(i); 
      dims[i].os = out_layout.stride(i); 
    }
    if (complex_storage_format == split_complex)
    {
      std::pair<SCALAR_TYPE*,SCALAR_TYPE*> in = 
	array_cast<split_complex>(in_buffer_);
      SCALAR_TYPE *out = out_buffer_.get();
      plan_ = FFTW(plan_guru_split_dft_c2r)(D, dims, 0, 0,
					    in.first, in.second, out,
					    flags);
    }
    else
    {
      FFTW(complex) *in = reinterpret_cast<FFTW(complex)*>(in_buffer_.get());
      SCALAR_TYPE *out = out_buffer_.get();
      plan_ = FFTW(plan_guru_dft_c2r)(D, dims, 0, 0, in, out, flags);
    }
    if (!plan_) OVXX_DO_THROW(std::bad_alloc());
  }
  ~planner() VSIP_NOTHROW { FFTW(destroy_plan)(plan_);}

  aligned_array<complex<SCALAR_TYPE> > in_buffer_;
  aligned_array<SCALAR_TYPE> out_buffer_;
  FFTW(plan) plan_;
  bool aligned_;
  length_type mult_;
};

// 1D complex -> complex FFT

template <int S>
class fft<1, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, S>
  : public fft_backend<1, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, S>,
    planner<1, complex<SCALAR_TYPE>, complex<SCALAR_TYPE> >
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  fft(Domain<1> const &dom, unsigned number)
    : planner<1, ctype, ctype>
      (dom, exponent<ctype, ctype, S>::value, make_flags<ctype>(dom, number))
  {}
  virtual void query_layout(Rt_layout<1> &inout)
  {
    inout.packing = this->aligned_ ? aligned : dense;
    inout.alignment = OVXX_ALLOC_ALIGNMENT;
    inout.order = tuple<0, 1, 2>();
    inout.storage_format = complex_storage_format;
  }
  virtual void query_layout(Rt_layout<1> &in, Rt_layout<1> &out)
  {
    query_layout(in);
    out = in;
  }
  virtual ctype *input_buffer() { return in_buffer_.get();}
  virtual ctype *output_buffer() { return out_buffer_.get();}
  virtual void in_place(ctype *inout, stride_type s, length_type l)
  {
    FFTW(execute_dft)(plan_ip_,
		      reinterpret_cast<FFTW(complex)*>(inout),
		      reinterpret_cast<FFTW(complex)*>(inout));
  }
  virtual void in_place(ztype inout, stride_type s, length_type l)
  {
    if (S == fft_fwd)
      FFTW(execute_split_dft)(plan_ip_,
			      inout.first, inout.second,
			      inout.first, inout.second);
    else
      FFTW(execute_split_dft)(plan_ip_,
			      inout.second, inout.first,
			      inout.second, inout.first);
  }
  virtual void out_of_place(ctype *in, stride_type in_stride,
			    ctype *out, stride_type out_stride,
			    length_type length)
  {
    FFTW(execute_dft)(plan_op_,
		      reinterpret_cast<FFTW(complex)*>(in),
		      reinterpret_cast<FFTW(complex)*>(out));
  }
  virtual void out_of_place(ztype in, stride_type in_stride,
			    ztype out, stride_type out_stride,
			    length_type length)
  {
    // the inverse DFT is equal to the forwards DFT 
    // with the real and imaginary parts swapped
    if (S == fft_fwd)
      FFTW(execute_split_dft)(plan_op_,
			      in.first,  in.second,
			      out.first, out.second);
    else
      FFTW(execute_split_dft)(plan_op_,
			      in.second,  in.first,
			      out.second, out.first);
  }
};

// 1D real -> complex FFT

template <>
class fft<1, SCALAR_TYPE, complex<SCALAR_TYPE>, 0>
  : public fft_backend<1, SCALAR_TYPE, complex<SCALAR_TYPE>, 0>,
    planner<1, SCALAR_TYPE, complex<SCALAR_TYPE> >
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  fft(Domain<1> const &dom, unsigned number)
    : planner<1, rtype, ctype>(dom, 0, make_flags<rtype>(dom, number))
  {}
  virtual void query_layout(Rt_layout<1> &in, Rt_layout<1> &out)
  {
    in.packing = this->aligned_ ? aligned : dense;
    in.alignment = OVXX_ALLOC_ALIGNMENT;
    in.order = tuple<0, 1, 2>();
    in.storage_format = array;
    out = in;
    out.storage_format = complex_storage_format;
  }
  virtual rtype *input_buffer() { return in_buffer_.get();}
  virtual ctype *output_buffer() { return out_buffer_.get();}
  virtual void out_of_place(rtype *in, stride_type,
			    ctype *out, stride_type,
			    length_type)
  {
    FFTW(execute_dft_r2c)(plan_,
			  in, reinterpret_cast<FFTW(complex)*>(out));
  }
  virtual void out_of_place(rtype *in, stride_type,
			    ztype out, stride_type,
			    length_type)
  {
    FFTW(execute_split_dft_r2c)(plan_,
			  in, out.first, out.second);
  }
};

// 1D complex -> real FFT

template <>
class fft<1, complex<SCALAR_TYPE>, SCALAR_TYPE, 0>
  : public fft_backend<1, complex<SCALAR_TYPE>, SCALAR_TYPE, 0>,
    planner<1, complex<SCALAR_TYPE>, SCALAR_TYPE>
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  fft(Domain<1> const &dom, unsigned number)
    : planner<1, ctype, rtype>(dom, 0, make_flags<rtype>(dom, number))
  {}
  virtual void query_layout(Rt_layout<1> &in, Rt_layout<1> &out)
  {
    in.packing = this->aligned_ ? aligned : dense;
    in.alignment = OVXX_ALLOC_ALIGNMENT;
    in.order = tuple<0, 1, 2>();
    in.storage_format = complex_storage_format;
    out = in;
    out.storage_format = array;
  }
  virtual ctype *input_buffer() { return in_buffer_.get();}
  virtual rtype *output_buffer() { return out_buffer_.get();}
  virtual void out_of_place(ctype *in, stride_type,
			    rtype *out, stride_type,
			    length_type)
  {
    FFTW(execute_dft_c2r)(plan_, reinterpret_cast<FFTW(complex)*>(in), out);
  }
  virtual void out_of_place(ztype in, stride_type,
			    rtype *out, stride_type,
			    length_type length)
  {
    FFTW(execute_split_dft_c2r)(plan_, in.first, in.second, out);
  }
};

// 2D complex -> complex FFT

template <int S>
class fft<2, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, S>
  : public fft_backend<2, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, S>,
    planner<2, complex<SCALAR_TYPE>, complex<SCALAR_TYPE> >
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  fft(Domain<2> const &dom, unsigned number)
    : planner<2, ctype, ctype>
      (dom, exponent<ctype, ctype, S>::value, make_flags(number))
  {}
  virtual void query_layout(Rt_layout<2> &inout)
  {
    inout.packing = this->aligned_ ? aligned : dense;
    inout.alignment = OVXX_ALLOC_ALIGNMENT;
    inout.order = tuple<0,1,2>();
    inout.storage_format = complex_storage_format;
  }
  virtual void query_layout(Rt_layout<2> &in, Rt_layout<2> &out)
  {
    query_layout(in);
    out = in;
  }
  virtual ctype *input_buffer() { return in_buffer_.get();}
  virtual ctype *output_buffer() { return out_buffer_.get();}
  virtual void in_place(ctype *inout,
			stride_type, stride_type,
			length_type, length_type)
  {
    FFTW(execute_dft)(plan_ip_,
		      reinterpret_cast<FFTW(complex)*>(inout),
		      reinterpret_cast<FFTW(complex)*>(inout));
  }
  virtual void in_place(ztype inout, stride_type, stride_type,
			length_type, length_type)
  {
    // the inverse DFT is equal to the forwards DFT 
    // with the real and imaginary parts swapped
    if (S == fft_fwd)
      FFTW(execute_split_dft)(plan_ip_,
			      inout.first, inout.second,
			      inout.first, inout.second);
    else
      FFTW(execute_split_dft)(plan_ip_,
			      inout.second, inout.first,
			      inout.second, inout.first);
  }
  virtual void out_of_place(ctype *in, stride_type, stride_type,
			    ctype *out, stride_type, stride_type,
			    length_type, length_type)
  {
    FFTW(execute_dft)(plan_op_,
		      reinterpret_cast<FFTW(complex)*>(in), 
		      reinterpret_cast<FFTW(complex)*>(out));
  }
  virtual void out_of_place(ztype in, stride_type, stride_type,
			    ztype out, stride_type, stride_type,
			    length_type, length_type cols)
  {
    // the inverse DFT is equal to the forwards DFT 
    // with the real and imaginary parts swapped
    if (S == fft_fwd)
      FFTW(execute_split_dft)(plan_op_,
			      in.first, in.second,
			      out.first, out.second);
    else
      FFTW(execute_split_dft)(plan_op_,
			      in.second, in.first,
			      out.second, out.first);
  }
};

// 2D real -> complex FFT

template <int A>
class fft<2, SCALAR_TYPE, complex<SCALAR_TYPE>, A>
  : public fft_backend<2, SCALAR_TYPE, complex<SCALAR_TYPE>, A>,
    planner<2, SCALAR_TYPE, complex<SCALAR_TYPE> >
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  fft(Domain<2> const &dom, unsigned number)
    : planner<2, rtype, ctype>(dom, A, make_flags(number))
  {}
  virtual void query_layout(Rt_layout<2> &in, Rt_layout<2> &out)
  {
    in.packing = this->aligned_ ? aligned : dense;
    in.alignment = OVXX_ALLOC_ALIGNMENT;
    if (A == 0) in.order = tuple<1, 0, 2>();
    else in.order = tuple<0, 1, 2>();
    in.storage_format = array;
    out = in;
    out.storage_format = complex_storage_format;
  }
  virtual rtype *input_buffer() { return in_buffer_.get();}
  virtual ctype *output_buffer() { return out_buffer_.get();}
  virtual void out_of_place(rtype *in,
			    stride_type, stride_type,
			    ctype *out,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    FFTW(execute_dft_r2c)(plan_, in, reinterpret_cast<FFTW(complex)*>(out));
  }
  virtual void out_of_place(rtype *in,
			    stride_type, stride_type,
			    ztype out,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    FFTW(execute_split_dft_r2c)(plan_, in, out.first, out.second);
  }
};

// 2D complex -> real FFT

template <int A>
class fft<2, complex<SCALAR_TYPE>, SCALAR_TYPE, A>
  : public fft_backend<2, complex<SCALAR_TYPE>, SCALAR_TYPE, A>,
    planner<2, complex<SCALAR_TYPE>, SCALAR_TYPE>
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  fft(Domain<2> const &dom, unsigned number)
    : planner<2, ctype, rtype>(dom, A, make_flags(number, false))
  {}
  virtual void query_layout(Rt_layout<2> &in, Rt_layout<2> &out)
  {
    in.packing = this->aligned_ ? aligned : dense;
    in.alignment = OVXX_ALLOC_ALIGNMENT;
    if (A == 0) in.order = tuple<1, 0, 2>();
    else in.order = tuple<0, 1, 2>();
    in.storage_format = complex_storage_format;
    out = in;
    out.storage_format = array;
  }
  virtual ctype *input_buffer() { return in_buffer_.get();}
  virtual rtype *output_buffer() { return out_buffer_.get();}
  // Multi-dimensional C2R FFTs overwrite the input buffer.
  virtual bool requires_copy(Rt_layout<2> &) { return true;}
  virtual void out_of_place(ctype *in, stride_type, stride_type,
			    rtype *out, stride_type, stride_type,
			    length_type, length_type)
  {
    FFTW(execute_dft_c2r)(plan_, reinterpret_cast<FFTW(complex)*>(in), out);
  }
  virtual void out_of_place(ztype in, stride_type, stride_type,
			    rtype *out, stride_type, stride_type,
			    length_type, length_type)
  {
    FFTW(execute_split_dft_c2r)(plan_, in.first, in.second, out);
  }
};

// 3D complex -> complex FFT

template <int S>
class fft<3, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, S>
  : public fft_backend<3, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, S>,
    planner<3, complex<SCALAR_TYPE>, complex<SCALAR_TYPE> >

{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  fft(Domain<3> const &dom, unsigned number)
    : planner<3, ctype, ctype>(dom, exponent<ctype, ctype, S>::value, make_flags(number))
  {}
  virtual void query_layout(Rt_layout<3> &inout)
  {
    inout.packing = this->aligned_ ? aligned : dense;
    inout.alignment = OVXX_ALLOC_ALIGNMENT;
    inout.order = tuple<0,1,2>();
    inout.storage_format = complex_storage_format;
  }
  virtual void query_layout(Rt_layout<3> &in, Rt_layout<3> &out)
  {
    query_layout(in);
    out = in;
  }
  virtual ctype *input_buffer() { return in_buffer_.get();}
  virtual ctype *output_buffer() { return out_buffer_.get();}
  virtual void in_place(ctype *inout,
			stride_type, stride_type, stride_type,
			length_type, length_type, length_type)
  {
    FFTW(execute_dft)(plan_ip_,
		      reinterpret_cast<FFTW(complex)*>(inout),
		      reinterpret_cast<FFTW(complex)*>(inout));
  }
  virtual void in_place(ztype inout,
			stride_type, stride_type, stride_type,
			length_type, length_type, length_type)
  {
    // the inverse DFT is equal to the forwards DFT 
    // with the real and imaginary parts swapped
    if (S == fft_fwd)
      FFTW(execute_split_dft)(plan_ip_,
			      inout.first, inout.second,
			      inout.first, inout.second);
    else
      FFTW(execute_split_dft)(plan_ip_,
			      inout.second, inout.first,
			      inout.second, inout.first);
  }
  virtual void out_of_place(ctype *in, stride_type, stride_type, stride_type,
			    ctype *out, stride_type, stride_type, stride_type,
			    length_type, length_type, length_type)
  {
    FFTW(execute_dft)(plan_op_,
		      reinterpret_cast<FFTW(complex)*>(in), 
		      reinterpret_cast<FFTW(complex)*>(out));
  }
  virtual void out_of_place(ztype in, stride_type, stride_type, stride_type,
			    ztype out, stride_type, stride_type, stride_type,
			    length_type, length_type, length_type)
  {
    // the inverse DFT is equal to the forwards DFT 
    // with the real and imaginary parts swapped
    if (S == fft_fwd)
      FFTW(execute_split_dft)(plan_op_,
			      in.first, in.second,
			      out.first, out.second);
    else
      FFTW(execute_split_dft)(plan_op_,
			      in.second, in.first,
			      out.second, out.first);
  }
};

// 3D real -> complex FFT

template <int A>
class fft<3, SCALAR_TYPE, complex<SCALAR_TYPE>, A>
  : public fft_backend<3, SCALAR_TYPE, complex<SCALAR_TYPE>, A>,
    planner<3, SCALAR_TYPE, complex<SCALAR_TYPE> >
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  fft(Domain<3> const &dom, unsigned number)
    : planner<3, rtype, ctype>(dom, A, make_flags(number))
  {}
  virtual void query_layout(Rt_layout<3> &in, Rt_layout<3> &out)
  {
    in.packing = this->aligned_ ? aligned : dense;
    in.alignment = OVXX_ALLOC_ALIGNMENT;
    switch (A)
    {
      case 0: in.order = tuple<2, 1, 0>(); break;
      case 1: in.order = tuple<0, 2, 1>(); break;
      default: in.order = tuple<0, 1, 2>(); break;
    }
    in.storage_format = array;
    out = in;
    out.storage_format = complex_storage_format;
  }
  virtual rtype *input_buffer() { return in_buffer_.get();}
  virtual ctype *output_buffer() { return out_buffer_.get();}
  virtual void out_of_place(rtype *in, stride_type, stride_type, stride_type,
			    ctype *out, stride_type, stride_type, stride_type,
			    length_type, length_type, length_type)
  {
    FFTW(execute_dft_r2c)(plan_,
			  in, reinterpret_cast<FFTW(complex)*>(out));
  }
  virtual void out_of_place(rtype *in, stride_type, stride_type, stride_type,
			    ztype out, stride_type, stride_type, stride_type,
			    length_type, length_type, length_type)
  {
    FFTW(execute_split_dft_r2c)(plan_,
				in, out.first, out.second);
  }
};

// 3D complex -> real FFT

template <int A>
class fft<3, complex<SCALAR_TYPE>, SCALAR_TYPE, A>
  : public fft_backend<3, complex<SCALAR_TYPE>, SCALAR_TYPE, A>,
    planner<3, complex<SCALAR_TYPE>, SCALAR_TYPE>
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  fft(Domain<3> const &dom, unsigned number)
    : planner<3, ctype, rtype>(dom, A, make_flags(number, false))
  {}
  virtual void query_layout(Rt_layout<3> &in, Rt_layout<3> &out)
  {
    in.packing = this->aligned_ ? aligned : dense;
    in.alignment = OVXX_ALLOC_ALIGNMENT;
    switch (A)
    {
      case 0: in.order = tuple<2, 1, 0>(); break;
      case 1: in.order = tuple<0, 2, 1>(); break;
      default: in.order = tuple<0, 1, 2>(); break;
    }
    in.storage_format = complex_storage_format;
    out = in;
    out.storage_format = array;
  }
  virtual ctype *input_buffer() { return in_buffer_.get();}
  virtual rtype *output_buffer() { return out_buffer_.get();}
  // Multi-dimensional C2R FFTs overwrite the input buffer.
  virtual bool requires_copy(Rt_layout<3> &) { return true;}
  virtual void out_of_place(ctype *in, stride_type, stride_type, stride_type,
			    rtype *out, stride_type, stride_type, stride_type,
			    length_type, length_type, length_type)
  {
    FFTW(execute_dft_c2r)(plan_,
			  reinterpret_cast<FFTW(complex)*>(in), out);
  }
  virtual void out_of_place(ztype in, stride_type, stride_type, stride_type,
			    rtype *out, stride_type, stride_type, stride_type,
			    length_type, length_type, length_type)
  {
    FFTW(execute_split_dft_c2r)(plan_,
			  in.first, in.second, out);
  }
};

// real -> complex FFTM

template <int A>
class fftm<SCALAR_TYPE, complex<SCALAR_TYPE>, A, fft_fwd>
  : public fftm_backend<SCALAR_TYPE, complex<SCALAR_TYPE>, A, fft_fwd>,
    planner<1, SCALAR_TYPE, complex<SCALAR_TYPE> >
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const axis = A == vsip::col ? 0 : 1;

public:
  fftm(Domain<2> const &dom, unsigned number)
    : planner<1, rtype, ctype>(dom[axis], 0, make_flags<rtype>(dom[axis], number, false),
			       dom[1 - axis].length())
  {
  }
  virtual void query_layout(Rt_layout<2> &in, Rt_layout<2> &out)
  {
    in.packing = this->aligned_ ? aligned : dense;
    in.alignment = OVXX_ALLOC_ALIGNMENT;
    if (axis == 0) in.order = tuple<1, 0, 2>();
    else in.order = tuple<0, 1, 2>();
    in.storage_format = array;
    out = in;
    out.storage_format = complex_storage_format;
  }
  virtual rtype *input_buffer() { return in_buffer_.get();}
  virtual ctype *output_buffer() { return out_buffer_.get();}
  virtual void out_of_place(rtype *in, stride_type i_str_0, stride_type i_str_1,
			    ctype *out, stride_type o_str_0, stride_type o_str_1,
			    length_type rows, length_type cols)
  {
    // To support distributed blocks we need to be ready to handle subsizes
    // of what we planned for. Thus, mult <= mult_.
    length_type const mult = axis == 1 ? rows : cols;
    length_type const in_stride = axis == 1 ? i_str_0 : i_str_1;
    length_type const out_stride = axis == 1 ? o_str_0 : o_str_1;

    // For the same reason, we can't have FFTW do the parallelization, since
    // that would require knowing the true (local) dimensions at planning time.
    for (index_type i = 0; i != mult; ++i, in += in_stride, out += out_stride)
      FFTW(execute_dft_r2c)(plan_, in, reinterpret_cast<FFTW(complex)*>(out));
  }
  virtual void out_of_place(rtype *in, stride_type i_str_0, stride_type i_str_1,
			    ztype out, stride_type o_str_0, stride_type o_str_1,
			    length_type rows, length_type cols)
  {
    // To support distributed blocks we need to be ready to handle subsizes
    // of what we planned for. Thus, mult <= mult_.
    length_type const mult = axis == 1 ? rows : cols;
    length_type const in_stride = axis == 1 ? i_str_0 : i_str_1;
    length_type const out_stride = axis == 1 ? o_str_0 : o_str_1;

    rtype* out_r = out.first;
    rtype* out_i = out.second;

    // For the same reason, we can't have FFTW do the parallelization, since
    // that would require knowing the true (local) dimensions at planning time.
    for (index_type i = 0; i != mult; ++i,
	   in += in_stride, out_r += out_stride, out_i += out_stride)
      FFTW(execute_split_dft_r2c)(plan_, in, out_r, out_i);
  }
};

// complex -> real FFTM

template <int A>
class fftm<complex<SCALAR_TYPE>, SCALAR_TYPE, A, fft_inv>
  : public fftm_backend<complex<SCALAR_TYPE>, SCALAR_TYPE, A, fft_inv>,
    planner<1, complex<SCALAR_TYPE>, SCALAR_TYPE>
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const axis = A == vsip::col ? 0 : 1;

public:
  fftm(Domain<2> const &dom, unsigned number)
    : planner<1, ctype, rtype>(dom[axis], 0, make_flags<rtype>(dom[axis], number), dom[1-axis].length())
  {
  }
  virtual void query_layout(Rt_layout<2> &in, Rt_layout<2> &out)
  {
    in.packing = this->aligned_ ? aligned : dense;
    in.alignment = OVXX_ALLOC_ALIGNMENT;
    if (axis == 0) in.order = tuple<1, 0, 2>();
    else  in.order = tuple<0, 1, 2>();
    in.storage_format = complex_storage_format;
    out = in;
    out.storage_format = array;
  }
  virtual ctype *input_buffer() { return in_buffer_.get();}
  virtual rtype *output_buffer() { return out_buffer_.get();}
  virtual void out_of_place(ctype *in, stride_type i_str_0, stride_type i_str_1,
			    rtype *out, stride_type o_str_0, stride_type o_str_1,
			    length_type rows, length_type cols)
  {
    length_type const mult = axis == 1 ? rows : cols;
    length_type const in_stride = axis == 1 ? i_str_0 : i_str_1;
    length_type const out_stride = axis == 1 ? o_str_0 : o_str_1;

    for (index_type i = 0; i != mult; ++i, in += in_stride, out += out_stride)
    {
      FFTW(execute_dft_c2r)(plan_, 
			    reinterpret_cast<FFTW(complex)*>(in), out);
    }
  }
  virtual void out_of_place(ztype in, stride_type i_str_0, stride_type i_str_1,
			    rtype *out, stride_type o_str_0, stride_type o_str_1,
			    length_type rows, length_type cols)
  {
    length_type const mult = axis == 1 ? rows : cols;
    length_type const in_stride = axis == 1 ? i_str_0 : i_str_1;
    length_type const out_stride = axis == 1 ? o_str_0 : o_str_1;

    rtype *in_r = in.first;
    rtype *in_i = in.second;
    for (index_type i = 0; i != mult;
	 ++i, in_r += in_stride, in_i += in_stride, out += out_stride)
      FFTW(execute_split_dft_c2r)(plan_, in_r, in_i, out);
  }
};

// complex -> complex FFTM

template <int A, int D>
class fftm<complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, A, D>
  : public fftm_backend<complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, A, D>,
    planner<1, complex<SCALAR_TYPE>, complex<SCALAR_TYPE> >
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const axis = A == vsip::col ? 0 : 1;

public:
  fftm(Domain<2> const &dom, int number)
    : planner<1, ctype, ctype>
      (dom[axis], exponent<ctype, ctype, D>::value, make_flags<ctype>(dom[axis], number), dom[1-axis].size())
  {}
  virtual void query_layout(Rt_layout<2> &inout)
  {
    inout.packing = this->aligned_ ? aligned : dense;
    inout.alignment = OVXX_ALLOC_ALIGNMENT;
    if (axis == 0) inout.order = tuple<1, 0, 2>();
    else inout.order = tuple<0, 1, 2>();
    inout.storage_format = complex_storage_format;
  }
  virtual void query_layout(Rt_layout<2> &in, Rt_layout<2> &out)
  {
    query_layout(in);
    out = in;
  }
  virtual ctype *input_buffer() { return in_buffer_.get();}
  virtual ctype *output_buffer() { return out_buffer_.get();}
  virtual void in_place(ctype *inout, stride_type str_0, stride_type str_1,
			length_type rows, length_type cols)
  {
    length_type const mult = axis == 1 ? rows : cols;
    stride_type const stride  = axis == 1 ? str_0 : str_1;

    for (index_type i = 0; i != mult; ++i, inout += stride)
      FFTW(execute_dft)(this->plan_ip_, 
 			reinterpret_cast<FFTW(complex)*>(inout),
 			reinterpret_cast<FFTW(complex)*>(inout));
  }
  virtual void in_place(ztype inout, stride_type str_0, stride_type str_1,
			length_type rows, length_type cols)
  {
    length_type const mult = axis == 1 ? rows : cols;
    stride_type const stride = axis == 1 ? str_0 : str_1;

    rtype* real_ptr = inout.first;
    rtype* imag_ptr = inout.second;

    for (index_type i = 0; i != mult; ++i, real_ptr += stride, imag_ptr += stride)
      if (D == fft_fwd)
	FFTW(execute_split_dft)(this->plan_ip_,
				real_ptr, imag_ptr,
				real_ptr, imag_ptr);
      else
	FFTW(execute_split_dft)(this->plan_ip_,
				imag_ptr, real_ptr,
				imag_ptr, real_ptr);
  }
  virtual void out_of_place(ctype *in, stride_type i_str_0, stride_type i_str_1,
			    ctype *out, stride_type o_str_0, stride_type o_str_1,
			    length_type rows, length_type cols)
  {
    length_type const mult = axis == 1 ? rows : cols;
    length_type const in_stride = axis == 1 ? i_str_0 : i_str_1;
    length_type const out_stride = axis == 1 ? o_str_0 : o_str_1;

    for (index_type i = 0; i != mult; ++i, in += in_stride, out += out_stride)
      FFTW(execute_dft)(plan_op_,
			reinterpret_cast<FFTW(complex)*>(in), 
			reinterpret_cast<FFTW(complex)*>(out));
  }
  virtual void out_of_place(ztype in, stride_type i_str_0, stride_type i_str_1,
			    ztype out, stride_type o_str_0, stride_type o_str_1,
			    length_type rows, length_type cols)
  {
    length_type const mult = axis == 1 ? rows : cols;
    length_type const in_stride  = axis == 1 ? i_str_0 : i_str_1;
    length_type const out_stride = axis == 1 ? o_str_0 : o_str_1;

    rtype *in_real  = in.first;
    rtype *in_imag  = in.second;
    rtype *out_real = out.first;
    rtype *out_imag = out.second;

    for (index_type i = 0; i != mult;
	 ++i,
	   in_real += in_stride, in_imag += in_stride,
	   out_real += out_stride, out_imag += out_stride)
      if (D == fft_fwd)
	FFTW(execute_split_dft)(plan_op_, 
				in_real, in_imag, out_real, out_imag);
      else
	FFTW(execute_split_dft)(plan_op_, 
				in_imag, in_real, out_imag, out_real);
  }
};

#define OVXX_FFTW_CREATE_FFT(D, I, O, S)	       \
template <>					       \
std::unique_ptr<fft_backend<D, I, O, S> >	               \
create(Domain<D> const &dom, unsigned number)	       \
{                                                      \
  return std::unique_ptr<fft_backend<D, I, O, S> >       \
    (new fft<D, I, O, S>(dom, number));		       \
}

OVXX_FFTW_CREATE_FFT(1, SCALAR_TYPE, complex<SCALAR_TYPE>, 0)
OVXX_FFTW_CREATE_FFT(1, complex<SCALAR_TYPE>, SCALAR_TYPE, 0)
OVXX_FFTW_CREATE_FFT(1, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, fft_fwd)
OVXX_FFTW_CREATE_FFT(1, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, fft_inv)
OVXX_FFTW_CREATE_FFT(2, SCALAR_TYPE, complex<SCALAR_TYPE>, 0)
OVXX_FFTW_CREATE_FFT(2, SCALAR_TYPE, complex<SCALAR_TYPE>, 1)
OVXX_FFTW_CREATE_FFT(2, complex<SCALAR_TYPE>, SCALAR_TYPE, 0)
OVXX_FFTW_CREATE_FFT(2, complex<SCALAR_TYPE>, SCALAR_TYPE, 1)
OVXX_FFTW_CREATE_FFT(2, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, fft_fwd)
OVXX_FFTW_CREATE_FFT(2, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, fft_inv)
OVXX_FFTW_CREATE_FFT(3, SCALAR_TYPE, complex<SCALAR_TYPE>, 0)
OVXX_FFTW_CREATE_FFT(3, SCALAR_TYPE, complex<SCALAR_TYPE>, 1)
OVXX_FFTW_CREATE_FFT(3, SCALAR_TYPE, complex<SCALAR_TYPE>, 2)
OVXX_FFTW_CREATE_FFT(3, complex<SCALAR_TYPE>, SCALAR_TYPE, 0)
OVXX_FFTW_CREATE_FFT(3, complex<SCALAR_TYPE>, SCALAR_TYPE, 1)
OVXX_FFTW_CREATE_FFT(3, complex<SCALAR_TYPE>, SCALAR_TYPE, 2)
OVXX_FFTW_CREATE_FFT(3, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, fft_fwd)
OVXX_FFTW_CREATE_FFT(3, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, fft_inv)

#undef OVXX_FFTW_CREATE_FFT

#define OVXX_FFTW_CREATE_FFTM(I, O, A, D)	       \
template <>                                            \
std::unique_ptr<fftm_backend<I, O, A, D> >	       \
create(Domain<2> const &dom, unsigned number)	       \
{                                                      \
  return std::unique_ptr<fftm_backend<I, O, A, D> >      \
    (new fftm<I, O, A, D>(dom, number));	       \
}

OVXX_FFTW_CREATE_FFTM(SCALAR_TYPE, complex<SCALAR_TYPE>, 0, fft_fwd)
OVXX_FFTW_CREATE_FFTM(SCALAR_TYPE, complex<SCALAR_TYPE>, 1, fft_fwd)
OVXX_FFTW_CREATE_FFTM(complex<SCALAR_TYPE>, SCALAR_TYPE, 0, fft_inv)
OVXX_FFTW_CREATE_FFTM(complex<SCALAR_TYPE>, SCALAR_TYPE, 1, fft_inv)
OVXX_FFTW_CREATE_FFTM(complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, 0, fft_fwd)
OVXX_FFTW_CREATE_FFTM(complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, 1, fft_fwd)
OVXX_FFTW_CREATE_FFTM(complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, 0, fft_inv)
OVXX_FFTW_CREATE_FFTM(complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, 1, fft_inv)

#undef OVXX_FFTW_CREATE_FFTM

} // namespace ovxx::fftw
} // namespace ovxx
