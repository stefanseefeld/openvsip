/* Copyright (c) 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/fftw3/fft_impl.cpp
    @author  Stefan Seefeld
    @date    2006-04-10
    @brief   VSIPL++ Library: FFT wrappers and traits to bridge with 
             FFTW3.
*/

#include <cassert>

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/fft/backend.hpp>
#include <vsip/core/fft/util.hpp>
#include <vsip/opt/fftw3/fft.hpp>
#include <vsip/core/equal.hpp>
#include <vsip/dense.hpp>
#include <fftw3.h>
#include <cstring> // for memcpy

namespace vsip
{
namespace impl
{
namespace fftw3
{

template <dimension_type D>
struct Fft_base<D, complex<SCALAR_TYPE>, complex<SCALAR_TYPE> >
{
  Fft_base(Domain<D> const& dom, int exp, int flags, bool aligned = false)
    VSIP_THROW((std::bad_alloc))
      : in_buffer_(dom.size()),
	out_buffer_(dom.size()),
        aligned_(aligned)
  {
    if (!aligned) flags |= FFTW_UNALIGNED;
    // For multi-dimensional transforms, these plans assume both
    // input and output data is dense, row-major, interleave-complex
    // format.
    
    for(index_type i=0;i<D;i++) size_[i] = dom[i].size();
    plan_in_place_ =
      Create_plan<fftw3_storage_format>
        ::create<FFTW(plan), FFTW(iodim)>
        (in_buffer_.ptr(), in_buffer_.ptr(), exp, flags, dom);
    
    if (!plan_in_place_) VSIP_IMPL_THROW(std::bad_alloc());

    plan_out_of_place_ = Create_plan<fftw3_storage_format>
      ::create<FFTW(plan), FFTW(iodim)>
      (in_buffer_.ptr(), out_buffer_.ptr(), exp, flags, dom);

    if (!plan_out_of_place_)
    {
      FFTW(destroy_plan)(plan_in_place_);
      VSIP_IMPL_THROW(std::bad_alloc());
    }
  }
  ~Fft_base() VSIP_NOTHROW
  {
    if (plan_in_place_) FFTW(destroy_plan)(plan_in_place_);
    if (plan_out_of_place_) FFTW(destroy_plan)(plan_out_of_place_);
  }

  Cmplx_buffer<fftw3_storage_format, SCALAR_TYPE> in_buffer_;
  Cmplx_buffer<fftw3_storage_format, SCALAR_TYPE> out_buffer_;
  FFTW(plan) plan_in_place_;
  FFTW(plan) plan_out_of_place_;
  int size_[D];
  bool aligned_;
};

template <vsip::dimension_type D>
struct Fft_base<D, SCALAR_TYPE, complex<SCALAR_TYPE> >
{
  Fft_base(Domain<D> const& dom, int A, int flags, bool aligned = false)
    VSIP_THROW((std::bad_alloc))
    : in_buffer_(32, dom.size()),
      out_buffer_(dom.size()),
      aligned_(aligned)
  { 
    if (!aligned) flags |= FFTW_UNALIGNED;
    for (vsip::dimension_type i = 0; i < D; ++i) size_[i] = dom[i].size();  
    // FFTW3 assumes A == D - 1.
    // See also query_layout().
    if (A != D - 1) std::swap(size_[A], size_[D - 1]);
    plan_out_of_place_ = Create_plan<fftw3_storage_format>::
      create<FFTW(plan), FFTW(iodim)>
      (in_buffer_.get(), out_buffer_.ptr(), A, flags, dom);
    if (!plan_out_of_place_) VSIP_IMPL_THROW(std::bad_alloc());
  }
  ~Fft_base() VSIP_NOTHROW
  {
    if (plan_out_of_place_) FFTW(destroy_plan)(plan_out_of_place_);
  }

  aligned_array<SCALAR_TYPE> in_buffer_;
  Cmplx_buffer<fftw3_storage_format, SCALAR_TYPE> out_buffer_;
  FFTW(plan) plan_out_of_place_;
  int size_[D];
  bool aligned_;
};

template <vsip::dimension_type D>
struct Fft_base<D, complex<SCALAR_TYPE>, SCALAR_TYPE>
{
  Fft_base(Domain<D> const& dom, int A, int flags, bool aligned = false)
    VSIP_THROW((std::bad_alloc))
    : in_buffer_(dom.size()),
      out_buffer_(32, dom.size()),
      aligned_(aligned)
  {
    if (!aligned) flags |= FFTW_UNALIGNED;
    for (vsip::dimension_type i = 0; i < D; ++i) size_[i] = dom[i].size();
    // FFTW3 assumes A == D - 1.
    // See also query_layout().
    if (A != D - 1) std::swap(size_[A], size_[D - 1]);
    plan_out_of_place_ = Create_plan<fftw3_storage_format>::
      create<FFTW(plan), FFTW(iodim)>
      (in_buffer_.ptr(), out_buffer_.get(), A, flags, dom);

    if (!plan_out_of_place_) VSIP_IMPL_THROW(std::bad_alloc());
  }
  ~Fft_base() VSIP_NOTHROW
  {
    if (plan_out_of_place_) FFTW(destroy_plan)(plan_out_of_place_);
  }

  Cmplx_buffer<fftw3_storage_format, SCALAR_TYPE> in_buffer_;
  aligned_array<SCALAR_TYPE>              out_buffer_;
  FFTW(plan) plan_out_of_place_;
  int size_[D];
  bool aligned_;
};

// 1D complex -> complex FFT

template <int S>
class Fft_impl<1, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, S>
  : private Fft_base<1, complex<SCALAR_TYPE>, complex<SCALAR_TYPE> >,
    public fft::Fft_backend<1, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>,
			    S>

{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const E = S == fft_fwd ? -1 : 1;

public:
  Fft_impl(Domain<1> const &dom, unsigned number)
    : Fft_base<1, ctype, ctype>(dom, E, convert_NoT(number),
                                // Only require aligned arrays if FFTW3 can actually take 
                                // advantage from it by using SIMD kernels.
                                !(VSIP_IMPL_ALLOC_ALIGNMENT % sizeof(ctype)) &&
                                !((sizeof(ctype) * dom.length()) % VSIP_IMPL_ALLOC_ALIGNMENT))
  {}
  virtual char const* name() { return "fft-fftw3-1D-complex"; }
  virtual void query_layout(Rt_layout<1> &rtl_inout)
  {
    rtl_inout.packing = this->aligned_ ? aligned : dense;
    rtl_inout.alignment = VSIP_IMPL_ALLOC_ALIGNMENT;
    rtl_inout.order = tuple<0, 1, 2>();
    // make default based on library
    rtl_inout.storage_format = Create_plan<fftw3_storage_format>::storage_format;
  }
  virtual void query_layout(Rt_layout<1> &rtl_in, Rt_layout<1> &rtl_out)
  {
    rtl_in.packing = this->aligned_ ? aligned : dense;
    rtl_in.alignment = VSIP_IMPL_ALLOC_ALIGNMENT;
    rtl_in.order = tuple<0, 1, 2>();
    rtl_in.storage_format = Create_plan<fftw3_storage_format>::storage_format;
    rtl_out = rtl_in;
  }
  virtual void in_place(ctype *inout, stride_type s, length_type l)
  {
    assert(s == 1 && static_cast<int>(l) == this->size_[0]);
    FFTW(execute_dft)(plan_in_place_,
		      reinterpret_cast<FFTW(complex)*>(inout),
		      reinterpret_cast<FFTW(complex)*>(inout));
  }
  virtual void in_place(ztype inout, stride_type s, length_type l)
  {
    assert(s == 1 && static_cast<int>(l) == this->size_[0]);

#if USE_BROKEN_FFTW_SPLIT
    if (E == -1)
      FFTW(execute_split_dft)(plan_in_place_,
			      inout.first, inout.second,
			      inout.first, inout.second);
    else
      FFTW(execute_split_dft)(plan_in_place_,
			      inout.second, inout.first,
			      inout.second, inout.first);
#else
    typedef Storage<fftw3_storage_format, ctype> storage_type;
    rtype* real = storage_type::get_real_ptr(in_buffer_.ptr());
    rtype* imag = storage_type::get_imag_ptr(in_buffer_.ptr());
    memcpy(real, inout.first, l*sizeof(rtype));
    memcpy(imag, inout.second, l*sizeof(rtype));
    if (E == -1)
      FFTW(execute_split_dft)(plan_out_of_place_,
			      real, imag,
			      inout.first, inout.second);
    else
      FFTW(execute_split_dft)(plan_out_of_place_,
			      imag, real,
			      inout.second, inout.first);
#endif
  }
  virtual void out_of_place(ctype *in, stride_type in_stride,
			    ctype *out, stride_type out_stride,
			    length_type length)
  {
    assert(in_stride == 1 && out_stride == 1 &&
	   static_cast<int>(length) == this->size_[0]);
    FFTW(execute_dft)(plan_out_of_place_,
		      reinterpret_cast<FFTW(complex)*>(in),
		      reinterpret_cast<FFTW(complex)*>(out));
  }
  virtual void out_of_place(ztype in, stride_type in_stride,
			    ztype out, stride_type out_stride,
			    length_type length)
  {
    assert(in_stride == 1 && out_stride == 1 &&
	   static_cast<int>(length) == this->size_[0]);

    if (E == -1)
      FFTW(execute_split_dft)(plan_out_of_place_,
			      in.first,  in.second,
			      out.first, out.second);
    else
      FFTW(execute_split_dft)(plan_out_of_place_,
			      in.second,  in.first,
			      out.second, out.first);
  }
};

// 1D real -> complex FFT

template <>
class Fft_impl<1, SCALAR_TYPE, complex<SCALAR_TYPE>, 0>
  : private Fft_base<1, SCALAR_TYPE, complex<SCALAR_TYPE> >,
    public fft::Fft_backend<1, SCALAR_TYPE, complex<SCALAR_TYPE>, 0>
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<1> const &dom, unsigned number)
    : Fft_base<1, rtype, ctype>(dom, 0, convert_NoT(number),
                                // Only require aligned arrays if FFTW3 can actually take 
                                // advantage from it by using SIMD kernels.
                                !(VSIP_IMPL_ALLOC_ALIGNMENT % sizeof(rtype)) &&
                                !((sizeof(rtype) * dom.length()) % VSIP_IMPL_ALLOC_ALIGNMENT))
  {}
  virtual char const* name() { return "fft-fftw3-1D-real-forward"; }
  virtual void query_layout(Rt_layout<1> &rtl_in, Rt_layout<1> &rtl_out)
  {
    rtl_in.packing = this->aligned_ ? aligned : dense;
    rtl_in.alignment = VSIP_IMPL_ALLOC_ALIGNMENT;
    rtl_in.order = tuple<0, 1, 2>();
    rtl_in.storage_format = Create_plan<fftw3_storage_format>::storage_format;
    rtl_out = rtl_in;
  }
  virtual void out_of_place(rtype *in, stride_type,
			    ctype *out, stride_type,
			    length_type)
  {
    FFTW(execute_dft_r2c)(plan_out_of_place_, 
			  in, reinterpret_cast<FFTW(complex)*>(out));
  }
  virtual void out_of_place(rtype *in, stride_type is,
			    ztype out, stride_type os,
			    length_type length)
  {
    assert(is == 1);
    assert(os == 1);
#if USE_BROKEN_FFTW_SPLIT
    FFTW(execute_split_dft_r2c)(plan_out_of_place_, 
			  in, out.first, out.second);
#else
    typedef Storage<fftw3_storage_format, ctype> storage_type;
    rtype* out_r = storage_type::get_real_ptr(out_buffer_.ptr());
    rtype* out_i = storage_type::get_imag_ptr(out_buffer_.ptr());
    FFTW(execute_split_dft_r2c)(plan_out_of_place_, 
				in, out_r, out_i);
    memcpy(out.first,  out_r, (length/2+1)*sizeof(rtype));
    memcpy(out.second, out_i, (length/2+1)*sizeof(rtype));
#endif
  }
};

// 1D complex -> real FFT

template <>
class Fft_impl<1, complex<SCALAR_TYPE>, SCALAR_TYPE, 0>
  : Fft_base<1, complex<SCALAR_TYPE>, SCALAR_TYPE>,
    public fft::Fft_backend<1, complex<SCALAR_TYPE>, SCALAR_TYPE, 0>
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<1> const &dom, unsigned number)
    : Fft_base<1, ctype, rtype>(dom, 0, convert_NoT(number),
                                // Only require aligned arrays if FFTW3 can actually take 
                                // advantage from it by using SIMD kernels.
                                !(VSIP_IMPL_ALLOC_ALIGNMENT % sizeof(rtype)) &&
                                !((sizeof(rtype) * dom.length()) % VSIP_IMPL_ALLOC_ALIGNMENT))
  {}

  virtual char const* name() { return "fft-fftw3-1D-real-inverse"; }
  virtual void query_layout(Rt_layout<1> &rtl_in, Rt_layout<1> &rtl_out)
  {
    rtl_in.packing = this->aligned_ ? aligned : dense;
    rtl_in.alignment = VSIP_IMPL_ALLOC_ALIGNMENT;
    rtl_in.order = tuple<0, 1, 2>();
    rtl_in.storage_format = Create_plan<fftw3_storage_format>::storage_format;
    rtl_out = rtl_in;
  }

  virtual bool requires_copy(Rt_layout<1> &) { return true;}

  virtual void out_of_place(ctype *in, stride_type,
			    rtype *out, stride_type,
			    length_type)
  {
    FFTW(execute_dft_c2r)(plan_out_of_place_,
			  reinterpret_cast<FFTW(complex)*>(in), out);
  }
  virtual void out_of_place(ztype in, stride_type is,
			    rtype *out, stride_type os,
			    length_type length)
  {
    assert(is == 1);
    assert(os == 1);
#if USE_BROKEN_FFTW_SPLIT
    FFTW(execute_split_dft_c2r)(plan_out_of_place_,
			  in.first, in.second, out);
#else
    typedef Storage<fftw3_storage_format, ctype> storage_type;
    rtype* in_r = storage_type::get_real_ptr(in_buffer_.ptr());
    rtype* in_i = storage_type::get_imag_ptr(in_buffer_.ptr());
    memcpy(in_r, in.first, (length/2+1)*sizeof(rtype));
    memcpy(in_i, in.second, (length/2+1)*sizeof(rtype));
    FFTW(execute_split_dft_c2r)(plan_out_of_place_,
			  in_r, in_i, out);
#endif
  }
};

// 2D complex -> complex FFT

template <int S>
class Fft_impl<2, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, S>
  : private Fft_base<2, complex<SCALAR_TYPE>, complex<SCALAR_TYPE> >,
    public fft::Fft_backend<2, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>,
			    S>
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const E = S == fft_fwd ? -1 : 1;

public:
  Fft_impl(Domain<2> const &dom, unsigned number)
    : Fft_base<2, ctype, ctype>(dom, E, convert_NoT(number))
  {}
  virtual char const* name() { return "fft-fftw3-2D-complex"; }
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    rtl_in.packing = dense;
    rtl_in.order = row2_type();
    rtl_in.storage_format = Create_plan<fftw3_storage_format>::storage_format;
    rtl_out = rtl_in;
  }
  virtual void in_place(ctype *inout,
			stride_type r_stride,
			stride_type c_stride,
			length_type /*rows*/, length_type cols)
  {
    // Check that data is dense row-major.
    assert(r_stride == static_cast<stride_type>(cols));
    assert(c_stride == 1);

    FFTW(execute_dft)(plan_in_place_,
		      reinterpret_cast<FFTW(complex)*>(inout),
		      reinterpret_cast<FFTW(complex)*>(inout));
  }
  /// complex (split) in-place
  virtual void in_place(ztype inout,
			stride_type, stride_type,
			length_type, length_type)
  {
    FFTW(execute_split_dft)(plan_in_place_,
		      inout.first, inout.second,
		      inout.first, inout.second);
  }
  virtual void out_of_place(ctype *in,
			    stride_type in_r_stride,
			    stride_type in_c_stride,
			    ctype *out,
			    stride_type out_r_stride,
			    stride_type out_c_stride,
			    length_type /*rows*/, length_type cols)
  {
    // Check that data is dense row-major.
    assert(in_r_stride == static_cast<stride_type>(cols));
    assert(in_c_stride == 1);
    assert(out_r_stride == static_cast<stride_type>(cols));
    assert(out_c_stride == 1);

    FFTW(execute_dft)(plan_out_of_place_,
		      reinterpret_cast<FFTW(complex)*>(in), 
		      reinterpret_cast<FFTW(complex)*>(out));
  }
  virtual void out_of_place(ztype in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    ztype out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type, length_type cols)
  {
    // Check that data is dense row-major.
    assert(in_r_stride == static_cast<stride_type>(cols));
    assert(in_c_stride == 1);
    assert(out_r_stride == static_cast<stride_type>(cols));
    assert(out_c_stride == 1);

    FFTW(execute_split_dft)(plan_out_of_place_,
                            in.first, in.second,
                            out.first, out.second);
  }
};

// 2D real -> complex FFT

template <int A>
class Fft_impl<2, SCALAR_TYPE, complex<SCALAR_TYPE>, A>
  : private Fft_base<2, SCALAR_TYPE, complex<SCALAR_TYPE> >,
    public fft::Fft_backend<2, SCALAR_TYPE, complex<SCALAR_TYPE>, A>
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<2> const &dom, unsigned number)
    : Fft_base<2, rtype, ctype>(dom, A, convert_NoT(number))
  {}

  virtual char const* name() { return "fft-fftw3-2D-real-forward"; }

  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    rtl_in.packing = dense;
    // FFTW3 assumes A is the last dimension.
    if (A == 0) rtl_in.order = tuple<1, 0, 2>();
    else rtl_in.order = tuple<0, 1, 2>();
    rtl_in.storage_format = Create_plan<fftw3_storage_format>::storage_format;
    rtl_out = rtl_in;
  }
  virtual bool requires_copy(Rt_layout<2> &) { return true;}

  virtual void out_of_place(rtype *in,
			    stride_type, stride_type,
			    ctype *out,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    FFTW(execute_dft_r2c)(plan_out_of_place_,
			  in, reinterpret_cast<FFTW(complex)*>(out));
  }
  virtual void out_of_place(rtype *in,
			    stride_type, stride_type,
			    ztype out,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    FFTW(execute_split_dft_r2c)(plan_out_of_place_,
				in, out.first, out.second);
  }

};

// 2D complex -> real FFT

template <int A>
class Fft_impl<2, complex<SCALAR_TYPE>, SCALAR_TYPE, A>
  : Fft_base<2, complex<SCALAR_TYPE>, SCALAR_TYPE>,
    public fft::Fft_backend<2, complex<SCALAR_TYPE>, SCALAR_TYPE, A>
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<2> const &dom, unsigned number)
    : Fft_base<2, ctype, rtype>(dom, A, convert_NoT(number))
  {}

  virtual char const* name() { return "fft-fftw3-2D-real-inverse"; }

  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    rtl_in.packing = dense;
    // FFTW3 assumes A is the last dimension.
    if (A == 0) rtl_in.order = tuple<1, 0, 2>();
    else rtl_in.order = tuple<0, 1, 2>();
    rtl_in.storage_format = Create_plan<fftw3_storage_format>::storage_format;
    rtl_out = rtl_in;
  }
  virtual bool requires_copy(Rt_layout<2> &) { return true;}

  virtual void out_of_place(ctype *in,
			    stride_type, stride_type,
			    rtype *out,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    FFTW(execute_dft_c2r)(plan_out_of_place_, 
			  reinterpret_cast<FFTW(complex)*>(in), out);
  }
  virtual void out_of_place(ztype in,
			    stride_type, stride_type,
			    rtype *out,
			    stride_type, stride_type,
			    length_type, length_type)
  {
    FFTW(execute_split_dft_c2r)(plan_out_of_place_, 
			  in.first, in.second, out);
  }

};

// 3D complex -> complex FFT

template <int S>
class Fft_impl<3, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, S>
  : private Fft_base<3, complex<SCALAR_TYPE>, complex<SCALAR_TYPE> >,
    public fft::Fft_backend<3, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>,
			    S>

{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const E = S == fft_fwd ? -1 : 1;

public:
  Fft_impl(Domain<3> const &dom, unsigned number)
    : Fft_base<3, ctype, ctype>(dom, E, convert_NoT(number))
  {}
  virtual char const* name() { return "fft-fftw3-3D-complex"; }
  virtual void query_layout(Rt_layout<3> &rtl_in, Rt_layout<3> &rtl_out)
  {
    rtl_in.packing = dense;
    rtl_in.order = row3_type();
    rtl_in.storage_format = Create_plan<fftw3_storage_format>::storage_format;
    rtl_out = rtl_in;
  }
  virtual void in_place(ctype *inout,
			stride_type x_stride,
			stride_type y_stride,
			stride_type z_stride,
			length_type x_length,
			length_type y_length,
			length_type z_length)
  {
    assert(static_cast<int>(x_length) == this->size_[0]);
    assert(static_cast<int>(y_length) == this->size_[1]);
    assert(static_cast<int>(z_length) == this->size_[2]);

    // Check that data is dense row-major.
    assert(x_stride == static_cast<stride_type>(y_length*z_length));
    assert(y_stride == static_cast<stride_type>(z_length));
    assert(z_stride == 1);

    FFTW(execute_dft)(plan_in_place_,
		      reinterpret_cast<FFTW(complex)*>(inout),
		      reinterpret_cast<FFTW(complex)*>(inout));
  }
  virtual void in_place(ztype inout,
			stride_type x_stride,
			stride_type y_stride,
			stride_type z_stride,
			length_type x_length,
			length_type y_length,
			length_type z_length)
  {
    assert(static_cast<int>(x_length) == this->size_[0]);
    assert(static_cast<int>(y_length) == this->size_[1]);
    assert(static_cast<int>(z_length) == this->size_[2]);

    // Check that data is dense row-major.
    assert(x_stride == static_cast<stride_type>(y_length*z_length));
    assert(y_stride == static_cast<stride_type>(z_length));
    assert(z_stride == 1);

    FFTW(execute_split_dft)(plan_in_place_,
		      inout.first, inout.second,
		      inout.first, inout.second);
  }
  virtual void out_of_place(ctype *in,
			    stride_type in_x_stride,
			    stride_type in_y_stride,
			    stride_type in_z_stride,
			    ctype *out,
			    stride_type out_x_stride,
			    stride_type out_y_stride,
			    stride_type out_z_stride,
			    length_type x_length,
			    length_type y_length,
			    length_type z_length)
  {
    assert(static_cast<int>(x_length) == this->size_[0]);
    assert(static_cast<int>(y_length) == this->size_[1]);
    assert(static_cast<int>(z_length) == this->size_[2]);

    // Check that data is dense row-major.
    assert(in_x_stride == static_cast<stride_type>(y_length*z_length));
    assert(in_y_stride == static_cast<stride_type>(z_length));
    assert(in_z_stride == 1);
    assert(out_x_stride == static_cast<stride_type>(y_length*z_length));
    assert(out_y_stride == static_cast<stride_type>(z_length));
    assert(out_z_stride == 1);

    FFTW(execute_dft)(plan_out_of_place_,
		      reinterpret_cast<FFTW(complex)*>(in), 
		      reinterpret_cast<FFTW(complex)*>(out));
  }
  virtual void out_of_place(ztype in,
			    stride_type in_x_stride,
			    stride_type in_y_stride,
			    stride_type in_z_stride,
			    ztype out,
			    stride_type out_x_stride,
			    stride_type out_y_stride,
			    stride_type out_z_stride,
			    length_type x_length,
			    length_type y_length,
			    length_type z_length)
  {
    assert(static_cast<int>(x_length) == this->size_[0]);
    assert(static_cast<int>(y_length) == this->size_[1]);
    assert(static_cast<int>(z_length) == this->size_[2]);

    // Check that data is dense row-major.
    assert(in_x_stride == static_cast<stride_type>(y_length*z_length));
    assert(in_y_stride == static_cast<stride_type>(z_length));
    assert(in_z_stride == 1);
    assert(out_x_stride == static_cast<stride_type>(y_length*z_length));
    assert(out_y_stride == static_cast<stride_type>(z_length));
    assert(out_z_stride == 1);

    FFTW(execute_split_dft)(plan_out_of_place_,
                      in.first, in.second,
                      out.first, out.second);
  }
};

// 3D real -> complex FFT

template <int A>
class Fft_impl<3, SCALAR_TYPE, complex<SCALAR_TYPE>, A>
  : private Fft_base<3, SCALAR_TYPE, complex<SCALAR_TYPE> >,
    public fft::Fft_backend<3, SCALAR_TYPE, complex<SCALAR_TYPE>, A>
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<3> const &dom, unsigned number)
    : Fft_base<3, rtype, ctype>(dom, A, convert_NoT(number))
  {}

  virtual char const* name() { return "fft-fftw3-3D-real-forward"; }

  virtual void query_layout(Rt_layout<3> &rtl_in, Rt_layout<3> &rtl_out)
  {
    rtl_in.packing = dense;
    // FFTW3 assumes A is the last dimension.
    switch (A)
    {
      case 0: rtl_in.order = tuple<2, 1, 0>(); break;
      case 1: rtl_in.order = tuple<0, 2, 1>(); break;
      default: rtl_in.order = tuple<0, 1, 2>(); break;
    }
    rtl_in.storage_format = Create_plan<fftw3_storage_format>::storage_format;
    rtl_out = rtl_in;
  }
  virtual bool requires_copy(Rt_layout<3> &) { return true;}

  virtual void out_of_place(rtype *in,
			    stride_type,
			    stride_type,
			    stride_type,
			    ctype *out,
			    stride_type,
			    stride_type,
			    stride_type,
			    length_type,
			    length_type,
			    length_type)
  {
    FFTW(execute_dft_r2c)(plan_out_of_place_,
			  in, reinterpret_cast<FFTW(complex)*>(out));
  }
  virtual void out_of_place(rtype *in,
			    stride_type,
			    stride_type,
			    stride_type,
			    ztype out,
			    stride_type,
			    stride_type,
			    stride_type,
			    length_type,
			    length_type,
			    length_type)
  {
    FFTW(execute_split_dft_r2c)(plan_out_of_place_,
			  in, out.first, out.second);
  }

};

// 3D complex -> real FFT

template <int A>
class Fft_impl<3, complex<SCALAR_TYPE>, SCALAR_TYPE, A>
  : Fft_base<3, complex<SCALAR_TYPE>, SCALAR_TYPE>,
    public fft::Fft_backend<3, complex<SCALAR_TYPE>, SCALAR_TYPE, A>
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<3> const &dom, unsigned number)
    : Fft_base<3, ctype, rtype>(dom, A, convert_NoT(number))
  {}

  virtual char const* name() { return "fft-fftw3-3D-real-inverse"; }
  
  virtual void query_layout(Rt_layout<3> &rtl_in, Rt_layout<3> &rtl_out)
  {
    rtl_in.packing = dense;
    // FFTW3 assumes A is the last dimension.
    switch (A)
    {
      case 0: rtl_in.order = tuple<2, 1, 0>(); break;
      case 1: rtl_in.order = tuple<0, 2, 1>(); break;
      default: rtl_in.order = tuple<0, 1, 2>(); break;
    }
    rtl_in.storage_format = Create_plan<fftw3_storage_format>::storage_format;
    rtl_out = rtl_in;
  }
  virtual bool requires_copy(Rt_layout<3> &) { return true;}

  virtual void out_of_place(ctype *in,
			    stride_type,
			    stride_type,
			    stride_type,
			    rtype *out,
			    stride_type,
			    stride_type,
			    stride_type,
			    length_type,
			    length_type,
			    length_type)
  {
    FFTW(execute_dft_c2r)(plan_out_of_place_,
			  reinterpret_cast<FFTW(complex)*>(in), out);
  }
  virtual void out_of_place(ztype in,
			    stride_type,
			    stride_type,
			    stride_type,
			    rtype *out,
			    stride_type,
			    stride_type,
			    stride_type,
			    length_type,
			    length_type,
			    length_type)
  {
    FFTW(execute_split_dft_c2r)(plan_out_of_place_,
			  in.first, in.second, out);
  }

};

// real -> complex FFTM

template <int A>
class Fftm_impl<SCALAR_TYPE, complex<SCALAR_TYPE>, A, fft_fwd>
  : private Fft_base<1, SCALAR_TYPE, complex<SCALAR_TYPE> >,
    public fft::Fftm_backend<SCALAR_TYPE, complex<SCALAR_TYPE>, A, fft_fwd>
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const axis = A == vsip::col ? 0 : 1;

public:
  Fftm_impl(Domain<2> const &dom, unsigned number)
    : Fft_base<1, SCALAR_TYPE, complex<SCALAR_TYPE> >
      (dom[axis], 0, convert_NoT(number),
       // Only require aligned arrays if FFTW3 can actually take 
       // advantage from it by using SIMD kernels.
       !(VSIP_IMPL_ALLOC_ALIGNMENT % sizeof(rtype)) &&
       !((sizeof(rtype) * dom[axis].length()) % VSIP_IMPL_ALLOC_ALIGNMENT)),
      mult_(dom[1 - axis].size())
  {}
  virtual char const* name() { return "fftm-fftw3-real-forward"; }
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    rtl_in.packing = this->aligned_ ? aligned : dense;
    rtl_in.alignment = VSIP_IMPL_ALLOC_ALIGNMENT;
    if (axis == 0) rtl_in.order = tuple<1, 0, 2>();
    else  rtl_in.order = tuple<0, 1, 2>();
    // make default based on library
    rtl_in.storage_format = Create_plan<fftw3_storage_format>::storage_format;
    rtl_out = rtl_in;
  }
  virtual void out_of_place(rtype *in,
			    stride_type i_str_0, stride_type i_str_1,
			    ctype *out,
			    stride_type o_str_0, stride_type o_str_1,
			    length_type rows, length_type cols)
  {
    length_type const n_fft          = (axis == 1) ? rows : cols;
    length_type const in_fft_stride  = (axis == 1) ? i_str_0 : i_str_1;
    length_type const out_fft_stride = (axis == 1) ? o_str_0 : o_str_1;

    if (axis == 1) assert(rows <= mult_ && static_cast<int>(cols) == size_[0]);
    else           assert(cols <= mult_ && static_cast<int>(rows) == size_[0]);

    for (index_type i = 0; i < n_fft; ++i)
    {
      FFTW(execute_dft_r2c)(plan_out_of_place_, 
			    in, reinterpret_cast<FFTW(complex)*>(out));
      in  += in_fft_stride;
      out += out_fft_stride;
    }
  }
  virtual void out_of_place(rtype*      in,
			    stride_type i_str_0,
			    stride_type i_str_1,
			    ztype       out,
			    stride_type o_str_0,
			    stride_type o_str_1,
			    length_type rows,
			    length_type cols)
  {
    length_type const n_fft          = (axis == 1) ? rows : cols;
    length_type const in_fft_stride  = (axis == 1) ? i_str_0 : i_str_1;
    length_type const out_fft_stride = (axis == 1) ? o_str_0 : o_str_1;

    if (axis == 1) assert(rows <= mult_ && static_cast<int>(cols) == size_[0]);
    else           assert(cols <= mult_ && static_cast<int>(rows) == size_[0]);

    rtype* out_r = out.first;
    rtype* out_i = out.second;

#if !USE_BROKEN_FFTW_SPLIT
    typedef Storage<fftw3_storage_format, ctype> storage_type;
    rtype* tmp_out_r = storage_type::get_real_ptr(out_buffer_.ptr());
    rtype* tmp_out_i = storage_type::get_imag_ptr(out_buffer_.ptr());
#endif

    for (index_type i = 0; i < n_fft; ++i)
    {
#if USE_BROKEN_FFTW_SPLIT
      FFTW(execute_split_dft_r2c)(plan_out_of_place_,
				  in, out_r, out_i);
#else
      FFTW(execute_split_dft_r2c)(plan_out_of_place_,
				  in, tmp_out_r, tmp_out_i);
      memcpy(out_r, tmp_out_r, (size_[0]/2+1)*sizeof(rtype));
      memcpy(out_i, tmp_out_i, (size_[0]/2+1)*sizeof(rtype));
#endif
      in    += in_fft_stride;
      out_r += out_fft_stride;
      out_i += out_fft_stride;
    }
  }

private:
  length_type mult_;
};

// complex -> real FFTM

template <int A>
class Fftm_impl<complex<SCALAR_TYPE>, SCALAR_TYPE, A, fft_inv>
  : private Fft_base<1, complex<SCALAR_TYPE>, SCALAR_TYPE>,
    public fft::Fftm_backend<complex<SCALAR_TYPE>, SCALAR_TYPE, A, fft_inv>
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const axis = A == vsip::col ? 0 : 1;

public:
  Fftm_impl(Domain<2> const &dom, unsigned number)
    : Fft_base<1, complex<SCALAR_TYPE>, SCALAR_TYPE>
      (dom[axis], 0, convert_NoT(number),
       // Only require aligned arrays if FFTW3 can actually take 
       // advantage from it by using SIMD kernels.
       !(VSIP_IMPL_ALLOC_ALIGNMENT % sizeof(rtype)) &&
       !((sizeof(rtype) * dom[axis].length()) % VSIP_IMPL_ALLOC_ALIGNMENT)),
      mult_(dom[1-axis].size())
  {
  }

  virtual char const* name() { return "fftm-fftw3-real-inverse"; }

  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    rtl_in.packing = this->aligned_ ? aligned : dense;
    rtl_in.alignment = VSIP_IMPL_ALLOC_ALIGNMENT;
    if (axis == 0) rtl_in.order = tuple<1, 0, 2>();
    else  rtl_in.order = tuple<0, 1, 2>();
    // make default based on library
    rtl_in.storage_format = Create_plan<fftw3_storage_format>::storage_format;
    rtl_out = rtl_in;
  }
  virtual bool requires_copy(Rt_layout<2> &) { return true;}

  virtual void out_of_place(ctype *in,
			    stride_type i_str_0, stride_type i_str_1,
			    rtype *out,
			    stride_type o_str_0, stride_type o_str_1,
			    length_type rows, length_type cols)
  {
    length_type const n_fft          = (axis == 1) ? rows : cols;
    length_type const in_fft_stride  = (axis == 1) ? i_str_0 : i_str_1;
    length_type const out_fft_stride = (axis == 1) ? o_str_0 : o_str_1;

    if (axis == 1) assert(rows <= mult_ && static_cast<int>(cols) == size_[0]);
    else           assert(cols <= mult_ && static_cast<int>(rows) == size_[0]);

    for (index_type i = 0; i < n_fft; ++i)
    {
      FFTW(execute_dft_c2r)(plan_out_of_place_, 
			    reinterpret_cast<FFTW(complex)*>(in), out);
      in  += in_fft_stride;
      out += out_fft_stride;
    }
  }
  virtual void out_of_place(ztype       in,
			    stride_type i_str_0,
			    stride_type i_str_1,
			    rtype*      out,
			    stride_type o_str_0,
			    stride_type o_str_1,
			    length_type rows,
			    length_type cols)
  {
    length_type const n_fft          = (axis == 1) ? rows : cols;
    length_type const in_fft_stride  = (axis == 1) ? i_str_0 : i_str_1;
    length_type const out_fft_stride = (axis == 1) ? o_str_0 : o_str_1;

    if (axis == 1) assert(rows <= mult_ && static_cast<int>(cols) == size_[0]);
    else           assert(cols <= mult_ && static_cast<int>(rows) == size_[0]);

    rtype* in_r = in.first;
    rtype* in_i = in.second;

#if !USE_BROKEN_FFTW_SPLIT
    typedef Storage<fftw3_storage_format, ctype> storage_type;
    rtype* tmp_in_r = storage_type::get_real_ptr(in_buffer_.ptr());
    rtype* tmp_in_i = storage_type::get_imag_ptr(in_buffer_.ptr());
#endif

    for (index_type i = 0; i < n_fft; ++i)
    {
#if USE_BROKEN_FFTW_SPLIT
      FFTW(execute_split_dft_c2r)(plan_out_of_place_,
				  in_r, in_i, out);
#else
      memcpy(tmp_in_r, in_r, (size_[0]/2+1)*sizeof(rtype));
      memcpy(tmp_in_i, in_i, (size_[0]/2+1)*sizeof(rtype));
      FFTW(execute_split_dft_c2r)(plan_out_of_place_,
				  tmp_in_r, tmp_in_i, out);
#endif
      in_r += in_fft_stride;
      in_i += in_fft_stride;
      out  += out_fft_stride;
    }
  }

private:
  length_type mult_;
};

// complex -> complex FFTM

template <int A, int D>
class Fftm_impl<complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, A, D>
  : private Fft_base<1, complex<SCALAR_TYPE>, complex<SCALAR_TYPE> >,
    public fft::Fftm_backend<complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, A, D>
{
  typedef SCALAR_TYPE rtype;
  typedef complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const axis = A == vsip::col ? 0 : 1;
  static int const E = D == fft_fwd ? -1 : 1;

public:
  Fftm_impl(Domain<2> const &dom, int number)
    : Fft_base<1, ctype, ctype>
      (dom[axis], E, convert_NoT(number),
       // Only require aligned arrays if FFTW3 can actually take 
       // advantage from it by using SIMD kernels.
       !(VSIP_IMPL_ALLOC_ALIGNMENT % sizeof(ctype)) &&
       !((sizeof(ctype) * dom[axis].length()) % VSIP_IMPL_ALLOC_ALIGNMENT)),
      mult_(dom[1-axis].size())
  {
  }

  virtual char const* name() { return "fftm-fftw3-complex"; }

  virtual void query_layout(Rt_layout<2> &rtl_inout)
  {
    // By default use unit_stride,
    rtl_inout.packing = this->aligned_ ? aligned : dense;
    // an order that gives unit strides on the axis perpendicular to A,
    if (axis == 0) rtl_inout.order = tuple<1, 0, 2>();
    else rtl_inout.order = tuple<0, 1, 2>();
    // make default based on library
    rtl_inout.storage_format = Create_plan<fftw3_storage_format>::storage_format;
    rtl_inout.alignment = VSIP_IMPL_ALLOC_ALIGNMENT;
  }

  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    rtl_in.packing = this->aligned_ ? aligned : dense;
    rtl_in.alignment = VSIP_IMPL_ALLOC_ALIGNMENT;
    if (axis == 0) rtl_in.order = tuple<1, 0, 2>();
    else  rtl_in.order = tuple<0, 1, 2>();
    // make default based on library
    rtl_in.storage_format = Create_plan<fftw3_storage_format>::storage_format;
    rtl_out = rtl_in;
  }

  virtual void in_place(ctype *inout,
			stride_type str_0, stride_type str_1,
			length_type rows, length_type cols)
  {
    assert(fftw3_storage_format == interleaved_complex);

    length_type const n_fft       = (axis == 1) ? rows : cols;
    stride_type const fft_stride  = (axis == 1) ? str_0 : str_1;

    if (axis == 1) assert(rows <= mult_ && static_cast<int>(cols) == size_[0]);
    else           assert(cols <= mult_ && static_cast<int>(rows) == size_[0]);
    assert(((axis == 1) ? str_1 : str_0) == 1);

    for (index_type i = 0; i != n_fft; ++i)
    {
      FFTW(execute_dft)(this->plan_in_place_, 
 			reinterpret_cast<FFTW(complex)*>(inout),
 			reinterpret_cast<FFTW(complex)*>(inout));
      inout += fft_stride;
    }
  }

  virtual void in_place(ztype       inout,
			stride_type str_0,
			stride_type str_1,
			length_type rows,
			length_type cols)
  {
    assert(fftw3_storage_format == split_complex);

    length_type const n_fft       = (axis == 1) ? rows : cols;
    stride_type const fft_stride  = (axis == 1) ? str_0 : str_1;

    if (axis == 1) assert(rows <= mult_ && static_cast<int>(cols) == size_[0]);
    else           assert(cols <= mult_ && static_cast<int>(rows) == size_[0]);
    assert(((axis == 1) ? str_1 : str_0) == 1);

    rtype* real_ptr = inout.first;
    rtype* imag_ptr = inout.second;

    for (index_type i = 0; i != n_fft; ++i)
    {
#if USE_BROKEN_FFTW_SPLIT
      if (E == -1)
	FFTW(execute_split_dft)(this->plan_in_place_,
				real_ptr, imag_ptr,
				real_ptr, imag_ptr);
      else
	FFTW(execute_split_dft)(this->plan_in_place_,
				imag_ptr, real_ptr,
				imag_ptr, real_ptr);
#else
      typedef Storage<fftw3_storage_format, ctype> storage_type;
      rtype* tmp_in_r = storage_type::get_real_ptr(in_buffer_.ptr());
      rtype* tmp_in_i = storage_type::get_imag_ptr(in_buffer_.ptr());
      memcpy(tmp_in_r, real_ptr, size_[0]*sizeof(rtype));
      memcpy(tmp_in_i, imag_ptr, size_[0]*sizeof(rtype));
      if (E == -1)
	FFTW(execute_split_dft)(plan_out_of_place_,
				tmp_in_r, tmp_in_i,
				real_ptr, imag_ptr);
      else
	FFTW(execute_split_dft)(plan_out_of_place_,
				tmp_in_i, tmp_in_r,
				imag_ptr, real_ptr);
#endif

      real_ptr += fft_stride;
      imag_ptr += fft_stride;
    }
  }

  virtual void out_of_place(ctype *in,
			    stride_type i_str_0, stride_type i_str_1,
			    ctype *out,
			    stride_type o_str_0, stride_type o_str_1,
			    length_type rows, length_type cols)
  {
    // If the inputs to the Fftm are distributed, the number of FFTs may
    // be less than mult_.
    length_type const n_fft          = (axis == 1) ? rows : cols;
    length_type const in_fft_stride  = (axis == 1) ? i_str_0 : i_str_1;
    length_type const out_fft_stride = (axis == 1) ? o_str_0 : o_str_1;

    if (axis == 1) assert(rows <= mult_ && static_cast<int>(cols) == size_[0]);
    else           assert(cols <= mult_ && static_cast<int>(rows) == size_[0]);

    for (index_type i = 0; i != n_fft; ++i)
    {
      FFTW(execute_dft)(plan_out_of_place_, 
			reinterpret_cast<FFTW(complex)*>(in), 
			reinterpret_cast<FFTW(complex)*>(out));
      in  += in_fft_stride;
      out += out_fft_stride;
    }
  }

  virtual void out_of_place(ztype       in,
			    stride_type i_str_0,
			    stride_type i_str_1,
			    ztype       out,
			    stride_type o_str_0,
			    stride_type o_str_1,
			    length_type rows,
			    length_type cols)
  {
    // If the inputs to the Fftm are distributed, the number of FFTs may
    // be less than mult_.
    length_type const n_fft          = (axis == 1) ? rows : cols;
    length_type const in_fft_stride  = (axis == 1) ? i_str_0 : i_str_1;
    length_type const out_fft_stride = (axis == 1) ? o_str_0 : o_str_1;

    if (axis == 1) assert(rows <= mult_ && static_cast<int>(cols) == size_[0]);
    else           assert(cols <= mult_ && static_cast<int>(rows) == size_[0]);

    rtype* in_real  = in.first;
    rtype* in_imag  = in.second;
    rtype* out_real = out.first;
    rtype* out_imag = out.second;

    for (index_type i = 0; i != n_fft; ++i)
    {
      if (E == -1)
	FFTW(execute_split_dft)(plan_out_of_place_, 
				in_real, in_imag, out_real, out_imag);
      else
	FFTW(execute_split_dft)(plan_out_of_place_, 
				in_imag, in_real, out_imag, out_real);
      in_real  += in_fft_stride;
      in_imag  += in_fft_stride;
      out_real += out_fft_stride;
      out_imag += out_fft_stride;
    }
  }

private:
  length_type mult_;
};

#define FFT_DEF(D, I, O, S)	                       \
template <>                                            \
std::auto_ptr<fft::Fft_backend<D, I, O, S> >	       \
create(Domain<D> const &dom, unsigned number)	       \
{                                                      \
  return std::auto_ptr<fft::Fft_backend<D, I, O, S> >  \
    (new Fft_impl<D, I, O, S>(dom, number));           \
}

FFT_DEF(1, SCALAR_TYPE, complex<SCALAR_TYPE>, 0)
FFT_DEF(1, complex<SCALAR_TYPE>, SCALAR_TYPE, 0)
FFT_DEF(1, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, fft_fwd)
FFT_DEF(1, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, fft_inv)
FFT_DEF(2, SCALAR_TYPE, complex<SCALAR_TYPE>, 0)
FFT_DEF(2, SCALAR_TYPE, complex<SCALAR_TYPE>, 1)
FFT_DEF(2, complex<SCALAR_TYPE>, SCALAR_TYPE, 0)
FFT_DEF(2, complex<SCALAR_TYPE>, SCALAR_TYPE, 1)
FFT_DEF(2, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, fft_fwd)
FFT_DEF(2, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, fft_inv)
FFT_DEF(3, SCALAR_TYPE, complex<SCALAR_TYPE>, 0)
FFT_DEF(3, SCALAR_TYPE, complex<SCALAR_TYPE>, 1)
FFT_DEF(3, SCALAR_TYPE, complex<SCALAR_TYPE>, 2)
FFT_DEF(3, complex<SCALAR_TYPE>, SCALAR_TYPE, 0)
FFT_DEF(3, complex<SCALAR_TYPE>, SCALAR_TYPE, 1)
FFT_DEF(3, complex<SCALAR_TYPE>, SCALAR_TYPE, 2)
FFT_DEF(3, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, fft_fwd)
FFT_DEF(3, complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, fft_inv)

#undef FFT_DEF

#define FFTM_DEF(I, O, A, D)		               \
template <>                                            \
std::auto_ptr<fft::Fftm_backend<I, O, A, D> >	       \
create(Domain<2> const &dom, unsigned number)	       \
{                                                      \
  return std::auto_ptr<fft::Fftm_backend<I, O, A, D> > \
    (new Fftm_impl<I, O, A, D>(dom, number));          \
}

FFTM_DEF(SCALAR_TYPE, complex<SCALAR_TYPE>, 0, fft_fwd)
FFTM_DEF(SCALAR_TYPE, complex<SCALAR_TYPE>, 1, fft_fwd)
FFTM_DEF(complex<SCALAR_TYPE>, SCALAR_TYPE, 0, fft_inv)
FFTM_DEF(complex<SCALAR_TYPE>, SCALAR_TYPE, 1, fft_inv)
FFTM_DEF(complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, 0, fft_fwd)
FFTM_DEF(complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, 1, fft_fwd)
FFTM_DEF(complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, 0, fft_inv)
FFTM_DEF(complex<SCALAR_TYPE>, complex<SCALAR_TYPE>, 1, fft_inv)

#undef FFTM_DEF

} // namespace vsip::impl::fftw3
} // namespace vsip::impl
} // namespace vsip
