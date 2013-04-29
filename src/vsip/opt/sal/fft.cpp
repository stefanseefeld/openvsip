/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/sal/fft.cpp
    @author  Stefan Seefeld
    @date    2006-02-20
    @brief   VSIPL++ Library: FFT wrappers and traits to bridge with 
             Mercury's SAL.
*/

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/fft/backend.hpp>
#include <vsip/core/fft/util.hpp>
#include <vsip/opt/sal/fft.hpp>
#include <vsip/core/equal.hpp>
#include <sal.h>
#include <complex>

namespace vsip
{
namespace impl
{
namespace sal
{

long const ESAL = 0;

inline unsigned long
ilog2(length_type size)    // assume size = 2^n, != 0, return n.
{
  unsigned int n = 0;
  while (size >>= 1) ++n;
  return n;
}

inline length_type
get_sizes(Domain<1> const &dom, length_type *size, length_type *l2)
{
  *size = dom.size();
  *l2 = ilog2(*size);
  return *l2;
}
inline length_type
get_sizes(Domain<2> const &dom, length_type *size, length_type *l2)
{
  size[0] = dom[0].size();
  size[1] = dom[1].size();
  l2[0] = ilog2(size[0]);
  l2[1] = ilog2(size[1]);
  return std::max(l2[0], l2[1]);
}
inline length_type
get_sizes(Domain<3> const &dom, length_type *size, length_type *l2)
{
  size[0] = dom[0].size();
  size[1] = dom[1].size();
  size[2] = dom[2].size();
  l2[0] = ilog2(size[0]);
  l2[1] = ilog2(size[1]);
  l2[2] = ilog2(size[2]);
  return std::max(l2[0], std::max(l2[1], l2[2]));
}


template <typename T>
inline void 
unpack(T *data, unsigned long cols, unsigned long rows)
{
  // unpack the data (see SAL reference, figure 3.6, for details).
  unsigned long const rows2 = rows/2 + 1;
  T t10r = data[cols];         // (1, 0).real()
  T t10i = data[cols + 1];     // (1, 0).imag()
  data[cols + 1] = 0.;         // (0, 0).imag()
  data[cols - 2] = data[1];    // (0,-1).real()
  data[cols - 1] = 0.;         // (0,-1).imag()
  data[1] = 0.;
  for (unsigned long r = 1; r != rows2 - 1; ++r)
  {
    // set last column (r,-1)
    data[(r + 1) * cols - 2] = data[2 * r * cols + 1];
    data[(r + 1) * cols - 1] = data[(2 * r + 1) * cols + 1];
    data[2 * r * cols + 1] = 0.;
    data[(2 * r + 1) * cols + 1] = 0.;
    // set first column (r, 0)
    data[r * cols] = data[2 * r * cols];
    data[r * cols + 1] = data[(2 * r + 1) * cols];
    data[2 * r * cols] = 0.;
    data[(2 * r + 1) * cols] = 0.;
  }
  data[(rows2 - 1) * cols] = t10r;
  data[(rows2 - 1) * cols + 1] = 0.;
  data[rows2  * cols - 2] = t10i;
  data[rows2  * cols - 1] = 0;

  // Now fill in the missing cells by symmetry.
  for (unsigned long r = rows2; r != rows; ++r)
  {
    // first column (r, 0)
    data[r * cols] = data[(rows - r) * cols];
    data[r * cols + 1] = - data[(rows - r) * cols + 1];
    // last column (r, -1)
    data[(r + 1) * cols - 2] = data[(rows - r + 1) * cols - 2];
    data[(r + 1) * cols - 1] = - data[(rows - r + 1) * cols - 1];
  }
}

// pack a 2D block according to SAL_Reference.pdf (in particular, Figure 3.6)
// cols is the number of columns (in reals, not complex) of the input block
// rows is the number of rows
template <typename T>
inline void 
pack(T *data, unsigned long cols, unsigned long rows)
{
  unsigned long const rows2 = rows/2 + 1;
  T tmp = data[(rows2 - 1) * cols];                          // (1, 0)
  for (unsigned long r = rows2 - 2; r; --r)
  {
    // pack first column (r, 0)
    data[2 * r * cols] = data[r * cols];                     // (r, 0).real()
    data[(2 * r + 1) * cols] = data[r * cols + 1];           // (r, 0).imag()
    // pack last column (r,-1)
    data[2 * r * cols + 1] = data[(r + 1) * cols - 2];       // (r,-1).real()
    data[(2 * r + 1) * cols + 1] = data[(r + 1) * cols - 1]; // (r,-1).imag()
  }

  data[1] = data[cols - 2];                                  // (0, 1)
  data[cols] = tmp;                                          // (1, 0)
  data[cols + 1] = data[rows2 * cols - 2];                   // (1, 1)
}

template <dimension_type D, bool single> struct Fft_base;

template <dimension_type D> struct Fft_base<D, true /* single precision */>
{
  typedef float rtype;
  typedef COMPLEX ctype;
  typedef COMPLEX_SPLIT ztype;

  Fft_base(Domain<D> const &dom, long options, rtype scale)
    : scale_(scale)
  {
    length_type size = get_sizes(dom, size_, l2size_);
    unsigned long nbytes = 0;
    fft_setup(size, options, &setup_, &nbytes);
    buffer_ = alloc_align<ctype>(32, dom.size());
  }
  ~Fft_base() 
  {
    free_align(buffer_);
    fft_free(&setup_);
  }
  
  void scale(std::complex<rtype> *data, length_type size, rtype s)
  {
    rtype *d = reinterpret_cast<rtype*>(data);
    vsmulx(d, 1, &s, d, 1, 2 * size, ESAL);
  }
  void scale(std::pair<rtype*, rtype*> data, length_type size, rtype s)
  {
    vsmulx(data.first, 1, &s, data.first, 1, size, ESAL);
    vsmulx(data.second, 1, &s, data.second, 1, size, ESAL);
  }
  void scale(rtype *data, length_type size, rtype s)
  {
    vsmulx(data, 1, &s, data, 1, size, ESAL);
  }

  void cip(std::complex<rtype> *data, long dir)
  {
    fft_cipx(&setup_, reinterpret_cast<ctype *>(data), 2, l2size_[0], dir, ESAL);
  }
  void zip(std::pair<rtype*, rtype*> inout, stride_type stride, long dir)
  {
    ztype data = {inout.first, inout.second};
    ztype tmp = {reinterpret_cast<rtype*>(buffer_),
		 reinterpret_cast<rtype*>(buffer_) + size_[0]};
    fft_ziptx(&setup_, &data, stride, &tmp, l2size_[0], dir, ESAL);
  }
  void cip2d(std::complex<rtype> *data, dimension_type axis, long dir)
  {
    fft2d_cipx(&setup_, reinterpret_cast<ctype *>(data), 2, 2 * size_[axis],
	       l2size_[axis], l2size_[1 - axis], dir, ESAL);
  }
  void zip2d(std::pair<rtype*, rtype*> inout, dimension_type axis, long dir)
  {
    ztype data = {inout.first, inout.second};
    fft2d_zipx(&setup_, &data, 1, size_[axis],
	       l2size_[axis], l2size_[1 - axis], dir, ESAL);
  }
  void rop(rtype *in, std::complex<rtype> *out)
  {
    rtype *data = reinterpret_cast<rtype*>(out);
    fft_roptx(&setup_, in, 1, data, 1, reinterpret_cast<rtype *>(buffer_), 
	     l2size_[0], FFT_FORWARD, ESAL);
    // unpack the data (see SAL reference for details).
    int const N = size_[0] + 2;
    data[N - 2] = data[1];
    data[1] = 0.f;
    data[N - 1] = 0.f;
  }
  void rop(std::complex<rtype> *in, rtype *out)
  {
    rtype *data = reinterpret_cast<rtype*>(in);
    // pack the data (see SAL reference for details).
    int const N = size_[0] + 2;
    data[1] = data[N - 2];
    data[N - 2] = data[N - 1] = 0.f;
    fft_ropx(&setup_, data, 1, out, 1, l2size_[0], FFT_INVERSE, ESAL);
  }
  void cop(std::complex<rtype> *in_arg, std::complex<rtype> *out_arg, long dir)
  {
    ctype *in = reinterpret_cast<ctype*>(in_arg);
    ctype *out = reinterpret_cast<ctype*>(out_arg);
    fft_coptx(&setup_, in, 2, out, 2, buffer_, l2size_[0], dir, ESAL);
  }
  void zop(std::pair<rtype*, rtype*> in_arg, stride_type in_stride,
	   std::pair<rtype*, rtype*> out_arg, stride_type out_stride, long dir)
  {
    ztype in = {in_arg.first, in_arg.second};
    ztype out = {out_arg.first, out_arg.second};
    ztype tmp = {reinterpret_cast<rtype*>(buffer_),
		 reinterpret_cast<rtype*>(buffer_) + size_[0]};
    fft_zoptx(&setup_, &in, in_stride, &out, out_stride,
	      &tmp, l2size_[0], dir, ESAL);
  }
  void rop2d(rtype *in, stride_type in_stride,
	     std::complex<rtype> *out_arg, stride_type out_stride,
	     dimension_type axis)
  {
    rtype *out = reinterpret_cast<rtype*>(out_arg);
    fft2d_roptx(&setup_, in, 1, in_stride, out, 1, 2 * out_stride,
		reinterpret_cast<rtype*>(buffer_),
		l2size_[axis], l2size_[1 - axis], FFT_FORWARD, ESAL);
    // unpack the data (see SAL reference, figure 3.6, for details).
    unpack(out, size_[axis] + 2, size_[1 - axis]);
  }
  void rop2d(std::complex<rtype> *in_arg, stride_type in_stride,
	     rtype *out, stride_type out_stride, dimension_type axis)
  {
    rtype *in = reinterpret_cast<rtype*>(in_arg);
    // pack the data (see SAL reference, figure 3.6, for details).
    pack(in, size_[axis] + 2, size_[1 - axis]);
    fft2d_ropx(&setup_, in, 1, 2 * in_stride, out, 1, out_stride,
	       l2size_[axis], l2size_[1 - axis], FFT_INVERSE, ESAL);
  }
  void cop2d(std::complex<rtype> *in_arg, stride_type in_stride,
	     std::complex<rtype> *out_arg, stride_type out_stride,
	     dimension_type axis, long dir)
  {
    ctype *in = reinterpret_cast<ctype*>(in_arg);
    ctype *out = reinterpret_cast<ctype*>(out_arg);
    fft2d_coptx(&setup_, in, 2, 2 * in_stride, out, 2, 2 * out_stride,
		reinterpret_cast<ctype*>(buffer_),
		l2size_[axis], l2size_[1 - axis], dir, ESAL);
  }
  void zop2d(std::pair<rtype*, rtype*> in_arg,
	     stride_type in_r_stride, stride_type in_c_stride,
	     std::pair<rtype*, rtype*> out_arg,
	     stride_type out_r_stride, stride_type out_c_stride,
	     dimension_type axis, long dir)
  {
    ztype in = {in_arg.first, in_arg.second};
    ztype out = {out_arg.first, out_arg.second};
    ztype tmp = {reinterpret_cast<rtype*>(buffer_),
		 reinterpret_cast<rtype*>(buffer_) +  size_[0] * size_[1]};
    fft2d_zoptx(&setup_, &in, in_r_stride, in_c_stride,
		&out, out_r_stride, out_c_stride,
		&tmp, l2size_[0], l2size_[1], dir, ESAL);
  }
  void cipm(std::complex<rtype> *inout, stride_type stride,
	    length_type n_fft,
	    dimension_type axis, long dir)
  {
    assert(n_fft <= size_[1-axis]);
    fftm_cipx(&setup_, reinterpret_cast<ctype *>(inout),
	      2, 2 * stride, l2size_[axis], n_fft, dir, ESAL);
  }
  void zipm(std::pair<rtype*, rtype*> inout, stride_type stride,
	    length_type n_fft,
	    dimension_type axis, long dir)
  {
    ztype data = {inout.first, inout.second};
    assert(n_fft <= size_[1-axis]);
    fftm_zipx(&setup_, &data, 1, stride,
	      l2size_[axis], n_fft, dir, ESAL);
  }
  void ropm(rtype *in, stride_type in_stride,
	    std::complex<rtype> *out_arg, stride_type out_stride,
	    dimension_type axis)
  {
    rtype *out = reinterpret_cast<rtype*>(out_arg);
    fftm_roptx(&setup_, in, 1, in_stride,
	       out, 1, 2 * out_stride,
	       reinterpret_cast<rtype*>(buffer_),
	       l2size_[axis], size_[1 - axis], FFT_FORWARD, ESAL);
    // Unpack the data (see SAL reference for details), and scale back by 1/2.
    int const N = size_[axis] + 2;
    rtype scale = scale_ * 0.5f;
    for (unsigned int i = 0; i != size_[1 - axis]; ++i, out += 2 * out_stride)
    {
      out[N - 2] = out[1];
      out[N - 1] = 0.;
      out[1] = 0.f;
      vsmulx(out, 1, &scale, out, 1, N, ESAL);
    }
  }
  void ropm(std::complex<rtype> *in_arg, stride_type in_stride,
	    rtype *out, stride_type out_stride,
	    dimension_type axis)
  {
    rtype *in = reinterpret_cast<rtype*>(in_arg);
    // Pack the data (see SAL reference for details).
    int const N = size_[axis] + 2;
    for (unsigned int i = 0; i != size_[1 - axis]; ++i, in += 2 * in_stride)
    {
      in[1] = in[N - 2];
      in[N - 2] = in[N - 1] = 0.f;
    }
    in = reinterpret_cast<rtype*>(in_arg);
    fftm_ropx(&setup_, in, 1, 2 * in_stride, out, 1, out_stride,
	      l2size_[axis], size_[1 - axis], FFT_INVERSE, ESAL);
    for (unsigned int i = 0; i != size_[1 - axis]; ++i, out += out_stride)
      vsmulx(out, 1, &scale_, out, 1, size_[axis], ESAL);
  }
  void copm(std::complex<rtype> *in, stride_type in_stride,
	    std::complex<rtype> *out, stride_type out_stride,
	    length_type n_fft,
	    dimension_type axis, long dir)
  {
    assert(n_fft <= size_[1-axis]);
    fftm_coptx(&setup_, reinterpret_cast<ctype *>(in), 2, 2 * in_stride,
	       reinterpret_cast<ctype *>(out), 2, 2 * out_stride,
	       reinterpret_cast<ctype *>(buffer_),
	       l2size_[axis], n_fft, dir, ESAL);
  }
  void zopm(std::pair<rtype*,rtype*> in_arg, stride_type in_stride,
	    std::pair<rtype*,rtype*> out_arg, stride_type out_stride,
	    length_type n_fft,
	    dimension_type axis, long dir)
  {
    ztype in = {in_arg.first, in_arg.second};
    ztype out = {out_arg.first, out_arg.second};
    ztype tmp = {reinterpret_cast<rtype*>(buffer_),
		 reinterpret_cast<rtype*>(buffer_) + size_[0] * size_[1]};
    assert(n_fft <= size_[1-axis]);
    fftm_zoptx(&setup_, &in, 1, in_stride, &out, 1, out_stride, &tmp,
	       l2size_[axis], n_fft, dir, ESAL);
  }

  FFT_setup setup_;
  length_type size_[D];
  length_type l2size_[D];
  rtype scale_;
  ctype *buffer_;
};

template <dimension_type D> struct Fft_base<D, false /* single precision */>
{
  typedef double rtype;
  typedef DOUBLE_COMPLEX ctype;
  typedef DOUBLE_COMPLEX_SPLIT ztype;

  Fft_base(Domain<D> const &dom, long options, rtype scale)
    : scale_(scale)
  {
    length_type size = get_sizes(dom, size_, l2size_);
    unsigned long nbytes = 0;
    fft_setupd(size, options, &setup_, &nbytes);
    buffer_ = alloc_align<ctype>(32, dom.size());
  }
  ~Fft_base() 
  {
    free_align(buffer_);
    fft_freed(&setup_);
  }
  
  void scale(std::complex<rtype> *data, length_type size, rtype s)
  {
    rtype *d = reinterpret_cast<rtype*>(data);
    vsmuldx(d, 1, &s, d, 1, 2 * size, ESAL);
  }
  void scale(std::pair<rtype*, rtype*> data, length_type size, rtype s)
  {
    vsmuldx(data.first, 1, &s, data.first, 1, size, ESAL);
    vsmuldx(data.second, 1, &s, data.second, 1, size, ESAL);
  }
  void scale(rtype *data, length_type size, rtype s)
  {
    vsmuldx(data, 1, &s, data, 1, size, ESAL);
  }

  void cip(std::complex<rtype> *data, long dir)
  {
    fft_cipdx(&setup_, reinterpret_cast<ctype *>(data), 2, l2size_[0], dir, ESAL);
  }
  void zip(std::pair<rtype*, rtype*> inout, stride_type stride, long dir)
  {
    ztype data = {inout.first, inout.second};
    ztype tmp = {reinterpret_cast<rtype*>(buffer_),
		 reinterpret_cast<rtype*>(buffer_) + size_[0]};
    fft_ziptdx(&setup_, &data, stride, &tmp, l2size_[0], dir, ESAL);
  }
  void cip2d(std::complex<rtype> *data, dimension_type axis, long dir)
  {
    fft2d_cipdx(&setup_, reinterpret_cast<ctype *>(data), 2, 2*size_[axis],
		l2size_[axis], l2size_[1 - axis], dir, ESAL);
  }
  void zip2d(std::pair<rtype*, rtype*> inout, dimension_type axis, long dir)
  {
    ztype data = {inout.first, inout.second};
    fft2d_zipdx(&setup_, &data, 1, size_[axis],
		l2size_[axis], l2size_[1 - axis], dir, ESAL);
  }
  void rop(rtype *in, std::complex<rtype> *out)
  {
    rtype *data = reinterpret_cast<rtype*>(out);
    fft_roptdx(&setup_, in, 1, data, 1, reinterpret_cast<rtype *>(buffer_),
	       l2size_[0], FFT_FORWARD, ESAL);
    // unpack the data (see SAL reference for details).
    int const N = size_[0] + 2;
    data[N - 2] = data[1];
    data[1] = 0.f;
    data[N - 1] = 0.f;
  }
  void rop(std::complex<rtype> *in, rtype *out)
  {
    rtype *data = reinterpret_cast<rtype*>(in);
    // pack the data (see SAL reference for details).
    int const N = size_[0] + 2;
    data[1] = data[N - 2];
    data[N - 2] = data[N - 1] = 0.f;

    fft_roptdx(&setup_, data, 1, out, 1, reinterpret_cast<rtype *>(buffer_),
	       l2size_[0], FFT_INVERSE, ESAL);
  }
  void cop(std::complex<rtype> *in_arg, std::complex<rtype> *out_arg, long dir)
  {
    ctype *in = reinterpret_cast<ctype*>(in_arg);
    ctype *out = reinterpret_cast<ctype*>(out_arg);
    fft_coptdx(&setup_, in, 2, out, 2, buffer_, l2size_[0], dir, ESAL);
  }
  void zop(std::pair<rtype*, rtype*> in_arg, stride_type in_stride,
	   std::pair<rtype*, rtype*> out_arg, stride_type out_stride, long dir)
  {
    ztype in = {in_arg.first, in_arg.second};
    ztype out = {out_arg.first, out_arg.second};
    ztype tmp = {reinterpret_cast<rtype*>(buffer_),
		 reinterpret_cast<rtype*>(buffer_) + size_[0]};
    fft_zoptdx(&setup_, &in, in_stride, &out, out_stride,
	       &tmp, l2size_[0], dir, ESAL);
  }
  void rop2d(rtype *in, stride_type in_stride,
	     std::complex<rtype> *out_arg, stride_type out_stride,
	     dimension_type axis)
  {
    rtype *out = reinterpret_cast<rtype*>(out_arg);
    fft2d_roptdx(&setup_, in, 1, in_stride, out, 1, 2 * out_stride,
		 reinterpret_cast<rtype*>(buffer_),
		 l2size_[axis], l2size_[1 - axis], FFT_FORWARD, ESAL);
    // unpack the data (see SAL reference, figure 3.6, for details).
    unpack(out, size_[axis] + 2, size_[1 - axis]);
  }
  void rop2d(std::complex<rtype> *in_arg, stride_type in_stride,
	     rtype *out, stride_type out_stride, dimension_type axis)
  {
    rtype *in = reinterpret_cast<rtype*>(in_arg);
    // pack the data (see SAL reference, figure 3.6, for details).
    pack(in, size_[axis] + 2, size_[1 - axis]);
    fft2d_ropdx(&setup_, in, 1, 2 * in_stride, out, 1, out_stride,
		l2size_[axis], l2size_[1 - axis], FFT_INVERSE, ESAL);
  }
  void cop2d(std::complex<rtype> *in_arg, stride_type in_r_stride,
	     std::complex<rtype> *out_arg, stride_type out_r_stride,
	     dimension_type axis, long dir)
  {
    ctype *in = reinterpret_cast<ctype*>(in_arg);
    ctype *out = reinterpret_cast<ctype*>(out_arg);
    fft2d_coptdx(&setup_, in, 2, 2 * in_r_stride, out, 2, 2 * out_r_stride,
		 reinterpret_cast<ctype*>(buffer_),
		 l2size_[axis], l2size_[1 - axis], dir, ESAL);
  }
  void zop2d(std::pair<rtype*, rtype*> in_arg,
	     stride_type in_r_stride, stride_type in_c_stride,
	     std::pair<rtype*, rtype*> out_arg,
	     stride_type out_r_stride, stride_type out_c_stride,
	     dimension_type axis, long dir)
  {
    ztype in = {in_arg.first, in_arg.second};
    ztype out = {out_arg.first, out_arg.second};
    ztype tmp = {reinterpret_cast<rtype*>(buffer_),
		 reinterpret_cast<rtype*>(buffer_) + size_[0] * size_[1]};
    fft2d_zoptdx(&setup_, &in, in_r_stride, in_c_stride,
		 &out, out_r_stride, out_c_stride,
		 &tmp, l2size_[0], l2size_[1], dir, ESAL);
  }
  void cipm(std::complex<rtype> *inout, stride_type stride,
	    length_type n_fft,
	    dimension_type axis, long dir)
  {
    assert(n_fft <= size_[1-axis]);
    fftm_cipdx(&setup_, reinterpret_cast<ctype *>(inout),
	       2, 2 * stride, l2size_[axis], n_fft, dir, ESAL);
  }
  void zipm(std::pair<rtype*, rtype*> inout, stride_type stride,
	    length_type n_fft,
	    dimension_type axis, long dir)
  {
    ztype data = {inout.first, inout.second};
    assert(n_fft <= size_[1-axis]);
    fftm_zipdx(&setup_, &data, 1, stride,
	       l2size_[axis], n_fft, dir, ESAL);
  }
  void ropm(rtype *in, stride_type in_stride,
	    std::complex<rtype> *out_arg, stride_type out_stride,
	    dimension_type axis)
  {
    rtype *out = reinterpret_cast<rtype*>(out_arg);
    fftm_roptdx(&setup_, in, 1, in_stride,
		out, 1, 2 * out_stride,
		reinterpret_cast<rtype*>(buffer_),
		l2size_[axis], size_[1 - axis], FFT_FORWARD, ESAL);
    // Unpack the data (see SAL reference for details), and scale back by 1/2.
    int const N = size_[axis] + 2;
    rtype scale = scale_ * 0.5f;
    for (unsigned int i = 0; i != size_[1 - axis]; ++i, out += 2 * out_stride)
    {
      out[N - 2] = out[1];
      out[N - 1] = 0.;
      out[1] = 0.f;
      vsmuldx(out, 1, &scale, out, 1, N, ESAL);
    }
  }
  void ropm(std::complex<rtype> *in_arg, stride_type in_stride,
	    rtype *out, stride_type out_stride,
	    dimension_type axis)
  {
    rtype *in = reinterpret_cast<rtype*>(in_arg);
    // Pack the data (see SAL reference for details).
    int const N = size_[axis] + 2;
    for (unsigned int i = 0; i != size_[1 - axis]; ++i, in += 2 * in_stride)
    {
      in[1] = in[N - 2];
      in[N - 2] = in[N - 1] = 0.f;
    }
    in = reinterpret_cast<rtype*>(in_arg);
    fftm_ropdx(&setup_, in, 1, 2 * in_stride, out, 1, out_stride,
	       l2size_[axis], size_[1 - axis], FFT_INVERSE, ESAL);
    for (unsigned int i = 0; i != size_[1 - axis]; ++i, out += out_stride)
      vsmuldx(out, 1, &scale_, out, 1, size_[axis], ESAL);
  }
  void copm(std::complex<rtype> *in, stride_type in_stride,
	    std::complex<rtype> *out, stride_type out_stride,
	    length_type n_fft,
	    dimension_type axis, long dir)
  {
    assert(n_fft <= size_[1-axis]);
    fftm_coptdx(&setup_, reinterpret_cast<ctype *>(in), 2, 2 * in_stride,
		reinterpret_cast<ctype *>(out), 2, 2 * out_stride,
		reinterpret_cast<ctype *>(buffer_),
		l2size_[axis], n_fft, dir, ESAL);
  }
  void zopm(std::pair<rtype*,rtype*> in_arg, stride_type in_stride,
	    std::pair<rtype*,rtype*> out_arg, stride_type out_stride,
	    length_type n_fft,
	    dimension_type axis, long dir)
  {
    ztype in = {in_arg.first, in_arg.second};
    ztype out = {out_arg.first, out_arg.second};
    ztype tmp = {reinterpret_cast<rtype*>(buffer_),
		 reinterpret_cast<rtype*>(buffer_) + size_[0] * size_[1]};
    assert(n_fft <= size_[1-axis]);
    fftm_zoptdx(&setup_, &in, 1, in_stride, &out, 1, out_stride, &tmp,
		l2size_[axis], n_fft, dir, ESAL);
  }

  FFT_setupd setup_;
  length_type size_[D];
  length_type l2size_[D];
  rtype scale_;
  ctype *buffer_;
};

template <typename T> struct precision;
template <> struct precision<float> { static bool const single = true;};
template <> struct precision<double> { static bool const single = false;};

template <dimension_type D, typename I, typename O, int S>
class Fft_impl;

template <typename T, int S>
class Fft_impl<1, std::complex<T>, std::complex<T>, S>
  : private Fft_base<1, precision<T>::single>,
    public fft::Fft_backend<1, std::complex<T>, std::complex<T>, S>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;
  static int const direction = S == fft_fwd ? FFT_FORWARD : FFT_INVERSE;

public:
  Fft_impl(Domain<1> const &dom, T scale)
    : Fft_base<1, precision<T>::single>(dom, 0, scale) {}

  virtual bool supports_scale() { return true;}
  virtual void query_layout(Rt_layout<1> &rtl_inout)
  {
    rtl_inout.packing = dense;
    rtl_inout.order = tuple<0, 1, 2>();
  }
  virtual void query_layout(Rt_layout<1> &rtl_in, Rt_layout<1> &rtl_out)
  {
    rtl_in.packing = dense;
    rtl_in.order = tuple<0, 1, 2>();
    rtl_in.storage_format = rtl_out.storage_format;
    rtl_out.packing = dense;
    rtl_out.order = tuple<0, 1, 2>();
  }

  virtual void in_place(ctype *data, stride_type stride, length_type size)
  {
    assert(stride == 1);
    assert(size == this->size_[0]);
    cip(data, direction);
    if (this->scale_ != T(1.))
      scale(data, this->size_[0], this->scale_);
  }

  virtual void in_place(ztype data, stride_type stride, length_type size)
  {
    assert(size == this->size_[0]);
    zip(data, stride, direction);
    if (this->scale_ != T(1.))
    {
      scale(data.first, this->size_[0], this->scale_);
      scale(data.second, this->size_[0], this->scale_);
    }
  }

  virtual void out_of_place(ctype *in, stride_type in_stride,
			    ctype *out, stride_type out_stride,
			    length_type size)
  {
    assert(in_stride == 1 && out_stride == 1);
    assert(size == this->size_[0]);
    cop(in, out, direction);
    if (this->scale_ != T(1.))
      scale(out, this->size_[0], this->scale_);
  }
  virtual void out_of_place(ztype in, stride_type in_stride,
			    ztype out, stride_type out_stride,
			    length_type size)
  {
    assert(size == this->size_[0]);
    zop(in, in_stride, out, out_stride, direction);
    if (this->scale_ != T(1.))
    {
      scale(out.first, this->size_[0], this->scale_);
      scale(out.second, this->size_[0], this->scale_);
    }
  }
};

template <typename T, int A>
class Fft_impl<1, T, std::complex<T>, A>
  : private Fft_base<1, precision<T>::single>,
    public fft::Fft_backend<1, T, std::complex<T>, A>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<1> const &dom, T scale)
    : Fft_base<1, precision<T>::single>(dom, 0, scale) {}

  virtual bool supports_scale() { return true;}
  virtual void query_layout(Rt_layout<1> &rtl_in, Rt_layout<1> &rtl_out)
  {
    rtl_in.packing = dense;
    rtl_in.order = tuple<0, 1, 2>();
    rtl_in.storage_format = interleaved_complex;
    rtl_out.packing = dense;
    rtl_out.order = tuple<0, 1, 2>();
    rtl_out.storage_format = interleaved_complex;
  }

  virtual void out_of_place(T *in, stride_type in_stride,
			    std::complex<T> *out, stride_type out_stride,
			    length_type size)
  {
    assert(in_stride == 1 && out_stride == 1);
    rop(in, out);
    T s = this->scale_ * 0.5;
    if (!almost_equal(s, T(1.)))
      scale(out, size/2 + 1, s);
  }
};

template <typename T, int A>
class Fft_impl<1, std::complex<T>, T, A>
  : private Fft_base<1, precision<T>::single>,
    public fft::Fft_backend<1, std::complex<T>, T, A>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fft_impl(Domain<1> const &dom, T scale)
    : Fft_base<1, precision<T>::single>(dom, 0, scale) {}

  virtual bool supports_scale() { return true;}
  virtual void query_layout(Rt_layout<1> &rtl_in, Rt_layout<1> &rtl_out)
  {
    rtl_in.packing = dense;
    rtl_in.order = tuple<0, 1, 2>();
    rtl_in.storage_format = interleaved_complex;
    rtl_out.packing = dense;
    rtl_out.order = tuple<0, 1, 2>();
    rtl_out.storage_format = interleaved_complex;
  }
  // SAL requires the input to be packed, so we will modify the input
  // before passing it along.
  virtual bool requires_copy(Rt_layout<1> &) { return true;}

  virtual void out_of_place(ctype *in, stride_type in_stride,
			    T *out, stride_type out_stride,
			    length_type size)
  {
    assert(in_stride == 1 && out_stride == 1);
    assert(size == this->size_[0]);
    rop(in, out);
    if (this->scale_ != T(1.))
      scale(out, this->size_[0], this->scale_);
  }
};

template <typename T, int S>
class Fft_impl<2, std::complex<T>, std::complex<T>, S>
  : private Fft_base<2, precision<T>::single>,
    public fft::Fft_backend<2, std::complex<T>, std::complex<T>, S>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;
  static int const direction = S == fft_fwd ? FFT_FORWARD : FFT_INVERSE;

public:
  Fft_impl(Domain<2> const &dom, T scale)
    : Fft_base<2, precision<T>::single>(dom, 0, scale) {}

  virtual bool supports_scale() { return true;}
  virtual void query_layout(Rt_layout<2> &rtl_inout)
  {
    // We require unit-stride in either direction.
    // Whichever it is determines the order in which sub-operations
    // are applied.
    rtl_inout.packing = dense;
  }
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    rtl_in.packing = dense;
    rtl_in.storage_format = rtl_out.storage_format;
    rtl_out.packing = rtl_in.packing;
    rtl_out.order = rtl_in.order;
  }

  virtual void in_place(std::complex<T> *inout,
			stride_type r_stride, stride_type c_stride,
			length_type rows, length_type cols)
  {
    int axis = r_stride == 1 ? 0 : 1;
    cip2d(inout, axis, direction);
    if (this->scale_ != T(1.))
      if (axis == 0)
	for (length_type i = 0; i != cols; ++i)
	  scale(inout + i * c_stride, rows, this->scale_);
      else
	for (length_type i = 0; i != rows; ++i)
	  scale(inout + i * r_stride, cols, this->scale_);
  }

  virtual void in_place(std::pair<T *, T *> inout,
			stride_type r_stride, stride_type c_stride,
			length_type rows, length_type cols)
  {
    int axis = r_stride == 1 ? 0 : 1;
    zip2d(inout, axis, direction);
    if (this->scale_ != T(1.))
      if (axis == 0)
	for (length_type i = 0; i != cols; ++i)
	{
 	  scale(inout.first + i * c_stride, rows, this->scale_);
 	  scale(inout.second + i * c_stride, rows, this->scale_);
	}
      else
	for (length_type i = 0; i != rows; ++i)
	{
	  scale(inout.first + i * r_stride, cols, this->scale_);    
	  scale(inout.second + i * r_stride, cols, this->scale_);    
	}
  }

  virtual void out_of_place(std::complex<T> *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    std::complex<T> *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    assert(rows == this->size_[0] && cols == this->size_[1]);
    int axis = in_r_stride == 1 ? 0 : 1;
    if (axis == 0)
    {
      assert(in_r_stride == 1 && out_r_stride == 1);
      cop2d(in, in_c_stride, out, out_c_stride, axis, direction);
      if (this->scale_ != T(1.))
	for (length_type i = 0; i != cols; ++i)
	  scale(out + i * out_c_stride, rows, this->scale_);
    }
    else
    {
      assert(in_c_stride == 1 && out_c_stride == 1);
      cop2d(in, in_r_stride, out, out_r_stride, axis, direction);
      if (this->scale_ != T(1.))
	for (length_type i = 0; i != rows; ++i)
	  scale(out + i * out_r_stride, cols, this->scale_);
    }
  }
  virtual void out_of_place(std::pair<T *, T *> in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    std::pair<T *, T *> out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    int axis = in_r_stride == 1 ? 0 : 1;
    assert(rows == this->size_[0] && cols == this->size_[1]);
    zop2d(in, in_r_stride, in_c_stride, out, out_r_stride, out_c_stride,
	  axis, direction);
    if (this->scale_ != T(1.))
      if (axis == 0)
	for (length_type i = 0; i != cols; ++i)
	{
	  scale(out.first + i * out_c_stride, rows, this->scale_);
	  scale(out.second + i * out_c_stride, rows, this->scale_);
	}
      else
	for (length_type i = 0; i != rows; ++i)
	{
	  scale(out.first + i * out_r_stride, cols, this->scale_);
	  scale(out.second + i * out_r_stride, cols, this->scale_);
	}
  }
};

template <typename T, int S>
class Fft_impl<2, T, std::complex<T>, S>
  : private Fft_base<2, precision<T>::single>,
    public fft::Fft_backend<2, T, std::complex<T>, S>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const axis = S;

public:
  Fft_impl(Domain<2> const &dom, T scale)
    : Fft_base<2, precision<T>::single>(dom, 0, scale) {}

  virtual bool supports_scale() { return true;}
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    rtl_in.packing = dense;
    if (axis == 0) rtl_in.order = col2_type();
    else rtl_in.order = row2_type();
    rtl_out.packing = dense;
    rtl_out.order = rtl_in.order;
    rtl_out.storage_format = interleaved_complex;
  }

  virtual void out_of_place(T *in, stride_type in_r_stride, stride_type in_c_stride,
			    std::complex<T> *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    assert(rows == this->size_[0] && cols == this->size_[1]);
    T s = this->scale_ * 0.5;
    if (axis == 0)
    {
      assert(in_r_stride == 1 && out_r_stride == 1);
      rop2d(in, in_c_stride, out, out_c_stride, axis);
      if (!almost_equal(s, T(1.)))
 	for (length_type i = 0; i != cols; ++i)
 	  scale(out + i * out_c_stride, rows / 2 + 1, s);
    }
    else
    {
      assert(in_c_stride == 1 && out_c_stride == 1);
      rop2d(in, in_r_stride, out, out_r_stride, axis);
      if (!almost_equal(s, T(1.)))
 	for (length_type i = 0; i != rows; ++i)
 	  scale(out + i * out_r_stride, cols / 2 + 1, s);
    }
  }
};

template <typename T, int S>
class Fft_impl<2, std::complex<T>, T, S>
  : private Fft_base<2, precision<T>::single>,
    public fft::Fft_backend<2, std::complex<T>, T, S>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

  static int const axis = S;

public:
  Fft_impl(Domain<2> const &dom, T scale)
    : Fft_base<2, precision<T>::single>(dom, 0, scale) {}

  virtual bool supports_scale() { return true;}
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    rtl_in.packing = dense;
    if (axis == 0) rtl_in.order = col2_type();
    else rtl_in.order = row2_type();
    rtl_in.storage_format = interleaved_complex;
    rtl_out.packing = dense;
    rtl_out.order = rtl_in.order;
  }
  // SAL requires the input to be packed, so we will modify the input
  // before passing it along.
  virtual bool requires_copy(Rt_layout<2> &) { return true;}

  virtual void out_of_place(std::complex<T> *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    T *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    assert(rows == this->size_[0] && cols == this->size_[1]);
    if (axis == 0)
    {
      assert(in_r_stride == 1 && out_r_stride == 1);
      rop2d(in, in_c_stride, out, out_c_stride, axis);
      if (this->scale_ != T(1.))
	for (length_type i = 0; i != cols; ++i)
	  scale(out + i * out_c_stride, rows, this->scale_);
    }
    else
    {
      assert(in_c_stride == 1 && out_c_stride == 1);
      rop2d(in, in_r_stride, out, out_r_stride, axis);
      if (this->scale_ != T(1.))
	for (length_type i = 0; i != rows; ++i)
	  scale(out + i * out_r_stride, cols, this->scale_);
    }
  }
};

template <typename I, typename O, int A, int D> class Fftm_impl;

template <typename T, int A>
class Fftm_impl<T, std::complex<T>, A, fft_fwd>
  : private Fft_base<2, precision<T>::single>,
    public fft::Fftm_backend<T, std::complex<T>, A, fft_fwd>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fftm_impl(Domain<2> const &dom, T scale)
    : Fft_base<2, precision<T>::single>(dom, 0, scale) {}

  virtual bool supports_scale() { return true;}
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    rtl_in.packing = dense;
    if (A == vsip::col) rtl_in.order = col2_type();
    else rtl_in.order = row2_type();
    rtl_out.packing = rtl_in.packing;
    rtl_out.order = rtl_in.order;
    rtl_out.storage_format = interleaved_complex;
  }
  virtual void out_of_place(T *in, stride_type in_r_stride, stride_type in_c_stride,
			    std::complex<T> *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    assert(rows == this->size_[0] && cols == this->size_[1]);
    if (A == vsip::row)
    {
      assert(in_c_stride == 1 && out_c_stride == 1);
      ropm(in, in_r_stride, out, out_r_stride, 1);
      // Scaling done in ropm()
    }
    else
    {
      assert(in_r_stride == 1 && out_r_stride == 1);
      ropm(in, in_c_stride, out, out_c_stride, 0);
      // Scaling done in ropm()
    }
  }
};

template <typename T, int A>
class Fftm_impl<std::complex<T>, T, A, fft_inv>
  : private Fft_base<2, precision<T>::single>,
    public fft::Fftm_backend<std::complex<T>, T, A, fft_inv>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;

public:
  Fftm_impl(Domain<2> const &dom, T scale)
    : Fft_base<2, precision<T>::single>(dom, 0, scale) {}

  virtual bool supports_scale() { return true;}
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    rtl_in.packing = dense;
    if (A == vsip::col) rtl_in.order = col2_type();
    else rtl_in.order = row2_type();
    rtl_in.storage_format = interleaved_complex;
    rtl_out.packing = rtl_in.packing;
    rtl_out.order = rtl_in.order;
  }
  // SAL requires the input to be packed, so we will modify the input
  // before passing it along.
  virtual bool requires_copy(Rt_layout<2> &) { return true;}
  virtual void out_of_place(std::complex<T> *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    T *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    assert(rows == this->size_[0] && cols == this->size_[1]);
    if (A == vsip::col)
    {
      assert(in_r_stride == 1 && out_r_stride == 1);
      ropm(in, in_c_stride, out, out_c_stride, 0);
      // Scaling done in ropm()
    }
    else
    {
      assert(in_c_stride == 1 && out_c_stride == 1);
      ropm(in, in_r_stride, out, out_r_stride, 1);
      // Scaling done in ropm()
    }
  }
};

template <typename T, int A, int D>
class Fftm_impl<std::complex<T>, std::complex<T>, A, D>
  : private Fft_base<2, precision<T>::single>,
    public fft::Fftm_backend<std::complex<T>, std::complex<T>, A, D>
{
  typedef T rtype;
  typedef std::complex<rtype> ctype;
  typedef std::pair<rtype*, rtype*> ztype;
  static int const direction = D == fft_fwd ? FFT_FORWARD : FFT_INVERSE;

public:
  Fftm_impl(Domain<2> const &dom, T scale)
    : Fft_base<2, precision<T>::single>(dom, 0, scale) {}

  virtual bool supports_scale() { return true;}
  virtual void query_layout(Rt_layout<2> &rtl_inout)
  {
    rtl_inout.packing = dense;
    if (A == vsip::col) rtl_inout.order = col2_type();
    else rtl_inout.order = row2_type();
  }
  virtual void query_layout(Rt_layout<2> &rtl_in, Rt_layout<2> &rtl_out)
  {
    rtl_in.packing = dense;
    if (A == vsip::col) rtl_in.order = col2_type();
    else rtl_in.order = row2_type();
    rtl_in.storage_format = rtl_out.storage_format;
    rtl_out.packing = rtl_in.packing;
    rtl_out.order = rtl_in.order;
  }
  virtual void in_place(std::complex<T> *inout,
			stride_type r_stride, stride_type c_stride,
			length_type rows, length_type cols)
  {
    if (A == vsip::row)
    {
      assert(rows <= this->size_[0]); // OK if rows are distributed.
      assert(cols == this->size_[1]); // Columns must be whole.
      assert(c_stride == 1);
      cipm(inout, r_stride, rows, 1, direction);
      if (this->scale_ != T(1.))
	for (length_type i = 0; i != rows; ++i)
	  scale(inout + i * r_stride, cols, this->scale_);
    }
    else
    {
      assert(rows == this->size_[0]); // Rows must be whole.
      assert(cols <= this->size_[1]); // OK if columns are distributed.
      assert(r_stride == 1);
      cipm(inout, c_stride, cols, 0, direction);
      if (this->scale_ != T(1.))
	for (length_type i = 0; i != cols; ++i)
	  scale(inout + i * c_stride, rows, this->scale_);
    }
  }

  virtual void in_place(std::pair<T *, T *> inout,
			stride_type r_stride, stride_type c_stride,
			length_type rows, length_type cols)
  {
    if (A == vsip::row)
    {
      assert(rows <= this->size_[0]); // OK if rows are distributed.
      assert(cols == this->size_[1]); // Columns must be whole.
      assert(c_stride == 1);
      zipm(inout, r_stride, rows, 1, direction);
      if (this->scale_ != T(1.))
	for (length_type i = 0; i != rows; ++i)
	{
	  scale(inout.first + i * r_stride, cols, this->scale_);
	  scale(inout.second + i * r_stride, cols, this->scale_);
	}
    }
    else
    {
      assert(rows == this->size_[0]); // Rows must be whole.
      assert(cols <= this->size_[1]); // OK if columns are distributed.
      assert(r_stride == 1);
      zipm(inout, c_stride, cols, 0, direction);
      if (this->scale_ != T(1.))
	for (length_type i = 0; i != cols; ++i)
	{
	  scale(inout.first + i * c_stride, rows, this->scale_);
	  scale(inout.second + i * c_stride, rows, this->scale_);
	}
    }
  }

  virtual void out_of_place(std::complex<T> *in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    std::complex<T> *out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    if (A == vsip::row)
    {
      assert(rows <= this->size_[0]); // OK if rows are distributed.
      assert(cols == this->size_[1]); // Columns must be whole.
      assert(in_c_stride == 1 && out_c_stride == 1);
      copm(in, in_r_stride, out, out_r_stride, rows, 1, direction);
      if (this->scale_ != T(1.))
	for (length_type i = 0; i != rows; ++i)
	  scale(out + i * out_r_stride, cols, this->scale_);
    }
    else
    {
      assert(rows == this->size_[0]); // Rows must be whole.
      assert(cols <= this->size_[1]); // OK if columns are distributed.
      assert(in_r_stride == 1 && out_r_stride == 1);
      copm(in, in_c_stride, out, out_c_stride, cols, 0, direction);
      if (this->scale_ != T(1.))
	for (length_type i = 0; i != cols; ++i)
	  scale(out + i * out_c_stride, rows, this->scale_);
    }
  }
  virtual void out_of_place(std::pair<T *, T *> in,
			    stride_type in_r_stride, stride_type in_c_stride,
			    std::pair<T *, T *> out,
			    stride_type out_r_stride, stride_type out_c_stride,
			    length_type rows, length_type cols)
  {
    if (A == vsip::row)
    {
      assert(rows <= this->size_[0]); // OK if rows are distributed.
      assert(cols == this->size_[1]); // Columns must be whole.
      assert(in_c_stride == 1 && out_c_stride == 1);
      zopm(in, in_r_stride, out, out_r_stride, rows, 1, direction);
      if (this->scale_ != T(1.))
	for (length_type i = 0; i != rows; ++i)
	{
	  scale(out.first + i * out_r_stride, cols, this->scale_);
	  scale(out.second + i * out_r_stride, cols, this->scale_);
	}
    }
    else
    {
      assert(rows == this->size_[0]); // Rows must be whole.
      assert(cols <= this->size_[1]); // OK if columns are distributed.
      assert(in_r_stride == 1 && out_r_stride == 1);
      zopm(in, in_c_stride, out, out_c_stride, cols, 0, direction);
      if (this->scale_ != T(1.))
	for (length_type i = 0; i != cols; ++i)
	{
	  scale(out.first + i * out_c_stride, rows, this->scale_);
	  scale(out.second + i * out_c_stride, rows, this->scale_);
	}
    }
  }
};

#define FFT_DEF(D, I, O, S)				                     \
template <>                                                                  \
std::auto_ptr<fft::Fft_backend<D, I, O, S> >				     \
create(Domain<D> const &dom, fft::Fft_backend<D, I, O, S>::scalar_type scale)\
{                                                                            \
  return std::auto_ptr<fft::Fft_backend<D, I, O, S> >			     \
    (new Fft_impl<D, I, O, S>(dom, scale));				     \
}

#if VSIP_IMPL_HAVE_SAL_FLOAT
FFT_DEF(1, float, std::complex<float>, 0)
FFT_DEF(1, std::complex<float>, float, 0)
FFT_DEF(1, std::complex<float>, std::complex<float>, fft_fwd)
FFT_DEF(1, std::complex<float>, std::complex<float>, fft_inv)
#endif

#if VSIP_IMPL_HAVE_SAL_DOUBLE
FFT_DEF(1, double, std::complex<double>, 0)
FFT_DEF(1, std::complex<double>, double, 0)
FFT_DEF(1, std::complex<double>, std::complex<double>, fft_fwd)
FFT_DEF(1, std::complex<double>, std::complex<double>, fft_inv)
#endif

#if VSIP_IMPL_HAVE_SAL_FLOAT
FFT_DEF(2, float, std::complex<float>, 0)
FFT_DEF(2, float, std::complex<float>, 1)
FFT_DEF(2, std::complex<float>, float, 0)
FFT_DEF(2, std::complex<float>, float, 1)
FFT_DEF(2, std::complex<float>, std::complex<float>, fft_fwd)
FFT_DEF(2, std::complex<float>, std::complex<float>, fft_inv)
#endif

#if VSIP_IMPL_HAVE_SAL_DOUBLE
FFT_DEF(2, double, std::complex<double>, 0)
FFT_DEF(2, double, std::complex<double>, 1)
FFT_DEF(2, std::complex<double>, double, 0)
FFT_DEF(2, std::complex<double>, double, 1)
FFT_DEF(2, std::complex<double>, std::complex<double>, fft_fwd)
FFT_DEF(2, std::complex<double>, std::complex<double>, fft_inv)
#endif

#undef FFT_DEF

#define FFTM_DEF(I, O, A, D)				          \
template <>                                                       \
std::auto_ptr<fft::Fftm_backend<I, O, A, D> >		     	  \
create(Domain<2> const &dom, impl::scalar_of<I>::type scale)      \
{                                                                 \
  return std::auto_ptr<fft::Fftm_backend<I, O, A, D> >	       	  \
    (new Fftm_impl<I, O, A, D>(dom, scale));	                  \
}

#if VSIP_IMPL_HAVE_SAL_FLOAT
FFTM_DEF(float, std::complex<float>, 0, fft_fwd)
FFTM_DEF(float, std::complex<float>, 1, fft_fwd)
FFTM_DEF(std::complex<float>, float, 0, fft_inv)
FFTM_DEF(std::complex<float>, float, 1, fft_inv)
FFTM_DEF(std::complex<float>, std::complex<float>, 0, fft_fwd)
FFTM_DEF(std::complex<float>, std::complex<float>, 1, fft_fwd)
FFTM_DEF(std::complex<float>, std::complex<float>, 0, fft_inv)
FFTM_DEF(std::complex<float>, std::complex<float>, 1, fft_inv)
#endif

#if VSIP_IMPL_HAVE_SAL_DOUBLE
FFTM_DEF(double, std::complex<double>, 0, fft_fwd)
FFTM_DEF(double, std::complex<double>, 1, fft_fwd)
FFTM_DEF(std::complex<double>, double, 0, fft_inv)
FFTM_DEF(std::complex<double>, double, 1, fft_inv)
FFTM_DEF(std::complex<double>, std::complex<double>, 0, fft_fwd)
FFTM_DEF(std::complex<double>, std::complex<double>, 1, fft_fwd)
FFTM_DEF(std::complex<double>, std::complex<double>, 0, fft_inv)
FFTM_DEF(std::complex<double>, std::complex<double>, 1, fft_inv)
#endif

#undef FFTM_DEF

} // namespace vsip::impl::sal
} // namespace vsip::impl
} // namespace vsip
