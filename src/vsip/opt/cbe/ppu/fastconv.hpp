/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_OPT_CBE_PPU_FASTCONV_H
#define VSIP_OPT_CBE_PPU_FASTCONV_H

#include <vsip/core/allocation.hpp>
#include <vsip/core/config.hpp>
#include <vsip/dda.hpp>
#include <vsip/opt/cbe/fconv_params.h>
#include <vsip/opt/cbe/ppu/bindings.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
extern "C"
{
#include <libspe2.h>
}

namespace vsip
{
namespace impl
{
namespace cbe
{
void fconv(complex<float> const *in,
	   complex<float> const *kernel,
	   complex<float> *out,
	   length_type rows,
	   length_type cols,
	   bool transform_kernels);
void fconvm(complex<float> const *in,
	    complex<float> const *kernel,
	    complex<float> *out,
	    length_type rows,
	    length_type cols,
	    bool transform_kernels);
void fconv(std::pair<float const *,float const *> in,
	   std::pair<float const *,float const *> kernel,
	   std::pair<float*,float*> out,
	   length_type rows,
	   length_type cols,
	   bool transform_kernels);
void fconvm(std::pair<float const *,float const *> in,
	    std::pair<float const *,float const *> kernel,
	    std::pair<float*,float*> out,
	    length_type rows,
	    length_type cols,
	    bool transform_kernels);

template <typename T, storage_format_type C> 
struct Fastconv_traits;

template <>
struct Fastconv_traits<complex<float>, interleaved_complex>
{
  static length_type const min_size = VSIP_IMPL_MIN_FCONV_SIZE;
  static length_type const max_size = VSIP_IMPL_MAX_FCONV_SIZE;
};

template <>
struct Fastconv_traits<complex<float>, split_complex>
{
  static length_type const min_size = VSIP_IMPL_MIN_FCONV_SPLIT_SIZE;
  static length_type const max_size = VSIP_IMPL_MAX_FCONV_SPLIT_SIZE;
};

/// Fast convolution base class.
///
/// Template parameters:
///
///  :D: For D==1 a 1D kernel is applied row-wise,
///      for D==2, kernel and input matrices are convolved row-wise.
///  :T: The value-type.
///  :F: the complex format (interleaved_complex or split_complex)
template <dimension_type D,
          typename T,
	  storage_format_type C = interleaved_complex>
class Fastconv;

template <typename T, storage_format_type C>
class Fastconv<1, T, C>
{
  typedef Fastconv_traits<T, C> traits;
  typedef Layout<1, row1_type, dense, C> layout1_type;
  typedef Layout<2, row2_type, dense, C> layout2_type;
  typedef layout1_type kernel_layout_type;
  typedef Strided<1, T, kernel_layout_type, Local_map> kernel_block_type;
  typedef Vector<T, kernel_block_type> kernel_view_type;

public:
  template <typename Block>
  Fastconv(Vector<T, Block> coeffs,
	   length_type input_size,
	   bool transform_kernel = true) VSIP_THROW((std::bad_alloc))
    : size_(input_size),
      kernel_(input_size),
      transform_kernel_(transform_kernel)
  {
    assert(coeffs.size(0) <= this->size());
    if (transform_kernel)
    {
      kernel_ = T();
      kernel_(view_domain(coeffs.local())) = coeffs.local();
    }
    else
      kernel_ = coeffs.local();
  }
  ~Fastconv() VSIP_NOTHROW {}

  length_type size() { return size_;}
  static bool is_size_valid(length_type size)
  {
    return (size >= traits::min_size &&
	    size <= traits::max_size &&
	    fft::is_power_of_two(size));
  }

  template <typename B1, typename B2>
  Vector<T, B2>
  operator()(const_Vector<T, B1> in, Vector<T, B2> out) VSIP_NOTHROW
  {
    assert(in.size() == this->size());
    assert(out.size() == this->size());
    convolve(in.local(), out.local());
    return out;
  }

  template <typename B1, typename B2>
  Matrix<T, B2>
  operator()(const_Matrix<T, B1> in, Matrix<T, B2> out) VSIP_NOTHROW
  {
    assert(in.size(1) == this->size());
    assert(out.size(1) == this->size());
    convolve(in.local(), out.local());
    return out;
  }

private:
  template <typename B1, typename B2>
  void
  convolve(const_Vector<T, B1> in, Vector<T, B2> out)
  {
    dda::Data<B1, dda::in, layout1_type> data_in(in.block());
    dda::Data<kernel_block_type, dda::in> data_kernel(kernel_.block());
    dda::Data<B2, dda::out, layout1_type> data_out(out.block());
    assert(data_in.stride(0) == 1);
    assert(data_kernel.stride(0) == 1);
    assert(data_out.stride(0) == 1);

    fconv(data_in.ptr(), data_kernel.ptr(), data_out.ptr(), 1, out.size(),
	  transform_kernel_);
  }

  template <typename B1, typename B2>
  void
  convolve(const_Matrix<T, B1> in, Matrix<T, B2> out)
  {
    dda::Data<B1, dda::in, layout2_type> data_in(in.block());
    dda::Data<kernel_block_type, dda::in> data_kernel(kernel_.block());
    dda::Data<B2, dda::out, layout2_type> data_out(out.block());
    assert(data_in.stride(1) == 1);
    assert(data_kernel.stride(0) == 1);
    assert(data_out.stride(1) == 1);

    fconv(data_in.ptr(), data_kernel.ptr(), data_out.ptr(), out.size(0), out.size(1),
	  transform_kernel_);
  }

  length_type size_;
  kernel_view_type kernel_;
  bool transform_kernel_;
};

template <typename T, storage_format_type C>
class Fastconv<2, T, C>
{
  typedef Fastconv_traits<T, C> traits;
  typedef Layout<2, row2_type, dense, C> layout_type;
  typedef Strided<2, T, layout_type, Local_map> kernel_block_type;
  typedef Matrix<T, kernel_block_type> kernel_view_type;

public:
  template <typename B>
  Fastconv(Matrix<T, B> coeffs,
	   length_type size,
	   bool transform_kernel = true) VSIP_THROW((std::bad_alloc))
    : size_(size),
      kernel_(coeffs.local().size(0), size),
      transform_kernel_(transform_kernel)
  {
    assert(coeffs.size(1) <= this->size());
    if (transform_kernel)
    {
      kernel_ = T();
      kernel_(view_domain(coeffs.local())) = coeffs.local();
    }
    else
      kernel_ = coeffs.local();
  }
  ~Fastconv() VSIP_NOTHROW {}

  length_type size() { return size_;}
  static bool is_size_valid(length_type size)
  {
    return (size >= traits::min_size &&
	    size <= traits::max_size &&
	    fft::is_power_of_two(size));
  }

  template <typename B1, typename B2>
  Matrix<T, B2>
  operator()(const_Matrix<T, B1> in, Matrix<T, B2> out) VSIP_NOTHROW
  {
    assert(in.size(1) == this->size());
    assert(out.size(1) == this->size());
    convolve(in.local(), out.local());
    return out;
  }

private:
  template <typename B1, typename B2>
  void
  convolve(const_Matrix<T, B1> in, Matrix<T, B2> out) VSIP_NOTHROW
  {
    dda::Data<B1, dda::in, layout_type> data_in(in.block());
    dda::Data<kernel_block_type, dda::in> data_kernel(kernel_.block());
    dda::Data<B2, dda::out, layout_type> data_out(out.block());
    assert(data_in.stride(1) == 1);
    assert(data_kernel.stride(1) == 1);
    assert(data_out.stride(1) == 1);

    fconvm(data_in.ptr(), data_kernel.ptr(), data_out.ptr(), in.size(0), in.size(1),
	   transform_kernel_);
  }

  length_type size_;
  kernel_view_type kernel_;
  bool transform_kernel_;
};

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip

#endif
