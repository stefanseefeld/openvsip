/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. 

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef vsip_opt_cuda_device_storage_hpp_
#define vsip_opt_cuda_device_storage_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/core/complex_decl.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/dda_pointer.hpp>
#ifndef NDEBUG
# include <iostream>
#endif

extern "C"
{
#if !defined(__CUDA_RUNTIME_API_H__)
// From cuda_runtime_api.h
//
enum cudaError
{ 
  cudaSuccess = 0
};
typedef cudaError cudaError_t;

enum cudaMemcpyKind
{
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice,
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice
};

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
		       enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, 
			 size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyToSymbol(char const *symbol, void const *src, size_t count, 
			       size_t offset, enum cudaMemcpyKind kind);
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaGetLastError(void);
const char* cudaGetErrorString(cudaError_t error);
cudaError_t cudaThreadSynchronize(void);

#endif // !defined(__CUDA_RUNTIME_API_H__)
}

#ifndef NDEBUG
# define ASSERT_CUDA_OK()					\
{								\
  cudaError_t error = cudaGetLastError();			\
  if (error != cudaSuccess)					\
  {								\
    std::cerr << "CUDA problem encountered (error "		\
	      << error << ")" << std::endl;			\
    std::cerr << cudaGetErrorString(error) << std::endl;	\
  }								\
  assert(error == cudaSuccess);					\
}
#else
# define ASSERT_CUDA_OK()
#endif

namespace vsip
{
namespace impl
{
namespace cuda
{

// Convenience wrappers around cudaMemcpy
template <typename T>
void cp_to_dev(T *dest, T const *src, length_type size)
{
  namespace p = vsip::impl::profile;
  p::Scope<p::copy>("cuda::copy_to_device", size * sizeof(T));
  cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyHostToDevice);
  ASSERT_CUDA_OK();
}
template <typename T>
void cp_to_host(T *dest, T const *src, length_type size)
{
  namespace p = vsip::impl::profile;
  p::Scope<p::copy>("cuda::copy_to_host", size * sizeof(T));
  cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyDeviceToHost);
  ASSERT_CUDA_OK();
}
template <typename T>
void cp_to_dev(T *dest, stride_type ds, T const *src, stride_type ss,
	       length_type width, length_type height)
{
  namespace p = vsip::impl::profile;
  p::Scope<p::copy>("cuda::copy_to_device", width * height * sizeof(T));
  cudaMemcpy2D(dest, ds * sizeof(T), src, ss * sizeof(T),
	       width * sizeof(T), height, cudaMemcpyHostToDevice);
  ASSERT_CUDA_OK();
}
template <typename T>
void cp_to_host(T *dest, stride_type ds, T const *src, stride_type ss,
		length_type width, length_type height)
{
  namespace p = vsip::impl::profile;
  p::Scope<p::copy>("cuda::copy_to_host", width * height * sizeof(T));
  cudaMemcpy2D(dest, ds * sizeof(T), src, ss * sizeof(T),
	       width * sizeof(T), height, cudaMemcpyDeviceToHost);
  ASSERT_CUDA_OK();
}

template <typename T>
class Rt_device_storage_base
{
public:
  typedef dda::impl::Pointer<T> type;
  typedef dda::impl::const_Pointer<T> const_type;

  enum state_type
  {
    alloc_data,
    user_data,
    no_data
  };

  static dda::impl::Pointer<T> allocate_(storage_format_type cformat,
                                         length_type size)
  {
    void* ptr;

    if (!is_complex<T>::value || cformat == interleaved_complex)
    {
      namespace p = vsip::impl::profile;
      p::event<p::memory>("cuda::Device_storage()", size * sizeof(T));
      cudaError_t error = cudaMalloc((void**)&ptr, size * sizeof(T));
      ASSERT_CUDA_OK();
      if (error != cudaSuccess)
	VSIP_IMPL_THROW(std::bad_alloc());

      return dda::impl::Pointer<T>(static_cast<T*>(ptr));
    }
    else
    {
      typedef typename scalar_of<T>::type scalar_type;

      namespace p = vsip::impl::profile;
      p::event<p::memory>("cuda::Device_storage()", size * sizeof(T));
      cudaError_t error = cudaMalloc((void**)&ptr, size * 2 * sizeof(scalar_type));
      ASSERT_CUDA_OK();
      if (error != cudaSuccess) VSIP_IMPL_THROW(std::bad_alloc());

      return dda::impl::Pointer<T>(std::pair<scalar_type*, scalar_type*>(
	static_cast<scalar_type*>(ptr), static_cast<scalar_type*>(ptr) + size));
    }
  }

  static void deallocate_(dda::impl::Pointer<T>   ptr,
			  storage_format_type cformat,
			  length_type     size)
  {
    namespace p = vsip::impl::profile;
    p::event<p::memory>("cuda::~Device_storage()", size * sizeof(T));
    cudaFree(ptr.as_inter());
    ASSERT_CUDA_OK();
  }

  // Potentially "split" an interleaved buffer into two segments
  // of equal size, if split format is requested and user provides
  // an interleaved buffer.
  static dda::impl::Pointer<T> partition_(storage_format_type cformat,
					  dda::impl::Pointer<T>   buffer,
					  length_type     size)
  {
    if (cformat == split_complex && is_complex<T>::value &&
	buffer.as_split().second == 0)
    {
      // We're allocating split-storage but user gave us
      // interleaved storage.
      typedef typename scalar_of<T>::type scalar_type;
      scalar_type* ptr = reinterpret_cast<scalar_type*>(buffer.as_inter());
      return dda::impl::Pointer<T>(std::pair<scalar_type*, scalar_type*>(
			     ptr, ptr + size));
    }
    else
    {
      // Check that user didn't give us split-storage when we wanted
      // interleaved.  We can't fix this, but we can throw an
      // exception.
      assert(!(cformat == interleaved_complex && is_complex<T>::value &&
	       buffer.as_split().second != 0));
      return buffer;
    }
  }

  Rt_device_storage_base(length_type size,
                      storage_format_type cformat,
                      type buffer = type())
    : size_(size),
      cformat_(cformat),
      state_(size == 0         ? no_data   :
             buffer.is_null()  ? alloc_data
                               : user_data),
      data_(state_ == alloc_data ? allocate_ (cformat_, size) :
            state_ == user_data  ? partition_(cformat, buffer, size)
                                 : type())
  {}

  ~Rt_device_storage_base()
  {
    if (state_ == alloc_data)
    {
      deallocate_(data_, cformat_, size_);
      data_ = type();
    }
  }

  void resize(length_type size)
  {
    namespace p = vsip::impl::profile;
    if (state_ == alloc_data)
    {
      deallocate_(data_, cformat_, size_);
    }
    size_ = size;
    data_ = allocate_(cformat_, size);
  }

  void deallocate(length_type size)
  {
    if (state_ == alloc_data)
    {
      deallocate_(data_, cformat_, size);
      data_ = type();
    }
  }

  bool is_alloc() const {return state_ == alloc_data;}

  length_type total_size() const { return size_;}
  type ptr() { return data_;}
  const_type ptr() const { return const_type(data_);}
  storage_format_type storage_format() const { return cformat_;}

  void from_host(const_type src) { cp_to_dev(data_.as_inter(), src.as_inter(), size_);}
  void to_host(type dest) const { cp_to_host(dest.as_inter(), data_.as_inter(), size_);}

private:
  length_type size_;
  storage_format_type cformat_;
  state_type state_;
  type data_;
};

/// Common base class for Device_storage.
/// Device storage is always dense.
/// In case of split-complex data, the two planes are placed
/// next to each other. This makes it easier to convert between 
/// split and interleaved data.
/// The storage is only valid for size > 0.
template <typename T, storage_format_type C = interleaved_complex>
class Device_storage_base
{
public:
  typedef T *type;
  typedef T const *const_type;

  Device_storage_base(length_type size)
    : size_(size),
      data_(0)
  {
    if (size_)
    {
      namespace p = vsip::impl::profile;
      p::event<p::memory>("cuda::Device_storage()", size_ * sizeof(T));
      cudaError_t error = cudaMalloc((void**)&data_, size_ * sizeof(T));
      ASSERT_CUDA_OK();
      if (error != cudaSuccess)
	VSIP_IMPL_THROW(std::bad_alloc());
    }
  }

  ~Device_storage_base()
  {
    if (size_)
    {
      namespace p = vsip::impl::profile;
      p::event<p::memory>("cuda::~Device_storage()", size_ * sizeof(T));
      cudaFree(data_);
      ASSERT_CUDA_OK();
    }
  }

  void resize(length_type size)
  {
    namespace p = vsip::impl::profile;
    if (size_)
    {
      p::event<p::memory>("cuda::Device_storage::resize() free", size_ * sizeof(T));
      cudaFree(data_);
      ASSERT_CUDA_OK();
    }
    size_ = size;
    p::event<p::memory>("cuda::Device_storage::resize() alloc", size_ * sizeof(T));
    cudaError_t error = cudaMalloc((void**)&data_, size_ * sizeof(T));
    ASSERT_CUDA_OK();
    if (error != cudaSuccess) VSIP_IMPL_THROW(std::bad_alloc());
  }

  length_type total_size() const { return size_;}
  type ptr() { return data_;}
  const_type ptr() const { return data_;}

  void from_host(const_type src) { cp_to_dev(data_, src, size_);}
  void to_host(type dest) const { cp_to_host(dest, data_, size_);}

private:
  length_type size_;
  T *data_;
};

template <typename T>
class Device_storage_base<vsip::complex<T>, split_complex>
{
public:
  typedef std::pair<T*,T*> type;
  typedef std::pair<T const*, T const*> const_type;

  Device_storage_base(length_type size)
    : size_(size)
  {
    if (size_)
    {
      namespace p = vsip::impl::profile;
      p::event<p::memory>("cuda::Device_storage()", size_ * sizeof(T));
      cudaError_t error = cudaMalloc((void**)&data_.first, size * 2 * sizeof(T));
      ASSERT_CUDA_OK();
      if (error != cudaSuccess) VSIP_IMPL_THROW(std::bad_alloc());
      data_.second = data_.first + size_;
    }
  }

  ~Device_storage_base()
  {
    if (size_)
    {
      namespace p = vsip::impl::profile;
      p::event<p::memory>("cuda::~Device_storage()", size_ * sizeof(T));
      cudaFree(data_.first);
      ASSERT_CUDA_OK();
    }
  }

  void resize(length_type size)
  {
    namespace p = vsip::impl::profile;
    if (size_)
    {
      p::event<p::memory>("cuda::~Device_storage::resize() free", size_ * sizeof(T));
      cudaFree(data_.first);
      ASSERT_CUDA_OK();
    }
    size_ = size;
    p::event<p::memory>("cuda::~Device_storage::resize() alloc", size_ * sizeof(T));
    cudaError_t error = cudaMalloc((void**)&data_.first, size_ * 2 * sizeof(T));
    ASSERT_CUDA_OK();
    if (error != cudaSuccess) VSIP_IMPL_THROW(std::bad_alloc());
    data_.second = data_.first + size_;
  }

  length_type total_size() const { return size_;}
  type ptr() { return data_;}
  const_type ptr() const { return std::make_pair(data_.first, data_.second);}

  void from_host(const_type src)
  {
    cp_to_dev(data_.first, src.first, size_);
    cp_to_dev(data_.second, src.second, size_);
  }
  void to_host(type dest) const
  {
    cp_to_host(dest.first, data_.first, size_);
    cp_to_host(dest.second, data_.second, size_);
  }

private:
  length_type size_;
  std::pair<T*,T*> data_;
};

/// Device_storage manages data storage on a GPU.
/// Template parameters:
///
///   :T: The value-type
///   :L: The layout in host-memory. For dense data
///       this is the same as the layout in device-memory.
///       For non-dense data (such as stride-unit-align) the
///       data is packed densely when transferred to the device.
template <typename T, typename L>
class Device_storage;

/// Device storage for 1D data.
/// (For 1D storage the pack-type doesn't matter.)
template <typename T, pack_type P, storage_format_type C>
class Device_storage<T, Layout<1, tuple<0,1,2>, P, C> >
  : public Device_storage_base<T, C>
{
  typedef Layout<1, tuple<0,1,2>, P, C> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  Device_storage(applied_layout_type const &l) VSIP_NOTHROW
  : Device_storage_base<T, C>(l.total_size()) {}

  stride_type stride(dimension_type) const VSIP_NOTHROW { return 1;}
  length_type size(dimension_type) const VSIP_NOTHROW { return this->total_size();}
};

/// Device storage for 2D dense data.
template <typename T, storage_format_type C>
class Device_storage<T, Layout<2, row2_type, dense, C> >
  : public Device_storage_base<T, C>
{
  typedef Layout<2, row2_type, dense, C> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  Device_storage(applied_layout_type const &l) VSIP_NOTHROW
  : Device_storage_base<T, C>(l.total_size()), layout_(l) {}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return layout_.stride(d);}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return layout_.size(d);}

private:
  applied_layout_type layout_;
};

/// Specialization needed for disambiguation only.
template <typename T, storage_format_type C>
class Device_storage<T, Layout<2, col2_type, dense, C> >
  : public Device_storage_base<T, C>
{
  typedef Layout<2, col2_type, dense, C> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  Device_storage(applied_layout_type const &l) VSIP_NOTHROW
  : Device_storage_base<T, C>(l.total_size()), layout_(l) {}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return layout_.stride(d);}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return layout_.size(d);}

private:
  applied_layout_type layout_;
};

/// Specialization needed for disambiguation only.
template <typename T>
class Device_storage<complex<T>, 
  Layout<2, row2_type, dense, interleaved_complex> >
  : public Device_storage_base<complex<T>, interleaved_complex>
{
  typedef Layout<2, row2_type, dense, interleaved_complex> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  Device_storage(applied_layout_type const &l) VSIP_NOTHROW
  : Device_storage_base<complex<T>, interleaved_complex>(l.total_size()), layout_(l) {}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return layout_.stride(d);}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return layout_.size(d);}

private:
  applied_layout_type layout_;
};

/// Specialization needed for disambiguation only.
template <typename T>
class Device_storage<complex<T>, 
  Layout<2, col2_type, dense, interleaved_complex> >
  : public Device_storage_base<complex<T>, interleaved_complex>
{
  typedef Layout<2, col2_type, dense, interleaved_complex> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  Device_storage(applied_layout_type const &l) VSIP_NOTHROW
  : Device_storage_base<complex<T>, interleaved_complex>(l.total_size()), layout_(l) {}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return layout_.stride(d);}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return layout_.size(d);}

private:
  applied_layout_type layout_;
};

/// Specialization needed for disambiguation only.
template <typename T>
class Device_storage<complex<T>, 
  Layout<2, row2_type, dense, split_complex> >
  : public Device_storage_base<complex<T>, split_complex>
{
  typedef Layout<2, row2_type, dense, split_complex> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  Device_storage(applied_layout_type const &l) VSIP_NOTHROW
  : Device_storage_base<complex<T>, split_complex>(l.total_size()), layout_(l) {}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return layout_.stride(d);}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return layout_.size(d);}

private:
  applied_layout_type layout_;
};

/// Specialization needed for disambiguation only.
template <typename T>
class Device_storage<complex<T>, 
  Layout<2, col2_type, dense, split_complex> >
  : public Device_storage_base<complex<T>, split_complex>
{
  typedef Layout<2, col2_type, dense, split_complex> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  Device_storage(applied_layout_type const &l) VSIP_NOTHROW
  : Device_storage_base<complex<T>, split_complex>(l.total_size()), layout_(l) {}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return layout_.stride(d);}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return layout_.size(d);}

private:
  applied_layout_type layout_;
};

/// Device storage for 3D dense data.
template <typename T, typename O, storage_format_type C>
class Device_storage<T, Layout<3, O, dense, C> >
  : public Device_storage_base<T, C>
{
  typedef Layout<3, O, dense, C> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  Device_storage(applied_layout_type const &l) VSIP_NOTHROW
  : Device_storage_base<T, C>(l.total_size()), layout_(l) {}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return layout_.stride(d);}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return layout_.size(d);}

private:
  applied_layout_type layout_;
};

/// Specialization needed for disambiguation only.
template <typename T, typename O>
class Device_storage<complex<T>, Layout<3, O, dense, interleaved_complex> >
  : public Device_storage_base<complex<T>, interleaved_complex>
{
  typedef Layout<3, O, dense, interleaved_complex> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  Device_storage(applied_layout_type const &l) VSIP_NOTHROW
  : Device_storage_base<complex<T>, interleaved_complex>(l.total_size()), layout_(l) {}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return layout_.stride(d);}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return layout_.size(d);}

private:
  applied_layout_type layout_;
};

/// Specialization needed for disambiguation only.
template <typename T, typename O>
class Device_storage<complex<T>, Layout<3, O, dense, split_complex> >
  : public Device_storage_base<complex<T>, split_complex>
{
  typedef Layout<3, O, dense, split_complex> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  Device_storage(applied_layout_type const &l) VSIP_NOTHROW
  : Device_storage_base<complex<T>, split_complex>(l.total_size()), layout_(l) {}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return layout_.stride(d);}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return layout_.size(d);}

private:
  applied_layout_type layout_;
};

/// Device storage for 2D non-dense row-major data.
template <typename T, pack_type P, storage_format_type C>
class Device_storage<T, Layout<2, row2_type, P, C> >
  : public Device_storage_base<T, C>
{
  typedef Layout<2, row2_type, P, C> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  Device_storage(applied_layout_type const &l) VSIP_NOTHROW
  : Device_storage_base<T, C>(l.size(0)*l.size(1)), host_layout_(l) {}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return d == 0 ? size(1) : 1;}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return host_layout_.size(d);}

  void from_host(T const *src)
  {
    cp_to_dev(this->ptr(), stride(0), src, host_layout_.stride(0),
	      host_layout_.size(1), host_layout_.size(0));
  }
  void to_host(T *dest) const
  {
    cp_to_host(dest, host_layout_.stride(0), this->ptr(), stride(0),
	       host_layout_.size(1), host_layout_.size(0));
  }

private:
  applied_layout_type host_layout_;
};

/// Device storage for 2D non-dense column-major data.
template <typename T, pack_type P, storage_format_type C>
class Device_storage<T, Layout<2, col2_type, P, C> >
  : public Device_storage_base<T, C>
{
  typedef Layout<2, col2_type, P, C> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  Device_storage(applied_layout_type const &l) VSIP_NOTHROW
  : Device_storage_base<T, C>(l.size(0)*l.size(1)), host_layout_(l) {}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return d == 0 ? 1 : size(0);}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return host_layout_.size(d);}


  void from_host(T const *src)
  {
    cp_to_dev(this->ptr(), stride(1), src, host_layout_.stride(1),
	      host_layout_.size(0), host_layout_.size(1));
  }
  void to_host(T *dest) const
  {
    cp_to_host(dest, host_layout_.stride(1), this->ptr(), stride(1),
	       host_layout_.size(0), host_layout_.size(1));
  }

private:
  applied_layout_type host_layout_;
};

/// Device storage for 2D non-dense row-major split-complex data.
template <typename T, pack_type P>
class Device_storage<complex<T>, Layout<2, row2_type, P, split_complex> >
  : public Device_storage_base<complex<T>, split_complex>
{
  typedef Device_storage_base<complex<T>, split_complex> base_type;
  typedef Layout<2, row2_type, P, split_complex> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  Device_storage(applied_layout_type const &l) VSIP_NOTHROW
  : base_type(l.size(0)*l.size(1)), host_layout_(l) {}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return d == 0 ? size(1) : 1;}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return host_layout_.size(d);}

  void from_host(std::pair<T const*, T const*> src)
  {
    std::pair<T*,T*> data = base_type::ptr();
    cp_to_dev(data.first, stride(0), src.first, host_layout_.stride(0),
	      host_layout_.size(1), host_layout_.size(0));
    cp_to_dev(data.second, stride(0), src.second, host_layout_.stride(0),
	      host_layout_.size(1), host_layout_.size(0));
  }
  void to_host(std::pair<T*,T*> dest) const
  {
    std::pair<T const*,T const*> data = base_type::ptr();
    cp_to_host(dest.first, host_layout_.stride(0), data.first, stride(0),
	       host_layout_.size(1), host_layout_.size(0));
    cp_to_host(dest.second, host_layout_.stride(0), data.second, stride(0),
	       host_layout_.size(1), host_layout_.size(0));
  }

private:
  applied_layout_type host_layout_;
};

/// Device storage for 2D non-dense column-major split-complex data.
template <typename T, pack_type P>
class Device_storage<complex<T>, Layout<2, col2_type, P, split_complex> >
  : public Device_storage_base<complex<T>, split_complex>
{
  typedef Device_storage_base<complex<T>, split_complex> base_type;
  typedef Layout<2, col2_type, P, split_complex> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  Device_storage(applied_layout_type const &l) VSIP_NOTHROW
  : base_type(l.size(0)*l.size(1)), host_layout_(l) {}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return d == 0 ? 1 : size(0);}
  length_type size(dimension_type d) const VSIP_NOTHROW
  { return host_layout_.size(d);}

  void from_host(std::pair<T const *, T const *> src)
  {
    std::pair<T*,T*> data = base_type::ptr();
    cp_to_dev(data.first, stride(1), src.first, host_layout_.stride(1),
	      host_layout_.size(0), host_layout_.size(1));
    cp_to_dev(data.second, stride(1), src.second, host_layout_.stride(1),
	      host_layout_.size(0), host_layout_.size(1));
  }
  void to_host(std::pair<T*,T*> dest) const
  {
    std::pair<T const*,T const*> data = base_type::ptr();
    cp_to_host(dest.first, host_layout_.stride(1), data.first, stride(1),
	       host_layout_.size(0), host_layout_.size(1));
    cp_to_host(dest.second, host_layout_.stride(1), data.second, stride(1),
	       host_layout_.size(0), host_layout_.size(1));
  }

private:
  applied_layout_type host_layout_;
};

/// Device storage for 1D rt-laid-out data.
template <typename T>
class Device_storage<T, Rt_layout<1> > : Rt_device_storage_base<T>
{
  typedef Rt_device_storage_base<T> base_type;
  typedef Rt_layout<1> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  typedef dda::impl::Pointer<T> type;
  typedef dda::impl::const_Pointer<T> const_type;
  typedef typename scalar_of<T>::type scalar_type;

  Device_storage(applied_layout_type const &l, bool use_direct = false, type buffer = type()) VSIP_NOTHROW
  : base_type(use_direct ? 0 : l.total_size(), l.storage_format(), buffer), layout_(l)
  {
    assert(layout_.total_size() == 0 || layout_.stride(0) == 1);
  }

  using base_type::total_size;

  length_type size(dimension_type) const { return total_size();}
  stride_type stride(dimension_type) const { return 1;}
  storage_format_type storage_format() const { return base_type::storage_format();}
  type ptr() { return type(base_type::ptr());}
  const_type ptr() const { return const_type(base_type::ptr());}

  void from_host(const_type src)
  {
    if (!is_complex<T>::value)
    {
      cp_to_dev(ptr().as_real(), src.as_real(), total_size());
    }
    else if (storage_format() == split_complex)
    {
      std::pair<scalar_type const*,scalar_type const*> host = src.as_split();
      std::pair<scalar_type*,scalar_type*> dev = ptr().as_split();
      cp_to_dev(dev.first, host.first, total_size());
      cp_to_dev(dev.second, host.second, total_size());
    }
    else
    {
      cp_to_dev(ptr().as_inter(), src.as_inter(), total_size());
    }
  }

  void to_host(type dest) const
  {
    if (!is_complex<T>::value)
    {
      cp_to_host(dest.as_real(), ptr().as_real(), total_size());
    }
    else if (storage_format() == split_complex)
    {
      std::pair<scalar_type const*,scalar_type const*> dev = ptr().as_split();
      std::pair<scalar_type*,scalar_type*> host = dest.as_split();
      cp_to_host(host.first, dev.first, total_size());
      cp_to_host(host.second, dev.second, total_size());
    }
    else
    {
      cp_to_host(dest.as_inter(), ptr().as_inter(), total_size());
    }
  }

private:
  applied_layout_type layout_;
};

/// Device storage for 2D rt-laid-out data.
template <typename T>
class Device_storage<T, Rt_layout<2> > : Rt_device_storage_base<T>
{
  typedef Rt_device_storage_base<T> base_type;
  typedef Rt_layout<2> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  typedef dda::impl::Pointer<T> type;
  typedef dda::impl::const_Pointer<T> const_type;
  typedef typename scalar_of<T>::type scalar_type;

  Device_storage(applied_layout_type const &l, bool use_direct = false, type buffer = type()) VSIP_NOTHROW
  : base_type(use_direct ? 0 : l.size(0)*l.size(1), l.storage_format(), buffer), layout_(l)
  {
    // Only dense layouts are valid in this context.
    assert(layout_.total_size() == 0 ||
	   (layout_.stride(1) == 1 &&
	    static_cast<length_type>(layout_.stride(0)) == layout_.size(1)) ||
	   (layout_.stride(0) == 1 &&
	    static_cast<length_type>(layout_.stride(1)) == layout_.size(0)));
  }

  using base_type::total_size;
  length_type size(dimension_type d) const { return layout_.size(d);}
  stride_type stride(dimension_type d) const { return layout_.stride(d);}
  storage_format_type storage_format() const { return base_type::storage_format();}
  type ptr() { return type(base_type::ptr());}
  const_type ptr() const { return const_type(base_type::ptr());}

  void from_host(const_type src)
  {
    if (layout_.stride(1) == 1)  // row major
    {
      if (!is_complex<T>::value)
      {
        cp_to_dev(ptr().as_real(), stride(0), src.as_real(), layout_.stride(0),
                  layout_.size(1), layout_.size(0));
      }
      else if (storage_format() == split_complex)
      {
        std::pair<scalar_type const*,scalar_type const*> host = src.as_split();
        std::pair<scalar_type*,scalar_type*> dev = ptr().as_split();

	cp_to_dev(dev.first, stride(0), host.first, layout_.stride(0),
		  layout_.size(1), layout_.size(0));
	cp_to_dev(dev.second, stride(0), host.second, layout_.stride(0),
		  layout_.size(1), layout_.size(0));
      }
      else
      {
	cp_to_dev(ptr().as_inter(), stride(0), src.as_inter(), layout_.stride(0),
		  layout_.size(1), layout_.size(0));
      }
    }
    else                     // column major
    {
      if (!is_complex<T>::value)
      {
        cp_to_dev(ptr().as_real(), stride(1), src.as_real(), layout_.stride(1),
                  layout_.size(0), layout_.size(1));
      }
      else if (storage_format() == split_complex)
      {
        std::pair<scalar_type const*,scalar_type const*> host = src.as_split();
        std::pair<scalar_type*,scalar_type*> dev = ptr().as_split();

	cp_to_dev(dev.first, stride(1), host.first, layout_.stride(1),
		  layout_.size(0), layout_.size(1));
	cp_to_dev(dev.second, stride(1), host.second, layout_.stride(1),
		  layout_.size(0), layout_.size(1));
      }
      else
      {
	cp_to_dev(ptr().as_inter(), stride(1), src.as_inter(), layout_.stride(1),
		  layout_.size(0), layout_.size(1));
      }
    }
  }
  void to_host(type dest) const
  {
    if (layout_.stride(1) == 1)  // row major
    {
      if (!is_complex<T>::value)
      {
        cp_to_host(dest.as_real(), layout_.stride(0), ptr().as_real(), stride(0),
		   layout_.size(1), layout_.size(0));
      }
      else if (storage_format() == split_complex)
      {
        std::pair<scalar_type const*,scalar_type const*> dev = ptr().as_split();
        std::pair<scalar_type*,scalar_type*> host = dest.as_split();

	cp_to_host(host.first, layout_.stride(0), dev.first, stride(0),
		   layout_.size(1), layout_.size(0));
	cp_to_host(host.second, layout_.stride(0), dev.second, stride(0),
		   layout_.size(1), layout_.size(0));
      }
      else
      {
	cp_to_host(dest.as_inter(), layout_.stride(0), ptr().as_inter(), stride(0),
		   layout_.size(1), layout_.size(0));
      }
    }
    else                     // column major
    {
      if (!is_complex<T>::value)
      {
        cp_to_host(dest.as_real(), layout_.stride(1), ptr().as_real(), stride(1),
		   layout_.size(0), layout_.size(1));
      }
      else if (storage_format() == split_complex)
      {
        std::pair<scalar_type const*,scalar_type const*> dev = ptr().as_split();
        std::pair<scalar_type*,scalar_type*> host = dest.as_split();

	cp_to_host(host.first, layout_.stride(1), dev.first, stride(1),
		   layout_.size(0), layout_.size(1));
	cp_to_host(host.second, layout_.stride(1), dev.second, stride(1),
		   layout_.size(0), layout_.size(1));
      }
      else
      {
	cp_to_host(dest.as_inter(), layout_.stride(1), ptr().as_inter(), stride(1),
		   layout_.size(0), layout_.size(1));
      }
    }
  }

private:
  applied_layout_type layout_;
};

/// Device storage for 3D rt-laid-out data.
/// This only supports dense data at this time.
template <typename T>
class Device_storage<T, Rt_layout<3> > : Rt_device_storage_base<T>
{
  typedef Rt_device_storage_base<T> base_type;
  typedef Rt_layout<3> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  typedef dda::impl::Pointer<T> type;
  typedef dda::impl::const_Pointer<T> const_type;
  typedef typename scalar_of<T>::type scalar_type;

  Device_storage(applied_layout_type const &l, bool use_direct = false, type buffer = type()) VSIP_NOTHROW
  : base_type(use_direct ? 0 : l.size(0)*l.size(1)*l.size(2), l.storage_format(), buffer), layout_(l)
  {
    // Make sure data is dense.
    assert(layout_.total_size() == 0 ||
	   (l.stride(2) == 1 &&
	    l.stride(1) > 0 &&
	    static_cast<length_type>(l.stride(1)) == l.size(2) &&
	    l.stride(0) > 0 &&
	    static_cast<length_type>(l.stride(0)) == l.size(1) * l.size(2)));
  }

  using base_type::total_size;
  length_type size(dimension_type d) const { return layout_.size(d);}
  stride_type stride(dimension_type d) const { return layout_.stride(d);}
  storage_format_type storage_format() const { return base_type::storage_format();}
  type ptr() { return type(base_type::ptr());}
  const_type ptr() const { return const_type(base_type::ptr());}

  void from_host(const_type src)
  {
    if (!is_complex<T>::value)
    {
      cp_to_dev(ptr().as_real(), src.as_real(), total_size());
    }
    else if (storage_format() == split_complex)
    {
      std::pair<scalar_type const*,scalar_type const*> host = src.as_split();
      std::pair<scalar_type*,scalar_type*> dev = ptr().as_split();
      cp_to_dev(dev.first, host.first, total_size());
      cp_to_dev(dev.second, host.second, total_size());
    }
    else
    {
      cp_to_dev(ptr().as_inter(), src.as_inter(), total_size());
    }
  }

  void to_host(type dest) const
  {
    if (!is_complex<T>::value)
    {
      cp_to_host(dest.as_real(), ptr().as_real(), total_size());
    }
    else if (storage_format() == split_complex)
    {
      std::pair<scalar_type const*,scalar_type const*> dev = ptr().as_split();
      std::pair<scalar_type*,scalar_type*> host = dest.as_split();
      cp_to_host(host.first, dev.first, total_size());
      cp_to_host(host.second, dev.second, total_size());
    }
    else
    {
      cp_to_host(dest.as_inter(), ptr().as_inter(), total_size());
    }
  }

private:
  applied_layout_type layout_;
};

} // namespace cuda
} // namespace impl
} // namespace vsip

#endif
