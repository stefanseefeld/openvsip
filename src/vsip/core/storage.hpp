/* Copyright (c) 2005, 2006 by CodeSourcery, LLC.  All rights reserved. */

/// Description
///   Data storage within a block.

#ifndef vsip_core_storage_hpp_
#define vsip_core_storage_hpp_

#include <vsip/core/layout.hpp>
#include <vsip/core/aligned_allocator.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/noncopyable.hpp>
#include <vsip/core/dda_pointer.hpp>

namespace vsip
{
namespace impl
{
/// Storage: abstracts storage of data, in particular handling the storage
/// of split complex data.
///
/// Description:
///   For non-complex data and complex data stored in interleaved
///   format, data is stored in memory as a single regular array.
///   For complex data stored in split format, separate arrays for
///   the real and imaginary parts are maintained.
///
///   An optional specialization for complex data stored in
///   interleaved format that explicitly stores values as interleave
///   real and imaginary.  If std::complex<> stores its values in a
///   format compatible with interleaved storage (packed, with real
///   value first, imaginary value second), this specialization is not
///   necessary.
///
/// Notes:
///   Specializations for complex storage do not support getting a
///   reference to storaged values.
///
///   Storage does not remember the size of memory allocated.
///   As a result its destructor cannot perform deallocate (STL
///   allocator deallocater requires a size).  Prior to destruction,
///   Storage's user must call the deallocate() function.
template <storage_format_type C, typename T>
struct Storage
{
  typedef T*       type;
  typedef T const* const_type;
  typedef T        alloc_type;
  typedef typename scalar_of<T>::type scalar_type;

  // Used by block-copy
  static T    get(const_type data, index_type idx) { return data[idx];}
  static void put(type data, index_type idx, T val) { data[idx] = val;}

  static bool is_null(type data) { return data == 0; }
  static type null() { return 0; }


  // Allocator based allocation

  template <typename AllocT>
  static T*   allocate(AllocT& allocator, length_type size) 
  {
    return allocator.allocate(size);
  }

  template <typename AllocT>
  static void deallocate(AllocT& allocator, type data, length_type size) 
  {
    allocator.deallocate(data, size);
  }


  // Direct allocation

  static T*   allocate(length_type size) 
  {
    return alloc_align<T>(VSIP_IMPL_ALLOC_ALIGNMENT, size);
  }

  static void deallocate(type data)
  {
    free_align(data);
  }


  static type offset(type ptr, stride_type stride)
  { return ptr + stride; }

  static scalar_type *get_real_ptr(type ptr)
  { return (scalar_type*)ptr; }
  static scalar_type const *get_real_ptr(const_type ptr)
  { return (scalar_type*)ptr; }
  static scalar_type* get_imag_ptr(type /*ptr*/)
  {
    VSIP_IMPL_THROW(std::runtime_error("Accessing imaginary part of non-complex pointer"));
    return 0;
  }
  static scalar_type const *get_imag_ptr(const_type /*ptr*/)
  {
    VSIP_IMPL_THROW(std::runtime_error("Accessing imaginary part of non-complex pointer"));
    return 0; 
  }

};



template <typename T>
struct Storage<split_complex, vsip::complex<T> >
{
  typedef std::pair<T*, T*>             type;
  typedef std::pair<T const*, T const*> const_type;
  typedef T                             alloc_type;

  // Used by block-copy
  static complex<T> get(const_type data, index_type idx)
  { return vsip::complex<T>(data.first[idx], data.second[idx]);}
  static void       put(type data, index_type idx, complex<T> val)
  {
    data.first [idx] = val.real();
    data.second[idx] = val.imag();
  }

  static bool is_null(type data)
    { return data.first == 0 || data.second == 0; }
  static type null() { return type(0, 0); }


  // Allocator based allocation

  template <typename AllocT>
  static std::pair<T*, T*> allocate(AllocT& allocator, length_type size) 
  {
    return std::pair<T*, T*>(allocator.allocate(size),
			     allocator.allocate(size));
  }

  template <typename AllocT>
  static void deallocate(AllocT& allocator, type data, length_type size) 
  {
    allocator.deallocate(data.first, size);
    allocator.deallocate(data.second, size);
  }


  // Direct allocation

  static std::pair<T*, T*> allocate(length_type size) 
  {
    return std::pair<T*, T*>(alloc_align<T>(VSIP_IMPL_ALLOC_ALIGNMENT, size),
			     alloc_align<T>(VSIP_IMPL_ALLOC_ALIGNMENT, size));
  }

  static void deallocate(type data) 
  {
    free_align(data.first);
    free_align(data.second);
  }

  static type offset(type ptr, stride_type stride)
  { return type(ptr.first + stride, ptr.second + stride); }

  static T* get_real_ptr(type ptr) { return ptr.first;}
  static T const *get_real_ptr(const_type ptr) { return ptr.first;}
  static T* get_imag_ptr(type ptr){ return ptr.second;}
  static T const *get_imag_ptr(const_type ptr){ return ptr.second;}
};



template <storage_format_type C,
	  typename T,
	  typename AllocT = vsip::impl::Aligned_allocator<
                              typename Storage<C, T>::alloc_type> >
class Allocated_storage
{
  // Compile-time values and types.
public:
  typedef Storage<C, T> storage_type;

  typedef typename storage_type::type       type;
  typedef typename storage_type::const_type const_type;
  typedef typename storage_type::alloc_type alloc_type;

  enum state_type
  {
    alloc_data,
    user_data,
    no_data
  };

  // Constructors and destructor.
public:
  Allocated_storage(length_type   size,
		    type          buffer = storage_type::null(),
		    AllocT const& alloc  = AllocT())
    VSIP_THROW((std::bad_alloc))
    : alloc_ (alloc),
      state_ (size == 0                     ? no_data   :
	      storage_type::is_null(buffer) ? alloc_data
		                            : user_data),
      data_  (state_ == alloc_data ? storage_type::allocate(alloc_, size) :
	      state_ == user_data  ? buffer
	                           : storage_type::null())
  {}

  Allocated_storage(length_type   size,
		    T             val,
		    type          buffer = storage_type::null(),
		    AllocT const& alloc  = AllocT())
  VSIP_THROW((std::bad_alloc))
    : alloc_ (alloc),
      state_ (size == 0                     ? no_data   :
	      storage_type::is_null(buffer) ? alloc_data
		                            : user_data),
      data_  (state_ == alloc_data ? storage_type::allocate(alloc_, size) :
	      state_ == user_data  ? buffer
	                           : storage_type::null())
  {
    for (index_type i=0; i<size; ++i)
      storage_type::put(data_, i, val);
  }

  ~Allocated_storage()
  {
    // it using class's responsiblity to call deallocate().
    if (state_ == alloc_data)
      assert(storage_type::is_null(data_));
  }

  // Accessors.
public:
  void rebind(length_type size, type buffer);

  void deallocate(length_type size)
  {
    if (state_ == alloc_data)
    {
      storage_type::deallocate(alloc_, data_, size);
      data_ = storage_type::null();
    }
  }

  bool is_alloc() const { return state_ == alloc_data; }

  T    get(index_type idx) const { return storage_type::get(data_, idx); }
  void put(index_type idx, T val){ storage_type::put(data_, idx, val); }

  // T&       ref(index_type idx)       { return data_[idx]; }
  // T const& ref(index_type idx) const { return data_[idx]; }

  type       ptr()       { return data_; }
  const_type ptr() const { return data_; }

  // Member data.
private:
  typename AllocT::template rebind<alloc_type>::other alloc_;
  state_type                                          state_;
  type                                                data_;
};



/// Allocated storage, with complex format determined at run-time.
template <typename T>
class Rt_allocated_storage : Non_copyable
{
  // Compile-time values and types.
public:
  // typedef Storage<ComplexFmt, T> storage_type;

  typedef dda::impl::Pointer<T> type;
  typedef dda::impl::Pointer<T> const_type;
  // typedef typename storage_type::type       type;
  // typedef typename storage_type::const_type const_type;
  // typedef typename storage_type::alloc_type alloc_type;

  enum state_type
  {
    alloc_data,
    user_data,
    no_data
  };

  // Constructors and destructor.
public:
  static dda::impl::Pointer<T> allocate_(storage_format_type cformat,
					 length_type size)
  {
    if (!is_complex<T>::value || cformat == interleaved_complex)
    {
      return dda::impl::Pointer<T>(alloc_align<T>(VSIP_IMPL_ALLOC_ALIGNMENT, size));
    }
    else
    {
      typedef typename scalar_of<T>::type scalar_type;
      return dda::impl::Pointer<T>(std::pair<scalar_type*, scalar_type*>(
	alloc_align<scalar_type>(VSIP_IMPL_ALLOC_ALIGNMENT, size),
	alloc_align<scalar_type>(VSIP_IMPL_ALLOC_ALIGNMENT, size)));
    }
  }

  static void deallocate_(dda::impl::Pointer<T>   ptr,
			  storage_format_type cformat,
			  length_type     /*size*/)
  {
    if (cformat == interleaved_complex)
    {
      free_align(ptr.as_inter());
    }
    else
    {
      free_align(ptr.as_split().first);
      free_align(ptr.as_split().second);
    }
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
      // interleaved.  We can't fix this, but we can through an
      // exception.
      assert(!(cformat == interleaved_complex && is_complex<T>::value &&
	       buffer.as_split().second != 0));
      return buffer;
    }
  }

  Rt_allocated_storage(length_type     size,
		       storage_format_type cformat,
		       type            buffer = type())
    VSIP_THROW((std::bad_alloc))
    : cformat_(cformat),
      state_ (size == 0         ? no_data   :
	      buffer.is_null()  ? alloc_data
		                : user_data),
      data_  (state_ == alloc_data ? allocate_ (cformat_, size) :
	      state_ == user_data  ? partition_(cformat, buffer, size)
	                           : type())
  {}

  Rt_allocated_storage(length_type   size,
		       storage_format_type cformat,
		       T               val,
		       type            buffer = type())
  VSIP_THROW((std::bad_alloc))
    : cformat_(cformat),
      state_ (size == 0         ? no_data   :
	      buffer.is_null() ? alloc_data
		                : user_data),
      data_  (state_ == alloc_data ? allocate_ (cformat, size) :
	      state_ == user_data  ? partition_(cformat, buffer, size)
	                           : type())
  {
    if (cformat == interleaved_complex)
    {
      typedef Storage<interleaved_complex, T> inter_storage_type;
      for (index_type i=0; i<size; ++i)
	inter_storage_type::put(data_.as_inter(), i, val);
    }
    else /* (cformat == cmplx_split_fmt) */
    {
      typedef Storage<split_complex, T> split_storage_type;
      for (index_type i=0; i<size; ++i)
	split_storage_type::put(data_.as_inter(), i, val);
    }
  }

  ~Rt_allocated_storage()
  {
    // it using class's responsiblity to call deallocate().
    if (state_ == alloc_data)
      assert(data_.is_null());
  }

  // Accessors.
public:
  void rebind(length_type size, type buffer);

  void deallocate(length_type size)
  {
    if (state_ == alloc_data)
    {
      deallocate_(data_, cformat_, size);
      data_ = type();
    }
  }

  bool is_alloc() const { return state_ == alloc_data; }

  type       ptr()       { return data_; }
  const_type ptr() const { return data_; }

  // Member data.
private:
  storage_format_type cformat_;
  state_type      state_;
  type            data_;
};



/***********************************************************************
  Definitions
***********************************************************************/

/// Rebind the memory referred to by Allocated_storage object
///
/// Requires:
///   SIZE to be size object was constructed with.
template <storage_format_type C,
	  typename T,
	  typename AllocT>
void
Allocated_storage<C, T, AllocT>::rebind(length_type size,
					type        buffer)
{
  if (!storage_type::is_null(buffer))
  {
    if (state_ == alloc_data)
      storage_type::deallocate(alloc_, data_, size);
    
    state_ = user_data;
    data_  = buffer;
  }
  else // is_null(buffer)
  {
    if (state_ == user_data)
    {
      state_ = alloc_data;
      data_  = storage_type::allocate(alloc_, size);
    }
    /* else do nothing - we already own our data */
  }
}

} // namespace vsip::impl
} // namespace vsip

#endif
