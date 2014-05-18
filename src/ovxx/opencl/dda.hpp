//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_dda_hpp_
#define ovxx_opencl_dda_hpp_

#include <ovxx/opencl/buffer.hpp>
#include <ovxx/dda.hpp>
#include <vsip/dense.hpp>

namespace ovxx
{
namespace opencl
{
namespace detail
{

template <typename T, cl_mem (T::*)()>
struct ptmf_helper;

template <typename T>
ovxx::detail::no_tag
has_buffer_helper(...);

template <typename T>
ovxx::detail::yes_tag
has_buffer_helper(int, ptmf_helper<T, &T::buffer>* p = 0);

template <typename B>
struct has_buffer
{
  static bool const value = 
    sizeof(has_buffer_helper<B>(0)) == sizeof(ovxx::detail::yes_tag);
};

template <typename B> struct has_buffer<B const> : has_buffer<B> {};

template <dimension_type D, typename T, typename L>
struct has_buffer<Strided<D, T, L> >
{
  static bool const value = true;
};

template <dimension_type D, typename T, typename O>
struct has_buffer<Dense<D, T, O> >
{
  static bool const value = true;
};

template <typename B, dda::sync_policy S, typename L,
	  bool direct = has_buffer<B>::value>
class accessor;

// For direct access we can just forward the call to buffer().
template <typename B, dda::sync_policy S, typename L>
class accessor<B, S, L, true>
{
  typedef typename B::value_type T;
public:
  static int const ct_cost = 0;
  accessor(B &block) : block_(block) {}
  void sync_in() {}
  void sync_out() {}
  buffer ptr() { return block_.buffer();}
  length_type size(dimension_type d) const { return block_.size(L::dim, d);}
  length_type size() const { return block_.size();}
  stride_type stride(dimension_type d) const { return block_.stride(L::dim, d);}
  length_type storage_size() const { return block_.size()*sizeof(T);}
private:
  B &block_;
};

// Otherwise we copy the block's data to a temporary buffer.
template <typename B, dda::sync_policy S, typename L>
class accessor<B, S, L, false>
{
  typedef typename B::value_type T;
  typedef dda::Data<B, S, L> data_type;
  // int const buffer_flags = CL_MEM_READ_WRITE; // TODO: refine that.
public:
  static int const ct_cost = 10;

  accessor(B &block)
    : block_(block),
      data_(block_),
      buffer_(default_context(), data_.storage_size()*sizeof(T), buffer::read_write)
  { sync_in();}
  ~accessor() { sync_out();}
  void sync_in()
  {
    if (S&dda::in)
      default_queue().write(data_.ptr(), buffer_, data_.storage_size());
  }
  void sync_out()
  {
    if (S&dda::out)
      default_queue().read(buffer_, (T*)data_.ptr(), data_.storage_size());
  }
  buffer ptr() { return buffer_;}
  length_type size(dimension_type d) const { return data_.size(d);}
  length_type size() const { return data_.size();}
  stride_type stride(dimension_type d) const { return data_.stride(d);}
  length_type storage_size() const { return data_.storage_size();}
private:
  B &block_;
  data_type data_;
  opencl::buffer buffer_;
};

} // namespace ovxx::opencl::detail

// This is modeled after DDA, but differs slightly due to the differing
// OpenCL memory model
template <typename B,
	  dda::sync_policy S,
	  typename L = typename dda::dda_block_layout<B>::layout_type,
	  bool ReadOnly = !(S&dda::out)>
class Data;

template <typename B, dda::sync_policy S, typename L>
class Data<B, S, L, false> : ovxx::detail::noncopyable,
                             ovxx::ct_assert<ovxx::is_modifiable_block<B>::value>
{
  typedef detail::accessor<B, S, L> backend_type;

public:
  static int   const ct_cost = backend_type::ct_cost;

  Data(B &block) : backend_(block) {}

  void sync_in() { backend_.sync_in();}
  void sync_out() { backend_.sync_out();}

  buffer ptr() { return backend_.ptr();}
  length_type size(dimension_type d) const { return backend_.size(d);}
  length_type size() const { return backend_.size();}
  stride_type stride(dimension_type d) const { return backend_.stride(d);}
  length_type storage_size() const { return backend_.storage_size();}

private:
  backend_type backend_;
};

/// Specialization for read-only synchronization
template <typename B, dda::sync_policy S, typename L>
class Data<B, S, L, true> : ovxx::detail::noncopyable
{
  typedef typename ovxx::remove_const<B>::type non_const_block_type;
  typedef detail::accessor<B const, S, L> backend_type;

public:
  static int   const ct_cost = backend_type::ct_cost;

  Data(B const &block) : backend_(block) {}

  void sync_in() { backend_.sync_in();}
  void sync_out() { backend_.sync_out();}

  buffer ptr() { return backend_.ptr();}
  length_type size(dimension_type d) const { return backend_.size(d);}
  length_type size() const { return backend_.size();}
  stride_type stride(dimension_type d) const { return backend_.stride(d);}
  length_type storage_size() const { return backend_.storage_size();}

private:
  backend_type backend_;
};

} // namespace ovxx::opencl
} // namespace ovxx

#endif
