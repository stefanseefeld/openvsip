//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

/// Description
///   Reference count base class.
///
///   Classes requiring reference counts should derive from Ref_count.
///
///   Notes:
///    * Ref_count is templated on the type of the derived class.
///      This allows it to call the derived class destructor when
///      the reference count goes to zero (without declaring the
///      destructor virtual).  However, this means that Ref_count
///      should only be used when the static_cast is safe.  This
///      precludes multiple inheritence and more then level of
///      additional inheritence that overrides the destructor.
///
///    * Class Ref_count declares count_ as mutable.  This allows the
///      increment_count() and decrement_count() members to preserve
///      const-ness.  This is necessary since it is possible to create
///      a reference to a const blocks (for example, creating a
///      subview).

#ifndef VSIP_CORE_REFCOUNT_HPP
#define VSIP_CORE_REFCOUNT_HPP

#include <vsip/support.hpp>
#include <vsip/core/noncopyable.hpp>
#include <cassert>

namespace vsip
{
namespace impl
{

/// Reference counting base class.
///
/// Ref_count has the following behavior:
///  - when constructed, reference count starts out at one.
///  - when increment_count is called, the reference count is
///    incremented.
///  - when decrement_count is called, the reference count is
///    decremented, and checked against 0.  If the count is
///    0, the destructor of the derived class is called.
///
template <typename DerivedClass>
class Ref_count
{
  // Constructors.
public:
  Ref_count() VSIP_NOTHROW : count_(1) {}

  // Accessors
public:
  void increment_count() const;
  void decrement_count() const;

  // Debug interface
public:
  unsigned impl_debug_count() const { return count_;}

  // Hidden Functions.
  // Reference-counted objects should not be copied by value, rather a
  // new reference should be created (and the count incremented).
private:
  Ref_count(Ref_count const&);
  Ref_count& operator=(Ref_count const&);

  // Member Data.
private:
  mutable unsigned int count_;
};

/// Type and valueused as the second argument to a Ref_counted_ptr
/// constructor to indicate that it should not increment the object's
/// reference count.  This is purely a flag, rather like std::nothrow.
enum noincrement_type { noincrement };

/// Smart pointer to a reference counted object.
///
/// Ref_counted_ptr must be instantiated with a type that implements the
/// above Ref_count interface.  When constructed with an instance of
/// that type, you may specify whether or not the refcount of that
/// object should be incremented.  Copy-initialization and assignment
/// always increment the reference count of the now-pointed-to object,
/// and decrement that of the no-longer-pointed-to object if any.
/// When destructed, the refcount is always decremented.
///
/// Pointer arithmetic is not supported.
///
/// In all other regards Ref_counted_ptr<T> behaves as T*.
template <typename Obj>
class Ref_counted_ptr
{
  Obj* ptr_;  ///< Underlying pointer.

public:
  typedef Obj& equiv_type;

  Ref_counted_ptr() : ptr_(0) {}

  /// Basic constructor from an object.  This takes a pointer to the object
  /// so that you write "Ref_counted_ptr<T> ptr(&obj)" just like an ordinary
  /// pointer.
  explicit
  Ref_counted_ptr(Obj *p) : ptr_(p) { ptr_->increment_count(); }

  /// Variant constructor which does not increment the reference count.
  /// This is useful when the object has just been constructed, so it
  /// already has a reference count of 1, and this is the sole handle.
  Ref_counted_ptr(Obj *p, noincrement_type) VSIP_NOTHROW : ptr_(p) {}

  /// Copy-constructor, always increments the reference count.
  Ref_counted_ptr(Ref_counted_ptr const& p) : ptr_(p.ptr_)
  { if (ptr_) ptr_->increment_count(); }

  /// Destructor.
  ~Ref_counted_ptr() { if (ptr_) ptr_->decrement_count(); }

  /// operator= always increments the reference count of the new object
  /// and decrements the reference count of the old object.
  Ref_counted_ptr&
  operator=(Ref_counted_ptr const& from)
    {
      if (ptr_ != from.ptr_)
      {
        if (ptr_) ptr_->decrement_count();
        ptr_ = from.ptr_;
        ptr_->increment_count();
      }
      return *this;
    }

  void reset(Obj *p, bool increment = false)
  {
    if (ptr_) ptr_->decrement_count();
    ptr_ = p;
    // If increment was given, p is assumed to be non-zero
    assert(ptr_ || !increment);
    if (increment) ptr_->increment_count();
  }

  /// Dereference operator.
  Obj&
  operator*() const VSIP_NOTHROW
    { return *ptr_; }

  /// Arrow operator.
  Obj*
  operator->() const VSIP_NOTHROW
    { return ptr_; }

  /// An accessor for situations where you need a raw pointer.
  Obj*
  get() const VSIP_NOTHROW
    { return ptr_; }
};



/// Smart pointer to a reference counted object, using policy for 
/// increment and decrement.
///
/// RPPtr must be instantiated with a type that implements a Ref_count
/// interface accessed via a Ref_count policy RP.  When constructed
/// with an instance of that type, you may specify whether or not the
/// refcount of that object should be incremented.
/// Copy-initialization and assignment always increment the reference
/// count of the now-pointed-to object, and decrement that of the
/// no-longer-pointed-to object if any.  When destructed, the refcount
/// is always decremented.
///
/// Pointer arithmetic is not supported.
///
/// In all other regards RPPtr<T> behaves as T*.
template <typename Obj,
	  typename RP>
class RPPtr
{
  Obj* ptr_;  ///< Underlying pointer.

public:
  /// Basic constructor from an object.  This takes a pointer to the object
  /// so that you write "RPPtr<T> ptr(&obj)" just like an ordinary
  /// pointer.
  explicit
  RPPtr(Obj *p) : ptr_(p)
    { RP::inc(ptr_); }

  /// Variant constructor which does not increment the reference count.
  /// This is useful when the object has just been constructed, so it
  /// already has a reference count of 1, and this is the sole handle.
  RPPtr(Obj *p, noincrement_type) VSIP_NOTHROW : ptr_(p) {}

  /// Copy-constructor, always increments the reference count.
  RPPtr(RPPtr const& p) : ptr_(p.ptr_)
    { RP::inc(ptr_); }

  /// Destructor.
  ~RPPtr() { RP::dec(ptr_); }

  /// operator= always increments the reference count of the new object
  /// and decrements the reference count of the old object.
  RPPtr&
  operator=(RPPtr const& from)
    {
      RP::dec(ptr_);
      ptr_ = from.ptr_;
      RP::inc(ptr_);
      return *this;
    }

  /// Dereference operator.
  Obj&
  operator*() const VSIP_NOTHROW
    { return *ptr_; }

  /// Arrow operator.
  Obj*
  operator->() const VSIP_NOTHROW
    { return ptr_; }

  /// An accessor for situations where you need a raw pointer.
  Obj*
  get() const VSIP_NOTHROW
    { return ptr_; }
};

/// Store a mutable object.
///
/// Mutable hosts an object by value, but uses a mutable data member,
/// unless the object is const, since const objects cannot be mutable.
template <typename Obj>
class Mutable
{
public:
  // Constructor.
  Mutable(Obj& data) : data_(data) {}
 
  Obj& get() const { return data_; }

private:
  mutable Obj data_;
};

template <typename Obj>
class Mutable<const Obj>
{
public:
  // Constructor.
  Mutable(const Obj& data) : data_(data) {}

  // Return a reference to the object.
  const Obj& get() const { return data_; }
 
private:
  const Obj data_;
};

/// Store object by value
///
/// Stored_value holds an object by-value, but presents an interface
/// similar to Ref_counted_ptr.  This allows views to use Stored_value
/// and Ref_counted_ptr interchangably.
///
/// Stored_value does not call increment_count and decrement_count on
/// stored value.
///
/// Note 050321: Making Stored_value Non_assignable and using default
/// copy-constructor would work fine.  However, views do not use the
/// copy-constructor directly, instead they construct a copy by
/// dereferencing the original Stored_value and using the "Obj*"
/// constructor.  Making Stored_value Non_copyable emphasizes this usage.
template <typename Obj>
class Stored_value : public Non_copyable
{
  // Constructors and destructor.
public:
  typedef Obj equiv_type;

  /// Basic constructor from an object.  This takes a pointer to the object
  /// so that you write "Stored_value<T> ptr(&obj)" just like Ref_counted_ptr.
  explicit
  Stored_value(Obj *p) : data_(*p) {}

  /// Variant constructor to maintain interface compatibility with
  /// Ref_counted_ptr.
  Stored_value(Obj *p, noincrement_type) : data_(*p) {}

  ~Stored_value() {}


  // Accessors.
public:
  Obj& operator*() const VSIP_NOTHROW { return data_.get(); }
  Obj* operator->() const VSIP_NOTHROW { return get(); }
  Obj* get() const VSIP_NOTHROW { return &data_.get(); }

  // Member data.
private:
  Mutable<Obj> data_; ///< Underlying stored value.
};

/// Increment reference count.
template <typename DerivedClass>
inline void
Ref_count<DerivedClass>::increment_count() const
{
  // Only make this thread-safe if specifically asked to
#if VSIP_IMPL_ENABLE_THREADING && __GNUC__
  __sync_fetch_and_add(&count_, 1);
#else
  ++count_;
#endif
}

/// Decrement reference count and delete if it goes to zero.
template <typename DerivedClass>
inline void
Ref_count<DerivedClass>::decrement_count() const
{
  // Only make this thread-safe if specifically asked to
#if VSIP_IMPL_ENABLE_THREADING && __GNUC__
  __sync_fetch_and_sub(&count_, 1);
#else
  --count_;
#endif
  if (count_ == 0)
    delete static_cast<DerivedClass*>(const_cast<Ref_count*>(this));
}

/// Pass view through, while decrement reference count of a view's block
/// (Useful for subview creation).
template <typename View>
View
decrement_block_count(View view)
{
  view.block().decrement_count();
  return view;
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_REFCOUNT_HPP
