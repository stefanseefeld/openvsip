/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */
#ifndef vsip_core_dda_pointer_hpp_
#define vsip_core_dda_pointer_hpp_

#include <vsip/core/c++0x.hpp>

namespace vsip
{
namespace dda
{
namespace impl
{

template <typename T>
T *offset(T *d, stride_type o) { return d + o;}

template <typename T>
T const *offset(T const *d, stride_type o) { return d + o;}

template <typename T>
std::pair<T*,T*> offset(std::pair<T*,T*> d, stride_type o) 
{ return std::pair<T*,T*>(d.first + o, d.second + o);}

template <typename T>
std::pair<T const*,T const*> offset(std::pair<T const*,T const*> d, stride_type o) 
{ return std::pair<T const*,T const *>(d.first + o, d.second + o);}


/// Class to represent either a interleaved-pointer or a split-pointer.
///
/// Primary definition handles non-complex types.  Functions
/// corresponding to split-pointer cause a runtime error, since
/// "split" does not make sense for non-complex types.
template <typename T>
class Pointer
{
public:
  // Default constructor creates a NULL pointer.
  Pointer() : ptr_(0) {}

  // Some places unfortunately require a Pointer const_cast
  // (FFT workspaces, notably).
  // Pointer(T const *ptr) : ptr_(const_cast<T*>(ptr)) {}
  Pointer(T* ptr) : ptr_(ptr) {}
  Pointer(std::pair<T*, T*> const&) { assert(0);}

  // Accessors.
  T *as_real() const { return ptr_;}
  T *as_inter() const { return ptr_;}
  std::pair<T*,T*> as_split() const { assert(0); return std::pair<T*,T*>(0,0);}

  bool is_null() const { return ptr_ == 0;}

private:
  T* ptr_;
};

/// Specialization for complex types.  Whether a Pointer refers to a
/// interleaved or split pointer is determined by the using code.
/// (However, when initializing an interleaved pointer, `ptr1_` is set
/// to 0).
template <typename T>
class Pointer<complex<T> >
{
public:
  Pointer() : ptr0_(0), ptr1_(0) {}
  Pointer(complex<T>* ptr) : ptr0_(reinterpret_cast<T*>(ptr)), ptr1_(0) {}
  Pointer(std::pair<T*, T*> const& ptr) : ptr0_(ptr.first), ptr1_(ptr.second) {}

  T *as_real() const { assert(0); return 0; }
  complex<T> *as_inter() const { return reinterpret_cast<complex<T>*>(ptr0_); }
  std::pair<T*, T*> as_split() const { return std::pair<T*,T*>(ptr0_, ptr1_); }

  bool is_null() const { return ptr0_ == 0; }

private:
  T* ptr0_;
  T* ptr1_;
};

template <typename T>
class const_Pointer
{
public:
  const_Pointer() : ptr_(0) {}

  const_Pointer(T const *ptr) : ptr_(ptr) {}
  const_Pointer(std::pair<T const *, T const *> const&) { assert(0);}
  const_Pointer(std::pair<T *, T *> const&) { assert(0);}
  const_Pointer(Pointer<T> const &r) : ptr_(r.as_real()) {}

  T const *as_real() const { return ptr_;}
  T const *as_inter() const { return ptr_;}
  std::pair<T const *,T const *> as_split() const 
  { assert(0); return std::pair<T const *,T const *>(0,0);}

  bool is_null() const { return ptr_ == 0;}

private:
  T const *ptr_;
};

template <typename T>
class const_Pointer<complex<T> >
{
public:
  const_Pointer() : ptr0_(0), ptr1_(0) {}
  const_Pointer(complex<T> const *ptr) : ptr0_(reinterpret_cast<T const*>(ptr)), ptr1_(0) {}
  const_Pointer(std::pair<T const *, T const *> const &ptr)
    : ptr0_(ptr.first), ptr1_(ptr.second) {}
  const_Pointer(std::pair<T *, T *> const &ptr)
    : ptr0_(ptr.first), ptr1_(ptr.second) {}
  const_Pointer(Pointer<complex<T> > const &ptr)
  {
    std::pair<T*,T*> tmp = ptr.as_split();
    ptr0_ = tmp.first;
    ptr1_ = tmp.second;
  }

  T const *as_real() const { assert(0); return 0;}
  complex<T> const *as_inter() const
  { return reinterpret_cast<complex<T> const *>(ptr0_);}
  std::pair<T const *, T const *> as_split() const 
  { return std::pair<T const *,T const *>(ptr0_, ptr1_);}

  bool is_null() const { return ptr0_ == 0;}

private:
  T const *ptr0_;
  T const *ptr1_;
};

template <typename T> 
struct Const_cast
{
  static T *cast(T *p) { return p;}
};

template <typename T>
struct Const_cast<T*> 
{
  static T *cast(T *p) { return p;}
  static T *cast(T const*p) { return const_cast<T*>(p);}
};

template <typename T>
struct Const_cast<std::pair<T*,T*> >
{
  static std::pair<T*,T*> cast(std::pair<T*,T*> p) { return p;}
  static std::pair<T*,T*> cast(std::pair<T const*,T const*> p)
  { return std::make_pair(const_cast<T*>(p.first), const_cast<T*>(p.second));}
};

template <typename T>
struct Const_cast<Pointer<T> >
{
  static Pointer<T> cast(Pointer<T> p) { return p;}
  static Pointer<T> cast(const_Pointer<T> p)
  { return Pointer<T>(Const_cast<T*>::cast(p.as_real()));}
};

template <typename T>
struct Const_cast<Pointer<complex<T> > >
{
  static Pointer<complex<T> > cast(Pointer<complex<T> > p) { return p;}
  static Pointer<complex<T> > cast(const_Pointer<complex<T> > p)
  { return Pointer<complex<T> >(Const_cast<std::pair<T*,T*> >::cast(p.as_split()));}
};

template <typename T1, typename T2>
inline T1
const_cast_(T2 p) { return Const_cast<T1>::cast(p);}

} // namespace vsip::dda::impl
} // namespace vsip::dda
} // namespace vsip

#endif
