//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_domain_hpp_
#define vsip_domain_hpp_

#include <ovxx/support.hpp>
#include <ovxx/vertex.hpp>

namespace vsip
{

template <dimension_type D> struct Index;

template <> class Index<1> : public ovxx::Vertex<index_type, 1>
{
public:
  Index() VSIP_NOTHROW {}
  Index(index_type x) VSIP_NOTHROW : ovxx::Vertex<index_type, 1>(x) {}
};

// mathematical operations for Index

template <dimension_type Dim>
inline bool 
operator==(Index<Dim> const& i, Index<Dim> const& j) VSIP_NOTHROW
{
  for (dimension_type d=0; d<Dim; ++d)
    if (i[d] != j[d])
      return false;
  return true;
}



// Specialize for Index<1>.  This is identical to the template
// operator==, but it allows implicit conversions to be considered.
// In particular, it allows Index<1> values to be compared with
// index_type values.

inline bool 
operator==(Index<1> const& i, Index<1> const& j) VSIP_NOTHROW
{
  return i[0] == j[0];
}



template <dimension_type Dim>
inline
Index<Dim>
operator-(
  Index<Dim> const& op1,
  Index<Dim> const& op2)
{
  Index<Dim> res;
  for (dimension_type d=0; d<Dim; ++d)
    res[d] = op1[d] - op2[d];
  return res;
}



template <dimension_type Dim>
inline
Index<Dim>
operator+(
  Index<Dim> const& op1,
  Index<Dim> const& op2)
{
  Index<Dim> res;
  for (dimension_type d=0; d<Dim; ++d)
    res[d] = op1[d] + op2[d];
  return res;
}


template <> class Index<2> : public Vertex<index_type, 2>
{
public:
  Index() VSIP_NOTHROW {}
  Index(index_type x, index_type y) VSIP_NOTHROW 
  : Vertex<index_type, 2>(x, y) {}
};

template <> class Index<3> : public Vertex<index_type, 3>
{
public:
  Index() VSIP_NOTHROW {}
  Index(index_type x, index_type y, index_type z) VSIP_NOTHROW
  : Vertex<index_type, 3>(x, y, z) {}
};

/// A Domain is a non-empty set of non-negative indexes.
//
/// Validation checks are only performed in Domain<1> constructors,
/// since Domain<D> is expressed in terms of Domain<1> and all modifying
/// operations on domains are expressed by assigning new Domain<1> objects.

template <dimension_type D> class Domain;

template <>
class Domain<1>
{
public:
  static dimension_type const dim = 1;

  Domain() VSIP_NOTHROW
    : index_(0), stride_(1), length_(1) { }
  Domain(index_type i, stride_type s, length_type len) VSIP_NOTHROW
    : index_(i), stride_(s), length_(len) { assert(this->impl_is_valid()); }
  Domain(length_type len) VSIP_NOTHROW
    : index_(0), stride_(1), length_(len) { assert(this->impl_is_valid()); }
  Domain(Domain const& d) VSIP_NOTHROW
    : index_(d.index_), stride_(d.stride_), length_(d.length_) { }
  Domain<1> const& operator[](dimension_type d OVXX_UNUSED) const VSIP_NOTHROW
    { assert(d == 0); return *this; } 

  Domain<1>& impl_at(dimension_type d OVXX_UNUSED) VSIP_NOTHROW
    { assert(d == 0); return *this; } 

  // these are defined explicitly just so we can paste on VSIP_NOTHROW.
 
  Domain& operator=(Domain const& d) VSIP_NOTHROW;

  bool element_conformant(Domain<1> const& dom) const VSIP_NOTHROW;
  bool product_conformant(Domain<2> const&) const VSIP_NOTHROW
    { return false; }

  index_type  first()  const VSIP_NOTHROW { return this->index_;}
  stride_type stride() const VSIP_NOTHROW { return this->stride_;}
  length_type length() const VSIP_NOTHROW { return this->length_;}
  length_type size()   const VSIP_NOTHROW { return this->length_;}

  // These functions are an extension to the specs.  impl_nth() is not
  // operator[] because operator[] needs to remain the identity function
  // for convenience when writing cross-dimensional templates.

  index_type impl_nth(index_type const i) const VSIP_NOTHROW
    {
      // We need to do the multiplication using signed arithmetic,
      // because the stride may be negative.  The addition can happen
      // in unsigned arithmetic; if the stride is negative, conversion
      // will produce a very large number and the unsigned sum will
      // wrap around to the value we want.  (C++ guarantees this
      // property for unsigned arithmetic, see [conv.integral] and
      // [basic.fundamental]/4.)

      return this->index_ + this->stride_ * static_cast<stride_type>(i);
    }
  index_type impl_last() const VSIP_NOTHROW
    { return this->impl_nth(this->length() - 1); }

  /// A Domain is valid if it is non-empty and non-negative.
  inline bool impl_is_valid() const VSIP_NOTHROW;

  // These are used to implement [domains.arithmetic].
 
  inline void impl_add_in(index_difference_type s) VSIP_NOTHROW;
  inline void impl_sub_out(index_difference_type s) VSIP_NOTHROW;
  inline void impl_mul_by(stride_scalar_type s) VSIP_NOTHROW;
  inline void impl_div_by(stride_scalar_type s) VSIP_NOTHROW;

  inline bool impl_equals(Domain const&) const VSIP_NOTHROW;

  enum impl_No_init { impl_no_init };
  Domain(Domain::impl_No_init) VSIP_NOTHROW { }

private:
  index_type  index_;
  stride_type stride_;
  length_type length_;
};

inline Domain<1>& 
Domain<1>::operator=(Domain const& d) VSIP_NOTHROW
{
  this->index_ = d.index_;
  this->stride_ = d.stride_;
  this->length_ = d.length_;
  return *this;
}

inline bool 
Domain<1>::impl_is_valid() const VSIP_NOTHROW 
{
  // As the stride can be negative, we have to test whether 
  // the upper end-point is greater than or equal to zero.
  return(this->size() == 0
         || (this->size() > 0 &&
	     static_cast<stride_type>(this->impl_last()) >= 0));
}

#define OVXX_DEF_DOM1_MEMBER(Mem,Arg,Datum,Assign_op) \
inline void					      \
Domain<1>::Mem(Arg const a) VSIP_NOTHROW	      \
{						      \
  this->Datum Assign_op a;			      \
  assert(this->impl_is_valid());		      \
}

OVXX_DEF_DOM1_MEMBER(impl_add_in, index_difference_type, index_, +=)
OVXX_DEF_DOM1_MEMBER(impl_sub_out, index_difference_type, index_, -=)
OVXX_DEF_DOM1_MEMBER(impl_mul_by, stride_scalar_type, stride_, *=)
OVXX_DEF_DOM1_MEMBER(impl_div_by, stride_scalar_type, stride_, /=)

#undef OVXX_DEF_DOM1_MEMBER


inline bool 
Domain<1>::element_conformant(Domain<1> const& d) const VSIP_NOTHROW
{ return this->length() == d.length(); }

inline bool 
Domain<1>::impl_equals(Domain const& d) const VSIP_NOTHROW
{
  return this->length() == d.length() &&
    ((this->stride() == d.stride() && this->first() == d.first()) ||
     (this->stride() == -d.stride() && 
      this->first() == d.first() + (d.length() - 1) * d.stride()));
}


namespace impl
{

// definitions of Op types to pass to Vec_mem_op_scalar
#define OVXX_DEF_OP_OBJ(Tag, Op, Arg)				\
struct Tag							\
{								\
  static void apply(Domain<1>& dom, const Arg u) VSIP_NOTHROW	\
  { dom.Op(u); }						\
}

OVXX_DEF_OP_OBJ(Add_in,  impl_add_in, index_difference_type);
OVXX_DEF_OP_OBJ(Sub_out, impl_sub_out, index_difference_type);
OVXX_DEF_OP_OBJ(Mul_by,  impl_mul_by, stride_scalar_type);
OVXX_DEF_OP_OBJ(Div_by,  impl_div_by, stride_scalar_type);

#undef OVXX_DEF_OP_OBJ

////////////////////////////////////////////////////////////////
// Vec_mem_op_scalar: recursively apply a modifying operation and 
// an argument to elements of an array.

template <typename Op, typename Arg1, typename Arg2, dimension_type D>
struct Vec_mem_op_scalar
{
  static void apply(Arg1* const dom, Arg2 d) VSIP_NOTHROW
    { Op::apply(*dom, d);
      Vec_mem_op_scalar<Op,Arg1,Arg2,D-1>::apply(dom + 1, d); }
};

template <typename Op, typename Arg1, typename Arg2>
struct Vec_mem_op_scalar<Op,Arg1,Arg2,0>
{ 
  static void apply(Arg1*, Arg2) VSIP_NOTHROW { }
};

struct Equal
{
  static bool apply(Domain<1> const& a, Domain<1> const& b) VSIP_NOTHROW 
    { return a.impl_equals(b); }
};

struct Conforms
{
  static bool apply(Domain<1> const& a, Domain<1> const& b) VSIP_NOTHROW 
    { return a.element_conformant(b); }
};

////////////////////////////////////////////////////////////////
// Vec_mem_pred: recursively apply a member predicate to corresponding
//   elements of vectors, short-circuiting the result.

template <typename Pred, typename Arg, dimension_type D>
struct Vec_mem_pred
{
  static bool apply(Arg* const a, Arg* const b) VSIP_NOTHROW
  { return Pred::apply(*a, *b) &&
      Vec_mem_pred<Pred,Arg,D-1>::apply(a + 1, b + 1); }
};

template <typename Pred, typename Arg>
  struct Vec_mem_pred<Pred,Arg,0>
{ 
  static bool apply(Arg*, Arg*) VSIP_NOTHROW { return true; }
};


////////////////////////////////////////////////////////////////
// Vec_copy: recursively corresponding elements of vectors.

template <typename Arg, dimension_type D>
struct Vec_copy
{
  static void apply(Arg* const a, Arg const* const b) VSIP_NOTHROW
    { *a = *b;
      Vec_copy<Arg,D-1>::apply(a + 1, b + 1); }
};

template <typename Arg>
struct Vec_copy<Arg,1>
{
  static void apply(Arg* const a, Arg const* const b) VSIP_NOTHROW
    { *a = *b; }
};

////////////////////////////////////////////////////////////////
// vec_domain_size: return product of sizes of D Domain<1>s.

template <dimension_type Dim>
inline length_type 
vec_domain_size(Domain<1> const* const d) VSIP_NOTHROW
  { return d->size() * vec_domain_size<Dim-1>(d + 1); }

template <>
inline length_type 
vec_domain_size<1>(Domain<1> const* const d) VSIP_NOTHROW
  { return d->size(); }

////////////////////////////////////////////////////////////////

// Domain1n is just like a Domain<1> except its default constructor
// doesn't initialize it (that's what the "n" stands for).  Used for 
// the array member Domain_base<D>::domains_[D].

struct Domain1n : Domain<1>
{
  Domain1n() VSIP_NOTHROW : Domain<1>(Domain<1>::impl_no_init) { }
  Domain1n(Domain1n const& d) VSIP_NOTHROW : Domain<1>(d) { }
  Domain1n& operator=(Domain<1> const& d) VSIP_NOTHROW
    { Domain<1>::operator=(d); return *this; }
};

template <dimension_type D>
struct Domain_base
{
  static dimension_type const dim = D;

  Domain_base() VSIP_NOTHROW { }  // leave members uninited!
  Domain_base(const Domain_base& dom) VSIP_NOTHROW;

  Domain<1> const& operator[](dimension_type d) const VSIP_NOTHROW
    { assert(d < D); return this->domains_[d]; }

  Domain<1>& impl_at(dimension_type d) VSIP_NOTHROW
    { assert(d < D); return this->domains_[d]; } 

  length_type size() const VSIP_NOTHROW
    { return vec_domain_size<D>(this->domains_); }

  bool impl_equals(Domain_base<D> const& dom) const VSIP_NOTHROW;

  bool element_conformant(const Domain<D>& dom) const VSIP_NOTHROW
    { return Vec_mem_pred<Conforms,const Domain1n,D>::apply(
               this->domains_, dom.domains_); }

  bool product_conformant(const Domain<2>& dom) const VSIP_NOTHROW;

  void operator=(Domain_base<D> const& d) VSIP_NOTHROW
    { Vec_copy<Domain1n,D>::apply(this->domains_, d.domains_); }

  #define OVXX_DEF_UPDATE(Op,Arg,Op_type) \
  inline void Op(Arg const d) VSIP_NOTHROW \
    { Vec_mem_op_scalar<Op_type,Domain1n,const Arg,D>::apply( \
        this->domains_, d); }

  // These are used to implement [domains.arithmetic].
  OVXX_DEF_UPDATE(impl_add_in, index_difference_type, Add_in)
  OVXX_DEF_UPDATE(impl_sub_out, index_difference_type, Sub_out)
  OVXX_DEF_UPDATE(impl_mul_by, stride_scalar_type, Mul_by)
  OVXX_DEF_UPDATE(impl_div_by, stride_scalar_type, Div_by)

  #undef OVXX_DEF_UPDATE

  Domain1n domains_[dim];
};


template <dimension_type D>
inline 
Domain_base<D>::Domain_base(const Domain_base& dom) VSIP_NOTHROW
{ Vec_copy<Domain1n,D>::apply(this->domains_, dom.domains_); }

template <dimension_type D>
bool
Domain_base<D>::impl_equals(Domain_base<D> const& dom) const VSIP_NOTHROW
{
  return Vec_mem_pred<Equal,const Domain1n,D>::apply(
    this->domains_, dom.domains_);
}



// Return size of dimension.

// Generic function.  Overloaded for structures that can encode
// extents (Domain, Length)

template <dimension_type D>
inline length_type
size_of_dim(Domain<D> const& len, dimension_type d)
{
  return len[d].size();
}

} // namespace impl


// We have to specialize Domain<D> for various D explicitly just
// so we can declare the special constructors.

template <> 
class Domain<2> : public impl::Domain_base<2>
{
public:
  Domain() VSIP_NOTHROW
    { this->domains_[0] = this->domains_[1] = Domain<1>(); }
  Domain(const Domain<1>& d0, const Domain<1>& d1) VSIP_NOTHROW 
    { this->domains_[0] = d0; this->domains_[1] = d1; }
  Domain(const Domain& dom) VSIP_NOTHROW 
    : impl::Domain_base<2>(dom) { }
  Domain& operator=(Domain const& dom) VSIP_NOTHROW
    { impl::Domain_base<2>::operator=(dom); return *this; } 
};

template <> 
class Domain<3> : public impl::Domain_base<3>
{
public:
  Domain() VSIP_NOTHROW;
  Domain(const Domain<1>&, const Domain<1>&, const Domain<1>&) VSIP_NOTHROW;
  Domain(const Domain& dom) VSIP_NOTHROW : impl::Domain_base<3>(dom) { }
  Domain& operator=(Domain const& dom) VSIP_NOTHROW
    { impl::Domain_base<3>::operator=(dom); return *this; } 
};

inline
Domain<3>::Domain() VSIP_NOTHROW
{ this->domains_[0] = this->domains_[1] = this->domains_[2] = Domain<1>(); }

inline
Domain<3>::Domain(
  const Domain<1>& d0,
  const Domain<1>& d1,
  const Domain<1>& d2) VSIP_NOTHROW
{
  domains_[0] = d0;
  domains_[1] = d1;
  domains_[2] = d2;
}

template <dimension_type D>
inline bool
impl::Domain_base<D>::product_conformant(
  Domain<2> const& dom) const VSIP_NOTHROW
{
  // note: this is OK because Domain<1> has its own, specialized definition.
  return D == 2 && this->domains_[1].length() == dom.domains_[0].length();
}

/***********************************************************************
  Globals
***********************************************************************/

/// [domains.index.equality]
/// Domain<D>s can be compared for equality and inequality.  Two
/// Domain<D>s are equal if and only if they contain exactly the same
/// Index<D>es.

template <dimension_type D> 
inline bool
operator==(const Domain<D>& d1, const Domain<D>& d2) VSIP_NOTHROW
  { return d1.impl_equals(d2); }

template <dimension_type D>
inline bool 
operator!=(Domain<D> const& d0, Domain<D> const& d1) VSIP_NOTHROW
{ return !operator==(d0, d1); }

#define OVXX_DEF_DOMAIN_OP(Op,Arg,Mem) \
  template <dimension_type D> \
  inline const Domain<D> \
  operator Op(Domain<D> const& dom, const Arg a) VSIP_NOTHROW \
  { \
    Domain<D> retn(dom); \
    retn.Mem(a); \
    return retn;   \
  }
OVXX_DEF_DOMAIN_OP(+, index_difference_type, impl_add_in)
OVXX_DEF_DOMAIN_OP(-, index_difference_type, impl_sub_out)
OVXX_DEF_DOMAIN_OP(*, stride_scalar_type, impl_mul_by)
OVXX_DEF_DOMAIN_OP(/, stride_scalar_type, impl_div_by)

#undef OVXX_DEF_DOMAIN_OP

template <dimension_type D>
inline const Domain<D>
operator+(index_difference_type diff, Domain<D> const& dom) VSIP_NOTHROW
{ return dom + diff; }

template <dimension_type D>
inline const Domain<D>
operator*(stride_scalar_type s, Domain<D> const& dom) VSIP_NOTHROW
{ return dom * s; }

} // namespace ovxx

#endif
