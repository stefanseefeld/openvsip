/* Copyright (c) 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/simd/expr_iterator.hpp
    @author  Stefan Seefeld
    @date    2006-07-18
    @brief   VSIPL++ Library: SIMD expression iterators.

*/

#ifndef VSIP_IMPL_SIMD_EXPR_ITERATOR_HPP
#define VSIP_IMPL_SIMD_EXPR_ITERATOR_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>
#include <vsip/opt/simd/simd.hpp>
#include <vsip/core/expr/operations.hpp>
#include <vsip/core/metaprogramming.hpp>

/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace simd
{

template <typename T,                  // value_type
	  template <typename> class O> // operator
struct Unary_operator_map
{
  // The general case, in particular unary functors, are not supported.
  static bool const is_supported = false;
};

template <typename T> 
struct Unary_operator_map<T, expr::op::Plus>
{
  typedef typename Simd_traits<T>::simd_type simd_type;
  static bool const is_supported = true;
  static simd_type 
  apply(simd_type const &op)
  { return Simd_traits<T>::add(Simd_traits<T>::zero(), op);}
};

template <typename T> 
struct Unary_operator_map<T, expr::op::Minus>
{
  typedef typename Simd_traits<T>::simd_type simd_type;
  static bool const is_supported = true;
  static simd_type 
  apply(simd_type const &op)
  { return Simd_traits<T>::sub(Simd_traits<T>::zero(), op);}
};

template <typename T,                            // value_type
	  template <typename, typename> class O> // operator
struct Binary_operator_map
{
  // The general case, in particular binary functors, are not supported.
  static bool const is_supported = false;
  typedef T value_type;
  typedef T return_type;
};

template <typename T,                                      // type
          template <typename, typename, typename> class O> // operator
struct Ternary_operator_map
{
  // The general case, in particular ternary functors, are not supported.
  static bool const is_supported = false;
};


template <typename T>
struct Binary_operator_map<T, expr::op::Add>
{
  typedef T value_type;
  typedef T return_type;
  typedef typename Simd_traits<T>::simd_type simd_type;
  static bool const is_supported = true;
  static simd_type 
  apply(simd_type const &left, simd_type const &right)
  { return Simd_traits<T>::add(left, right);}
};

template <typename T>
struct Binary_operator_map<T, expr::op::Sub>
{
  typedef T value_type;
  typedef T return_type;
  typedef typename Simd_traits<T>::simd_type simd_type;
  static bool const is_supported = true;
  static simd_type 
  apply(simd_type const &left, simd_type const &right)
  { return Simd_traits<T>::sub(left, right);}
};

template <typename T>
struct Binary_operator_map<T, expr::op::Mult>
{
  typedef T value_type;
  typedef T return_type;
  typedef typename Simd_traits<T>::simd_type simd_type;
  static bool const is_supported = true;
  static simd_type 
  apply(simd_type const &left, simd_type const &right)
  { return Simd_traits<T>::mul(left, right);}
};

template <typename T>
struct Binary_operator_map<T, expr::op::Div>
{
  typedef T value_type;
  typedef T return_type;
  typedef typename Simd_traits<T>::simd_type simd_type;
  static bool const is_supported = Simd_traits<T>::has_div;
  static simd_type 
  apply(simd_type const &left, simd_type const &right)
  { return Simd_traits<T>::div(left, right);}
};



#define VSIP_OPT_DECL_UNARY_OP(FCN, OP)					\
template <typename T>							\
struct Unary_operator_map<T, OP>					\
{									\
  typedef typename Simd_traits<T>::simd_type simd_type;			\
  typedef T                             return_type;			\
  typedef T                             value_type;			\
  									\
  static bool const is_supported = true;				\
  static inline simd_type						\
  apply(simd_type const &arg)						\
  {									\
    return Simd_traits<T>::FCN(arg);					\
  }									\
};

#define VSIP_OPT_DECL_UNARY_OP_COMPLEX_BLOCK(OP)			\
template <typename T>							\
struct Unary_operator_map<complex<T>, OP>				\
{									\
  static bool const is_supported = false;				\
};



#define VSIP_OPT_DECL_BINARY_OP(FCN,OP)					\
template <typename T>							\
struct Binary_operator_map<T, OP>					\
{									\
  typedef typename Simd_traits<T>::simd_type simd_type;			\
  typedef T                             return_type;			\
  typedef T                             value_type;			\
  									\
  static bool const is_supported = true;				\
  static inline simd_type						\
  apply(simd_type const &left, simd_type const &right)			\
  {									\
    return Simd_traits<T>::FCN(left, right);				\
  }									\
};



// Binary operators that return different type than type of operands
#define VSIP_OPT_DECL_BINARY_CMP_OP(FCN,O) \
template <typename T> \
struct Binary_operator_map<T, O> \
{ \
  typedef typename Simd_traits<int>::simd_type simd_itype; \
  typedef typename Simd_traits<T>::simd_type   simd_type;  \
  typedef int                           return_type;       \
  typedef T                             value_type;        \
                                                           \
  static bool const is_supported = true;                   \
  static inline simd_itype                                 \
  apply(simd_type const &left, simd_type const &right)     \
  {                                                        \
    simd_itype mask = simd_itype(Simd_traits<T>::FCN(left, right)); \
    return mask;                                           \
  }                                                        \
};




VSIP_OPT_DECL_BINARY_CMP_OP(gt,  expr::op::Gt)
VSIP_OPT_DECL_BINARY_CMP_OP(lt,  expr::op::Lt)
VSIP_OPT_DECL_BINARY_CMP_OP(ge,  expr::op::Ge)
VSIP_OPT_DECL_BINARY_CMP_OP(le,  expr::op::Le)

VSIP_OPT_DECL_BINARY_OP(max, expr::op::Max)
VSIP_OPT_DECL_BINARY_OP(min, expr::op::Min)

VSIP_OPT_DECL_UNARY_OP(mag, expr::op::Mag)
VSIP_OPT_DECL_UNARY_OP_COMPLEX_BLOCK(expr::op::Mag)

#undef VSIP_OPT_DECL_BINARY_CMP_OP
#undef VSIP_OPT_DECL_BINARY_OP
#undef VSIP_OPT_DECL_UNARY_OP
#undef VSIP_OPT_DECL_UNARY_OP_COMPLEX_BLOCK

// Support for ternary maps
template <typename T>
struct Ternary_operator_map<T, expr::op::Ite>
{
  typedef Simd_traits<T>                       simd;
  typedef typename simd::simd_type             simd_type;
  typedef Simd_traits<int>                     simdi;
  typedef typename simdi::simd_type            simd_itype;

  static bool const is_supported = false;
  static simd_type
  apply(simd_itype const& mask, simd_type const& a, simd_type const& k)
  {
    simd_itype nmask        = simdi::bnot(mask);
    simd_itype xor_val      = simdi::bxor(simd_itype(a),simd_itype(k));
    simd_itype and_val      = simdi::band(xor_val,nmask);
    simd_itype res          = simdi::bxor(and_val,simd_itype(a));
    return simd_type(res);
  }
};

// Access trait for direct access to contiguous aligned memory.
template <typename T> struct Direct_access_traits 
{
  typedef T value_type;
};

// Access trait for direct lvalue access to contiguous aligned memory.
template <typename T> struct LValue_access_traits 
{
  typedef T value_type;
};

// Access trait for unaccelerated access. Either non-contiguous or not aligned.
template <typename T> struct Indirect_access_traits 
{
  typedef T value_type;
};

// Access trait for scalar blocks.
template <typename T> struct Scalar_access_traits 
{
  typedef T value_type;
};

template <typename T> struct Complex_inter_access_traits 
{
  typedef T value_type;
};

template <typename T> struct Complex_split_access_traits 
{ 
  typedef T value_type;
};

// Access trait for unary expressions.
template <typename ProxyT,             // operatory proxy
	  template <typename> class O> // operator
struct Unary_access_traits
{
  typedef typename ProxyT::value_type value_type;
};

// Access trait for binary expressions. Both operands have the same value_type.
// TODO: Support (T, std::complex<T>) and (std::complex<T>, T) binary 
// operations.
template <typename L,                            // left operator proxy
	  typename R,                            // right operator proxy
	  template <typename, typename> class O> // operator
struct Binary_access_traits
{
  typedef Binary_operator_map<typename L::value_type,O> bin_op;

  typedef typename bin_op::value_type  value_type;
  typedef typename bin_op::return_type return_type;
};

template <typename P1, typename P2, typename P3,
          template <typename,typename,typename> class O>
struct Ternary_access_traits
{
  typedef typename P1::value_type value_type;
};



// Helper class for loading SIMD values.
//
// Two specializatons are provided:
//   - has_perm=true is for SIMD instruction sets that have a permute
//                   instruction (and no unaligned load), such as
//                   AltiVec.
//   - has_perm=false is for SIMD instruction sets that do not have
//                   a permute instruction, but allow unaligned loads,
//                   such as SSE.
//
// The SIMD traits class has a 'has_perm' trait.

template <typename T,
          bool has_perm = Simd_traits<T>::has_perm>
struct Simd_unaligned_loader;

template <typename T>
struct Simd_unaligned_loader<T, true>
{
  typedef Simd_traits<T>                 simd;
  typedef typename simd::simd_type       simd_type;
  typedef typename simd::perm_simd_type  perm_simd_type;
  typedef typename simd::value_type      value_type;

  Simd_unaligned_loader(value_type const* ptr)
  {
    ptr_aligned_    = (value_type*)((intptr_t)ptr & ~(simd::alignment-1));

    x0_  = simd::load((value_type*)ptr_aligned_);
    sh_  = simd::shift_for_addr(ptr);
  }

  simd_type load() const
  {
    x1_  = simd::load((value_type*)(ptr_aligned_+simd::vec_size));
    return simd::perm(x0_, x1_, sh_);
  }

  void increment(length_type n = 1)
  {
    ptr_aligned_   += n * simd::vec_size;
  
    // Update x0.
    //
    // Note: this requires load() to be called at least once before each
    //       call to increment().
    x0_ = (n == 1) ? x1_ : simd::load((value_type*)ptr_aligned_);
  }

  value_type const*            ptr_aligned_;
  simd_type                    x0_;
  mutable simd_type            x1_;
  perm_simd_type               sh_;

};

template <typename T>
struct Simd_unaligned_loader<T, false>
{
  typedef Simd_traits<T>            simd;
  typedef typename simd::simd_type  simd_type;
  typedef typename simd::value_type value_type;

  Simd_unaligned_loader(value_type const* ptr) : ptr_unaligned_(ptr) {}

  simd_type load() const { return simd::load_unaligned(ptr_unaligned_); }

  void increment(length_type n = 1)
  { ptr_unaligned_ += n * Simd_traits<value_type>::vec_size; }

  void increment_by_element(length_type) { assert(0); }

  value_type const*            ptr_unaligned_;
};

template <typename T, bool IsAligned> class Proxy;

// Optimized proxy for direct SIMD access to block data, i.e. the data
// is contiguous (unit stride) and correctly aligned.
template <typename T>
class Proxy<Direct_access_traits<T>, true>
{
public:
  typedef T value_type;
  typedef typename Simd_traits<value_type>::simd_type simd_type;

  Proxy(value_type const *ptr) : ptr_(ptr)
  {}

  simd_type load() const { return Simd_traits<value_type>::load(ptr_);}

  void increment(length_type n = 1)
  { ptr_ += n * Simd_traits<value_type>::vec_size;}

  void increment_by_element(length_type n)
  { ptr_ += n; }

private:
  value_type const *ptr_;
};

template <typename T>
class Proxy<Direct_access_traits<T>,false >
{
public:
  typedef T value_type;
  typedef Simd_traits<value_type>                     simd;
  typedef typename simd::simd_type                    simd_type;

  Proxy(value_type const *ptr) : simd_loader_(ptr) {}

  simd_type load() const
  { return simd_loader_.load(); }

  void increment(length_type n = 1) 
  { simd_loader_.increment(n); }

  void increment_by_element(length_type n)
  { simd_loader_.increment_by_element(n); }

private:
  Simd_unaligned_loader<T>      simd_loader_;
};


// Optimized proxy for direct SIMD access to writable block data, i.e. 
// the data is contiguous (unit stride) and correctly aligned.
template <typename T>
class Proxy<LValue_access_traits<T>,true >
{
public:
  typedef T value_type;
  typedef typename Simd_traits<value_type>::simd_type simd_type;

  Proxy(value_type *ptr) : ptr_(ptr)
  {}

  template <typename T1>
  Proxy operator = (Proxy<T1,true> const &o) 
  {
    store(o.load());
    return *this;
  }

  simd_type 
  load() const { return Simd_traits<value_type>::load(ptr_);}
  void 
  store(simd_type const &value) 
  { Simd_traits<value_type>::store(ptr_, value);}

  void increment(length_type n = 1)
  { ptr_ += n * Simd_traits<value_type>::vec_size;}

  void increment_by_element(length_type n)
  { ptr_ += n; }

private:
  value_type *ptr_;
};

template <typename T, bool IsAligned>
class Proxy<Scalar_access_traits<T>, IsAligned>
{
public:
  typedef T value_type;
  typedef typename Simd_traits<value_type>::simd_type simd_type;

  Proxy(value_type value)
    : value_(Simd_traits<value_type>::load_scalar_all(value))
  {}

  simd_type load() const 
  { return value_; }

  void increment(length_type = 1) {}
  void increment_by_element(length_type) {}

private:
  simd_type value_;
};

// Proxy for unary expressions.
template <typename ProxyT, template <typename> class O, bool IsAligned>
class Proxy<Unary_access_traits<ProxyT, O>, IsAligned>
{
public:
  typedef Unary_access_traits<ProxyT, O> access_traits;
  typedef typename access_traits::value_type value_type;
  typedef typename Simd_traits<value_type>::simd_type simd_type;

  Proxy(ProxyT const &o) : op_(o) {}

  simd_type load() const 
  {
    simd_type op = op_.load();
    return Unary_operator_map<value_type, O>::apply(op);
  }

  void increment(length_type n = 1) { op_.increment(n);}
  void increment_by_element(length_type n) { op_.increment_by_element(n); }

private:
  ProxyT op_;
};

// Proxy for binary expressions. The two proxy operands L and R are 
// combined using binary operator O.
template <typename L, typename R, template <typename, typename> class O,
          bool IsAligned>
class Proxy<Binary_access_traits<L, R, O>, IsAligned>
{
public:
  typedef Binary_access_traits<L, R, O> access_traits;
  typedef typename access_traits::value_type  value_type;
  typedef typename access_traits::return_type return_type;
  typedef typename Simd_traits<value_type>::simd_type  simd_type;
  typedef typename Simd_traits<return_type>::simd_type simd_ret_type;

  Proxy(L const &l, R const &r) : left_(l), right_(r) {}

  L const &left() const { return left_;}
  R const &right() const { return right_;}

  simd_ret_type load() const 
  {
    simd_type l = left_.load();
    simd_type r = right_.load();
    return Binary_operator_map<value_type, O>::apply(l, r);
  }

  void increment(length_type n = 1)
  {
    left_.increment(n);
    right_.increment(n);
  }

  void increment_by_element(length_type n)
  {
    left_.increment_by_element(n);
    right_.increment_by_element(n);
  }

private:
  L left_;
  R right_;
};

// Proxy for ternary 'multiply-add' expression (a * b + c)
template <typename A, typename B, typename C, bool IsAligned>
class Proxy<Binary_access_traits<
	      Proxy<Binary_access_traits<A, B, expr::op::Mult>, IsAligned>,
	      C, expr::op::Add>, IsAligned>
{
public:
  typedef Proxy<Binary_access_traits<A, B, expr::op::Mult>, IsAligned> AB;
  typedef Binary_access_traits<AB, C, expr::op::Add> access_traits;
  typedef typename access_traits::value_type value_type;
  typedef typename Simd_traits<value_type>::simd_type simd_type;

  Proxy(AB const &left, C const &right)
    : left_(left), right_(right) {}

  AB const &left() const { return left_;}
  C const &right() const { return right_;}

  simd_type load() const
  {
    simd_type a = left_.left().load();
    simd_type b = left_.right().load();
    simd_type c = right_.load();
    return Simd_traits<value_type>::fma(a, b, c);
  }

  void increment(length_type n = 1)
  {
    left_.increment(n);
    right_.increment(n);
  }

  void increment_by_element(length_type n)
  {
    left_.increment_by_element(n);
    right_.increment_by_element(n);
  }

private:
  AB left_;
  C right_;
};

// Proxy for ternary 'add-multiply' expression (a + b * c)
template <typename A, typename B, typename C, bool IsAligned>
class Proxy<Binary_access_traits<A,
		 Proxy<Binary_access_traits<B, C, expr::op::Mult>, IsAligned>,
		 expr::op::Add>, IsAligned>
{
public:
  typedef Proxy<Binary_access_traits<B, C, expr::op::Mult>, IsAligned> BC;
  typedef Binary_access_traits<A, BC, expr::op::Add> access_traits;
  typedef typename access_traits::value_type value_type;
  typedef typename Simd_traits<value_type>::simd_type simd_type;

  Proxy(A const &left, BC const &right)
    : left_(left), right_(right) {}

  A const &left() const { return left_;}
  BC const &right() const { return right_;}

  simd_type load() const 
  {
    simd_type a = left_.load();
    simd_type b = right_.left().load();
    simd_type c = right_.right().load();
    return Simd_traits<value_type>::fma(b, c, a);
  }

  void increment(length_type n = 1)
  {
    left_.increment(n);
    right_.increment(n);
  }

  void increment_by_element(length_type n)
  {
    left_.increment_by_element(n);
    right_.increment_by_element(n);
  }

private:
  A left_;
  BC right_;
};

// Proxy for quaternary 'add-multiply' expression (a * b + c * d)
// (needed for disambiguation).
template <typename A, typename B, typename C, typename D, bool IsAligned>
class Proxy<Binary_access_traits<
	Proxy<Binary_access_traits<A, B, expr::op::Mult>, IsAligned>,
	Proxy<Binary_access_traits<C, D, expr::op::Mult>, IsAligned>,
	expr::op::Add>, IsAligned>
{
public:
  typedef Proxy<Binary_access_traits<A, B, expr::op::Mult>, IsAligned> AB;
  typedef Proxy<Binary_access_traits<C, D, expr::op::Mult>, IsAligned> CD;
  typedef Binary_access_traits<AB, CD, expr::op::Add> access_traits;
  typedef typename access_traits::value_type value_type;
  typedef typename Simd_traits<value_type>::simd_type simd_type;

  Proxy(AB const &left, CD const &right)
    : left_(left), right_(right) {}

  AB const &left() const { return left_;}
  CD const &right() const { return right_;}

  simd_type load() const 
  {
    // Implement 'a * b + c * d' as '(a*b) + c * d'.
    simd_type ab = left_.load();
    simd_type c = right_.left().load();
    simd_type d = right_.right().load();
    return Simd_traits<value_type>::fma(c, d, ab);
  }

  void increment(length_type n = 1)
  {
    left_.increment(n);
    right_.increment(n);
  }

  void increment_by_element(length_type n)
  {
    left_.increment_by_element(n);
    right_.increment_by_element(n);
  }

private:
  AB left_;
  CD right_;
};

// Proxy for ternary access traits
template <typename A, typename B, typename C,
          template <typename,typename,typename> class O,
	  bool IsAligned>
class Proxy<Ternary_access_traits<A,B,C,O>, IsAligned>
{
  typedef typename A::access_traits                   access_traits;
  typedef typename access_traits::value_type          value_type;
  typedef typename Simd_traits<value_type>::simd_type simd_type;

public:
  Proxy(A const &a, B const &b, C const &c)
    : a_(a), b_(b), c_(c) {}

  simd_type load() const 
  {
    typedef typename A::access_traits::return_type return_type;
    typedef typename A::access_traits::value_type  value_type;
    typedef typename Simd_traits<return_type>::simd_type simd_ret_type;
    typedef typename Simd_traits<value_type>::simd_type  simd_val_type;
    
    simd_ret_type a_ret  = a_.load(); // this is the mask
    simd_val_type  b     = b_.load(); // if true
    simd_val_type  c     = c_.load(); // if false
    // apply
    return Ternary_operator_map<value_type,O>::apply(a_ret,b,c);
  }

  void increment(length_type n = 1)
  {
    a_.increment(n);
    b_.increment(n);
    c_.increment(n);
  }

  void increment_by_element(length_type n)
  {
    a_.increment_by_element(n);
    b_.increment_by_element(n);
    c_.increment_by_element(n);
  }

private:
  A a_;
  B b_;
  C c_;
};

/*
template <typename T>
struct Iterator
{
public:
  Iterator(Proxy<T> const &c) : cursor_(c) {}
  bool operator== (Iterator const &i) const { return cursor_ == i.cursor_;}
  bool operator!= (Iterator const &i) const { return !(*this==i);}
  Proxy<T> &operator* () { return cursor_;}
  Proxy<T> *operator-> () { return &cursor_;}
  Iterator &operator++() { cursor_.increment(); return *this;}
  Iterator operator++(int) { Iterator i(*this); cursor_.increment(); return i;}
  Iterator &operator+=(length_type n) { cursor_.increment(n); return *this;}

private:
  Proxy<T> cursor_;
};

template <typename T>
inline Iterator<T> 
operator+(Iterator<T> const i, length_type n) 
{
  Iterator<T> r(i);
  r += n;
  return r;
}
*/

} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip

#endif
