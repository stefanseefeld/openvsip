//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_LVALUE_PROXY_HPP
#define VSIP_CORE_LVALUE_PROXY_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/noncopyable.hpp>
#include <vsip/core/block_traits.hpp>

/***********************************************************************
  Declarations
***********************************************************************/
  
namespace vsip
{
namespace impl
{

namespace lvalue_detail
{

/// A mix-in template which provides `+=`, `-=`, `*=`, `/=` operators for any
/// class T that defines T::value_type, conversion to T::value_type,
/// and conversion or assignment from T::value_type.  Note that in
/// most uses, T::value_type will have to be given explicitly as the
/// second template argument.
template <typename T, typename V = typename T::value_type>
struct Modify_operators
{
  typedef V value_type;
  value_type operator+= (value_type n)
    { T& self = static_cast<T&>(*this); return self = self + n; }
  value_type operator-= (value_type n)
    { T& self = static_cast<T&>(*this); return self = self - n; }
  value_type operator*= (value_type n)
    { T& self = static_cast<T&>(*this); return self = self * n; }
  value_type operator/= (value_type n)
    { T& self = static_cast<T&>(*this); return self = self / n; }
};

template <typename Block>
inline typename Block::value_type
get(
  Block const& block,
  Index<1> const& idx)
{
  return block.get(idx[0]);
}

template <typename Block>
inline typename Block::value_type
get(
  Block const& block,
  Index<2> const& idx)
{
  return block.get(idx[0], idx[1]);
}

template <typename Block>
inline typename Block::value_type
get(
  Block const& block,
  Index<3> const& idx)
{
  return block.get(idx[0], idx[1], idx[2]);
}

template <typename Block>
void
put(
  Block&                     block,
  Index<1> const&            idx,
  typename Block::value_type value)
{
  block.put(idx[0], value);
}

template <typename Block>
void
put(
  Block&                     block,
  Index<2> const&            idx,
  typename Block::value_type value)
{
  block.put(idx[0], idx[1], value);
}
template <typename Block>
void
put(
  Block&                     block,
  Index<3> const&            idx,
  typename Block::value_type value)
{
  block.put(idx[0], idx[1], idx[2], value);
}

} // namespace lvalue_detail



/// The generic lvalue proxy class.  Note that the default copy
/// constructor and destructor are correct for this class.
template <typename       T,
	  typename       Block,
	  dimension_type Dim>
class Lvalue_proxy
  : public lvalue_detail::Modify_operators<Lvalue_proxy<T, Block, Dim>,
					   typename Block::value_type>
{
  // Type members.
public:
  typedef Block                        block_type;
  typedef typename Block::value_type   value_type;

  // Data members.
protected:
  block_type& block_;
  Index<Dim> coord_;
  
public:
  /// Constructor.
  Lvalue_proxy (block_type& b, Index<Dim> const& i)
    : block_(b), coord_(i) {};

  /// Read access, by implicit conversion to the value type.
  operator value_type() const
    { return lvalue_detail::get(block_, coord_); }

  /// Write access, by assignment from the value type.
  Lvalue_proxy& operator= (value_type v)
    { lvalue_detail::put(block_, coord_, v); return *this; }

  /// Write access, by assignment from another instance of this class.
  Lvalue_proxy& operator= (Lvalue_proxy& other)
    { return *this = value_type(other); }
};



/// Lvalue proxy specialization for complex.
template <typename       T,
	  typename       Block,
	  dimension_type Dim>
class Lvalue_proxy<std::complex<T>, Block, Dim>
  : public std::complex<T>
{
  typedef std::complex<T> base_type;

  // Type members.
public:
  typedef Block                        block_type;
  typedef typename Block::value_type   value_type;

  // Data members.
protected:
  block_type& block_;
  Index<Dim> coord_;
  
public:
  /// Constructor.
  Lvalue_proxy(block_type& b, Index<Dim> const& i)
    : base_type(lvalue_detail::get(b, i)),
      block_(b), coord_(i)
  {};

  /// Since proxy derives from value type, implicit conversion
  /// is not necessary.

  /// Write access, by assignment from the value type.
  Lvalue_proxy& operator= (value_type const& v)
  {
    this->base_type::operator=(v);
    lvalue_detail::put(block_, coord_, v);
    return *this;
  }

  /// Write access, by assignment from another instance of this class.
  Lvalue_proxy& operator= (Lvalue_proxy& other)
    { return *this = value_type(other); }


  // Trying to mix these in with Modify_operators creates an
  // ambiguity with the same operators derived from std::complex.
  // Therefor, we define them in class.

  value_type operator+= (value_type n)
    { return *this = *this + n; }
  value_type operator-= (value_type n)
    { return *this = *this - n; }
  value_type operator*= (value_type n)
    { return *this = *this * n; }
  value_type operator/= (value_type n)
    { return *this = *this / n; }
};



/// Proxy_lvalue_factory takes an arbitrary Block and generates
/// Lvalue_proxy instances from it.
template <typename Block>
class Proxy_lvalue_factory : Non_copyable
{
  /// The block we have been asked to produce proxy lvalues for.  As
  /// this object is extremely short-lived, we do not bother counting
  /// this reference.
  Block& block_;

  typedef typename Block::value_type value_type;
  static dimension_type const dim = Block::dim;

public:
  /// The type of the reference that will be returned.
  typedef Lvalue_proxy<value_type, Block, dim>       reference_type;
  typedef Lvalue_proxy<value_type, Block, dim> const const_reference_type;

  /// Constructor.
  Proxy_lvalue_factory (Block& b) : block_(b) {}

  /// Retrieve the proxy.
  reference_type impl_ref(index_type i)
    { return Lvalue_proxy<value_type, Block, dim>(block_, Index<1>(i)); }
  reference_type impl_ref(index_type i, index_type j)
    { return Lvalue_proxy<value_type, Block, dim>(block_, Index<2>(i, j)); }
  reference_type impl_ref(index_type i, index_type j, index_type k)
    { return Lvalue_proxy<value_type, Block, dim>(block_, Index<3>(i, j, k)); }
};

/// True_lvalue_factory takes a Block that implements impl_ref(),
/// and generates true lvalues from it by applying that function.
template <typename Block>
class True_lvalue_factory : Non_copyable
{
  /// The block we have been asked to produce proxy lvalues for.  As
  /// this object is extremely short-lived, we do not bother counting
  /// this reference.
  Block& block_;

public:
  /// The type of the reference that will be returned.
  typedef typename Block::reference_type       reference_type;
  typedef typename Block::const_reference_type const_reference_type;

  /// Constructor.
  True_lvalue_factory (Block& b) : block_(b) {}

  /// Retrieve the proxy.
  reference_type impl_ref(index_type i)
    { return block_.impl_ref(i); }
  reference_type impl_ref(index_type i, index_type j)
    { return block_.impl_ref(i, j); }
  reference_type impl_ref(index_type i, index_type j, index_type k)
    { return block_.impl_ref(i, j, k); }
};



// Trait to determine if type is an lvalue_proxy.

template <typename T>
struct Is_lvalue_proxy_type
{
  static bool const value = false;
  typedef T value_type;
};

template <typename       T,
	  typename       BlockT,
	  dimension_type Dim>
struct Is_lvalue_proxy_type<Lvalue_proxy<T, BlockT, Dim> >
{
  static bool const value = true;
  typedef T value_type;
};

} // namespace vsip::impl
} // namespace vsip

#endif
