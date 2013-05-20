//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_vector_hpp_
#define vsip_impl_vector_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/dense.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/expr.hpp>
#include <ovxx/element_proxy.hpp>
#include <ovxx/view/traits.hpp>
#include <ovxx/assign.hpp>
#include <ovxx/view.hpp>

namespace vsip
{

/// View which appears as a one-dimensional, read-only vector.
template <typename T, typename Block>
class const_Vector : public ovxx::const_View<const_Vector,Block>
{
  typedef ovxx::const_View<vsip::const_Vector,Block> base_type;
  typedef typename ovxx::lvalue_factory_type<Block, 1>::type ref_factory;

public:
  static dimension_type const dim = 1;
  typedef Block block_type;
  typedef typename block_type::value_type value_type;
  typedef typename ref_factory::reference_type reference_type;
  typedef typename ref_factory::const_reference_type const_reference_type;

protected:
  typedef typename block_type::map_type  map_type;
  typedef ovxx::expr::Subset<block_type> impl_subblock_type;

public:
  // [view.vector.subview_types]
  typedef const_Vector<T, impl_subblock_type> subview_type;
  typedef const_Vector<T, impl_subblock_type> const_subview_type;

  typedef typename base_type::impl_const_view_type impl_const_view_type;

  // [view.vector.constructors]

  const_Vector(length_type len, T const &value, map_type const &map = map_type())
    : base_type(new block_type(Domain<1>(len), value, map), false)
  {}
  explicit const_Vector(length_type len, map_type const &map = map_type())
    : base_type(new block_type(Domain<1>(len), map), false)
  {}
  const_Vector(Block &block) VSIP_NOTHROW : base_type (&block) {}
  const_Vector(const_Vector const &v) VSIP_NOTHROW
    : base_type(&v.block())
  { ovxx::expr::evaluate(this->block());}
  ~const_Vector() VSIP_NOTHROW {}

  // [view.vector.valaccess]
  value_type get(index_type i) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < this->size(0));
    return this->block().get(i);
  }

  // Supported for some, but not all, underlying Blocks.
  const_reference_type operator()(index_type i) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < this->size(0));
    return ref_factory::ref(this->block(), i);
  }

  // [view.vector.subviews]
  const_subview_type
  get(Domain<1> const& dom) const VSIP_THROW((std::bad_alloc))
  {
    impl_subblock_type block(dom, this->block());
    return const_subview_type(block);
  }
  const_subview_type
  operator()(Domain<1> const& dom) const VSIP_THROW((std::bad_alloc))
  {
    impl_subblock_type block(dom, this->block());
    return const_subview_type(block);
  }
  const_Vector const &
  operator()(whole_domain_type) const { return *this;}

  // [view.vector.accessors]
  length_type length() const VSIP_NOTHROW { return this->block().size();}
};



#define VSIP_IMPL_ELEMENTWISE_SCALAR(op)        		   \
  *this = *this op val

#define VSIP_IMPL_ELEMENTWISE_SCALAR_NOFWD(op)			   \
  for (index_type i = 0; i < this->size(); i++)			   \
    this->put(i, this->get(i) op val)

#define VSIP_IMPL_ELEMENTWISE_VECTOR(op)	                   \
  *this = *this op v;

#define VSIP_IMPL_ELEMENTWISE_VECTOR_NOFWD(op)			   \
  for (index_type i = (assert(this->size() == v.size()), 0);	   \
       i < this->size(); i++)					   \
    this->put(i, this->get(i) op v.get(i))
  
#define VSIP_IMPL_ASSIGN_OP(asop, op)			   	   \
  template <typename T0>                                           \
  Vector& operator asop(T0 const& val) VSIP_NOTHROW                \
  { VSIP_IMPL_ELEMENTWISE_SCALAR(op); return *this;}               \
  template <typename T0, typename Block0>                          \
  Vector& operator asop(const_Vector<T0, Block0> v) VSIP_NOTHROW  \
  { VSIP_IMPL_ELEMENTWISE_VECTOR(op); return *this;}               \
  template <typename T0, typename Block0>                          \
  Vector& operator asop(const Vector<T0, Block0> v) VSIP_NOTHROW  \
  { VSIP_IMPL_ELEMENTWISE_VECTOR(op); return *this;}

#define VSIP_IMPL_ASSIGN_OP_NOFWD(asop, op)			   \
  template <typename T0>                                           \
  Vector& operator asop(T0 const& val) VSIP_NOTHROW                \
  { VSIP_IMPL_ELEMENTWISE_SCALAR_NOFWD(op); return *this;}	   \
  template <typename T0, typename Block0>                          \
  Vector& operator asop(const_Vector<T0, Block0> v) VSIP_NOTHROW   \
  { VSIP_IMPL_ELEMENTWISE_VECTOR_NOFWD(op); return *this;}	   \
  template <typename T0, typename Block0>                          \
  Vector& operator asop(const Vector<T0, Block0> v) VSIP_NOTHROW   \
  { VSIP_IMPL_ELEMENTWISE_VECTOR_NOFWD(op); return *this;}


/// View which appears as a one-dimensional, modifiable vector.  This
/// inherits from const_Vector, so only the members that const_Vector
/// does not carry, or that are different, need be specified.
template <typename T, typename Block>
class Vector : public ovxx::View<Vector,Block>
{
  typedef ovxx::View<vsip::Vector,Block> base_type;
  typedef typename ovxx::lvalue_factory_type<Block, 1>::type ref_factory;

  // Implementation compile-time values.
  typedef typename Block::map_type             map_type;
  // [view.vector.subview_types]
  // Override subview_type and make it writable.
  typedef ovxx::expr::Subset<Block> impl_subblock_type;

public:
  // Compile-time values.
  static dimension_type const dim = 1;
  typedef Block block_type;
  typedef typename block_type::value_type value_type;
  typedef typename ref_factory::reference_type reference_type;
  typedef typename ref_factory::const_reference_type
    const_reference_type;

public:
  typedef Vector<T, impl_subblock_type>      subview_type;

  // [view.vector.constructors]
  Vector(length_type len, T const &value, map_type const &map = map_type())
    : base_type(len, value, map, ovxx::detail::disambiguate)
  {}
  explicit Vector(length_type len, map_type const &map = map_type())
    : base_type(len, map)
  {}
  Vector(Block &block) VSIP_NOTHROW : base_type(block) {}
  Vector(Vector const &v) VSIP_NOTHROW : base_type(v.block()) {}
  template <typename T0, typename Block0>
  Vector(const_Vector<T0, Block0> const& v) VSIP_NOTHROW
    : base_type(v.length(), v.block().map())
  { *this = v;}
  ~Vector() VSIP_NOTHROW {}

  // [view.vector.valaccess]
  void put(index_type i, value_type val) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < this->size(0));
    this->block().put(i, val);
  }

  reference_type operator()(index_type i) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < this->size(0));
    return ref_factory::ref(this->block(), i);
  }

  template <typename T0, typename Block0>
  Vector(Vector<T0, Block0> const &v)
    : base_type(v.size(), map_type())
  { *this = v;}

  Vector &operator=(Vector const &v) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(this->size() == v.size());
    ovxx::assign<1>(this->block(), v.block());
    return *this;
  }

  Vector &operator=(const_reference_type val) VSIP_NOTHROW
  {
    ovxx::expr::Scalar<1, T> scalar(val);
    ovxx::assign<1>(this->block(), scalar);
    return *this;
  }
  template <typename T0>
  Vector &operator=(T0 const &val) VSIP_NOTHROW
  {
    ovxx::expr::Scalar<1, T0> scalar(val);
    ovxx::assign<1>(this->block(), scalar);
    return *this;
  }
  template <typename T0, typename Block0>
  Vector &operator=(const_Vector<T0, Block0> const &v) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(this->size() == v.size());
    ovxx::assign<1>(this->block(), v.block());
    return *this;
  }
  template <typename T0, typename Block0>
  Vector &operator=(Vector<T0, Block0> const &v) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(this->size() == v.size());
    ovxx::assign<1>(this->block(), v.block());
    return *this;
  }

  // [view.vector.subviews]
  subview_type
  operator()(Domain<1> const &dom) VSIP_THROW((std::bad_alloc))
  {
    impl_subblock_type block(dom, this->block());
    return subview_type(block);
  }
  Vector const &operator()(whole_domain_type) const { return *this;}
  Vector &operator()(whole_domain_type) { return *this;}

  // [view.vector.assign]
  VSIP_IMPL_ASSIGN_OP(+=, +)
  VSIP_IMPL_ASSIGN_OP(-=, -)
  VSIP_IMPL_ASSIGN_OP(*=, *)
  VSIP_IMPL_ASSIGN_OP(/=, /)
  // ghs claims the use of operator& in 'view1 & view2' is ambiguous,
  // thus we implement operator&= in terms of the scalar operator&.
  // Likewise for operator=| and operator=^.
  VSIP_IMPL_ASSIGN_OP_NOFWD(&=, &)
  VSIP_IMPL_ASSIGN_OP_NOFWD(|=, |)
  VSIP_IMPL_ASSIGN_OP_NOFWD(^=, ^)

};

#undef VSIP_IMPL_ASSIGN_OP
#undef VSIP_IMPL_ELEMENTWISE_SCALAR
#undef VSIP_IMPL_ELEMENTWISE_VECTOR
#undef VSIP_IMPL_ASSIGN_OP_NOFWD
#undef VSIP_IMPL_ELEMENTWISE_SCALAR_NOFWD
#undef VSIP_IMPL_ELEMENTWISE_VECTOR_NOFWD

// [view.vector.convert]
template <typename T, typename Block>
struct ViewConversion<Vector, T, Block>
{
  typedef const_Vector<T, Block> const_view_type;
  typedef Vector<T, Block> view_type;
};

template <typename T, typename Block>
struct ViewConversion<const_Vector, T, Block>
{
  typedef const_Vector<T, Block> const_view_type;
  typedef Vector<T, Block> view_type;
};

} // namespace vsip

namespace ovxx
{
template <typename B>
struct view_of<B, 1>
{
  typedef Vector<typename B::value_type, B> type;
  typedef const_Vector<typename B::value_type, B> const_type;
};

template <typename T, typename Block> 
struct is_view_type<Vector<T, Block> >
{
  typedef Vector<T, Block> type;
  static bool const value = true;
};

template <typename T, typename Block> 
struct is_view_type<const_Vector<T, Block> >
{
  typedef const_Vector<T, Block> type;
  static bool const value = true;
};

template <typename T, typename Block> 
struct is_const_view_type<const_Vector<T, Block> >
{
  typedef const_Vector<T, Block> type;
  static bool const value = true;
};

template <typename T, typename Block>
T
get(const_Vector<T, Block> view, Index<1> const &i)
{
  return view.get(i[0]);
}

template <typename T, typename Block>
void
put(Vector<T, Block> view, Index<1> const &i, T value)
{
  view.put(i[0], value);
}

/// Return the view extent as a domain.
template <typename T,
	  typename Block>
inline Domain<1>
view_domain(const_Vector<T, Block> const& view)
{
  return Domain<1>(view.size(0));
}

/// Get the extent of a vector view, as a Length. 
template <typename T,
	  typename Block>
inline Length<1>
extent(const_Vector<T, Block> const &v)
{
  return Length<1>(v.size(0));
}

} // namespace ovxx

#endif
