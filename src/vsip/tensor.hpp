//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_tensor_hpp_
#define vsip_tensor_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/dense.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>

namespace vsip
{

/// View which appears as a three-dimensional, read-only tensor.
template <typename T, typename Block>
class const_Tensor : public ovxx::const_View<const_Tensor,Block>
{
  typedef ovxx::const_View<vsip::const_Tensor,Block> base_type;
  typedef typename ovxx::lvalue_factory_type<Block, 3>::type ref_factory;

  // [view.tensor.subview_types]
protected:
  typedef typename Block::map_type map_type;
  typedef ovxx::expr::Subset<Block> impl_subblock_type;

public:
  static const dimension_type dim = 3;
  typedef Block block_type;
  typedef typename block_type::value_type value_type;
  typedef typename ref_factory::reference_type reference_type;
  typedef typename ref_factory::const_reference_type
    const_reference_type;

  typedef vsip::whole_domain_type whole_domain_type;
  static whole_domain_type const whole_domain = vsip::whole_domain;

  typedef typename base_type::impl_const_view_type impl_const_view_type;

  typedef const_Tensor<T, impl_subblock_type> subview_type;
  typedef const_Tensor<T, impl_subblock_type> const_subview_type;

  template <dimension_type D0, dimension_type D1, dimension_type D2>
  struct transpose_view
  {
    typedef ovxx::expr::Permuted<Block, tuple<D0, D1, D2> > block_type;
    typedef const_Tensor<T, block_type> type;
    typedef const_Tensor<T, block_type> const_type;
  }; 

  template <dimension_type D>
  struct submatrix : ovxx::ct_assert<(D < 3)>
  {
    typedef ovxx::expr::Sliced<Block, D> block_type;
    typedef const_Matrix<T, block_type> type;
    typedef const_Matrix<T, block_type> const_type;

    typedef ovxx::expr::Subset<block_type> subset_block_type;
    typedef const_Matrix<T, subset_block_type> subset_type;
    typedef const_Matrix<T, subset_block_type> const_subset_type;
  };

  template <dimension_type D1, dimension_type D2>
  struct subvector : ovxx::ct_assert<(D1 < D2 && D2 < 3)>
  {
    typedef ovxx::expr::Sliced2<Block, D1, D2> block_type;
    typedef const_Vector<T, block_type> type;
    typedef const_Vector<T, block_type> const_type;

    typedef ovxx::expr::Subset<block_type> subset_block_type;
    typedef const_Vector<T, subset_block_type> subset_type;
    typedef const_Vector<T, subset_block_type> const_subset_type;
  };

  // [view.tensor.constructors]
  const_Tensor(length_type i, length_type j, length_type k, T const &value,
	       map_type const &map = map_type())
    : base_type(new block_type(Domain<3>(i, j, k), value, map), false)
  {}
  const_Tensor(length_type i, length_type j, length_type k,
	       map_type const &map = map_type())
    : base_type(new block_type(Domain<3>(i, j, k), map), false)
  {}
  const_Tensor(Block &block) VSIP_NOTHROW : base_type(&block) {}
  const_Tensor(const_Tensor const &v) VSIP_NOTHROW : base_type(&v.block()) {}
  ~const_Tensor() VSIP_NOTHROW {}

  // [view.tensor.transpose]
  template <dimension_type D0, dimension_type D1, dimension_type D2>
  typename transpose_view<D0, D1, D2>::const_type  
  transpose() const VSIP_NOTHROW
  {
    typename transpose_view<D0, D1, D2>::block_type  block(this->block());
    return typename transpose_view<D0, D1, D2>::const_type(block);
  }

  // [view.tensor.valaccess]
  value_type get(index_type i, index_type j, index_type k) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < this->size(0));
    OVXX_PRECONDITION(j < this->size(1));
    OVXX_PRECONDITION(k < this->size(2));
    return this->block().get(i, j, k);
  }

  // Supported for some, but not all, underlying Blocks.
  const_reference_type 
  operator()(index_type i, index_type j, index_type k) const VSIP_NOTHROW
  { return ref_factory::ref(this->block(), i, j, k);}

  // [view.tensor.subviews]
  const_subview_type get(Domain<3> const &dom) const VSIP_THROW((std::bad_alloc))
  {
    impl_subblock_type block(dom, this->block());
    return const_subview_type(block);
  }

  const_subview_type operator()(Domain<3> const &dom)
    const VSIP_THROW((std::bad_alloc))
  {
    impl_subblock_type block(dom, this->block());
    return const_subview_type(block);
  }

  typename subvector<1, 2>::const_subset_type
  operator()(Domain<1> const &d, index_type j, index_type k)
    const VSIP_THROW((std::bad_alloc))
  {
    typename subvector<1, 2>::block_type block(this->block(), j, k);
    typename subvector<1, 2>::subset_block_type sblock(d, block);
    return typename subvector<1, 2>::const_subset_type(sblock);
  }
  typename subvector<0, 2>::const_subset_type
  operator()(index_type i, Domain<1> const& d, index_type k)
    const VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 2>::block_type block(this->block(), i, k);
    typename subvector<0, 2>::subset_block_type sblock(d, block);
    return typename subvector<0, 2>::const_subset_type(sblock);
  }
  typename subvector<0, 1>::const_subset_type
  operator()(index_type i, index_type j, Domain<1> const& d)
    const VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 1>::block_type block(this->block(), i, j);
    typename subvector<0, 1>::subset_block_type sblock(d, block);
    return typename subvector<0, 1>::const_subset_type(sblock);
  }
  typename submatrix<0>::const_subset_type
  operator()(index_type i, Domain<1> const& d1, Domain<1> const& d2)
    const VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<0>::block_type block(this->block(), i);
    typename submatrix<0>::subset_block_type sblock(Domain<2>(d1, d2), block);
    return typename submatrix<0>::const_subset_type(sblock);
  }
  typename submatrix<1>::const_subset_type
  operator()(Domain<1> const& d1, index_type j, Domain<1> const& d2)
    const VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<1>::block_type block(this->block(), j);
    typename submatrix<1>::subset_block_type sblock(Domain<2>(d1, d2), block);
    return typename submatrix<1>::const_subset_type(sblock);
  }
  typename submatrix<2>::const_subset_type
  operator()(Domain<1> const& d1, Domain<1> const& d2, index_type k)
    const VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<2>::block_type block(this->block(), k);
    typename submatrix<2>::subset_block_type sblock(Domain<2>(d1, d2), block);
    return typename submatrix<2>::const_subset_type(sblock);
  }

  typename subvector<1, 2>::const_type
  operator()(whole_domain_type, index_type j, index_type k)
    const VSIP_THROW((std::bad_alloc))
  {
    typename subvector<1, 2>::block_type block(this->block(), j, k);
    return typename subvector<1, 2>::const_type(block);
  }
  typename subvector<0, 2>::const_type
  operator()(index_type i, whole_domain_type, index_type k)
    const VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 2>::block_type block(this->block(), i, k);
    return typename subvector<0, 2>::const_type(block);
  }
  typename subvector<0, 1>::const_type
  operator()(index_type i, index_type j, whole_domain_type)
    const VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 1>::block_type block(this->block(), i, j);
    return typename subvector<0, 1>::const_type(block);
  }
  typename submatrix<0>::const_type
  operator()(index_type i, whole_domain_type, whole_domain_type)
    const VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<0>::block_type block(this->block(), i);
    return typename submatrix<0>::const_type(block);
  }
  typename submatrix<1>::const_type
  operator()(whole_domain_type, index_type j, whole_domain_type)
    const VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<1>::block_type block(this->block(), j);
    return typename submatrix<1>::const_type(block);
  }
  typename submatrix<2>::const_type
  operator()(whole_domain_type, whole_domain_type, index_type k)
    const VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<2>::block_type block(this->block(), k);
    return typename submatrix<2>::const_type(block);
  }
};


/// View which appears as a three-dimensional, modifiable tensor.  This
/// inherits from const_Tensor, so only the members that const_Tensor
/// does not carry, or that are different, need be specified.
template <typename T, typename Block>
class Tensor : public ovxx::View<Tensor, Block>
{
  typedef ovxx::View<vsip::Tensor, Block> base_type;
  typedef typename ovxx::lvalue_factory_type<Block, 3>::type ref_factory;

protected:
  typedef typename Block::map_type map_type;

  // [view.tensor.subview_types]
  // Override subview_type and make it writable.
  typedef ovxx::expr::Subset<Block> impl_subblock_type;

public:
  typedef typename const_Tensor<T, Block>::whole_domain_type whole_domain_type;
  typedef Tensor<T, impl_subblock_type>      subview_type;

  static const dimension_type dim = 3;
  typedef Block block_type;
  typedef typename block_type::value_type value_type;
  typedef typename ref_factory::reference_type reference_type;
  typedef typename ref_factory::const_reference_type
    const_reference_type;

  template <dimension_type D0, dimension_type D1, dimension_type D2>
  struct transpose_view
  {
    typedef ovxx::expr::Permuted<Block, tuple<D0, D1, D2> > block_type;
    typedef       Tensor<T, block_type> type;
    typedef const_Tensor<T, block_type> const_type;
  }; 

  template <dimension_type D>
  struct submatrix : ovxx::ct_assert<(D < 3)>
  {
    typedef ovxx::expr::Sliced<Block, D> block_type;
    typedef Matrix<T, block_type> type;
    typedef const_Matrix<T, block_type> const_type;

    typedef ovxx::expr::Subset<block_type> subset_block_type;
    typedef Matrix<T, subset_block_type> subset_type;
    typedef const_Matrix<T, subset_block_type> const_subset_type;
  };

  template <dimension_type D1, dimension_type D2>
  struct subvector : ovxx::ct_assert<(D1 < D2 && D2 < 3)>
  {
    typedef ovxx::expr::Sliced2<Block, D1, D2> block_type;
    typedef Vector<T, block_type> type;
    typedef const_Vector<T, block_type> const_type;

    typedef ovxx::expr::Subset<block_type> subset_block_type;
    typedef Vector<T, subset_block_type> subset_type;
    typedef const_Vector<T, subset_block_type> const_subset_type;
  };

  // [view.tensor.constructors]
  Tensor(length_type i, length_type j, length_type k, const T &value,
	 map_type const &map = map_type())
    : base_type(i, j, k, value, map, ovxx::detail::disambiguate)
  {}
  Tensor(length_type i, length_type j, length_type k,
	 map_type const &map = map_type())
    : base_type(i, j, k, map)
  {}
  Tensor(Block &block) VSIP_NOTHROW : base_type(block) {}
  Tensor(Tensor const &t) VSIP_NOTHROW : base_type(t.block()) {}
  template <typename T0, typename Block0>
  Tensor(const_Tensor<T0, Block0> const &t) VSIP_NOTHROW
    : base_type(t.size(0), t.size(1), t.size(2), t.block().map())
  { *this = t;}
  template <typename T0, typename Block0>
  Tensor(Tensor<T0, Block0> const &t)
    : base_type(t.size(0), t.size(1), t.size(2), t.block().map())
  { *this = t;}
  ~Tensor() VSIP_NOTHROW {}

  // [view.tensor.transpose]
  template <dimension_type D0, dimension_type D1, dimension_type D2>
  typename transpose_view<D0, D1, D2>::const_type
  transpose() const VSIP_NOTHROW
  {
    return base_type::transpose();
  }
  template <dimension_type D0, dimension_type D1, dimension_type D2>
  typename transpose_view<D0, D1, D2>::type
  transpose() VSIP_NOTHROW
  {
    typename transpose_view<D0, D1, D2>::block_type block(this->block());
    return typename transpose_view<D0, D1, D2>::type(block);
  }

  // [view.tensor.valaccess]
  void put(index_type i,
	   index_type j,
	   index_type k,
	   value_type val) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < this->size(0));
    OVXX_PRECONDITION(j < this->size(1));
    OVXX_PRECONDITION(k < this->size(2));
    this->block().put(i, j, k, val);
  }

  reference_type operator()(index_type i,
                            index_type j,
                            index_type k) VSIP_NOTHROW
  { return ref_factory::ref(this->block(), i, j, k);}

  Tensor& operator=(Tensor const& t) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(this->size(0) == t.size(0) &&
	   this->size(1) == t.size(1) &&
	   this->size(2) == t.size(2));
    ovxx::assign<3>(this->block(), t.block());
    return *this;
  }

  Tensor& operator=(const_reference_type val) VSIP_NOTHROW
  {
    for (index_type i=0; i<this->size(0); ++i)
      for (index_type j=0; j<this->size(1); ++j)
	for (index_type k=0; k<this->size(2); ++k)
	  this->put(i, j, k, val);
    return *this;
  }
  template <typename T0>
  Tensor& operator=(T0 const& val) VSIP_NOTHROW
  {
    for (index_type i=0; i<this->size(0); ++i)
      for (index_type j=0; j<this->size(1); ++j)
	for (index_type k=0; k<this->size(2); ++k)
	  this->put(i, j, k, val);
    return *this;
  }
  template <typename T0, typename Block0>
  Tensor& operator=(const_Tensor<T0, Block0> const& t) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(this->size(0) == t.size(0) &&
	   this->size(1) == t.size(1) &&
	   this->size(2) == t.size(2));
    ovxx::assign<3>(this->block(), t.block());
    return *this;
  }
  template <typename T0, typename Block0>
  Tensor& operator=(Tensor<T0, Block0> const& t) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(this->size(0) == t.size(0) &&
	   this->size(1) == t.size(1) &&
	   this->size(2) == t.size(2));
    ovxx::assign<3>(this->block(), t.block());
    return *this;
  }

  // [view.tensor.subviews]
  using base_type::operator(); // Pull in all the const versions.
  subview_type
  operator()(const Domain<3>& dom) VSIP_THROW((std::bad_alloc))
  {
    impl_subblock_type block(dom, this->block());
    return subview_type(block);
  }

  typename subvector<1, 2>::subset_type
  operator()(Domain<1> const& d, index_type j, index_type k)
    VSIP_THROW((std::bad_alloc))
  {
    typename subvector<1, 2>::block_type block(this->block(), j, k);
    typename subvector<1, 2>::subset_block_type sblock(d, block);
    return typename subvector<1, 2>::subset_type(sblock);
  }
  typename subvector<0, 2>::subset_type
  operator()(index_type i, Domain<1> const& d, index_type k)
    VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 2>::block_type block(this->block(), i, k);
    typename subvector<0, 2>::subset_block_type sblock(d, block);
    return typename subvector<0, 2>::subset_type(sblock);
  }
  typename subvector<0, 1>::subset_type
  operator()(index_type i, index_type j, Domain<1> const& d)
    VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 1>::block_type block(this->block(), i, j);
    typename subvector<0, 1>::subset_block_type sblock(d, block);
    return typename subvector<0, 1>::subset_type(sblock);
  }
  typename submatrix<0>::subset_type
  operator()(index_type i, Domain<1> const& d1, Domain<1> const& d2)
    VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<0>::block_type block(this->block(), i);
    typename submatrix<0>::subset_block_type sblock(Domain<2>(d1, d2), block);
    return typename submatrix<0>::subset_type(sblock);
  }
  typename submatrix<1>::subset_type
  operator()(Domain<1> const& d1, index_type j, Domain<1> const& d2)
    VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<1>::block_type block(this->block(), j);
    typename submatrix<1>::subset_block_type sblock(Domain<2>(d1, d2), block);
    return typename submatrix<1>::subset_type(sblock);
  }
  typename submatrix<2>::subset_type
  operator()(Domain<1> const& d1, Domain<1> const& d2, index_type k)
    VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<2>::block_type block(this->block(), k);
    typename submatrix<2>::subset_block_type sblock(Domain<2>(d1, d2), block);
    return typename submatrix<2>::subset_type(sblock);
  }

  typename subvector<1, 2>::type
  operator()(whole_domain_type, index_type j, index_type k)
    VSIP_THROW((std::bad_alloc))
  {
    typename subvector<1, 2>::block_type block(this->block(), j, k);
    return typename subvector<1, 2>::type(block);
  }
  typename subvector<0, 2>::type
  operator()(index_type i, whole_domain_type, index_type k)
    VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 2>::block_type block(this->block(), i, k);
    return typename subvector<0, 2>::type(block);
  }
  typename subvector<0, 1>::type
  operator()(index_type i, index_type j, whole_domain_type)
    VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 1>::block_type block(this->block(), i, j);
    return typename subvector<0, 1>::type(block);
  }
  typename submatrix<0>::type
  operator()(index_type i, whole_domain_type, whole_domain_type)
    VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<0>::block_type block(this->block(), i);
    return typename submatrix<0>::type(block);
  }
  typename submatrix<1>::type
  operator()(whole_domain_type, index_type j, whole_domain_type)
    VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<1>::block_type block(this->block(), j);
    return typename submatrix<1>::type(block);
  }
  typename submatrix<2>::type
  operator()(whole_domain_type, whole_domain_type, index_type k)
    VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<2>::block_type block(this->block(), k);
    return typename submatrix<2>::type(block);
  }
#define VSIP_IMPL_ELEMENTWISE_SCALAR(op)				\
  for (index_type i = 0; i < this->size(0); ++i)			\
    for (index_type j = 0; j < this->size(1); ++j)			\
      for (index_type k = 0; k < this->size(2); ++k)			\
	this->put(i, j, k, this->get(i, j, k) op val)

#define VSIP_IMPL_ELEMENTWISE_TENSOR(op)				\
  OVXX_PRECONDITION(this->size(0) == m.size(0) &&			\
         this->size(1) == m.size(1) &&                                  \
         this->size(2) == m.size(2));					\
  for (index_type i = 0; i < this->size(0); ++i)			\
    for (index_type j = 0; j < this->size(1); ++j)			\
      for (index_type k = 0; k < this->size(2); ++k)			\
	this->put(i, j, k, this->get(i, j, k) op m.get(i, j, k))
  
#define VSIP_IMPL_ASSIGN_OP(asop, op)			   	   \
  template <typename T0>                                           \
  Tensor& operator asop(T0 const& val) VSIP_NOTHROW                \
  { VSIP_IMPL_ELEMENTWISE_SCALAR(op); return *this;}               \
  template <typename T0, typename Block0>                          \
  Tensor& operator asop(const_Tensor<T0, Block0> m) VSIP_NOTHROW   \
  { VSIP_IMPL_ELEMENTWISE_TENSOR(op); return *this;}               \
  template <typename T0, typename Block0>                          \
  Tensor& operator asop(const Tensor<T0, Block0> m) VSIP_NOTHROW   \
  { VSIP_IMPL_ELEMENTWISE_TENSOR(op); return *this;}

  // [view.tensor.assign]
  VSIP_IMPL_ASSIGN_OP(+=, +)
  VSIP_IMPL_ASSIGN_OP(-=, -)
  VSIP_IMPL_ASSIGN_OP(*=, *)
  VSIP_IMPL_ASSIGN_OP(/=, /)
  VSIP_IMPL_ASSIGN_OP(&=, &)
  VSIP_IMPL_ASSIGN_OP(|=, |)
  VSIP_IMPL_ASSIGN_OP(^=, ^)
};

#undef VSIP_IMPL_ASSIGN_OP
#undef VSIP_IMPL_ELEMENTWISE_SCALAR
#undef VSIP_IMPL_ELEMENTWISE_TENSOR

// [view.tensor.convert]
template <typename T, typename Block>
struct ViewConversion<Tensor, T, Block>
{
  typedef const_Tensor<T, Block> const_view_type;
  typedef Tensor<T, Block>       view_type;
};

template <typename T, typename Block>
struct ViewConversion<const_Tensor, T, Block>
{
  typedef const_Tensor<T, Block> const_view_type;
  typedef Tensor<T, Block>       view_type;
};

} // namespace vsip

namespace ovxx
{
template <typename B>
struct view_of<B, 3>
{
  typedef Tensor<typename B::value_type, B> type;
  typedef const_Tensor<typename B::value_type, B> const_type;
};

template <typename T, typename Block>
struct is_view_type<Tensor<T, Block> >
{
  typedef Tensor<T, Block> type; 
  static bool const value = true;
};

template <typename T, typename Block>
struct is_view_type<const_Tensor<T, Block> >
{
  typedef const_Tensor<T, Block> type; 
  static bool const value = true;
};

template <typename T, typename Block>
struct is_const_view_type<const_Tensor<T, Block> >
{
  typedef const_Tensor<T, Block> type; 
  static bool const value = true;
};

template <typename T, typename Block>
T
get(const_Tensor<T, Block> view, Index<3> const &i)
{
  return view.get(i[0], i[1], i[2]);
}

template <typename T, typename Block>
void
put(Tensor<T, Block> view, Index<3> const &i, T value)
{
  view.put(i[0], i[1], i[2], value);
}

/// Return the view extent as a domain.
template <typename T,
	  typename Block>
inline Domain<3>
view_domain(const_Tensor<T, Block> const& view)
{
  return Domain<3>(view.size(0), view.size(1), view.size(2));
}

/// Get the extent of a tensor view, as a Length.
template <typename T,
	  typename Block>
inline Length<3>
extent(const_Tensor<T, Block> const &v)
{
  return Length<3>(v.size(0), v.size(1), v.size(2));
}

} // namespace ovxx

#endif
