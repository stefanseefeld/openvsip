//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_TENSOR_HPP
#define VSIP_TENSOR_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/dense.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/subblock.hpp>
#include <vsip/core/refcount.hpp>
#include <vsip/core/view_traits.hpp>
#include <vsip/core/assign.hpp>
#include <vsip/core/lvalue_proxy.hpp>
#ifndef VSIP_IMPL_REF_IMPL
# include <vsip_csl/pi/iterator.hpp>
#endif

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

/// View which appears as a three-dimensional, read-only tensor.
template <typename T, typename Block>
class const_Tensor : public impl::Non_assignable,
		     public  vsip::impl_const_View<vsip::const_Tensor,Block>
{
  typedef vsip::impl_const_View<vsip::const_Tensor,Block> impl_base_type;
  typedef typename impl::Lvalue_factory_type<Block>::type impl_factory_type;

public:
  // Compile-time values.
  static const dimension_type dim = 3;
  typedef Block                                      block_type;
  typedef typename block_type::value_type            value_type;
  typedef typename impl_factory_type::reference_type reference_type;
  typedef typename impl_factory_type::const_reference_type
		const_reference_type;

  typedef vsip::whole_domain_type whole_domain_type;
  static whole_domain_type const whole_domain = vsip::whole_domain;

  typedef typename impl_base_type::impl_const_view_type impl_const_view_type;

  // [view.tensor.subview_types]
protected:
  typedef typename Block::map_type  impl_map_type;
  typedef impl::Subset_block<Block>  impl_subblock_type;

public:
  typedef const_Tensor<T, impl_subblock_type> subview_type;
  typedef const_Tensor<T, impl_subblock_type> const_subview_type;

  template <dimension_type D0, dimension_type D1, dimension_type D2>
  struct transpose_view
  {
    typedef impl::Permuted_block<block_type, tuple<D0, D1, D2> >
      impl_transblock_type;
    typedef const_Tensor<T, impl_transblock_type> type;
    typedef const_Tensor<T, impl_transblock_type> const_type;
  }; 

  template <dimension_type D>
  struct submatrix : public impl::Compile_time_assert<(D < 3)>
  {
    typedef impl::Sliced_block<block_type, D> block;
    typedef const_Matrix<T, block> type;
    typedef const_Matrix<T, block> const_type;

    typedef impl::Subset_block<block> subblock;
    typedef const_Matrix<T, subblock> subset_type;
    typedef const_Matrix<T, subblock> const_subset_type;
  };

  template <dimension_type D1, dimension_type D2>
  struct subvector : public impl::Compile_time_assert<(D1 < D2 && D2 < 3)>
  {
    typedef impl::Sliced2_block<block_type, D1, D2> block;
    typedef const_Vector<T, block> type;
    typedef const_Vector<T, block> const_type;

    typedef impl::Subset_block<block> subblock;
    typedef const_Vector<T, subblock> subset_type;
    typedef const_Vector<T, subblock> const_subset_type;
  };

public:

  // [view.tensor.constructors]
  const_Tensor(length_type i, length_type j, length_type k, T const& value,
	       impl_map_type const& map = impl_map_type())
    : impl_base_type(new block_type(Domain<3>(i, j, k), value, map), impl::noincrement)
  {}
  const_Tensor(length_type i, length_type j, length_type k,
	       impl_map_type const& map = impl_map_type())
    : impl_base_type(new block_type(Domain<3>(i, j, k), map), impl::noincrement)
  {}
  const_Tensor(Block& blk) VSIP_NOTHROW
    : impl_base_type (&blk)
  {}
  const_Tensor(const_Tensor const& v) VSIP_NOTHROW
    : impl_base_type(&v.block())
  {}
  ~const_Tensor() VSIP_NOTHROW
  {}

  // [view.tensor.transpose]
  template <dimension_type D0, dimension_type D1, dimension_type D2>
  typename transpose_view<D0, D1, D2>::const_type  
  transpose() const VSIP_NOTHROW
  {
    typename transpose_view<D0, D1, D2>::impl_transblock_type  block(
      this->block());
    return typename transpose_view<D0, D1, D2>::const_type(block);
  }

  // [view.tensor.valaccess]
  value_type get(index_type i, index_type j, index_type k) const VSIP_NOTHROW
  {
    assert(i < this->size(0));
    assert(j < this->size(1));
    assert(k < this->size(2));
    return this->block().get(i, j, k);
  }

  // Supported for some, but not all, underlying Blocks.
  const_reference_type operator()(index_type i, index_type j, index_type k)
    const VSIP_NOTHROW
  { impl_factory_type f(this->block()); return f.impl_ref(i, j, k); }

  // [view.tensor.subviews]
  const_subview_type get(Domain<3> const& dom)
    const VSIP_THROW((std::bad_alloc))
  {
    impl_subblock_type block(dom, this->block());
    return const_subview_type(block);
  }

  const_subview_type operator()(Domain<3> const& dom)
    const VSIP_THROW((std::bad_alloc))
  {
    impl_subblock_type block(dom, this->block());
    return const_subview_type(block);
  }

  typename subvector<1, 2>::const_subset_type
  operator()(Domain<1> const& d, index_type j, index_type k)
    const VSIP_THROW((std::bad_alloc))
  {
    typename subvector<1, 2>::block block(this->block(), j, k);
    typename subvector<1, 2>::subblock sblock(d, block);
    return typename subvector<1, 2>::const_subset_type(sblock);
  }
  typename subvector<0, 2>::const_subset_type
  operator()(index_type i, Domain<1> const& d, index_type k)
    const VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 2>::block block(this->block(), i, k);
    typename subvector<0, 2>::subblock sblock(d, block);
    return typename subvector<0, 2>::const_subset_type(sblock);
  }
  typename subvector<0, 1>::const_subset_type
  operator()(index_type i, index_type j, Domain<1> const& d)
    const VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 1>::block block(this->block(), i, j);
    typename subvector<0, 1>::subblock sblock(d, block);
    return typename subvector<0, 1>::const_subset_type(sblock);
  }
  typename submatrix<0>::const_subset_type
  operator()(index_type i, Domain<1> const& d1, Domain<1> const& d2)
    const VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<0>::block block(this->block(), i);
    typename submatrix<0>::subblock sblock(Domain<2>(d1, d2), block);
    return typename submatrix<0>::const_subset_type(sblock);
  }
  typename submatrix<1>::const_subset_type
  operator()(Domain<1> const& d1, index_type j, Domain<1> const& d2)
    const VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<1>::block block(this->block(), j);
    typename submatrix<1>::subblock sblock(Domain<2>(d1, d2), block);
    return typename submatrix<1>::const_subset_type(sblock);
  }
  typename submatrix<2>::const_subset_type
  operator()(Domain<1> const& d1, Domain<1> const& d2, index_type k)
    const VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<2>::block block(this->block(), k);
    typename submatrix<2>::subblock sblock(Domain<2>(d1, d2), block);
    return typename submatrix<2>::const_subset_type(sblock);
  }

  typename subvector<1, 2>::const_type
  operator()(whole_domain_type, index_type j, index_type k)
    const VSIP_THROW((std::bad_alloc))
  {
    typename subvector<1, 2>::block block(this->block(), j, k);
    return typename subvector<1, 2>::const_type(block);
  }
  typename subvector<0, 2>::const_type
  operator()(index_type i, whole_domain_type, index_type k)
    const VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 2>::block block(this->block(), i, k);
    return typename subvector<0, 2>::const_type(block);
  }
  typename subvector<0, 1>::const_type
  operator()(index_type i, index_type j, whole_domain_type)
    const VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 1>::block block(this->block(), i, j);
    return typename subvector<0, 1>::const_type(block);
  }
  typename submatrix<0>::const_type
  operator()(index_type i, whole_domain_type, whole_domain_type)
    const VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<0>::block block(this->block(), i);
    return typename submatrix<0>::const_type(block);
  }
  typename submatrix<1>::const_type
  operator()(whole_domain_type, index_type j, whole_domain_type)
    const VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<1>::block block(this->block(), j);
    return typename submatrix<1>::const_type(block);
  }
  typename submatrix<2>::const_type
  operator()(whole_domain_type, whole_domain_type, index_type k)
    const VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<2>::block block(this->block(), k);
    return typename submatrix<2>::const_type(block);
  }
#ifndef VSIP_IMPL_REF_IMPL
  template <typename I, typename J, typename K>
  typename impl::enable_if_c<vsip_csl::pi::is_iterator<I>::value || 
			     vsip_csl::pi::is_iterator<J>::value || 
			     vsip_csl::pi::is_iterator<K>::value,
			     vsip_csl::pi::Call<block_type, I, J, K> >::type
  operator()(I const &i, J const &j, K const &k)
  {
    using namespace vsip_csl::pi;
    return Call<block_type, I, J, K>(this->block(), i, j, k);
  }
#endif

};


/// View which appears as a three-dimensional, modifiable tensor.  This
/// inherits from const_Tensor, so only the members that const_Tensor
/// does not carry, or that are different, need be specified.
template <typename T, typename Block>
class Tensor : public vsip::impl_View<vsip::Tensor, Block>
{
  typedef vsip::impl_View<vsip::Tensor, Block> impl_base_type;
  typedef typename impl::Lvalue_factory_type<Block>::type impl_factory_type;

  // Implementation compile-time values.
protected:
  typedef typename Block::map_type impl_map_type;

  // [view.tensor.subview_types]
  // Override subview_type and make it writable.
  typedef impl::Subset_block<Block> impl_subblock_type;

public:
  typedef typename 
    vsip::const_Tensor<T, Block>::whole_domain_type whole_domain_type;
  typedef Tensor<T, impl_subblock_type>      subview_type;

  // Compile-time values.
  static const dimension_type dim = 3;
  typedef Block                                       block_type;
  typedef typename block_type::value_type             value_type;
  typedef typename impl_factory_type::reference_type  reference_type;
  typedef typename impl_factory_type::const_reference_type
		const_reference_type;

  template <dimension_type D0, dimension_type D1, dimension_type D2>
  struct transpose_view
  {
    typedef impl::Permuted_block<block_type, tuple<D0, D1, D2> >
      impl_transblock_type;
    typedef       Tensor<T, impl_transblock_type> type;
    typedef const_Tensor<T, impl_transblock_type> const_type;
  }; 

  template <dimension_type D>
  struct submatrix : public impl::Compile_time_assert<(D < 3)>
  {
    typedef impl::Sliced_block<block_type, D> block;
    typedef Matrix<T, block> type;
    typedef const_Matrix<T, block> const_type;

    typedef impl::Subset_block<block> subblock;
    typedef Matrix<T, subblock> subset_type;
    typedef const_Matrix<T, subblock> const_subset_type;
  };

  template <dimension_type D1, dimension_type D2>
  struct subvector : public impl::Compile_time_assert<(D1 < D2 && D2 < 3)>
  {
    typedef impl::Sliced2_block<block_type, D1, D2> block;
    typedef Vector<T, block> type;
    typedef const_Vector<T, block> const_type;

    typedef impl::Subset_block<block> subblock;
    typedef Vector<T, subblock> subset_type;
    typedef const_Vector<T, subblock> const_subset_type;
  };

public:

  // [view.tensor.constructors]
  Tensor(length_type i, length_type j, length_type k, const T& value,
	 impl_map_type const& map = impl_map_type())
    : impl_base_type(i, j, k, value, map, impl::disambiguate)
  {}
  Tensor(length_type i, length_type j, length_type k,
	 impl_map_type const& map = impl_map_type())
    : impl_base_type(i, j, k, map)
  {}
  Tensor(Block& blk) VSIP_NOTHROW : impl_base_type(blk) {}
  Tensor(Tensor const& v) VSIP_NOTHROW : impl_base_type(v.block())
  {}
  template <typename T0, typename Block0>
  Tensor(const_Tensor<T0, Block0> const& t) VSIP_NOTHROW
    : impl_base_type (t.size(0), t.size(1), t.size(2), t.block().map())
  { *this = t;}
  template <typename T0, typename Block0>
  Tensor(Tensor<T0, Block0> const& t)
    : impl_base_type(t.size(0), t.size(1), t.size(2), t.block().map())
  { *this = t;}
  ~Tensor() VSIP_NOTHROW {}

  // [view.tensor.transpose]
  template <dimension_type D0, dimension_type D1, dimension_type D2>
  typename transpose_view<D0, D1, D2>::const_type
  transpose() const VSIP_NOTHROW
  {
    return impl_base_type::transpose();
  }
  template <dimension_type D0, dimension_type D1, dimension_type D2>
  typename transpose_view<D0, D1, D2>::type
  transpose() VSIP_NOTHROW
  {
    typename transpose_view<D0, D1, D2>::impl_transblock_type  block(
      this->block());
    return typename transpose_view<D0, D1, D2>::type(block);
  }

  // [view.tensor.valaccess]
  void put(index_type i,
	   index_type j,
	   index_type k,
	   value_type val) const VSIP_NOTHROW
  {
    assert(i < this->size(0));
    assert(j < this->size(1));
    assert(k < this->size(2));
    this->block().put(i, j, k, val);
  }

  reference_type operator()(index_type i,
                            index_type j,
                            index_type k) VSIP_NOTHROW
  { impl_factory_type f(this->block()); return f.impl_ref(i, j, k); }

  Tensor& operator=(Tensor const& t) VSIP_NOTHROW
  {
    assert(this->size(0) == t.size(0) &&
	   this->size(1) == t.size(1) &&
	   this->size(2) == t.size(2));
    impl::assign<3>(this->block(), t.block());
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
    assert(this->size(0) == t.size(0) &&
	   this->size(1) == t.size(1) &&
	   this->size(2) == t.size(2));
    impl::assign<3>(this->block(), t.block());
    return *this;
  }
  template <typename T0, typename Block0>
  Tensor& operator=(Tensor<T0, Block0> const& t) VSIP_NOTHROW
  {
    assert(this->size(0) == t.size(0) &&
	   this->size(1) == t.size(1) &&
	   this->size(2) == t.size(2));
    impl::assign<3>(this->block(), t.block());
    return *this;
  }

  // [view.tensor.subviews]
  using impl_base_type::operator(); // Pull in all the const versions.
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
    typename subvector<1, 2>::block block(this->block(), j, k);
    typename subvector<1, 2>::subblock sblock(d, block);
    return typename subvector<1, 2>::subset_type(sblock);
  }
  typename subvector<0, 2>::subset_type
  operator()(index_type i, Domain<1> const& d, index_type k)
    VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 2>::block block(this->block(), i, k);
    typename subvector<0, 2>::subblock sblock(d, block);
    return typename subvector<0, 2>::subset_type(sblock);
  }
  typename subvector<0, 1>::subset_type
  operator()(index_type i, index_type j, Domain<1> const& d)
    VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 1>::block block(this->block(), i, j);
    typename subvector<0, 1>::subblock sblock(d, block);
    return typename subvector<0, 1>::subset_type(sblock);
  }
  typename submatrix<0>::subset_type
  operator()(index_type i, Domain<1> const& d1, Domain<1> const& d2)
    VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<0>::block block(this->block(), i);
    typename submatrix<0>::subblock sblock(Domain<2>(d1, d2), block);
    return typename submatrix<0>::subset_type(sblock);
  }
  typename submatrix<1>::subset_type
  operator()(Domain<1> const& d1, index_type j, Domain<1> const& d2)
    VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<1>::block block(this->block(), j);
    typename submatrix<1>::subblock sblock(Domain<2>(d1, d2), block);
    return typename submatrix<1>::subset_type(sblock);
  }
  typename submatrix<2>::subset_type
  operator()(Domain<1> const& d1, Domain<1> const& d2, index_type k)
    VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<2>::block block(this->block(), k);
    typename submatrix<2>::subblock sblock(Domain<2>(d1, d2), block);
    return typename submatrix<2>::subset_type(sblock);
  }

  typename subvector<1, 2>::type
  operator()(whole_domain_type, index_type j, index_type k)
    VSIP_THROW((std::bad_alloc))
  {
    typename subvector<1, 2>::block block(this->block(), j, k);
    return typename subvector<1, 2>::type(block);
  }
  typename subvector<0, 2>::type
  operator()(index_type i, whole_domain_type, index_type k)
    VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 2>::block block(this->block(), i, k);
    return typename subvector<0, 2>::type(block);
  }
  typename subvector<0, 1>::type
  operator()(index_type i, index_type j, whole_domain_type)
    VSIP_THROW((std::bad_alloc))
  {
    typename subvector<0, 1>::block block(this->block(), i, j);
    return typename subvector<0, 1>::type(block);
  }
  typename submatrix<0>::type
  operator()(index_type i, whole_domain_type, whole_domain_type)
    VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<0>::block block(this->block(), i);
    return typename submatrix<0>::type(block);
  }
  typename submatrix<1>::type
  operator()(whole_domain_type, index_type j, whole_domain_type)
    VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<1>::block block(this->block(), j);
    return typename submatrix<1>::type(block);
  }
  typename submatrix<2>::type
  operator()(whole_domain_type, whole_domain_type, index_type k)
    VSIP_THROW((std::bad_alloc))
  {
    typename submatrix<2>::block block(this->block(), k);
    return typename submatrix<2>::type(block);
  }

#ifndef VSIP_IMPL_REF_IMPL
  template <typename I, typename J, typename K>
  typename impl::enable_if_c<vsip_csl::pi::is_iterator<I>::value || 
			     vsip_csl::pi::is_iterator<J>::value || 
			     vsip_csl::pi::is_iterator<K>::value,
			     vsip_csl::pi::Call<block_type, I, J, K> >::type
  operator()(I const &i, J const &j, K const &k)
  {
    using namespace vsip_csl::pi;
    return Call<block_type, I, J, K>(this->block(), i, j, k);
  }
#endif

#define VSIP_IMPL_ELEMENTWISE_SCALAR(op)				\
  for (index_type i = 0; i < this->size(0); ++i)			\
    for (index_type j = 0; j < this->size(1); ++j)			\
      for (index_type k = 0; k < this->size(2); ++k)			\
	this->put(i, j, k, this->get(i, j, k) op val)

#define VSIP_IMPL_ELEMENTWISE_TENSOR(op)				\
  assert(this->size(0) == m.size(0) &&                                  \
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

namespace impl
{
template <typename B>
struct view_of<B, 3>
{
  typedef Tensor<typename B::value_type, B> type;
  typedef const_Tensor<typename B::value_type, B> const_type;
};

template <typename T, typename Block>
struct Is_view_type<Tensor<T, Block> >
{
  typedef Tensor<T, Block> type; 
  static bool const value = true;
};

template <typename T, typename Block>
struct Is_view_type<const_Tensor<T, Block> >
{
  typedef const_Tensor<T, Block> type; 
  static bool const value = true;
};

template <typename T, typename Block>
struct Is_const_view_type<const_Tensor<T, Block> >
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

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_TENSOR_HPP
