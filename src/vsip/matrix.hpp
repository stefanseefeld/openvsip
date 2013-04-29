//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

/// Description
///   [view.matrix] views implementing two-dimensional matrices.
///
///   This file declares the `const_Matrix` and `Matrix` classes,
///   which provide the generic View interface, and implement the
///   mathematical idea of matrices, i.e. two-dimensional storage and
///   access to values.  A `const_Matrix` view is not modifiable, but a
///   `Matrix` view is modifiable.

#ifndef VSIP_MATRIX_HPP
#define VSIP_MATRIX_HPP

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/dense.hpp>
#include <vsip/vector.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/block_fill.hpp>
#include <vsip/core/subblock.hpp>
#include <vsip/core/refcount.hpp>
#include <vsip/core/view_traits.hpp>
#include <vsip/core/assign.hpp>
#include <vsip/core/lvalue_proxy.hpp>
#include <vsip/core/metaprogramming.hpp>
#ifndef VSIP_IMPL_REF_IMPL
# include <vsip_csl/pi/iterator.hpp>
#endif

namespace vsip
{

/// View which appears as a two-dimensional, read-only matrix.
template <typename T, typename Block>
class const_Matrix : public impl::Non_assignable,
		     public  vsip::impl_const_View<vsip::const_Matrix,Block>
{
  typedef vsip::impl_const_View<vsip::const_Matrix,Block> impl_base_type;
  typedef typename impl::Lvalue_factory_type<Block>::type impl_factory_type;

public:
  // Compile-time values.
  static const dimension_type dim = 2;
  typedef Block                                      block_type;
  typedef typename block_type::value_type            value_type;
  typedef typename impl_factory_type::reference_type reference_type;
  typedef typename impl_factory_type::const_reference_type
		const_reference_type;
  typedef typename impl_base_type::impl_const_view_type impl_const_view_type;

  // [view.matrix.subview_types]
protected:
  typedef typename block_type::map_type impl_map_type;
  typedef impl::Subset_block<block_type> impl_subblock_type;
  typedef impl::Transposed_block<block_type> impl_transblock_type;
  typedef impl::Sliced_block<block_type, 1> impl_coblock_type;
  typedef impl::Sliced_block<block_type, 0> impl_roblock_type;
  typedef impl::Diag_block<block_type> impl_diagblock_type;

public:
  typedef const_Matrix<T, impl_subblock_type> subview_type;
  typedef const_Matrix<T, impl_subblock_type> const_subview_type;
  typedef const_Vector<T, impl_coblock_type> col_type;
  typedef const_Vector<T, impl_coblock_type> const_col_type;
  typedef const_Vector<T, impl_diagblock_type> diag_type;
  typedef const_Vector<T, impl_diagblock_type> const_diag_type;
  typedef const_Vector<T, impl_roblock_type> row_type;
  typedef const_Vector<T, impl_roblock_type> const_row_type;
  typedef const_Matrix<T, impl_transblock_type> transpose_type;
  typedef const_Matrix<T, impl_transblock_type> const_transpose_type;

  // [view.matrix.constructors]
  const_Matrix(length_type num_rows, length_type num_cols, T const& value,
	       impl_map_type const& map = impl_map_type())
    : impl_base_type(new block_type(Domain<2>(num_rows, num_cols), value, map),
	   impl::noincrement)
  {}
  const_Matrix(length_type num_rows, length_type num_cols,
	       impl_map_type const& map = impl_map_type())
    : impl_base_type(new block_type(Domain<2>(num_rows, num_cols), map),
	   impl::noincrement)
  {}
  const_Matrix(Block& blk) VSIP_NOTHROW
    : impl_base_type(&blk)
  {}
  const_Matrix(const_Matrix const& m) VSIP_NOTHROW
    : impl_base_type(&m.block())
  {}
  ~const_Matrix() VSIP_NOTHROW
  {}

  // [view.matrix.valaccess]
  value_type get(vsip::index_type r, vsip::index_type c) const VSIP_NOTHROW
  {
    assert(r < this->size(0));
    assert(c < this->size(1));
    return this->block().get(r, c);
  }

  // Supported for some, but not all, underlying Blocks.
  const_reference_type operator()(vsip::index_type r, vsip::index_type c)
    const VSIP_NOTHROW
  {
    assert(r < this->size(0));
    assert(c < this->size(1));
    impl_factory_type f(this->block());
    return f.impl_ref(r, c);
  }

  // [view.matrix.subviews]
  const_subview_type get(const Domain<2>& dom)
    const VSIP_THROW((std::bad_alloc))
  {
    impl_subblock_type block(dom, this->block());
    return const_subview_type(block);
  }

  const_subview_type operator()(const Domain<2>& dom)
    const VSIP_THROW((std::bad_alloc))
  {
    impl_subblock_type block(dom, this->block());
    return const_subview_type(block);
  }

  const_transpose_type transpose() const
    VSIP_THROW((std::bad_alloc))
  {
    impl_transblock_type block(this->block());
    return const_transpose_type(block);
  }

  // [view.matrix.accessors]

  const_col_type col(vsip::index_type i) const VSIP_THROW((std::bad_alloc))
  {
    assert(i < this->size(1));
    impl_coblock_type block(this->block(), i);
    return const_col_type(block);
  }
  const_row_type row(vsip::index_type i) const VSIP_THROW((std::bad_alloc))
  {
    assert(i < this->size(0));
    impl_roblock_type block(this->block(), i);
    return const_row_type(block);
  }
  const_col_type operator()(whole_domain_type, vsip::index_type i) const
  {
    return col(i);
  }
  const_row_type operator()(vsip::index_type i, whole_domain_type) const
  {
    return row(i);
  }

  const_diag_type diag(index_difference_type diagonal_offset = 0)
    const VSIP_THROW((std::bad_alloc))
  {
    impl_diagblock_type block(this->block(), diagonal_offset);
    return const_diag_type(block);
  }
#ifndef VSIP_IMPL_REF_IMPL
  template <typename I, typename J>
  typename impl::enable_if_c<vsip_csl::pi::is_iterator<I>::value || 
			     vsip_csl::pi::is_iterator<J>::value,
			     vsip_csl::pi::Call<block_type, I, J> >::type
  operator()(I const &i, J const &j)
  {
    using namespace vsip_csl::pi;
    return Call<block_type, I, J>(this->block(), i, j);
  }
  template <typename J>
  typename impl::enable_if_c<vsip_csl::pi::is_iterator<J>::value,
			     vsip_csl::pi::Call<block_type, whole_domain_type, J> >::type
  col(J const &j)
  {
    using namespace vsip_csl::pi;
    return Call<block_type, whole_domain_type, J>(this->block(), j);
  }
  template <typename I>
  typename impl::enable_if_c<vsip_csl::pi::is_iterator<I>::value,
			     vsip_csl::pi::Call<block_type, I, whole_domain_type> >::type
  row(I const &i)
  {
    using namespace vsip_csl::pi;
    return Call<block_type, I, whole_domain_type>(this->block(), i);
  }
#endif
};

/// View which appears as a two-dimensional, modifiable matrix.  This
/// inherits from const_Matrix, so only the members that const_Matrix
/// does not carry, or that are different, need be specified.
template <typename T, typename Block>
class Matrix : public vsip::impl_View<vsip::Matrix,Block>
{
  typedef vsip::impl_View<vsip::Matrix,Block> impl_base_type;
  typedef typename impl::Lvalue_factory_type<Block>::type impl_factory_type;

public:
  // Compile-time values.
  static const dimension_type dim = 2;
  typedef Block                                     block_type;
  typedef typename block_type::value_type           value_type;
  typedef typename impl_factory_type::reference_type reference_type;
  typedef typename impl_factory_type::const_reference_type
		const_reference_type;

  // [view.matrix.subview_types]
protected:
  typedef typename block_type::map_type impl_map_type;
  typedef impl::Subset_block<block_type> impl_subblock_type;
  typedef impl::Transposed_block<block_type> impl_transblock_type;  
  typedef impl::Sliced_block<Block, 1> impl_coblock_type;
  typedef impl::Sliced_block<Block, 0> impl_roblock_type;
  typedef impl::Diag_block<Block> impl_diagblock_type;

public:
  typedef       Matrix<T, impl_subblock_type> subview_type;
  typedef const_Matrix<T, impl_subblock_type> const_subview_type;
  typedef       Vector<T, impl_coblock_type> col_type;
  typedef const_Vector<T, impl_coblock_type> const_col_type;
  typedef       Vector<T, impl_diagblock_type> diag_type;
  typedef const_Vector<T, impl_diagblock_type> const_diag_type;
  typedef       Vector<T, impl_roblock_type> row_type;
  typedef const_Vector<T, impl_roblock_type> const_row_type;
  typedef       Matrix<T, impl_transblock_type> transpose_type;
  typedef const_Matrix<T, impl_transblock_type> const_transpose_type;

  // [view.matrix.constructors]
  Matrix(length_type num_rows, length_type num_cols, const T& value,
	 impl_map_type const& map = impl_map_type())
    : impl_base_type(num_rows, num_cols, value, map, impl::disambiguate)
  {}
  Matrix(length_type num_rows, length_type num_cols,
	 impl_map_type const& map = impl_map_type())
    : impl_base_type(num_rows, num_cols, map)
  {}
  Matrix(Block& blk) VSIP_NOTHROW : impl_base_type(blk) {}
  Matrix(Matrix const& v) VSIP_NOTHROW : impl_base_type(v.block()) 
  { impl::expr::evaluate(this->block());}
  template <typename T0, typename Block0>
  Matrix(const_Matrix<T0,Block0> const& m)
    : impl_base_type(m.size(0), m.size(1), m.block().map()) { *this = m;}
  ~Matrix()VSIP_NOTHROW {}

  // [view.matrix.valaccess]
  void put(vsip::index_type r, vsip::index_type c, value_type val) const VSIP_NOTHROW
  {
    assert(r < this->size(0));
    assert(c < this->size(1));
    this->block().put(r, c, val);
  }

  reference_type operator()(vsip::index_type r, vsip::index_type c)
    VSIP_NOTHROW
  {
    assert(r < this->size(0));
    assert(c < this->size(1));
    impl_factory_type f(this->block());
    return f.impl_ref(r, c);
  }
  
#ifndef VSIP_IMPL_REF_IMPL
  template <typename I, typename J>
  typename impl::enable_if_c<vsip_csl::pi::is_iterator<I>::value || 
			     vsip_csl::pi::is_iterator<J>::value,
			     vsip_csl::pi::Call<block_type, I, J> >::type
  operator()(I const &i, J const &j)
  {
    using namespace vsip_csl::pi;
    return Call<block_type, I, J>(this->block(), i, j);
  }
  template <typename J>
  typename impl::enable_if_c<vsip_csl::pi::is_iterator<J>::value,
			     vsip_csl::pi::Call<block_type, whole_domain_type, J> >::type
  col(J const &j)
  {
    using namespace vsip_csl::pi;
    return Call<block_type, whole_domain_type, J>(this->block(), j);
  }
  template <typename I>
  typename impl::enable_if_c<vsip_csl::pi::is_iterator<I>::value,
			     vsip_csl::pi::Call<block_type, I, whole_domain_type> >::type
  row(I const &i)
  {
    using namespace vsip_csl::pi;
    return Call<block_type, I, whole_domain_type>(this->block(), i);
  }
#endif

  Matrix& operator=(Matrix const& m) VSIP_NOTHROW
  {
    assert(this->size(0) == m.size(0) && this->size(1) == m.size(1));
    impl::assign<2>(this->block(), m.block());
    return *this;
  }

  Matrix& operator=(const_reference_type val) VSIP_NOTHROW
  {
    impl::expr::Scalar<2, T> scalar(val);
    impl::assign<2>(this->block(), scalar);
    return *this;
  }
  template <typename T0>
  Matrix& operator=(T0 const& val) VSIP_NOTHROW
  {
    impl::expr::Scalar<2, T0> scalar(val);
    impl::assign<2>(this->block(), scalar);
    return *this;
  }
  template <typename T0, typename Block0>
  Matrix& operator=(const_Matrix<T0, Block0> const& m) VSIP_NOTHROW
  {
    assert(this->size(0) == m.size(0) && this->size(1) == m.size(1));
    impl::assign<2>(this->block(), m.block());
    return *this;
  }
  template <typename T0, typename Block0>
  Matrix& operator=(Matrix<T0, Block0> const& m) VSIP_NOTHROW
  {
    assert(this->size(0) == m.size(0) && this->size(1) == m.size(1));
    impl::assign<2>(this->block(), m.block());
    return *this;
  }

  // [view.matrix.subviews]
  subview_type
  operator()(const Domain<2>& dom) VSIP_THROW((std::bad_alloc))
  {
    impl_subblock_type block(dom, this->block());
    return subview_type(block);
  }

  transpose_type
  transpose() const
    VSIP_THROW((std::bad_alloc))
  {
    impl_transblock_type block(this->block());
    return transpose_type(block);
  }

  const_col_type col(vsip::index_type i) const VSIP_THROW((std::bad_alloc))
  { return impl_base_type::col(i); } 
  col_type col(vsip::index_type i) VSIP_THROW((std::bad_alloc))
  {
    assert(i < this->size(1));
    impl_coblock_type block(this->block(), i);
    return col_type(block);
  }

  const_row_type row(vsip::index_type i) const VSIP_THROW((std::bad_alloc))
  { return impl_base_type::row(i); } 
  row_type row(vsip::index_type i) VSIP_THROW((std::bad_alloc))
  {
    assert(i < this->size(0));
    impl_roblock_type block(this->block(), i);
    return row_type(block);
  }
  const_col_type operator()(whole_domain_type, vsip::index_type i) const
  {
    return col(i);
  }
  col_type operator()(whole_domain_type, vsip::index_type i)
  {
    return col(i);
  }
  const_row_type operator()(vsip::index_type i, whole_domain_type) const
  {
    return row(i);
  }
  row_type operator()(vsip::index_type i, whole_domain_type)
  {
    return row(i);
  }

  const_diag_type diag(index_difference_type diagonal_offset = 0)
    const VSIP_THROW((std::bad_alloc))
  { return impl_base_type::diag(diagonal_offset); }
  diag_type diag(index_difference_type diagonal_offset = 0)
    VSIP_THROW((std::bad_alloc))
  {
    impl_diagblock_type block(this->block(), diagonal_offset);
    return diag_type(block);
  }

#define VSIP_IMPL_ELEMENTWISE_SCALAR(op)        			\
  *this = *this op val

#define VSIP_IMPL_ELEMENTWISE_SCALAR_NOFWD(op)				\
  for (vsip::index_type r = 0; r < this->size(0); ++r)			\
    for (vsip::index_type c = 0; c < this->size(1); ++c)		\
      this->put(r, c, this->get(r, c) op val)

#define VSIP_IMPL_ELEMENTWISE_MATRIX(op)				\
  *this = *this op m;

#define VSIP_IMPL_ELEMENTWISE_MATRIX_NOFWD(op)				\
  assert(this->size(0) == m.size(0) && this->size(1) == m.size(1));	\
  for (vsip::index_type r = 0; r < this->size(0); ++r)			\
    for (vsip::index_type c = 0; c < this->size(1); ++c)		\
      this->put(r, c, this->get(r, c) op m.get(r, c))
  
#define VSIP_IMPL_ASSIGN_OP(asop, op)			   	   \
  template <typename T0>                                           \
  Matrix& operator asop(T0 const& val) VSIP_NOTHROW                \
  { VSIP_IMPL_ELEMENTWISE_SCALAR(op); return *this;}               \
  template <typename T0, typename Block0>                          \
  Matrix& operator asop(const_Matrix<T0, Block0> m) VSIP_NOTHROW   \
  { VSIP_IMPL_ELEMENTWISE_MATRIX(op); return *this;}               \
  template <typename T0, typename Block0>                          \
  Matrix& operator asop(const Matrix<T0, Block0> m) VSIP_NOTHROW   \
  { VSIP_IMPL_ELEMENTWISE_MATRIX(op); return *this;}

#define VSIP_IMPL_ASSIGN_OP_NOFWD(asop, op)			   	   \
  template <typename T0>                                           \
  Matrix& operator asop(T0 const& val) VSIP_NOTHROW                \
  { VSIP_IMPL_ELEMENTWISE_SCALAR_NOFWD(op); return *this;}               \
  template <typename T0, typename Block0>                          \
  Matrix& operator asop(const_Matrix<T0, Block0> m) VSIP_NOTHROW   \
  { VSIP_IMPL_ELEMENTWISE_MATRIX_NOFWD(op); return *this;}               \
  template <typename T0, typename Block0>                          \
  Matrix& operator asop(const Matrix<T0, Block0> m) VSIP_NOTHROW   \
  { VSIP_IMPL_ELEMENTWISE_MATRIX_NOFWD(op); return *this;}

  // [view.matrix.assign]
  VSIP_IMPL_ASSIGN_OP(+=, +)
  VSIP_IMPL_ASSIGN_OP(-=, -)
  VSIP_IMPL_ASSIGN_OP(*=, *)
  VSIP_IMPL_ASSIGN_OP(/=, /)
  // For vector, ghs claims the use of operator& in 'view1 & view2' is
  // ambiguous, thus we implement operator&= in terms of the scalar
  // operator&.  Likewise for operator=| and operator=^.
  VSIP_IMPL_ASSIGN_OP_NOFWD(&=, &)
  VSIP_IMPL_ASSIGN_OP_NOFWD(|=, |)
  VSIP_IMPL_ASSIGN_OP_NOFWD(^=, ^)
};


#undef VSIP_IMPL_ASSIGN_OP
#undef VSIP_IMPL_ELEMENTWISE_SCALAR
#undef VSIP_IMPL_ELEMENTWISE_MATRIX
#undef VSIP_IMPL_ASSIGN_OP_NOFWD
#undef VSIP_IMPL_ELEMENTWISE_SCALAR_NOFWD
#undef VSIP_IMPL_ELEMENTWISE_MATRIX_NOFWD

// [view.matrix.convert]
template <typename T, typename Block>
struct ViewConversion<Matrix, T, Block>
{
  typedef const_Matrix<T, Block> const_view_type;
  typedef Matrix<T, Block>       view_type;
};

template <typename T, typename Block>
struct ViewConversion<const_Matrix, T, Block>
{
  typedef const_Matrix<T, Block> const_view_type;
  typedef Matrix<T, Block>       view_type;
};

namespace impl
{
template <typename B>
struct view_of<B, 2>
{
  typedef Matrix<typename B::value_type, B> type;
  typedef const_Matrix<typename B::value_type, B> const_type;
};

template <typename T, typename Block>
struct Is_view_type<Matrix<T, Block> >
{
  typedef Matrix<T, Block> type; 
  static bool const value = true;
};

template <typename T, typename Block> 
struct Is_view_type<const_Matrix<T, Block> >
{
  typedef const_Matrix<T, Block> type; 
  static bool const value = true;
};

template <typename T, typename Block> 
struct Is_const_view_type<const_Matrix<T, Block> >
{
  typedef const_Matrix<T, Block> type; 
  static bool const value = true;
};

template <typename T, typename Block>
T
get(const_Matrix<T, Block> view, Index<2> const &i)
{
  return view.get(i[0], i[1]);
}

template <typename T, typename Block>
void
put(Matrix<T, Block> view, Index<2> const &i, T value)
{
  view.put(i[0], i[1], value);
}

/// Return the view extent as a domain.
template <typename T,
	  typename Block>
inline Domain<2>
view_domain(const_Matrix<T, Block> const& view)
{
  return Domain<2>(view.size(0), view.size(1));
}

/// Get the extent of a matrix view, as a Length.
template <typename T,
	  typename Block>
inline Length<2>
extent(const_Matrix<T, Block> const &v)
{
  return Length<2>(v.size(0), v.size(1));
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_MATRIX_HPP
