//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_matrix_hpp_
#define vsip_matrix_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/dense.hpp>
#include <vsip/vector.hpp>

namespace vsip
{

/// View which appears as a two-dimensional, read-only matrix.
template <typename T, typename Block>
class const_Matrix : public ovxx::const_View<const_Matrix,Block>
{
  typedef ovxx::const_View<vsip::const_Matrix,Block> base_type;
  typedef typename ovxx::lvalue_factory_type<Block, 2>::type ref_factory;

public:
  static const dimension_type dim = 2;
  typedef Block block_type;
  typedef typename block_type::value_type value_type;
  typedef typename ref_factory::reference_type reference_type;
  typedef typename ref_factory::const_reference_type
    const_reference_type;
  typedef typename base_type::impl_const_view_type impl_const_view_type;

  // [view.matrix.subview_types]
protected:
  typedef typename block_type::map_type map_type;
  typedef ovxx::expr::Subset<block_type> impl_subblock_type;
  typedef ovxx::expr::Transposed<block_type> impl_transblock_type;
  typedef ovxx::expr::Sliced<block_type, 1> impl_coblock_type;
  typedef ovxx::expr::Sliced<block_type, 0> impl_roblock_type;
  typedef ovxx::expr::Diag<block_type> impl_diagblock_type;

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
  const_Matrix(length_type num_rows, length_type num_cols, T const &value,
	       map_type const &map = map_type())
    : base_type(new block_type(Domain<2>(num_rows, num_cols), value, map), false)
  {}
  const_Matrix(length_type num_rows, length_type num_cols,
	       map_type const &map = map_type())
    : base_type(new block_type(Domain<2>(num_rows, num_cols), map), false)
  {}
  const_Matrix(Block &block) VSIP_NOTHROW : base_type(&block) {}
  const_Matrix(const_Matrix const &m) VSIP_NOTHROW : base_type(&m.block()) {}
  ~const_Matrix() VSIP_NOTHROW {}

  // [view.matrix.valaccess]
  value_type get(index_type r, index_type c) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(r < this->size(0));
    OVXX_PRECONDITION(c < this->size(1));
    return this->block().get(r, c);
  }

  // Supported for some, but not all, underlying Blocks.
  const_reference_type operator()(index_type r, index_type c) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(r < this->size(0));
    OVXX_PRECONDITION(c < this->size(1));
    return ref_factory::ref(this->block(), r, c);
  }

  // [view.matrix.subviews]
  const_subview_type get(const Domain<2>& dom) const VSIP_THROW((std::bad_alloc))
  {
    impl_subblock_type block(dom, this->block());
    return const_subview_type(block);
  }

  const_subview_type operator()(const Domain<2>& dom) const VSIP_THROW((std::bad_alloc))
  {
    impl_subblock_type block(dom, this->block());
    return const_subview_type(block);
  }

  const_transpose_type transpose() const VSIP_THROW((std::bad_alloc))
  {
    impl_transblock_type block(this->block());
    return const_transpose_type(block);
  }

  // [view.matrix.accessors]
  const_col_type col(index_type i) const VSIP_THROW((std::bad_alloc))
  {
    OVXX_PRECONDITION(i < this->size(1));
    impl_coblock_type block(this->block(), i);
    return const_col_type(block);
  }
  const_row_type row(index_type i) const VSIP_THROW((std::bad_alloc))
  {
    OVXX_PRECONDITION(i < this->size(0));
    impl_roblock_type block(this->block(), i);
    return const_row_type(block);
  }
  const_col_type operator()(whole_domain_type, vsip::index_type i) const
  {
    return col(i);
  }
  const_row_type operator()(index_type i, whole_domain_type) const
  {
    return row(i);
  }

  const_diag_type diag(index_difference_type diagonal_offset = 0)
    const VSIP_THROW((std::bad_alloc))
  {
    impl_diagblock_type block(this->block(), diagonal_offset);
    return const_diag_type(block);
  }
};

/// View which appears as a two-dimensional, modifiable matrix.  This
/// inherits from const_Matrix, so only the members that const_Matrix
/// does not carry, or that are different, need be specified.
template <typename T, typename Block>
class Matrix : public ovxx::View<Matrix,Block>
{
  typedef ovxx::View<vsip::Matrix,Block> base_type;
  typedef typename ovxx::lvalue_factory_type<Block, 2>::type ref_factory;

public:
  // Compile-time values.
  static const dimension_type dim = 2;
  typedef Block block_type;
  typedef typename block_type::value_type value_type;
  typedef typename ref_factory::reference_type reference_type;
  typedef typename ref_factory::const_reference_type
    const_reference_type;

  // [view.matrix.subview_types]
protected:
  typedef typename block_type::map_type map_type;
  typedef ovxx::expr::Subset<block_type> impl_subblock_type;
  typedef ovxx::expr::Transposed<block_type> impl_transblock_type;  
  typedef ovxx::expr::Sliced<Block, 1> impl_coblock_type;
  typedef ovxx::expr::Sliced<Block, 0> impl_roblock_type;
  typedef ovxx::expr::Diag<Block> impl_diagblock_type;

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
	 map_type const &map = map_type())
    : base_type(num_rows, num_cols, value, map, ovxx::detail::disambiguate)
  {}
  Matrix(length_type num_rows, length_type num_cols,
	 map_type const &map = map_type())
    : base_type(num_rows, num_cols, map)
  {}
  Matrix(Block &block) VSIP_NOTHROW : base_type(block) {}
  Matrix(Matrix const &m) VSIP_NOTHROW : base_type(m.block()) 
  { ovxx::expr::evaluate(this->block());}
  template <typename T0, typename Block0>
  Matrix(const_Matrix<T0,Block0> const &m)
    : base_type(m.size(0), m.size(1), m.block().map()) { *this = m;}
  ~Matrix()VSIP_NOTHROW {}

  // [view.matrix.valaccess]
  void put(index_type r, index_type c, value_type val) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(r < this->size(0));
    OVXX_PRECONDITION(c < this->size(1));
    this->block().put(r, c, val);
  }

  reference_type operator()(index_type r, index_type c)
    VSIP_NOTHROW
  {
    OVXX_PRECONDITION(r < this->size(0));
    OVXX_PRECONDITION(c < this->size(1));
    return ref_factory::ref(this->block(), r, c);
  }
  
  Matrix &operator=(Matrix const &m) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(this->size(0) == m.size(0) && this->size(1) == m.size(1));
    ovxx::assign<2>(this->block(), m.block());
    return *this;
  }

  Matrix &operator=(const_reference_type val) VSIP_NOTHROW
  {
    ovxx::expr::Scalar<2, T> scalar(val);
    ovxx::assign<2>(this->block(), scalar);
    return *this;
  }
  template <typename T0>
  Matrix& operator=(T0 const& val) VSIP_NOTHROW
  {
    ovxx::expr::Scalar<2, T0> scalar(val);
    ovxx::assign<2>(this->block(), scalar);
    return *this;
  }
  template <typename T0, typename Block0>
  Matrix &operator=(const_Matrix<T0, Block0> const &m) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(this->size(0) == m.size(0) && this->size(1) == m.size(1));
    ovxx::assign<2>(this->block(), m.block());
    return *this;
  }
  template <typename T0, typename Block0>
  Matrix &operator=(Matrix<T0, Block0> const &m) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(this->size(0) == m.size(0) && this->size(1) == m.size(1));
    ovxx::assign<2>(this->block(), m.block());
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
  transpose() const VSIP_THROW((std::bad_alloc))
  {
    impl_transblock_type block(this->block());
    return transpose_type(block);
  }

  const_col_type col(index_type i) const VSIP_THROW((std::bad_alloc))
  { return base_type::col(i); } 
  col_type col(index_type i) VSIP_THROW((std::bad_alloc))
  {
    OVXX_PRECONDITION(i < this->size(1));
    impl_coblock_type block(this->block(), i);
    return col_type(block);
  }

  const_row_type row(index_type i) const VSIP_THROW((std::bad_alloc))
  { return base_type::row(i); } 
  row_type row(index_type i) VSIP_THROW((std::bad_alloc))
  {
    OVXX_PRECONDITION(i < this->size(0));
    impl_roblock_type block(this->block(), i);
    return row_type(block);
  }
  const_col_type operator()(whole_domain_type, index_type i) const
  {
    return col(i);
  }
  col_type operator()(whole_domain_type, index_type i)
  {
    return col(i);
  }
  const_row_type operator()(index_type i, whole_domain_type) const
  {
    return row(i);
  }
  row_type operator()(index_type i, whole_domain_type)
  {
    return row(i);
  }

  const_diag_type diag(index_difference_type diagonal_offset = 0)
    const VSIP_THROW((std::bad_alloc))
  { return base_type::diag(diagonal_offset);}
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
  OVXX_PRECONDITION(this->size(0) == m.size(0) && this->size(1) == m.size(1));	\
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

#define VSIP_IMPL_ASSIGN_OP_NOFWD(asop, op)			   \
  template <typename T0>                                           \
  Matrix& operator asop(T0 const& val) VSIP_NOTHROW                \
  { VSIP_IMPL_ELEMENTWISE_SCALAR_NOFWD(op); return *this;}	   \
  template <typename T0, typename Block0>                          \
  Matrix& operator asop(const_Matrix<T0, Block0> m) VSIP_NOTHROW   \
  { VSIP_IMPL_ELEMENTWISE_MATRIX_NOFWD(op); return *this;}	   \
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

} // namespace vsip

namespace ovxx
{
template <typename B>
struct view_of<B, 2>
{
  typedef Matrix<typename B::value_type, B> type;
  typedef const_Matrix<typename B::value_type, B> const_type;
};

template <typename T, typename Block>
struct is_view_type<Matrix<T, Block> >
{
  typedef Matrix<T, Block> type; 
  static bool const value = true;
};

template <typename T, typename Block> 
struct is_view_type<const_Matrix<T, Block> >
{
  typedef const_Matrix<T, Block> type; 
  static bool const value = true;
};

template <typename T, typename Block> 
struct is_const_view_type<const_Matrix<T, Block> >
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

} // namespace ovxx

#endif
