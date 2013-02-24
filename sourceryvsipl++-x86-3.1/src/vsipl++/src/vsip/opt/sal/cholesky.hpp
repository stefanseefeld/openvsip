/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_IMPL_SAL_SOLVER_CHOLESKY_HPP
#define VSIP_IMPL_SAL_SOLVER_CHOLESKY_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <algorithm>

#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/math_enum.hpp>
#include <vsip/core/temp_buffer.hpp>
#include <vsip/core/working_view.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>
#include <vsip/core/solver/common.hpp>

namespace vsip
{
namespace impl
{
namespace sal
{

// SAL Cholesky decomposition
#define CHOL_DEC(T, D_T, SAL_T, SALFCN)		\
inline void					\
mat_chol_dec(T *a, D_T *d, length_type n)	\
{						\
  SALFCN((SAL_T*) a, n,				\
	 (SAL_T*) a, n,				\
	 (D_T*) d, n);				\
}

#define CHOL_DEC_SPLIT(T, D_T, SAL_T, SALFCN) \
inline void				      \
mat_chol_dec(std::pair<T*,T*> a,	      \
	     D_T *d, length_type n)	      \
{					      \
  SALFCN((SAL_T*) &a, n, (SAL_T*) &a, n,      \
	 (D_T*) d, n);			      \
}

CHOL_DEC(float,                   float,float,        matchold)
CHOL_DEC(complex<float>,          float,COMPLEX,      cmatchold)
CHOL_DEC_SPLIT(float,             float,COMPLEX_SPLIT,zmatchold)

#undef CHOL_DEC
#undef CHOL_DEC_SPLIT

// SAL Cholesky solver
#define CHOL_SOL(T, D_T, SAL_T, SALFCN)		\
inline void					\
mat_chol_sol(T const *a, int atcols,		\
	     D_T *d, T *b, T *x, length_type n)	\
{						\
  SALFCN((SAL_T*) a, atcols,			\
	 (D_T*) d, (SAL_T*)b, (SAL_T*)x, n);	\
}

#define CHOL_SOL_SPLIT( T, D_T, SAL_T, SALFCN )	   \
inline void                                        \
mat_chol_sol(std::pair<T const*,T const*> a, int atcols,	\
	     D_T *d, std::pair<T*,T*> b,	   \
	     std::pair<T*,T*> x, length_type n)	   \
{                                                  \
  SALFCN((SAL_T*) &a, atcols,			   \
	 (D_T*) d, (SAL_T*)&b, (SAL_T*)&x, n);	   \
}

CHOL_SOL(float,         float,float,        matchols)
CHOL_SOL(complex<float>,float,COMPLEX,      cmatchols)
CHOL_SOL_SPLIT(float,   float,COMPLEX_SPLIT,zmatchols)

#undef CHOL_SOL
#undef CHOL_SOL_SPLIT

/// Cholesky factorization implementation class.  Common functionality
/// for chold by-value and by-reference classes.

template <typename T>
class Chold
{
  // The matrix to be decomposed using SAL must be in ROW major format. The
  // other matrix B will be in COL major format so that we can pass each
  // column to the solver. SAL supports both split and interleaved format.
  static storage_format_type const storage_format = dense_complex_format;
  typedef Storage<storage_format, T> storage_type;
  typedef typename storage_type::type ptr_type;

  typedef Layout<2, row2_type, dense, storage_format> data_LP;
  typedef Strided<2, T, data_LP> data_block_type;

  typedef Layout<2, col2_type, dense, storage_format> b_data_LP;
  typedef Strided<2, T, b_data_LP> b_data_block_type;

public:
  Chold(mat_uplo uplo, length_type length) VSIP_THROW((std::bad_alloc))
  : uplo_   (uplo),
    length_ (length),
    idv_    (length_),
    data_   (length_, length_)
  {
    assert(length_ > 0);
  }
  Chold(Chold const &chold) VSIP_THROW((std::bad_alloc))
  : uplo_       (chold.uplo_),
    length_     (chold.length_),
    idv_        (length_),
    data_       (length_, length_)
  {
    data_ = chold.data_;
  }
  mat_uplo    uplo()  const VSIP_NOTHROW { return uplo_; }
  length_type length()const VSIP_NOTHROW { return length_; }

  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW
  {
    assert(m.size(0) == length_ && m.size(1) == length_);

    data_ = m;
    dda::Data<data_block_type, dda::inout> data(data_.block());

    if(length_ > 1)
      mat_chol_dec(data.ptr(), &idv_[0], length_);
  return true;
}

  template <typename Block0,
	    typename Block1>
  bool solve(const_Matrix<T, Block0>, Matrix<T, Block1>) VSIP_NOTHROW;

private:
  typedef std::vector<float, Aligned_allocator<float> > vector_type;

  Chold& operator=(Chold const&) VSIP_NOTHROW;

  mat_uplo     uplo_;			// A upper/lower triangular
  length_type  length_;			// Order of A.
  vector_type  idv_;			// Daignal vector from decompose

  Matrix<T, data_block_type> data_;	// Factorized Cholesky matrix (A)
};

/// Solve A x = b (where A previously given to decompose)
template <typename T>
template <typename Block0, typename Block1>
bool
Chold<T>::solve(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
{
  assert(b.size(0) == length_);
  assert(b.size(0) == x.size(0) && b.size(1) == x.size(1));

  Matrix<T, b_data_block_type> b_int(b.size(0), b.size(1));
  Matrix<T, b_data_block_type> x_int(b.size(0), b.size(1));
  b_int = b;

  if (length_ > 1) 
  {
    dda::Data<b_data_block_type, dda::inout> b_data(b_int.block());
    dda::Data<b_data_block_type, dda::inout> x_data(x_int.block());
    dda::Data<data_block_type, dda::in> a_data(data_.block());

    ptr_type b_ptr = b_data.ptr();
    ptr_type x_ptr = x_data.ptr();

    for(index_type i=0;i<b.size(1);i++)
    {
      mat_chol_sol(a_data.ptr(), a_data.stride(0),
		   &idv_[0],
		   dda::impl::offset(b_ptr,i*length_),
		   dda::impl::offset(x_ptr,i*length_),
		   length_);
    }
  }
  else 
  {
    for(index_type i=0;i<b.size(1);i++)
      x_int.put(0,i,b.get(0,i)/data_.get(0,0));
  }
  x = x_int;
  return true;
}

} // namespace vsip::impl::sal
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <typename T>
struct Evaluator<op::chold, be::mercury_sal, T>
{
  static bool const ct_valid = is_same<T, float>::value ||
    is_same<T, complex<float> >::value;
  typedef impl::sal::Chold<T> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
