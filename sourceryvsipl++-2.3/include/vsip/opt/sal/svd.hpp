/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef vsip_opt_sal_svd_hpp_
#define vsip_opt_sal_svd_hpp_

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
#include <sal.h>

namespace vsip
{
namespace impl
{
namespace sal
{

// SAL SVD decomposition
// SAL only supports SVD decomposition using COMPLEX SPLIT format. If we are
// dealing with real numbers, we assign the imaginary part of the pointer to
// a matrix of zeros.
#define SVD_DEC(T, D_T, SALFCN)				    \
inline bool                                                 \
sal_mat_svd_dec(T* z,					    \
		T* a, int tcols_a,			    \
		std::pair<float*,float*> u, int tcols_u,    \
		std::pair<float*,float*> v, int tcols_v,    \
		D_T *D,					    \
		int m, int n,				    \
		int flag)				    \
{                                                           \
  int rank;                                                 \
  COMPLEX_SPLIT temp_ptr = {a, z};			    \
                                                            \
  return(SALFCN(&temp_ptr, tcols_a,			    \
         D,                                                 \
	 (COMPLEX_SPLIT*)&u, tcols_u,                       \
	 (COMPLEX_SPLIT*)&v, tcols_v,                       \
	 m,n,                                               \
	 NULL,NULL,&rank,flag) == SAL_SUCCESS);             \
}

#define SVD_DEC_CPLX(T, D_T, SALFCN)			    \
inline bool                                                 \
sal_mat_svd_dec(T* /*z*/,				    \
		std::pair<T*,T*> a, int tcols_a,	    \
		std::pair<float*,float*> u, int tcols_u,    \
		std::pair<float*,float*> v, int tcols_v,    \
		D_T *D,					    \
		int m, int n,				    \
		int flag)				    \
{                                                           \
  int rank;                                                 \
                                                            \
  return(SALFCN((COMPLEX_SPLIT*)&a, tcols_a,                \
         D,                                                 \
	 (COMPLEX_SPLIT*)&u, tcols_u,                       \
	 (COMPLEX_SPLIT*)&v, tcols_v,                       \
	 m,n,                                               \
	 NULL,NULL,&rank,flag) == SAL_SUCCESS);             \
}

SVD_DEC(float, float,zmat_svd)
SVD_DEC_CPLX(float, float,zmat_svd)

#undef SVD_DEC
#undef SVD_DEC_CPLX

/// SVD decomposition implementation class.  Common functionality for
/// svd by-value and by-reference classes.
template <typename T>
class Svd
{
  typedef typename Scalar_of<T>::type scalar_type;
  typedef complex<scalar_type> uv_type;

  // SAL requires complex to be in split format
  typedef Layout<2, row2_type, Stride_unit_dense, Cmplx_split_fmt> data_LP;
  typedef Strided<2, T, data_LP>           data_block_type;
  typedef Strided<2, scalar_type, data_LP> rl_data_block_type;

  typedef Layout<2, col2_type, Stride_unit_dense, Cmplx_split_fmt> cp_data_LP;
  typedef Strided<2, uv_type, cp_data_LP> cp_data_block_type;

  typedef Storage<Cmplx_split_fmt, T>       my_storage_type;
  typedef typename my_storage_type::type    ptr_type;

public:
  Svd(length_type, length_type, storage_type, storage_type)
    VSIP_THROW((std::bad_alloc));
  Svd(Svd const&) VSIP_THROW((std::bad_alloc));

  length_type  rows()     const VSIP_NOTHROW { return m_;}
  length_type  columns()  const VSIP_NOTHROW { return n_;}
  storage_type ustorage() const VSIP_NOTHROW { return ust_;}
  storage_type vstorage() const VSIP_NOTHROW { return vst_;}

  template <typename Block0, typename Block1>
  bool decompose(Matrix<T, Block0>,
		 Vector<scalar_f, Block1>) VSIP_NOTHROW;

  template <mat_op_type       tr,
	    product_side_type ps,
	    typename          Block0,
	    typename          Block1>
  bool produ(const_Matrix<T, Block0>, Matrix<T, Block1>) VSIP_NOTHROW;

  template <mat_op_type       tr,
	    product_side_type ps,
	    typename          Block0,
	    typename          Block1>
  bool prodv(const_Matrix<T, Block0>, Matrix<T, Block1>) VSIP_NOTHROW;

  template <typename          Block>
  bool u(index_type, index_type, Matrix<T, Block>) VSIP_NOTHROW;

  template <typename          Block>
  bool v(index_type, index_type, Matrix<T, Block>) VSIP_NOTHROW;

  length_type order() const VSIP_NOTHROW { return p_;}

private:
  typedef std::vector<float, Aligned_allocator<float> > vector_type;
  typedef std::vector<scalar_type, Aligned_allocator<scalar_type> >
		svector_type;

  Svd &operator=(Svd const&) VSIP_NOTHROW;

  length_type  m_;			// Number of rows.
  length_type  n_;			// Number of cols.
  length_type  p_;			// min(rows, cols)
  storage_type ust_;			// U storage type
  storage_type vst_;			// V storage type

  Matrix<T, data_block_type> data_;		// Factorized matrix
  Matrix<scalar_type, rl_data_block_type> zeros_; // A matrix of zeros. This is 
                                                // used for imaginary part of 
						// float types.
  Matrix<uv_type, cp_data_block_type> u_;	// U matrix
  Matrix<uv_type, cp_data_block_type> v_;	// V matrix
  Matrix<uv_type, cp_data_block_type> ut_;	// U' matrix
  Matrix<uv_type, cp_data_block_type> vt_;	// V' matrix
  vector_type  d_;                      	// The diagonal vector
};

template <typename T>
Svd<T>::Svd(length_type rows, length_type cols,
	    storage_type ust, storage_type vst) VSIP_THROW((std::bad_alloc))
  : m_          (rows),
    n_          (cols),
    p_          (std::min(m_, n_)),
    ust_        (ust),
    vst_        (vst),

    data_       (m_, n_),
    zeros_      (m_, n_),
    u_          (m_, ((ust_ == svd_uvpart)? n_:m_)),
    v_          (n_, n_),
    ut_         (((ust_ == svd_uvpart)? n_:m_), m_),
    vt_         (n_, n_),
    d_          (p_)
{
  assert(m_ > 0 && n_ > 0);
  assert(ust_ == svd_uvnos || ust_ == svd_uvpart || ust_ == svd_uvfull);
  assert(vst_ == svd_uvnos || vst_ == svd_uvpart || vst_ == svd_uvfull);
  zeros_ = 0;

  if (n_ > m_)
    VSIP_IMPL_THROW(impl::unimplemented("SAL Svd only supports m >= n"));
}

template <typename T>
Svd<T>::Svd(Svd const& sv) VSIP_THROW((std::bad_alloc))
  : m_          (sv.m_),
    n_          (sv.n_),
    p_          (sv.p_),
    ust_        (sv.ust_),
    vst_        (sv.vst_),

    data_       (m_, n_),
    zeros_      (m_, n_),
    u_          (m_, ((ust_ == svd_uvpart)? n_:m_)),
    v_          (n_, n_),
    ut_         (((ust_ == svd_uvpart)? n_:m_), m_),
    vt_         (n_, n_),
    d_          (p_)
{
  assert(m_ >= n_);
  data_ = sv.data_;
  ut_   = sv.ut_;
  vt_   = sv.vt_;
  u_    = sv.u_;
  v_    = sv.v_;
  zeros_ = 0;
}

/// Decompose matrix M into
///
/// Return
///   DEST contains M's singular values.
///
/// Requires
///   M to be a full rank, modifiable matrix of ROWS x COLS.
template <typename T>
template <typename Block0, typename Block1>
bool
Svd<T>::decompose(Matrix<T, Block0> m,
		  Vector<vsip::scalar_f, Block1> dest) VSIP_NOTHROW
{
  int flag;

  assign_local(data_,m);
  assert(m.size(0) == m_ && m.size(1) == n_);

  Ext_data<data_block_type> data_ext(data_.block());
  Ext_data<cp_data_block_type> u_ext(u_.block());
  Ext_data<cp_data_block_type> v_ext(v_.block());
  Ext_data<rl_data_block_type> z_ext(zeros_.block());
  Ext_data<Block1> d_ext(dest.block());


  flag = ((ust_ == svd_uvpart)? SAL_SVD_THIN:
          (ust_ == svd_uvfull)? SAL_SVD_FULL:0) | SAL_SVD_V;

  bool success =
         sal_mat_svd_dec(z_ext.data(),
                         data_ext.data(), n_,
                         u_ext.data(), m_,
		         v_ext.data(), n_,
		         d_ext.data(),
		         m_,n_,
		         flag);

  return success;
}

// This function is a helper function that allows to axb. Because the U and V
// matrixes are complex with imag = 0 for real svd, a normal multiplication
// will not work. So, we need to multiply just the real part of a of b is real
// and vice-versa.
template <typename Block0, typename Block1, typename Block2>
void sal_svd_prod_uv(const_Matrix<complex<float>,Block0> a,
                     const_Matrix<float,Block1> b,
                     Matrix<float,Block2> p)
{
  generic_prod(a.real(),b,p);
}

template <typename Block0, typename Block1, typename Block2>
void sal_svd_prod_uv(const_Matrix<complex<float>,Block0> a,
                     const_Matrix<complex<float>,Block1> b,
                     Matrix<complex<float>,Block2> p)
{
  generic_prod(a,b,p);
}

template <typename Block0, typename Block1, typename Block2>
void sal_svd_prod_uv(const_Matrix<float,Block0> a,
                     const_Matrix<complex<float>,Block1> b,
                     Matrix<float,Block2> p)
{
  generic_prod(a,b.real(),p);
}


/// Compute product of U and b
///
/// If svd_uvpart: U is (m, p)
/// If svd_uvfull: U is (m, m)
///
/// ustorage   | ps        | tr         | product | b (in) | x (out)
/// svd_uvpart | mat_lside | mat_ntrans | U b     | (p, s) | (m, s)
/// svd_uvpart | mat_lside | mat_trans  | U' b    | (m, s) | (p, s)
/// svd_uvpart | mat_lside | mat_herm   | U* b    | (m, s) | (p, s)
///
/// svd_uvpart | mat_rside | mat_ntrans | b U     | (s, m) | (s, p)
/// svd_uvpart | mat_rside | mat_trans  | b U'    | (s, p) | (s, m)
/// svd_uvpart | mat_rside | mat_herm   | b U*    | (s, p) | (s, m)
///
/// svd_uvfull | mat_lside | mat_ntrans | U b     | (m, s) | (m, s)
/// svd_uvfull | mat_lside | mat_trans  | U' b    | (m, s) | (m, s)
/// svd_uvfull | mat_lside | mat_herm   | U* b    | (m, s) | (m, s)
///
/// svd_uvfull | mat_rside | mat_ntrans | b U     | (s, m) | (s, m)
/// svd_uvfull | mat_rside | mat_trans  | b U'    | (s, m) | (s, m)
/// svd_uvfull | mat_rside | mat_herm   | b U*    | (s, m) | (s, m)
template <typename T>
template <mat_op_type       tr,
	  product_side_type ps,
	  typename          Block0,
	  typename          Block1>
bool
Svd<T>::produ(const_Matrix<T, Block0> b,
	      Matrix<T, Block1>       x) VSIP_NOTHROW
{
  length_type prod_m;
  length_type prod_n;
  length_type m_int = u_.size(0);
  length_type n_int = u_.size(1);

  // SAL actually returns U as transposed. However, because our block type
  // is setup to be column major, the matrix looks normal.
  if(tr == mat_trans || tr == mat_herm) 
  {
    ut_ = u_.transpose();
    std::swap(m_int,n_int);
    if(tr == mat_herm) 
    {
      ut_ = impl_conj(ut_);
    }
  } 

  // left side or right side?
  if(ps == mat_lside)
  {
    prod_m = m_int;
    prod_n = b.size(1);
    assert(b.size(0) == n_int);
    assert(x.size(0) == prod_m && x.size(1) == prod_n);
    sal_svd_prod_uv(((tr == mat_ntrans)? u_:ut_),b,x);
  }
  else {
    prod_m = b.size(0);
    prod_n = n_int;
    assert(b.size(1) == m_int);
    assert(x.size(0) == prod_m && x.size(1) == prod_n);
    sal_svd_prod_uv(b,((tr == mat_ntrans)? u_:ut_),x);
  }
  return true;
}



template <typename T>
template <mat_op_type       tr,
	  product_side_type ps,
	  typename          Block0,
	  typename          Block1>
bool
Svd<T>::prodv(const_Matrix<T, Block0> b,
	      Matrix<T, Block1>       x) VSIP_NOTHROW
{
  length_type prod_m;
  length_type prod_n;
  length_type m_int = v_.size(0);
  length_type n_int = v_.size(1);

  // SAL actually returns V as transposed. However, because our block type
  // is setup to be column major, the matrix looks normal.
  if(tr == mat_trans || tr == mat_herm) 
  {
    vt_ = v_.transpose();
    std::swap(m_int,n_int);
    if(tr == mat_herm) 
    {
      vt_ = impl_conj(vt_);
    }
  } 

  // left side or right side?
  if(ps == mat_lside)
  {
    prod_m = m_int;
    prod_n = b.size(1);
    assert(b.size(0) == n_int);
    assert(x.size(0) == prod_m && x.size(1) == prod_n);
    sal_svd_prod_uv(((tr == mat_ntrans)? v_:vt_),b,x);
  }
  else
  {
    prod_m = b.size(0);
    prod_n = n_int;
    assert(b.size(1) == m_int);
    assert(x.size(0) == prod_m && x.size(1) == prod_n);
    sal_svd_prod_uv(b,((tr == mat_ntrans)? v_:vt_),x);
  }
  return true;
}

// This helper function is necessary because when we want a submatrix of u or
// v, we need to return a real matrix if we are using real svd. Because u and
// v are always complex but have 0 for the imaginary parts, this function
// returns the real part if the result must be real.
template <typename Block0, typename Block1>
void sal_svd_get_uv(const_Matrix<complex<float>,Block0> a,
		    Matrix<float,Block1> ar)
{
  ar = a.real();
}

template <typename Block0, typename Block1>
void sal_svd_get_uv(const_Matrix<complex<float>,Block0> a,
		    Matrix<complex<float>,Block1> ar)
{
  ar = a;
}



/// Return the submatrix U containing columns (low .. high) inclusive.
template <typename T>
template <typename Block>
bool
Svd<T>::u(index_type low, index_type high, Matrix<T, Block> u) VSIP_NOTHROW
{
  assert((ust_ == svd_uvpart && high < p_) || (ust_ == svd_uvfull && high < m_));
  assert(u.size(0) == m_);
  assert(u.size(1) == high - low + 1);

  sal_svd_get_uv(u_(Domain<2>(m_, Domain<1>(low, 1, high-low+1))),u);

  return true;
}

/// Return the submatrix V containing columns (low .. high) inclusive.
template <typename T>
template <typename Block>
bool
Svd<T>::v(index_type low, index_type high, Matrix<T, Block> v) VSIP_NOTHROW
{
  assert((vst_ == svd_uvpart && high < p_) || (vst_ == svd_uvfull && high < n_));
  assert(v.size(0) == n_);
  assert(v.size(1) == high - low + 1);

  sal_svd_get_uv(v_(Domain<2>(Domain<1>(low, 1, high-low+1), n_)),v);

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
struct Evaluator<op::svd, be::mercury_sal, T>
{
  static bool const ct_valid = is_same<T, float>::value ||
    is_same<T, complex<float> >::value;
  typedef impl::sal::Svd<T> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
