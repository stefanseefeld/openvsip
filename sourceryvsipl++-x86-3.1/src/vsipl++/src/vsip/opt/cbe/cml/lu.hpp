/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_OPT_CBE_CML_LU_HPP
#define VSIP_OPT_CBE_CML_LU_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/dda.hpp>
#include <vsip/core/signal/fir_backend.hpp>
#include <vsip/opt/dispatch.hpp>

#include <cml.h>

namespace vsip
{
namespace impl
{
namespace cml
{

/// LUD implementation using CML library.
///
/// Requires:
///   T to be a value type supported by CML's LU routines (i.e., float)
template <typename T>
class Lud
{
  static storage_format_type const storage_format = dense_complex_format;

  typedef Layout<2, row2_type, dense, storage_format> data_LP;
  typedef Strided<2, T, data_LP> data_block_type;

  typedef Layout<2, col2_type, dense, storage_format> t_data_LP;
  typedef Strided<2, T, t_data_LP> t_data_block_type;

public:
  Lud(length_type rows) VSIP_THROW((std::bad_alloc))
  : length_     (rows),
    data_       (length_, length_)
  {
    assert(length_ > 0);
    if (!cml_lud_create_f(&lud_obj_handle, rows))
      VSIP_IMPL_THROW(std::bad_alloc());
  }
  Lud(Lud const &lu) VSIP_THROW((std::bad_alloc))
  : length_(lu.length_),
    data_(length_, length_)
  {
    // Copy the LU product data.
    data_ = lu.data_;

    // Create a new LU object of the appropriate size.
    if (!cml_lud_create_f(&lud_obj_handle, length_))
      VSIP_IMPL_THROW(std::bad_alloc());

    // Set an appropriate stride value for the LU matrix.  (Note that
    // the actual LU matrix pointer is always reset before use.)
    lud_obj_handle.LU_stride_0 = length_;

    // Copy permutation matrix.
    memcpy(lu.lud_obj_handle.P, lud_obj_handle.P, length_ * sizeof(size_t));
  }

  ~Lud() VSIP_NOTHROW { cml_lud_destroy_f(&lud_obj_handle);}

  length_type length() const VSIP_NOTHROW { return length_;}

  /// Form LU factorization of matrix M
  ///
  /// Requires
  ///   M to be a square matrix
  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW
  {
    assert(m.size(0) == length_ && m.size(1) == length_);
    assign_local(data_, m);
    dda::Data<data_block_type, dda::inout> data(data_.block());
    return cml_lud_decompose_f(&lud_obj_handle, data.ptr(), length_); 
  }

  template <mat_op_type tr,
            typename    Block0,
            typename    Block1>
  bool solve(const_Matrix<T, Block0>, Matrix<T, Block1>) VSIP_NOTHROW;

  length_type max_decompose_size();

private:
  typedef std::vector<T, Aligned_allocator<T> > vector_type;

  Lud &operator=(Lud const&) VSIP_NOTHROW;

  length_type  length_;                 // Order of A.
  Matrix<T, data_block_type> data_;	// Factorized matrix A
  cml_lud_f lud_obj_handle;
};

/// Solve Op(A) x = b (where A previously given to decompose)
///
/// Op(A) is
///   A   if tr == mat_ntrans
///   A'T if tr == mat_trans
///
/// Requires
///   B to be a (length, P) matrix
///   X to be a (length, P) matrix
///
/// Effects:
///   X contains solution to Op(A) X = B

template <typename T>
template <mat_op_type tr,
          typename    Block0,
          typename    Block1>
bool
Lud<T>::solve(const_Matrix<T, Block0> b,
	      Matrix<T, Block1>       x) VSIP_NOTHROW
{
  // CML only supports float LU decompositions, so calling this with
  // a hermitian is invalid.
  assert(tr != mat_herm);

  // Check that input and output sizes match.
  assert(x.size(0) == length_);
  assert(b.size(0) == x.size(0) && b.size(1) == x.size(1));

  // We need to put the LU storage array back into ext_data form,
  // and make sure the object pointer points to the correct place
  // still.
  dda::Data<data_block_type, dda::inout> data(data_.block());
  lud_obj_handle.LU = data.ptr();

  // CML does an in-place solve, so we must first do a copy.  We
  // use a local temporary to ensure it has the appropriate block
  // type.
  Matrix<T, data_block_type> x_local(x.size(0), x.size(1));
  assign_local(x_local, b);

  {
    dda::Data<data_block_type, dda::inout> x_data(x_local.block());

    if(tr == mat_ntrans) 
    {
      cml_lud_sol_f(&lud_obj_handle,
		    x_data.ptr(),
		    x.size(1),
		    x.size(0),
		    x.size(1));
    }
    else
    {
      cml_lud_solt_f(&lud_obj_handle,
		     x_data.ptr(),
		     x.size(1),
		     x.size(0),
		     x.size(1));
    }
  }

  assign_local(x, x_local);
  
  return true;
}

} // namespace vsip::impl::cml
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <typename T>
struct Evaluator<op::lud, be::cml, T>
{
  static bool const ct_valid = is_same<T, float>::value;
  typedef impl::cml::Lud<T> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
