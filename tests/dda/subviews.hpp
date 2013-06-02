//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef dda_subviews_hpp_
#define dda_subviews_hpp_

#include <iostream>
#include <cassert>
#include <vsip/support.hpp>
#include <vsip/dense.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include "test.hpp"
#include "output.hpp"

using namespace vsip;
using namespace ovxx;

template <typename BlockT>
void
dump_access_details()
{
  std::cout << "Access details (block_type = " << ovxx::type_name<BlockT>() << std::endl;

  typedef vsip::dda::dda_block_layout<BlockT> dbl_type;
  typedef typename dbl_type::access_type access_type;
  typedef typename dbl_type::order_type  order_type;
  typedef typename dbl_type::packing   packing;
  typedef typename dbl_type::layout_type layout_type;

  std::cout << "  dbl access_type = " << ovxx::type_name<access_type>() << std::endl;
  std::cout << "  dbl order_type  = " << ovxx::type_name<order_type>() << std::endl;
  std::cout << "  dbl packing   = " << ovxx::type_name<packing>() << std::endl;

  typedef vsip::get_block_layout<BlockT> bl_type;
  typedef typename bl_type::access_type bl_access_type;
  typedef typename bl_type::order_type  bl_order_type;
  typedef typename bl_type::packing   bl_packing;
  typedef typename bl_type::layout_type bl_layout_type;

  std::cout << "  bl  access_type = " << ovxx::type_name<bl_access_type>() << std::endl;
  std::cout << "  bl  order_type  = " << ovxx::type_name<bl_order_type>() << std::endl;
  std::cout << "  bl  packing   = " << ovxx::type_name<bl_packing>() << std::endl;

  typedef typename ovxx::dda::get_block_access<BlockT, layout_type, vsip::dda::in>::type
    use_access_type;

  std::cout << "  use_access_type = " << ovxx::type_name<use_access_type>() << std::endl;

  std::cout << "  cost            = " << vsip::dda::Data<BlockT, vsip::dda::in>::ct_cost
	    << std::endl;
}

/// Vector subview of a vector.

template <typename       T,
	  typename       BlockT>
void
test_vector_subview(
  Vector<T, BlockT> view,
  Domain<1> const&  subdom)
{
  typedef Vector<T, BlockT>                 view_type;
  typedef typename view_type::subview_type  subview_type;
  typedef typename subview_type::block_type subblock_type;

  typedef storage_traits<typename subblock_type::value_type, 
			 get_block_layout<subblock_type>::storage_format>
    storage;
  typedef typename storage::ptr_type ptr_type;

  view = T();
  test_assert((vsip::dda::Data<BlockT, vsip::dda::in>::ct_cost == 0));
  test_assert((vsip::dda::Data<subblock_type, vsip::dda::in>::ct_cost == 0));

  subview_type subv = view(subdom);

  {
    vsip::dda::Data<subblock_type, vsip::dda::inout> data(subv.block());

    // This should be direct access
    test_assert(data.ct_cost         == 0);
    test_assert(data.size(0) == subdom[0].size());

    ptr_type    ptr     = data.ptr();
    stride_type stride0 = data.stride(0);
    
    for (index_type i=0; i<data.size(0); ++i)
    {
      test_assert(equal(storage::get(ptr, i*stride0), T()));
      storage::put(ptr, i*stride0, T(i));
    }
  }

  for (index_type i=0; i<subdom[0].size(); ++i)
  {
    index_type ni = subdom[0].impl_nth(i);

    test_assert(equal(view.get(ni), T(i)));
    view.put(ni, T(i + 100));
  }

  {
    vsip::dda::Data<subblock_type, vsip::dda::inout> data(subv.block());

    // This should be direct access
    test_assert(data.ct_cost         == 0);
    test_assert(data.size(0) == subdom[0].size());

    ptr_type    ptr     = data.ptr();
    stride_type stride0 = data.stride(0);
    
    for (index_type i=0; i<data.size(0); ++i)
    {
      test_assert(equal(storage::get(ptr, i*stride0), T(i + 100)));
    }
  }
}



template <typename       T,
	  typename       BlockT>
void
test_vector_subview_cases(
  Vector<T, BlockT> view)
{
  length_type size = view.size(0);

  test_vector_subview(view, Domain<1>(size-2));
  test_vector_subview(view, Domain<1>(2, 1, size-2));
  test_vector_subview(view, Domain<1>(0, 2, size/2));
  test_vector_subview(view, Domain<1>(1, 2, size/2));
  test_vector_subview(view, Domain<1>(size-2, -1, size-2));
  test_vector_subview(view, Domain<1>(size-1, -1, size-2));
}




template <typename       T,
	  typename       BlockT>
void
test_vector_realimag(Vector<T, BlockT>, false_type)
{
}



template <typename       T,
	  typename       BlockT>
void
test_vector_realimag(Vector<complex<T>, BlockT> view, true_type)
{
  typedef Vector<complex<T>, BlockT>          view_type;
  typedef typename view_type::realview_type   realview_type;
  typedef typename realview_type::block_type  realblock_type;
  typedef typename view_type::imagview_type   imagview_type;
  typedef typename imagview_type::block_type  imagblock_type;

  view = complex<T>();

  test_assert((vsip::dda::Data<BlockT, vsip::dda::inout>::ct_cost == 0));
  test_assert((vsip::dda::Data<realblock_type, vsip::dda::inout>::ct_cost == 0));
  test_assert((vsip::dda::Data<imagblock_type, vsip::dda::inout>::ct_cost == 0));

  realview_type real = view.real();
  imagview_type imag = view.imag();

  // Initialize the view using DDA on the real subview.
  {
    vsip::dda::Data<realblock_type, vsip::dda::inout> data(real.block());

    // This should be direct access
    test_assert(data.ct_cost         == 0);
    test_assert(data.size(0) == view.size(0));

    T* ptr              = data.ptr();
    stride_type stride0 = data.stride(0);
    
    for (index_type i=0; i<data.size(0); ++i)
    {
      test_assert(equal(ptr[i*stride0], T()));
      ptr[i*stride0] = T(3*i+1);
    }
  }

  // Initialize the view using DDA on the imag subview.
  {
    vsip::dda::Data<imagblock_type, vsip::dda::inout> data(imag.block());

    // This should be direct access
    test_assert(data.ct_cost         == 0);
    test_assert(data.size(0) == view.size(0));

    T* ptr              = data.ptr();
    stride_type stride0 = data.stride(0);
    
    for (index_type i=0; i<data.size(0); ++i)
    {
      test_assert(equal(ptr[i*stride0], T()));
      ptr[i*stride0] = T(4*i+1);
    }
  }

  // Check the original views using get/put.
  for (index_type i=0; i<view.size(0); ++i)
  {
    test_assert(equal(real.get(i), T(3*i+1)));
    test_assert(equal(imag.get(i), T(4*i+1)));
    test_assert(equal(view.get(i), complex<T>(3*i+1, 4*i+1)));

    view.put(i, complex<T>(5*i+1, 7*i+1));
  }

  // Check and change the view using DDA on the real & imag subviews.
  {
    vsip::dda::Data<realblock_type, vsip::dda::inout> rdata(real.block());
    vsip::dda::Data<imagblock_type, vsip::dda::inout> idata(imag.block());

    test_assert(rdata.size(0) == view.size(0));
    test_assert(idata.size(0) == view.size(0));

    T* rptr              = rdata.ptr();
    T* iptr              = idata.ptr();
    stride_type rstride0 = rdata.stride(0);
    stride_type istride0 = idata.stride(0);
    
    for (index_type i=0; i<view.size(0); ++i)
    {
      test_assert(equal(rptr[i*rstride0], T(5*i+1)));
      test_assert(equal(iptr[i*istride0], T(7*i+1)));

      rptr[i*rstride0] = T(3*i+2);
      iptr[i*istride0] = T(2*i+1);
    }
  }

  // Check the original views using get/put.
  for (index_type i=0; i<view.size(0); ++i)
  {
    test_assert(equal(real.get(i), T(3*i+2)));
    test_assert(equal(imag.get(i), T(2*i+1)));
    test_assert(equal(view.get(i), complex<T>(3*i+2, 2*i+1)));
  }
}



template <typename T,
	  typename BlockT>
void
test_vector(Vector<T, BlockT> view)
{
  length_type size = view.size(0);

  test_vector_subview_cases(view);
  test_vector_subview_cases(view(Domain<1>(size-2)));

  test_vector_realimag(view,
		       integral_constant<bool, is_complex<T>::value>());
  test_vector_realimag(view(Domain<1>(size-2)),
		       integral_constant<bool, is_complex<T>::value>());
}



/// Row subviews of matrix.

template <typename T,
	  typename BlockT>
void
test_row_subview(
  Matrix<T, BlockT> mat)
{
  typedef Matrix<T, BlockT>             view_type;
  typedef typename view_type::row_type  row_type;
  typedef typename row_type::block_type row_block_type;
  typedef typename vsip::dda::Data<row_block_type, vsip::dda::inout>::ptr_type ptr_type;
  typedef storage_traits<typename row_block_type::value_type,
			 get_block_layout<row_block_type>::storage_format> storage;

  length_type rows = mat.size(0);
  length_type cols = mat.size(1);

  mat = T();

  test_assert((vsip::dda::Data<BlockT, vsip::dda::inout>::ct_cost == 0));
  test_assert((vsip::dda::Data<row_block_type, vsip::dda::inout>::ct_cost == 0));

  // Initialize the matrix using DDA by row subview.
  for (index_type r=0; r<rows; ++r)
  {
    row_type row = mat.row(r);

    {
      vsip::dda::Data<row_block_type, vsip::dda::inout> data(row.block());

      test_assert(data.ct_cost         == 0);
      test_assert(data.size(0) == cols);

      ptr_type ptr    = data.ptr();
      stride_type  stride = data.stride(0);

      for (index_type i=0; i<data.size(0); ++i)
      {
	test_assert(equal(storage::get(ptr, i*stride), T()));
	storage::put(ptr, i*stride, T(r*cols+i));
      }
    }
  }

  // Check the matrix using get/put.
  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
    {
      test_assert(equal(mat(r, c), T(r*cols+c)));
      mat(r, c) += T(1);
    }

  // Check the matrix using DDA by row subview.
  for (index_type r=0; r<rows; ++r)
  {
    row_type row = mat.row(r);

    {
      vsip::dda::Data<row_block_type, vsip::dda::inout> data(row.block());

      test_assert(data.ct_cost         == 0);
      test_assert(data.size(0) == cols);

      ptr_type ptr   = data.ptr();
      stride_type stride = data.stride(0);

      for (index_type i=0; i<data.size(0); ++i)
	test_assert(equal(storage::get(ptr, i*stride), T(r*cols+i + 1)));
    }
  }
}



template <typename T,
	  typename BlockT>
void
test_col_subview(
  Matrix<T, BlockT> mat)
{
  typedef Matrix<T, BlockT>             view_type;
  typedef typename view_type::col_type  col_type;
  typedef typename col_type::block_type col_block_type;
  typedef typename vsip::dda::Data<col_block_type, vsip::dda::inout>::ptr_type ptr_type;
  typedef storage_traits<typename col_block_type::value_type,
			 get_block_layout<col_block_type>::storage_format> storage;

  length_type rows = mat.size(0);
  length_type cols = mat.size(1);

  mat = T();

  test_assert((vsip::dda::Data<BlockT, vsip::dda::inout>::ct_cost == 0));
  test_assert((vsip::dda::Data<col_block_type, vsip::dda::inout>::ct_cost == 0));

  // Initialize the matrix using DDA by col subview.
  for (index_type c=0; c<cols; ++c)
  {
    col_type col = mat.col(c);

    {
      vsip::dda::Data<col_block_type, vsip::dda::inout> data(col.block());

      test_assert(data.ct_cost         == 0);
      test_assert(data.size(0) == rows);

      ptr_type ptr    = data.ptr();
      stride_type  stride = data.stride(0);

      for (index_type i=0; i<data.size(0); ++i)
      {
	test_assert(equal(storage::get(ptr, i*stride), T()));
	storage::put(ptr, i*stride, T(c*rows+i));
      }
    }
  }

  // Check the matrix using get/put.
  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
    {
      test_assert(equal(mat(r, c), T(c*rows+r)));
      mat(r, c) += T(1);
    }

  // Check the matrix using DDA by col subview.
  for (index_type c=0; c<cols; ++c)
  {
    col_type col = mat.col(c);

    {
      vsip::dda::Data<col_block_type, vsip::dda::inout> data(col.block());

      test_assert(data.ct_cost         == 0);
      test_assert(data.size(0) == rows);

      ptr_type ptr    = data.ptr();
      stride_type  stride = data.stride(0);

      for (index_type i=0; i<data.size(0); ++i)
	test_assert(equal(storage::get(ptr, i*stride), T(c*rows+i + 1)));
    }
  }
}



template <typename T,
          typename BlockT>
void
test_diag_subview(
  Matrix<T, BlockT> mat)
{
  typedef Matrix<T, BlockT>                  view_type;
  typedef typename view_type::diag_type      diagview_type;
  typedef typename diagview_type::block_type subblock_type;
  typedef typename vsip::dda::Data<subblock_type, vsip::dda::inout>::ptr_type ptr_type;
  typedef storage_traits<typename subblock_type::value_type,
			 get_block_layout<subblock_type>::storage_format> storage;

  length_type rows = mat.size(0);
  length_type cols = mat.size(1);

  mat = T();

  test_assert((vsip::dda::Data<BlockT, vsip::dda::inout>::ct_cost == 0));
  // If the access cost to diagview_type::block_type is 0,
  // then direct access is being used.
  test_assert((vsip::dda::Data<subblock_type, vsip::dda::inout>::ct_cost == 0));

  // Initialize the matrix using put with count
  for (index_type r=0; r<rows; ++r) {
    for (index_type c=0; c<cols; ++c) {
      mat.put(r, c, T(r * cols + c));
    }
  }

  // Check Diagonal view
  index_difference_type max_col = static_cast<index_difference_type>(cols);
  for ( index_difference_type d = 1 - rows; d < max_col; d++ )
  {
    diagview_type diagv = mat.diag(d);
    length_type size = diagv.size();

    { // use a block to limit the lifetime of data object.
      vsip::dda::Data<subblock_type, vsip::dda::inout> data(diagv.block());

      ptr_type ptr = data.ptr();
      stride_type  str = data.stride(0);

      for (index_type i = 0; i < size; ++i )
      {
        if ( d >= 0 )
          test_assert(equal(storage::get(ptr, i*str), T(i * cols + i + d) ) );
        else
          test_assert(equal(storage::get(ptr, i*str), T(i * cols + i - (d * cols)) ) );

        storage::put(ptr, i*str, T(OVXX_PI + i));
      }
    } 

    // Check them through the Matrix view
    for ( index_type i = 0; i < size; i++ )
    {
      if ( d >= 0 )
        test_assert( equal( mat(i, i + d), T(OVXX_PI + i) ) );
      else
        test_assert( equal( mat(i - d, i), T(OVXX_PI + i) ) );
    }
  }
}


/// Matrix subview of a matrix.

template <typename       T,
	  typename       BlockT>
void
test_matrix_subview(
  Matrix<T, BlockT> view,
  Domain<2> const&  subdom)
{
  typedef Matrix<T, BlockT>                 view_type;
  typedef typename view_type::subview_type  subview_type;
  typedef typename subview_type::block_type subblock_type;
  typedef typename vsip::dda::Data<subblock_type, vsip::dda::inout>::ptr_type ptr_type;
  typedef storage_traits<typename subblock_type::value_type,
			 get_block_layout<subblock_type>::storage_format> storage;

  view = T();

  test_assert((vsip::dda::Data<BlockT, vsip::dda::inout>::ct_cost == 0));
  test_assert((vsip::dda::Data<subblock_type, vsip::dda::inout>::ct_cost == 0));

  subview_type subv = view(subdom);

  {
    vsip::dda::Data<subblock_type, vsip::dda::inout> data(subv.block());

    // This should be direct access
    test_assert(data.ct_cost         == 0);
    test_assert(data.size(0) == subdom[0].size());
    test_assert(data.size(1) == subdom[1].size());

    ptr_type ptr     = data.ptr();
    stride_type  stride0 = data.stride(0);
    stride_type  stride1 = data.stride(1);
    
    for (index_type i=0; i<data.size(0); ++i)
      for (index_type j=0; j<data.size(1); ++j)
      {
	test_assert(equal(storage::get(ptr, i*stride0 + j*stride1), T()));
	storage::put(ptr, i*stride0 + j*stride1, T(i*data.size(1)+j));
      }
  }

  for (index_type i=0; i<subdom[0].size(); ++i)
    for (index_type j=0; j<subdom[1].size(); ++j)
    {
      index_type ni = subdom[0].impl_nth(i);
      index_type nj = subdom[1].impl_nth(j);

      test_assert(view.get(ni, nj) == T(i*subdom[1].size()+j));
      view.put(ni, nj, T(i + j*subdom[0].size() + 100));
    }

  {
    vsip::dda::Data<subblock_type, vsip::dda::inout> data(subv.block());

    // This should be direct access
    test_assert(data.ct_cost         == 0);
    test_assert(data.size(0) == subdom[0].size());
    test_assert(data.size(1) == subdom[1].size());

    ptr_type ptr     = data.ptr();
    stride_type  stride0 = data.stride(0);
    stride_type  stride1 = data.stride(1);
    
    for (index_type i=0; i<data.size(0); ++i)
      for (index_type j=0; j<data.size(1); ++j)
      {
	test_assert(equal(storage::get(ptr, i*stride0 + j*stride1),
			  T(i + j*data.size(0) + 100)));
      }
  }
}



template <typename       T,
	  typename       BlockT>
void
test_matrix_subview_cases(
  Matrix<T, BlockT> view)
{
  length_type rows = view.size(0);
  length_type cols = view.size(1);

  test_matrix_subview(view, Domain<2>(rows-2, cols-2));
  test_matrix_subview(view, Domain<2>(Domain<1>(2, 1, rows-2),
				     Domain<1>(2, 1, cols-2)));
  test_matrix_subview(view, Domain<2>(Domain<1>(0, 2, rows/2),
				     Domain<1>(0, 2, cols/2)));
  test_matrix_subview(view, Domain<2>(Domain<1>(1, 2, rows/2),
				     Domain<1>(1, 2, cols/2)));

  test_matrix_subview(view, Domain<2>(Domain<1>(rows-2, -1, rows-2),
				     Domain<1>(cols-2, -1, cols-2)));
}



template <typename       T,
	  typename       BlockT>
void
test_matrix_realimag(Matrix<T, BlockT>, false_type)
{
}



template <typename       T,
	  typename       BlockT>
void
test_matrix_realimag(Matrix<complex<T>, BlockT> view, true_type)
{
  typedef Matrix<complex<T>, BlockT>          view_type;
  typedef typename view_type::realview_type   realview_type;
  typedef typename realview_type::block_type  realblock_type;
  typedef typename view_type::imagview_type   imagview_type;
  typedef typename imagview_type::block_type  imagblock_type;

  view = complex<T>();

  test_assert((vsip::dda::Data<BlockT, vsip::dda::inout>::ct_cost == 0));
  test_assert((vsip::dda::Data<realblock_type, vsip::dda::inout>::ct_cost == 0));
  test_assert((vsip::dda::Data<imagblock_type, vsip::dda::inout>::ct_cost == 0));

  realview_type real = view.real();
  imagview_type imag = view.imag();

  // Initialize the view using DDA on the real subview.
  {
    vsip::dda::Data<realblock_type, vsip::dda::inout> data(real.block());

    // This should be direct access
    test_assert(data.ct_cost         == 0);
    test_assert(data.size(0) == view.size(0));
    test_assert(data.size(1) == view.size(1));

    T* ptr              = data.ptr();
    stride_type stride0 = data.stride(0);
    stride_type stride1 = data.stride(1);
    
    for (index_type i=0; i<data.size(0); ++i)
      for (index_type j=0; j<data.size(1); ++j)
      {
	index_type idx = (i*data.size(1)+j);
	test_assert(equal(ptr[i*stride0 + j*stride1], T()));
	ptr[i*stride0 + j*stride1] = T(3*idx+1);
      }
  }

  // Initialize the view using DDA on the imag subview.
  {
    vsip::dda::Data<imagblock_type, vsip::dda::inout> data(imag.block());

    // This should be direct access
    test_assert(data.ct_cost         == 0);
    test_assert(data.size(0) == view.size(0));
    test_assert(data.size(1) == view.size(1));

    T* ptr              = data.ptr();
    stride_type stride0 = data.stride(0);
    stride_type stride1 = data.stride(1);
    
    for (index_type i=0; i<data.size(0); ++i)
      for (index_type j=0; j<data.size(1); ++j)
      {
	index_type idx = (i*data.size(1)+j);
	test_assert(equal(ptr[i*stride0 + j*stride1], T()));
	ptr[i*stride0 + j*stride1] = T(4*idx+1);
      }
  }

  // Check the original views using get/put.
  for (index_type i=0; i<view.size(0); ++i)
    for (index_type j=0; j<view.size(1); ++j)
    {
      index_type idx = (i*view.size(1)+j);

      test_assert(equal(real.get(i, j), T(3*idx+1)));
      test_assert(equal(imag.get(i, j), T(4*idx+1)));
      test_assert(equal(view.get(i, j), complex<T>(3*idx+1, 4*idx+1)));

      view.put(i, j, complex<T>(5*idx+1, 7*idx+1));
  }

  // Check and change the view using DDA on the real & imag subviews.
  {
    vsip::dda::Data<realblock_type, vsip::dda::inout> rdata(real.block());
    vsip::dda::Data<imagblock_type, vsip::dda::inout> idata(imag.block());

    T* rptr              = rdata.ptr();
    T* iptr              = idata.ptr();
    stride_type rstride0 = rdata.stride(0);
    stride_type rstride1 = rdata.stride(1);
    stride_type istride0 = idata.stride(0);
    stride_type istride1 = idata.stride(1);
    
    for (index_type i=0; i<view.size(0); ++i)
      for (index_type j=0; j<view.size(1); ++j)
      {
	index_type idx = (i*view.size(1)+j);

	test_assert(equal(rptr[i*rstride0+j*rstride1], T(5*idx+1)));
	test_assert(equal(iptr[i*istride0+j*istride1], T(7*idx+1)));

	rptr[i*rstride0+j*rstride1] = T(3*idx+2);
	iptr[i*istride0+j*rstride1] = T(2*idx+1);
      }
  }

  // Check the original views using get/put.
  for (index_type i=0; i<view.size(0); ++i)
    for (index_type j=0; j<view.size(1); ++j)
    {
      index_type idx = (i*view.size(1)+j);

      test_assert(equal(real.get(i, j), T(3*idx+2)));
      test_assert(equal(imag.get(i, j), T(2*idx+1)));
      test_assert(equal(view.get(i, j), complex<T>(3*idx+2, 2*idx+1)));
    }
}



/// Transpose subview of a matrix.

template <typename       T,
	  typename       BlockT>
void
test_matrix_transpose(
  Matrix<T, BlockT> /*view*/)
{
}



template <typename T,
	  typename BlockT>
void
test_matrix(Matrix<T, BlockT> view)
{
  test_row_subview(view);
  test_col_subview(view);

  length_type rows = view.size(0);
  length_type cols = view.size(1);

  test_matrix_subview_cases(view);
  test_matrix_subview_cases(view(Domain<2>(rows-2, cols-2)));

  test_matrix_realimag(view,
		       integral_constant<bool, is_complex<T>::value>());
  test_matrix_realimag(view(Domain<2>(rows-2, cols-2)),
		       integral_constant<bool, is_complex<T>::value>());

  test_diag_subview(view);

  test_matrix_transpose(view);

  for (index_type i=0; i<view.size(0); ++i)
    test_vector(view.row(i));

  for (index_type i=0; i<view.size(1); ++i)
    test_vector(view.col(i));
}

template <typename       T,
	  typename       BlockT,
	  dimension_type FixedDim>
struct Tensor_vector_subview;



template <typename       T,
	  typename       BlockT>
struct Tensor_vector_subview<T, BlockT, 0>
{
  static dimension_type const D1 = 1;
  static dimension_type const D2 = 2;

  typedef typename Tensor<T, BlockT>::template subvector<D1, D2>::type
    subv_type;

  
  static subv_type subv(Tensor<T, BlockT> view, index_type j, index_type k)
    { return view(vsip::whole_domain, j, k); }

  static T value(Tensor<T, BlockT> view,
		 index_type i, index_type j, index_type k)
  { return T(i+j*view.size(0) + k*view.size(0)*view.size(D1)); }

  static T get(Tensor<T, BlockT> view,
	       index_type i, index_type j, index_type k)
  { return view.get(i, j, k); }

  static void put(Tensor<T, BlockT> view,
	       index_type i, index_type j, index_type k,
	       T value)
  { view.put(i, j, k, value); }
};



template <typename       T,
	  typename       BlockT>
struct Tensor_vector_subview<T, BlockT, 1>
{
  static dimension_type const D1 = 0;
  static dimension_type const D2 = 2;

  typedef typename Tensor<T, BlockT>::template subvector<D1, D2>::type
    subv_type;

  
  static subv_type subv(Tensor<T, BlockT> view, index_type j, index_type k)
    { return view(j, vsip::whole_domain, k); }

  static T value(Tensor<T, BlockT> view,
		 index_type i, index_type j, index_type k)
  { return T(i+j*view.size(1) + k*view.size(1)*view.size(D1)); }

  static T get(Tensor<T, BlockT> view,
	       index_type i, index_type j, index_type k)
    { return view.get(j, i, k); }

  static void put(Tensor<T, BlockT> view,
	       index_type i, index_type j, index_type k,
	       T value)
    { view.put(j, i, k, value); }
};



template <typename       T,
	  typename       BlockT>
struct Tensor_vector_subview<T, BlockT, 2>
{
  static dimension_type const D1 = 0;
  static dimension_type const D2 = 1;

  typedef typename Tensor<T, BlockT>::template subvector<D1, D2>::type
    subv_type;

  
  static subv_type subv(Tensor<T, BlockT> view, index_type j, index_type k)
    { return view(j, k, vsip::whole_domain); }

  static T value(Tensor<T, BlockT> view,
		 index_type i, index_type j, index_type k)
  { return T(i+j*view.size(2) + k*view.size(2)*view.size(D1)); }

  static T get(Tensor<T, BlockT> view,
	       index_type i, index_type j, index_type k)
    { return view.get(j, k, i); }

  static void put(Tensor<T, BlockT> view,
	       index_type i, index_type j, index_type k,
	       T value)
    { view.put(j, k, i, value); }
};



template <dimension_type FreeDim,
	  typename       T,
	  typename       BlockT>
void
test_tensor_vector_subview(
  Tensor<T, BlockT> view)
{
  typedef Tensor_vector_subview<T, BlockT, FreeDim> info_type;

  typedef Tensor<T, BlockT>              view_type;
  typedef typename info_type::subv_type  subv_type;
  typedef typename subv_type::block_type subv_block_type;
  typedef typename vsip::dda::Data<subv_block_type, vsip::dda::inout>::ptr_type ptr_type;
  typedef storage_traits<typename subv_block_type::value_type,
			 get_block_layout<subv_block_type>::storage_format> storage;

  dimension_type const D1 = info_type::D1;
  dimension_type const D2 = info_type::D2;

  view = T();

  test_assert((vsip::dda::Data<BlockT, vsip::dda::inout>::ct_cost == 0));
  // test_assert(dda::Data<subv_block_type>::ct_cost == 0);

  // dump_access_details<BlockT>();
  // dump_access_details<subv_block_type>();

  // Initialize the view using DDA by 0'th dim vector subview.
  for (index_type j=0; j<view.size(D1); ++j)
    for (index_type k=0; k<view.size(D2); ++k)
    {
      subv_type subv = info_type::subv(view, j, k);

    {
      vsip::dda::Data<subv_block_type, vsip::dda::inout> data(subv.block());

      test_assert(data.ct_cost         == 0);
      test_assert(data.size(0) == view.size(FreeDim));

      ptr_type ptr    = data.ptr();
      stride_type  stride = data.stride(0);

      for (index_type i=0; i<data.size(0); ++i)
      {
	test_assert(equal(storage::get(ptr, i*stride), T()));
	storage::put(ptr, i*stride, info_type::value(view, i, j, k));
      }
    }
  }

  // Check the view using get/put.
  for (index_type i=0; i<view.size(FreeDim); ++i)
    for (index_type j=0; j<view.size(D1); ++j)
      for (index_type k=0; k<view.size(D2); ++k)
      {
	test_assert(equal(info_type::get(view, i, j, k),
		     info_type::value(view, i, j, k)));
	info_type::put(view, i, j, k,
		       info_type::value(view, i, j, k) + T(1));
      }

  // Check the view using DDA by 0'th dim vector subview.
  for (index_type j=0; j<view.size(D1); ++j)
    for (index_type k=0; k<view.size(D2); ++k)
    {
      subv_type subv = info_type::subv(view, j, k);

    {
      vsip::dda::Data<subv_block_type, vsip::dda::inout> data(subv.block());

      test_assert(data.ct_cost         == 0);
      test_assert(data.size(0) == view.size(FreeDim));

      ptr_type ptr    = data.ptr();
      stride_type  stride = data.stride(0);

      for (index_type i=0; i<data.size(0); ++i)
      {
	test_assert(equal(storage::get(ptr, i*stride),
		     info_type::value(view, i, j, k) + T(1) ));
      }
    }
  }
}



/// Tensor subview of a tensor.

template <typename       T,
	  typename       BlockT>
void
test_tensor_subview(
  Tensor<T, BlockT> view,
  Domain<3> const&  subdom)
{
  typedef Tensor<T, BlockT>                 view_type;
  typedef typename view_type::subview_type  subview_type;
  typedef typename subview_type::block_type subblock_type;
  typedef typename vsip::dda::Data<subblock_type, vsip::dda::inout>::ptr_type ptr_type;
  typedef storage_traits<typename subblock_type::value_type,
			 get_block_layout<subblock_type>::storage_format> storage;

  view = T();

  test_assert((vsip::dda::Data<BlockT, vsip::dda::inout>::ct_cost == 0));
  test_assert((vsip::dda::Data<subblock_type, vsip::dda::inout>::ct_cost == 0));

  subview_type subv = view(subdom);

  {
    vsip::dda::Data<subblock_type, vsip::dda::inout> data(subv.block());

    // This should be direct access
    test_assert(data.ct_cost         == 0);
    test_assert(data.size(0) == subdom[0].size());
    test_assert(data.size(1) == subdom[1].size());
    test_assert(data.size(2) == subdom[2].size());

    ptr_type ptr     = data.ptr();
    stride_type  stride0 = data.stride(0);
    stride_type  stride1 = data.stride(1);
    stride_type  stride2 = data.stride(2);
    
    for (index_type i=0; i<data.size(0); ++i)
      for (index_type j=0; j<data.size(1); ++j)
	for (index_type k=0; k<data.size(2); ++k)
	{
	  test_assert(equal(storage::get(ptr, i*stride0 + j*stride1 + k*stride2), T()));
	  storage::put(ptr, i*stride0 + j*stride1 + k*stride2,
		       T(i*data.size(1)*data.size(2)+j*data.size(2)+k));
	}
  }

  for (index_type i=0; i<subdom[0].size(); ++i)
    for (index_type j=0; j<subdom[1].size(); ++j)
      for (index_type k=0; k<subdom[2].size(); ++k)
      {
	index_type ni = subdom[0].impl_nth(i);
	index_type nj = subdom[1].impl_nth(j);
	index_type nk = subdom[2].impl_nth(k);

	test_assert(equal(view.get(ni, nj, nk),
		     T(i*subdom[1].size()*subdom[2].size() +
		       j*subdom[2].size()+k)));
	view.put(ni, nj, nk,
		 T(i +
		   j*subdom[0].size() +
		   k*subdom[0].size()*subdom[1].size() + 100));
    }

  {
    vsip::dda::Data<subblock_type, vsip::dda::inout> data(subv.block());

    // This should be direct access
    test_assert(data.ct_cost         == 0);
    test_assert(data.size(0) == subdom[0].size());
    test_assert(data.size(1) == subdom[1].size());
    test_assert(data.size(2) == subdom[2].size());

    ptr_type ptr     = data.ptr();
    stride_type  stride0 = data.stride(0);
    stride_type  stride1 = data.stride(1);
    stride_type  stride2 = data.stride(2);
    
    for (index_type i=0; i<data.size(0); ++i)
      for (index_type j=0; j<data.size(1); ++j)
	for (index_type k=0; k<data.size(2); ++k)
	{
	  test_assert(equal(storage::get(ptr, i*stride0 + j*stride1 + k*stride2),
		       T(i +
			 j*data.size(0) +
			 k*data.size(0)*data.size(1) + 100)));
	}
  }
}



template <typename T,
	  typename BlockT>
void
test_tensor_subview_cases(Tensor<T, BlockT> view)
{
  length_type size0 = view.size(0);
  length_type size1 = view.size(1);
  length_type size2 = view.size(2);

  test_tensor_subview(view, Domain<3>(size0-2, size1-2, size2-2));
  test_tensor_subview(view, Domain<3>(Domain<1>(2, 1, size0-2),
				      Domain<1>(2, 1, size1-2),
				      Domain<1>(2, 1, size2-2)));

}



template <typename       T,
	  typename       BlockT>
void
test_tensor_realimag(Tensor<T, BlockT>, false_type)
{
}



template <typename       T,
	  typename       BlockT>
void
test_tensor_realimag(Tensor<complex<T>, BlockT> view, true_type)
{
  typedef Tensor<complex<T>, BlockT>          view_type;
  typedef typename view_type::realview_type   realview_type;
  typedef typename realview_type::block_type  realblock_type;
  typedef typename view_type::imagview_type   imagview_type;
  typedef typename imagview_type::block_type  imagblock_type;

  view = complex<T>();

  test_assert((vsip::dda::Data<BlockT, vsip::dda::inout>::ct_cost == 0));
  test_assert((vsip::dda::Data<realblock_type, vsip::dda::inout>::ct_cost == 0));
  test_assert((vsip::dda::Data<imagblock_type, vsip::dda::inout>::ct_cost == 0));

  realview_type real = view.real();
  imagview_type imag = view.imag();

  // Initialize the view using DDA on the real subview.
  {
    vsip::dda::Data<realblock_type, vsip::dda::inout> data(real.block());

    // This should be direct access
    test_assert(data.ct_cost         == 0);
    test_assert(data.size(0) == view.size(0));
    test_assert(data.size(1) == view.size(1));
    test_assert(data.size(2) == view.size(2));

    T* ptr              = data.ptr();
    stride_type stride0 = data.stride(0);
    stride_type stride1 = data.stride(1);
    stride_type stride2 = data.stride(2);
    
    for (index_type i=0; i<data.size(0); ++i)
      for (index_type j=0; j<data.size(1); ++j)
	for (index_type k=0; k<data.size(2); ++k)
	{
	  index_type idx = (i*view.size(1)*view.size(2)+j*view.size(2)+k);
	  test_assert(equal(ptr[i*stride0 + j*stride1 + k*stride2], T()));
	  ptr[i*stride0 + j*stride1 + k*stride2] = T(3*idx+1);
	}
  }

  // Initialize the view using DDA on the imag subview.
  {
    vsip::dda::Data<imagblock_type, vsip::dda::inout> data(imag.block());

    // This should be direct access
    test_assert(data.ct_cost         == 0);
    test_assert(data.size(0) == view.size(0));
    test_assert(data.size(1) == view.size(1));
    test_assert(data.size(2) == view.size(2));

    T* ptr              = data.ptr();
    stride_type stride0 = data.stride(0);
    stride_type stride1 = data.stride(1);
    stride_type stride2 = data.stride(2);
    
    for (index_type i=0; i<data.size(0); ++i)
      for (index_type j=0; j<data.size(1); ++j)
	for (index_type k=0; k<data.size(2); ++k)
	{
	  index_type idx = (i*view.size(1)*view.size(2)+j*view.size(2)+k);
	  test_assert(equal(ptr[i*stride0 + j*stride1 + k*stride2], T()));
	  ptr[i*stride0 + j*stride1 + k*stride2] = T(4*idx+1);
	}
  }

  // Check the original views using get/put.
  for (index_type i=0; i<view.size(0); ++i)
    for (index_type j=0; j<view.size(1); ++j)
      for (index_type k=0; k<view.size(2); ++k)
      {
	index_type idx = (i*view.size(1)*view.size(2)+j*view.size(2)+k);

	test_assert(equal(real.get(i, j, k), T(3*idx+1)));
	test_assert(equal(imag.get(i, j, k), T(4*idx+1)));
	test_assert(equal(view.get(i, j, k), complex<T>(3*idx+1, 4*idx+1)));

	view.put(i, j, k, complex<T>(5*idx+1, 7*idx+1));
      }

  // Check and change the view using DDA on the real & imag subviews.
  {
    vsip::dda::Data<realblock_type, vsip::dda::inout> rdata(real.block());
    vsip::dda::Data<imagblock_type, vsip::dda::inout> idata(imag.block());

    T* rptr              = rdata.ptr();
    T* iptr              = idata.ptr();
    stride_type rstride0 = rdata.stride(0);
    stride_type rstride1 = rdata.stride(1);
    stride_type rstride2 = rdata.stride(2);
    stride_type istride0 = idata.stride(0);
    stride_type istride1 = idata.stride(1);
    stride_type istride2 = idata.stride(2);
    
    for (index_type i=0; i<view.size(0); ++i)
      for (index_type j=0; j<view.size(1); ++j)
	for (index_type k=0; k<view.size(2); ++k)
	{
	  index_type idx = (i*view.size(1)*view.size(2)+j*view.size(2)+k);

	  test_assert(equal(rptr[i*rstride0+j*rstride1+k*rstride2], T(5*idx+1)));
	  test_assert(equal(iptr[i*istride0+j*istride1+k*istride2], T(7*idx+1)));

	  rptr[i*rstride0+j*rstride1+k*rstride2] = T(3*idx+2);
	  iptr[i*istride0+j*rstride1+k*istride2] = T(2*idx+1);
	}
  }

  // Check the original views using get/put.
  for (index_type i=0; i<view.size(0); ++i)
    for (index_type j=0; j<view.size(1); ++j)
      for (index_type k=0; k<view.size(2); ++k)
      {
	index_type idx = (i*view.size(1)*view.size(2)+j*view.size(2)+k);

	test_assert(equal(real.get(i, j, k), T(3*idx+2)));
	test_assert(equal(imag.get(i, j, k), T(2*idx+1)));
	test_assert(equal(view.get(i, j, k), complex<T>(3*idx+2, 2*idx+1)));
      }
}



template <typename T,
	  typename BlockT>
void
test_tensor(Tensor<T, BlockT> view)
{
  test_tensor_vector_subview<0>(view);
  test_tensor_vector_subview<1>(view);
  test_tensor_vector_subview<2>(view);

  length_type size0 = view.size(0);
  length_type size1 = view.size(1);
  length_type size2 = view.size(2);

  test_tensor_subview_cases(view);
  test_tensor_subview_cases(view(Domain<3>(size0-2, size1-2, size2-2)));

  test_tensor_realimag(view,
		       integral_constant<bool, is_complex<T>::value>());
  test_tensor_realimag(view(Domain<3>(size0-2, size1-2, size2-2)),
		       integral_constant<bool, is_complex<T>::value>());

  for (index_type i=0; i<view.size(0); ++i)
    test_matrix(view(i, vsip::whole_domain, vsip::whole_domain));

  for (index_type i=0; i<view.size(1); ++i)
    test_matrix(view(vsip::whole_domain, i, vsip::whole_domain));

  for (index_type i=0; i<view.size(2); ++i)
    test_matrix(view(vsip::whole_domain, vsip::whole_domain, i));
}

#endif
