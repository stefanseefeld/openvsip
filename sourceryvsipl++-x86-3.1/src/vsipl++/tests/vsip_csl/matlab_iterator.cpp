/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/matlab_bin_file_test.cpp
    @author  Assem Salama
    @date    2006-07-18
    @brief   VSIPL++ Library: Test for reading and writing Matlab .mat files 
             using iterators
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <vsip_csl/matlab_file.hpp>
#include <vsip_csl/output.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;

float           increment(float v)  { return v+1; }
double          increment(double v) { return v+1; }
complex<float>  increment(complex<float>  v) { complex<float> i(1.,1.);
					       return v+i; }
complex<double> increment(complex<double> v) { complex<double> i(1.,1.);
                                               return v+i; }

template <typename T,
	  typename Block0,
	  template <typename,typename> class View>

void view_test(View<T,Block0> view)
{
  dimension_type const View_dim = View<T,Block0>::dim;
  vsip::impl::Length<View_dim> v_extent = extent(view);
  Index<View_dim> my_index;
  T view_data,comp_data;

  comp_data = 0;
  for(index_type i=0;i<view.size();i++)
  {
    comp_data=increment(comp_data);
    view_data = get(view,my_index);
    
    test_assert(comp_data == view_data);
      
    my_index = next(v_extent,my_index);
  }
}	

template <typename T>
void read_view_test(Matlab_file::iterator iterator, Matlab_file &mf)
{
  Matlab_view_header *header = *iterator;
  if(header->num_dims == 2 && (header->dims[0] == 1 || header->dims[1] == 1))
  {
    // vector
    Vector<T> a(std::max(header->dims[0],header->dims[1]));
    mf.read_view(a,iterator);
    view_test(a);
  } else if(header->num_dims == 2)
  {
    // matrix
    Matrix<T> a(header->dims[0],header->dims[1]);
    mf.read_view(a,iterator);
    view_test(a);
  } else if(header->num_dims == 3)
  {
    // tensor
    Tensor<T> a(header->dims[0],header->dims[1],header->dims[2]);
    mf.read_view(a,iterator);
    view_test(a);
  }
}

template <typename T>
void tensor_test(length_type m, length_type n, length_type o,
		 std::ofstream &ofs, char const *name)
{
  Tensor<T> a(m,n,o);
  T         value;

  value = 0;
  for(length_type i=0;i<m;i++) {
    for(length_type j=0;j<n;j++) {
      for(length_type k=0;k<o;k++) {
        value = increment(value);
        a.put(i,j,k,value);
      }
    }
  }

  // write it out to file
  ofs << Matlab_bin_formatter<Tensor<T> >(a,name);
}


template <typename T>
void matrix_test(length_type m, length_type n, std::ofstream &ofs, char const *name)
{
  Matrix<T> a(m,n);
  T         value;

  value = 0;
  for(length_type i=0;i<m;i++) {
    for(length_type j=0;j<n;j++) {
      value = increment(value);
      a.put(i,j,value);
    }
  }

  // write it out to file
  ofs << Matlab_bin_formatter<Matrix<T> >(a,name);
}

template <typename T>
void vector_test(length_type m, std::ofstream &ofs, char const *name)
{
  Vector<T> a(m);
  T         value;

  value = 0;
  for(length_type i=0;i<m;i++) {
    value = increment(value);
    a.put(i,value);
  }

  // write it out to file
  ofs << Matlab_bin_formatter<Vector<T> >(a,name);
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  // We need to generate the matlab file first.
  {
    std::ofstream ofs("temp.mat");

    // write header
    ofs << Matlab_bin_hdr("example");

    // tests
  
    vector_test<float>            (5,ofs,"float_vector");
    vector_test<double>           (5,ofs,"double_vector");
    vector_test<complex<float> >  (5,ofs,"cplx_float_vector");
    vector_test<complex<double> > (5,ofs,"cplx_double_vector");
  
    matrix_test<float>            (2,3,ofs,"float_matrix");
    matrix_test<double>           (2,3,ofs,"double_matrix");
    matrix_test<complex<float> >  (2,3,ofs,"cplx_float_matrix");
    matrix_test<complex<double> > (2,3,ofs,"cplx_double_matrix");

    tensor_test<float>            (3,2,3,ofs,"float_tensor");
    tensor_test<double>           (3,2,3,ofs,"double_tensor");
    tensor_test<complex<float> >  (3,2,3,ofs,"cplx_float_tensor");
    tensor_test<complex<double> > (3,2,3,ofs,"cplx_double_tensor");
  }
  
  
  Matlab_file mf("temp.mat");
  Matlab_file::iterator begin = mf.begin();
  Matlab_file::iterator end   = mf.end();
  Matlab_view_header *temp_p;

  while(begin != end)
  {
    temp_p = *begin;
    if(temp_p->is_complex) 
    {
      if(temp_p->class_type == matlab::mxSINGLE_CLASS)
        read_view_test<complex<float> >(begin,mf);
      if(temp_p->class_type == matlab::mxDOUBLE_CLASS)
        read_view_test<complex<double> >(begin,mf);
    }
    else
    {
      if(temp_p->class_type == matlab::mxSINGLE_CLASS)
        read_view_test<float>(begin,mf);
      if(temp_p->class_type == matlab::mxDOUBLE_CLASS)
        read_view_test<double>(begin,mf);
    }
    ++begin;
  }
  return 0;
}
