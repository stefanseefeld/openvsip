/* Copyright (c) 2006, 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/matlab_bin_file/matlab_bin_file.cpp
    @author  Assem Salama
    @date    2006-07-18
    @brief   VSIPL++ Library: Test for reading and writing Matlab .mat files
*/

/* A sample input file can be created by the following Matlab code:
=====================================================================

% Clear all pre-existing variables.
clear all

% Define base arrays.

m = 0;
for i = 1:5
  m = m + 1;
  base_vector(i) = m;
end

m = 0;
for i = 1:2
  for j = 1:3
    m = m + 1;
    base_matrix(i,j) = m;
  end
end

m = 0;
for i = 1:3
  for j = 1:2
    for k = 1:3
      m = m + 1;
      base_tensor(i,j,k) = m;
    end
  end
end

% Convert base arrays to various data formats.

int_vector = int32(base_vector);
uint_vector = uint32(base_vector);
float_vector = single(base_vector);
double_vector = double(base_vector);
cplx_float_vector = single(base_vector * (1+1i));
cplx_double_vector = double(base_vector * (1+1i));

float_matrix = single(base_matrix);
double_matrix = double(base_matrix);
cplx_float_matrix = single(base_matrix * (1+1i));
cplx_double_matrix = double(base_matrix * (1+1i));

float_tensor = single(base_tensor);
double_tensor = double(base_tensor);
cplx_float_tensor = single(base_tensor * (1+1i));
cplx_double_tensor = double(base_tensor * (1+1i));

float_colvector = single(base_vector');

floattodouble_vector = float_vector;
floattocplx_vector = float_vector;
doubletofloat_vector = double_vector;
uinttoint_vector = uint_vector;
inttofloat_vector = int_vector;

% Create output file.

save -v6 matlab-ref-testdata.mat ...
  int_vector uint_vector ...
  float_vector double_vector cplx_float_vector cplx_double_vector ...
  float_matrix double_matrix cplx_float_matrix cplx_double_matrix ...
  float_tensor double_tensor cplx_float_tensor cplx_double_tensor ...
  float_colvector ...
  floattodouble_vector floattocplx_vector doubletofloat_vector ...
  uinttoint_vector inttofloat_vector ...

=====================================================================
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <iomanip>
#include <fstream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <vsip_csl/matlab_text_formatter.hpp>
#include <vsip_csl/matlab_bin_formatter.hpp>
#include <vsip_csl/output.hpp>
#include <vsip_csl/test.hpp>

#define DEBUG 0

using namespace vsip;
using namespace vsip_csl;
using namespace std;

int             increment(int v)  { return v+1; }
unsigned int    increment(unsigned int v)  { return v+1; }
float           increment(float v)  { return v+1; }
double          increment(double v) { return v+1; }
complex<float>  increment(complex<float>  v) { complex<float> i(1.,1.);
					       return v+i; }
complex<double> increment(complex<double> v) { complex<double> i(1.,1.);
                                               return v+i; }

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

template <typename T>
void vector_input_test(length_type m, std::ifstream &ifs,
		       char const *name, Matlab_bin_hdr &h)
{
  Vector<T> a(m);
  T         value,input_value;

  ifs >> Matlab_bin_formatter<Vector<T> >(a,name,h);

  value = 0;
  for(length_type i=0;i<m;i++) {
    value = increment(value);
    input_value = a.get(i);
    test_assert(value == input_value);
  }
}

template <typename T>
void matrix_input_test(length_type m, length_type n, std::ifstream &ifs,
		       char const *name, Matlab_bin_hdr &h)
{
  Matrix<T> a(m,n);
  T         value,input_value;

  ifs >> Matlab_bin_formatter<Matrix<T> >(a,name,h);

  value = 0;
  for(length_type i=0;i<m;i++) {
    for(length_type j=0;j<n;j++) {
      value = increment(value);
      input_value = a.get(i,j);
      test_assert(value == input_value);
    }
  }
}

template <typename T>
void tensor_input_test(length_type m, length_type n, length_type o,
		       std::ifstream &ifs, char const *name, Matlab_bin_hdr &h)
{
  Tensor<T> a(m,n,o);
  T         value,input_value;

  ifs >> Matlab_bin_formatter<Tensor<T> >(a,name,h);

  value = 0;
  for(length_type i=0;i<m;i++) {
    for(length_type j=0;j<n;j++) {
      for(length_type k=0;k<o;k++) {
        value = increment(value);
        input_value = a.get(i,j,k);
        test_assert(value == input_value);
      }
    }
  }
}

template <typename T>
void vector_promote_test(length_type m, std::ifstream &ifs,
			 char const *name, Matlab_bin_hdr &h)
{
  Vector<T> a(m);
  T         value,input_value;

  ifs >> Matlab_bin_formatter<Vector<T> >(a,name,h);

  value = 0;
  for(length_type i=0;i<m;i++) {
    value = value + T(1);
    input_value = a.get(i);
    test_assert(value == input_value);
  }
}


void
write_file(char const* name)
{
  std::ofstream ofs(name);

  // write header
  ofs << Matlab_bin_hdr("example");

  // Tests of various View types
  vector_test<int>              (5,ofs,"int_vector");
  vector_test<unsigned int>     (5,ofs,"uint_vector");
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

  // Tests of input/output mismatches
  vector_test<float>            (5,ofs,"float_colvector");
  vector_test<float>            (5,ofs,"floattodouble_vector");
  vector_test<float>            (5,ofs,"floattocplx_vector");
  vector_test<double>           (5,ofs,"doubletofloat_vector");
  vector_test<unsigned int>     (5,ofs,"uinttoint_vector");
  vector_test<int>              (5,ofs,"inttofloat_vector");
}



void
read_file(char const* name)
{
  std::ifstream ifs(name);

  // Skip header
  Matlab_bin_hdr h;
  ifs >> h;

  // Tests of various View types
  vector_input_test<int>                  (5,ifs,"int_vector",h);
  vector_input_test<unsigned int>         (5,ifs,"uint_vector",h);
  vector_input_test<float>                (5,ifs,"float_vector",h);
  vector_input_test<double>               (5,ifs,"double_vector",h);
  vector_input_test<complex<float> >      (5,ifs,"float_vector",h);
  vector_input_test<complex<double> >     (5,ifs,"double_vector",h);

  matrix_input_test<float>                (2,3,ifs,"float_matrix",h);
  matrix_input_test<double>               (2,3,ifs,"double_matrix",h);
  matrix_input_test<complex<float> >      (2,3,ifs,"cplx_float_matrix",h);
  matrix_input_test<complex<double> >     (2,3,ifs,"cplx_double_matrix",h);
  
  tensor_input_test<float>                (3,2,3,ifs,"float_tensor",h);
  tensor_input_test<double>               (3,2,3,ifs,"double_tensor",h);
  tensor_input_test<complex<float> >      (3,2,3,ifs,"cplx_float_tensor",h);
  tensor_input_test<complex<double> >     (3,2,3,ifs,"cplx_double_tensor",h);

  // Tests of input/output mismatches
  // (In "real Matlab" test files, float_colvector is a column vector)
  vector_promote_test<float>              (5,ifs,"float_colvector",h);
  vector_promote_test<double>             (5,ifs,"floattodouble_vector",h);
  vector_promote_test<complex<float> >    (5,ifs,"floattocplx_vector",h);
  vector_promote_test<float>              (5,ifs,"doubletofloat_vector",h);
  vector_promote_test<int>                (5,ifs,"uinttoint_vector",h);
  vector_promote_test<float>              (5,ifs,"inttofloat_vector",h);
}



int main(int ac, char** av)
{
  vsipl init(ac, av);

  write_file("temp.mat");

  // Read what we just wrote.
  read_file("temp.mat");		

  // Read a reference file if given.
  if (ac == 2)
  {
    char const* file = av[1];
    read_file(file);
  }

  return 0;
}
