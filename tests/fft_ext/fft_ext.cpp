//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions
***********************************************************************/

/// Possible types of FFT
enum fft_type
{
  complex_to_complex = 0,
  complex_to_real,
  real_to_complex
};

/// Possible precision selection
enum precision_type
{
  both_precisions = 0,
  single_precision,
  double_precision
};
     
/// Used by main to pass to argp_parse()
struct arguments
{
  precision_type precision;
  fft_type fft; 
  char *filename;
};



/***********************************************************************
  Utility Functions
***********************************************************************/
     
/// Error metric between two vectors.
template <typename T1,
	  typename T2,
	  typename Block1,
	  typename Block2>
double
error_db(
  const_Vector<T1, Block1> v1,
  const_Vector<T2, Block2> v2)
{
  double refmax = 0.0;
  double maxsum = -250;
  double sum;

  for (index_type i=0; i<v1.size(); ++i)
  {
    double val = magsq(v1.get(i));
    if (val > refmax)
      refmax = val;
  }


  for (index_type i=0; i<v1.size(); ++i)
  {
    double val = magsq(v1.get(i) - v2.get(i));

    if (val < 1.e-20)
      sum = -201.;
    else
      sum = 10.0 * log10(val/(2.0*refmax));

    if (sum > maxsum)
      maxsum = sum;
  }

  return maxsum;
}


/// Test by-value Fft.
template <typename F, typename T1, typename T2>
void
fft_by_value(Vector<T1> in, Vector<T2> test, scalar_f scale)
{
  length_type N = 0;

  if ( in.size() >= test.size() )
    N = in.size();
  else
    N = test.size();

  F t_fft(Domain<1>(N), scale);

  Vector<T2> out(test.size());
  out = t_fft(in);

  test_assert(error_db(test, out) < -100);
}


/// Complex to Complex FFT
template<typename T>
void test_fft_cc (char *filename)
{
  ifstream ifile(filename, ios::in);
  if (ifile.fail()) 
  {
    cerr << "Failed to open file " << filename << endl;
    test_assert(0);
    return;
  }

  length_type size = 0;
  scalar_f scale = 0;
  int ntimes = 0;
  char line[120];

  // first line contains three values separated by spaces
  ifile.getline(line, sizeof(line));
  istringstream (line) >> size >> scale >> ntimes; 

  Vector<complex<T> > input(size);
  Vector<complex<T> > expected(size);

  // format: 
  //   <r_input>,<c_input>  <r_expected>,<c_expected>
  index_type i;
  T val1, val2, val3, val4;
  for ( i = 0; i < size; ++i )
  {
    ifile.getline(line, sizeof(line), ',');
    istringstream (line) >> val1;
    
    ifile.getline(line, sizeof(line), ',');
    istringstream (line) >> val2 >> val3;
    
    ifile.getline(line, sizeof(line));
    istringstream (line) >> val4;
    
    input.put( i, complex<float>(val1, val2) );
    expected.put( i, complex<float>(val3, val4) );
  }

  typedef Fft<const_Vector, complex<T>, complex<T>, fft_fwd, by_value, 1, alg_space>
    cc_fft_type;

  fft_by_value<cc_fft_type>(input, expected, scale);
}


// Complex to Real FFT
template<typename T>
void test_fft_cr (char *filename)
{
  ifstream ifile(filename, ios::in);
  if (ifile.fail()) 
  {
    cerr << "Failed to open file " << filename << endl;
    test_assert(0);
    return;
  }

  length_type size = 0;
  scalar_f scale = 0;
  int ntimes = 0;
  char line[120];

  ifile.getline(line, sizeof(line));
  istringstream (line) >> size >> scale >> ntimes; 

  test_assert( (size / 2) * 2 == size );

  Vector<complex<T> > input(size / 2 + 1);
  Vector<T> expected(size);

  // format: 
  //   <r_input>,<c_input>  <r_expected>
  index_type i;
  T val1, val2, val3, val4;
  for ( i = 0; i < size; ++i )
  {
    ifile.getline(line, sizeof(line), ',');
    istringstream (line) >> val1;
    
    ifile.getline(line, sizeof(line));
    istringstream (line) >> val2 >> val3;
    val4 = 0;
    
    if ( i < size / 2 + 1 )
      input.put( i, complex<T>(val1, val2) );
    expected.put( i, T(val3) );
  }

  typedef Fft<const_Vector, complex<T>, T, 0, by_value, 1, alg_space>
	cr_fft_type;

  fft_by_value<cr_fft_type>(input, expected, scale);
}


// Real to Complex FFT
template<typename T>
void test_fft_rc (char *filename)
{
  ifstream ifile(filename, ios::in);
  if (ifile.fail()) 
  {
    cerr << "Failed to open file " << filename << endl;
    test_assert(0);
    return;
  }

  length_type size = 0;
  scalar_f scale = 0;
  int ntimes = 0;
  char line[120];

  ifile.getline(line, sizeof(line));
  istringstream (line) >> size >> scale >> ntimes; 

  test_assert( (size / 2) * 2 == size );

  Vector<T> input(size);
  Vector<complex<T> > expected(size / 2 + 1);

  // format: 
  //   <r_input>  <r_expected>,<c_expected>
  index_type i;
  T val1, val2, val3, val4;
  for ( i = 0; i < size; ++i )
  {
    ifile.getline(line, sizeof(line), ',');
    istringstream (line) >> val1 >> val3;
    val2 = 0;
    
    ifile.getline(line, sizeof(line));
    istringstream (line) >> val4;
    
    input.put( i, T(val1) );
    if ( i < size / 2 + 1 )
      expected.put( i, complex<T>(val3, val4) );
  }

  typedef Fft<const_Vector, T, complex<T>, 0, by_value, 1, alg_space>
	rc_fft_type;

  fft_by_value<rc_fft_type>(input, expected, scale);
}


// macro to make running the tests a little neater

#if defined(VSIP_IMPL_FFT_USE_FLOAT) && defined(VSIP_IMPL_FFT_USE_DOUBLE)
#define TEST_RUN_FFT( func, name, prec ) \
    { \
      if ( prec != double_precision ) \
        func<float>(name); \
      if ( prec != single_precision ) \
        func<double>(name); \
    }

#elif defined(VSIP_IMPL_FFT_USE_DOUBLE)
#define TEST_RUN_FFT( func, name, prec ) \
    { \
      if ( prec != single_precision ) \
        func<double>(name); \
    }

#elif defined(VSIP_IMPL_FFT_USE_FLOAT)
#define TEST_RUN_FFT( func, name, prec ) \
    { \
      if ( prec != double_precision ) \
        func<float>(name); \
    }

#else
#define TEST_RUN_FFT( func, name, prec )

#endif




/***********************************************************************
  Main
***********************************************************************/

/// This program does the following:
///   Reads two vectors from a data file.
///   Performs an FFT on the first vector.
///   Computes a relative measure of the error between
///     the result and the second vector read from the file.
///
int main (int argc, char **argv)
{
  vsipl init(argc, argv);
  struct arguments arguments;
     
  /* Default values. */
  arguments.precision = both_precisions;
  arguments.fft = complex_to_complex;
  arguments.filename = NULL;
     
  if ( argc == 2 )
    arguments.filename = argv[1];
  else
  {
    std::cerr << "Invalid number of arguments." << std::endl;
    return EXIT_FAILURE;
  }


  // Check the first two letters of the filename to see if they
  // match a pattern that will allow us to deduce the fft type.  
  // Currently, these are accepted:
  //   "cc*" - complex-complex
  //   "cr*" - complex-real
  //   "rc*" - real-complex
  std::string fullpath(arguments.filename);

  // strip path information
  int index = fullpath.rfind( '/' );
  if ( index >= 0 )
  {
    std::string filename( fullpath.substr( index + 1, fullpath.length() - index - 1 ) );
    
    if ( filename.length() >= 2 )
    {
      if ( filename.substr(0, 2) == "cc" )      arguments.fft = complex_to_complex;
      else if ( filename.substr(0, 2) == "cr" ) arguments.fft = complex_to_real;
      else if ( filename.substr(0, 2) == "rc" ) arguments.fft = real_to_complex;
    }
  }


  switch ( arguments.fft )
  {
  case complex_to_real:
    TEST_RUN_FFT( test_fft_cr, arguments.filename, arguments.precision );
    break;
  case real_to_complex:
    TEST_RUN_FFT( test_fft_rc, arguments.filename, arguments.precision );
    break;
  case complex_to_complex:
    TEST_RUN_FFT( test_fft_cc, arguments.filename, arguments.precision );
    break;
  }


  return EXIT_SUCCESS;
}
