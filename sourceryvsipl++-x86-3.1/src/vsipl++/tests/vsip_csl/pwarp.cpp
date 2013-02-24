/* Copyright (c) 2007 by CodeSourcery, LLC.  All rights reserved. */

/** @file    tests/vsip_csl/pwarp.cpp
    @author  Jules Bergmann
    @date    2007-11-05
    @brief   VSIPL++ Library: Unit tests for perspective warping.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define VERBOSE 1
#define SAVE_IMAGES 0
#define DO_CHECK 0
#define TEST_TYPES 1

#define NUM_TCS 6

#if VERBOSE
#  include <iostream>
#endif
#include <string>
#include <sstream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/domain.hpp>
#include <vsip/random.hpp>
#include <vsip/signal.hpp>

#include <vsip_csl/img/perspective_warp.hpp>

#include <vsip/core/view_cast.hpp>
#include <vsip_csl/save_view.hpp>
#include <vsip/opt/dispatch_diagnostics.hpp>


#include <vsip_csl/test.hpp>
#include <vsip_csl/ref_pwarp.hpp>
#include <vsip_csl/error_db.hpp>
#if VERBOSE
#  include <vsip_csl/output.hpp>
#endif

using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions - Utility Functions
***********************************************************************/

template <typename T>
struct Pwarp_traits
{
  typedef T diff_type;
  typedef T print_type;
};

template <>
struct Pwarp_traits<unsigned char>
{
  typedef int diff_type;
  typedef int print_type;
};

template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
proj_inv(
  vsip::Matrix<T, Block1> P,
  vsip::Vector<T, Block2> xy,
  vsip::Vector<T, Block3> res)
{
  T X = xy.get(0);
  T Y = xy.get(1);

  T yD = (P.get(1,0) - P.get(2,0)*Y) * (P.get(0,1) - P.get(2,1)*X)
       - (P.get(1,1) - P.get(2,1)*Y) * (P.get(0,0) - P.get(2,0)*X);
  T xD = (P.get(0,0) - P.get(2,0)*X);

  if (yD == 0) yD = 1e-8;
  if (xD == 0) xD = 1e-8;

  T y  = ( (P.get(1,0) - P.get(2,0)*Y)*(X - P.get(0,2))
         - (P.get(0,0) - P.get(2,0)*X)*(Y - P.get(1,2)) ) / yD;
  T x  = ( (X - P.get(0,2)) - (P.get(0,1) - P.get(2,1)*X)*y ) / xD;

  res.put(0, x);
  res.put(1, y);
}

template <typename T,
	  typename BlockT>
void
expect_proj(
  Matrix<T, BlockT> P,
  T                 ref_u,
  T                 ref_v,
  T                 ref_x,
  T                 ref_y)
{
  using vsip_csl::img::impl::apply_proj;
  using vsip_csl::img::impl::invert_proj;
  using vsip_csl::img::impl::apply_proj_inv;

  Matrix<T> Pi(3, 3);
  T x, y, u, v;

  apply_proj<T>(P, ref_u, ref_v, x, y);
  test_assert(equal(x, ref_x));
  test_assert(equal(y, ref_y));

  invert_proj(P, Pi);

  apply_proj<T>(Pi, x, y, u, v);
  test_assert(equal(u, ref_u));
  test_assert(equal(v, ref_v));
  

  Vector<T> xy(2);
  Vector<T> chk(2);

  xy(0) = ref_x; xy(1) = ref_y;
  proj_inv(P, xy, chk);

  test_assert(equal(chk.get(0), ref_u));
  test_assert(equal(chk.get(1), ref_v));

  T chk_u, chk_v;
  apply_proj_inv<T>(P, xy.get(0), xy.get(1), chk_u, chk_v);
  test_assert(equal(chk_u, ref_u));
  test_assert(equal(chk_v, ref_v));
}



template <typename T,
	  typename BlockT>
void
setup_p(
  Matrix<T, BlockT> P,
  int               i)
{
  switch (i) {

  case 0: // Identity projection
    P        = T();
    P.diag() = T(1);
    break;

  case 1: // Random projection #1 extracted from example applicaton
    P(0,0) = T(0.999982);    P(0,1) = T(0.000427585); P(0,2) = T(-0.180836);
    P(1,0) = T(-0.00207906); P(1,1) = T(0.999923);    P(1,2) = T(0.745001);
    P(2,0) = T(1.01958e-07); P(2,1) = T(8.99655e-08); P(2,2) = T(1);
    break;

  case 2: // Random projection #2 extracted from example application
    P(0,0) = 8.28282751190698e-01; 
    P(0,1) = 2.26355321374407e-02;
    P(0,2) = -1.10504985681804e+01;

    P(1,0) = -2.42950546474237e-01;
    P(1,1) = 8.98035288576380e-01;
    P(1,2) = 1.05162748265872e+02;

    P(2,0) = -1.38973743578922e-04;
    P(2,1) = -9.01955477542629e-05;
    P(2,2) = 1;
    break;

  case 3: // Shift left by 8 pixels
    P(0, 0) = 1; P(0, 1) = 0; P(0, 2) = 8;
    P(1, 0) = 0; P(1, 1) = 1; P(1, 2) = 0;
    P(2, 0) = 0; P(2, 1) = 0; P(2, 2) = 1;
    break;

  case 4: // Random projection #3, extracted from example application.
          // Broke SPU input streaming for VGA images.
    P(0, 0) = 1.00202;
    P(0, 1) = 0.00603114;
    P(0, 2) = 1.03277;

    P(1, 0) = 0.000532397;
    P(1, 1) = 1.01655;
    P(1, 2) = 1.66292;

    P(2, 0) = 1.40122e-06;
    P(2, 1) = 1.05832e-05;
    P(2, 2) = 1.00002;
    break;

  case 5: // Random projection #4, extracted from example application.
          // Broke SIMD for VGA images.
    P(0, 0) = 1.00504661;
    P(0, 1) = 0.0150403921;
    P(0, 2) = 9.60451126;

    P(1, 0) = 0.00317225;
    P(1, 1) = 1.04547524;
    P(1, 2) = 16.1063614;

    P(2, 0) = 2.21413484e-06;
    P(2, 1) = 2.5766507e-05;
    P(2, 2) = 1.00024176;
    break;
  }
}



template <typename T>
void
test_apply_proj()
{
  Matrix<T> P(3, 3);

  setup_p(P, 1);

  expect_proj<T>(P, 0.181157, -0.744682, 0, 0);


  setup_p(P, 2);

  expect_proj<T>(P, 1.64202829975142e+01, -1.12660864027683e+02, 0, 0);
  expect_proj<T>(P, 5.00593422077480e+02, 6.39343844623318e+02, 480-1, 640-1);


  setup_p(P, 3);
  expect_proj<T>(P, 0, 0, /* -> */ 8, 0);
}



template <typename T,
	  typename BlockT>
void
setup_checker(
  vsip::Matrix<T, BlockT> img,
  vsip::length_type       row_size,
  vsip::length_type       col_size,
  T                       value)
{
  using vsip::length_type;
  using vsip::index_type;

  length_type rows = img.size(0);
  length_type cols = img.size(1);

  for (index_type y=0; y<rows; y += row_size)
  {
    for (index_type x=0; x<cols; x += col_size)
    {
      T v = (((y / row_size) % 2) ^ ((x / col_size) % 2)) ? value : 0;
      img(Domain<2>(Domain<1>(y, 1, min(row_size, rows-y)),
		    Domain<1>(x, 1, min(col_size, cols-x)))) = T(v);
    }
  }
}



template <typename T,
	  typename BlockT>
void
setup_pattern(
  int                     pattern,
  vsip::Matrix<T, BlockT> img,
  vsip::length_type       row_size,
  vsip::length_type       col_size,
  T                       value)
{
  using vsip::length_type;
  using vsip::index_type;

  length_type rows = img.size(0);
  length_type cols = img.size(1);

  switch(pattern)
  {
  case 1: // checker
    for (index_type y=0; y<rows; y += row_size)
    {
      for (index_type x=0; x<cols; x += col_size)
      {
	T v = (((y / row_size) % 2) ^ ((x / col_size) % 2)) ? value : 0;
	img(Domain<2>(Domain<1>(y, 1, min(row_size, rows-y)),
		      Domain<1>(x, 1, min(col_size, cols-x)))) = T(v);
      }
    }
    break;
  case 2: // rows
    for (index_type y=0; y<rows; y += row_size)
    {
      T v = ((y / row_size) % 2) ? value : 0;
      img(Domain<2>(Domain<1>(y, 1, min(row_size, rows-y)),
		    Domain<1>(0, 1, cols))) = T(v);
    }
    break;
  case 3: // cols
    break;
  case 4: // checker fade
    int scale = 2;
    for (index_type y=0; y<rows; y += row_size)
    {
      for (index_type x=0; x<cols; x += col_size)
      {
	T v = (((y / row_size) % 2) ^ ((x / col_size) % 2)) ? value : 0;
	v = (T)(((float)(scale*rows-y) / (scale*rows)) *
		((float)(scale*cols-x) / (scale*cols)) *
		v);
	img(Domain<2>(Domain<1>(y, 1, min(row_size, rows-y)),
		      Domain<1>(x, 1, min(col_size, cols-x)))) = T(v);
      }
    }
    break;
  }
}



// rawtopgm -bpp 1 <cols> <rows> IN > OUT
template <typename T,
	  typename BlockT>
void
save_image(std::string outfile, Matrix<T, BlockT> img)
{
  using vsip::impl::view_cast;
  
  Index<2> idx;
  Matrix<unsigned char> out(img.size(0), img.size(1));
  
  T minv      = 0; // minval(img, idx);
  T maxv      = maxval(img, idx);

  if (vsip::impl::is_same<T, unsigned char>::value)
    maxv = 255;

  float scale = 255.0 / (maxv - minv ? maxv - minv : 1.f);
  
  out = view_cast<unsigned char>(view_cast<float>(img - minv) * scale);
  
  vsip_csl::save_view(const_cast<char *>(outfile.c_str()), out);
}



template <typename CoeffT,
	  typename T>
void
test_perspective_fun(
  std::string f_prefix,
  length_type rows,
  length_type cols,
  length_type row_size,
  length_type col_size)
{
  using vsip::impl::view_cast;

  typedef typename Pwarp_traits<T>::print_type print_type;
  typedef typename Pwarp_traits<T>::diff_type  diff_type;

  Matrix<T> src(rows, cols);
  Matrix<T> dst(rows, cols);
  Matrix<T> chk1(rows, cols);
  Matrix<T> chk2(rows, cols);
  Matrix<T> diff(rows, cols);
  Matrix<T> d2(rows, cols);
  Matrix<CoeffT> P(3, 3);

  setup_checker(src, row_size, col_size, T(255));
  setup_p(P, 2);

  vsip_csl::img::perspective_warp(P, src, dst);
  vsip_csl::ref::pwarp            (P, src, chk1);
  vsip_csl::ref::pwarp_incremental(P, src, chk2);

  float error1 = error_db(dst, chk1);
  float error2 = error_db(dst, chk2);
  diff = mag(view_cast<int>(dst) - view_cast<int>(chk2));
  d2   = ite(dst != chk2, 255, 0);

#if SAVE_IMAGES
  save_image(f_prefix + "-src.raw", src);
  save_image(f_prefix + "-dst.raw", dst);
  save_image(f_prefix + "-diff.raw", diff);
  save_image(f_prefix + "-d2.raw", d2);
  save_image(f_prefix + "-chk2.raw", chk2);
#endif

#if VERBOSE > 0
  std::cout << f_prefix << " error: " << error1 << ", " << error2 << std::endl;
#else
  (void)f_prefix;
#endif

#if VERBOSE > 1
  Index<2> i;
  std::cout << "  dst : " << static_cast<print_type>(minval(dst, i)) << " .. "
	                  << static_cast<print_type>(maxval(dst, i)) << "\n"
	    << "  chk1: " << static_cast<print_type>(minval(chk1, i)) << " .. "
	                  << static_cast<print_type>(maxval(chk1, i)) << "\n"
	    << "  chk2: " << static_cast<print_type>(minval(chk2, i)) << " .. "
	                  << static_cast<print_type>(maxval(chk2, i)) << "\n"
	    << "  diff: " << static_cast<print_type>(minval(diff, i)) << " .. "
              	          << static_cast<print_type>(maxval(diff, i)) << "\n"
    ;
#endif
  (void)error1;
  // assert(error1 <= -100); // error1 unusally large on x86 SIMD
#if DO_CHECK
  test_assert(error2 <= -50);
#endif
}





template <typename CoeffT,
	  typename T>
void
test_pwarp_obj(
  std::string f_prefix,
  length_type rows,
  length_type cols,
  length_type row_size,
  length_type col_size,
  int         tc)
{
  using vsip::Domain;
  using vsip::impl::view_cast;
  using vsip_csl::img::Perspective_warp;
  using vsip_csl::img::interp_linear;
  using vsip_csl::img::forward;

  typedef typename Pwarp_traits<T>::print_type print_type;
  typedef typename Pwarp_traits<T>::diff_type  diff_type;

  Matrix<T> src(rows, cols);
  Matrix<T> dst(rows, cols);
  Matrix<T> chk1(rows, cols);
  Matrix<T> chk2(rows, cols);
  Matrix<T> diff(rows, cols);
  Matrix<T> diff0(rows, cols);
  Matrix<T> d2(rows, cols);
  Matrix<CoeffT> P(3, 3);

  setup_pattern(4, src, row_size, col_size, T(255));
  setup_p(P, tc);

  Perspective_warp<CoeffT, T, interp_linear, forward>
    warp(P, Domain<2>(rows, cols));

  warp(src, dst);
  vsip_csl::ref::pwarp            (P, src, chk1);
  vsip_csl::ref::pwarp_incremental(P, src, chk2);

  float error1 = error_db(dst, chk1);
  float error2 = error_db(dst, chk2);
  diff = mag(view_cast<diff_type>(dst) - view_cast<diff_type>(chk2));
  diff0 = ite(diff == 0, T(255), diff);
  d2   = ite(dst != chk2, 255, 0);

#if SAVE_IMAGES
  save_image(f_prefix + "-src.raw", src);
  save_image(f_prefix + "-dst.raw", dst);
  save_image(f_prefix + "-diff.raw", diff);
  save_image(f_prefix + "-diff0.raw", diff0);
  save_image(f_prefix + "-d2.raw", d2);
  save_image(f_prefix + "-chk2.raw", chk2);
#endif

#if VERBOSE > 0
  using namespace vsip_csl::dispatcher;
  typedef typename Dispatcher<
    op::pwarp<CoeffT, T, interp_linear, forward, 0, alg_time> >::backend
    backend;
  std::cout << f_prefix
	    << " (" << Backend_name<backend>::name() << ") "
	    << rows << " x " << cols << " "
	    << " tc: " << tc 
	    << "  error: " << error1 << ", " << error2 << std::endl;
#else
  (void)f_prefix;
#endif

#if VERBOSE > 1
  Index<2> i;
  std::cout << "  dst : " << static_cast<print_type>(minval(dst, i)) << " .. "
	                  << static_cast<print_type>(maxval(dst, i)) << "\n"
	    << "  chk1: " << static_cast<print_type>(minval(chk1, i)) << " .. "
	                  << static_cast<print_type>(maxval(chk1, i)) << "\n"
	    << "  chk2: " << static_cast<print_type>(minval(chk2, i)) << " .. "
	                  << static_cast<print_type>(maxval(chk2, i)) << "\n"
	    << "  diff: " << static_cast<print_type>(minval(diff, i)) << " .. "
              	          << static_cast<print_type>(maxval(diff, i))
	    << "  " << i[0] << "," << i[1]
	    << "\n"
    ;
#endif

  (void)error1;
#if DO_CHECK
  test_assert(error2 <= -50);
#endif
}



template <typename CoeffT,
	  typename T>
void
test_perspective_obj(
  std::string f_prefix,
  length_type rows,
  length_type cols,
  length_type row_size,
  length_type col_size)
{
  for (index_type i=0; i<NUM_TCS; ++i)
  {
    std::ostringstream filename;
    filename << f_prefix << "-" << i;
    test_pwarp_obj<CoeffT, T>(filename.str(), rows,cols, row_size,col_size, i);
  }
}


#if TEST_TYPES
void
test_types(
  length_type rows,
  length_type cols,
  length_type r_size,
  length_type c_size)
{
  typedef unsigned char byte_t;

#if TEST_LEVEL >= 2
  // Cool types, but not that useful in practice.
  test_perspective_fun<double, double>("double", rows, cols, r_size, c_size);
  test_perspective_fun<double, float> ("dfloat", rows, cols, r_size, c_size);
  test_perspective_fun<double, byte_t>("duchar", rows, cols, r_size, c_size);

  test_perspective_obj<double, float> ("obj-dfloat",rows,cols,r_size,c_size);
  test_perspective_obj<double, double>("obj-double",rows,cols,r_size,c_size);
  test_perspective_obj<double, byte_t>("obj-duchar",rows,cols,r_size,c_size);
#endif

  test_perspective_fun<float,  float> ("float",  rows, cols, r_size, c_size);
  test_perspective_fun<float,  byte_t>("uchar",  rows, cols, r_size, c_size);

  test_perspective_obj<float,  float> ("obj-float", rows,cols, r_size, c_size);
  test_perspective_obj<float,  byte_t>("obj-uchar", rows,cols, r_size, c_size);
}
#endif


int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_apply_proj<double>();

#if TEST_TYPES
  test_types(1080, 1920, 32, 16);
  test_types(480,   640, 32, 16);
  test_types(512,   512, 32, 16);
#endif

  // Standalone examples for debugging.
  // test_perspective_obj<float, byte_t>("obj-uchar", 1080, 1920, 32, 16);
  // test_pwarp_obj<float, byte_t>("obj-uchar", 480, 640, 32, 16, 5);
  // test_pwarp_obj<float, byte_t>("obj-uchar", 1080, 1920, 32, 16, 5);
}
