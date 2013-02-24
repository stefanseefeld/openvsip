/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    tests/ukernels/interp.cpp
    @author  Don McCoy
    @date    2008-08-26
    @brief   VSIPL++ Library: User-defined kernel for polar to rectangular
               interpolation for SSAR images.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/ukernel/kernels/host/interp.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;


#define DBG_SHOW_IO      0
#define DBG_SHOW_ERRORS  0


namespace ref
{

template <typename IT,
	  typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3,
	  typename Block4>
void
interpolate(
  const_Matrix<IT, Block1>	   indices,  // n x m
  Tensor<T, Block2>                window,   // n x m x I
  const_Matrix<complex<T>, Block3> in,       // n x m
  Matrix<complex<T>, Block4>       out,      // nx x m
  length_type                      depth)
{
  length_type n = indices.size(0);
  length_type m = indices.size(1);
  length_type nx = out.size(0);
  length_type I = depth; // window.size(2) may include padding
  assert(n == in.size(0));
  assert(m == in.size(1));
  assert(m == out.size(1));
  assert(window.size(0) == n);
  assert(window.size(1) == m);

  out = complex<T>(0);

  for (index_type j = 0; j < m; ++j)
  {
    for (index_type i = 0; i < n; ++i)
    {
      index_type ikxrows = indices.get(i, j);
      index_type i_shift = (i + n/2) % n;
      for (index_type h = 0; h < I; ++h)
      {

        out.put(ikxrows + h, j, out.get(ikxrows + h, j) + 
          (in.get(i_shift, j) * window.get(i, j, h)));
      }
    }
    out.col(j)(Domain<1>(j%2, 2, nx/2)) *= T(-1);
  }
}

} // namespace ref



/***********************************************************************
  Definitions
***********************************************************************/

template <typename T>
void
test_ukernel(length_type rows, length_type cols, length_type depth)
{
  cout << "rows: " << rows << ", cols: " << cols << endl;

  typedef uint32_t I;
  typedef std::complex<T>  C;
  typedef tuple<1, 0, 2> order_type;

  typedef vsip::Layout<2, col2_type, vsip::aligned_128,
			     vsip::impl::dense_complex_format>
		col_layout_type;
  typedef vsip::impl::Strided<2, C, col_layout_type> col_block_type;

  // interp_f.hpp ukernel is hard coded for depth of 17
  test_assert(depth == 17);

  length_type padded_depth = depth;
  if (padded_depth % 4 != 0)
    padded_depth += (4 - (padded_depth % 4));

  Matrix<I, Dense<2, I, order_type> > indices(rows, cols);
  Tensor<T, Dense<3, T, order_type> > window(rows, cols, padded_depth);
  Matrix<C, col_block_type> input(rows, cols);
  // filled with non-zero values to ensure all are overwritten
  Matrix<C, col_block_type> out(rows + depth - 1, cols, C(-4, 4));
  Matrix<C, col_block_type> ref(rows + depth - 1, cols, C(4, -4));

  cout << "  generating data..." << endl;

  // set up input data, weights and indices
  Rand<C> gen(0, 0);
  input = gen.randu(rows, cols);
  Rand<T> gen_real(1, 0);
  for (index_type k = 0; k < depth; ++k)
    window(whole_domain, whole_domain, k) = gen_real.randu(rows, cols);

  // The size of the output is determined by the way the indices are
  // set up.  Here, they are mapped one-to-one, but the output ends up
  // being larger by an amount determined by the depth of the window
  // function used.
  for (index_type i = 0; i < rows; ++i)
    indices.row(i) = i;

  cout << "  computing reference..." << endl;

  // Compute reference output image
  ref::interpolate(indices, window, input, ref, depth);

  cout << "  calling user kernel..." << endl;

  // Compute output image using user-defined kernel.  Data must be 
  // transposed to place it in row-major format.
  ukernel::Interp obj;
  ukernel::Ukernel<ukernel::Interp> interpolate(obj);
  interpolate(
    indices.transpose(), 
    window.template transpose<1, 0, 2>(), 
    input.transpose(), 
    out.transpose() );


  // verify results
#if  DBG_SHOW_IO
  cout << "window = " << endl << window.template transpose<2, 0, 1>() << endl;
  cout << "indices = " << endl << indices << endl;
  cout << "input = " << endl << input << endl;
  cout << "ref = " << endl << ref << endl;
  cout << "out = " << endl << out << endl;
#endif

#if DBG_SHOW_ERRORS
  int err_count = 0;
  for (index_type i = 0; i < out.size(0); ++i)
    for (index_type j = 0; j < out.size(1); ++j)
    {
      if (!equal(out.get(i, j), ref.get(i, j)) && err_count++ < 10)
        cout << "[" << i << ", " << j << "] : " << out.get(i, j) << " != " << ref.get(i, j) 
             << "    " << ref.get(i, j) - out.get(i, j) << endl;
    }
#endif

  test_assert(view_equal(out, ref));
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_ukernel<float>(8, 4, 17);
  test_ukernel<float>(512, 256, 17);
  test_ukernel<float>(1144, 1072, 17);

  return 0;
}
