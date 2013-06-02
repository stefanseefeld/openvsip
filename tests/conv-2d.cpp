//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/vector.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip/initfin.hpp>
#include <test.hpp>

#define VERBOSE 0

using namespace ovxx;

length_type expected_output_size(
  support_region_type supp,
  length_type         M,    // kernel length
  length_type         N,    // input  length
  length_type         D)    // decimation factor
{
  if      (supp == support_full)
    return ((N + M - 2)/D) + 1;
  else if (supp == support_same)
    return ((N - 1)/D) + 1;
  else //(supp == support_min)
  {
#if VSIP_IMPL_CONV_CORRECT_MIN_SUPPORT_SIZE
    return ((N - M + 1) / D) + ((N - M + 1) % D == 0 ? 0 : 1);
#else
    return ((N - 1)/D) - ((M-1)/D) + 1;
#endif
  }
}



length_type expected_shift(
  support_region_type supp,
  length_type         M,     // kernel length
  length_type         /*D*/) // decimation factor
{
  if      (supp == support_full)
    return 0;
  else if (supp == support_same)
    return (M/2);
  else //(supp == support_min)
    return (M-1);
}


template <typename T,
	  typename Block>
void
init_in(Matrix<T, Block> in, int k)
{
  for (index_type i=0; i<in.size(0); ++i)
    for (index_type j=0; j<in.size(1); ++j)
      in(i, j) = T(i*in.size(0)+j+k);
}



/// Test general 2-D convolution.

template <typename            T,
	  symmetry_type       symmetry,
	  support_region_type support,
	  dimension_type      Dim,
	  typename            T1,
	  typename            Block1>
void
test_conv(
  Domain<Dim> const&       dom_in,
  length_type              D,		// decimation
  const_Matrix<T1, Block1> coeff,	// coefficients
  length_type const        n_loop = 2,
  stride_type const        stride = 1)
{
  length_type Nr = dom_in[0].size();	// input rows
  length_type Nc = dom_in[1].size();	// input cols

  typedef Convolution<const_Matrix, symmetry, support, T> conv_type;
  typedef typename Matrix<T>::subview_type matrix_subview_type;

  length_type M2r = coeff.size(0);
  length_type M2c = coeff.size(1);
  length_type Mr, Mc;

  if (symmetry == nonsym)
  {
    Mr = coeff.size(0);
    Mc = coeff.size(1);
  }
  else if (symmetry == sym_even_len_odd)
  {
    Mr = 2*coeff.size(0)-1;
    Mc = 2*coeff.size(1)-1;
  }
  else /* (symmetry == sym_even_len_even) */
  {
    Mr = 2*coeff.size(0);
    Mc = 2*coeff.size(1);
  }

  length_type const Pr = expected_output_size(support, Mr, Nr, D);
  length_type const Pc = expected_output_size(support, Mc, Nc, D);

  int shift_r = expected_shift(support, Mr, D);
  int shift_c = expected_shift(support, Mc, D);

  Matrix<T> kernel(Mr, Mc, T());

  // Apply symmetry (if any) to get kernel form coefficients.
  if (symmetry == nonsym)
  {
    kernel = coeff;
  }
  else if (symmetry == sym_even_len_odd)
  {
    kernel(Domain<2>(M2r, M2c))   = coeff;
    kernel(Domain<2>(M2r, Domain<1>(M2c, 1, M2c-1))) =
      coeff(Domain<2>(M2r, Domain<1>(M2c-2, -1, M2c-1)));

    kernel(Domain<2>(Domain<1>(M2r, 1, M2r-1), 2*M2c-1)) = 
      kernel(Domain<2>(Domain<1>(M2r-2, -1, M2r-1), 2*M2c-1));
  }
  else /* (symmetry == sym_even_len_even) */
  {
    kernel(Domain<2>(M2r, M2c))   = coeff;
    kernel(Domain<2>(M2r, Domain<1>(M2c, 1, M2c))) =
      coeff(Domain<2>(M2r, Domain<1>(M2c-1, -1, M2c)));
    kernel(Domain<2>(Domain<1>(M2r, 1, M2r), 2*M2c)) = 
      kernel(Domain<2>(Domain<1>(M2r-1, -1, M2r), 2*M2c));
  }


  conv_type conv(coeff, Domain<2>(Nr, Nc), D);

  test_assert(conv.symmetry() == symmetry);
  test_assert(conv.support()  == support);

  test_assert(conv.kernel_size()[0].size()  == Mr);
  test_assert(conv.kernel_size()[1].size()  == Mc);

  test_assert(conv.filter_order()[0].size() == Mr);
  test_assert(conv.filter_order()[1].size() == Mc);

  test_assert(conv.input_size()[0].size()   == Nr);
  test_assert(conv.input_size()[1].size()   == Nc);

  test_assert(conv.output_size()[0].size()  == Pr);
  test_assert(conv.output_size()[1].size()  == Pc);

  Matrix<T> in_base(Nr * stride, Nc * stride);
  Matrix<T> out_base(Pr * stride, Pc * stride, T(100));
  Matrix<T> ex(Pr, Pc, T(101));

  matrix_subview_type in  = in_base (Domain<2>(
				       Domain<1>(0, stride, Nr),
				       Domain<1>(0, stride, Nc)));
  matrix_subview_type out = out_base(Domain<2>(
				       Domain<1>(0, stride, Pr),
				       Domain<1>(0, stride, Pc)));

  Matrix<T> sub(Mr, Mc);

  for (index_type loop=0; loop<n_loop; ++loop)
  {
    init_in(in, 3*loop);

    conv(in, out);

    // Check result
    bool good = true;
    for (index_type i=0; i<Pr; ++i)
      for (index_type j=0; j<Pc; ++j)
      {
	sub = T();
	index_type ii = i*D + shift_r;
	index_type jj = j*D + shift_c;

	Domain<1> sub_d0, sub_d1;
	Domain<1> rhs_d0, rhs_d1;

	// Determine rows to copy
	if (ii+1 < Mr)
	{
	  sub_d0 = Domain<1>(0, 1, ii+1);
	  rhs_d0 = Domain<1>(ii, -1, ii+1);
	}
	else if (ii >= Nr)
	{
	  index_type start = ii - Nr + 1;
	  sub_d0 = Domain<1>(start, 1, Mr-start);
	  rhs_d0 = Domain<1>(Nr-1, -1, Mr-start);
	}
	else
	{
	  sub_d0 = Domain<1>(0, 1, Mr);
	  rhs_d0 = Domain<1>(ii, -1, Mr);
	}

	// Determine cols to copy
	if (jj+1 < Mc)
	{
	  sub_d1 = Domain<1>(0, 1, jj+1);
	  rhs_d1 = Domain<1>(jj, -1, jj+1);
	}
	else if (jj >= Nc)
	{
	  index_type start = jj - Nc + 1;
	  sub_d1 = Domain<1>(start, 1, Mc-start);
	  rhs_d1 = Domain<1>(Nc-1, -1, Mc-start);
	}
	else
	{
	  sub_d1 = Domain<1>(0, 1, Mc);
	  rhs_d1 = Domain<1>(jj, -1, Mc);
	}

	sub(Domain<2>(sub_d0, sub_d1)) = in(Domain<2>(rhs_d0, rhs_d1));
	  
	T val = out(i, j);
	T chk = sumval(kernel * sub);

	ex(i, j) = chk;

	// test_assert(equal(val, chk));
	if (!equal(val, chk))
	  good = false;
      }

    if (!good)
    {
#if VERBOSE
      cout << "in = ("       << Nr << ", " << Nc
	   << ")  coeff = (" << Mr << ", " << Mc << ")  D=" << D << endl;
      cout << "out =\n" << out << endl;
      cout << "ex =\n" << ex << endl;
#endif
      test_assert(0);
    }
  }
}



/// Test convolution with nonsym symmetry.

template <typename            T,
	  support_region_type support>
void
test_conv_nonsym(
  length_type Nr,	// input rows
  length_type Nc,	// input cols
  length_type Mr,	// coeff rows
  length_type Mc,	// coeff cols
  index_type  r,
  index_type  c,
  int         k1)
{
  symmetry_type const        symmetry = nonsym;

  typedef Convolution<const_Matrix, symmetry, support, T> conv_type;

  length_type const D = 1;				// decimation

  length_type const Pr = expected_output_size(support, Mr, Nr, D);
  length_type const Pc = expected_output_size(support, Mc, Nc, D);

  int shift_r = expected_shift(support, Mr, D);
  int shift_c = expected_shift(support, Mc, D);

  Matrix<T> coeff(Mr, Mc, T());

  coeff(r, c) = T(k1);

  conv_type conv(coeff, Domain<2>(Nr, Nc), D);

  test_assert(conv.symmetry() == symmetry);
  test_assert(conv.support()  == support);

  test_assert(conv.kernel_size()[0].size()  == Mr);
  test_assert(conv.kernel_size()[1].size()  == Mc);

  test_assert(conv.filter_order()[0].size() == Mr);
  test_assert(conv.filter_order()[1].size() == Mc);

  test_assert(conv.input_size()[0].size()   == Nr);
  test_assert(conv.input_size()[1].size()   == Nc);

  test_assert(conv.output_size()[0].size()  == Pr);
  test_assert(conv.output_size()[1].size()  == Pc);


  Matrix<T> in(Nr, Nc);
  Matrix<T> out(Pr, Pc, T(100));
  Matrix<T> ex(Pr, Pc, T(100));

  init_in(in, 0);

  conv(in, out);

  bool good = true;
  for (index_type i=0; i<Pr; ++i)
  {
    for (index_type j=0; j<Pc; ++j)
    {
      T val;

      if ((int)i + shift_r - (int)r < 0 || i + shift_r - r >= Nr ||
	  (int)j + shift_c - (int)c < 0 || j + shift_c - c >= Nc)
	val = T();
      else
	val = in(i + shift_r - r, j + shift_c - c);

      ex(i, j) = T(k1) * val;
      if (!equal(out(i, j), ex(i, j)))
	good = false;
    }
  }

  if (!good)
  {
#if VERBOSE
    cout << "in =\n" << in << endl;
    cout << "coeff =\n" << coeff << endl;
    cout << "out =\n" << out << endl;
    cout << "ex =\n" << ex << endl;
#endif
    test_assert(0);
  }
}



// Run a set of convolutions for given type and size
//   (with symmetry = nonsym and decimation = 1).

template <typename T>
void
cases_nonsym(
  length_type i_r,	// input rows
  length_type i_c,	// input cols
  length_type k_r,	// kernel rows
  length_type k_c)	// kernel cols
{
  test_conv_nonsym<T, support_min>(i_r, i_c, k_r, k_c,     0,     0, +1);
  test_conv_nonsym<T, support_min>(i_r, i_c, k_r, k_c, k_r/2, k_c/2, -2);
  test_conv_nonsym<T, support_min>(i_r, i_c, k_r, k_c,     0, k_c-1, +3);
  test_conv_nonsym<T, support_min>(i_r, i_c, k_r, k_c, k_r-1,     0, -4);
  test_conv_nonsym<T, support_min>(i_r, i_c, k_r, k_c, k_r-1, k_c-1, +5);

  test_conv_nonsym<T, support_same>(i_r, i_c, k_r, k_c,     0,     0, +1);
  test_conv_nonsym<T, support_same>(i_r, i_c, k_r, k_c, k_r/2, k_c/2, -2);
  test_conv_nonsym<T, support_same>(i_r, i_c, k_r, k_c,     0, k_c-1, +3);
  test_conv_nonsym<T, support_same>(i_r, i_c, k_r, k_c, k_r-1,     0, -4);
  test_conv_nonsym<T, support_same>(i_r, i_c, k_r, k_c, k_r-1, k_c-1, +5);

  test_conv_nonsym<T, support_full>(i_r, i_c, k_r, k_c,     0,     0, +1);
  test_conv_nonsym<T, support_full>(i_r, i_c, k_r, k_c, k_r/2, k_c/2, -2);
  test_conv_nonsym<T, support_full>(i_r, i_c, k_r, k_c,     0, k_c-1, +3);
  test_conv_nonsym<T, support_full>(i_r, i_c, k_r, k_c, k_r-1,     0, -4);
  test_conv_nonsym<T, support_full>(i_r, i_c, k_r, k_c, k_r-1, k_c-1, +5);
}



// Run a set of convolutions for given type and size
//   (using vectors with strides other than one).

template <typename       T,
	  dimension_type Dim>
void
cases_nonunit_stride(Domain<Dim> const& size)
{
  length_type const n_loop = 2;
  length_type const D      = 1;

  Rand<T> rgen(0);
  Matrix<T> coeff33(3, 3, T()); coeff33 = rgen.randu(3, 3);
  Matrix<T> coeff23(2, 3, T()); coeff23 = rgen.randu(2, 3);
  Matrix<T> coeff32(3, 2, T()); coeff32 = rgen.randu(3, 2);

  test_conv<T, nonsym, support_min>(size, D, coeff33, n_loop, 3);
  test_conv<T, nonsym, support_min>(size, D, coeff33, n_loop, 2);

  test_conv<T, nonsym, support_full>(size, D, coeff32, n_loop, 3);
  test_conv<T, nonsym, support_full>(size, D, coeff23, n_loop, 2);

  test_conv<T, nonsym, support_same>(size, D, coeff23, n_loop, 3);
  test_conv<T, nonsym, support_same>(size, D, coeff32, n_loop, 2);
}



// Run a set of convolutions for given type, symmetry, input size, coeff size
// and decmiation.

template <typename       T,
	  symmetry_type  Sym,
	  dimension_type Dim>
void
cases_conv(
  Domain<Dim> const& size,
  Domain<Dim> const& M,
  length_type        D,
  bool               rand)
{
  typename vsip::impl::view_of<Dense<Dim, T> >::type
		coeff(M[0].size(), M[1].size(), T());

  if (rand)
  {
    Rand<T> rgen(0);
    coeff = rgen.randu(M[0].size(), M[1].size());
  }
  else
  {
    coeff(0, 0)                         = T(-1);
    coeff(M[0].size()-1, M[1].size()-1) = T(2);
  }

  test_conv<T, Sym, support_min> (size, D, coeff);
  test_conv<T, Sym, support_same>(size, D, coeff);
  test_conv<T, Sym, support_full>(size, D, coeff);
}



// Run a single convolutions for given type, symmetry, support, input
// size, coeff size and decmiation.

template <typename            T,
	  symmetry_type       Sym,
	  support_region_type Sup,
	  dimension_type      Dim>
void
single_conv(
  Domain<Dim> const& size, 
  Domain<Dim> const& M,
  length_type        D,
  length_type        n_loop,
  bool               rand)
{
  typename vsip::impl::view_of<Dense<Dim, T> >::type
		coeff(M[0].size(), M[1].size(), T());

  if (rand)
  {
    Rand<T> rgen(0);
    coeff = rgen.randu(M[0].size(), M[1].size());
  }
  else
  {
    coeff(0, 0)                         = T(-1);
    coeff(M[0].size()-1, M[1].size()-1) = T(2);
  }

  test_conv<T, Sym, Sup>(size, D, coeff, n_loop);
}



template <typename T>
void
cases(bool rand)
{
  // check that M == N works
  cases_conv<T, nonsym>(Domain<2>(8, 8), Domain<2>(8, 8), 1, rand);
  cases_conv<T, nonsym>(Domain<2>(5, 5), Domain<2>(5, 5), 1, rand);
  cases_conv<T, sym_even_len_even>(Domain<2>(8, 8), Domain<2>(4, 4), 1, rand);
  cases_conv<T, sym_even_len_odd> (Domain<2>(7, 7), Domain<2>(4, 4), 1, rand);

  cases_conv<T, nonsym>(Domain<2>(5, 5), Domain<2>(4, 4), 1, rand);
  cases_conv<T, nonsym>(Domain<2>(5, 5), Domain<2>(4, 4), 2, rand);
  cases_conv<T, nonsym>(Domain<2>(5, 5), Domain<2>(4, 4), 3, rand);
  cases_conv<T, nonsym>(Domain<2>(5, 5), Domain<2>(4, 4), 4, rand);

  for (length_type size=8; size<=256; size *= 8)
  {
    cases_nonsym<T>(size,     size, 3, 3);
    cases_nonsym<T>(size+3, size-1, 3, 2);

    cases_nonunit_stride<T>(Domain<2>(size, size));

    cases_conv<T, nonsym>(Domain<2>(size,   size),   Domain<2>(3, 3), 1, rand);
    cases_conv<T, nonsym>(Domain<2>(2*size, size-1), Domain<2>(4, 3), 2, rand);
  }

  length_type fixed_size = 64;

  cases_conv<T, sym_even_len_even>(Domain<2>(fixed_size, fixed_size),
				   Domain<2>(2, 3), 1, rand);
  cases_conv<T, sym_even_len_even>(Domain<2>(fixed_size-1, fixed_size+2),
				   Domain<2>(3, 2), 2, rand);

  cases_conv<T, sym_even_len_odd>(Domain<2>(fixed_size, fixed_size),
				  Domain<2>(2, 3), 3, rand);
  cases_conv<T, sym_even_len_odd>(Domain<2>(fixed_size+3, fixed_size-2),
				  Domain<2>(3, 2), 4, rand);
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0
  // small sets of tests, covered by 'cases()' below
  cases_nonsym<int>(8, 8, 3, 3);
  cases_nonsym<float>(8, 8, 3, 3);
  cases_nonsym<complex<float> >(8, 8, 3, 3);

  // individual tests, covered by 'cases()' below
  test_conv_nonsym<int, support_min>(8, 8, 3, 3, 1, 1, 1);
  test_conv_nonsym<int, support_min>(8, 8, 3, 3, 0, 2, -1);
  test_conv_nonsym<int, support_min>(8, 8, 3, 3, 2, 0, 2);
  test_conv_nonsym<int, support_min>(8, 8, 3, 3, 2, 2, -2);

  test_conv_nonsym<int, support_same>(8, 8, 3, 3, 1, 1, 1);
  test_conv_nonsym<int, support_same>(8, 8, 3, 3, 0, 2, -1);
  test_conv_nonsym<int, support_same>(8, 8, 3, 3, 2, 0, 2);

  test_conv_nonsym<int, support_full>(8, 8, 3, 3, 1, 1, 1);
  test_conv_nonsym<int, support_full>(8, 8, 3, 3, 0, 0, 2);
  test_conv_nonsym<int, support_full>(8, 8, 3, 3, 2, 2, -1);
#endif

#if VSIP_IMPL_TEST_LEVEL >= 1
  // General tests.
  bool rand = true;
  cases<short>(rand);
  cases<int>(rand);
  cases<float>(rand);
  // cases<complex<int> >(rand);
  cases<complex<float> >(rand);

  cases_nonsym<complex<int> >(8, 8, 3, 3);
  cases_nonsym<complex<float> >(8, 8, 3, 3);

#  if VSIP_IMPL_TEST_DOUBLE
  cases<double>(rand);
  // cases<complex<double> >(rand);

  cases_nonsym<complex<double> >(8, 8, 3, 3);
#  endif // VSIP_IMPL_TEST_DOUBLE
#endif

}
