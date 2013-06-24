//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef test_ref_corr_hpp_
#define test_ref_corr_hpp_

#include <vsip/vector.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip/selgen.hpp>
#include <test/ref/matvec.hpp>

namespace test
{
namespace ref
{

vsip::length_type
corr_output_size(vsip::support_region_type supp,
		 vsip::length_type         M,    // kernel length
		 vsip::length_type         N)    // input  length
{
  if      (supp == vsip::support_full)
    return (N + M - 1);
  else if (supp == vsip::support_same)
    return N;
  else //(supp == vsip::support_min)
    return (N - M + 1);
}



vsip::stride_type
expected_shift(vsip::support_region_type supp,
	       vsip::length_type         M)     // kernel length
{
  if      (supp == vsip::support_full)
    return -(M-1);
  else if (supp == vsip::support_same)
    return -(M/2);
  else //(supp == vsip::support_min)
    return 0;
}



template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
corr(vsip::bias_type               bias,
     vsip::support_region_type     sup,
     vsip::const_Vector<T, Block1> ref,
     vsip::const_Vector<T, Block2> in,
     vsip::Vector<T, Block3>       out)
{
  using vsip::index_type;
  using vsip::length_type;
  using vsip::stride_type;
  using vsip::Vector;
  using vsip::Domain;
  using vsip::unbiased;

  typedef typename scalar_of<T>::type scalar_type;

  length_type M = ref.size(0);
  length_type N = in.size(0);
  length_type P = out.size(0);

  stride_type shift      = expected_shift(sup, M);

  // expected_P = corr_output_size(sup, M, N)
  assert(corr_output_size(sup, M, N) == P);

  Vector<T> sub(M);

  // compute correlation
  for (index_type i=0; i<P; ++i)
  {
    sub = T();
    stride_type pos = static_cast<stride_type>(i) + shift;
    scalar_type scale;

    if (pos < 0)
    {
      sub(Domain<1>(-pos, 1, M + pos)) = in(Domain<1>(0, 1, M+pos));
      scale = scalar_type(M + pos);
    }
    else if (pos + M > N)
    {
      sub(Domain<1>(0, 1, N-pos)) = in(Domain<1>(pos, 1, N-pos));
      scale = scalar_type(N - pos);
    }
    else
    {
      sub = in(Domain<1>(pos, 1, M));
      scale = scalar_type(M);
    }

#if VSIP_IMPL_CORR_CORRECT_SAME_SUPPORT_SCALING
#else
    if (sup == vsip::support_same)
    {
      if      (i < (M/2))     scale = i + (M+1)/2;         // i + ceil(M/2)
      else if (i < N - (M/2)) scale = M;                   // M
      else                    scale = N - 1 + (M+1)/2 - i; // N-1+ceil(M/2)-i
    }
#endif
      
    T val = ref::dot(ref, impl_conj(sub));
    if (bias == vsip::unbiased)
      val /= scale;

    out(i) = val;
  }
}



template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
corr(vsip::bias_type               bias,
     vsip::support_region_type     sup,
     vsip::const_Matrix<T, Block1> ref,
     vsip::const_Matrix<T, Block2> in,
     vsip::Matrix<T, Block3>       out)
{
  using vsip::index_type;
  using vsip::length_type;
  using vsip::stride_type;
  using vsip::Matrix;
  using vsip::Domain;
  using vsip::unbiased;

  typedef typename scalar_of<T>::type scalar_type;

  length_type Mr = ref.size(0);
  length_type Mc = ref.size(1);
  length_type Nr = in.size(0);
  length_type Nc = in.size(1);
  length_type Pr = out.size(0);
  length_type Pc = out.size(1);

  stride_type shift_r     = expected_shift(sup, Mr);
  stride_type shift_c     = expected_shift(sup, Mc);

  // expected_Pr = corr_output_size(sup, Mr, Nr)
  // expected_Pc = corr_output_size(sup, Mc, Nc)
  assert(corr_output_size(sup, Mr, Nr) == Pr);
  assert(corr_output_size(sup, Mc, Nc) == Pc);

  Matrix<T> sub(Mr, Mc);
  Domain<1> sub_dom_r;
  Domain<1> sub_dom_c;
  Domain<1> in_dom_r;
  Domain<1> in_dom_c;

  // compute correlation
  for (index_type r=0; r<Pr; ++r)
  {
    stride_type pos_r = static_cast<stride_type>(r) + shift_r;

    for (index_type c=0; c<Pc; ++c)
    {

      stride_type pos_c = static_cast<stride_type>(c) + shift_c;

      scalar_type scale = scalar_type(1);

      if (pos_r < 0)
      {
	sub_dom_r = Domain<1>(-pos_r, 1, Mr + pos_r); 
	in_dom_r  = Domain<1>(0, 1, Mr+pos_r);
	scale *= scalar_type(Mr + pos_r);
      }
      else if (pos_r + Mr > Nr)
      {
	sub_dom_r = Domain<1>(0, 1, Nr-pos_r);
	in_dom_r  = Domain<1>(pos_r, 1, Nr-pos_r);
	scale *= scalar_type(Nr - pos_r);
      }
      else
      {
	sub_dom_r = Domain<1>(0, 1, Mr);
	in_dom_r  = Domain<1>(pos_r, 1, Mr);
	scale *= scalar_type(Mr);
      }

      if (pos_c < 0)
      {
	sub_dom_c = Domain<1>(-pos_c, 1, Mc + pos_c); 
	in_dom_c  = Domain<1>(0, 1, Mc+pos_c);
	scale *= scalar_type(Mc + pos_c);
      }
      else if (pos_c + Mc > Nc)
      {
	sub_dom_c = Domain<1>(0, 1, Nc-pos_c);
	in_dom_c  = Domain<1>(pos_c, 1, Nc-pos_c);
	scale *= scalar_type(Nc - pos_c);
      }
      else
      {
	sub_dom_c = Domain<1>(0, 1, Mc);
	in_dom_c  = Domain<1>(pos_c, 1, Mc);
	scale *= scalar_type(Mc);
      }

      sub = T();
      sub(Domain<2>(sub_dom_r, sub_dom_c)) = in(Domain<2>(in_dom_r, in_dom_c));
      
      T val = sumval(ref * impl_conj(sub));
      if (bias == unbiased)
	val /= scale;
      
      out(r, c) = val;
    }
  }
}

} // namespace test::ref
} // namespace test

#endif
