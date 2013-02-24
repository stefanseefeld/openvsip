/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    convolution.c
    @author  Jules Bergmann, Stefan Seefeld
    @date    2008-06-16
    @brief   VSIPL++ Library: Unit tests for [signal.convolution] items.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip.h>
#include "test.h"
#include "output.h"
#include <assert.h>

#define VERBOSE 1

// The following is defined in vsip/core/signal/conv_common.hpp,
// and not visible here, so we have to redefine it for consistency.
#define VSIP_IMPL_CONV_CORRECT_MIN_SUPPORT_SIZE 1
double const ERROR_THRESH = -70;

/***********************************************************************
  Definitions
***********************************************************************/

vsip_vview_d *ref_kernel_from_coeff(vsip_symmetry symmetry, vsip_vview_d const *coeff)
{
  vsip_length M2 = vsip_vgetlength_d(coeff);
  vsip_length M;

  if (symmetry == VSIP_NONSYM) M = M2;
  else if (symmetry == VSIP_SYM_EVEN_LEN_ODD) M = 2*M2-1;
  else M = 2*M2;

  vsip_vview_d *kernel = vsip_vcreate_d(M, VSIP_MEM_NONE);

  if (symmetry == VSIP_NONSYM)
  {
    vsip_vcopy_d_d(coeff, kernel);
  }
  else if (symmetry == VSIP_SYM_EVEN_LEN_ODD)
  {
    vsip_vview_d *sub = vsip_vsubview_d(kernel, 0, M2);
    vsip_vcopy_d_d(coeff, sub);
    vsip_vdestroy_d(sub);
    sub = vsip_vsubview_d(kernel, M2, M2-1);
    vsip_block_d *block = vsip_vgetblock_d(coeff);
    vsip_vview_d *subcoeff = vsip_vbind_d(block, M2 - 2, -1, M2 - 1);
    vsip_vcopy_d_d(subcoeff, sub);
    vsip_vdestroy_d(subcoeff);
    vsip_vdestroy_d(sub);
  }
  else
  {
    vsip_vview_d *sub = vsip_vsubview_d(kernel, 0, M2);
    vsip_vcopy_d_d(coeff, sub);
    vsip_vdestroy_d(sub);

    sub = vsip_vsubview_d(kernel, M2, M2);
    vsip_block_d *block = vsip_vgetblock_d(coeff);
    vsip_vview_d *subcoeff = vsip_vbind_d(block, M2 - 1, -1, M2);
    vsip_vcopy_d_d(subcoeff, sub);
    vsip_vdestroy_d(sub);
    vsip_vdestroy_d(subcoeff);
  }

  return kernel;
}

vsip_length
ref_conv_output_size(vsip_support_region supp,
                     vsip_length M,
                     vsip_length N,
                     vsip_length D)
{
  if      (supp == VSIP_SUPPORT_FULL) return ((N + M - 2)/D) + 1;
  else if (supp == VSIP_SUPPORT_SAME) return ((N - 1)/D) + 1;
  else
  {
#if VSIP_IMPL_CONV_CORRECT_MIN_SUPPORT_SIZE
    return ((N - M + 1) / D) + ((N - M + 1) % D == 0 ? 0 : 1);
#else
    return ((N - 1)/D) - ((M-1)/D) + 1;
#endif
  }
}

vsip_stride
ref_conv_expected_shift(vsip_support_region supp, vsip_length M)
{
  if (supp == VSIP_SUPPORT_FULL) return 0;
  else if (supp == VSIP_SUPPORT_SAME) return (M/2);
  else return (M-1);
}

void
ref_conv_d(vsip_symmetry sym,
           vsip_support_region sup,
           vsip_vview_d const *coeff,
           vsip_vview_d const *in,
           vsip_vview_d *out,
           vsip_length D)
{
  vsip_vview_d *kernel = ref_kernel_from_coeff(sym, coeff);

  vsip_length M = vsip_vgetlength_d(kernel);
  vsip_length N = vsip_vgetlength_d(in);
  vsip_length P = vsip_vgetlength_d(out);

  vsip_length expected_P = ref_conv_output_size(sup, M, N, D);
  vsip_stride shift      = ref_conv_expected_shift(sup, M);


  assert(expected_P == P);

  vsip_vview_d *sub = vsip_vcreate_d(M, VSIP_MEM_NONE);

  // Check result
  vsip_index i;
  for (i=0; i<P; ++i)
  {
    vsip_vfill_d(0, sub);
    vsip_index pos = i*D + shift;

    if (pos+1 < M)
    {
      vsip_vview_d *subsub = vsip_vsubview_d(sub, 0, pos + 1);
      vsip_block_d *block = vsip_vgetblock_d(in);
      vsip_vview_d *insub = vsip_vcloneview_d(in);
      vsip_vattr_d attr;
      vsip_vgetattrib_d(in, &attr);
      attr.offset += pos * attr.stride;
      attr.stride *= -1;
      attr.length = pos + 1;
      vsip_vputattrib_d(insub, &attr);
      vsip_vcopy_d_d(insub, subsub);
      vsip_vdestroy_d(subsub);
      vsip_vdestroy_d(insub);
    }
    else if (pos >= N)
    {
      vsip_index start = pos - N + 1;
      vsip_vview_d *subsub = vsip_vsubview_d(sub, start, M - start);
      vsip_vview_d *insub = vsip_vcloneview_d(in);
      vsip_vattr_d attr;
      vsip_vgetattrib_d(in, &attr);
      attr.offset += (N - 1) * attr.stride;
      attr.stride *= -1;
      attr.length = M - start;
      vsip_vputattrib_d(insub, &attr);
      vsip_vcopy_d_d(insub, subsub);
      vsip_vdestroy_d(subsub);
      vsip_vdestroy_d(insub);
    }
    else
    {
      vsip_vview_d *insub = vsip_vcloneview_d(in);
      vsip_vattr_d attr;
      vsip_vgetattrib_d(in, &attr);
      attr.offset += pos * attr.stride;
      attr.stride *= -1;
      attr.length = M;
      vsip_vputattrib_d(insub, &attr);
      vsip_vcopy_d_d(insub, sub);
      vsip_vdestroy_d(insub);
    }      
    vsip_vput_d(out, i, vsip_vdot_d(kernel, sub));
  }
  vsip_valldestroy_d(kernel);
}

vsip_length
expected_kernel_size(vsip_symmetry symmetry, vsip_length coeff_size)
{
  if (symmetry == VSIP_NONSYM) return coeff_size;
  else if (symmetry == VSIP_SYM_EVEN_LEN_ODD) return 2*coeff_size-1;
  else return 2*coeff_size;
}	     

vsip_length
expected_output_size(vsip_support_region supp,
                     vsip_length M,    // kernel length
                     vsip_length N,    // input  length
                     vsip_length D)    // decimation factor
{
  if (supp == VSIP_SUPPORT_FULL) return ((N + M - 2)/D) + 1;
  else if (supp == VSIP_SUPPORT_SAME) return ((N - 1)/D) + 1;
  else
  {
#if VSIP_IMPL_CONV_CORRECT_MIN_SUPPORT_SIZE
    return ((N - M + 1) / D) + ((N - M + 1) % D == 0 ? 0 : 1);
#else
    return ((N - 1)/D) - ((M-1)/D) + 1;
#endif
  }
}

vsip_length
expected_shift(vsip_support_region supp,
               vsip_length M,
               vsip_length D)
{
  if      (supp == VSIP_SUPPORT_FULL) return 0;
  else if (supp == VSIP_SUPPORT_SAME) return (M/2);
  else return (M-1);
}

/// Test convolution with nonsym symmetry.

void
test_conv_nonsym_d(vsip_support_region support,
                   vsip_length N,
                   vsip_length M,
                   vsip_index c1,
                   vsip_index c2,
                   int k1,
                   int k2)
{
  vsip_symmetry const symmetry = VSIP_NONSYM;

  // length_type const M = 5;				// filter size
  // length_type const N = 100;				// input size
  vsip_length const D = 1;				// decimation
  vsip_length const P = expected_output_size(support, M, N, D);
  // ((N-1)/D) - ((M-1)/D) + 1;	// output size

  int shift = expected_shift(support, M, D);

  vsip_vview_d *coeff = vsip_vcreate_d(M, VSIP_MEM_NONE);
  vsip_vfill_d(0, coeff);
  vsip_vput_d(coeff, c1, k1);
  vsip_vput_d(coeff, c2, k2);

  vsip_conv1d_d *conv = vsip_conv1d_create_d(coeff, symmetry, N, D, support,
                                             0, VSIP_ALG_SPACE);
  vsip_conv1d_attr attr;
  vsip_conv1d_getattr_d(conv, &attr);
  test_assert(attr.symm == symmetry);
  test_assert(attr.support == support);
  test_assert(attr.kernel_len == M);
  test_assert(attr.data_len == N);
  test_assert(attr.out_len == P);
  
  vsip_vview_d *in = vsip_vcreate_d(N, VSIP_MEM_NONE);
  vsip_vview_d *out = vsip_vcreate_d(P, VSIP_MEM_NONE);
  vsip_vfill_d(100, out);
  vsip_vview_d *exp = vsip_vcreate_d(P, VSIP_MEM_NONE);
  vsip_vfill_d(201, exp);

  vsip_index i;
  for (i=0; i<N; ++i)
    vsip_vput_d(in, i, i);

  vsip_convolve1d_d(conv, in, out);

  for (i=0; i<P; ++i)
  {
    double val1, val2;

    if ((int)i + shift - (int)c1 < 0 || i + shift - c1 >= vsip_vgetlength_d(in))
      val1 = 0;
    else
      val1 = vsip_vget_d(in, i + shift - c1);

    if ((int)i + shift - (int)c2 < 0 || i + shift - c2 >= vsip_vgetlength_d(in))
      val2 = 0;
    else
      val2 = vsip_vget_d(in, i + shift - c2);

    vsip_vput_d(exp, i, k1 * val1 + k2 * val2);
  }

  double error = verror_db_d(out, exp);
  test_assert(error < ERROR_THRESH);
}

/// Test general 1-D convolution.

void
test_conv_base_d(vsip_symmetry symmetry,
                 vsip_support_region support,
                 vsip_vview_d const *in,
                 vsip_vview_d const *out,
                 vsip_vview_d const *coeff,
                 vsip_length D,
                 vsip_length n_loop)
{
  vsip_length M = expected_kernel_size(symmetry, vsip_vgetlength_d(coeff));
  vsip_length N = vsip_vgetlength_d(in);
  vsip_length P = vsip_vgetlength_d(out);
  vsip_vview_d *tmp = vsip_vcreate_d(P, VSIP_MEM_NONE);

  vsip_length expected_P = expected_output_size(support, M, N, D);

  test_assert(P == expected_P);

  vsip_conv1d_d *conv = vsip_conv1d_create_d(coeff, symmetry, N, D, support,
                                             0, VSIP_ALG_SPACE);
  vsip_conv1d_attr attr;
  vsip_conv1d_getattr_d(conv, &attr);

  test_assert(attr.symm == symmetry);
  test_assert(attr.support  == support);
  test_assert(attr.kernel_len == M);
  test_assert(attr.data_len == N);
  test_assert(attr.out_len == P);

  vsip_vview_d *exp = vsip_vcreate_d(P, VSIP_MEM_NONE);

  vsip_index loop;
  for (loop=0; loop<n_loop; ++loop)
  {
    vsip_index i;
    for (i=0; i<N; ++i)
      vsip_vput_d(in, i, 3*loop+i);

    vsip_convolve1d_d(conv, in, out);

    ref_conv_d(symmetry, support, coeff, in, exp, D);

    // Check result
    vsip_scalar_vi idx;
    double error = verror_db_d(out, exp);
    
    vsip_vsub_d(out, exp, tmp);
    vsip_vsq_d(tmp, tmp);
    double maxdiff = vsip_vmaxval_d(tmp, &idx);

#if VERBOSE
    if (error > ERROR_THRESH)
    {
      printf("exp :\n");
      voutput_d(exp);
      printf("out :\n");
      voutput_d(out);
      printf("diff :\n");
      voutput_d(tmp);
    }
#endif

    test_assert(error < ERROR_THRESH || maxdiff < 1e-4);
  }
}



/// Test convolution for non-unit strides.

void
test_conv_nonunit_stride_d(vsip_support_region support,
                           vsip_length N,
                           vsip_length M,
                           vsip_stride stride)
{
  vsip_symmetry const symmetry = VSIP_NONSYM;
  vsip_length const D = 1; // decimation

  vsip_length const P = expected_output_size(support, M, N, D);

  vsip_vview_d *kernel = vsip_vcreate_d(M, VSIP_MEM_NONE);
  vsip_randstate *rgen = vsip_randcreate(0, 1, 1, VSIP_PRNG);
  vsip_vrandu_d(rgen, kernel);

  vsip_vview_d *in_base = vsip_vcreate_d(N * stride, VSIP_MEM_NONE);
  vsip_vview_d *out_base = vsip_vcreate_d(P * stride, VSIP_MEM_NONE);
  vsip_vfill_d(100, out_base);
  vsip_block_d *block = vsip_vgetblock_d(in_base);
  vsip_vview_d *in = vsip_vbind_d(block, 0, stride, N);

  block = vsip_vgetblock_d(out_base);
  vsip_vview_d *out = vsip_vbind_d(block, 0, stride, P);
  test_conv_base_d(symmetry, support, in, out, kernel, D, 1);
  vsip_vdestroy_d(out);
  vsip_vdestroy_d(in);
}



/// Test general 1-D convolution.

void
test_conv_d(vsip_symmetry symmetry, vsip_support_region support,
            vsip_length N,		// input size
            vsip_length D,		// decimation
            vsip_vview_d const *coeff,
            int n_loop)
{
  vsip_length M = expected_kernel_size(symmetry, vsip_vgetlength_d(coeff));
  vsip_length P = expected_output_size(support, M, N, D);

  vsip_vview_d *in = vsip_vcreate_d(N, VSIP_MEM_NONE);
  vsip_vview_d *out = vsip_vcreate_d(P, VSIP_MEM_NONE);
  vsip_vfill_d(100, out);
  test_conv_base_d(symmetry, support, in, out, coeff, D, n_loop);
  vsip_valldestroy_d(out);
  vsip_valldestroy_d(in);
}

// Run a set of convolutions for given type and size
//   (with symmetry = nonsym and decimation = 1).

void
cases_nonsym_d(vsip_length size)
{
  test_conv_nonsym_d(VSIP_SUPPORT_MIN, size, 4, 0, 1, +1, +1);
  test_conv_nonsym_d(VSIP_SUPPORT_MIN, size, 5, 0, 1, +1, -1);

  test_conv_nonsym_d(VSIP_SUPPORT_SAME, size, 4, 0, 1, +1, +1);
  test_conv_nonsym_d(VSIP_SUPPORT_SAME, size, 5, 0, 1, +1, -1);

  test_conv_nonsym_d(VSIP_SUPPORT_FULL, size, 4, 0, 1, +1, +1);
  test_conv_nonsym_d(VSIP_SUPPORT_FULL, size, 5, 0, 1, +1, -1);
}

// Run a set of convolutions for given type and size
//   (using vectors with strides other than one).

void
cases_nonunit_stride_d(vsip_length size)
{
  test_conv_nonunit_stride_d(VSIP_SUPPORT_MIN, size, 4, 3);
  test_conv_nonunit_stride_d(VSIP_SUPPORT_MIN, size, 5, 2);

  test_conv_nonunit_stride_d(VSIP_SUPPORT_FULL, size, 4, 3);
  test_conv_nonunit_stride_d(VSIP_SUPPORT_FULL, size, 5, 2);

  test_conv_nonunit_stride_d(VSIP_SUPPORT_SAME, size, 4, 3);
  test_conv_nonunit_stride_d(VSIP_SUPPORT_SAME, size, 5, 2);
}



// Run a set of convolutions for given type, symmetry, input size, coeff size
// and decmiation.

void
cases_conv_d(vsip_symmetry symmetry, vsip_length size, vsip_length M, vsip_length D, vsip_bool rand)
{
  vsip_vview_d *coeff = vsip_vcreate_d(M, VSIP_MEM_NONE);

  if (rand)
  {
    vsip_randstate *rgen = vsip_randcreate(0, 1, 1, VSIP_PRNG);
    vsip_vrandu_d(rgen, coeff);
    vsip_randdestroy(rgen);
  }
  else
  {
    vsip_vfill_d(0, coeff);
    vsip_vput_d(coeff, 0, -1);
    vsip_vput_d(coeff, M-1, 2);
  }

  test_conv_d(symmetry, VSIP_SUPPORT_MIN, size, D, coeff, 2);
  test_conv_d(symmetry, VSIP_SUPPORT_SAME, size, D, coeff, 2);
  test_conv_d(symmetry, VSIP_SUPPORT_FULL, size, D, coeff, 2);
}

// Run a single convolutions for given type, symmetry, support, input
// size, coeff size and decmiation.

void
single_conv_d(vsip_symmetry symmetry, vsip_support_region support,
              vsip_length size, vsip_length M, vsip_length D,
              int n_loop, vsip_bool rand)
{
  vsip_vview_d *coeff = vsip_vcreate_d(M, VSIP_MEM_NONE);
  if (rand)
  {
    vsip_randstate *rgen = vsip_randcreate(0, 1, 1, VSIP_PRNG);
    vsip_vrandu_d(rgen, coeff);
    vsip_randdestroy(rgen);
  }
  else
  {
    vsip_vfill_d(0, coeff);
    vsip_vput_d(coeff, 0, -1);
    vsip_vput_d(coeff, M-1, 2);
  }

  test_conv_d(symmetry, support, size, D, coeff, n_loop);
}



void
cases_d(vsip_bool rand)
{
  // check that M == N works
  cases_conv_d(VSIP_NONSYM, 8, 8, 1, rand);
  cases_conv_d(VSIP_NONSYM, 5, 5, 1, rand);
  cases_conv_d(VSIP_SYM_EVEN_LEN_EVEN, 8, 4, 1, rand);
  cases_conv_d(VSIP_SYM_EVEN_LEN_ODD, 7, 4, 1, rand);

  cases_conv_d(VSIP_NONSYM, 5, 4, 1, rand);
  cases_conv_d(VSIP_NONSYM, 5, 4, 2, rand);
  cases_conv_d(VSIP_NONSYM, 5, 4, 3, rand);
  cases_conv_d(VSIP_NONSYM, 5, 4, 4, rand);

  cases_nonsym_d(100);

  vsip_length size;
  for (size=32; size<=1024; size *= 4)
  {
    cases_nonsym_d(size);
    cases_nonsym_d(size+3);
    cases_nonsym_d(2*size);

    cases_nonunit_stride_d(size);

    cases_conv_d(VSIP_NONSYM, size,      8,  1, rand);
    cases_conv_d(VSIP_NONSYM, 2*size,    7,  2, rand);
    cases_conv_d(VSIP_NONSYM, size+4,    6,  3, rand);

    cases_conv_d(VSIP_SYM_EVEN_LEN_EVEN, size,   5,  1, rand);
    cases_conv_d(VSIP_SYM_EVEN_LEN_EVEN, size+1, 6,  2, rand);

    cases_conv_d(VSIP_SYM_EVEN_LEN_ODD, size,   4,  1, rand);
    cases_conv_d(VSIP_SYM_EVEN_LEN_ODD, size+3, 3,  2, rand);
  }
}

int
main(int argc, char** argv)
{
  vsip_init(0);
  cases_d(1);
  return 0;
}
