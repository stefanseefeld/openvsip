/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/sal/is_op_supported.hpp
    @author  Jules Bergmann
    @date    2006-10-26
    @brief   VSIPL++ Library: Mercury SAL ops supported for dispatch.
*/

#ifndef VSIP_OPT_SAL_IS_OP_SUPPORTED_HPP
#define VSIP_OPT_SAL_IS_OP_SUPPORTED_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/expr/fns_elementwise.hpp>
#include <vsip/core/view_cast.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace sal
{

/// Traits class to help determine when SAL supports a given
/// binary operation.
///
/// Requirements:
///   OPERATOR is the Operator template class for the operation
///      (from the vsip::impl::op namespace).
///   LTYPE is the type of the left operand.
///   RTYPE is the type of the right operand.
///   RTYPE is the type of the result.
///
/// For LTYPE, RTYPE, and DSTTYPE, vector operands should be represented
/// by a pointer type (for example: float*, complex<float>*, and
/// std::pair<float*, float*>).  scalar operand should be represented
/// by a value type, such as float, complex<float>, std::pair<float, float>.

template <template <typename> class Operator,
	  typename                  SrcType,
	  typename                  DstType>
struct Is_op1_supported
{
  static bool const value = false;
};

template <template <typename, typename> class Operator,
	  typename                            LType,
	  typename                            RType,
	  typename                            DstType>
struct Is_op2_supported
{
  static bool const value = false;
};

template <template <typename, typename, typename> class Operator,
	  typename                            Type1,
	  typename                            Type2,
	  typename                            Type3,
	  typename                            DstType>
struct Is_op3_supported
{
  static bool const value = false;
};


// Tokens for ops not mapping directly to functors.

template <typename> struct copy_token;

template <typename, typename> struct veq_token;
template <typename, typename> struct vne_token;
template <typename, typename> struct vgt_token;
template <typename, typename> struct vge_token;
template <typename, typename> struct vlt_token;
template <typename, typename> struct vle_token;

template <typename, typename, typename> struct cma_token;


#define VSIP_IMPL_OP1SUP(Op, T1, DT)					\
  template <> struct Is_op1_supported<Op, T1, DT >			\
  { static bool const value = true; }

#define VSIP_IMPL_OP2SUP(Op, LT, RT, DT)				\
  template <> struct Is_op2_supported<Op, LT, RT, DT >			\
  { static bool const value = true; }

#define VSIP_IMPL_OP3SUP(Op, T1, T2, T3, DT)				\
  template <> struct Is_op3_supported<Op, T1, T2, T3, DT >		\
  { static bool const value = true; }

typedef std::pair<float*, float*>   split_float;
typedef std::pair<double*, double*> split_double;


/***********************************************************************
  Unary operators and functions provided by SAL
***********************************************************************/

VSIP_IMPL_OP1SUP(expr::op::Magsq, complex<float>*,  float*);
VSIP_IMPL_OP1SUP(expr::op::Magsq, split_float,      float*);
VSIP_IMPL_OP1SUP(expr::op::Minus,     int*,             int*);
VSIP_IMPL_OP1SUP(expr::op::Minus,     float*,           float*);
VSIP_IMPL_OP1SUP(expr::op::Minus,     complex<float>*,  complex<float>*);
VSIP_IMPL_OP1SUP(expr::op::Minus,     split_float,      split_float);

VSIP_IMPL_OP1SUP(expr::op::Mag,   int*,             int*);
VSIP_IMPL_OP1SUP(expr::op::Mag,   float*,           float*);
VSIP_IMPL_OP1SUP(expr::op::Mag,   complex<float>*,  float*);
VSIP_IMPL_OP1SUP(expr::op::Mag,   split_float,      float*);

VSIP_IMPL_OP1SUP(expr::op::Cos,   float*,           float*);

VSIP_IMPL_OP1SUP(expr::op::Sin,   float*,           float*);

VSIP_IMPL_OP1SUP(expr::op::Tan,   float*,           float*);

VSIP_IMPL_OP1SUP(expr::op::Atan,  float*,           float*);

VSIP_IMPL_OP1SUP(expr::op::Log,   float*,           float*);

VSIP_IMPL_OP1SUP(expr::op::Log10, float*,           float*);

VSIP_IMPL_OP1SUP(expr::op::Exp,   float*,           float*);

VSIP_IMPL_OP1SUP(expr::op::Exp10, float*,           float*);

VSIP_IMPL_OP1SUP(expr::op::Sqrt,  float*,           float*);
VSIP_IMPL_OP1SUP(expr::op::Sqrt,  complex<float>*,  complex<float>*);
VSIP_IMPL_OP1SUP(expr::op::Sqrt,  split_float,      split_float);

VSIP_IMPL_OP1SUP(expr::op::Rsqrt, float*,           float*);

VSIP_IMPL_OP1SUP(expr::op::Sq,    float*,           float*);

VSIP_IMPL_OP1SUP(expr::op::Recip, float*,           float*);
VSIP_IMPL_OP1SUP(expr::op::Recip, complex<float>*,  complex<float>*);
VSIP_IMPL_OP1SUP(expr::op::Recip, split_float,      split_float);

VSIP_IMPL_OP1SUP(copy_token,    float*,           float*);
VSIP_IMPL_OP1SUP(copy_token,    complex<float>*,  complex<float>*);
VSIP_IMPL_OP1SUP(copy_token,    split_float,      split_float);

VSIP_IMPL_OP1SUP(copy_token,    split_float,      complex<float>*);
VSIP_IMPL_OP1SUP(copy_token,    complex<float>*,  split_float);

VSIP_IMPL_OP1SUP(copy_token,    unsigned long*,   float*);
VSIP_IMPL_OP1SUP(copy_token,    unsigned short*,  float*);

VSIP_IMPL_OP1SUP(Cast_closure<long          >::Cast, float*, long*);
VSIP_IMPL_OP1SUP(Cast_closure<short         >::Cast, float*, short*);
#if VSIP_IMPL_SAL_USES_SIGNED == 1
VSIP_IMPL_OP1SUP(Cast_closure<signed char   >::Cast, float*, signed char*);
#else
VSIP_IMPL_OP1SUP(Cast_closure<char          >::Cast, float*, char*);
#endif
VSIP_IMPL_OP1SUP(Cast_closure<unsigned long >::Cast, float*, unsigned long*);
VSIP_IMPL_OP1SUP(Cast_closure<unsigned short>::Cast, float*, unsigned short*);
VSIP_IMPL_OP1SUP(Cast_closure<unsigned char >::Cast, float*, unsigned char*);

VSIP_IMPL_OP1SUP(Cast_closure<float>::Cast, long*, float*);
VSIP_IMPL_OP1SUP(Cast_closure<float>::Cast, short*, float*);
#if VSIP_IMPL_SAL_USES_SIGNED == 1
VSIP_IMPL_OP1SUP(Cast_closure<float>::Cast, signed char*, float*);
#else
VSIP_IMPL_OP1SUP(Cast_closure<float>::Cast, char*, float*);
#endif
VSIP_IMPL_OP1SUP(Cast_closure<float>::Cast, unsigned long*, float*);
VSIP_IMPL_OP1SUP(Cast_closure<float>::Cast, unsigned short*, float*);
VSIP_IMPL_OP1SUP(Cast_closure<float>::Cast, unsigned char*, float*);


#if VSIP_IMPL_HAVE_SAL_DOUBLE
VSIP_IMPL_OP1SUP(expr::op::Magsq, complex<double>*, double*);
VSIP_IMPL_OP1SUP(expr::op::Magsq, split_double,     double*);

VSIP_IMPL_OP1SUP(expr::op::Minus,     double*,          double*);
VSIP_IMPL_OP1SUP(expr::op::Minus,     complex<double>*, complex<double>*);
VSIP_IMPL_OP1SUP(expr::op::Minus,     split_double,     split_double);

VSIP_IMPL_OP1SUP(expr::op::Mag,   double*,          double*);

VSIP_IMPL_OP1SUP(expr::op::Cos,   double*,          double*);

VSIP_IMPL_OP1SUP(expr::op::Sin,   double*,          double*);

VSIP_IMPL_OP1SUP(expr::op::Atan,  double*,          double*);

VSIP_IMPL_OP1SUP(expr::op::Log,   double*,          double*);

VSIP_IMPL_OP1SUP(expr::op::Log10, double*,          double*);

VSIP_IMPL_OP1SUP(expr::op::Exp,   double*,          double*);

VSIP_IMPL_OP1SUP(expr::op::Exp10, double*,          double*);

VSIP_IMPL_OP1SUP(expr::op::Sqrt,  double*,          double*);

VSIP_IMPL_OP1SUP(expr::op::Rsqrt, double*,          double*);

VSIP_IMPL_OP1SUP(expr::op::Sq,    double*,          double*);

// recip: no scalar double
VSIP_IMPL_OP1SUP(expr::op::Recip, complex<double>*, complex<double>*);
VSIP_IMPL_OP1SUP(expr::op::Recip, split_double,     split_double);

VSIP_IMPL_OP1SUP(copy_token,    complex<double>*, complex<double>*);
VSIP_IMPL_OP1SUP(copy_token,    split_double,     split_double);

VSIP_IMPL_OP1SUP(copy_token,    split_double,     complex<double>*);
VSIP_IMPL_OP1SUP(copy_token,    complex<double>*, split_double);
#endif // VSIP_IMPL_HAVE_SAL_DOUBLE


/***********************************************************************
  Binary operators and functions provided by SAL
***********************************************************************/

// straight-up vector add
VSIP_IMPL_OP2SUP(expr::op::Add, int*,             int*,            int*);
VSIP_IMPL_OP2SUP(expr::op::Add, float*,           float*,          float*);
VSIP_IMPL_OP2SUP(expr::op::Add, complex<float>*,  complex<float>*, complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Add, split_float,      split_float,     split_float);

VSIP_IMPL_OP2SUP(expr::op::Add, float*,           complex<float>*, complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Add, complex<float>*,  float*,          complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Add, float*,           split_float,     split_float);
VSIP_IMPL_OP2SUP(expr::op::Add, split_float,      float*,          split_float);

// scalar-vector vector add
VSIP_IMPL_OP2SUP(expr::op::Add, int,              int*,            int*);
VSIP_IMPL_OP2SUP(expr::op::Add, int*,             int,             int*);
VSIP_IMPL_OP2SUP(expr::op::Add, float,            float*,          float*);
VSIP_IMPL_OP2SUP(expr::op::Add, float*,           float,           float*);
VSIP_IMPL_OP2SUP(expr::op::Add, complex<float>,   complex<float>*, complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Add, complex<float>*,  complex<float>,  complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Add, complex<float>,   split_float,     split_float);
VSIP_IMPL_OP2SUP(expr::op::Add, split_float,      complex<float>,  split_float);

VSIP_IMPL_OP2SUP(expr::op::Add, float,            complex<float>*, complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Add, complex<float>*,  float,           complex<float>*);

VSIP_IMPL_OP2SUP(expr::op::Add, float,            split_float,     split_float);
VSIP_IMPL_OP2SUP(expr::op::Add, split_float,      float,           split_float);


// straight-up vector sub
VSIP_IMPL_OP2SUP(expr::op::Sub, int*,             int*,            int*);
VSIP_IMPL_OP2SUP(expr::op::Sub, float*,           float*,          float*);
VSIP_IMPL_OP2SUP(expr::op::Sub, complex<float>*,  complex<float>*, complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Sub, split_float,      split_float,     split_float);

VSIP_IMPL_OP2SUP(expr::op::Sub, complex<float>*,  float*,          complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Sub, split_float,      float*,          split_float);

// scalar-vector vector sub
VSIP_IMPL_OP2SUP(expr::op::Sub, int*,             int,             int*);
// not in sal   (expr::op::Sub, float,            float*,          float*);
VSIP_IMPL_OP2SUP(expr::op::Sub, float*,           float,           float*);
// not in sal   (expr::op::Sub, complex<float>,   complex<float>*, complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Sub, complex<float>*,  complex<float>,  complex<float>*);
// not in sal   (expr::op::Sub, complex<float>,   split_float,     split_float);
VSIP_IMPL_OP2SUP(expr::op::Sub, split_float,      complex<float>,  split_float);

VSIP_IMPL_OP2SUP(expr::op::Sub, complex<float>*,  float,           complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Sub, split_float,      float,           split_float);


// straight-up vector multiply
VSIP_IMPL_OP2SUP(expr::op::Mult, int*,            int*,            int*);
VSIP_IMPL_OP2SUP(expr::op::Mult, float*,          float*,          float*);
VSIP_IMPL_OP2SUP(expr::op::Mult, complex<float>*, complex<float>*, complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Mult, split_float,     split_float,     split_float);

// real-complex vector multiply
VSIP_IMPL_OP2SUP(expr::op::Mult, complex<float>*, float*,          complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Mult, float*,          complex<float>*, complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Mult, split_float,     float*,          split_float);
VSIP_IMPL_OP2SUP(expr::op::Mult, float*,          split_float,     split_float);

// scalar-vector vector multiply
VSIP_IMPL_OP2SUP(expr::op::Mult, int,             int*,            int*);
VSIP_IMPL_OP2SUP(expr::op::Mult, int*,            int,             int*);
VSIP_IMPL_OP2SUP(expr::op::Mult, float,           float*,          float*);
VSIP_IMPL_OP2SUP(expr::op::Mult, float*,          float,           float*);
VSIP_IMPL_OP2SUP(expr::op::Mult, complex<float>,  complex<float>*, complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Mult, complex<float>*, complex<float>,  complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Mult, complex<float>,  split_float,     split_float);
VSIP_IMPL_OP2SUP(expr::op::Mult, split_float,     complex<float>,  split_float);

VSIP_IMPL_OP2SUP(expr::op::Mult, float,           complex<float>*, complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Mult, complex<float>*, float,           complex<float>*);

VSIP_IMPL_OP2SUP(expr::op::Mult, float,           split_float,     split_float);
VSIP_IMPL_OP2SUP(expr::op::Mult, split_float,     float,           split_float);



// straight-up vector division
VSIP_IMPL_OP2SUP(expr::op::Div, int*,             int*,            int*);
VSIP_IMPL_OP2SUP(expr::op::Div, float*,           float*,          float*);
VSIP_IMPL_OP2SUP(expr::op::Div, complex<float>*,  complex<float>*, complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Div, split_float,      split_float,     split_float);

VSIP_IMPL_OP2SUP(expr::op::Div, complex<float>*,  float*,          complex<float>*);
VSIP_IMPL_OP2SUP(expr::op::Div, split_float,      float*,          split_float);

// scalar-vector vector division
// not in sal  (expr::op::Div, int,             int*,            int*);
#if VSIP_IMPL_SAL_HAS_VSDIVIX
VSIP_IMPL_OP2SUP(expr::op::Div, int*,            int,             int*);
#endif
VSIP_IMPL_OP2SUP(expr::op::Div, float,           float*,          float*);
VSIP_IMPL_OP2SUP(expr::op::Div, float*,          float,           float*);
// not in sal   (expr::op::Div, complex<float>,  complex<float>*, complex<float>*);
// not in sal   (expr::op::Div, complex<float>*, complex<float>,  complex<float>*);


// Logical

VSIP_IMPL_OP2SUP(expr::op::Band, int*,             int*,            int*);
VSIP_IMPL_OP2SUP(expr::op::Bor,  int*,             int*,            int*);


// vector comparisons

VSIP_IMPL_OP2SUP(expr::op::Max, float*,             float*,            float*);

VSIP_IMPL_OP2SUP(expr::op::Min, float*,             float*,            float*);


// Vector comparisons to 1/0
VSIP_IMPL_OP2SUP(veq_token, float*,  float*,  float*);
VSIP_IMPL_OP2SUP(veq_token, int*,    int*,    int*);
VSIP_IMPL_OP2SUP(vne_token, float*,  float*,  float*);
VSIP_IMPL_OP2SUP(vne_token, int*,    int*,    int*);
VSIP_IMPL_OP2SUP(vgt_token, float*,  float*,  float*);
VSIP_IMPL_OP2SUP(vgt_token, int*,    int*,    int*);
VSIP_IMPL_OP2SUP(vge_token, float*,  float*,  float*);
VSIP_IMPL_OP2SUP(vge_token, int*,    int*,    int*);
VSIP_IMPL_OP2SUP(vlt_token, float*,  float*,  float*);
VSIP_IMPL_OP2SUP(vlt_token, int*,    int*,    int*);
VSIP_IMPL_OP2SUP(vle_token, float*,  float*,  float*);
VSIP_IMPL_OP2SUP(vle_token, int*,    int*,    int*);

VSIP_IMPL_OP2SUP(expr::op::Atan2, float*,  float*,  float*);


#if VSIP_IMPL_HAVE_SAL_DOUBLE
// straight-up vector add
VSIP_IMPL_OP2SUP(expr::op::Add, double*,          double*,         double*);
VSIP_IMPL_OP2SUP(expr::op::Add, complex<double>*, complex<double>*,complex<double>*);
VSIP_IMPL_OP2SUP(expr::op::Add, split_double,     split_double,    split_double);

// no crvadddx in SAL

// scalar-vector vector add
VSIP_IMPL_OP2SUP(expr::op::Add, double,           double*,         double*);
VSIP_IMPL_OP2SUP(expr::op::Add, double*,          double,          double*);
VSIP_IMPL_OP2SUP(expr::op::Add, complex<double>,  complex<double>*,complex<double>*);
VSIP_IMPL_OP2SUP(expr::op::Add, complex<double>*, complex<double>, complex<double>*);
VSIP_IMPL_OP2SUP(expr::op::Add, complex<double>,  split_double,    split_double);
VSIP_IMPL_OP2SUP(expr::op::Add, split_double,     complex<double>, split_double);

VSIP_IMPL_OP2SUP(expr::op::Add, double,           complex<double>*,complex<double>*);
VSIP_IMPL_OP2SUP(expr::op::Add, complex<double>*, double,          complex<double>*);

VSIP_IMPL_OP2SUP(expr::op::Add, double,           split_double,    split_double);
VSIP_IMPL_OP2SUP(expr::op::Add, split_double,     double,          split_double);


// straight-up vector sub
VSIP_IMPL_OP2SUP(expr::op::Sub, double*,          double*,         double*);
VSIP_IMPL_OP2SUP(expr::op::Sub, complex<double>*, complex<double>*,complex<double>*);
VSIP_IMPL_OP2SUP(expr::op::Sub, split_double,     split_double,    split_double);

// scalar-vector vector sub
// not in sal   (expr::op::Sub, double,           double*,         double*);
VSIP_IMPL_OP2SUP(expr::op::Sub, double*,          double,          double*);
// not in sal   (expr::op::Sub, complex<double>,  complex<double>*,complex<double>*);
VSIP_IMPL_OP2SUP(expr::op::Sub, complex<double>*, complex<double>, complex<double>*);
// not in sal   (expr::op::Sub, complex<double>,  split_double,    split_double);
VSIP_IMPL_OP2SUP(expr::op::Sub, split_double,     complex<double>, split_double);

VSIP_IMPL_OP2SUP(expr::op::Sub, complex<double>*, double,          complex<double>*);
VSIP_IMPL_OP2SUP(expr::op::Sub, split_double,     double,          split_double);


// straight-up vector multiply
VSIP_IMPL_OP2SUP(expr::op::Mult, double*,         double*,         double*);
VSIP_IMPL_OP2SUP(expr::op::Mult, complex<double>*,complex<double>*,complex<double>*);
VSIP_IMPL_OP2SUP(expr::op::Mult, split_double,    split_double,    split_double);

// real-complex vector multiply

// scalar-vector vector multiply
VSIP_IMPL_OP2SUP(expr::op::Mult, double,          double*,         double*);
VSIP_IMPL_OP2SUP(expr::op::Mult, double*,         double,          double*);
VSIP_IMPL_OP2SUP(expr::op::Mult, complex<double>, complex<double>*,complex<double>*);
VSIP_IMPL_OP2SUP(expr::op::Mult, complex<double>*,complex<double>, complex<double>*);
VSIP_IMPL_OP2SUP(expr::op::Mult, complex<double>, split_double,    split_double);
VSIP_IMPL_OP2SUP(expr::op::Mult, split_double,    complex<double>, split_double);

VSIP_IMPL_OP2SUP(expr::op::Mult, double,          complex<double>*,complex<double>*);
VSIP_IMPL_OP2SUP(expr::op::Mult, complex<double>*,double,          complex<double>*);

VSIP_IMPL_OP2SUP(expr::op::Mult, double,          split_double,    split_double);
VSIP_IMPL_OP2SUP(expr::op::Mult, split_double,    double,          split_double);



// straight-up vector division
VSIP_IMPL_OP2SUP(expr::op::Div, double*,          double*,         double*);
VSIP_IMPL_OP2SUP(expr::op::Div, complex<double>*, complex<double>*,complex<double>*);
VSIP_IMPL_OP2SUP(expr::op::Div, split_double,     split_double,    split_double);

// scalar-vector vector division
// not in sal   (expr::op::Div, double,          double*,         double*);
VSIP_IMPL_OP2SUP(expr::op::Div, double*,         double,          double*);
// not in sal   (expr::op::Div, complex<double>, complex<double>*,complex<double>*);
// not in sal   (expr::op::Div, complex<double>*,complex<double>, complex<double>*);

// vector min/max
VSIP_IMPL_OP2SUP(expr::op::Max, double*,            double*,           double*);
VSIP_IMPL_OP2SUP(expr::op::Min, double*,            double*,           double*);


// Vector comparisons to 1/0
VSIP_IMPL_OP2SUP(veq_token, double*, double*, double*);
VSIP_IMPL_OP2SUP(vne_token, double*, double*, double*);
VSIP_IMPL_OP2SUP(vgt_token, double*, double*, double*);
VSIP_IMPL_OP2SUP(vge_token, double*, double*, double*);
VSIP_IMPL_OP2SUP(vlt_token, double*, double*, double*);
VSIP_IMPL_OP2SUP(vle_token, double*, double*, double*);

VSIP_IMPL_OP2SUP(expr::op::Atan2, double*,  double*,  double*);
#endif // VSIP_IMPL_HAVE_SAL_DOUBLE



/***********************************************************************
  Ternary operators and functions provided by SAL.
***********************************************************************/

// Multiply-add

VSIP_IMPL_OP3SUP(expr::op::Ma, float,   float*,  float*, float*);
VSIP_IMPL_OP3SUP(expr::op::Ma, float*,  float,   float*, float*);
VSIP_IMPL_OP3SUP(expr::op::Ma, float*,  float*,  float,  float*);
VSIP_IMPL_OP3SUP(expr::op::Ma, float*,  float*,  float*, float*);

VSIP_IMPL_OP3SUP(expr::op::Ma, complex<float>*, complex<float>, complex<float>*,
		 complex<float>*);
VSIP_IMPL_OP3SUP(expr::op::Ma, complex<float>, complex<float>*, complex<float>*,
		 complex<float>*);
VSIP_IMPL_OP3SUP(expr::op::Ma, split_float, complex<float>, split_float,
		 split_float);
VSIP_IMPL_OP3SUP(expr::op::Ma, complex<float>, split_float, split_float,
		 split_float);

#if VSIP_IMPL_HAVE_SAL_DOUBLE
VSIP_IMPL_OP3SUP(expr::op::Ma, double,   double*,  double*, double*);
VSIP_IMPL_OP3SUP(expr::op::Ma, double*,  double,   double*, double*);
VSIP_IMPL_OP3SUP(expr::op::Ma, double*,  double*,  double,  double*);
VSIP_IMPL_OP3SUP(expr::op::Ma, double*,  double*,  double*, double*);

VSIP_IMPL_OP3SUP(expr::op::Ma,
		 complex<double>*, complex<double>, complex<double>*,
		 complex<double>*);
VSIP_IMPL_OP3SUP(expr::op::Ma,
		 complex<double>, complex<double>*, complex<double>*,
		 complex<double>*);
#endif // VSIP_IMPL_HAVE_SAL_DOUBLE


// Multiply-subtract

VSIP_IMPL_OP3SUP(expr::op::Msb, float*,  float,   float*, float*);
// not in sal   (expr::op::Msb, float*,  float*,  float,  float*);
VSIP_IMPL_OP3SUP(expr::op::Msb, float*,  float*,  float*, float*);

#if VSIP_IMPL_HAVE_SAL_DOUBLE
VSIP_IMPL_OP3SUP(expr::op::Msb, double*,  double,   double*, double*);
// not in sal   (expr::op::Msb, double*,  double*,  double,  double*);
VSIP_IMPL_OP3SUP(expr::op::Msb, double*,  double*,  double*, double*);
#endif // VSIP_IMPL_HAVE_SAL_DOUBLE

// no complex msb in SAL


// Add-multiply

// not in SAL   (expr::op::Am, float,   float*,  float*, float*);
// not in SAL   (expr::op::Am, float*,  float,   float*, float*);
VSIP_IMPL_OP3SUP(expr::op::Am, float*,  float*,  float,  float*);
VSIP_IMPL_OP3SUP(expr::op::Am, float*,  float*,  float*, float*);

#if VSIP_IMPL_HAVE_SAL_DOUBLE
// not in SAL   (expr::op::Am, double,  double*, double*,double*);
// not in SAL   (expr::op::Am, double*, double,  double*,double*);
// not in SAL   (expr::op::Am, double*, double*, double, double*);
VSIP_IMPL_OP3SUP(expr::op::Am, double*, double*, double*,double*);
#endif // VSIP_IMPL_HAVE_SAL_DOUBLE


// Subtract-multiply
VSIP_IMPL_OP3SUP(expr::op::Sbm, float*,  float*,  float,  float*);
VSIP_IMPL_OP3SUP(expr::op::Sbm, float*,  float*,  float*, float*);
#if VSIP_IMPL_HAVE_SAL_DOUBLE
VSIP_IMPL_OP3SUP(expr::op::Sbm, double*, double*, double*,double*);
#endif // VSIP_IMPL_HAVE_SAL_DOUBLE


// Conjugate(multiply)-add

VSIP_IMPL_OP3SUP(cma_token, complex<float>*, complex<float>*, complex<float>*,
		 complex<float>*);
VSIP_IMPL_OP3SUP(cma_token, split_float*, split_float*, split_float*,
		 split_float*);



#undef VSIP_IMPL_OP1SUP
#undef VSIP_IMPL_OP2SUP
#undef VSIP_IMPL_OP3SUP

} // namespace vsip::impl::sal
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_SAL_IS_OP_SUPPORTED_HPP
