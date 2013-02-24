/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/sal/elementwise.hpp
    @author  Don McCoy
    @date    2005-10-04
    @brief   VSIPL++ Library: Wrappers to bridge with Mercury SAL
             elementwise functions.
*/

#ifndef VSIP_OPT_SAL_ELEMENTWISE_HPP
#define VSIP_OPT_SAL_ELEMENTWISE_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <sal.h>

#include <vsip/support.hpp>
#include <vsip/core/coverage.hpp>
#include <vsip/opt/sal/bridge_util.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace sal
{

#define VSIP_IMPL_SAL_INLINE inline



/***********************************************************************
  Unary Functions
***********************************************************************/

// (real vector) -> real vector

#define VSIP_IMPL_SAL_V(FCN, T, SALFCN)					\
VSIP_IMPL_SAL_INLINE void						\
FCN(									\
  Sal_vector<T> const& A,						\
  Sal_vector<T> const& Z,						\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_V", SALFCN)					\
  SALFCN(A.ptr, A.stride, Z.ptr, Z.stride, len, 0);			\
}

VSIP_IMPL_SAL_V(vneg, int,    vnegix)
VSIP_IMPL_SAL_V(vneg, float,  vnegx)
VSIP_IMPL_SAL_V(vneg, double, vnegdx)

VSIP_IMPL_SAL_V(vmag, int,    vabsix)
VSIP_IMPL_SAL_V(vmag, float,  vabsx)
VSIP_IMPL_SAL_V(vmag, double, vabsdx)

VSIP_IMPL_SAL_V(vcos, float,  vcosx)
VSIP_IMPL_SAL_V(vcos, double, vcosdx)

VSIP_IMPL_SAL_V(vsin, float,  vsinx)
VSIP_IMPL_SAL_V(vsin, double, vsindx)

VSIP_IMPL_SAL_V(vtan, float,  vtanx)
// no vtandx in SAL

VSIP_IMPL_SAL_V(vatan, float,  vatanx)
VSIP_IMPL_SAL_V(vatan, double, vatandx)

VSIP_IMPL_SAL_V(vexp, float,  vexpx)
VSIP_IMPL_SAL_V(vexp, double, vexpdx)

VSIP_IMPL_SAL_V(vlog, float,  vlnx)
VSIP_IMPL_SAL_V(vlog, double, vlndx)

VSIP_IMPL_SAL_V(vlog10, float,  vlogx)
VSIP_IMPL_SAL_V(vlog10, double, vlogdx)

VSIP_IMPL_SAL_V(vexp10, float,  valogx)
VSIP_IMPL_SAL_V(vexp10, double, valogdx)

VSIP_IMPL_SAL_V(vsqrt, float,  vsqrtx)
VSIP_IMPL_SAL_V(vsqrt, double, vsqrtdx)

VSIP_IMPL_SAL_V(vrsqrt, float,  vrsqrtx)
VSIP_IMPL_SAL_V(vrsqrt, double, vrsqrtdx)

VSIP_IMPL_SAL_V(vsq, float,  vsqx)
VSIP_IMPL_SAL_V(vsq, double, vsqdx)

VSIP_IMPL_SAL_V(vcopy, int,    vmovix)
VSIP_IMPL_SAL_V(vcopy, float,  vmovx)
VSIP_IMPL_SAL_V(vcopy, double, vmovdx)

VSIP_IMPL_SAL_V(vrecip, float,  vrecipx)



// type conversion (vector) -> vector

#define VSIP_IMPL_SAL_V_CONVERT(FCN, ST, DT, SALFCN)			\
VSIP_IMPL_SAL_INLINE void						\
FCN(									\
  Sal_vector<ST> const& A,						\
  Sal_vector<DT> const& Z,						\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_V_CONVERT", SALFCN)				\
  SALFCN(A.ptr, A.stride, Z.ptr, Z.stride,				\
	 0 /* scale 1.0 */, 0 /* bias 0.0 */,				\
	 len, SAL_ROUND_ZERO, 0);					\
}

VSIP_IMPL_SAL_V_CONVERT(vcopy, unsigned long,  float, vconvert_u32_f32x)
VSIP_IMPL_SAL_V_CONVERT(vcopy, unsigned short, float, vconvert_u16_f32x)


// (complex vector) -> complex vector

#define VSIP_IMPL_SAL_CV(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& A,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_CV", SALFCN)					\
  typedef Sal_inter<T>::type inter_type;				\
  SALFCN((inter_type*)A.ptr, 2*A.stride,				\
         (inter_type*)Z.ptr, 2*Z.stride, len, 0);			\
}

#define VSIP_IMPL_SAL_ZV(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_split_fmt > const& A,			\
  Sal_vector<complex<T>, Cmplx_split_fmt > const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_ZV", SALFCN)					\
  typedef Sal_split<T>::type split_type;				\
  SALFCN((split_type*)&A.ptr, A.stride,					\
         (split_type*)&Z.ptr, Z.stride, len, 0);			\
}

VSIP_IMPL_SAL_CV(vneg, float,  cvnegx)
VSIP_IMPL_SAL_CV(vneg, double, cvnegdx)
VSIP_IMPL_SAL_ZV(vneg, float,  zvnegx)
VSIP_IMPL_SAL_ZV(vneg, double, zvnegdx)

VSIP_IMPL_SAL_CV(vsqrt, float,  cvsqrtx)
VSIP_IMPL_SAL_ZV(vsqrt, float,  zvsqrtx)

VSIP_IMPL_SAL_CV(vrecip, float,  cvrcipx)
VSIP_IMPL_SAL_CV(vrecip, double, cvrcipdx)
VSIP_IMPL_SAL_ZV(vrecip, float,  zvrcipx)
VSIP_IMPL_SAL_ZV(vrecip, double, zvrcipdx)

VSIP_IMPL_SAL_CV(vcopy, float,  cvmovx)
VSIP_IMPL_SAL_CV(vcopy, double, cvmovdx)
VSIP_IMPL_SAL_ZV(vcopy, float,  zvmovx)
VSIP_IMPL_SAL_ZV(vcopy, double, zvmovdx)


#define VSIP_IMPL_SAL_CVR(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_inter_fmt > const& A,			\
  Sal_vector<T> const& Z,						\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_CVR", SALFCN)				\
  typedef Sal_inter<T>::type inter_type;				\
  SALFCN((inter_type*)A.ptr, 2*A.stride, Z.ptr, Z.stride, len, 0);	\
}

#define VSIP_IMPL_SAL_ZVR(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_split_fmt > const& A,			\
  Sal_vector<T> const& Z,						\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_ZVR", SALFCN)				\
  typedef Sal_split<T>::type split_type;				\
  SALFCN((split_type*)&A.ptr, A.stride, Z.ptr, Z.stride, len, 0);	\
}

VSIP_IMPL_SAL_CVR(vmagsq, float,  cvmagsx)
VSIP_IMPL_SAL_CVR(vmagsq, double, cvmagsdx)

VSIP_IMPL_SAL_ZVR(vmagsq, float,  zvmagsx)
VSIP_IMPL_SAL_ZVR(vmagsq, double, zvmagsdx)

VSIP_IMPL_SAL_CVR(vmag, float,  cvabsx)
VSIP_IMPL_SAL_ZVR(vmag, float,  zvabsx)



#define VSIP_IMPL_SAL_CTOZ(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_inter_fmt > const& A,			\
  Sal_vector<complex<T>, Cmplx_split_fmt > const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_CTOZ", SALFCN)				\
  typedef Sal_inter<T>::type inter_type;				\
  typedef Sal_split<T>::type split_type;				\
  SALFCN((inter_type*) A.ptr, 2*A.stride,				\
         (split_type*)&Z.ptr,   Z.stride, len, 0);			\
}

#define VSIP_IMPL_SAL_ZTOC(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_split_fmt > const& A,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt > const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_ZTOC", SALFCN)				\
  typedef Sal_inter<T>::type inter_type;				\
  typedef Sal_split<T>::type split_type;				\
  SALFCN((split_type*)&A.ptr,   A.stride,				\
         (inter_type*) Z.ptr, 2*Z.stride, len, 0);			\
}

VSIP_IMPL_SAL_CTOZ(vcopy, float,  ctozx)
VSIP_IMPL_SAL_CTOZ(vcopy, double, ctozdx)

VSIP_IMPL_SAL_ZTOC(vcopy, float,  ztocx)
VSIP_IMPL_SAL_ZTOC(vcopy, double, ztocdx)



// Vector conversion
#define VSIP_IMPL_SAL_VCONV(FCN, ST, DT, SALFCN)			\
VSIP_IMPL_SAL_INLINE void						\
FCN(									\
  Sal_vector<ST> const& A,						\
  Sal_vector<DT> const& Z,						\
  length_type len)							\
{									\
  float scale = 1.f;							\
  float bias  = 0.f;							\
  VSIP_IMPL_COVER_FCN("SAL_VCONV", SALFCN)				\
  SALFCN(A.ptr, A.stride, Z.ptr, Z.stride, &scale, &bias, len,		\
	 SAL_ROUND_ZERO, 0);						\
}

VSIP_IMPL_SAL_VCONV(vconv, float,          long,  vconvert_f32_s32x);
VSIP_IMPL_SAL_VCONV(vconv, float,          short, vconvert_f32_s16x);
#if VSIP_IMPL_SAL_USES_SIGNED == 1
VSIP_IMPL_SAL_VCONV(vconv, float,   signed char,  vconvert_f32_s8x);
#else
VSIP_IMPL_SAL_VCONV(vconv, float,          char,  vconvert_f32_s8x);
#endif
VSIP_IMPL_SAL_VCONV(vconv, float, unsigned long,  vconvert_f32_u32x);
VSIP_IMPL_SAL_VCONV(vconv, float, unsigned short, vconvert_f32_u16x);
VSIP_IMPL_SAL_VCONV(vconv, float, unsigned char,  vconvert_f32_u8x);

VSIP_IMPL_SAL_VCONV(vconv,          long,  float, vconvert_s32_f32x);
VSIP_IMPL_SAL_VCONV(vconv,          short, float, vconvert_s16_f32x);
#if VSIP_IMPL_SAL_USES_SIGNED == 1
VSIP_IMPL_SAL_VCONV(vconv,   signed char,  float, vconvert_s8_f32x);
#else
VSIP_IMPL_SAL_VCONV(vconv,          char,  float, vconvert_s8_f32x);
#endif
VSIP_IMPL_SAL_VCONV(vconv, unsigned long,  float, vconvert_u32_f32x);
VSIP_IMPL_SAL_VCONV(vconv, unsigned short, float, vconvert_u16_f32x);
VSIP_IMPL_SAL_VCONV(vconv, unsigned char,  float, vconvert_u8_f32x);

#undef VSIP_IMPL_SAL_VCONV



/***********************************************************************
  Binary Functions
***********************************************************************/

// (real vector, real vector) -> real vector

// Notes:
//  - For subtraction
//    Real values use functions where C = B - A in SAL
//    Complex values use C = A - B
//
//  - For division
//    All values use functions where C = B / A in SAL

#define VSIP_IMPL_VV(FCN, T, SALFCN)					\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<T> const& A,						\
  Sal_vector<T> const& B,						\
  Sal_vector<T> const& Z,						\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_VV", SALFCN)					\
  SALFCN(A.ptr, A.stride, B.ptr, B.stride, Z.ptr, Z.stride, len, 0);	\
}

#define VSIP_IMPL_VV_R(FCN, T, SALFCN)					\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<T> const& A,						\
  Sal_vector<T> const& B,						\
  Sal_vector<T> const& Z,						\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_VV_R", SALFCN)				\
  SALFCN(B.ptr, B.stride, A.ptr, A.stride, Z.ptr, Z.stride, len, 0);	\
}

VSIP_IMPL_VV  (vband, int,  vandix)
VSIP_IMPL_VV  (vbor,  int,  vorix)

VSIP_IMPL_VV  (vadd,  int,  vaddix)
VSIP_IMPL_VV_R(vsub,  int,  vsubix)
VSIP_IMPL_VV  (vmul,  int,  vmulix)
VSIP_IMPL_VV_R(vdiv,  int,  vdivix)

VSIP_IMPL_VV  (vadd, float, vaddx)
VSIP_IMPL_VV_R(vsub, float, vsubx)
VSIP_IMPL_VV  (vmul, float, vmulx)
VSIP_IMPL_VV_R(vdiv, float, vdivx)

VSIP_IMPL_VV  (vadd, double, vadddx)
VSIP_IMPL_VV_R(vsub, double, vsubdx)
VSIP_IMPL_VV  (vmul, double, vmuldx)
VSIP_IMPL_VV_R(vdiv, double, vdivdx)

VSIP_IMPL_VV  (vmax,   float,  vmaxx)
VSIP_IMPL_VV  (vmax,   double, vmaxdx)
VSIP_IMPL_VV  (vmin,   float,  vminx)
VSIP_IMPL_VV  (vmin,   double, vmindx)
VSIP_IMPL_VV  (vmaxmg, float,  vmaxmgx)
VSIP_IMPL_VV  (vmaxmg, double, vmaxmgdx)
VSIP_IMPL_VV  (vminmg, float,  vminmgx)
VSIP_IMPL_VV  (vminmg, double, vminmgdx)

VSIP_IMPL_VV  (lveq,   float,  lveqx)
VSIP_IMPL_VV  (lveq,   double, lveqdx)
VSIP_IMPL_VV  (lveq,   int,    lveqix)
VSIP_IMPL_VV  (lvne,   float,  lvnex)
VSIP_IMPL_VV  (lvne,   double, lvnedx)
VSIP_IMPL_VV  (lvne,   int,    lvneix)
VSIP_IMPL_VV  (lvge,   float,  lvgex)
VSIP_IMPL_VV  (lvge,   double, lvgedx)
VSIP_IMPL_VV  (lvge,   int,    lvgeix)
VSIP_IMPL_VV  (lvgt,   float,  lvgtx)
VSIP_IMPL_VV  (lvgt,   double, lvgtdx)
VSIP_IMPL_VV  (lvgt,   int,    lvgtix)
VSIP_IMPL_VV  (lvle,   float,  lvlex)
VSIP_IMPL_VV  (lvle,   double, lvledx)
VSIP_IMPL_VV  (lvle,   int,    lvleix)
VSIP_IMPL_VV  (lvlt,   float,  lvltx)
VSIP_IMPL_VV  (lvlt,   double, lvltdx)
VSIP_IMPL_VV  (lvlt,   int,    lvltix)

VSIP_IMPL_VV_R(vatan2,   float,  vatan2x)
VSIP_IMPL_VV_R(vatan2,   double, vatan2dx)


// (inter vector, inter vector) -> inter vector

#define VSIP_IMPL_SAL_CVV(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& A,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_CVV", SALFCN)				\
  typedef Sal_inter<T>::type inter_type;				\
  /* complex elements call for a stride of 2 and not 1 (when		\
   * dealing with dense data for example). this differs from		\ 
   * the VSIPL++ definition of 1 == 1 pair of values. */		\
  SALFCN((inter_type*)A.ptr, 2*A.stride,				\
         (inter_type*)B.ptr, 2*B.stride,				\
         (inter_type*)Z.ptr, 2*Z.stride, len, 0);			\
}

// CF - Conjugate flag
#define VSIP_IMPL_SAL_CVV_CF(FCN, T, SALFCN, FLAG)			\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& A,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_CVV_CF", SALFCN)				\
  typedef Sal_inter<T>::type inter_type;				\
  /* complex elements call for a stride of 2 and not 1 (when		\
   * dealing with dense data for example). this differs from		\ 
   * the VSIPL++ definition of 1 == 1 pair of values. */		\
  SALFCN((inter_type*)A.ptr, 2*A.stride, (inter_type*)B.ptr, 2*B.stride, (inter_type*)Z.ptr, 2*Z.stride, len, FLAG, 0);	\
}

#define VSIP_IMPL_SAL_CVV_R(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& A,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_CVV_R", SALFCN)				\
  typedef Sal_inter<T>::type inter_type;				\
  /* complex elements call for a stride of 2 and not 1 (when		\
   * dealing with dense data for example). this differs from		\ 
   * the VSIPL++ definition of 1 == 1 pair of values. */		\
  SALFCN((inter_type*)B.ptr, 2*B.stride, (inter_type*)A.ptr, 2*A.stride, (inter_type*)Z.ptr, 2*Z.stride, len, 0);	\
}

VSIP_IMPL_SAL_CVV   (vadd, float, cvaddx)
VSIP_IMPL_SAL_CVV   (vsub, float, cvsubx)
VSIP_IMPL_SAL_CVV_CF(vmul, float, cvmulx, 1)
VSIP_IMPL_SAL_CVV_R (vdiv, float, cvdivx)

VSIP_IMPL_SAL_CVV   (vadd, double, cvadddx)
VSIP_IMPL_SAL_CVV   (vsub, double, cvsubdx)
VSIP_IMPL_SAL_CVV_CF(vmul, double, cvmuldx, 1)
VSIP_IMPL_SAL_CVV_R (vdiv, double, cvdivdx)



// (split vector, split vector) -> split vector

#define VSIP_IMPL_SAL_ZVV(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(			 		 			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& A,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
  length_type len)							\
{									\
  typedef Sal_split<T>::type split_type;				\
  SALFCN((split_type*)&A.ptr, A.stride,\
         (split_type*)&B.ptr, B.stride, (split_type*)&Z.ptr, Z.stride, len, 0);	\
}

// CF - conjugate flag
#define VSIP_IMPL_SAL_ZVV_CF(FCN, T, SALFCN, FLAG)			\
VSIP_IMPL_SAL_INLINE							\
void FCN(			 		 			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& A,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
  length_type len)							\
{									\
  typedef Sal_split<T>::type split_type;				\
  SALFCN((split_type*)&A.ptr, A.stride, (split_type*)&B.ptr, B.stride, (split_type*)&Z.ptr, Z.stride, len, FLAG, 0);\
}

// R - reversed operand order
#define VSIP_IMPL_SAL_ZVV_R(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(			 		 			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& A,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
  length_type len)							\
{									\
  typedef Sal_split<T>::type split_type;				\
  SALFCN((split_type*)&B.ptr, B.stride, (split_type*)&A.ptr, A.stride, (split_type*)&Z.ptr, Z.stride, len, 0);	\
}

VSIP_IMPL_SAL_ZVV   (vadd, float, zvaddx)
VSIP_IMPL_SAL_ZVV   (vsub, float, zvsubx)
VSIP_IMPL_SAL_ZVV_CF(vmul, float, zvmulx, 1)
VSIP_IMPL_SAL_ZVV_R (vdiv, float, zvdivx)

#if VSIP_IMPL_HAVE_SAL_DOUBLE
VSIP_IMPL_SAL_ZVV   (vadd, double, zvadddx)
VSIP_IMPL_SAL_ZVV   (vsub, double, zvsubdx)
VSIP_IMPL_SAL_ZVV_CF(vmul, double, zvmuldx, 1)
VSIP_IMPL_SAL_ZVV_R (vdiv, double, zvdivdx)
#endif



/***********************************************************************
  Ternary Functions
***********************************************************************/

#define VSIP_IMPL_SAL_CVVV(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& A,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& C,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  typedef Sal_inter<T>::type inter_type;				\
  SALFCN((inter_type*)A.ptr, 2*A.stride,				\
         (inter_type*)B.ptr, 2*B.stride,				\
         (inter_type*)C.ptr, 2*C.stride,				\
         (inter_type*)Z.ptr, 2*Z.stride, len, 0);			\
}

#define VSIP_IMPL_SAL_ZVVV(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(			 		 			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& A,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& C,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
  length_type len)							\
{									\
  typedef Sal_split<T>::type split_type;				\
  SALFCN((split_type*)&A.ptr, A.stride,					\
         (split_type*)&B.ptr, B.stride,					\
         (split_type*)&C.ptr, C.stride,					\
         (split_type*)&Z.ptr, Z.stride, len, 0);			\
}

VSIP_IMPL_SAL_CVVV(vcma, float, cvcmax)
VSIP_IMPL_SAL_ZVVV(vcma, float, zvcmax)



#define VSIP_IMPL_VS(FCN, T, SALFCN)					\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<T> const& A,						\
  Sal_scalar<T> const& B,						\
  Sal_vector<T> const& Z,						\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_VS", SALFCN)					\
  SALFCN(A.ptr, A.stride, const_cast<T*>(&B.value),			\
         Z.ptr, Z.stride, len, 0);					\
}

#define VSIP_IMPL_SV(FCN, T, SALFCN)					\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_scalar<T> const& A,						\
  Sal_vector<T> const& B,						\
  Sal_vector<T> const& Z,						\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_SV", SALFCN)					\
  SALFCN(const_cast<T*>(&A.value), B.ptr, B.stride, Z.ptr, Z.stride, len, 0);\
}

#define VSIP_IMPL_VS_COMM(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_scalar<T> const& A,						\
  Sal_vector<T> const& B,						\
  Sal_vector<T> const& Z,						\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_VS_COMM", SALFCN)				\
  SALFCN(B.ptr, B.stride, const_cast<T*>(&A.value),			\
         Z.ptr, Z.stride, len, 0);					\
}

#define VSIP_IMPL_VS_NEG(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<T> const& A,						\
  Sal_scalar<T> const& B,						\
  Sal_vector<T> const& Z,						\
  length_type len)							\
{									\
  T value = -B.value;							\
  VSIP_IMPL_COVER_FCN("SAL_VS", SALFCN)					\
  SALFCN(A.ptr, A.stride, const_cast<T*>(&value),			\
         Z.ptr, Z.stride, len, 0);					\
}

#define VSIP_IMPL_VS_BOTH(FCN, T, SALFCN)				\
  VSIP_IMPL_VS(FCN, T, SALFCN)						\
  VSIP_IMPL_VS_COMM(FCN, T, SALFCN)


VSIP_IMPL_VS_BOTH(vadd, int,    vsaddix)
VSIP_IMPL_VS_BOTH(vmul, int,    vsmulix)
VSIP_IMPL_VS_NEG (vsub, int,    vsaddix)
VSIP_IMPL_VS     (vdiv, int,    vsdivix)
// No svdivix in SAL

VSIP_IMPL_VS_BOTH(vadd, float,  vsaddx)
VSIP_IMPL_VS_BOTH(vmul, float,  vsmulx)
VSIP_IMPL_VS_NEG (vsub, float,  vsaddx)
VSIP_IMPL_VS     (vdiv, float,  vsdivx)
VSIP_IMPL_SV     (vdiv, float,  svdivx)

VSIP_IMPL_VS_BOTH(vadd, double, vsadddx)
VSIP_IMPL_VS_BOTH(vmul, double, vsmuldx)
VSIP_IMPL_VS_NEG (vsub, double, vsadddx)
VSIP_IMPL_VS     (vdiv, double, vsdivdx)
// No svdivdx in SAL




#define VSIP_IMPL_SAL_CVS(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& A,			\
  Sal_scalar<complex<T> > const&                 B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  typedef Sal_inter<T>::type inter_type;				\
  /* complex elements call for a stride of 2 and not 1 (when		\
   * dealing with dense data for example). this differs from		\ 
   * the VSIPL++ definition of 1 == 1 pair of values. */		\
  SALFCN((inter_type*)A.ptr, 2*A.stride, (inter_type*)&B.value, (inter_type*)Z.ptr, 2*Z.stride, len, 0);	\
}

#define VSIP_IMPL_SAL_CVS_COMM(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_scalar<complex<T> > const& B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& A,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  typedef Sal_inter<T>::type inter_type;				\
  /* complex elements call for a stride of 2 and not 1 (when		\
   * dealing with dense data for example). this differs from		\ 
   * the VSIPL++ definition of 1 == 1 pair of values. */		\
  SALFCN((inter_type*)A.ptr, 2*A.stride, (inter_type*)&B.value, (inter_type*)Z.ptr, 2*Z.stride, len, 0);	\
}

VSIP_IMPL_SAL_CVS(vmul, float,  cvcsmlx)
VSIP_IMPL_SAL_CVS(vmul, double, cvcsmldx)
VSIP_IMPL_SAL_CVS_COMM(vmul, float,  cvcsmlx)
VSIP_IMPL_SAL_CVS_COMM(vmul, double, cvcsmldx)



#define VSIP_IMPL_SAL_ZVS(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& A,			\
  Sal_scalar<complex<T> > const& B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
  length_type len)							\
{									\
  typedef Sal_split<T>::type split_type;				\
  T real = B.value.real();						\
  T imag = B.value.imag();						\
  split_type cB = { &real, &imag };					\
									\
  SALFCN((split_type*)&A.ptr, A.stride, &cB,				\
	 (split_type*)&Z.ptr, Z.stride, len, 0);			\
}

#define VSIP_IMPL_SAL_ZVS_COMM(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_scalar<complex<T> > const& A,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
  length_type len)							\
{									\
  typedef Sal_split<T>::type split_type;				\
  T real = A.value.real();						\
  T imag = A.value.imag();						\
  split_type cA = { &real, &imag };					\
									\
  SALFCN((split_type*)&B.ptr, B.stride, &cA, (split_type*)&Z.ptr, Z.stride, len, 0);		\
}

VSIP_IMPL_SAL_ZVS(vmul, float,  zvzsmlx)
VSIP_IMPL_SAL_ZVS(vmul, double, zvzsmldx)

VSIP_IMPL_SAL_ZVS_COMM(vmul, float,  zvzsmlx)
VSIP_IMPL_SAL_ZVS_COMM(vmul, double, zvzsmldx)



/// Add real-vector to complex-vector (interleaved)

#define VSIP_IMPL_SAL_CRV_FUNC(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& A,			\
  Sal_vector<T> const&                           B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  typedef Sal_inter<T>::type inter_type;				\
  SALFCN((inter_type*)A.ptr, 2 * A.stride, B.ptr, B.stride, (inter_type*)Z.ptr, 2 * Z.stride, len, 0 );	\
}


#define VSIP_IMPL_SAL_CRV_RFUNC(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<T> const&                           A,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  typedef Sal_inter<T>::type inter_type;				\
  SALFCN((inter_type*)B.ptr, 2 * B.stride, A.ptr, A.stride, (inter_type*)Z.ptr, 2 * Z.stride, len, 0 );	\
}

#define VSIP_IMPL_SAL_CRV_COMM(FCN, SALFCN, T)				\
   VSIP_IMPL_SAL_CRV_FUNC(FCN, SALFCN, T)				\
   VSIP_IMPL_SAL_CRV_RFUNC(FCN, SALFCN, T)

/// complex op real vector elementwise multiply, float, interleaved
VSIP_IMPL_SAL_CRV_COMM(vadd, float, crvaddx)
VSIP_IMPL_SAL_CRV_COMM(vmul, float, crvmulx)
VSIP_IMPL_SAL_CRV_FUNC(vdiv, float, crvdivx)
VSIP_IMPL_SAL_CRV_FUNC(vsub, float, crvsubx)



/// Add real-vector to complex-vector (split)

#define VSIP_IMPL_SAL_ZRV_FUNC(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& A,			\
  Sal_vector<T> const&                           B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
  length_type len)							\
{									\
  typedef Sal_split<T>::type split_type;				\
  SALFCN((split_type*)&A.ptr, A.stride, B.ptr, B.stride, (split_type*)&Z.ptr, Z.stride, len, 0);	\
}

#define VSIP_IMPL_SAL_ZRV_RFUNC(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<T> const&                           A,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
  length_type len)							\
{									\
  typedef Sal_split<T>::type split_type;				\
  SALFCN((split_type*)&B.ptr, B.stride, A.ptr, A.stride, (split_type*)&Z.ptr, Z.stride, len, 0);	\
}

#define VSIP_IMPL_SAL_ZRV_COMM(FCN, T, SALFCN)				\
   VSIP_IMPL_SAL_ZRV_FUNC(FCN, T, SALFCN)				\
   VSIP_IMPL_SAL_ZRV_RFUNC(FCN, T, SALFCN)
  

VSIP_IMPL_SAL_ZRV_COMM(vadd, float, zrvaddx)
VSIP_IMPL_SAL_ZRV_COMM(vmul, float, zrvmulx)
VSIP_IMPL_SAL_ZRV_FUNC(vdiv, float, zrvdivx)
VSIP_IMPL_SAL_ZRV_FUNC(vsub, float, zrvsubx)



#define VSIP_IMPL_SAL_VVV(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void									\
FCN(Sal_vector<T> const& A,						\
    Sal_vector<T> const& B,						\
    Sal_vector<T> const& C,						\
    Sal_vector<T> const& Z,						\
    int len )								\
{									\
  VSIP_IMPL_COVER_FCN("SAL_VVV", SALFCN)				\
  SALFCN(A.ptr, A.stride, B.ptr, B.stride, C.ptr, C.stride,		\
         Z.ptr, Z.stride, len, 0);					\
}

VSIP_IMPL_SAL_VVV(vma,  float,  vma_x)
VSIP_IMPL_SAL_VVV(vma,  double, vmadx)

VSIP_IMPL_SAL_VVV(vmsb, float,  vmsbx)
VSIP_IMPL_SAL_VVV(vmsb, double, vmsbdx)

VSIP_IMPL_SAL_VVV(vam,  float,  vamx)
VSIP_IMPL_SAL_VVV(vam,  double, vamdx)

VSIP_IMPL_SAL_VVV(vsbm,  float,  vsbmx)
VSIP_IMPL_SAL_VVV(vsbm,  double, vsbmdx)


#define VSIP_IMPL_SAL_VSV(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void									\
FCN(Sal_vector<T> const& A,						\
    Sal_scalar<T> const& B,						\
    Sal_vector<T> const& C,						\
    Sal_vector<T> const& Z,						\
    int len )								\
{									\
  VSIP_IMPL_COVER_FCN("SAL_VSV", SALFCN)				\
  SALFCN(A.ptr, A.stride, 						\
	 const_cast<T*>(&B.value), C.ptr, C.stride, Z.ptr,		\
         Z.stride, len, 0);						\
}

#define VSIP_IMPL_SAL_SVV_AS_VSV(FCN, T, SALFCN)			\
VSIP_IMPL_SAL_INLINE							\
void									\
FCN(Sal_scalar<T> const& A,						\
    Sal_vector<T> const& B,						\
    Sal_vector<T> const& C,						\
    Sal_vector<T> const& Z,						\
    int len )								\
{									\
  VSIP_IMPL_COVER_FCN("SAL_SVV_AS_VSV", SALFCN)				\
  SALFCN(B.ptr, B.stride, 						\
	 const_cast<T*>(&A.value), C.ptr, C.stride, Z.ptr,		\
         Z.stride, len, 0);						\
}

// Z = A*b+C -- Vector scalar multiply and vector add
VSIP_IMPL_SAL_VSV(vma, float,  vsmax)
VSIP_IMPL_SAL_VSV(vma, double, vsmadx)
VSIP_IMPL_SAL_SVV_AS_VSV(vma, float,  vsmax)
VSIP_IMPL_SAL_SVV_AS_VSV(vma, double, vsmadx)

// Z = A*b-C -- Vector scalar multiply and vector subtract
VSIP_IMPL_SAL_VSV(vmsb, float,  vsmsbx)
VSIP_IMPL_SAL_VSV(vmsb, double, vsmsbdx)



#define VSIP_IMPL_SAL_VVS(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void									\
FCN(									\
  Sal_vector<T> const& A,						\
  Sal_vector<T> const& B,						\
  Sal_scalar<T> const& C,						\
  Sal_vector<T> const& Z,						\
  int len )								\
{									\
  VSIP_IMPL_COVER_FCN("SAL_VVS", SALFCN)				\
  SALFCN(A.ptr, A.stride, B.ptr, B.stride,				\
         const_cast<T*>(&C.value),					\
         Z.ptr, Z.stride, len, 0);					\
}

VSIP_IMPL_SAL_VVS(vma, float,  vmsax)

VSIP_IMPL_SAL_VVS(vma, double, vmsadx)

// Z = (A + B)c
VSIP_IMPL_SAL_VVS(vam,  float,  vasmx)

// Z = (A - B)c
VSIP_IMPL_SAL_VVS(vsbm, float,  vsbsmx)



#define VSIP_IMPL_SAL_VSS(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE void						\
FCN(									\
  Sal_vector<T> const& A,						\
  Sal_scalar<T> const& B,						\
  Sal_scalar<T> const& C,						\
  Sal_vector<T> const& Z,						\
  int len )								\
{									\
  VSIP_IMPL_COVER_FCN("SAL_VSS", SALFCN)				\
  SALFCN(A.ptr, A.stride, const_cast<T*>(&B.value),			\
         const_cast<T*>(&C.value), Z.ptr, Z.stride, len, 0);		\
}

VSIP_IMPL_SAL_VSS(vma, float,  vsmsax)

VSIP_IMPL_SAL_VSS(vma, double, vsmsadx)

#if VSIP_IMPL_SAL_HAVE_VTHRX
VSIP_IMPL_VS     (vthresh,  float, vthrx)
#else
VSIP_IMPL_SAL_INLINE void
vthresh(
  Sal_vector<float> const& A,
  Sal_scalar<float> const& B,
  Sal_vector<float> const& Z,
  length_type len)
{
  VSIP_IMPL_COVER_FCN("SAL_VSS", vthrx_psuedo);
  float* a = A.ptr;
  float  b = B.value;
  for (index_type i=0; i<len; ++i)
  {
    Z.ptr[i*A.stride] = (a[i*A.stride] > b) ? a[i*A.stride] : b;
  }
}
#endif
VSIP_IMPL_VS     (vthresh0, float, vthresx)




#define VSIP_IMPL_SAL_CVSV(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE void						\
FCN(									\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& A,			\
  Sal_scalar<complex<T> > const&                 B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& C,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  int len )								\
{									\
  typedef Sal_inter<T>::type inter_type;				\
  									\
  VSIP_IMPL_COVER_FCN("SAL_CVSV", SALFCN)				\
  SALFCN((inter_type*)A.ptr, 2*A.stride,				\
         (inter_type*)&B.value,						\
         (inter_type*)C.ptr, 2*C.stride,				\
         (inter_type*)Z.ptr, 2*Z.stride, len, 0);			\
}

#define VSIP_IMPL_SAL_CSVV_AS_VSV(FCN, T, SALFCN)			\
VSIP_IMPL_SAL_INLINE void						\
FCN(									\
  Sal_scalar<complex<T> > const&                 A,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& C,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  int len )								\
{									\
  typedef Sal_inter<T>::type inter_type;				\
  									\
  VSIP_IMPL_COVER_FCN("SAL_CVSV", SALFCN)				\
  SALFCN((inter_type*)B.ptr, 2*B.stride,				\
         (inter_type*)&A.value,						\
         (inter_type*)C.ptr, 2*C.stride,				\
         (inter_type*)Z.ptr, 2*Z.stride, len, 0);			\
}

VSIP_IMPL_SAL_CVSV(vma, float,  cvsmax)
VSIP_IMPL_SAL_CVSV(vma, double, cvsmadx)

VSIP_IMPL_SAL_CSVV_AS_VSV(vma, float,  cvsmax)
VSIP_IMPL_SAL_CSVV_AS_VSV(vma, double, cvsmadx)



#define VSIP_IMPL_SAL_ZVSV(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE void						\
FCN(									\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& A,			\
  Sal_scalar<complex<T> > const&                 B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& C,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
    int len )								\
{									\
  typedef Sal_split<T>::type split_type;				\
  split_type* cA = (split_type*)&A.ptr;					\
  split_type* cC = (split_type*)&C.ptr;					\
  split_type* cZ = (split_type*)&Z.ptr;					\
  T real = B.value.real();						\
  T imag = B.value.imag();						\
  split_type cB = { &real, &imag };					\
  									\
  VSIP_IMPL_COVER_FCN("SAL_ZVSV", SALFCN)				\
  SALFCN(cA, A.stride, &cB, cC, C.stride, cZ, Z.stride, len, 0);	\
}

#define VSIP_IMPL_SAL_ZSVV_AS_VSV(FCN, T, SALFCN)			\
VSIP_IMPL_SAL_INLINE void						\
FCN(									\
  Sal_scalar<complex<T> > const&                 A,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& C,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
    int len )								\
{									\
  typedef Sal_split<T>::type split_type;				\
  split_type* cB = (split_type*)&B.ptr;					\
  split_type* cC = (split_type*)&C.ptr;					\
  split_type* cZ = (split_type*)&Z.ptr;					\
  T real = A.value.real();						\
  T imag = A.value.imag();						\
  split_type cA = { &real, &imag };					\
  									\
  VSIP_IMPL_COVER_FCN("SAL_ZSVV_AS_VSV", SALFCN)			\
  SALFCN(cB, B.stride, &cA, cC, C.stride, cZ, Z.stride, len, 0);	\
}

VSIP_IMPL_SAL_ZVSV(vma, float,  zvsmax)
VSIP_IMPL_SAL_ZVSV(vma, double, zvsmadx)

VSIP_IMPL_SAL_ZSVV_AS_VSV(vma, float,  zvsmax)
VSIP_IMPL_SAL_ZSVV_AS_VSV(vma, double, zvsmadx)



#define VSIP_IMPL_SAL_VSMUL( ScalarT, T, SAL_T, SALFCN, STRIDE_X ) \
VSIP_IMPL_SAL_INLINE void                                          \
vsmul( T* a, int as,                                               \
       ScalarT* scalar,                                            \
       T* c, int cs,                                               \
       int n )                                                     \
{                                                                  \
  SALFCN( (SAL_T *) a, STRIDE_X * as,                              \
          (SAL_T *) &scalar,                                       \
          (SAL_T *) c, STRIDE_X * cs,                              \
          n, 0 );                                                  \
}

#define VSIP_IMPL_SAL_VSMUL_SPLIT( ScalarT, T, SAL_T, SALFCN, STRIDE_X ) \
VSIP_IMPL_SAL_INLINE void                                                \
vsmul( std::pair<T *, T *> a, int as,                                    \
       ScalarT* scalar,                                                  \
       std::pair<T *, T *> c, int cs,                                    \
       int n )                                                           \
{                                                                        \
  T real = scalar->real();                                               \
  T imag = scalar->imag();                                               \
  SAL_T p_scalar = { &real, &imag };                                     \
                                                                         \
  SALFCN( (SAL_T *) &a, STRIDE_X * as,                                   \
          &p_scalar,                                                     \
          (SAL_T *) &c, STRIDE_X * cs,                                   \
          n, 0 );                                                        \
}

VSIP_IMPL_SAL_VSMUL( float, float, float,                              vsmulx,   1 )
VSIP_IMPL_SAL_VSMUL( double, double, double,                           vsmuldx,  1 )
VSIP_IMPL_SAL_VSMUL( complex<float>, complex<float>,   COMPLEX,        cvcsmlx,  2 )
VSIP_IMPL_SAL_VSMUL( complex<double>, complex<double>, DOUBLE_COMPLEX, cvcsmldx, 2 )
VSIP_IMPL_SAL_VSMUL_SPLIT( complex<float>, float,                        \
                           COMPLEX_SPLIT, zvzsmlx, 1 )
VSIP_IMPL_SAL_VSMUL_SPLIT( complex<double>, double,                      \
                           DOUBLE_COMPLEX_SPLIT, zvzsmldx, 1 )

#undef VSIP_IMPL_SAL_VSMUL
#undef VSIP_IMPL_SAL_VSMUL_SPLIT


/***********************************************************************
  Synthesized functions
***********************************************************************/

#define VSIP_IMPL_CVS_SYN(FCN, T, SALFCN, SCALAR_OP)			\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& A,			\
  Sal_scalar<complex<T> > const& B,					\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_CVS_SYN", SALFCN)				\
									\
  T real = SCALAR_OP B.value.real();					\
  T imag = SCALAR_OP B.value.imag();					\
									\
  SALFCN((T*)A.ptr,   2*A.stride, &real, (T*)Z.ptr,   2*Z.stride, len, 0);\
  SALFCN((T*)A.ptr+1, 2*A.stride, &imag, (T*)Z.ptr+1, 2*Z.stride, len, 0);\
}

#define VSIP_IMPL_CSV_SYN(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_scalar<complex<T> > const&                 A,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_CSV_SYN", SALFCN)				\
									\
  T real = A.value.real();						\
  T imag = A.value.imag();						\
									\
  SALFCN((T*)B.ptr,   2*B.stride, &real, (T*)Z.ptr,   2*Z.stride, len, 0);\
  SALFCN((T*)B.ptr+1, 2*B.stride, &imag, (T*)Z.ptr+1, 2*Z.stride, len, 0);\
}

#define VSIP_IMPL_ZVS_SYN(FCN, T, SALFCN, SCALAR_OP)			\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& A,			\
  Sal_scalar<complex<T> > const& B,					\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("ZVS_SYN", SALFCN)				\
									\
  T real = SCALAR_OP B.value.real();					\
  T imag = SCALAR_OP B.value.imag();					\
									\
  SALFCN(A.ptr.first,  A.stride, &real, Z.ptr.first,  Z.stride, len,0);\
  SALFCN(A.ptr.second, A.stride, &imag, Z.ptr.second, Z.stride, len,0);\
}

#define VSIP_IMPL_ZSV_SYN(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_scalar<complex<T> > const&                 A,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("ZSV_SYN", SALFCN)				\
									\
  T real = A.value.real();						\
  T imag = A.value.imag();						\
									\
  SALFCN(B.ptr.first,  B.stride, &real, Z.ptr.first,  Z.stride, len, 0);\
  SALFCN(B.ptr.second, B.stride, &imag, Z.ptr.second, Z.stride, len, 0);\
}


// complex scalar-vector add

VSIP_IMPL_CVS_SYN(vadd, float, vsaddx, +)
VSIP_IMPL_CSV_SYN(vadd, float, vsaddx)
VSIP_IMPL_CVS_SYN(vsub, float, vsaddx, -)

VSIP_IMPL_CVS_SYN(vadd, double, vsadddx, +)
VSIP_IMPL_CSV_SYN(vadd, double, vsadddx)
VSIP_IMPL_CVS_SYN(vsub, double, vsadddx, -)

VSIP_IMPL_ZVS_SYN(vadd, float, vsaddx, +)
VSIP_IMPL_ZSV_SYN(vadd, float, vsaddx)
VSIP_IMPL_ZVS_SYN(vsub, float, vsaddx, -)

VSIP_IMPL_ZVS_SYN(vadd, double, vsadddx, +)
VSIP_IMPL_ZSV_SYN(vadd, double, vsadddx)
VSIP_IMPL_ZVS_SYN(vsub, double, vsadddx, -)

#undef VSIP_IMPL_CVS_SYN
#undef VSIP_IMPL_CSV_SYN
#undef VSIP_IMPL_ZVS_SYN
#undef VSIP_IMPL_ZSV_SYN



#define VSIP_IMPL_CRVS_ADD_SYN(FCN, T, SALFCN, CPYFCN, SCALAR_OP)	\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& A,			\
  Sal_scalar<T> const&                           B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_CRVS_ADD_SYN", SALFCN)				\
									\
  T real = SCALAR_OP B.value;						\
									\
  SALFCN((T*)A.ptr,   2*A.stride, &real, (T*)Z.ptr,   2*Z.stride, len, 0);\
  CPYFCN((T*)A.ptr+1, 2*A.stride,        (T*)Z.ptr+1, 2*Z.stride, len, 0);\
}

#define VSIP_IMPL_CRSV_ADD_SYN(FCN, T, SALFCN, CPYFCN)			\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_scalar<T> const&                           A,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_CRSV_ADD_SYN", SALFCN)				\
									\
  T real = A.value;							\
									\
  SALFCN((T*)B.ptr, 2*B.stride, &real, (T*)Z.ptr, 2*Z.stride, len, 0);	\
  CPYFCN((T*)B.ptr+1, 2*B.stride, (T*)Z.ptr+1, 2*Z.stride, len, 0);	\
}

#define VSIP_IMPL_CRVS_MUL_SYN(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& A,			\
  Sal_scalar<T> const&                           B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_CRVS_MUL_SYN", SALFCN)			\
									\
  T real = B.value;							\
									\
  if (A.stride == 1 && Z.stride == 1)					\
    SALFCN((T*)A.ptr, A.stride, &real, (T*)Z.ptr, Z.stride, 2*len, 0);	\
  else									\
  {									\
    SALFCN((T*)A.ptr,   2*A.stride, &real, (T*)Z.ptr,   2*Z.stride, len, 0);\
    SALFCN((T*)A.ptr+1, 2*A.stride, &real, (T*)Z.ptr+1, 2*Z.stride, len, 0);\
  }									\
}

#define VSIP_IMPL_CRSV_MUL_SYN(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_scalar<T> const&                           A,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_inter_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("SAL_CRSV_MUL_SYN", SALFCN)			\
									\
  T real = A.value;							\
									\
  if (B.stride == 1 && Z.stride == 1)					\
    SALFCN((T*)B.ptr, B.stride, &real, (T*)Z.ptr, Z.stride, 2*len, 0);	\
  else									\
  {									\
    SALFCN((T*)B.ptr,   2*B.stride, &real, (T*)Z.ptr,   2*Z.stride, len, 0);\
    SALFCN((T*)B.ptr+1, 2*B.stride, &real, (T*)Z.ptr+1, 2*Z.stride, len, 0);\
  }									\
}


#define VSIP_IMPL_ZRVS_ADD_SYN(FCN, T, SALFCN, CPYFCN, SCALAR_OP)	\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& A,			\
  Sal_scalar<T> const&                           B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("ZRVS_ADD_SYN", SALFCN)				\
									\
  T real = SCALAR_OP B.value;						\
									\
  SALFCN(A.ptr.first,  A.stride, &real, Z.ptr.first,  Z.stride, len, 0);\
  CPYFCN(A.ptr.second, A.stride,        Z.ptr.second, Z.stride, len, 0);\
}

#define VSIP_IMPL_ZRSV_ADD_SYN(FCN, T, SALFCN, CPYFCN)			\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_scalar<T> const&                           A,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("ZRSV_ADD_SYN", SALFCN)				\
									\
  T real = A.value;							\
									\
  SALFCN(B.ptr.first,  B.stride, &real, Z.ptr.first,  Z.stride, len, 0);\
  CPYFCN(B.ptr.second, B.stride,        Z.ptr.second, Z.stride, len, 0);\
}

#define VSIP_IMPL_ZRVS_MUL_SYN(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& A,			\
  Sal_scalar<T> const&                           B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("ZRVS_MUL_SYN", SALFCN)				\
									\
  T real = B.value;							\
									\
  SALFCN(A.ptr.first,  A.stride, &real, Z.ptr.first,  Z.stride, len, 0);\
  SALFCN(A.ptr.second, A.stride, &real, Z.ptr.second, Z.stride, len, 0);\
}

#define VSIP_IMPL_ZRSV_MUL_SYN(FCN, T, SALFCN)				\
VSIP_IMPL_SAL_INLINE							\
void FCN(								\
  Sal_scalar<T> const&                           A,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& B,			\
  Sal_vector<complex<T>, Cmplx_split_fmt> const& Z,			\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("ZRSV_MUL_SYN", SALFCN)				\
									\
  T real = A.value;							\
									\
  SALFCN(B.ptr.first,  B.stride, &real, Z.ptr.first,  Z.stride, len, 0);\
  SALFCN(B.ptr.second, B.stride, &real, Z.ptr.second, Z.stride, len, 0);\
}

// complex-real scalar-vector add

VSIP_IMPL_CRVS_ADD_SYN(vadd, float, vsaddx, vmovx, +)
VSIP_IMPL_CRSV_ADD_SYN(vadd, float, vsaddx, vmovx)
VSIP_IMPL_CRVS_ADD_SYN(vsub, float, vsaddx, vmovx, -)
VSIP_IMPL_CRVS_MUL_SYN(vmul, float, vsmulx)
VSIP_IMPL_CRSV_MUL_SYN(vmul, float, vsmulx)

VSIP_IMPL_CRVS_ADD_SYN(vadd, double, vsadddx, vmovdx, +)
VSIP_IMPL_CRSV_ADD_SYN(vadd, double, vsadddx, vmovdx)
VSIP_IMPL_CRVS_ADD_SYN(vsub, double, vsadddx, vmovdx, -)
VSIP_IMPL_CRVS_MUL_SYN(vmul, double, vsmuldx)
VSIP_IMPL_CRSV_MUL_SYN(vmul, double, vsmuldx)

VSIP_IMPL_ZRVS_ADD_SYN(vadd, float, vsaddx, vmovx, +)
VSIP_IMPL_ZRSV_ADD_SYN(vadd, float, vsaddx, vmovx)
VSIP_IMPL_ZRVS_ADD_SYN(vsub, float, vsaddx, vmovx, -)
VSIP_IMPL_ZRVS_MUL_SYN(vmul, float, vsmulx)
VSIP_IMPL_ZRSV_MUL_SYN(vmul, float, vsmulx)

VSIP_IMPL_ZRVS_ADD_SYN(vadd, double, vsadddx, vmovdx, +)
VSIP_IMPL_ZRSV_ADD_SYN(vadd, double, vsadddx, vmovdx)
VSIP_IMPL_ZRVS_ADD_SYN(vsub, double, vsadddx, vmovdx, -)
VSIP_IMPL_ZRVS_MUL_SYN(vmul, double, vsmuldx)
VSIP_IMPL_ZRSV_MUL_SYN(vmul, double, vsmuldx)

#undef VSIP_IMPL_SAL_INLINE

} // namespace vsip::impl::sal
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_SAL_ELEMENTWISE_HPP
