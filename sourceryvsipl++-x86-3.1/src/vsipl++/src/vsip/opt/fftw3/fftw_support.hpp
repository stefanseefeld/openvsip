/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/fftw3/fftw_support.hpp
    @author  Assem Salama
    @date    2007-04-25
    @brief   VSIPL++ Library: File that has overloaded create functions for
                              fftw

*/
#ifndef VSIP_OPT_FFTW3_FFTW_SUPPORT_HPP
#define VSIP_OPT_FFTW3_FFTW_SUPPORT_HPP

namespace vsip
{
namespace impl
{
namespace fftw3
{

#define DCL_FFTW_PLAN_FUNC_C2C(T, fT) \
fT##_plan create_fftw_plan(int dim, int *sz, \
                      std::complex<T>* ptr1, std::complex<T>* ptr2,\
                      int exp, int flags) \
{ return fT##_plan_dft(dim,sz,reinterpret_cast<fT##_complex*>(ptr1), \
                     reinterpret_cast<fT##_complex*>(ptr2), exp, flags); \
} \
\
fT##_plan create_fftw_plan(int dim, fT##_iodim *iodim, \
                      std::pair<T*,T*> ptr1, std::pair<T*,T*> ptr2,\
                      int flags) \
{ return fT##_plan_guru_split_dft(dim,iodim,0,NULL, \
                            ptr1.first,ptr1.second,ptr2.first,ptr2.second, \
                            flags); \
}

#define DCL_FFTW_PLAN_FUNC_R2C(T, fT) \
fT##_plan create_fftw_plan(int dim, int *sz, \
                      T* ptr1, std::complex<T>* ptr2,\
                      int flags) \
{ return fT##_plan_dft_r2c(dim,sz,ptr1, \
                     reinterpret_cast<fT##_complex*>(ptr2), flags); \
} \
\
fT##_plan create_fftw_plan(int dim, fT##_iodim *iodim, \
                      T* ptr1, std::pair<T*,T*> ptr2,\
                      int flags) \
{ return fT##_plan_guru_split_dft_r2c(dim,iodim,0,NULL, \
                            ptr1,ptr2.first,ptr2.second, \
                            flags); \
}

#define DCL_FFTW_PLAN_FUNC_C2R(T, fT) \
fT##_plan create_fftw_plan(int dim, int *sz, \
                      std::complex<T>* ptr1, T* ptr2,\
                      int flags) \
{ return fT##_plan_dft_c2r(dim,sz,reinterpret_cast<fT##_complex*>(ptr1), \
                     ptr2, flags); \
} \
\
fT##_plan create_fftw_plan(int dim, fT##_iodim *iodim, \
                      std::pair<T*,T*> ptr1, T* ptr2,\
                      int flags) \
{ return fT##_plan_guru_split_dft_c2r(dim,iodim,0,NULL, \
                            ptr1.first,ptr1.second,ptr2, \
                            flags); \
}

#define DCL_FFTW_PLANS(T, fT) \
  DCL_FFTW_PLAN_FUNC_C2C(T, fT) \
  DCL_FFTW_PLAN_FUNC_R2C(T, fT) \
  DCL_FFTW_PLAN_FUNC_C2R(T, fT)


#ifdef VSIP_IMPL_FFTW3_HAVE_FLOAT
  DCL_FFTW_PLANS(float, fftwf)
#endif
#ifdef VSIP_IMPL_FFTW3_HAVE_DOUBLE
  DCL_FFTW_PLANS(double, fftw)
#endif
#ifdef VSIP_IMPL_FFTW3_HAVE_LONG_DOUBLE
  DCL_FFTW_PLANS(long double, fftwl)
#endif

} // namespace vsip::impl::fftw3
} // namespace vsip::impl
} // namespace vsip

#endif
