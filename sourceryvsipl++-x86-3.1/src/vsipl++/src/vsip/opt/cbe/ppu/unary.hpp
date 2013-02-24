/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef VSIP_OPT_CBE_PPU_UNARY_HPP
#define VSIP_OPT_CBE_PPU_UNARY_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/core/adjust_layout.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>

namespace vsip
{
namespace impl
{
namespace cbe
{
void vsqrt(float const *A, float *R, length_type len);
void vatan(float const *A, float *R, length_type len);

void vlog(float const *A, float *R, length_type len);
void vlog(complex<float> const *A, complex<float> *R, length_type len);
void vlog(std::pair<float const *,float const *> const &A, std::pair<float*,float*> const &R,
	    length_type len);

void vlog10(float const *A, float *R, length_type len);
void vlog10(complex<float> const *A, complex<float> *R, length_type len);
void vlog10(std::pair<float const *,float const *> const &A, std::pair<float*,float*> const &R,
	    length_type len);

void vcos(float const *A, float *R, length_type len);
void vcos(complex<float> const *A, complex<float> *R, length_type len);
void vcos(std::pair<float const *,float const *> const &A, std::pair<float*,float*> const &R,
	    length_type len);

void vsin(float const *A, float *R, length_type len);
void vsin(complex<float> const *A, complex<float> *R, length_type len);
void vsin(std::pair<float const *,float const *> const &A, std::pair<float*,float*> const &R,
	    length_type len);

void vminus(float const *A, float *R, length_type len);
void vminus(complex<float> const *A, complex<float> *R, length_type len);
void vminus(std::pair<float const *,float const *> const &A, std::pair<float*,float*> const &R,
	    length_type len);

void vsq(float const *A, float *R, length_type len);
void vsq(complex<float> const *A, complex<float> *R, length_type len);
void vsq(std::pair<float const *,float const *> const &A, std::pair<float*,float*> const &R,
	 length_type len);

void vmag(float const *A, float *R, length_type len);
void vmag(complex<float> const *A, float *R, length_type len);
void vmag(std::pair<float const *,float const *> const &A, float *R, length_type len);

void vmagsq(float const *A, float *R, length_type len);
void vmagsq(complex<float> const *A, float *R, length_type len);
void vmagsq(std::pair<float const *,float const *> const &A, float *R, length_type len);

void vconj(complex<float> const *A, complex<float> *R, length_type len);
void vconj(std::pair<float const *,float const *> const &A, std::pair<float*,float*> const &R,
	   length_type len);

template <template <typename> class Operator,
	  typename A, bool AIsSplit,
	  typename R, bool CIsSplit>
struct Is_un_op_supported { static bool const value = false;};

#define UN_OP_IS_SUPPORTED(OP, LTYPE, RTYPE)			\
  template <bool IsSplit>					\
  struct Is_un_op_supported<expr::op::OP,			\
			    LTYPE, IsSplit, RTYPE, IsSplit>	\
  {								\
    static bool const value = true;				\
  };

#define UN_MAG_OP_IS_SUPPORTED(OP, LTYPE, RTYPE)		\
  template <bool IsSplit>					\
  struct Is_un_op_supported<expr::op::OP,			\
			    LTYPE, false, RTYPE, IsSplit>	\
  {								\
    static bool const value = true;				\
  };

UN_OP_IS_SUPPORTED(Sqrt,  float, float)
UN_OP_IS_SUPPORTED(Atan,  float, float)
UN_OP_IS_SUPPORTED(Log,   float, float)
UN_OP_IS_SUPPORTED(Log,   complex<float>, complex<float>)
UN_OP_IS_SUPPORTED(Log10, float, float)
UN_OP_IS_SUPPORTED(Log10, complex<float>, complex<float>)
UN_OP_IS_SUPPORTED(Cos,   float, float)
UN_OP_IS_SUPPORTED(Cos,   complex<float>, complex<float>)
UN_OP_IS_SUPPORTED(Sin,   float, float)
UN_OP_IS_SUPPORTED(Sin,   complex<float>, complex<float>)
UN_OP_IS_SUPPORTED(Minus, float, float)
UN_OP_IS_SUPPORTED(Minus, complex<float>, complex<float>)
UN_OP_IS_SUPPORTED(Sq,    float, float)
UN_OP_IS_SUPPORTED(Sq,    complex<float>, complex<float>)
UN_OP_IS_SUPPORTED(Mag,   float, float)
UN_OP_IS_SUPPORTED(Magsq, float, float)
UN_OP_IS_SUPPORTED(Conj,  complex<float>, complex<float>)
UN_MAG_OP_IS_SUPPORTED(Mag,   float, complex<float>)
UN_MAG_OP_IS_SUPPORTED(Magsq, float, complex<float>)

#undef UN_OP_IS_SUPPORTED

#define SIZE_THRESHOLD(OP, ATYPE, VAL)		\
template <>					\
struct Size_threshold<expr::op::OP<ATYPE > >	\
{						\
  static length_type const value = VAL;		\
};

SIZE_THRESHOLD(Conj, complex<float>, 8192)
SIZE_THRESHOLD(Magsq,         float, 4096)
SIZE_THRESHOLD(Magsq,complex<float>, 8192)
SIZE_THRESHOLD(Mag,           float, 4096)
SIZE_THRESHOLD(Mag,  complex<float>, 8192)
SIZE_THRESHOLD(Minus,         float, 4096)
SIZE_THRESHOLD(Minus,complex<float>, 8192)
SIZE_THRESHOLD(Sq,            float, 4096)
SIZE_THRESHOLD(Sq,   complex<float>, 4096)
SIZE_THRESHOLD(Sqrt,          float,  256)
SIZE_THRESHOLD(Atan,          float,  256)
SIZE_THRESHOLD(Cos,           float,  256)
SIZE_THRESHOLD(Cos,   complex<float>,  16)
SIZE_THRESHOLD(Sin,           float,  256)
SIZE_THRESHOLD(Sin,   complex<float>,  16)
SIZE_THRESHOLD(Log,           float,   64)
SIZE_THRESHOLD(Log,   complex<float>,  16)
SIZE_THRESHOLD(Log10,         float,   32)
SIZE_THRESHOLD(Log10, complex<float>,   8)

#undef SIZE_THRESHOLD

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip

#endif
