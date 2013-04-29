/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef VSIP_OPT_CBE_PPU_BINARY_HPP
#define VSIP_OPT_CBE_PPU_BINARY_HPP

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
void vmul(float const* A, float const* B, float* R, length_type len);
void vmul(complex<float> const* A, complex<float> const* B, complex<float>* R,
	  length_type len);
void vmul(std::pair<float const *, float const *> const& A,
	  std::pair<float const *, float const *> const& B,
	  std::pair<float*, float*> const& R,
	  length_type len);

void vadd(float const* A, float const* B, float* R, length_type len);
void vadd(complex<float> const* A, complex<float> const* B, complex<float>* R,
	  length_type len);
void vadd(std::pair<float const *, float const *> const& A,
	  std::pair<float const *, float const *> const& B,
	  std::pair<float*, float*> const& R,
	  length_type len);

void vsub(float const* A, float const* B, float* R, length_type len);
void vsub(complex<float> const* A, complex<float> const* B, complex<float>* R,
	  length_type len);
void vsub(std::pair<float const *, float const *> const& A,
	  std::pair<float const *, float const *> const& B,
	  std::pair<float*, float*> const& R,
	  length_type len);

void vdiv(float const* A, float const* B, float* R, length_type len);
void vdiv(complex<float> const* A, complex<float> const* B, complex<float>* R,
	  length_type len);
void vdiv(std::pair<float const *, float const *> const& A,
          std::pair<float const *, float const *> const& B,
          std::pair<float*, float*> const& R,
          length_type              len);

void vatan2(float const* A, float const* B, float* R, length_type len);

void vhypot(float const* A, float const* B, float* R, length_type len);

template <template <typename, typename> class Operator,
	  typename A, bool AIsSplit,
	  typename B, bool BIsSplit,
	  typename C, bool CIsSplit>
struct Is_bin_op_supported { static bool const value = false;};

template <template <typename, typename> class O>
struct Is_bin_op_supported<O, complex<float>, false,
			   complex<float>, false,
			   complex<float>, false>
{
  static bool const value = true;
};

template <template <typename, typename> class O>
struct Is_bin_op_supported<O, complex<float>, true,
			   complex<float>, true,
			   complex<float>, true>
{
  static bool const value = true;
};

template <template <typename, typename> class O>
struct Is_bin_op_supported<O, float, false, float, false, float, false>
{
  static bool const value = true;
};

#define SIZE_THRESHOLD(OP, ATYPE, BTYPE, VAL)		\
template <>						\
struct Size_threshold<expr::op::OP<ATYPE, BTYPE > >	\
{							\
  static length_type const value = VAL;			\
};

SIZE_THRESHOLD(Add,           float,          float, 4096)
SIZE_THRESHOLD(Add,  complex<float>, complex<float>, 4096)
SIZE_THRESHOLD(Div,           float,          float,  256)
SIZE_THRESHOLD(Div,  complex<float>, complex<float>,  128)
SIZE_THRESHOLD(Mult,          float,          float, 4096)
SIZE_THRESHOLD(Mult, complex<float>, complex<float>, 2048)
SIZE_THRESHOLD(Sub,           float,          float, 4096)
SIZE_THRESHOLD(Sub,  complex<float>, complex<float>, 4096)
SIZE_THRESHOLD(Atan2,         float,          float,    0)
SIZE_THRESHOLD(Hypot,         float,          float,    0)

#undef SIZE_THRESHOLD

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip

#endif
