/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef VSIP_OPT_CBE_PPU_TERNARY_HPP
#define VSIP_OPT_CBE_PPU_TERNARY_HPP

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

/// A * B + C
void vma(float const* A, float const* B, float const* C, float* R, length_type len);
void vma(complex<float> const* A, complex<float> const* B, complex<float> const* C,
	 complex<float>* R,
	 length_type len);
void vma(std::pair<float const *, float const *> const &A,
	 std::pair<float const *, float const *> const &B,
	 std::pair<float const *, float const *> const &C,
	 std::pair<float*, float*> const &R,
	 length_type len);

// (A + B) * C
void vam(float const* A, float const* B, float const* C, float* R, length_type len);
void vam(complex<float> const* A, complex<float> const* B, complex<float> const* C,
	 complex<float>* R,
	 length_type len);
void vam(std::pair<float const *, float const *> const &A,
	 std::pair<float const *, float const *> const &B,
	 std::pair<float const *, float const *> const &C,
	 std::pair<float*, float*> const &R,
	 length_type len);

template <template <typename, typename, typename> class Operator,
	  typename A, bool AIsSplit,
	  typename B, bool BIsSplit,
	  typename C, bool CIsSplit,
	  typename R, bool RIsSplit>
struct Is_tern_op_supported { static bool const value = false;};

template <template <typename, typename, typename> class O>
struct Is_tern_op_supported<O,
			    float, false,
			    float, false,
			    float, false,
			    float, false>
{
  static bool const value = true;
};

template <template <typename, typename, typename> class O>
struct Is_tern_op_supported<O,
			    complex<float>, false,
			    complex<float>, false,
			    complex<float>, false,
			    complex<float>, false>
{
  static bool const value = true;
};

template <template <typename, typename, typename> class O>
struct Is_tern_op_supported<O,
			    complex<float>, true,
			    complex<float>, true,
			    complex<float>, true,
			    complex<float>, true>
{
  static bool const value = true;
};

#define SIZE_THRESHOLD(OP, ATYPE, BTYPE, CTYPE, VAL)          \
template <>                                                   \
struct Size_threshold<expr::op::OP<ATYPE, BTYPE,CTYPE > >     \
{							      \
  static length_type const value = VAL;			      \
};

SIZE_THRESHOLD(Am,           float,          float, float,          4096)
SIZE_THRESHOLD(Am,  complex<float>, complex<float>, complex<float>, 2048)
SIZE_THRESHOLD(Ma,           float,          float, float,          4096)
SIZE_THRESHOLD(Ma,  complex<float>, complex<float>, complex<float>, 2048)

#undef SIZE_THRESHOLD

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip

#endif
