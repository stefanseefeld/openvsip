/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/complex_decl.hpp
    @author  Jules Bergmann
    @date    2005-07-25
    @brief   VSIPL++ Library: Declarations for [complex] complex numbers.

    This file declares the vsip::complex and vsip::polar types (which
    are synonyms for std::complex and std::polar), and related
    functions as specified in the [complex] section of the VSIPL++
    specifiction.
*/

#ifndef VSIP_CORE_COMPLEX_DECL_HPP
#define VSIP_CORE_COMPLEX_DECL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <complex>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

// ISO 14882 [lib.complex.numbers] incorporated by reference
//
// [lib.complex.ops]
//   - operator+	(complex-decl.hpp)
//   - operator-	(complex-decl.hpp)
//   - operator*	(complex-decl.hpp)
//   - operator/	(complex-decl.hpp)
//   - operator!=	(complex-decl.hpp)
//   - operator==	(complex-decl.hpp)
//   - operator<<	(complex-decl.hpp)
//   - operator>>	(complex-decl.hpp)
//
// [lib.complex.value]
//   - real		(fns_elementwise.hpp)
//   - imag		(fns_elementwise.hpp)
//   - abs		(complex-decl.hpp)
//   - arg		(fns_elementwise.hpp)
//   - norm		(complex-decl.hpp)
//   - conj		(fns_elementwise.hpp)
//   - polar		(complex-decl.hpp)
//
// [lib.complex.transcendentals]
//   - cos		(fns_elementwise.hpp)
//   - cosh		(fns_elementwise.hpp)
//   - exp		(fns_elementwise.hpp)
//   - log		(fns_elementwise.hpp)
//   - log10		(fns_elementwise.hpp)
//   - pow		(fns_elementwise.hpp) (NOTE: all 4 cases handled)
//   - sin		(fns_elementwise.hpp)
//   - sinh		(fns_elementwise.hpp)
//   - sqrt		(fns_elementwise.hpp)
//   - tan		(fns_elementwise.hpp)
//   - tanh		(fns_elementwise.hpp)


using std::complex;

// [operators]
using std::operator+;
using std::operator-;
using std::operator*;
using std::operator/;
using std::operator!=;
using std::operator==;
using std::operator<<;
using std::operator>>;


// [value]
using std::abs;
using std::norm;
using std::polar;

} // namespace vsip

namespace vsip_csl
{
using vsip::complex;
}

#endif // VSIP_CORE_COMPLEX_DECL_HPP
