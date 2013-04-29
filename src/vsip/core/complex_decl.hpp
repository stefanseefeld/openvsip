//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

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
