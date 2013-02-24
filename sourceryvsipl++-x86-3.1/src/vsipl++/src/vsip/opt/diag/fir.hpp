/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/diag/fir.hpp
    @author  Jules Bergmann
    @date    2008-03-13
    @brief   VSIPL++ Library: Diagnostics for Fir.
*/

#ifndef VSIP_OPT_DIAG_FIR_HPP
#define VSIP_OPT_DIAG_FIR_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <iomanip>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{

namespace diag_detail
{

struct Diagnose_fir
{
  template <typename FirT>
  static void diag(std::string const& name, FirT const& fir)
  {
    using std::cout;
    using std::endl;

    cout << "diagnose_fir(" << name << ")" << endl
	 << "  be : " << fir.backend_.get()->name() << endl;
  }
};

} // namespace vsip::impl::diag_detail



template <typename FirT>
void
diagnose_fir(std::string const& name, FirT const& fir)
{
  diag_detail::Diagnose_fir::diag<FirT>(name, fir);
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_DIAG_FIR_HPP
