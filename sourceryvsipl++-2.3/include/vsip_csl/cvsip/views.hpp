/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    vsip_csl/cvsip/support.cpp
    @author  Stefan Seefeld
    @date    2008-05-02
    @brief   block / view support
*/

#ifndef vsip_csl_cvsip_views_hpp_
#define vsip_csl_cvsip_views_hpp_

#include <vsip.h>
#include <vsip/core/view_cast.hpp>

#define CVSIP(name) CVSIP_(name, TYPE)
#define CVSIP_(name, TYPE) CVSIP__(name, TYPE)
#define CVSIP__(name, TYPE) vsip_##name##_##TYPE

// The wrapper headers are not header-guarded, and so we have
// to be very careful about which files to include where.

// complex blocks / views are defined for d and f. All other types
// only have scalar blocks / views.

#define TYPE f
#include "cview_decl.hpp"
#undef TYPE
#define TYPE d
#include "cview_decl.hpp"
#undef TYPE
#define TYPE i
#include "view_decl.hpp"
#undef TYPE
#define TYPE si
#include "view_decl.hpp"
#undef TYPE
#define TYPE uc
#include "view_decl.hpp"
#undef TYPE
#define TYPE vi
#include "view_decl.hpp"
#undef TYPE
#define TYPE mi
#include "view_decl.hpp"
#undef TYPE
#define TYPE bl
#include "view_decl.hpp"
#undef TYPE

#undef CVSIP__
#undef CVSIP_
#undef CVSIP

#endif
