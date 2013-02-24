/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/cml/traits.hpp
    @author  Don McCoy
    @date    2008-05-07
    @brief   VSIPL++ Library: Traits for CML evaluators.
*/

#ifndef VSIP_OPT_CBE_CML_TRAITS_HPP
#define VSIP_OPT_CBE_CML_TRAITS_HPP


namespace vsip
{
namespace impl
{
namespace cml
{

// At present, this traits class helps determine whether or not
// CML supports a given block's value_type simply by checking
// whether or not the underlying scalar type is a single-precision 
// floating point type.  This makes it valid for scalar floats,
// or complex floats (regardless of the layout being split or 
// interleaved).
template <typename BlockT>
struct Cml_supports_block
{
private:
  typedef typename BlockT::value_type value_type;

public:
  static bool const valid =
    is_same<typename scalar_of<value_type>::type, float>::value;
};


} // namespace vsip::impl::cml
} // namespace vsip::impl
} // namespace vsip


#endif // VSIP_OPT_CBE_CML_TRAITS_HPP
