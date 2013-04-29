/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    vsip_csl/cvsip/view_decl.hpp
    @author  Stefan Seefeld
    @date    2008-05-21
    @brief   View types
*/

#include <vsip_csl/cvsip/view.hpp>
#include <vsip_csl/cvsip/type_conversion.hpp>
#include <vsip/dense.hpp>

extern "C"
{
/***********************************************************************
   block
************************************************************************/
struct CVSIP(blockattributes)
{
  typedef CVSIP(scalar) value_type;
  typedef vsip_csl::cvsip::type_conversion<CVSIP(block)> converter;
  typedef converter::impl_type impl_value_type;
  typedef vsip::Dense<1, impl_value_type> storage_type;

  CVSIP(blockattributes)(size_t l, vsip_memory_hint)
    : storage(l), derived(false) {} 
  CVSIP(blockattributes)(value_type *data, size_t l)
    : storage(l, converter::in(data)), derived(true) {} 
  CVSIP(blockattributes)(value_type *data, size_t l, vsip_memory_hint)
    : storage(l, converter::in(data)), derived(false) {} 
    
  value_type *rebind(value_type *data)
  {
    value_type *old = find();
    if (derived || storage.admitted()) return 0;
    storage.rebind(converter::in(data));
    return old;
  }
  // To be used by the complex parent block
  // to push new data into the derived block.
  void rebind_derived(value_type *data)
  {
    // The calls to release and admit are here
    // to make sure the 'admitted' flag is set
    // correctly. As the new data comes directly
    // from the storage's (Dense's) internal
    // data, this doesn't involve any copying.
    storage.release(false);
    storage.rebind(converter::in(data));
    storage.admit(false);
  }
  value_type *find()
  {
    impl_value_type *data;
    if (derived) return 0;
    storage.find(data);
    return converter::out(data);
  }
  int admit(bool update) 
  {
    if (!find()) return -1;
    storage.admit(update);
    return 0;
  }
  value_type *release(bool update) 
  {
    if (derived) return 0;
    impl_value_type *data;
    storage.release(update, data);
    return converter::out(data);
  }

  storage_type storage;
  bool         derived;
};

/***********************************************************************
   vector
************************************************************************/
struct CVSIP(vviewattributes) : 
  vsip_csl::cvsip::View<1, CVSIP(scalar), CVSIP(block)>
{
  CVSIP(vviewattributes)(vsip_length l, vsip_memory_hint h)
    : vsip_csl::cvsip::View<1, CVSIP(scalar), CVSIP(block)>(l, h) {}
  CVSIP(vviewattributes)(cblock_type *b, vsip_offset o, vsip_stride s, vsip_length l,
                         bool derived)
    : vsip_csl::cvsip::View<1, CVSIP(scalar), CVSIP(block)>(b, o, s, l, derived) {}
};

/***********************************************************************
   matrix
************************************************************************/
struct CVSIP(mviewattributes) : 
  vsip_csl::cvsip::View<2, CVSIP(scalar), CVSIP(block)>
{
  CVSIP(mviewattributes)(vsip_length r, vsip_length c, vsip_major o, vsip_memory_hint h)
    : vsip_csl::cvsip::View<2, CVSIP(scalar), CVSIP(block)>(r, c, o, h) {}
  CVSIP(mviewattributes)(cblock_type *b, vsip_offset o,
                         vsip_stride rs, vsip_length rl,
                         vsip_stride cs, vsip_length cl,
                         bool derived)
    : vsip_csl::cvsip::View<2, CVSIP(scalar), CVSIP(block)>(b, o, rs, rl, cs, cl,
                                                            derived) {}
};

/***********************************************************************
   tensor
************************************************************************/
struct CVSIP(tviewattributes) : 
  vsip_csl::cvsip::View<3, CVSIP(scalar), CVSIP(block)>
{
  CVSIP(tviewattributes)(vsip_length p, vsip_length m, vsip_length n, vsip_tmajor o, vsip_memory_hint h)
    : vsip_csl::cvsip::View<3, CVSIP(scalar), CVSIP(block)>(p, m, n, o, h) {}
  CVSIP(tviewattributes)(cblock_type *b, vsip_offset o,
                         vsip_stride zs, vsip_length zl,
                         vsip_stride ys, vsip_length yl,
                         vsip_stride xs, vsip_length xl,
                         bool derived)
    : vsip_csl::cvsip::View<3, CVSIP(scalar), CVSIP(block)>(b, o,
                                                            zs, zl, ys, yl, xs, xl,
                                                            derived) {}
};

} // extern "C"

namespace vsip_csl
{
namespace cvsip
{

inline CVSIP(vview)::view_type extract_vpp_view(CVSIP(vview) *v) { return v->view;}
inline CVSIP(mview)::view_type extract_vpp_view(CVSIP(mview) *m) { return m->view;}
inline CVSIP(tview)::view_type extract_vpp_view(CVSIP(tview) *t) { return t->view;}

} // namespace vsip_csl::cvsip
} // namespace vsip_csl
