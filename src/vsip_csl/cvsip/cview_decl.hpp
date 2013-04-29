/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/

/// Description
///   Generic View API

#include <vsip_csl/cvsip/view.hpp>
#include <vsip_csl/cvsip/type_conversion.hpp>
#include <vsip_csl/cvsip/view_decl.hpp>

extern "C"
{
/***********************************************************************
   complex block
************************************************************************/
struct CVSIP(cblockattributes)
{
  typedef CVSIP(scalar) scalar_type;
  typedef CVSIP(cscalar) value_type;
  typedef vsip_csl::cvsip::type_conversion<CVSIP(cblock)> converter;
  typedef converter::impl_type impl_value_type;
  typedef vsip::Dense<1, impl_value_type> storage_type;

  CVSIP(cblockattributes)(size_t l, vsip_memory_hint)
    : storage(l), real(0), imag(0) {} 
  CVSIP(cblockattributes)(scalar_type *data, size_t l, vsip_memory_hint)
    : storage(l, data), real(0), imag(0) {} 
  CVSIP(cblockattributes)(scalar_type *re, scalar_type *im,
                          size_t l, vsip_memory_hint)
    : storage(l, re, im), real(0), imag(0) {} 
  ~CVSIP(cblockattributes)()
  {
    /* From the C-VSIPL spec:
       The derived block object cannot be destroyed or released.
       The parent complex block object may be released (if it is
       bound to user data). Destroying the complex block is the only way to
       free the memory associated with the derived block object.
    */
    delete real;
    delete imag;
  }
  void rebind(scalar_type *re, scalar_type *im,
              scalar_type *&old_re, scalar_type *&old_im)
  {
    if (storage.admitted()) old_re = old_im = 0;
    else
    {
      find(old_re, old_im);
      if (im)
        storage.rebind(re, im);
      else
        storage.rebind(re);

      // Make sure the change properly propagates
      // to any derived block.
      storage_type::ptr_type data = storage.ptr();
#if VSIP_IMPL_PREFER_SPLIT_COMPLEX
      CVSIP(scalar) *real_data = data.first;
      CVSIP(scalar) *imag_data = data.second;
#else
      CVSIP(scalar) *real_data = reinterpret_cast<CVSIP(scalar)*>(data);
      CVSIP(scalar) *imag_data = reinterpret_cast<CVSIP(scalar)*>(data) + 1;
#endif
      if (real) real->rebind_derived(real_data);
      if (imag) imag->rebind_derived(imag_data);
    }
  }
  void find(scalar_type *&r, scalar_type *&i)
  {
    if (storage.user_storage() == vsip::split_format)
      storage.find(r, i);
    else if (storage.user_storage() == vsip::interleaved_format)
    {
      storage.find(r);
      i = 0;
    }
    else
      r = i = 0;
  }
  int admit(bool update)
  {
    scalar_type *r, *i;
    find(r, i);
    if (!r) return -1;
    storage.admit(update);
    return 0;
  }
  void release(bool update, scalar_type *&r, scalar_type *&i)
  {
    if (storage.user_storage() == vsip::split_format)
      storage.release(update, r, i);
    else if (storage.user_storage() == vsip::interleaved_format)
    {
      storage.release(update, r);
      i = 0;
    }
    else
      r = i = 0;
  }
  CVSIP(block) *get_real()
  {
    if (!real)
    {
      storage_type::ptr_type data = storage.ptr();
#if VSIP_IMPL_PREFER_SPLIT_COMPLEX
      CVSIP(scalar) *scalar_data = data.first;
      size_t const size = storage.size();
#else
      CVSIP(scalar) *scalar_data = reinterpret_cast<CVSIP(scalar)*>(data);
      size_t const size = 2 * storage.size();
#endif
      real = new CVSIP(block)(scalar_data, size);
    }
    return real;
  }

  CVSIP(block) *get_imag()
  {
    if (!imag)
    {
      storage_type::ptr_type data = storage.ptr();
#if VSIP_IMPL_PREFER_SPLIT_COMPLEX
      CVSIP(scalar) *scalar_data = data.second;
      size_t const size = storage.size();
#else
      CVSIP(scalar) *scalar_data = reinterpret_cast<CVSIP(scalar)*>(data) + 1;
      size_t const size = 2 * storage.size();
#endif
      imag = new CVSIP(block)(scalar_data, size);
    }
    return imag;
  }

  storage_type storage;
  CVSIP(block) *real;
  CVSIP(block) *imag;  
};

/***********************************************************************
   complex vector
************************************************************************/
struct CVSIP(cvviewattributes) : 
  vsip_csl::cvsip::View<1, CVSIP(cscalar), CVSIP(cblock)>
{
  CVSIP(cvviewattributes)(vsip_length l, vsip_memory_hint h)
    : vsip_csl::cvsip::View<1, CVSIP(cscalar), CVSIP(cblock)>(l, h) {}
  CVSIP(cvviewattributes)(cblock_type *b, vsip_offset o, vsip_stride s, vsip_length l, bool)
    : vsip_csl::cvsip::View<1, CVSIP(cscalar), CVSIP(cblock)>(b, o, s, l, false) {}

  CVSIP(vview) *
  realview() const
  {
    return new CVSIP(vview)(cblock->get_real(),
                            block.offset(), block.stride(), block.size(),
                            true);
  }

  CVSIP(vview) *
  imagview() const
  {
    return new CVSIP(vview)(cblock->get_imag(),
                            block.offset(), block.stride(), block.size(),
                            true);
  }
};

/***********************************************************************
   complex matrix
************************************************************************/
struct CVSIP(cmviewattributes) : 
  vsip_csl::cvsip::View<2, CVSIP(cscalar), CVSIP(cblock)>
{
  CVSIP(cmviewattributes)(vsip_length r, vsip_length c, vsip_major o, vsip_memory_hint h)
    : vsip_csl::cvsip::View<2, CVSIP(cscalar), CVSIP(cblock)>(r, c, o, h) {}
  CVSIP(cmviewattributes)(cblock_type *b, vsip_offset o,
                          vsip_stride rs, vsip_length rl,
                          vsip_stride cs, vsip_length cl,
                          bool)
    : vsip_csl::cvsip::View<2, CVSIP(cscalar), CVSIP(cblock)>(b, o, rs, rl, cs, cl,
                                                              false) {}

  CVSIP(mview) *
  realview() const
  {
    return new CVSIP(mview)(cblock->get_real(),
                            block.offset(),
                            block.col_stride(), block.rows(),
                            block.row_stride(), block.cols(),
                            true);
  }

  CVSIP(mview) *
  imagview() const
  {
    return new CVSIP(mview)(cblock->get_imag(),
                            block.offset(),
                            block.col_stride(), block.rows(),
                            block.row_stride(), block.cols(),
                            true);
  }
};

/***********************************************************************
   complex tensor
************************************************************************/
struct CVSIP(ctviewattributes) : 
  vsip_csl::cvsip::View<3, CVSIP(cscalar), CVSIP(cblock)>
{
  CVSIP(ctviewattributes)(vsip_length p, vsip_length m, vsip_length n, vsip_tmajor o, vsip_memory_hint h)
    : vsip_csl::cvsip::View<3, CVSIP(cscalar), CVSIP(cblock)>(p, m, n, o, h) {}
  CVSIP(ctviewattributes)(cblock_type *b, vsip_offset o,
                          vsip_stride zs, vsip_length zl,
                          vsip_stride ys, vsip_length yl,
                          vsip_stride xs, vsip_length xl,
                          bool)
    : vsip_csl::cvsip::View<3, CVSIP(cscalar), CVSIP(cblock)>(b, o,
                                                              zs, zl, ys, yl, xs, xl,
                                                              false) {}

  CVSIP(tview) *
  realview() const
  {
    return new CVSIP(tview)(cblock->get_real(), block.offset(),
                            block.z_stride(), block.z_length(),
                            block.y_stride(), block.y_length(),
                            block.x_stride(), block.x_length(),
                            true);
  }

  CVSIP(tview) *
  imagview() const
  {
    return new CVSIP(tview)(cblock->get_imag(), block.offset(),
                            block.z_stride(), block.z_length(),
                            block.y_stride(), block.y_length(), 
                            block.x_stride(), block.x_length(),
                            true);
  }
};

} // extern "C"
