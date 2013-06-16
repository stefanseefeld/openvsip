//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cvsip_view_hpp_
#define ovxx_cvsip_view_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/dda.hpp>
#include <ovxx/detail/noncopyable.hpp>
#include <ovxx/cvsip/block.hpp>
extern "C" {
#include <vsip.h>

// TVCPP defines, but does not declare, the following:
void vsip_vcopy_bl_bl(vsip_vview_bl const*, vsip_vview_bl const*); 
}

namespace ovxx
{
namespace cvsip
{

// Traits for C-VSIP views.
//
// [1] C-VSIP implementations may not be orthogonal.  Functions
//     below commented out with a '[1]' label are not implemented
//     in TVCPP.

template <dimension_type D, typename T> struct view_traits;

#if OVXX_CVSIP_HAVE_BOOL

template <>
struct view_traits<1, bool>
{
  typedef bool value_type;
  typedef vsip_block_bl block_type;
  typedef vsip_vview_bl view_type;

  static view_type* create(vsip_length l)
  {
    view_type* v = vsip_vcreate_bl(l, VSIP_MEM_NONE);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type* bind(block_type const* b,
                         vsip_offset o, vsip_stride s, vsip_length l)
  {
    view_type* v = vsip_vbind_bl(b, o, s, l);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type* clone(view_type* v) 
  {
    view_type* c = vsip_vcloneview_bl(v);
    if (!c) VSIP_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(view_type* v) { vsip_valldestroy_bl(v);}
  static void copy(view_type* s, view_type* d) { vsip_vcopy_bl_bl(s, d);}
  static block_type* block(view_type* v) { return vsip_vgetblock_bl(v);}
  static vsip_offset offset(view_type* v) { return vsip_vgetoffset_bl(v);}
  static vsip_stride stride(view_type* v) { return vsip_vgetstride_bl(v);}
  static vsip_length length(view_type* v) { return vsip_vgetlength_bl(v);}

  static bool get(view_type* v, index_type i)
    { return (bool)vsip_vget_bl(v, i); }
  static void put(view_type* v, index_type i, bool value)
    { vsip_vput_bl(v, i, (vsip_scalar_bl)value); }
};

template <>
struct view_traits<2, bool>
{
  typedef bool value_type;
  typedef vsip_block_bl block_type;
  typedef vsip_mview_bl view_type;

  static view_type* create(vsip_length r, vsip_length c, bool row_major)
  {
    view_type* v = vsip_mcreate_bl(r, c, row_major ? VSIP_ROW : VSIP_COL,
                                  VSIP_MEM_NONE);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type* bind(block_type const* b, vsip_offset o,
                         vsip_stride s_r, vsip_length rows,
                         vsip_stride s_c, vsip_length cols)
  {
    view_type* v = vsip_mbind_bl(const_cast<block_type*>(b), o, s_r, rows, s_c, cols);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type* clone(view_type* v)
  {
    view_type* c = vsip_mcloneview_bl(v);
    if (!c) VSIP_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(view_type* v) { vsip_malldestroy_bl(v);}
  static void copy(view_type* s, view_type* d) { vsip_mcopy_bl_bl(s, d);}
  static block_type* block(view_type* v) { return vsip_mgetblock_bl(v);}
  static vsip_offset offset(view_type* v) { return vsip_mgetoffset_bl(v);}
  static vsip_stride row_stride(view_type* v) { return vsip_mgetrowstride_bl(v);}
  static vsip_length row_length(view_type* v) { return vsip_mgetrowlength_bl(v);}
  static vsip_stride col_stride(view_type* v) { return vsip_mgetcolstride_bl(v);}
  static vsip_length col_length(view_type* v) { return vsip_mgetcollength_bl(v);}
  static bool get(view_type* v, index_type r, index_type c)
    { return (bool)vsip_mget_bl(v, r, c); }
  static void put(view_type* v, index_type r, index_type c, bool value)
    { vsip_mput_bl(v, r, c, (vsip_scalar_bl)value); }
};

#endif
#if OVXX_CVSIP_HAVE_INT

template <>
struct view_traits<1, int>
{
  typedef int value_type;
  typedef vsip_block_i block_type;
  typedef vsip_vview_i view_type;

  static view_type* create(vsip_length l)
  {
    view_type* v = vsip_vcreate_i(l, VSIP_MEM_NONE);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type* bind(block_type const* b,
                         vsip_offset o, vsip_stride s, vsip_length l)
  {
    view_type* v = vsip_vbind_i(b, o, s, l);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type* clone(view_type* v) 
  {
    view_type* c = vsip_vcloneview_i(v);
    if (!c) VSIP_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(view_type* v) { vsip_valldestroy_i(v);}
  static void copy(view_type* s, view_type* d) { vsip_vcopy_i_i(s, d);}
  static block_type* block(view_type* v) { return vsip_vgetblock_i(v);}
  static vsip_offset offset(view_type* v) { return vsip_vgetoffset_i(v);}
  static vsip_stride stride(view_type* v) { return vsip_vgetstride_i(v);}
  static vsip_length length(view_type* v) { return vsip_vgetlength_i(v);}
};

template <>
struct view_traits<2, int>
{
  typedef int value_type;
  typedef vsip_block_i block_type;
  typedef vsip_mview_i view_type;

  static view_type* create(vsip_length r, vsip_length c, bool row_major)
  {
    view_type* v = vsip_mcreate_i(r, c, row_major ? VSIP_ROW : VSIP_COL,
                                  VSIP_MEM_NONE);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type* bind(block_type const* b, vsip_offset o,
                         vsip_stride s_r, vsip_length rows,
                         vsip_stride s_c, vsip_length cols)
  {
    view_type* v = vsip_mbind_i(const_cast<block_type*>(b), o, s_r, rows, s_c, cols);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type* clone(view_type* v)
  {
    view_type* c = vsip_mcloneview_i(v);
    if (!c) VSIP_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(view_type* v) { vsip_malldestroy_i(v);}
  // [1] static void copy(view_type* s, view_type* d) { vsip_mcopy_i_i(s, d);}
  static block_type* block(view_type* v) { return vsip_mgetblock_i(v);}
  static vsip_offset offset(view_type* v) { return vsip_mgetoffset_i(v);}
  static vsip_stride row_stride(view_type* v) { return vsip_mgetrowstride_i(v);}
  static vsip_length row_length(view_type* v) { return vsip_mgetrowlength_i(v);}
  static vsip_stride col_stride(view_type* v) { return vsip_mgetcolstride_i(v);}
  static vsip_length col_length(view_type* v) { return vsip_mgetcollength_i(v);}
};

#endif
#if OVXX_CVSIP_HAVE_FLOAT

template <>
struct view_traits<1, float>
{
  typedef float value_type;
  typedef vsip_block_f block_type;
  typedef vsip_vview_f view_type;

  static view_type *create(vsip_length l)
  {
    view_type *v = vsip_vcreate_f(l, VSIP_MEM_NONE);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *bind(block_type const *b,
                         vsip_offset o, vsip_stride s, vsip_length l)
  {
    // TODO: Remove const-cast once AXISLIB is fixed.
    view_type *v = vsip_vbind_f(const_cast<block_type*>(b), o, s, l);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *clone(view_type *v) 
  {
    view_type *c = vsip_vcloneview_f(v);
    if (!c) VSIP_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(view_type *v) { vsip_valldestroy_f(v);}
  static void copy(view_type *s, view_type *d) { vsip_vcopy_f_f(s, d);}
  static block_type *block(view_type *v) { return vsip_vgetblock_f(v);}
  static vsip_offset offset(view_type *v) { return vsip_vgetoffset_f(v);}
  static vsip_stride stride(view_type *v) { return vsip_vgetstride_f(v);}
  static vsip_length length(view_type *v) { return vsip_vgetlength_f(v);}

  static float get(view_type* v, index_type i)
    { return (float)vsip_vget_f(v, i); }
  static void put(view_type* v, index_type i, float value)
    { vsip_vput_f(v, i, (vsip_scalar_f)value); }
};

template <>
struct view_traits<1, std::complex<float> >
{
  typedef std::complex<float> value_type;
  typedef vsip_cblock_f block_type;
  typedef vsip_cvview_f view_type;

  static view_type *create(vsip_length l)
  {
    view_type *v = vsip_cvcreate_f(l, VSIP_MEM_NONE);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *bind(block_type const *b,
                         vsip_offset o, vsip_stride s, vsip_length l)
  {
    view_type *v = vsip_cvbind_f(const_cast<block_type*>(b), o, s, l);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *clone(view_type *v)
  {
    view_type *c = vsip_cvcloneview_f(v);
    if (!c) VSIP_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(view_type *v) { vsip_cvalldestroy_f(v);}
  static void copy(view_type *s, view_type *d) { vsip_cvcopy_f_f(s, d);}
  static block_type *block(view_type *v) { return vsip_cvgetblock_f(v);}
  static vsip_offset offset(view_type *v) { return vsip_cvgetoffset_f(v);}
  static vsip_stride stride(view_type *v) { return vsip_cvgetstride_f(v);}
  static vsip_length length(view_type *v) { return vsip_cvgetlength_f(v);}
};

template <>
struct view_traits<2, float>
{
  typedef float value_type;
  typedef vsip_block_f block_type;
  typedef vsip_mview_f view_type;

  static view_type *create(vsip_length r, vsip_length c, bool row_major)
  {
    view_type *v = vsip_mcreate_f(r, c, row_major ? VSIP_ROW : VSIP_COL,
                                  VSIP_MEM_NONE);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *bind(block_type const *b, vsip_offset o,
                         vsip_stride s_r, vsip_length rows,
                         vsip_stride s_c, vsip_length cols)
  {
    view_type *v = vsip_mbind_f(const_cast<block_type*>(b), o, s_r, rows, s_c, cols);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *clone(view_type *v)
  {
    view_type *c = vsip_mcloneview_f(v);
    if (!c) VSIP_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(view_type *v) { vsip_malldestroy_f(v);}
  static void copy(view_type *s, view_type *d) { vsip_mcopy_f_f(s, d);}
  static block_type *block(view_type *v) { return vsip_mgetblock_f(v);}
  static vsip_offset offset(view_type *v) { return vsip_mgetoffset_f(v);}
  static vsip_stride row_stride(view_type *v) { return vsip_mgetrowstride_f(v);}
  static vsip_length row_length(view_type *v) { return vsip_mgetrowlength_f(v);}
  static vsip_stride col_stride(view_type *v) { return vsip_mgetcolstride_f(v);}
  static vsip_length col_length(view_type *v) { return vsip_mgetcollength_f(v);}
};

template <>
struct view_traits<2, std::complex<float> >
{
  typedef std::complex<float> value_type;
  typedef vsip_cblock_f block_type;
  typedef vsip_cmview_f view_type;

  static view_type *create(vsip_length r, vsip_length c, bool row_major)
  {
    view_type *v = vsip_cmcreate_f(r, c, row_major ? VSIP_ROW : VSIP_COL, VSIP_MEM_NONE);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *bind(block_type const *b, vsip_offset o,
                         vsip_stride s_r, vsip_length rows,
                         vsip_stride s_c, vsip_length cols)
  {
    view_type *v = vsip_cmbind_f(const_cast<block_type*>(b), o, s_r, rows, s_c, cols);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *clone(view_type *v)
  {
    view_type *c = vsip_cmcloneview_f(v);
    if (!c) VSIP_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(view_type *v) { vsip_cmalldestroy_f(v);}
  static void copy(view_type *s, view_type *d) { vsip_cmcopy_f_f(s, d);}
  static block_type *block(view_type *v) { return vsip_cmgetblock_f(v);}
  static vsip_offset offset(view_type *v) { return vsip_cmgetoffset_f(v);}
  static vsip_stride row_stride(view_type *v) { return vsip_cmgetrowstride_f(v);}
  static vsip_length row_length(view_type *v) { return vsip_cmgetrowlength_f(v);}
  static vsip_stride col_stride(view_type *v) { return vsip_cmgetcolstride_f(v);}
  static vsip_length col_length(view_type *v) { return vsip_cmgetcollength_f(v);}
};

#endif
#if OVXX_CVSIP_HAVE_DOUBLE

template <>
struct view_traits<1, double>
{
  typedef double value_type;
  typedef vsip_block_d block_type;
  typedef vsip_vview_d view_type;

  static view_type *create(vsip_length l)
  {
    view_type *v = vsip_vcreate_d(l, VSIP_MEM_NONE);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *bind(block_type const *b,
                         vsip_offset o, vsip_stride s, vsip_length l)
  {
    view_type *v = vsip_vbind_d(const_cast<block_type*>(b), o, s, l);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *clone(view_type *v) 
  {
    view_type *c = vsip_vcloneview_d(v);
    if (!c) VSIP_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(view_type *v) { vsip_valldestroy_d(v);}
  static void copy(view_type *s, view_type *d) { vsip_vcopy_d_d(s, d);}
  static block_type *block(view_type *v) { return vsip_vgetblock_d(v);}
  static vsip_offset offset(view_type *v) { return vsip_vgetoffset_d(v);}
  static vsip_stride stride(view_type *v) { return vsip_vgetstride_d(v);}
  static vsip_length length(view_type *v) { return vsip_vgetlength_d(v);}

  static value_type get(view_type* v, index_type i)
    { return (value_type)vsip_vget_d(v, i); }
  static void put(view_type* v, index_type i, value_type value)
    { vsip_vput_d(v, i, (vsip_scalar_d)value); }
};

template <>
struct view_traits<1, std::complex<double> >
{
  typedef std::complex<double> value_type;
  typedef vsip_cblock_d block_type;
  typedef vsip_cvview_d view_type;

  static view_type *create(vsip_length l)
  {
    view_type *v = vsip_cvcreate_d(l, VSIP_MEM_NONE);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *bind(block_type const *b,
                         vsip_offset o, vsip_stride s, vsip_length l)
  {
    view_type *v = vsip_cvbind_d(const_cast<block_type*>(b), o, s, l);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *clone(view_type *v)
  {
    view_type *c = vsip_cvcloneview_d(v);
    if (!c) VSIP_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(view_type *v) { vsip_cvalldestroy_d(v);}
  static void copy(view_type *s, view_type *d) { vsip_cvcopy_d_d(s, d);}
  static block_type *block(view_type *v) { return vsip_cvgetblock_d(v);}
  static vsip_offset offset(view_type *v) { return vsip_cvgetoffset_d(v);}
  static vsip_stride stride(view_type *v) { return vsip_cvgetstride_d(v);}
  static vsip_length length(view_type *v) { return vsip_cvgetlength_d(v);}
};

template <>
struct view_traits<2, double>
{
  typedef double value_type;
  typedef vsip_block_d block_type;
  typedef vsip_mview_d view_type;

  static view_type *create(vsip_length r, vsip_length c, bool row_major)
  {
    view_type *v = vsip_mcreate_d(r, c, row_major ? VSIP_ROW : VSIP_COL,
                                  VSIP_MEM_NONE);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *bind(block_type const *b, vsip_offset o,
                         vsip_stride s_r, vsip_length rows,
                         vsip_stride s_c, vsip_length cols)
  {
    view_type *v = vsip_mbind_d(const_cast<block_type*>(b), o, s_r, rows, s_c, cols);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *clone(view_type *v)
  {
    view_type *c = vsip_mcloneview_d(v);
    if (!c) VSIP_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(view_type *v) { vsip_malldestroy_d(v);}
  static void copy(view_type *s, view_type *d) { vsip_mcopy_d_d(s, d);}
  static block_type *block(view_type *v) { return vsip_mgetblock_d(v);}
  static vsip_offset offset(view_type *v) { return vsip_mgetoffset_d(v);}
  static vsip_stride row_stride(view_type *v) { return vsip_mgetrowstride_d(v);}
  static vsip_length row_length(view_type *v) { return vsip_mgetrowlength_d(v);}
  static vsip_stride col_stride(view_type *v) { return vsip_mgetcolstride_d(v);}
  static vsip_length col_length(view_type *v) { return vsip_mgetcollength_d(v);}
};

template <>
struct view_traits<2, std::complex<double> >
{
  typedef std::complex<double> value_type;
  typedef vsip_cblock_d block_type;
  typedef vsip_cmview_d view_type;

  static view_type *create(vsip_length r, vsip_length c, bool row_major)
  {
    view_type *v = vsip_cmcreate_d(r, c, row_major ? VSIP_ROW : VSIP_COL,
                                   VSIP_MEM_NONE);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *bind(block_type const *b, vsip_offset o,
                         vsip_stride s_r, vsip_length rows,
                         vsip_stride s_c, vsip_length cols)
  {
    view_type *v = vsip_cmbind_d(const_cast<block_type*>(b), o, s_r, rows, s_c, cols);
    if (!v) VSIP_THROW(std::bad_alloc());
    return v;
  }
  static view_type *clone(view_type *v)
  {
    view_type *c = vsip_cmcloneview_d(v);
    if (!c) VSIP_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(view_type *v) { vsip_cmalldestroy_d(v);}
  static void copy(view_type *s, view_type *d) { vsip_cmcopy_d_d(s, d);}
  static block_type *block(view_type *v) { return vsip_cmgetblock_d(v);}
  static vsip_offset offset(view_type *v) { return vsip_cmgetoffset_d(v);}
  static vsip_stride row_stride(view_type *v) { return vsip_cmgetrowstride_d(v);}
  static vsip_length row_length(view_type *v) { return vsip_cmgetrowlength_d(v);}
  static vsip_stride col_stride(view_type *v) { return vsip_cmgetcolstride_d(v);}
  static vsip_length col_length(view_type *v) { return vsip_cmgetcollength_d(v);}
};

#endif

template <dimension_type D, // dimension
          typename T,       // type
          bool U = true>    // use user-storage
class View;

template <typename T>
class View<1, T, false> : ovxx::detail::noncopyable
{
  typedef view_traits<1, T>           traits;
  typedef Block<T>                    block_type;
  typedef typename block_type::traits block_traits;
  friend class View<1, T, true>; // Needed for operator=

public:
  View(length_type size) : impl_(traits::create(size)) {}
  template <bool U>
  View &operator= (View<1, T, U> const &v)
  { traits::copy(v.impl_, impl_); return *this;}
  block_type block() { return block_type(traits::block(impl_));}
  length_type size() const { return traits::length(impl_);}
  typename traits::view_type *ptr() { return impl_;}

protected:
  typename traits::view_type *impl_;
};

template <typename T>
class View<1, T, true> : ovxx::detail::noncopyable
{
  typedef view_traits<1, T>           traits;
  typedef Block<T, true>              block_type;
  typedef typename block_type::traits block_traits;
  friend class View<1, T, false>; // Needed for operator=

public:
  View(T *data, length_type size)
    : impl_(traits::bind(block_traits::bind(data, size), 0, 1, size)) {}
  View(T *data, index_type offset, stride_type stride, length_type size)
    : impl_(traits::bind(block_traits::bind(data, offset + (size - 1) * stride + 1),
                         offset, stride, size))
  {}
  template <bool U>
  View &operator= (View<1, T, U> const &v)
  { traits::copy(v.impl_, impl_); return *this;}
  block_type block() { return block_type(traits::block(impl_));}
  length_type size() const { return traits::length(impl_);}
  typename traits::view_type *ptr() { return impl_;}

protected:
  typename traits::view_type *impl_;
};

template <typename T>
class View<1, std::complex<T>, true> : ovxx::detail::noncopyable
{
  typedef view_traits<1, std::complex<T> >  traits;
  typedef Block<std::complex<T>, true>      block_type;
  typedef typename block_type::traits       block_traits;
  friend class View<1, std::complex<T>, false>; // Needed for operator=

public:
  View(std::complex<T> *data, length_type size)
    : impl_(traits::bind(block_traits::bind(data, size), 0, 1, size)) {}
  View(std::pair<T *, T *> data, length_type size)
    : impl_(traits::bind(block_traits::bind(data, size), 0, 1, size)) {}
  View(std::complex<T> *data, index_type offset,
       stride_type stride, length_type size)
    : impl_(traits::bind(block_traits::bind(data, offset + (size - 1) * stride + 1),
                         offset, stride, size))
  {}
  View(std::pair<T *, T *> data, index_type offset,
       stride_type stride, length_type size)
    : impl_(traits::bind(block_traits::bind(data, offset + (size - 1) * stride + 1),
                         offset, stride, size))
  {}
  template <bool U>
  View &operator= (View<1, std::complex<T>, U> const &v)
  { traits::copy(v.impl_, impl_); return *this;}
  block_type block() { return block_type(traits::block(impl_));}
  length_type size() const { return traits::length(impl_);}
  typename traits::view_type *ptr() { return impl_;}

protected:
  typename traits::view_type *impl_;
};

#if OVXX_CVSIP_HAVE_BOOL

// Specialize View to avoid using admit/release for user-storage
// bool vectors.  Perform copy instead.
//
// C-VSIP and C++ have different size bool types (C-VSIP
// vsip_scalar_bl is usually an int (4 bytes), while C++ bool is 1
// byte).
//
template <>
class View<1, bool, true> : public View<1, bool, false>
{
  typedef bool T;
  typedef View<1, bool, false>        base_type;
  typedef view_traits<1, T>           traits;
  typedef Block<T, true>              block_type;
  typedef block_type::traits          block_traits;
  friend class View<1, T, false>; // Needed for operator=

public:
  View(T* data, length_type size)
    : base_type(size),
      data_    (data),
      offset_  (0),
      stride_  (1)
  {
    for (index_type i=0; i<size; ++i)
      traits::put(this->ptr(), i, data[i]);
  }

  View(T* data, index_type offset, stride_type stride, length_type size)
    : base_type(size),
      data_    (data),
      offset_  (offset),
      stride_  (stride)
  {
    for (index_type i=0; i<size; ++i)
      traits::put(this->ptr(), i, data[offset + i*stride]);
  }

  ~View()
  {
    for (index_type i=0; i<traits::length(this->ptr()); ++i)
      data_[offset_ + i*stride_] = traits::get(this->ptr(), i);
  }

private:
  T*          data_;
  index_type  offset_;
  stride_type stride_;
};

#endif

template <typename T>
class View<2, T, false> : ovxx::detail::noncopyable
{
  typedef view_traits<2, T>           traits;
  typedef Block<T>                    block_type;
  typedef typename block_type::traits block_traits;
  friend class View<2, T, true>; // Needed for operator=

public:
  View(length_type rows, length_type cols, bool row_major)
    : impl_(traits::create(rows, cols, row_major)) {}
  template <bool U>
  View &operator= (View<2, T, U> const &v)
  { traits::copy(v.impl_, impl_); return *this;}
  block_type block() { return block_type(traits::block(impl_));}
  typename traits::view_type *ptr() { return impl_;}

protected:
  typename traits::view_type *impl_;
};

template <typename T>
class View<2, T, true> : ovxx::detail::noncopyable
{
  typedef view_traits<2, T>           traits;
  typedef Block<T, true>              block_type;
  typedef typename block_type::traits block_traits;
  friend class View<2, T, false>; // Needed for operator=

public:
  View(T *data, length_type rows, length_type cols, bool row_major)
    : impl_(traits::bind(block_traits::bind(data, rows * cols), 0,
                         row_major ? cols : 1, rows, row_major ? 1 : rows, cols))
  {}
  View(T *data, index_type offset,
       stride_type s_r, length_type rows,
       stride_type s_c, length_type cols)
    : impl_(traits::bind(block_traits::bind(data,
                                            offset + (rows-1)*s_r + (cols-1)*s_c + 1),
                         offset, s_r, rows, s_c, cols))
  {}
  template <bool U>
  View &operator= (View<2, T, U> const &v)
  { traits::copy(v.impl_, impl_); return *this;}
  block_type block() { return block_type(traits::block(impl_));}
  typename traits::view_type *ptr() { return impl_;}

protected:
  typename traits::view_type *impl_;
};

template <typename T>
class View<2, std::complex<T>, true> : ovxx::detail::noncopyable
{
  typedef view_traits<2, std::complex<T> >  traits;
  typedef Block<std::complex<T>, true>      block_type;
  typedef typename block_type::traits       block_traits;
  friend class View<2, std::complex<T>, false>; // Needed for operator=

public:
  View(std::complex<T> *data, length_type rows, length_type cols, bool row_major)
    : impl_(traits::bind(block_traits::bind(data, rows * cols), 0,
                         row_major ? cols : 1, rows, row_major ? 1 : rows, cols))
  {}
  View(std::pair<T *, T *> data, length_type rows, length_type cols, bool row_major)
    : impl_(traits::bind(block_traits::bind(data, rows * cols), 0,
                         row_major ? cols : 1, rows, row_major ? 1 : rows, cols))
  {}
  View(std::complex<T> *data, index_type offset,
       stride_type s_r, length_type rows,
       stride_type s_c, length_type cols)
    : impl_(traits::bind(block_traits::bind(data,
                                            offset + (rows-1)*s_r + (cols-1)*s_c + 1),
                         offset, s_r, rows, s_c, cols))
  {}
  View(std::pair<T *, T *> data, index_type offset,
       stride_type s_r, length_type rows,
       stride_type s_c, length_type cols)
    : impl_(traits::bind(block_traits::bind(data,
                                            offset + (rows-1)*s_r + (cols-1)*s_c + 1),
                         offset, s_r, rows, s_c, cols))
  {}
  template <bool U>
  View &operator= (View<2, std::complex<T>, U> const &v)
  { traits::copy(v.impl_, impl_); return *this;}
  block_type block() { return block_type(traits::block(impl_));}
  typename traits::view_type *ptr() { return impl_;}

protected:
  typename traits::view_type *impl_;
};

#if OVXX_CVSIP_HAVE_BOOL

// Specialize View to avoid using admit/release for user-storage
// bool matrices.  Perform copy instead.
//
// See View<1, bool, true> specialization for details.
//
template <>
class View<2, bool, true> : public View<2, bool, false>
{
  typedef bool T;
  typedef View<2, bool, false>        base_type;
  typedef view_traits<2, T>           traits;
  typedef Block<T, true>              block_type;
  typedef block_type::traits          block_traits;
  friend class View<2, T, false>; // Needed for operator=

public:
  View(T* data, length_type rows, length_type cols, bool row_major)
    : base_type(rows, cols, row_major),
      data_    (data),
      offset_  (0),
      stride0_ (row_major ? cols : 1),
      stride1_ (row_major ? 1 : rows)
  {
    for (index_type r=0; r<rows; ++r)
      for (index_type c=0; c<cols; ++c)
	traits::put(this->ptr(), r, c, data[r*stride0_ + c*stride1_]);
  }

  View(T* data, index_type offset,
       stride_type s_r, length_type rows,
       stride_type s_c, length_type cols)
    : base_type(rows, cols, s_r > s_c),
      data_    (data),
      offset_  (offset),
      stride0_ (s_r),
      stride1_ (s_c)
  {
    for (index_type r=0; r<rows; ++r)
      for (index_type c=0; c<cols; ++c)
	traits::put(this->ptr(), r, c, data[offset_+r*stride0_ + c*stride1_]);
  }

  ~View()
  {
    // Remember: number of rows == length of column (and visa versa)
    for (index_type r=0; r<traits::col_length(this->ptr()); ++r)
      for (index_type c=0; c<traits::row_length(this->ptr()); ++c)
	data_[offset_+r*stride0_+c*stride1_] = traits::get(this->ptr(), r, c);
  }

private:
  T*          data_;
  index_type  offset_;
  stride_type stride0_;
  stride_type stride1_;
};

#endif

template <dimension_type D, // dimension
          typename T>       // type
class const_View;

/// Derive const_View from View as a convenience wrapper
/// that does all the const_casting
template <typename T>
class const_View<1, T> : public View<1, T, true>
{
  typedef View<1, T, true> base_type;
public:
  const_View(T const *data, length_type size)
    : base_type(const_cast<T*>(data), size) {}
  const_View(T const *data, index_type offset, stride_type stride, length_type size)
    : base_type(const_cast<T*>(data), offset, stride, size) {}
  // const_View &operator= (const_View<1, T> const &v)
  // { traits::copy(v.impl_, impl_); return *this;}
};

template <typename T>
class const_View<1, complex<T> > : public View<1, std::complex<T>, true>
{
  typedef View<1, complex<T>, true> base_type;
public:
  const_View(complex<T> const *data, length_type size)
    : base_type(const_cast<complex<T>*>(data), size) {}
  const_View(std::pair<T const *, T const *> data, length_type size)
    : base_type(std::make_pair(const_cast<T*>(data.first),
			       const_cast<T*>(data.second)), size) {}
  const_View(complex<T> const *data, index_type offset,
       stride_type stride, length_type size)
    : base_type(const_cast<complex<T>*>(data), offset, stride, size) {}
  const_View(std::pair<T const *, T const *> data, index_type offset,
       stride_type stride, length_type size)
    : base_type(std::make_pair(const_cast<T*>(data.first),
			       const_cast<T*>(data.second)), size) {}
};

template <typename T>
class const_View<2, T> : public View<2, T, true>
{
  typedef View<2, T, true> base_type;
public:
  const_View(T const *data, length_type rows, length_type cols, bool row_major)
    : base_type(const_cast<T*>(data), rows, cols, row_major) {}
  const_View(T const *data, index_type offset,
       stride_type s_r, length_type rows,
       stride_type s_c, length_type cols)
    : base_type(const_cast<T*>(data), offset, s_r, rows, s_c, cols) {}
};

template <typename T>
class const_View<2, complex<T> > : public View<2, std::complex<T>, true>
{
  typedef View<2, complex<T>, true> base_type;
public:
  const_View(complex<T> const *data, length_type rows, length_type cols, bool row_major)
    : base_type(const_cast<complex<T>*>(data), rows, cols, row_major) {}
  const_View(std::pair<T const *, T const *> data, length_type rows, length_type cols, bool row_major)
    : base_type(std::make_pair(const_cast<T*>(data.first),
			       const_cast<T*>(data.second)),
		rows, cols, row_major)
  {}
  const_View(complex<T> const *data, index_type offset,
       stride_type s_r, length_type rows,
       stride_type s_c, length_type cols)
    : base_type(const_cast<complex<T>*>(data), offset, s_r, rows, s_c, cols) {}
  const_View(std::pair<T const *, T const *> data, index_type offset,
       stride_type s_r, length_type rows,
       stride_type s_c, length_type cols)
    : base_type(std::make_pair(const_cast<T*>(data.first),
			       const_cast<T*>(data.second)),
		offset, s_r, rows, s_c, cols) {}
};


// Construct view directly from dda::Data API

template <dimension_type D, typename T, dda::sync_policy>
struct view_from_data;

template <typename T, dda::sync_policy S>
struct view_from_data<1, T, S>
{
  template <typename DDA>
  view_from_data(DDA &data)
    : view(data.ptr(), 0, data.stride(0), data.size(0)) {}

  View<1, T, true> view;
};

template <typename T>
struct view_from_data<1, T, dda::in>
{
  template <typename DDA>
  view_from_data(DDA &data)
    : view(data.ptr(), 0, data.stride(0), data.size(0)) {}

  const_View<1, T> view;
};

template <typename T, dda::sync_policy S>
struct view_from_data<2, T, S>
{
  template <typename DDA>
  view_from_data(DDA &data)
    : view(data.ptr(), 0, data.stride(0), data.size(0),
	   data.stride(1), data.size(1))
  {}

  View<2, T, true> view;
};

template <typename T>
struct view_from_data<2, T, dda::in>
{
  template <typename DDA>
  view_from_data(DDA &data)
    : view(data.ptr(), 0, data.stride(0), data.size(0),
	   data.stride(1), data.size(1))
  {}

  const_View<2, T> view;
};

} // namespace ovxx::cvsip
} // namespace ovxx

#endif
