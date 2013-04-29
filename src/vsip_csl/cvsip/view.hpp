/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    vsip_csl/cvsip/view.hpp
    @author  Stefan Seefeld
    @date    2008-05-02
    @brief   the C-VSIP view API.
*/

#ifndef vsip_csl_cvsip_view_hpp_
#define vsip_csl_cvsip_view_hpp_

#include <vsip_csl/cvsip/block.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>

namespace vsip_csl
{
namespace cvsip
{
template <vsip::dimension_type D, typename T, typename B> struct View;

template <typename T, typename B>
struct View<1, T, B>
{
  static vsip::dimension_type const dim = 1;
  typedef T value_type;
  typedef B cblock_type;
  typedef Block<dim, typename cblock_type::storage_type> block_type;
  typedef vsip::Vector<typename block_type::value_type, block_type> view_type;

  View(vsip_length l, vsip_memory_hint h)
    : cblock(new cblock_type(l, h)),
      block(0, 1, l, cblock->storage, false),
      view(block)
  {}

  View(View const &v)
    : cblock(v.cblock),
      block(v.block),
      view(block)
  {}

  View(cblock_type *b, vsip_offset o, vsip_stride s, vsip_length l, bool derived)
    : cblock(b),
      block(o, s, l, cblock->storage, derived),
      view(block)
  {}

  template <typename V>
  V *subview(vsip_index o, vsip_length l) const
  {
    return new V(cblock, block.offset() + o * block.stride(), block.stride(), l, block.is_derived());
  }

  cblock_type *cblock;
  block_type block;
  view_type view;
};

template <typename T, typename B>
struct View<2, T, B>
{
  static vsip::dimension_type const dim = 2;
  typedef T value_type;
  typedef B cblock_type;
  typedef Block<dim, typename cblock_type::storage_type> block_type;
  typedef vsip::Matrix<typename block_type::value_type, block_type> view_type;

  View(vsip_length r, vsip_length c, vsip_major o, vsip_memory_hint h)
    : cblock(new cblock_type(r * c, h)),
      block(0,
            o == VSIP_ROW ? c : 1, r,
            o == VSIP_ROW ? 1 : r, c,
            cblock->storage, false),
      view(block)
  {}

  View(View const &v)
    : cblock(v.cblock),
      block(v.block),
      view(block)
  {}

  View(cblock_type *b, vsip_offset o,
       vsip_stride rs, vsip_length rl,
       vsip_stride cs, vsip_length cl,
       bool derived)
    : cblock(b),
      block(o, rs, rl, cs, cl, cblock->storage, derived),
      view(block)
  {}

  template <typename V>
  V *subview(vsip_index r, vsip_index c, vsip_length m, vsip_length n) const
  {
    return new V(cblock,
                 block.offset() + r * block.col_stride() + c * block.row_stride(),
                 block.col_stride(), m, block.row_stride(), n, block.is_derived());
  }

  template <typename V>
  V *row(vsip_index r) const
  {
    return new V(cblock,
                 block.offset() + r * block.col_stride(),
                 block.row_stride(), block.cols(), block.is_derived());
  }

  template <typename V>
  V *col(vsip_index c) const
  {
    return new V(cblock,
                 block.offset() + c * block.row_stride(),
                 block.col_stride(), block.rows(), block.is_derived());
  }

  // An index of '0' specifies the main diagonal, positive indices are above
  // the main diagonal, and negative indices are below the main diagonal.
  template <typename V>
  V *diag(vsip_stride i) const
  {
    // row index of origin
    vsip_index r = i < 0 ? -i : 0;
    // col index of origin
    vsip_index c = i > 0 ?  i : 0;
    // rows from origin to end
    vsip_length rows = block.rows() - r;
    // cols from origin to end
    vsip_length	cols = block.cols() - c;
    return new V(cblock, block.offset() + r * block.col_stride() + c * block.row_stride(),
                 block.row_stride() + block.col_stride(), 
                 rows < cols ? rows : cols, block.is_derived());
  }

  template <typename V>
  V *trans() const
  {
    return new V(cblock,
                 block.offset(),
                 block.row_stride(), block.cols(),
                 block.col_stride(), block.rows(), block.is_derived());
  }

  cblock_type *cblock;
  block_type block;
  view_type view;
};

template <typename T, typename B>
struct View<3, T, B>
{
  static vsip::dimension_type const dim = 3;
  typedef T value_type;
  typedef B cblock_type;
  typedef Block<dim, typename cblock_type::storage_type> block_type;
  typedef vsip::Tensor<typename block_type::value_type, block_type> view_type;

  View(vsip_length p, vsip_length m, vsip_length n, vsip_tmajor o, vsip_memory_hint h)
    : cblock(new cblock_type(p * m * n, h)),
      block(0,
            o == VSIP_TRAILING ? m*n : 1, p,
            o == VSIP_TRAILING ? n : p, m,
            o == VSIP_TRAILING ? 1 : p*m, n,
            cblock->storage, false),
      view(block)
  {
  }

  View(View const &t)
    : cblock(t.cblock),
      block(t.block),
      view(block)
  {}

  View(cblock_type *b, vsip_offset o,
       vsip_stride zs, vsip_length zl,
       vsip_stride ys, vsip_length yl,
       vsip_stride xs, vsip_length xl,
       bool derived)
    : cblock(b),
      block(o, zs, zl, ys, yl, xs, xl, cblock->storage, derived),
      view(block)
  {}

  template <typename V>
  V *trans(vsip_ttrans t) const
  {
    vsip_stride z_stride = block.z_stride();
    vsip_stride y_stride = block.y_stride();
    vsip_stride x_stride = block.x_stride();
    vsip_length z_length = block.z_length();
    vsip_length y_length = block.y_length();
    vsip_length x_length = block.x_length();
    switch (t)
    {
      case VSIP_TTRANS_YX:
        std::swap(y_stride, x_stride);
        std::swap(y_length, x_length);
        break;
      case VSIP_TTRANS_ZY:
        std::swap(z_stride, y_stride);
        std::swap(z_length, y_length);
        break;
      case VSIP_TTRANS_ZX:
        std::swap(z_stride, x_stride);
        std::swap(z_length, x_length);
        break;
      case VSIP_TTRANS_YXZY:
        z_stride = x_stride;
        z_length = x_length;
        x_stride = y_stride;
        x_length = y_length;
        y_stride = block.z_stride();
        y_length = block.z_length();
        break;
      case VSIP_TTRANS_YXZX:
        z_stride = y_stride;
        z_length = y_length;
        y_stride = x_stride;
        y_length = x_length;
        x_stride = block.z_stride();
        x_length = block.z_length();
        break;
      case VSIP_TTRANS_NOP:
      default:
        break;
    }
    return new V(cblock, block.offset(),
                 z_stride, z_length,
                 y_stride, y_length,
                 x_stride, x_length, block.is_derived());
  }

  template <typename V>
  V *subview(vsip_index z, vsip_index y, vsip_index x,
             vsip_length p, vsip_length m, vsip_length n) const
  {
    vsip_offset o = 
      block.offset() + 
      z * block.z_stride() +
      y * block.y_stride() + 
      x * block.x_stride();
    return new V(cblock, o,
                 block.z_stride(), p,
                 block.y_stride(), m,
                 block.x_stride(), n, block.is_derived());
  }

  template <typename V>
  V *vector(vsip_tvslice s, vsip_index i, vsip_index j) const
  {
    vsip_offset offset = block.offset();
    vsip_stride stride = 1;
    vsip_length length = 0;
    switch (s)
    {
      case VSIP_TVX:
        offset += i * block.z_stride() + j * block.y_stride();
        stride = block.x_stride();
        length = block.x_length();
        break;
      case VSIP_TVY:
        offset += i * block.z_stride() + j * block.x_stride();
        stride = block.y_stride();
        length = block.y_length();
        break;
      case VSIP_TVZ:
      default:
        offset += i * block.y_stride() + j * block.x_stride();
        stride = block.z_stride();
        length = block.z_length();
        break;
    }
    return new V(cblock, offset, stride, length, block.is_derived());
  }

  template <typename V>
  V *matrix(vsip_tmslice s, vsip_index i) const
  {
    vsip_offset offset = block.offset();
    vsip_stride y_stride = 1;
    vsip_stride x_stride = 1;
    vsip_length y_length = 0;
    vsip_length x_length = 0;
    switch (s)
    {
      case VSIP_TMYX:
        offset += i * block.z_stride();
        y_stride = block.y_stride();
        y_length = block.y_length();
        x_stride = block.x_stride();
        x_length = block.x_length();
        break;
      case VSIP_TMZX:
        offset += i * block.y_stride();
        y_stride = block.z_stride();
        y_length = block.z_length();
        x_stride = block.x_stride();
        x_length = block.x_length();
        break;
      case VSIP_TMZY:
      default:
        offset += i * block.x_stride();
        y_stride = block.z_stride();
        y_length = block.z_length();
        x_stride = block.y_stride();
        x_length = block.y_length();
        break;
    }
    return new V(cblock, offset,
                 y_stride, y_length, x_stride, x_length, block.is_derived());
  }

  cblock_type *cblock;
  block_type block;
  view_type view;
};

} // namespace vsip_csl::cvsip
} // namespace vsip_csl

#endif
