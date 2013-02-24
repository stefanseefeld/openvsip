/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/save_view.cpp
    @author  Jules Bergmann
    @date    2005-09-30
    @brief   VSIPL++ CodeSourcery Library: Utility to save a view to disk.
*/

#ifndef VSIP_CSL_SAVE_VIEW_HPP
#define VSIP_CSL_SAVE_VIEW_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/map.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <vsip/core/adjust_layout.hpp>
#include <vsip/core/working_view.hpp>
#include <vsip/core/view_cast.hpp>

#include <vsip_csl/matlab.hpp>


namespace vsip_csl
{

/***********************************************************************
  Definitions
***********************************************************************/

template <typename             OrderT,
	  vsip::dimension_type Dim>
bool
is_subdomain_contiguous(
  vsip::Domain<Dim> const&       sub_dom,
  vsip::impl::Length<Dim> const& ext)
{
  using vsip::dimension_type;
  using vsip::stride_type;

  dimension_type const dim0 = OrderT::impl_dim0;
  dimension_type const dim1 = OrderT::impl_dim1;
  dimension_type const dim2 = OrderT::impl_dim2;

  assert(Dim <= VSIP_MAX_DIMENSION);

  if (Dim == 1)
  {
    return (sub_dom[dim0].stride() == 1);
  }
  else if (Dim == 2)
  {
    return (sub_dom[dim1].stride() == 1) &&
           (sub_dom[dim0].size() == 1 ||
	    (sub_dom[dim0].stride() == 1 &&
	     sub_dom[dim1].size() == ext[dim1]));
  }
  else /*  if (Dim == 2) */
  {
    return (sub_dom[dim2].stride() == 1)
      && ((sub_dom[dim0].size() == 1 && sub_dom[dim1].size() == 1) ||
	  (sub_dom[dim1].stride() == 1 && sub_dom[dim2].size() == ext[dim2]))
      && (sub_dom[dim0].size() == 1  ||
	  (sub_dom[dim0].stride() == 1 && sub_dom[dim1].size() == ext[dim1]));
  }
}



/// Save a view to a FILE*.
///
/// Requires:
///   FD to be a FILE open for writing.
///   VIEW to be a VSIPL++ view.

template <typename ViewT>
void
save_view(
  FILE* fd,
  ViewT view,
  bool  swap_bytes = false)
{
  using vsip::impl::Block_layout;
  using vsip::impl::Ext_data;
  using vsip::impl::Adjust_layout_complex;
  using vsip::impl::Cmplx_inter_fmt;

  if (subblock(view) != vsip::no_subblock)
  {
    vsip::dimension_type const Dim = ViewT::dim;

    typedef typename ViewT::value_type       value_type;
    typedef typename ViewT::local_type       l_view_type;
    typedef typename l_view_type::block_type l_block_type;
    typedef typename Block_layout<l_block_type>::order_type order_type;

    typedef typename Block_layout<l_block_type>::layout_type layout_type;
    typedef typename Adjust_layout_complex<Cmplx_inter_fmt, layout_type>::type
      use_layout_type;

    vsip::Domain<Dim> g_dom = global_domain(view);
    vsip::Domain<Dim> l_dom = subblock_domain(view);

    assert(is_subdomain_contiguous<order_type>(g_dom, extent(view)));

    long l_pos = 0;

    if (Dim >= 1)
    {
      l_pos += g_dom[order_type::impl_dim0].first();
    }

    if (Dim >= 2)
    {
      l_pos *= g_dom[order_type::impl_dim1].size();
      l_pos += g_dom[order_type::impl_dim1].first();
    }

    if (Dim >= 3)
    {
      l_pos *= g_dom[order_type::impl_dim2].size();
      l_pos += g_dom[order_type::impl_dim2].first();
    }

    l_pos *= sizeof(value_type);

    size_t l_size = l_dom.size();

    if (fseek(fd, l_pos, SEEK_SET) == -1)
    {
      fprintf(stderr, "save_view: error on fseek.\n");
      exit(1);
    }

    if ( swap_bytes )
    {
      // Make a copy in order to swap the bytes prior to writing to disk
      l_view_type l_view = vsip::impl::clone_view<l_view_type>(view.local());
      l_view = view.local();
      
      Ext_data<l_block_type, use_layout_type> ext(l_view.block());

      // Swap from either big- to little-endian, or vice versa.  We can do this
      // as if it were a 1-D view because it is guaranteed to be dense.
      value_type* p_data = ext.data();
      for (size_t i = 0; i < l_size; ++i)
        matlab::Swap_value<value_type,true>::swap(p_data++);

      if (fwrite(ext.data(), sizeof(value_type), l_size, fd) != l_size)
      {
        fprintf(stderr, "save_view: error reading file.\n");
        exit(1);
      }
    }
    else
    {
      l_view_type l_view = view.local();

      Ext_data<l_block_type, use_layout_type> ext(l_view.block());

      // Check that subblock is dense.
      assert(vsip::impl::is_ext_dense<order_type>(Dim, ext));

      if (fwrite(ext.data(), sizeof(value_type), l_size, fd) != l_size)
      {
        fprintf(stderr, "save_view: error reading file.\n");
        exit(1);
      }
    }
  }
}



/// Save a view to a file
///
/// Requires:
///   FILENAME to be filename.
///   VIEW to be a VSIPL++ view.

template <typename ViewT>
void
save_view(
   char const* filename,
   ViewT       view,
   bool        swap_bytes = false)
{
  if (subblock(view) != vsip::no_subblock)
  {
    FILE*  fd;
    
    if (!(fd = fopen(filename, "w")))
    {
      fprintf(stderr, "save_view: error opening '%s'.\n", filename);
      exit(1);
    }

    save_view(fd, view, swap_bytes);

    fclose(fd);
  }
}


/// Save a view to a file as another type
///
/// Requires:
///   T to be the desired type on disk.
///   FILENAME to be filename.
///   VIEW to be a VSIPL++ view.

template <typename T,
          typename ViewT>
void
save_view_as(
  char const* filename,
  ViewT       view,
  bool        swap_bytes = false)
{
  using vsip::impl::View_of_dim;

  typedef
    typename View_of_dim<ViewT::dim, T, vsip::Dense<ViewT::dim, T> >::type
    view_type;

  view_type disk_view = vsip::impl::clone_view<view_type>(view);

  disk_view = vsip::impl::view_cast<T>(view);
    
  vsip_csl::save_view(filename, disk_view, swap_bytes);
} 


} // namespace vsip_csl

#endif // VSIP_CSL_SAVE_VIEW_HPP
