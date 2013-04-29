/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/load_view.hpp
    @author  Jules Bergmann
    @date    2005-09-30
    @brief   VSIPL++ CodeSourcery Library: Utility to load a view from disk.
*/

#ifndef VSIP_CSL_LOAD_VIEW_HPP
#define VSIP_CSL_LOAD_VIEW_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <string.h>
#include <errno.h>
#include <memory>

#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <vsip/dda.hpp>
#include <vsip/core/noncopyable.hpp>
#include <vsip/core/working_view.hpp>
#include <vsip/core/view_cast.hpp>

#include <vsip_csl/endian.hpp>
#include <vsip_csl/dda.hpp>

namespace vsip_csl
{

/// Load values from a file descriptor into a VSIPL++ view.
/// Note: assumes complex data on disk is always interleaved. 
template <typename ViewT>
void
load_view(FILE* fd, ViewT view, bool  swap_bytes = false)
{
  using namespace vsip;
  using vsip::get_block_layout;
  using vsip::impl::adjust_layout_storage_format;

  if (subblock(view) != vsip::no_subblock && subblock_domain(view).size() > 0)
  {
    vsip::dimension_type const Dim = ViewT::dim;

    typedef typename ViewT::value_type       value_type;
    typedef typename ViewT::local_type       l_view_type;
    typedef typename l_view_type::block_type l_block_type;
    typedef typename get_block_layout<l_block_type>::order_type order_type;

    typedef typename get_block_layout<l_block_type>::type layout_type;
    typedef typename adjust_layout_storage_format<array, layout_type>::type
      use_layout_type;

    l_view_type l_view = view.local();

    vsip::Domain<Dim> g_dom = global_domain(view);
    vsip::Domain<Dim> l_dom = subblock_domain(view);

    dda::Data<l_block_type, dda::out, use_layout_type> data(l_view.block());

    // Check that subblock is dense.
    if (!vsip_csl::dda::is_data_dense<order_type>(Dim, data))
      VSIP_IMPL_THROW(vsip::impl::unimplemented(
	"load_view can only handle dense subblocks"));

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

    std::size_t l_size = l_dom.size();

    if (fseek(fd, l_pos, SEEK_SET) == -1)
      VSIP_IMPL_THROW(std::runtime_error("load_view: error on fseek."));

    std::size_t l_read = fread(data.ptr(), sizeof(value_type), l_size, fd);
    if (l_read != l_size)
      VSIP_IMPL_THROW(std::runtime_error("load_view: error on fread."));

    // Swap from either big- to little-endian, or vice versa.  We can do this
    // as if it were a 1-D view because it is guaranteed to be dense.
    if ( swap_bytes )
    {
      value_type* p_data = data.ptr();
      for (std::size_t i = 0; i < l_size; ++i)
        bswap(*p_data++);
    }
  }
}



/// Load values from a file into a VSIPL++ view.
template <typename ViewT>
void
load_view(char const* filename, ViewT view, bool swap_bytes = false)
{
  if (subblock(view) != vsip::no_subblock && subblock_domain(view).size() > 0)
  {
    FILE*  fd;
    
    if (!(fd = fopen(filename, "r")))
      VSIP_IMPL_THROW(std::runtime_error("load_view: error opening file."));
    load_view(fd, view, swap_bytes);
    fclose(fd);
  }
}



/// Load a view from a file with another value type.
///
/// Template parameters:
///   T: the type on disk.
///
/// Arguments:
///   filename: the name of the file
///   view: the view to be loaded
///
/// Note: 
///   All other layout parameters (dimension-order and parallel
///   distribution) are preserved.
template <typename T,
          typename ViewT>
void
load_view_as(char const* filename, ViewT view, bool swap_bytes = false)
{
  using vsip::get_block_layout;
  using vsip::impl::clone_view;

  typedef typename ViewT::block_type                    block_type;
  typedef typename get_block_layout<block_type>::order_type order_type;
  typedef typename ViewT::block_type::map_type          map_type;

  typedef vsip::Dense<ViewT::dim, T, order_type, map_type> new_block_type;
  typedef typename vsip::impl::view_of<new_block_type>::type view_type;

  view_type disk_view = clone_view<view_type>(view, view.block().map());

  load_view(filename, disk_view, swap_bytes);

  view = vsip::impl::view_cast<typename ViewT::value_type>(disk_view);
} 




/// Load values from a file into a VSIPL++ view.

/// Requires
///   DIM to be the dimension of the data/view,
///   T to be the value type of the data/view
///   ORDERT to be the dimension order of the data and view
///      (row-major by default).
///   MAPT to be the mapping of the view
///      (Local_map by default).

template <vsip::dimension_type Dim,
	  typename          T,
	  typename          OrderT = typename vsip::impl::Row_major<Dim>::type,
	  typename          MapT = vsip::Local_map>
class Load_view : vsip::impl::Non_copyable
{
public:
  typedef T value_type;
  typedef vsip::Dense<Dim, T, OrderT, MapT> block_type;
  typedef typename vsip::impl::view_of<block_type>::type view_type;

public:
  Load_view(char const*              filename,
	    vsip::Domain<Dim> const& dom,
            MapT const&              map = MapT(),
            bool                     swap_bytes = false)
    : block_ (dom, map),
      view_  (block_)
  {
    load_view(filename, view_, swap_bytes);
  }



  Load_view(FILE*                    fd,
	    vsip::Domain<Dim> const& dom,
            MapT const&              map = MapT(),
            bool                     swap_bytes = false)
    : block_ (dom, map),
      view_  (block_)
  {
    load_view(fd, view_, swap_bytes);
  }

  view_type view() { return view_; }

private:
  block_type                block_;
  view_type                 view_;
};





} // namespace vsip_csl

#endif // VSIP_CSL_LOAD_VIEW_HPP
