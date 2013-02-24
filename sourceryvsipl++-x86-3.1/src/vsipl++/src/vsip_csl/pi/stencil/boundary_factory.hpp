/* Copyright (c) 2007 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Harness to create boundary blocks.

#ifndef vsip_csl_pi_stencil_boundary_factory_hpp_
#define vsip_csl_pi_stencil_boundary_factory_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/core/strided.hpp>
#include <vsip/core/block_traits.hpp>

namespace vsip_csl
{
namespace pi
{
namespace stencil
{

enum Position { left, right, top, bottom,
                top_left, top_right, bottom_left, bottom_right};
enum Padding { zero, constant};

template <typename B, Position P, typename K> struct Boundary_traits;

template <typename B, typename K>
struct Boundary_traits<B, left, K>
{
  typedef typename vsip::get_block_layout<B>::order_type order_type;
  typedef vsip::Layout<2, order_type, vsip::dense> layout_type;
  
  // the size of the boundary block
  static vsip::Domain<2> size(B const &block, K const &k)
  {
    return vsip::Domain<2>(block.size(2, 0), k.size(1) + k.origin(1) - 1);
  }
  // the subblock the boundary block needs to mirror
  static vsip::Domain<2> src_sub_domain(B const &b, K const &k)
  {
    return vsip::Domain<2>(b.size(2, 0), k.size(1) - 1);
  }
  // the destination subblock containing the mirror.
  static vsip::Domain<2> dst_sub_domain(B const &b, K const &k)
  {
    return vsip::Domain<2>(b.size(2, 0),
                           vsip::Domain<1>(k.origin(1), 1, k.size(1) - 1));
  }
};


template <typename B, typename K>
struct Boundary_traits<B, right, K>
{
  typedef typename vsip::get_block_layout<B>::order_type order_type;
  typedef vsip::Layout<2, order_type, vsip::dense> layout_type;
  
  // the size of the boundary block
  static vsip::Domain<2> size(B const &block, K const &k)
  {
    return vsip::Domain<2>(block.size(2, 0), 2 * (k.size(1) - 1) - k.origin(1));
  }
  // the subblock the boundary block needs to mirror
  static vsip::Domain<2> src_sub_domain(B const &b, K const &k)
  {
    return vsip::Domain<2>
      (b.size(2, 0), vsip::Domain<1>(b.size(2, 1) - k.size(1) + 1, 1, k.size(1) - 1));
  }
  // the destination subblock containing the mirror.
  static vsip::Domain<2> dst_sub_domain(B const &b, K const &k)
  {
    return vsip::Domain<2>(b.size(2, 0), k.size(1) - 1);
  }
};

template <typename B, typename K>
struct Boundary_traits<B, top, K>
{
  typedef typename vsip::get_block_layout<B>::order_type order_type;
  typedef vsip::Layout<2, order_type, vsip::dense> layout_type;
  
  // the size of the boundary block
  static vsip::Domain<2> size(B const &block, K const &k)
  {
    return vsip::Domain<2>(k.size(0) + k.origin(0) - 1, block.size(2, 1));
  }
  // the subblock the boundary block needs to mirror
  static vsip::Domain<2> src_sub_domain(B const &b, K const &k)
  {
    return vsip::Domain<2>(k.size(0) - 1, b.size(2, 1));
  }
  // the destination subblock containing the mirror.
  static vsip::Domain<2> dst_sub_domain(B const &b, K const &k)
  {
    return vsip::Domain<2>
      (vsip::Domain<1>(k.origin(0), 1, k.size(0) - 1), b.size(2, 1));
  }
};


template <typename B, typename K>
struct Boundary_traits<B, bottom, K>
{
  typedef typename vsip::get_block_layout<B>::order_type order_type;
  typedef vsip::Layout<2, order_type, vsip::dense> layout_type;
  
  // the size of the boundary block
  static vsip::Domain<2> size(B const &block, K const &k)
  {
    return vsip::Domain<2>(2 * (k.size(0) - 1) - k.origin(0), block.size(2, 1));
  }
  // the subblock the boundary block needs to mirror
  static vsip::Domain<2> src_sub_domain(B const &b, K const &k)
  {
    return vsip::Domain<2>
      (vsip::Domain<1>(b.size(2, 0) - k.size(0) + 1, 1, k.size(0) - 1), b.size(2, 1));
  }
  // the destination subblock containing the mirror.
  static vsip::Domain<2> dst_sub_domain(B const &b, K const &k)
  {
    return vsip::Domain<2>(k.size(0) - 1, b.size(2, 1));
  }
};

template <typename B, typename K>
struct Boundary_traits<B, top_left, K>
{
  typedef typename vsip::get_block_layout<B>::order_type order_type;
  typedef vsip::Layout<2, order_type, vsip::dense> layout_type;
  
  // the size of the boundary block
  static vsip::Domain<2> size(B const &, K const &k)
  {
    return vsip::Domain<2>
      (k.size(0) + k.origin(0) - 1, k.size(1) + k.origin(1) - 1);
  }
  // the subblock the boundary block needs to mirror
  static vsip::Domain<2> src_sub_domain(B const &, K const &k)
  {
    return vsip::Domain<2>(k.size(0) - 1, k.size(1) - 1);
  }
  // the destination subblock containing the mirror.
  static vsip::Domain<2> dst_sub_domain(B const &, K const &k)
  {
    return vsip::Domain<2>(vsip::Domain<1>(k.origin(0), 1, k.size(0) - 1),
                     vsip::Domain<1>(k.origin(1), 1, k.size(1) - 1));
  }
};

template <typename B, typename K>
struct Boundary_traits<B, top_right, K>
{
  typedef typename vsip::get_block_layout<B>::order_type order_type;
  typedef vsip::Layout<2, order_type, vsip::dense> layout_type;
  
  // the size of the boundary block
  static vsip::Domain<2> size(B const &, K const &k)
  {
    return vsip::Domain<2>(k.size(0) + k.origin(0) - 1,
                     2 * (k.size(1) - 1) - k.origin(1));
  }
  // the subblock the boundary block needs to mirror
  static vsip::Domain<2> src_sub_domain(B const &b, K const &k)
  {
    return vsip::Domain<2>
      (k.size(0) + 1,
       vsip::Domain<1>(b.size(2, 1) - k.size(1) + 1, 1, k.size(1) - 1));
  }
  // the destination subblock containing the mirror.
  static vsip::Domain<2> dst_sub_domain(B const &, K const &k)
  {
    return vsip::Domain<2>
      (vsip::Domain<1>(k.origin(0), 1, k.size(0) - 1), k.size(1) - 1);
  }
};

template <typename B, typename K>
struct Boundary_traits<B, bottom_left, K>
{
  typedef typename vsip::get_block_layout<B>::order_type order_type;
  typedef vsip::Layout<2, order_type, vsip::dense> layout_type;
  
  // the size of the boundary block
  static vsip::Domain<2> size(B const &, K const &k)
  {
    return vsip::Domain<2>(2 * (k.size(0) - 1) - k.origin(0),
                           k.size(1) + k.origin(1) - 1);
  }
  // the subblock the boundary block needs to mirror
  static vsip::Domain<2> src_sub_domain(B const &b, K const &k)
  {
    return vsip::Domain<2>
      (vsip::Domain<1>(b.size(2, 0) - k.size(0) + 1, 1, k.size(0) - 1),
       k.size(1) - 1);
  }
  // the destination subblock containing the mirror.
  static vsip::Domain<2> dst_sub_domain(B const &, K const &k)
  {
    return vsip::Domain<2>
      (k.size(0) - 1, vsip::Domain<1>(k.origin(1), 1, k.size(1) - 1));
  }
};

template <typename B, typename K>
struct Boundary_traits<B, bottom_right, K>
{
  typedef typename vsip::get_block_layout<B>::order_type order_type;
  typedef vsip::Layout<2, order_type, vsip::dense> layout_type;
  
  // the size of the boundary block
  static vsip::Domain<2> size(B const &, K const &k)
  {
    return vsip::Domain<2>(2 * (k.size(0) - 1) - k.origin(0),
                           2 * (k.size(1) - 1) - k.origin(1));
  }
  // the subblock the boundary block needs to mirror
  static vsip::Domain<2> src_sub_domain(B const &b, K const &k)
  {
    return vsip::Domain<2>
      (vsip::Domain<1>(b.size(2, 0) - k.size(0) + 1, 1, k.size(0) - 1),
       vsip::Domain<1>(b.size(2, 1) - k.size(1) + 1, 1, k.size(1) - 1));
  }
  // the destination subblock containing the mirror.
  static vsip::Domain<2> dst_sub_domain(B const& /*b*/, K const &k)
  {
    return vsip::Domain<2>(k.size(0) - 1, k.size(1) - 1);
  }
};

template <typename B, Position P, typename K>
struct Boundary_factory
{
  typedef Boundary_traits<B, P, K> traits;
  typedef vsip::impl::Strided<2, typename B::value_type,
                           typename traits::layout_type> block_type;
  typedef vsip::Matrix<typename B::value_type, block_type> view_type;

  static view_type
  create(B const &b, K const &k, Padding p)
  {
    using namespace vsip;

    Domain<2> size = traits::size(b, k);
    view_type view(size[0].length(), size[1].length());
    // set self to 0
    if (p == zero)
      vsip::impl::Block_fill<2, block_type>::exec(view.block(), 0);

    // assign boundary values
#if 0
    std::cout << "size " << size[0].length() << ' ' << size[1].length() << std::endl;
    std::cout << "src subdomain: "
              << traits::src_sub_domain(b, k)[0].first() << ' '
              << traits::src_sub_domain(b, k)[0].length() << ' '
              << traits::src_sub_domain(b, k)[1].first() << ' ' 
              << traits::src_sub_domain(b, k)[1].length() << std::endl;
    std::cout << "dst subdomain: " 
              << traits::dst_sub_domain(b, k)[0].first() << ' '
              << traits::dst_sub_domain(b, k)[0].length() << ' '
              << traits::dst_sub_domain(b, k)[1].first() << ' '
              << traits::dst_sub_domain(b, k)[1].length() << std::endl;
    std::cout << "input " << b.size(2, 0) << ' ' << b.size(2, 1) << std::endl;
    std::cout << "boundary " << view.block().size(2, 0) << ' ' << view.block().size(2, 1) << std::endl;
#endif
    vsip::impl::Subset_block<B> src(traits::src_sub_domain(b, k), const_cast<B &>(b));
    vsip::impl::Subset_block<block_type> dst(traits::dst_sub_domain(b, k), view.block());
    vsip::impl::assign<2>(dst, src);

    // TODO: handle constant padding
    if (p == constant)
      {}

#if 0
    for (unsigned int y = 0; y != size(2, 0); ++y)
    {
      for (unsigned int x = 0; x != size(2, 1); ++x)
        std::cout << get(y, x) << ' ';
      std::cout << std::endl;
    }
#endif

    return view;
  }
};

} // namespace vsip_csl::pi::stencil
} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif
