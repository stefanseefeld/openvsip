//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/support.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/static_assert.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;

using vsip::impl::Length;
using vsip::impl::extent;
using vsip::impl::valid;
using vsip::impl::next;



/***********************************************************************
  Definitions
***********************************************************************/

/// Convert a multi-dimensional index into a linear index, using
/// a specific dimension-order.

template <typename OrderT>
index_type
linear_index(Length<1> const& /*ext*/, Index<1> const& idx)
{
  VSIP_IMPL_STATIC_ASSERT(OrderT::impl_dim0 == 0);
  return idx[OrderT::impl_dim0];
}

template <typename OrderT>
index_type
linear_index(Length<2> const& ext, Index<2> const& idx)
{
  VSIP_IMPL_STATIC_ASSERT(OrderT::impl_dim0 < 2);
  VSIP_IMPL_STATIC_ASSERT(OrderT::impl_dim1 < 2);

  return (idx[OrderT::impl_dim1] + ext[OrderT::impl_dim1]*
          idx[OrderT::impl_dim0]);
}

template <typename OrderT>
index_type
linear_index(Length<3> const& ext, Index<3> const& idx)
{
  VSIP_IMPL_STATIC_ASSERT(OrderT::impl_dim0 < 3);
  VSIP_IMPL_STATIC_ASSERT(OrderT::impl_dim1 < 3);
  VSIP_IMPL_STATIC_ASSERT(OrderT::impl_dim2 < 3);

  return (idx[OrderT::impl_dim2] + ext[OrderT::impl_dim2]*
         (idx[OrderT::impl_dim1] + ext[OrderT::impl_dim1]*
          idx[OrderT::impl_dim0]));
}



/// Test that the arbitrary next() traversal.

template <typename       OrderT,
	  dimension_type Dim>
void
test_traversal(Domain<Dim> const& dom)
{
  Length<Dim> ext = extent(dom);

  index_type count = 0;
  for (Index<Dim> idx; valid(ext, idx); next<OrderT>(ext, idx))
  {
    test_assert(linear_index<OrderT>(ext, idx) == count);
    count++;
  }
}



/// Test that the default next() traversal is row-major.

template <dimension_type Dim>
void
test_traversal_default(Domain<Dim> const& dom)
{
  typedef typename vsip::impl::Row_major<Dim>::type order_type;
  Length<Dim> ext = extent(dom);

  index_type count = 0;
  for (Index<Dim> idx; valid(ext, idx); next(ext, idx))
  {
    test_assert(linear_index<order_type>(ext, idx) == count);
    count++;
  }
}



int
main()
{
  test_traversal<row1_type>(Domain<1>(5));

  test_traversal<row2_type>(Domain<2>(5, 7));
  test_traversal<col2_type>(Domain<2>(5, 7));

  test_traversal<row3_type>(Domain<3>(5, 7, 3));
  test_traversal<col3_type>(Domain<3>(5, 7, 3));
  test_traversal<tuple<2, 0, 1> >(Domain<3>(5, 7, 3));

  test_traversal_default(Domain<1>(5));
  test_traversal_default(Domain<2>(5, 7));
  test_traversal_default(Domain<3>(5, 7, 3));
}
