//
// Copyright (c) 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_layout_hpp_
#define ovxx_layout_hpp_

#include <ovxx/support.hpp>
#include <vsip/layout.hpp>

namespace ovxx
{

/// Enum to indicate that an Applied_layout or Rt_layout object
/// will not be used and therefor should not be initialized when
/// constructed.
enum empty_layout_type { empty_layout};

///  Map an alignment to a pack-type
template <unsigned A> struct Aligned_packing;

#define OVXX_ALIGNED_PACKING(A)						\
template <>								\
struct Aligned_packing<A> { static pack_type const value = aligned_##A;};
  
OVXX_ALIGNED_PACKING(8)
OVXX_ALIGNED_PACKING(16)
OVXX_ALIGNED_PACKING(32)
OVXX_ALIGNED_PACKING(64)
OVXX_ALIGNED_PACKING(128)
OVXX_ALIGNED_PACKING(256)
OVXX_ALIGNED_PACKING(512)
OVXX_ALIGNED_PACKING(1024)

#undef OVXX_ALIGNED_PACKING

/// Runtime dimension-order (corresponds to compile-time tuples).
///
/// Member names are chosen to correspond to tuple's.
///
class Rt_tuple
{
public:
  Rt_tuple() : impl_dim0(0), impl_dim1(1), impl_dim2(2) {}
  Rt_tuple(dimension_type d0, dimension_type d1, dimension_type d2)
    : impl_dim0(d0), impl_dim1(d1), impl_dim2(d2)
  {}

  // Convenience constructor from a compile-time tuple.
  template <dimension_type D0,
	    dimension_type D1,
	    dimension_type D2>
  Rt_tuple(tuple<D0, D1, D2>)
    : impl_dim0(D0), impl_dim1(D1), impl_dim2(D2)
  {}

  dimension_type impl_dim0;
  dimension_type impl_dim1;
  dimension_type impl_dim2;
};

struct any_type;

template <pack_type>
struct is_packing_aligned
{
  static bool     const value = false;
  static unsigned const alignment = 0;
};

template <>
struct is_packing_aligned<aligned>
{
  static bool     const value = true;
  static unsigned const alignment = 0;
};

#define OVXX_ALIGNED(P)		      \
template <>			      \
struct is_packing_aligned<P>	      \
{				      \
  static bool const value = true;     \
  static unsigned const alignment = P;\
};

OVXX_ALIGNED(aligned_8)
OVXX_ALIGNED(aligned_16)
OVXX_ALIGNED(aligned_32)
OVXX_ALIGNED(aligned_64)
OVXX_ALIGNED(aligned_128)
OVXX_ALIGNED(aligned_256)
OVXX_ALIGNED(aligned_1024)

#undef OVXX_ALIGNED

inline bool is_aligned(pack_type p) { return p >= aligned && p <= aligned_1024;}

/// Runtime layout class encapsulating:
///
///  - Dimension
///  - Dimension order
///  - Packing format
///  - Complex format
template <dimension_type D>
struct Rt_layout
{
  // Dimension is fixed at compile-time.
  static dimension_type const dim = D;

  // Run-time layout.
  pack_type packing;
  Rt_tuple order;
  storage_format_type storage_format;
  unsigned alignment;  // Only valid if packing == aligned_*

  // Construct an empty Rt_layout object.
  Rt_layout() {}
  Rt_layout(pack_type p, Rt_tuple const &o, storage_format_type c, unsigned a = 0)
    : packing(p), order(o), storage_format(c), alignment(a) {}
};

template <dimension_type D>
bool operator==(Rt_layout<D> const &l1, Rt_layout<D> const &l2)
{
  return (l1.packing == l2.packing &&
	  l1.order == l2.order &&
	  l1.storage_format == l2.storage_format &&
	  l1.alignment == l2.alignment);
}

/// Applied_layout takes the layout policies encapsulated by a
/// Layout and applies them to map multi-dimensional indices into
/// memory offsets.
template <typename LP>
struct Applied_layout;

template <typename Order, storage_format_type C>
class Applied_layout<Layout<1, Order, dense, C> >
{
public:
  static dimension_type const dim = 1;
  typedef tuple<0, 1, 2>  order_type;
  static pack_type const packing = dense;
  static storage_format_type const storage_format = C;

  Applied_layout(length_type size) : size_(size) {}
  template <typename ExtentT>
  Applied_layout(ExtentT const &extent) : size_(size_of_dim(extent, 0)) {}

  index_type index(index_type i) const VSIP_NOTHROW { return i;}
  index_type index(Index<1> const &i) const VSIP_NOTHROW { return i[0];}
  stride_type stride(dimension_type) const VSIP_NOTHROW { return 1;}
  length_type size(dimension_type) const VSIP_NOTHROW { return size_;}
  length_type total_size() const VSIP_NOTHROW { return size_;}

private:
  length_type size_;
};

template <typename O, storage_format_type C>
class Applied_layout<Layout<1, O, unit_stride, C> >
  : public Applied_layout<Layout<1, O, dense, C> >
{
  // For 1D layout, all unit-stride packings are the same, and order is meaningless.
  typedef Applied_layout<Layout<1, O, dense, C> > base_type;
public:
  Applied_layout(length_type size) : base_type(size) {}
  template <typename ExtentT>
  Applied_layout(ExtentT const &extent) : base_type(extent) {}
};

#define OVXX_APPLIED_UNIT_STRIDE_LAYOUT(P)		      \
template <typename O, storage_format_type C>                  \
class Applied_layout<Layout<1, O, P, C> >		      \
  : public Applied_layout<Layout<1, O, dense, C> >	      \
{							      \
 typedef Applied_layout<Layout<1, O, dense, C> > base_type;   \
public:							      \
 Applied_layout(length_type size) : base_type(size) {}	      \
 template <typename ExtentT>				      \
 Applied_layout(ExtentT const &extent) : base_type(extent) {} \
};

OVXX_APPLIED_UNIT_STRIDE_LAYOUT(aligned)
OVXX_APPLIED_UNIT_STRIDE_LAYOUT(aligned_8)
OVXX_APPLIED_UNIT_STRIDE_LAYOUT(aligned_16)
OVXX_APPLIED_UNIT_STRIDE_LAYOUT(aligned_32)
OVXX_APPLIED_UNIT_STRIDE_LAYOUT(aligned_64)
OVXX_APPLIED_UNIT_STRIDE_LAYOUT(aligned_128)
OVXX_APPLIED_UNIT_STRIDE_LAYOUT(aligned_256)
OVXX_APPLIED_UNIT_STRIDE_LAYOUT(aligned_512)
OVXX_APPLIED_UNIT_STRIDE_LAYOUT(aligned_1024)

#undef OVXX_APPLIED_UNIT_STRIDE_LAYOUT

template <typename O, storage_format_type C>
class Applied_layout<Layout<1, O, any_packing, C> >
{
public:
  static dimension_type const dim = 1;
  typedef O order_type;
  static pack_type const packing = any_packing;
  static storage_format_type const storage_format = C;

  Applied_layout(length_type size, stride_type stride = 1)
    : size_(size), stride_(stride) {}

  template <typename ExtentT>
  Applied_layout(ExtentT const &extent)
    : size_(size_of_dim(extent, 0)), stride_(1) {}

  index_type index(index_type i) const VSIP_NOTHROW { return i * stride_;}
  index_type index(Index<1> const &i) const VSIP_NOTHROW { return i[0] * stride_;}
  stride_type stride(dimension_type) const VSIP_NOTHROW { return stride_;}
  length_type size(dimension_type) const VSIP_NOTHROW { return size_;}
  length_type total_size() const VSIP_NOTHROW { return size_;}

private:
  length_type size_;
  stride_type stride_;
};

template <storage_format_type C>
class Applied_layout<Layout<2, tuple<0, 1, 2>, dense, C> >
{
public:
  static dimension_type const dim = 2;
  typedef tuple<0, 1, 2>  order_type;
  static pack_type const packing = dense;
  static storage_format_type const storage_format = C;

  Applied_layout(length_type size0, length_type size1)
  {
    size_[0] = size0;
    size_[1] = size1;
  }

  template <typename ExtentT>
  Applied_layout(ExtentT const& extent)
  {
    size_[0] = size_of_dim(extent, 0);
    size_[1] = size_of_dim(extent, 1);
  }

  index_type index(index_type idx0, index_type idx1)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(idx0 < size_[0] && idx1 < size_[1]);
    return idx0 * size_[1] + idx1;
  }

  index_type index(Index<2> const &idx) const VSIP_NOTHROW
  { return idx[0] * size_[1] + idx[1];}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return d == 0 ? size_[1] : 1;}

  length_type size(dimension_type d) const VSIP_NOTHROW
  { return size_[d];}

  length_type total_size() const VSIP_NOTHROW
  { return size_[1] * size_[0];}

private:
  length_type size_[2];
};

template <storage_format_type C>
class Applied_layout<Layout<2, tuple<1, 0, 2>, dense, C> >
{
public:
  static dimension_type const dim = 2;
  typedef tuple<1, 0, 2>  order_type;
  static pack_type const packing = dense;
  static storage_format_type const storage_format = C;

  Applied_layout(length_type size0, length_type size1)
  {
    size_[0] = size0;
    size_[1] = size1;
  }

  template <typename ExtentT>
  Applied_layout(ExtentT const& extent)
  {
    size_[0] = size_of_dim(extent, 0);
    size_[1] = size_of_dim(extent, 1);
  }

  index_type index(index_type idx0, index_type idx1) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(idx0 < size_[0] && idx1 < size_[1]);
    return idx0 + idx1 * size_[0];
  }

  index_type index(Index<2> const &idx) const VSIP_NOTHROW
  { return idx[0] + idx[1] * size_[0];}

  stride_type stride(dimension_type d) const VSIP_NOTHROW
  { return d == 0 ? 1 : size_[0];}

  length_type size(dimension_type d) const VSIP_NOTHROW
  { return size_[d];}

  length_type total_size() const VSIP_NOTHROW
  { return size_[1] * size_[0];}

private:
  length_type size_[2];
};

#define OVXX_APPLIED_ALIGNED_LAYOUT(A)			   \
template <storage_format_type C>			   \
class Applied_layout<Layout<2, tuple<0, 1, 2>, A, C> >	   \
{							   \
public:							   \
  static dimension_type const dim = 2;			   \
  typedef tuple<0, 1, 2>         order_type;		   \
  static pack_type const packing = A;			   \
  static storage_format_type const storage_format = C;	   \
  						           \
  Applied_layout(length_type size0, length_type size1)	   \
  {							   \
    size_[0] = size0;					   \
    size_[1] = size1;					   \
    stride_ = size_[1];					   \
    if (stride_ % A != 0) stride_ += (A - stride_%A);	   \
  }							   \
  						           \
  template <typename ExtentT>				   \
  Applied_layout(ExtentT const& extent)			   \
  {							   \
    size_[0] = size_of_dim(extent, 0);			   \
    size_[1] = size_of_dim(extent, 1);			   \
    stride_ = size_[1];					   \
    if (stride_ % A != 0) stride_ += (A - stride_%A);	   \
  }							   \
							   \
  index_type index(index_type idx0, index_type idx1) const \
  {							   \
    OVXX_PRECONDITION(idx0 < size_[0] && idx1 < size_[1]); \
    return idx0 * stride_ + idx1;			   \
  }							   \
  index_type index(Index<2> const &idx) const		   \
  { return idx[0] * stride_ + idx[1];}			   \
  stride_type stride(dimension_type d) const 		   \
  { return d == 0 ? stride_ : 1;}			   \
  length_type size(dimension_type d) const		   \
  { return size_[d];}					   \
  length_type total_size() const			   \
  { return stride_ * size_[0] - (stride_-size_[1]);}	   \
							   \
private:						   \
  length_type size_[2];					   \
  length_type stride_;					   \
};

OVXX_APPLIED_ALIGNED_LAYOUT(aligned_8)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_16)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_32)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_64)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_128)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_256)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_512)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_1024)

#undef OVXX_APPLIED_ALIGNED_LAYOUT

#define OVXX_APPLIED_ALIGNED_LAYOUT(A)			   \
template <storage_format_type C>			   \
class Applied_layout<Layout<2, tuple<1, 0, 2>, A, C> >	   \
{							   \
public:							   \
  static dimension_type const dim = 2;			   \
  typedef tuple<1, 0, 2> order_type;			   \
  static pack_type const packing = A;			   \
  static storage_format_type const storage_format = C;	   \
							   \
  Applied_layout(length_type size0, length_type size1)	   \
  {							   \
    size_[0] = size0;					   \
    size_[1] = size1;					   \
    stride_ = size_[1];					   \
    if (stride_ % A != 0) stride_ += (A - stride_%A);	   \
  }							   \
							   \
  template <typename ExtentT>				   \
  Applied_layout(ExtentT const& extent)			   \
  {							   \
    size_[0] = size_of_dim(extent, 0);			   \
    size_[1] = size_of_dim(extent, 1);			   \
    stride_ = size_[0];					   \
    if (stride_ % A != 0) stride_ += (A - stride_%A);	   \
  }							   \
							   \
  index_type index(index_type idx0, index_type idx1) const \
  {							   \
    OVXX_PRECONDITION(idx0 < size_[0] && idx1 < size_[1]); \
    return idx0 + idx1 * stride_;			   \
  }							   \
  index_type index(Index<2> const &idx) const		   \
  { return idx[0] + idx[1] * stride_;}			   \
  stride_type stride(dimension_type d) const		   \
  { return d == 1 ? stride_ : 1;}			   \
  length_type size(dimension_type d) const		   \
  { return size_[d];}					   \
  length_type total_size() const			   \
  { return stride_ * size_[1] - (stride_-size_[0]);}	   \
							   \
private:						   \
  length_type size_[2];					   \
  length_type stride_;					   \
};

OVXX_APPLIED_ALIGNED_LAYOUT(aligned_8)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_16)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_32)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_64)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_128)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_256)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_512)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_1024)

#undef OVXX_APPLIED_ALIGNED_LAYOUT

template <typename O, storage_format_type C>
class Applied_layout<Layout<2, O, any_packing, C> >
{
public:
  static dimension_type const dim = 2;
  typedef O order_type;
  static pack_type const packing = any_packing;
  static storage_format_type const storage_format = C;

  Applied_layout(length_type size0, stride_type stride0,
		 length_type size1, stride_type stride1)
  {
    size_[0] = size0;
    size_[1] = size1;
    stride_[0] = stride0;
    stride_[1] = stride1;
  }

  index_type index(index_type i, index_type j) const
  { return i * stride_[0] + j * stride_[1];}
  index_type index(Index<2> const &i) const
  { return i[0] * stride_[0] + i[1] * stride_[1];}
  stride_type stride(dimension_type d) const
  { return stride_[d];}
  length_type size(dimension_type d) const
  { return size_[d];}
  length_type total_size() const
  { return size_[0] * size_[1];}

private:
  length_type size_[dim];
  stride_type stride_[dim];
};

template <dimension_type D0,
	  dimension_type D1,
	  dimension_type D2,
	  storage_format_type C>
class Applied_layout<Layout<3, tuple<D0, D1, D2>, dense, C> >
{
public:
  static dimension_type const dim = 3;
  typedef tuple<D0, D1, D2> order_type;
  static pack_type const packing = dense;
  static storage_format_type const storage_format = C;

  template <typename ExtentT>
  Applied_layout(ExtentT const& extent)
  {
    size_[0] = size_of_dim(extent, 0);
    size_[1] = size_of_dim(extent, 1);
    size_[2] = size_of_dim(extent, 2);
  }

  index_type index(Index<3> const &idx) const
  {
    OVXX_PRECONDITION(idx[0] < size_[0] && idx[1] < size_[1] && idx[2] < size_[2]);
    return idx[D0]*size_[D1]*size_[D2] + idx[D1]*size_[D2] + idx[D2];
  }

  index_type index(index_type idx0, index_type idx1, index_type idx2) const
  { return index(Index<3>(idx0, idx1, idx2));}

  stride_type stride(dimension_type d) const
  {
    return d == D2 ? 1 : d == D1 ? size_[D2] : size_[D1] * size_[D2];
  }

  length_type size(dimension_type d) const
  { return size_[d];}

  length_type total_size() const
  { return size_[2] * size_[1] * size_[0];}

private:
  length_type size_[3];
};

#define OVXX_APPLIED_ALIGNED_LAYOUT(A)			   \
template <dimension_type D0,				   \
	  dimension_type D1,			           \
	  dimension_type D2,				   \
	  storage_format_type C>			   \
class Applied_layout<Layout<3, tuple<D0, D1, D2>, A, C> >  \
{							   \
public:							   \
  static dimension_type const dim = 3;			   \
  typedef tuple<D0, D1, D2> order_type;			   \
  static pack_type const packing = A;			   \
  static storage_format_type const storage_format = C;	   \
							   \
  template <typename ExtentT>				   \
  Applied_layout(ExtentT const& extent)			   \
  {							   \
    size_[0] = size_of_dim(extent, 0);			   \
    size_[1] = size_of_dim(extent, 1);			   \
    size_[2] = size_of_dim(extent, 2);			   \
    stride_[D2] = 1;					   \
    stride_[D1] = size_[D2];				   \
    if (stride_[D1] % A != 0)				   \
      stride_[D1] += (A - stride_[D1]%A);		   \
    stride_[D0] = size_[D1] * stride_[D1];		   \
  }							   \
							   \
  index_type index(Index<3> const &idx) const		   \
  {							   \
    OVXX_PRECONDITION(idx[0] < size_[0] &&		   \
		      idx[1] < size_[1] &&		   \
		      idx[2] < size_[2]);		   \
    return idx[D0]*stride_[D0] + idx[D1]*stride_[D1] + idx[D2];		\
  }		  					   \
  index_type index(index_type idx0,		  	   \
		   index_type idx1,		      	   \
 index_type idx2)				   	   \
    const						   \
  { return index(Index<3>(idx0, idx1, idx2));}		   \
  stride_type stride(dimension_type d) const		   \
  { return stride_[d];}					   \
  length_type size(dimension_type d) const	           \
  { return size_[d];}					   \
  length_type total_size() const			   \
  { return size_[D0] * stride_[D0];}			   \
  							   \
 private:						   \
  length_type size_  [3];				   \
  stride_type stride_[3];				   \
};

OVXX_APPLIED_ALIGNED_LAYOUT(aligned_8)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_16)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_32)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_64)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_128)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_256)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_512)
OVXX_APPLIED_ALIGNED_LAYOUT(aligned_1024)

#undef OVXX_APPLIED_ALIGNED_LAYOUT

/// Applied run-time layout.
/// This object gets created for run-time extdata access.
/// Efficiency is important to reduce library interface overhead.
/// Don't store the whole Rt_layout, only the parts we need:
/// the storage_format and part of the dimension-order.
template <dimension_type D>
class Applied_layout<Rt_layout<D> >
{
public:
  static dimension_type const dim = D;

  // Construct an empty Applied_layout.  Used when it is known that object
  // will not be used.
  Applied_layout(empty_layout_type)
    : cformat_(interleaved_complex)
  {
    for (dimension_type d = 0; d < D; ++d)
    {
      order_[d] = 0;
      size_[d] = 0;
      stride_[d] = 1;
    }
  }

  /// Construct Applied_layout object.
  ///
  /// Template parameters:
  ///
  ///   :ExtentT: a type capable of encoding an extent (Length or Domain)
  ///
  /// Arguments:
  ///
  ///   :layout:    the run-time layout.
  ///   :extent:    the extent of the data to layout.
  ///   :elem_size: the size of a data element (in bytes).
  template <typename ExtentT>
  Applied_layout(Rt_layout<D> const& layout,
		 ExtentT const &extent,
		 length_type elem_size = 1)
    : cformat_(layout.storage_format)
  {
    OVXX_PRECONDITION(!ovxx::is_aligned(layout.packing) ||
	   layout.alignment == 0 || layout.alignment % elem_size == 0);

    for (dimension_type d=0; d<D; ++d)
      size_[d] = size_of_dim(extent, d);

    if (D == 3)
    {
      order_[2] = layout.order.impl_dim2;
      order_[1] = layout.order.impl_dim1;
      order_[0] = layout.order.impl_dim0;

      stride_[order_[2]] = 1;
      stride_[order_[1]] = size_[order_[2]];
      if (is_aligned(layout.packing) && layout.alignment != 0 &&
	  (elem_size*stride_[order_[1]]) % layout.alignment != 0)
      {
	stride_type adjust =
	  layout.alignment - (stride_[order_[1]] * elem_size)%layout.alignment;
	OVXX_PRECONDITION(adjust > 0 && adjust % elem_size == 0);
	adjust /= elem_size;
	stride_[order_[1]] += adjust;
	OVXX_PRECONDITION((stride_[order_[1]] * elem_size)%layout.alignment == 0);
      }
      stride_[order_[0]] = size_[order_[1]] * stride_[order_[1]];
    }
    else if (D == 2)
    {
      // Copy only the portion of the dimension-order that we use.
      order_[1] = layout.order.impl_dim1;
      order_[0] = layout.order.impl_dim0;

      stride_[order_[1]] = 1;
      stride_[order_[0]] = size_[order_[1]];

      if (is_aligned(layout.packing) && layout.alignment != 0 &&
	  (elem_size*stride_[order_[0]]) % layout.alignment != 0)
      {
	stride_type adjust =
	  layout.alignment - (stride_[order_[0]] * elem_size)%layout.alignment;
	OVXX_PRECONDITION(adjust > 0 && adjust % elem_size == 0);
	adjust /= elem_size;
	stride_[order_[0]] += adjust;
	OVXX_PRECONDITION((stride_[order_[0]] * elem_size)%layout.alignment == 0);
      }
    }
    else  // (D == 1)
    {
      // Copy only the portion of the dimension-order that we use.
      order_[0] = layout.order.impl_dim0;

      stride_[0] = 1;
    }
  }

  index_type index(Index<D> const &idx) const VSIP_NOTHROW
  {
    if (D == 3)
    {
      OVXX_PRECONDITION(idx[0] < size_[0] && idx[1] < size_[1] && idx[2] < size_[2]);
      return idx[order_[0]]*stride_[order_[0]] +
	     idx[order_[1]]*stride_[order_[1]] + 
	     idx[order_[2]];
    }
    else if (D == 2)
    {
      OVXX_PRECONDITION(idx[0] < size_[0] && idx[1] < size_[1]);
      return idx[order_[0]]*stride_[order_[0]] +
	     idx[order_[1]];
    }
    else // (D == 1)
    {
      OVXX_PRECONDITION(idx[0] < size_[0]);
      return idx[0];
    }
  }
  stride_type stride(dimension_type d) const VSIP_NOTHROW { return stride_[d];}
  length_type size(dimension_type d) const VSIP_NOTHROW { return size_[d];}

  length_type total_size() const VSIP_NOTHROW
  { return size_[order_[0]] * stride_[order_[0]];}
  storage_format_type storage_format() const VSIP_NOTHROW { return cformat_;}

private:
  storage_format_type cformat_;
  dimension_type order_[D];
  length_type size_[D];
  stride_type stride_[D];
};

template <dimension_type D, typename O, pack_type P, storage_format_type C>
inline storage_format_type 
layout_storage_format(Layout<D, O, P, C>)
{ return C;}

template <dimension_type D>
inline storage_format_type 
layout_storage_format(Rt_layout<D> const &rtl) 
{ return rtl.storage_format;}

template <dimension_type D>
inline storage_format_type 
layout_storage_format(Applied_layout<Rt_layout<D> > const& appl)
{ return appl.storage_format();}

template <dimension_type D, typename O, pack_type P, storage_format_type C>
inline pack_type 
layout_packing(Layout<D, O, P, C>)
{ return P;}

template <dimension_type D>
inline pack_type 
layout_packing(Rt_layout<D> const &rtl)
{ return rtl.packing;}

template <dimension_type D, typename O, pack_type P, storage_format_type C>
inline unsigned 
layout_alignment(Layout<D, O, P, C>)
{ return is_packing_aligned<P>::alignment;}

template <dimension_type D>
inline unsigned 
layout_alignment(Rt_layout<D> const &rtl)
{ return rtl.alignment;}

template <dimension_type D, typename O, pack_type P, storage_format_type C>
inline dimension_type 
layout_nth_dim(Layout<D, O, P, C> const&, dimension_type const d)
{
  if      (d == 0) return O::impl_dim0;
  else if (d == 1) return O::impl_dim1;
  else             return O::impl_dim2;
}

template <dimension_type D>
inline dimension_type 
layout_nth_dim(Rt_layout<D> const& rtl, dimension_type const d)
{
  if      (d == 0) return rtl.order.impl_dim0;
  else if (d == 1) return rtl.order.impl_dim1;
  else             return rtl.order.impl_dim2;
}

template <dimension_type D, typename B>
inline Rt_layout<D>
block_layout(B const&)
{
  return Rt_layout<D>(get_block_layout<B>::packing,
		      Rt_tuple(typename get_block_layout<B>::order_type()),
		      get_block_layout<B>::storage_format,
		      is_packing_aligned<get_block_layout<B>::packing>::alignment);
}

} // namespace ovxx

#endif
