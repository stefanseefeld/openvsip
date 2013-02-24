/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

#ifndef VSIP_CSL_BLOCK_MARSHAL_HPP
#define VSIP_CSL_BLOCK_MARSHAL_HPP

#include <vsip/dense.hpp>
#include <vsip_csl/strided.hpp>
#include <vsip_csl/dda.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/inttypes.hpp>

namespace vsip_csl
{
namespace impl
{
namespace block_marshal
{
enum { CHAR, INT, FLOAT, DOUBLE, CFLOAT, CDOUBLE};
enum { SPLIT = vsip::impl::cmplx_split_fmt,
       INTERLEAVED = vsip::impl::cmplx_inter_fmt};

struct Descriptor
{
  vsip::impl::uint8_type value_type;
  vsip::impl::uint8_type dimensions;
  vsip::impl::uint8_type complex_type;
  vsip::impl::uint8_type block_type;
  vsip::impl::uint32_type size0;
  vsip::impl::uint32_type stride0;
  vsip::impl::uint32_type size1;
  vsip::impl::uint32_type stride1;
  vsip::impl::uint32_type size2;
  vsip::impl::uint32_type stride2;
};

template <typename Block,
	  bool Complex = vsip::impl::Is_complex<typename Block::value_type>::value>
struct Rebinder;

template <typename Block>
struct Rebinder<Block, false>
{
  typedef Block block_type;
  typedef typename block_type::value_type value_type;
  static vsip::dimension_type const dim = block_type::dim;
  static void rebind(block_type &b,
		     void *data, void *, vsip::impl::uint8_type,
		     vsip::Domain<dim> const &domain)
  {
    b.rebind(static_cast<value_type *>(data), domain);
    b.admit();
  }
};

template <typename Block>
struct Rebinder<Block, true>
{
  typedef Block block_type;
  typedef typename block_type::value_type value_type;
  typedef typename vsip::impl::Scalar_of<value_type>::type scalar_type;
  static vsip::dimension_type const dim = block_type::dim;
  static void rebind(block_type &b,
		     void *real, void *imag, vsip::impl::uint8_type format,
		     vsip::Domain<dim> const &domain)
  {
    if (format == SPLIT)
      b.rebind(static_cast<scalar_type *>(real),
	       static_cast<scalar_type *>(imag),
               domain);
    else
      b.rebind(static_cast<scalar_type *>(real), domain);
    b.admit();
  }
};

template <typename Block>
void dda_marshal(Block const &b, void **data, Descriptor &d)
{
  using namespace vsip::impl;
  typedef Block block_type;
  typedef typename block_type::value_type value_type;
  if (is_same<value_type, char>::value) d.value_type = CHAR;
  else if (is_same<value_type, int>::value) d.value_type = INT;
  else if (is_same<value_type, float>::value) d.value_type = FLOAT;
  else if (is_same<value_type, double>::value) d.value_type = DOUBLE;
  else if (is_same<value_type, complex<float> >::value) d.value_type = CFLOAT;
  else if (is_same<value_type, complex<double> >::value) d.value_type = CDOUBLE;
  else VSIP_IMPL_THROW(unimplemented("unsupported value-type"));
  d.dimensions = block_type::dim;
  if (Is_split_block<block_type>::value)
    d.complex_type = SPLIT;
  else
    d.complex_type = INTERLEAVED;
  dda::Rt_ext_data<block_type> ext_data
    (b, vsip::impl::block_layout<block_type::dim>(b), dda::SYNC_IN);
  d.size0 = ext_data.size(0);
  d.stride0 = ext_data.stride(0);
  if (d.dimensions > 1)
  {
    d.size1 = ext_data.size(1);
    d.stride1 = ext_data.stride(1);
  }
  if (d.dimensions > 2)
  {
    d.size2 = ext_data.size(2);
    d.stride2 = ext_data.stride(2);
  }
  if (d.complex_type == SPLIT)
  {
    data[0] = static_cast<void *>(ext_data.data().as_split().first);
    data[1] = static_cast<void *>(ext_data.data().as_split().second);
  }
  else
  {
    data[0] = static_cast<void *>(ext_data.data().as_inter());
    data[1] = 0;
  }
}

/// Releases the Block 'b' into the data pointer(s) 'data', and puts a
/// description of the released data into Descriptor 'd'.
template <typename Block>
void recover_marshal(Block &b, void **data, Descriptor &d)
{
  using namespace vsip::impl;
  typedef Block block_type;
  typedef typename block_type::value_type value_type;
  if (is_same<value_type, char>::value) d.value_type = CHAR;
  else if (is_same<value_type, int>::value) d.value_type = INT;
  else if (is_same<value_type, float>::value) d.value_type = FLOAT;
  else if (is_same<value_type, double>::value) d.value_type = DOUBLE;
  else if (is_same<value_type, complex<float> >::value) d.value_type = CFLOAT;
  else if (is_same<value_type, complex<double> >::value) d.value_type = CDOUBLE;
  else VSIP_IMPL_THROW(unimplemented("unsupported value-type"));
  d.dimensions = block_type::dim;

  Applied_layout<typename Block_layout<block_type>::layout_type> layout
    (extent<block_type::dim>(b));
  d.size0 = layout.size(0);
  d.stride0 = layout.stride(0);
  if (d.dimensions > 1)
  {
    d.size1 = layout.size(1);
    d.stride1 = layout.stride(1);
  }
  if (d.dimensions > 2)
  {
    d.size2 = layout.size(2);
    d.stride2 = layout.stride(2);
  }

  typedef typename Scalar_of<value_type>::type scalar_type;
  if (b.user_storage() == split_format)
  {
    d.complex_type = SPLIT;
    b.release(true, (scalar_type*&)data[0], (scalar_type*&)data[1]);
  }
  else
  {
    d.complex_type = INTERLEAVED;
    b.release(true, (scalar_type*&)data[0]);
    data[1] = 0;
  }
}

/// A generic block marshal may be able to marshal
/// a block, but can't unmarshal it. 
/// Whether it can unmarshal the block depends on what
/// kind of DDA is available for it.
template <typename Block> 
class Marshal : Rebinder<Block>
{
  typedef Block block_type;
  typedef typename block_type::value_type value_type;
  typedef void *data_pointers[2];

public:
  static void marshal(block_type const &b, data_pointers &data, Descriptor &d)
  { dda_marshal(b, data, d);}
  static bool can_unmarshal(Descriptor const &) { return false;}
  static void unmarshal(data_pointers const &data, Descriptor const &, Block &)
  { VSIP_IMPL_THROW(std::invalid_argument("Incompatible Block type"));}
  static void recover(block_type &b, data_pointers &data, Descriptor &d)
  { recover_marshal(b, data, d);}
};

/// Specialization for Strided.
/// In this case, unmarshaling is possible, depending on
/// the data layout.
template <vsip::dimension_type D, typename T, typename O, typename F>
class Marshal<Strided<D, T,
  dda::Layout<D, O, dda::Stride_unit_dense, F>, vsip::Local_map> >
: Rebinder<Strided<D, T,
  dda::Layout<D, O, dda::Stride_unit_dense, F>, vsip::Local_map> >
{
  typedef Strided<D, T,
    dda::Layout<D, O, dda::Stride_unit_dense, F>, vsip::Local_map> 
    block_type;
  typedef void *data_pointers[2];

public:
  static void marshal(block_type const &b, data_pointers &data, Descriptor &d)
  { dda_marshal(b, data, d);};

  static bool can_unmarshal(Descriptor const &d)
  {
    using namespace vsip::impl;
    if (!((is_same<T, char>::value && d.value_type == CHAR) ||
	  (is_same<T, int>::value && d.value_type == INT) ||
	  (is_same<T, float>::value && d.value_type == FLOAT) ||
	  (is_same<T, double>::value && d.value_type == DOUBLE) ||
	  (is_same<T, complex<float> >::value && d.value_type == CFLOAT) ||
	  (is_same<T, complex<double> >::value && d.value_type == CDOUBLE)))
      return false; // type mismatch
    else if (D != d.dimensions)
      return false; // dimension mismatch

    // The minor dimension needs to be unit-stride.
    if (D == 1 && d.stride0 != 1) return false;
    else if (D == 2 && d.stride1 != 1) return false;
    else if (D == 3 && d.stride2 != 1) return false;
    // Make sure strides match sizes.
    if (D == 2 && d.stride0 != d.size1) return false;
    else if (D == 3 && d.stride1 != d.size2 * d.size1) return false;
    return true;
  }
  static void unmarshal(data_pointers const &data, Descriptor const &d, block_type &b)
  {
    using namespace vsip;
    if (!can_unmarshal(d))
      VSIP_IMPL_THROW(std::invalid_argument("Incompatible Block type"));
    Domain<block_type::dim> domain;
    switch (d.dimensions)
    {
      case 3:
	domain.impl_at(2) = Domain<1>(0, d.stride2, d.size2);
      case 2:
	domain.impl_at(1) = Domain<1>(0, d.stride1, d.size1);
      case 1:
	domain.impl_at(0) = Domain<1>(0, d.stride0, d.size0);
	break;
      default:
	VSIP_IMPL_THROW(std::invalid_argument("Invalid dimension"));
    }
    rebind(b, data[0], data[1], d.complex_type, domain);
  }

  static void recover(block_type &b, data_pointers &data, Descriptor &d)
  { recover_marshal(b, data, d);}
};

/// A Dense Marshal is just a Strided Marshal...
template <vsip::dimension_type D, typename T, typename O>
class Marshal<vsip::Dense<D, T, O, vsip::Local_map> >
  : public Marshal<Strided<D, T, 
      dda::Layout<D, O, dda::Stride_unit_dense, vsip::impl::dense_complex_type>,
	Local_map> >
{};

} // namespace vsip_csl::impl::block_marshal

struct Block_marshal
{
  /// Changes this to point at Ext_data from Block b.
  template <typename Block>
  void marshal(Block const &b)
  { block_marshal::Marshal<Block>::marshal(b, data, descriptor);}

  template <typename Block>
  bool can_unmarshal()
  { return block_marshal::Marshal<Block>::can_unmarshal(descriptor);}

  /// Changes Block b to refer to data referenced by this.
  template <typename Block>
  void unmarshal(Block &b)
  {
    block_marshal::Marshal<Block>::unmarshal(data, descriptor, b);
  }

  /// Releases Block b, and changes this to point at released data.
  template <typename Block>
  void recover(Block &b)
  {
    block_marshal::Marshal<Block>::recover(b, data, descriptor);
  }

  block_marshal::Descriptor descriptor;
  void *data[2];
};

} // namespace vsip_csl::impl
} // namespace vsip_csl

#endif
