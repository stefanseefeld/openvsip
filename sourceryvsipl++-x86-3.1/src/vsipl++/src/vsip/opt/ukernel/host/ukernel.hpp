/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/ukernel/host/ukernel.hpp
    @author  Jules Bergmann
    @date    2008-06-10
    @brief   VSIPL++ Library: User-defined Kernel.
*/

#ifndef VSIP_OPT_UKERNEL_HOST_UKERNEL_HPP
#define VSIP_OPT_UKERNEL_HOST_UKERNEL_HPP

#define DEBUG             0
#define DEBUG_SPATT_EXTRA 0

#if DEBUG_SPATT_EXTRA
#  include <iostream>
#endif
#include <vsip/opt/cbe/dma.h>
#include <vsip/opt/cbe/ppu/util.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/ukernel/ukernel_params.hpp>
#include <vsip/opt/cbe/overlay_params.h>

namespace vsip_csl
{
namespace ukernel
{
/// Provide a map from operation / signature to plugin.
/// Specializations need to implement static member functions::
///
///   static char const *plugin();
///
/// providing the name of the library plugin, as well as the image
/// name within that.
template <typename O, typename S> struct Task_map;

}
}

// For backward compatibility only:
// O : operation
// S : signature
// D : directory
// P : plugin file (without extension)
# define DEFINE_UKERNEL_TASK(O, S, D, P)		   \
namespace vsip_csl { namespace ukernel {		   \
template <>                                                \
struct Task_map<O, S>					   \
{                                                          \
  static char const *plugin() { return D "/" #P ".img";}   \
};                                                         \
}}


namespace vsip
{
namespace impl
{
namespace ukernel
{

/// Base class for host kernel classes.
/// Deprecated.
struct Host_kernel
{
  typedef Empty_params param_type;
  static unsigned int const pre_argc = 0;
  static unsigned int const extra_ok = 0;

  template <typename T>
  void fill_params(T&) const {}

  void* get_param_stream() const { return 0; }

  length_type stack_size() const { return 4096; }

  length_type num_accel(length_type avail) const { return avail; }
};

class Whole_sdist
{
public:
  Whole_sdist()
    {}
};

class Blocksize_sdist
{
public:
  Blocksize_sdist(length_type max, length_type min = 0,
		  length_type mult = 32)
    : max_chunk_size_  (max)
    , min_chunk_size_  (min == 0 ? max : min)
    , chunk_multiple_  (mult)
    { assert(max_chunk_size_ >= min_chunk_size_); }

public:
  length_type max_chunk_size_;
  length_type min_chunk_size_;
  length_type chunk_multiple_;
};



class Blockoverlap_sdist
{
public:
  Blockoverlap_sdist(length_type size,
		     length_type overlap,
		     length_type mult = 0)
    : chunk_size_         (size)
    , chunk_multiple_     (mult == 0 ? size : mult)
    , leading_overlap_    (overlap)
    , trailing_overlap_   (overlap)
    , skip_first_overlap_ (1)
    , skip_last_overlap_  (1)
  {}

  Blockoverlap_sdist(length_type size,
		     length_type chunk_multiple,
		     length_type leading_overlap,
		     length_type trailing_overlap,
		     length_type skip_first_overlap,
		     length_type skip_last_overlap)
    : chunk_size_         (size)
    , chunk_multiple_     (chunk_multiple)
    , leading_overlap_    (leading_overlap)
    , trailing_overlap_   (trailing_overlap)
    , skip_first_overlap_ (skip_first_overlap)
    , skip_last_overlap_  (skip_last_overlap)
  {}

public:
  length_type chunk_size_;
  length_type chunk_multiple_;
  length_type leading_overlap_;
  length_type trailing_overlap_;
  int         skip_first_overlap_;
  int         skip_last_overlap_;
};



class Blockcount_sdist
{
public:
  Blockcount_sdist(length_type chunks)
    : num_chunks_      (chunks)
  {}

public:
  length_type num_chunks_;
};



class Full_sdist
{
public:
  Full_sdist(Whole_sdist const&)
    : num_chunks_      (1)
    , max_chunk_size_  (0)
    , min_chunk_size_  (0)
    , chunk_multiple_  (0)
    , leading_overlap_ (0)
    , trailing_overlap_(0)
    , skip_first_overlap_ (0)
    , skip_last_overlap_  (0)
  {}

  Full_sdist(Blocksize_sdist const& sdist)
    : num_chunks_      (0)
    , max_chunk_size_  (sdist.max_chunk_size_)
    , min_chunk_size_  (sdist.min_chunk_size_)
    , chunk_multiple_  (sdist.chunk_multiple_)
    , leading_overlap_ (0)
    , trailing_overlap_(0)
    , skip_first_overlap_ (0)
    , skip_last_overlap_  (0)
  {}

  Full_sdist(Blockoverlap_sdist const& sdist)
    : num_chunks_      (0)
    , max_chunk_size_  (sdist.chunk_size_)
    , min_chunk_size_  (sdist.chunk_size_)
    , chunk_multiple_  (sdist.chunk_multiple_)
    , leading_overlap_ (sdist.leading_overlap_)
    , trailing_overlap_(sdist.trailing_overlap_)
    , skip_first_overlap_ (sdist.skip_first_overlap_)
    , skip_last_overlap_  (sdist.skip_last_overlap_)
  {}

  Full_sdist(Blockcount_sdist const& sdist)
    : num_chunks_      (sdist.num_chunks_)
    , max_chunk_size_  (0)
    , min_chunk_size_  (0)
    , chunk_multiple_  (0)
    , leading_overlap_ (0)
    , trailing_overlap_(0)
    , skip_first_overlap_ (0)
    , skip_last_overlap_  (0)
  {}

public:
  length_type num_chunks_;
  length_type max_chunk_size_;
  length_type min_chunk_size_;
  length_type chunk_multiple_;
  length_type leading_overlap_;
  length_type trailing_overlap_;
  int         skip_first_overlap_;
  int         skip_last_overlap_;
};

class Stream_pattern
{
public:
  template <typename SDist0>
  Stream_pattern(SDist0 const& sdist0)
    : sdist0_(sdist0),
      sdist1_(Whole_sdist()),
      sdist2_(Whole_sdist())
  {}

  template <typename SDist0,
	    typename SDist1>
  Stream_pattern(SDist0 const& sdist0,
		 SDist1 const& sdist1)
    : sdist0_(sdist0),
      sdist1_(sdist1),
      sdist2_(Whole_sdist())
  {}

public:
  Full_sdist sdist0_;
  Full_sdist sdist1_;
  Full_sdist sdist2_;
};



template <typename T>
struct Set_addr
{
  static void set(Uk_stream& stream, T ptr)
  { stream.addr = cbe::ea_from_ptr(ptr); }
};

template <typename T>
struct Set_addr<std::pair<T*, T*> >
{
  static void set(Uk_stream& stream, std::pair<T*, T*> const& ptr)
  {
    stream.addr       = cbe::ea_from_ptr(ptr.first);
    stream.addr_split = cbe::ea_from_ptr(ptr.second);
  }
};


/// This class has methods used by Stream_spe::apply() to set up streaming for
/// views of different dimensions.
///
///   ::fill()     Takes information from a Stream_pattern as well as a count
///                of the number of available SPEs and fills in a Uk_stream
///                struct with information about how the data is to be sub-
///                divided for each iteration of data ('chunk') sent to an SPE.
///
template <typename ViewT,
	  dimension_type Dim = ViewT::dim>
struct Ps_helper;

template <typename ViewT>
struct Ps_helper<ViewT, 2>
{
  typedef typename ViewT::block_type block_type;
  typedef typename ViewT::value_type value_type;
  static storage_format_type const storage_format = get_block_layout<block_type>::storage_format;
  typedef Storage<storage_format, value_type> storage_type;
  typedef typename storage_type::type       ptr_type;
  typedef typename storage_type::alloc_type alloc_type;

  Ps_helper(ViewT& view, Stream_pattern const& spatt, bool input)
    : data_   (view.block())
    , addr_  (data_.ptr())
    , spatt_ (spatt)
    , input_ (input)
  {}

  void fill(Uk_stream& stream, length_type spes)
  {
    (void)spes;

    num_chunks_ = 1;
    chunk_size0_ = data_.size(0);
    chunk_size1_ = data_.size(1);
    chunk_size0_last_ = 0;
    chunk_size1_last_ = 0;

    if (spatt_.sdist0_.num_chunks_ > 0)
    {
      num_chunks0_ = spatt_.sdist0_.num_chunks_;
      chunk_size0_ = data_.size(0) / num_chunks0_;
      chunk_size0_xtra_ = data_.size(0) - (chunk_size0_ * num_chunks0_);
    }
    else if (spatt_.sdist0_.num_chunks_ == 0)
    {
      chunk_size0_ = spatt_.sdist0_.max_chunk_size_;
      num_chunks0_ = data_.size(0) / chunk_size0_;
      chunk_size0_last_ = data_.size(0) - (chunk_size0_ * num_chunks0_);
      if (chunk_size0_last_ % spatt_.sdist0_.chunk_multiple_)
      {
	chunk_size0_xtra_ =
	  chunk_size0_last_ % spatt_.sdist0_.chunk_multiple_;
	chunk_size0_last_  -= chunk_size0_xtra_;
      }
      else chunk_size0_xtra_ = 0;
      if (chunk_size0_last_) num_chunks0_++;
    }
    else assert(0);

    if (spatt_.sdist1_.num_chunks_ > 0)
    {
      num_chunks1_ = spatt_.sdist1_.num_chunks_;
      chunk_size1_ = data_.size(1) / num_chunks1_;
      chunk_size1_xtra_ = data_.size(1) - (chunk_size1_ * num_chunks1_);
    }
    else if (spatt_.sdist1_.num_chunks_ == 0)
    {
      chunk_size1_ = spatt_.sdist1_.max_chunk_size_;
      num_chunks1_ = data_.size(1) / chunk_size1_;
      chunk_size1_last_ = data_.size(1) - (chunk_size1_ * num_chunks1_);
      if (chunk_size1_last_ % spatt_.sdist1_.chunk_multiple_)
      {
	chunk_size1_xtra_ =
	  chunk_size1_last_ % spatt_.sdist1_.chunk_multiple_;
	chunk_size1_last_  -= chunk_size1_xtra_;
      }
      else chunk_size1_xtra_ = 0;
      if (chunk_size1_last_) num_chunks1_++;
    }
    else assert(0);

    num_chunks_              = num_chunks0_ * num_chunks1_;
    chunks_per_spe_          = num_chunks_ / spes;
    chunk_index_             = 0;

    stream.dim               = 2;
    stream.align_shift       = 0;
    stream.num_chunks0       = num_chunks0_;
    stream.num_chunks1       = num_chunks1_;
    stream.chunk_size0       = chunk_size0_;
    stream.chunk_size1       = chunk_size1_;
    stream.chunk_size0_extra = chunk_size0_last_;
    stream.chunk_size1_extra = chunk_size1_last_;
    stream.stride0           = data_.stride(0);
    stream.stride1           = data_.stride(1);
    stream.leading_overlap0  = spatt_.sdist0_.leading_overlap_;
    stream.leading_overlap1  =
      VSIP_IMPL_INCREASE_TO_DMA_SIZE(spatt_.sdist1_.leading_overlap_, float);
    stream.trailing_overlap0 = spatt_.sdist0_.trailing_overlap_;
    stream.trailing_overlap1 =
      VSIP_IMPL_INCREASE_TO_DMA_SIZE(spatt_.sdist1_.trailing_overlap_, float);
    stream.skip_first_overlap0 = spatt_.sdist0_.skip_first_overlap_;
    stream.skip_first_overlap1 = spatt_.sdist1_.skip_first_overlap_;
    stream.skip_last_overlap0  = spatt_.sdist0_.skip_last_overlap_;
    stream.skip_last_overlap1  = spatt_.sdist1_.skip_last_overlap_;

    chunk_size0_ += stream.leading_overlap0 +
                    stream.trailing_overlap0;
    chunk_size1_ += stream.leading_overlap1 +
                    stream.trailing_overlap1;

    chunk_size_ = chunk_size0_ * chunk_size1_;

    num_lines_ = std::max(stream.chunk_size0, stream.chunk_size0_extra);
    num_lines_ += stream.leading_overlap0 + stream.trailing_overlap0;

    Set_addr<ptr_type>::set(stream, addr_);
  }

  void set_workblock(Uk_stream& stream, length_type chunks_this_spe)
  {
    stream.chunk_offset = chunk_index_;
    chunk_index_ += chunks_this_spe;
  }

  int extra_size()
  { return chunk_size0_xtra_ * chunk_size1_xtra_; }

  int extra_size(dimension_type dim)
  { return (dim == 0) ? chunk_size0_xtra_ : chunk_size1_xtra_; }

  length_type buffer_size()
  { return sizeof(value_type)*chunk_size_; }

  length_type dtl_size()       { return num_lines_; }
  length_type num_chunks()     { return num_chunks_; }
  length_type chunks_per_spe() { return chunks_per_spe_; }

  void dump(char const* name)
  {
#if DEBUG_SPATT_EXTRA
    std::cout << "Ps_helper<2, View>: " << name << "\n"
	      << "  num_chunks_      : " << num_chunks_ << "\n"
	      << "  num_chunks0_     : " << num_chunks0_ << "\n"
	      << "  num_chunks1_     : " << num_chunks1_ << "\n"
	      << "  chunks_per_spe_  : " << chunks_per_spe_ << "\n"
	      << "  chunk_size_      : " << chunk_size_ << "\n"
	      << "  chunk_size0_     : " << chunk_size0_ << "\n"
	      << "  chunk_size1_     : " << chunk_size1_ << "\n"
	      << "  chunk_size0_last_: " << chunk_size0_last_ << "\n"
	      << "  chunk_size1_last_: " << chunk_size1_last_ << "\n"
	      << "  chunk_size0_xtra_: " << chunk_size0_xtra_ << "\n"
	      << "  chunk_size1_xtra_: " << chunk_size1_xtra_ << "\n"
	      << "  chunk_index_     : " << chunk_index_ << "\n"
      ;
#else
    (void)name;
#endif
  }

private:
  dda::Data<block_type, dda::inout>  data_;
  ptr_type              addr_;
  Stream_pattern const& spatt_;
  bool                  input_;
  length_type           num_chunks_;
  length_type           num_chunks0_;
  length_type           num_chunks1_;
  length_type           chunks_per_spe_;
  length_type           chunk_size_;
  length_type           chunk_size0_;
  length_type           chunk_size1_;
  length_type           chunk_size0_last_;
  length_type           chunk_size1_last_;
  length_type           chunk_size0_xtra_;
  length_type           chunk_size1_xtra_;
  index_type            chunk_index_;
  length_type		num_lines_;
};


template <typename T>
int
find_align_shift(T* addr)
{
  return ((uintptr_t)(addr) % VSIP_IMPL_DMA_SIZE_QUANTUM) / sizeof(T);
}

template <typename T>
int
find_align_shift(std::pair<T*, T*> const& addr)
{
  assert(((uintptr_t)(addr.first)  % VSIP_IMPL_DMA_SIZE_QUANTUM) / sizeof(T) ==
	 ((uintptr_t)(addr.second) % VSIP_IMPL_DMA_SIZE_QUANTUM) / sizeof(T));
  return ((uintptr_t)(addr.first) % VSIP_IMPL_DMA_SIZE_QUANTUM) / sizeof(T);
}


template <typename ViewT>
struct Ps_helper<ViewT, 1>
{
  typedef typename ViewT::block_type block_type;
  typedef typename ViewT::value_type value_type;
  static storage_format_type const storage_format = get_block_layout<block_type>::storage_format;
  typedef Storage<storage_format, value_type> storage_type;
  typedef typename storage_type::type       ptr_type;
  typedef typename storage_type::alloc_type alloc_type;

  Ps_helper(ViewT& view, Stream_pattern const& spatt, bool input)
    : data_  (view.block())
    , addr_ (data_.ptr())
    , spatt_(spatt)
    , input_(input)
  {
    using cbe::is_dma_addr_ok;
    assert(data_.stride(0) == 1);
    assert(input || is_dma_addr_ok(addr_));
  }

  void fill(Uk_stream& stream, length_type spes)
  {
    stream.align_shift = find_align_shift(addr_);
    total_size_  = data_.size(0);
    if (spatt_.sdist0_.num_chunks_ != 0)
    {
      num_chunks_      = spatt_.sdist0_.num_chunks_;
      chunks_per_spe_  = num_chunks_ / spes;
      chunk_size_      = total_size_ / num_chunks_;
      chunk_size_last_ = 0;
      chunk_size_xtra_ = total_size_ - (chunk_size_ * num_chunks_);
    }
    else
    {
      chunk_size_  = total_size_ / spes;

      if (chunk_size_ > spatt_.sdist0_.max_chunk_size_)
	chunk_size_ = spatt_.sdist0_.max_chunk_size_;
      else if (chunk_size_ < spatt_.sdist0_.min_chunk_size_)
	chunk_size_ = spatt_.sdist0_.min_chunk_size_;
      else if (chunk_size_ % spatt_.sdist0_.chunk_multiple_)
	chunk_size_ -= chunk_size_ % spatt_.sdist0_.chunk_multiple_;

      num_chunks_       = total_size_ / chunk_size_;
      chunk_size_last_ = total_size_ - (chunk_size_ * num_chunks_);

      if (chunk_size_last_ % spatt_.sdist0_.chunk_multiple_)
      {
	chunk_size_xtra_ =
	  chunk_size_last_ % spatt_.sdist0_.chunk_multiple_;
	chunk_size_last_  -= chunk_size_xtra_;
      }
      else chunk_size_xtra_ = 0;

      if (chunk_size_last_) num_chunks_++;
      chunks_per_spe_   = num_chunks_ / spes;
    }

    chunk_index_       = 0;

    Set_addr<ptr_type>::set(stream, addr_);
    stream.dim               = 1;
    stream.num_chunks0       = 1;
    stream.num_chunks1       = num_chunks_;
    stream.chunk_size0       = 1;
    stream.chunk_size1       = chunk_size_;
    stream.chunk_size0_extra = 0;
    stream.chunk_size1_extra = chunk_size_last_;
    stream.stride0           = 0;
    stream.stride1           = 1;
    stream.leading_overlap0  = 0;
    stream.leading_overlap1  =
      VSIP_IMPL_INCREASE_TO_DMA_SIZE(spatt_.sdist0_.leading_overlap_, float);
    stream.trailing_overlap0 = 0;
    stream.trailing_overlap1 =
      VSIP_IMPL_INCREASE_TO_DMA_SIZE(spatt_.sdist0_.trailing_overlap_, float);
    stream.skip_first_overlap0 = 0;
    stream.skip_first_overlap1 = spatt_.sdist0_.skip_first_overlap_;
    stream.skip_last_overlap0  = 0;
    stream.skip_last_overlap1  = spatt_.sdist0_.skip_last_overlap_;

    chunk_size_ += stream.leading_overlap1 + stream.trailing_overlap1
      +  VSIP_IMPL_INCREASE_TO_DMA_SIZE(stream.align_shift, float);
  }



  void set_workblock(Uk_stream& stream, length_type chunks_this_spe)
  {
    stream.chunk_offset = chunk_index_;
    chunk_index_ += chunks_this_spe;
    total_size_ -= chunks_this_spe * chunk_size_;
  }

  int extra_size()               { return chunk_size_xtra_; }
  int extra_size(dimension_type) { return chunk_size_xtra_; }

  length_type buffer_size()
  { return sizeof(value_type)*chunk_size_; }

  length_type dtl_size()       { return 1; }
  length_type num_chunks()     { return num_chunks_; }
  length_type chunks_per_spe() { return chunks_per_spe_; }

  void dump(char const* name)
  {
#if DEBUG_SPATT_EXTRA
    std::cout << "Ps_helper<1, View>: " << name << "\n"
	      << "  total_size_      : " << total_size_ << "\n"
	      << "  chunk_size_      : " << chunk_size_ << "\n"
	      << "  chunk_size_last_: " << chunk_size_last_ << "\n"
	      << "  chunk_size_xtra_: " << chunk_size_xtra_ << "\n"
	      << "  chunk_index_     : " << chunk_index_ << "\n"
	      << "  chunks_per_spe_  : " << chunks_per_spe_ << "\n"
	      << "  num_chunks_      : " << num_chunks_ << "\n"
      ;
#else
    (void)name;
#endif
  }

private:
  dda::Data<block_type, dda::inout>  data_;
  ptr_type              addr_;
  Stream_pattern const& spatt_;
  bool                  input_;
  length_type           total_size_;
  length_type           chunk_size_;
  length_type           chunk_size_last_;
  length_type           chunk_size_xtra_;
  length_type           chunk_index_;
  length_type           chunks_per_spe_;
  length_type           num_chunks_;
};


template <typename ViewT>
struct Ps_helper<ViewT, 3>
{
  typedef typename ViewT::block_type block_type;
  typedef typename ViewT::value_type value_type;
  static storage_format_type const storage_format = get_block_layout<block_type>::storage_format;
  typedef Storage<storage_format, value_type> storage_type;
  typedef typename storage_type::type       ptr_type;
  typedef typename storage_type::alloc_type alloc_type;

  Ps_helper(ViewT& view, Stream_pattern const& spatt, bool input)
    : data_   (view.block())
    , addr_  (data_.ptr())
    , spatt_ (spatt)
    , input_ (input)
  {}

  void fill(Uk_stream& stream, length_type spes)
  {
    num_chunks_ = 1;
    chunk_size0_ = data_.size(0);
    chunk_size1_ = data_.size(1);
    chunk_size2_ = data_.size(2);
    chunk_size0_last_ = 0;
    chunk_size1_last_ = 0;
    chunk_size2_last_ = 0;

    if (spatt_.sdist0_.num_chunks_ > 0)
    {
      num_chunks0_ = spatt_.sdist0_.num_chunks_;
      chunk_size0_ = data_.size(0) / num_chunks0_;
      chunk_size0_xtra_ = data_.size(0) - (chunk_size0_ * num_chunks0_);
    }
    else if (spatt_.sdist0_.num_chunks_ == 0)
    {
      chunk_size0_ = spatt_.sdist0_.max_chunk_size_;
      if (chunk_size0_ > data_.size(0)) 
        chunk_size0_ = data_.size(0);  // make chunk smaller if we're under the max
      num_chunks0_ = data_.size(0) / chunk_size0_;
      chunk_size0_last_ = data_.size(0) - (chunk_size0_ * num_chunks0_);
      if (chunk_size0_last_ % spatt_.sdist0_.chunk_multiple_)
      {
	chunk_size0_xtra_ =
	  chunk_size0_last_ % spatt_.sdist0_.chunk_multiple_;
	chunk_size0_last_  -= chunk_size0_xtra_;
      }
      else chunk_size0_xtra_ = 0;
      if (chunk_size0_last_) num_chunks0_++;
    }
    else assert(0);

    if (spatt_.sdist1_.num_chunks_ > 0)
    {
      num_chunks1_ = spatt_.sdist1_.num_chunks_;
      chunk_size1_ = data_.size(1) / num_chunks1_;
      chunk_size1_xtra_ = data_.size(1) - (chunk_size1_ * num_chunks1_);
    }
    else if (spatt_.sdist1_.num_chunks_ == 0)
    {
      chunk_size1_ = spatt_.sdist1_.max_chunk_size_;
      if (chunk_size1_ > data_.size(1)) 
        chunk_size1_ = data_.size(1);  // make chunk smaller if we're under the max
      num_chunks1_ = data_.size(1) / chunk_size1_;
      chunk_size1_last_ = data_.size(1) - (chunk_size1_ * num_chunks1_);
      if (chunk_size1_last_ % spatt_.sdist1_.chunk_multiple_)
      {
	chunk_size1_xtra_ =
	  chunk_size1_last_ % spatt_.sdist1_.chunk_multiple_;
	chunk_size1_last_  -= chunk_size1_xtra_;
      }
      else chunk_size1_xtra_ = 0;
      if (chunk_size1_last_) num_chunks1_++;
    }
    else assert(0);

    if (spatt_.sdist2_.num_chunks_ > 0)
    {
      num_chunks2_ = spatt_.sdist2_.num_chunks_;
      chunk_size2_ = data_.size(2) / num_chunks2_;
      chunk_size2_xtra_ = data_.size(2) - (chunk_size2_ * num_chunks2_);
    }
    else if (spatt_.sdist2_.num_chunks_ == 0)
    {
      chunk_size2_ = spatt_.sdist2_.max_chunk_size_;
      if (chunk_size2_ > data_.size(2)) 
        chunk_size2_ = data_.size(2);  // make chunk smaller if we're under the max
      num_chunks2_ = data_.size(2) / chunk_size2_;
      chunk_size2_last_ = data_.size(2) - (chunk_size2_ * num_chunks2_);
      if (chunk_size2_last_ % spatt_.sdist2_.chunk_multiple_)
      {
	chunk_size2_xtra_ =
	  chunk_size2_last_ % spatt_.sdist2_.chunk_multiple_;
	chunk_size2_last_  -= chunk_size2_xtra_;
      }
      else chunk_size2_xtra_ = 0;
      if (chunk_size2_last_) num_chunks2_++;
    }
    else assert(0);

    num_chunks_              = num_chunks0_ * num_chunks1_  * num_chunks2_;
    chunks_per_spe_          = num_chunks_ / spes;
    chunk_index_             = 0;

    stream.dim               = 3;
    stream.align_shift       = 0;
    stream.num_chunks0       = num_chunks0_;
    stream.num_chunks1       = num_chunks1_;
    stream.num_chunks2       = num_chunks2_;
    stream.chunk_size0       = chunk_size0_;
    stream.chunk_size1       = chunk_size1_;
    stream.chunk_size2       = chunk_size2_;
    stream.chunk_size0_extra = chunk_size0_last_;
    stream.chunk_size1_extra = chunk_size1_last_;
    stream.chunk_size2_extra = chunk_size2_last_;
    stream.stride0           = data_.stride(0);
    stream.stride1           = data_.stride(1);
    stream.stride2           = data_.stride(2);
    stream.leading_overlap0  = spatt_.sdist0_.leading_overlap_;
    stream.leading_overlap1  = spatt_.sdist1_.leading_overlap_;
    stream.leading_overlap2  =
      VSIP_IMPL_INCREASE_TO_DMA_SIZE(spatt_.sdist2_.leading_overlap_, float);
    stream.trailing_overlap0 = spatt_.sdist0_.trailing_overlap_;
    stream.trailing_overlap1 = spatt_.sdist1_.trailing_overlap_;
    stream.trailing_overlap2 =
      VSIP_IMPL_INCREASE_TO_DMA_SIZE(spatt_.sdist2_.trailing_overlap_, float);
    stream.skip_first_overlap0 = spatt_.sdist0_.skip_first_overlap_;
    stream.skip_first_overlap1 = spatt_.sdist1_.skip_first_overlap_;
    stream.skip_first_overlap2 = spatt_.sdist2_.skip_first_overlap_;
    stream.skip_last_overlap0  = spatt_.sdist0_.skip_last_overlap_;
    stream.skip_last_overlap1  = spatt_.sdist1_.skip_last_overlap_;
    stream.skip_last_overlap2  = spatt_.sdist2_.skip_last_overlap_;

    chunk_size0_ += stream.leading_overlap0 +
                    stream.trailing_overlap0;
    chunk_size1_ += stream.leading_overlap1 +
                    stream.trailing_overlap1;
    chunk_size2_ += stream.leading_overlap2 +
                    stream.trailing_overlap2;

    chunk_size_ = chunk_size0_ * chunk_size1_ * chunk_size2_;

    Set_addr<ptr_type>::set(stream, addr_);
  }

  void set_workblock(Uk_stream& stream, length_type chunks_this_spe)
  {
    stream.chunk_offset = chunk_index_;
    chunk_index_ += chunks_this_spe;
  }

  int extra_size()
  { return chunk_size0_xtra_ * chunk_size1_xtra_ * chunk_size2_xtra_; }

  int extra_size(dimension_type dim)
  { return (dim == 0) ? chunk_size0_xtra_ :
           (dim == 1) ? chunk_size1_xtra_ : chunk_size2_xtra_; }

  length_type buffer_size()
  { return sizeof(value_type)*chunk_size_; }

  length_type dtl_size()       { return chunk_size0_ * chunk_size1_; }
  length_type num_chunks()     { return num_chunks_; }
  length_type chunks_per_spe() { return chunks_per_spe_; }

  void dump(char const* name)
  {
#if DEBUG_SPATT_EXTRA
    std::cout << "Ps_helper<3, View>: " << name << "\n"
	      << "  num_chunks_      : " << num_chunks_ << "\n"
	      << "  num_chunks0_     : " << num_chunks0_ << "\n"
	      << "  num_chunks1_     : " << num_chunks1_ << "\n"
	      << "  num_chunks2_     : " << num_chunks2_ << "\n"
	      << "  chunks_per_spe_  : " << chunks_per_spe_ << "\n"
	      << "  chunk_size_      : " << chunk_size_ << "\n"
	      << "  chunk_size0_     : " << chunk_size0_ << "\n"
	      << "  chunk_size1_     : " << chunk_size1_ << "\n"
	      << "  chunk_size2_     : " << chunk_size2_ << "\n"
	      << "  chunk_size0_last_: " << chunk_size0_last_ << "\n"
	      << "  chunk_size1_last_: " << chunk_size1_last_ << "\n"
	      << "  chunk_size2_last_: " << chunk_size2_last_ << "\n"
	      << "  chunk_size0_xtra_: " << chunk_size0_xtra_ << "\n"
	      << "  chunk_size1_xtra_: " << chunk_size1_xtra_ << "\n"
	      << "  chunk_size2_xtra_: " << chunk_size2_xtra_ << "\n"
	      << "  chunk_index_     : " << chunk_index_ << "\n"
      ;
#endif
  }

private:
  dda::Data<block_type, dda::inout>  data_;
  ptr_type              addr_;
  Stream_pattern const& spatt_;
  bool                  input_;
  length_type           num_chunks_;
  length_type           num_chunks0_;
  length_type           num_chunks1_;
  length_type           num_chunks2_;
  length_type           chunks_per_spe_;
  length_type           chunk_size_;
  length_type           chunk_size0_;
  length_type           chunk_size1_;
  length_type           chunk_size2_;
  length_type           chunk_size0_last_;
  length_type           chunk_size1_last_;
  length_type           chunk_size2_last_;
  length_type           chunk_size0_xtra_;
  length_type           chunk_size1_xtra_;
  length_type           chunk_size2_xtra_;
  index_type            chunk_index_;
};



template <typename FuncT>
struct Stream_spe
{
  static unsigned int const pre_argc = FuncT::pre_argc;
  static unsigned int const in_argc  = FuncT::in_argc;
  static unsigned int const out_argc = FuncT::out_argc;
  typedef typename FuncT::param_type kernel_param_type;

  typedef Ukernel_params<pre_argc, in_argc, out_argc, kernel_param_type>
		param_type;

  static void apply(FuncT const &func)
  {
    using cbe::Task_manager;
    using cbe::lwp::Workblock;
    using cbe::lwp::Task;
    using cbe::lwp::load_plugin;
    using cbe::is_dma_addr_ok;
    using cbe::is_dma_size_ok;

    assert(pre_argc + in_argc  == 0);
    assert(out_argc == 0);

    cbe::Task_manager* mgr = cbe::Task_manager::instance();
    length_type spes       = func.num_accel(mgr->num_spes());
    length_type stack_size = func.stack_size();

    length_type isize    = 0;
    length_type osize    = 0;
    length_type dtl_size = 0;

    param_type ukp;

    func.fill_params(ukp.kernel_params);


    char const *plugin =
      vsip_csl::ukernel::Task_map<FuncT, void()>::plugin();

    static char* code_ea = 0;
    static int   code_size;
    if (code_ea == 0) load_plugin(code_ea, code_size, plugin);

    assert(stack_size <= VSIP_IMPL_OVERLAY_STACK_SIZE);
    assert(isize+osize <= VSIP_IMPL_OVERLAY_BUFFER_SIZE);
    assert(dtl_size <= VSIP_IMPL_OVERLAY_DTL_SIZE);

    std::auto_ptr<Task> task =
      mgr->reserve_lwp_task(0, 1, (uintptr_t)code_ea, code_size, 0);

    ukp.nspe       = spes;
    ukp.pre_chunks = 0;

    for (index_type i=0; i<spes; ++i)
    {
      Workblock block = task.get()->create_workblock(1);
      ukp.rank = i;
      block.set_parameters(ukp);
      block.enqueue();
    }

    task.get()->sync();
  }

  template <typename View0, typename View2>
  static void apply(FuncT const &func, View0 in0, View2 out)
  {
    using cbe::Task_manager;
    using cbe::lwp::Workblock;
    using cbe::lwp::Task;
    using cbe::lwp::load_plugin;
    using cbe::is_dma_addr_ok;
    using cbe::is_dma_size_ok;

    typedef Ps_helper<View0> vh0_t;
    typedef Ps_helper<View2> vh2_t;

    typedef typename vh0_t::ptr_type ptr0_type;
    typedef typename vh2_t::ptr_type ptr2_type;

    vh0_t vh0(in0, func.in_spatt(0),  1);
    vh2_t vh2(out, func.out_spatt(0), 0);

    assert(pre_argc + in_argc  == 1);
    assert(out_argc == 1);

    cbe::Task_manager* mgr = cbe::Task_manager::instance();

    length_type spes       = func.num_accel(mgr->num_spes());
    length_type stack_size = func.stack_size();

    param_type ukp;
    vh0.fill(ukp.in_stream[0],  spes);
    vh2.fill(ukp.out_stream[0], spes);

    assert(vh0.extra_size() == 0);
    assert(vh2.extra_size() == 0);

    func.fill_params(ukp.kernel_params);

    length_type chunks         = vh2.num_chunks();
    length_type chunks_per_spe = vh2.chunks_per_spe();
    assert(chunks_per_spe * spes <= chunks);

    length_type isize = vh0.buffer_size();
    length_type osize = vh2.buffer_size();
    length_type dtl_size = vh0.dtl_size() + vh2.dtl_size();

#if DEBUG
    printf("chunk num: %d (%d x %d)  size: %d (%d x %d)  block_size: %d %d  dtl_size: %d\n",
	   vh2.num_chunks(),
	   ukp.out_stream[0].num_chunks0, ukp.out_stream[0].num_chunks1,
	   0, /* vh2.chunk_size_, */
	   ukp.out_stream[0].chunk_size0, ukp.out_stream[0].chunk_size1,
	   isize, osize, dtl_size);
#endif

    char const *plugin =
      vsip_csl::ukernel::Task_map<FuncT, void(ptr0_type, ptr2_type)>::plugin();

    static char* code_ea = 0;
    static int   code_size;
    if (code_ea == 0) load_plugin(code_ea, code_size, plugin);

    assert(stack_size <= VSIP_IMPL_OVERLAY_STACK_SIZE);
    assert(isize+osize <= VSIP_IMPL_OVERLAY_BUFFER_SIZE);
    assert(dtl_size <= VSIP_IMPL_OVERLAY_DTL_SIZE);

    std::auto_ptr<Task> task =
      mgr->reserve_lwp_task(VSIP_IMPL_OVERLAY_BUFFER_SIZE, 2,
			    (uintptr_t)code_ea, code_size, 0);

    ukp.nspe       = std::min(spes, chunks);
    ukp.pre_chunks = 0;

    for (index_type i=0; i<spes && i<chunks; ++i)
    {
      // If chunks don't divide evenly, give the first SPEs one extra.
      length_type chunks_this_spe = (i < chunks % spes) ? chunks_per_spe + 1
	                                          : chunks_per_spe;

      vh0.set_workblock(ukp.in_stream[0],  chunks_this_spe);
      vh2.set_workblock(ukp.out_stream[0], chunks_this_spe);

      Workblock block = task.get()->create_workblock(chunks_this_spe +
						     ukp.pre_chunks);
      ukp.rank = i;
      block.set_parameters(ukp);
      block.enqueue();
    }

    task.get()->sync();
  }

  template <typename View0, typename View1, typename View2>
  static void apply(FuncT const &func, View0 in0, View1 in1, View2 out)
  {
    using cbe::Task_manager;
    using cbe::lwp::Workblock;
    using cbe::lwp::Task;
    using cbe::lwp::load_plugin;
    using cbe::is_dma_addr_ok;
    using cbe::is_dma_size_ok;

    typedef Ps_helper<View0> vh0_t;
    typedef Ps_helper<View1> vh1_t;
    typedef Ps_helper<View2> vh2_t;

    typedef typename vh0_t::ptr_type ptr0_type;
    typedef typename vh1_t::ptr_type ptr1_type;
    typedef typename vh2_t::ptr_type ptr2_type;

    vh0_t vh0(in0, func.in_spatt(0),  1);
    vh1_t vh1(in1, func.in_spatt(1),  1);
    vh2_t vh2(out, func.out_spatt(0), 0);

    assert(pre_argc + in_argc  == 2);
    assert(out_argc == 1);

    cbe::Task_manager* mgr = cbe::Task_manager::instance();
    length_type spes       = func.num_accel(mgr->num_spes());
    length_type stack_size = func.stack_size();

    param_type ukp;
    func.fill_params(ukp.kernel_params);

    vh0.fill(ukp.in_stream[0],  spes);
    vh1.fill(ukp.in_stream[1],  spes);
    vh2.fill(ukp.out_stream[0], spes);

    if (!FuncT::extra_ok &&
	(vh0.extra_size() != 0 || vh1.extra_size() != 0 ||
	 vh2.extra_size() != 0))
    {
      if (vh0.extra_size() != 0) vh0.dump("vh0");
      if (vh1.extra_size() != 0) vh1.dump("vh1");
      if (vh2.extra_size() != 0) vh2.dump("vh2");
      // TODO: THROW
      assert(0);
    }

    length_type isize;
    length_type osize;
    length_type dtl_size;
    if (FuncT::pre_argc > 0)
    {
      ukp.pre_chunks = vh0.num_chunks();
      isize = std::max(vh0.buffer_size(), vh1.buffer_size());
      dtl_size = std::max(vh0.dtl_size(), vh1.dtl_size() + vh2.dtl_size());
      osize = vh2.buffer_size();
    }
    else
    {
      ukp.pre_chunks = 0;
      isize = vh0.buffer_size() + vh1.buffer_size();
      osize = vh2.buffer_size();
      dtl_size = vh0.dtl_size() + vh1.dtl_size() + vh2.dtl_size();
    }

    length_type chunks         = vh2.num_chunks();
    length_type chunks_per_spe = vh2.chunks_per_spe();
    assert(chunks_per_spe * spes <= chunks);

#if DEBUG
    printf("chunk num: %d (%d x %d)  size: %d (%d x %d)  block_size: %d %d  dtl_size: %d\n",
	   vh2.num_chunks(),
	   ukp.out_stream[0].num_chunks0, ukp.out_stream[0].num_chunks1,
	   0, /* vh2.chunk_size_, */
	   ukp.out_stream[0].chunk_size0, ukp.out_stream[0].chunk_size1,
	   isize, osize, dtl_size);
#endif

    char const *plugin =
      vsip_csl::ukernel::Task_map<FuncT, void(ptr0_type, ptr1_type, ptr2_type)>::plugin();

    static char* code_ea = 0;
    static int   code_size;
    if (code_ea == 0) load_plugin(code_ea, code_size, plugin);

    assert(stack_size <= VSIP_IMPL_OVERLAY_STACK_SIZE);
    assert(isize+osize <= VSIP_IMPL_OVERLAY_BUFFER_SIZE);
    assert(dtl_size <= VSIP_IMPL_OVERLAY_DTL_SIZE);

    std::auto_ptr<Task> task =
      mgr->reserve_lwp_task(VSIP_IMPL_OVERLAY_BUFFER_SIZE, 2,
			    (uintptr_t)code_ea, code_size, 0);

    ukp.code_ea    = (uintptr_t)code_ea;
    ukp.code_size  = code_size;
    ukp.nspe       = std::min(spes, chunks);

    for (index_type i=0; i<spes && i<chunks; ++i)
    {
      // If chunks don't divide evenly, give the first SPEs one extra.
      length_type chunks_this_spe = (i < chunks % spes) ? chunks_per_spe + 1
	                                          : chunks_per_spe;

      vh0.set_workblock(ukp.in_stream[0],  chunks_this_spe);
      vh1.set_workblock(ukp.in_stream[1],  chunks_this_spe);
      vh2.set_workblock(ukp.out_stream[0], chunks_this_spe);

      Workblock block = task.get()->create_workblock(chunks_this_spe+ukp.pre_chunks);
      ukp.rank = i;
      block.set_parameters(ukp);
      block.enqueue();
    }

    task.get()->sync();
  }


  template <typename View0,
	    typename View1,
	    typename View2,
	    typename View3>
  static void apply(FuncT const &func,
		    View0 in0, View1 in1, View2 in2, View3 out)
  {
    using cbe::Task_manager;
    using cbe::lwp::Workblock;
    using cbe::lwp::Task;
    using cbe::lwp::load_plugin;
    using cbe::is_dma_addr_ok;
    using cbe::is_dma_size_ok;

    typedef Ps_helper<View0> vh0_t;
    typedef Ps_helper<View1> vh1_t;
    typedef Ps_helper<View2> vh2_t;
    typedef Ps_helper<View3> vh3_t;

    typedef typename vh0_t::ptr_type ptr0_type;
    typedef typename vh1_t::ptr_type ptr1_type;
    typedef typename vh2_t::ptr_type ptr2_type;
    typedef typename vh3_t::ptr_type ptr3_type;

    vh0_t vh0(in0, func.in_spatt(0),  1);
    vh1_t vh1(in1, func.in_spatt(1),  1);
    vh2_t vh2(in2, func.in_spatt(2),  1);
    vh3_t vh3(out, func.out_spatt(0), 0);

    assert(pre_argc + in_argc  == 3);
    assert(out_argc == 1);

    cbe::Task_manager* mgr = cbe::Task_manager::instance();
    length_type spes       = func.num_accel(mgr->num_spes());
    length_type stack_size = func.stack_size();

    param_type ukp;
    func.fill_params(ukp.kernel_params);

    vh0.fill(ukp.in_stream[0],  spes);
    vh1.fill(ukp.in_stream[1],  spes);
    vh2.fill(ukp.in_stream[2],  spes);
    vh3.fill(ukp.out_stream[0], spes);

    assert(vh0.extra_size() == 0);
    assert(vh1.extra_size() == 0);
    assert(vh2.extra_size() == 0);
    assert(vh3.extra_size() == 0);

#if DEBUG
    vh0.dump("vh0");
    vh1.dump("vh1");
    vh2.dump("vh2");
    vh3.dump("vh3");
#endif

    length_type isize;
    length_type osize;
    length_type buffer_size;
    length_type dtl_size;
    ukp.pre_chunks = 0;
    isize = vh0.buffer_size() + vh1.buffer_size() + vh2.buffer_size();
    osize = vh3.buffer_size();
    buffer_size = isize + osize;
    dtl_size = vh0.dtl_size() + vh1.dtl_size() + vh2.dtl_size() + vh3.dtl_size();

    length_type chunks         = vh3.num_chunks();
    length_type chunks_per_spe = vh3.chunks_per_spe();
    assert(chunks_per_spe * spes <= chunks);

#if DEBUG
    printf("chunk num: %d (%d x %d)  size: %d (%d x %d)  block_size: %d %d  dtl_size: %d\n",
	   vh3.num_chunks(),
	   ukp.out_stream[0].num_chunks0, ukp.out_stream[0].num_chunks1,
	   0, /* vh3.chunk_size_, */
	   ukp.out_stream[0].chunk_size0, ukp.out_stream[0].chunk_size1,
	   isize, osize, dtl_size);
#endif

    char const *plugin =
      vsip_csl::ukernel::Task_map<FuncT, void(ptr0_type, ptr1_type, ptr2_type, ptr3_type)>::plugin();

    static char* code_ea = 0;
    static int   code_size;
    if (code_ea == 0) load_plugin(code_ea, code_size, plugin);

    // Attempt to use standard sizes if possible
    if (stack_size <= VSIP_IMPL_OVERLAY_STACK_SIZE &&
	buffer_size <= VSIP_IMPL_OVERLAY_BUFFER_SIZE &&
	dtl_size <= VSIP_IMPL_OVERLAY_DTL_SIZE)
    {
      stack_size  = VSIP_IMPL_OVERLAY_STACK_SIZE;
      buffer_size = VSIP_IMPL_OVERLAY_BUFFER_SIZE;
      dtl_size    = VSIP_IMPL_OVERLAY_DTL_SIZE;
    }

    length_type buffer_space =   256 * 1024  // Total local store
			       -   4 * 1024  // code start address
			       -  code_size
			       - stack_size
			       -   1280 * 4  // DTL size from ALF kernel.c
			       -   8 * 1024; // safety margin of 8kb

    if (buffer_size > buffer_space)
      VSIP_THROW(std::bad_alloc());
    
    int num_buffers = (buffer_size * 2 > buffer_space) ? 1 : 2;

    std::auto_ptr<Task> task =
      mgr->reserve_lwp_task(buffer_size, num_buffers,
			    (uintptr_t)code_ea, code_size, 0);

    ukp.code_ea    = (uintptr_t)code_ea;
    ukp.code_size  = code_size;
    ukp.nspe       = std::min(spes, chunks);

    for (index_type i=0; i<spes && i<chunks; ++i)
    {
      // If chunks don't divide evenly, give the first SPEs one extra.
      length_type chunks_this_spe = (i < chunks % spes) ? chunks_per_spe + 1
	                                                : chunks_per_spe;

      vh0.set_workblock(ukp.in_stream[0],  chunks_this_spe);
      vh1.set_workblock(ukp.in_stream[1],  chunks_this_spe);
      vh2.set_workblock(ukp.in_stream[2],  chunks_this_spe);
      vh3.set_workblock(ukp.out_stream[0], chunks_this_spe);

      Workblock block = task.get()->create_workblock(chunks_this_spe + ukp.pre_chunks);
      ukp.rank = i;
      block.set_parameters(ukp);
      block.enqueue();
    }

    task.get()->sync();
  }
};



template <typename FuncT>
class Ukernel
{
public:
  Ukernel(FuncT& func)
    : func_(func)
  {}

  template <typename View1, typename View2>
  void apply_vpp(View1 in, View2 out)
  {
    func_.apply(in, out);
  }

  void operator()()
  {
    Stream_spe<FuncT>::apply(func_);
  }

  template <typename View1, typename View2>
  void operator()(View1 in, View2 out)
  {
    Stream_spe<FuncT>::apply(func_, in, out);
  }

  template <typename View0, typename View1, typename View2>
  void operator()(View0 in0, View1 in1, View2 out)
  {
    Stream_spe<FuncT>::apply(func_, in0, in1, out);
  }

  template <typename View0,
	    typename View1,
	    typename View2,
	    typename View3>
  void operator()(View0 in0, View1 in1, View2 in2, View3 out)
  {
    Stream_spe<FuncT>::apply(func_, in0, in1, in2, out);
  }

private:
  FuncT &func_;
};

} // namespace vsip::impl::ukernel
} // namespace vsip::impl
} // namespace vsip

#endif
