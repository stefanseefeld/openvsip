/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/fft/functor.cpp
    @author  Jules Bergmann, Stefan Seefeld
    @date    2009-07-19
    @brief   VSIPL++ Library: FFT expression functor
*/

#ifndef VSIP_OPT_FFT_FUNCTOR_HPP
#define VSIP_OPT_FFT_FUNCTOR_HPP

#include <vsip/core/block_traits.hpp>
#include <vsip/core/parallel/support_block.hpp>
#include <vsip/core/c++0x.hpp>

namespace vsip_csl
{
namespace expr
{
namespace op
{

template <dimension_type D, typename Backend, typename Workspace>
struct fft
{
  template <typename Block>
  class Functor
  {
  public:
    static dimension_type const dim = Backend::dim;
    typedef Backend backend_type;
    typedef Workspace workspace_type;
    typedef typename backend_type::output_value_type result_type;

    typedef Block block_type;
    typedef typename block_type::map_type map_type;
    typedef typename impl::View_block_storage<block_type>::plain_type block_ref_type;
    typedef Functor<typename impl::Distributed_local_block<Block>::type> local_type;

    template <typename View>
    Functor(View arg,
	    Domain<D> const &output_size,
	    backend_type &backend,
	    workspace_type &workspace)
      : arg_(arg.block()),
	output_size_(output_size),
	backend_(backend),
	workspace_(workspace)
    {}

    Functor(block_ref_type arg,
	    Domain<D> const &output_size,
	    backend_type &backend,
	    workspace_type &workspace)
      : arg_(arg),
	output_size_(output_size),
	backend_(backend),
	workspace_(workspace)
    {}

    Functor(Functor const &o)
      : arg_(o.arg_),
	output_size_(o.output_size_),
	backend_(o.backend_),
	workspace_(o.workspace_)
    {}

    length_type size() const
    {
      return output_size_.size();
    }

    length_type size(dimension_type block_dim ATTRIBUTE_UNUSED, dimension_type d) const
    {
      assert(block_dim == D);
      return output_size_[d].size();
    }

    local_type local() const
    {
      // The local output size is the same as the global output size
      // along the dimension of the FFT.  Its size along the other
      // dimension matches that of the input local block.
      length_type rows = output_size_[0].size();
      length_type cols = output_size_[1].size();
      if (Backend::axis == 0)
      {
	cols = impl::block_subblock_domain<2>(arg_)[1].size();
	rows = (cols == 0) ? 0 : rows;
      }
      else
      {
	rows = impl::block_subblock_domain<2>(arg_)[0].size();
	cols = (rows == 0) ? 0 : cols;
      }
      Domain<2> l_output_size(rows, cols);

      return local_type(get_local_block(arg_),
			l_output_size,
			backend_,
			workspace_);
    }

    map_type const &map() const { return arg_.map();}
    block_type const &arg() const { return arg_;}
    backend_type const &backend() const { return backend_;}
    workspace_type const &workspace() const { return workspace_;}

    template <typename ResBlock>
    void apply(ResBlock &result) const
    {
      workspace_.out_of_place_blk(&backend_, arg_, result);
    }

  private:
    block_ref_type arg_;
    Domain<D> output_size_;
    backend_type &backend_;
    workspace_type &workspace_;
  };
};

} // namespace vsip_csl::expr::op

template <typename F, typename Enable = void>
struct is_fft_functor { static bool const value = false;};

template <typename F>
struct is_fft_functor<F, 
  typename vsip::impl::enable_if<vsip::impl::fft::is_fft_backend<
    typename F::backend_type> >::type>
{
  static bool const value = true;
};

template <typename F, typename Enable = void>
struct is_fftm_functor { static bool const value = false;};

template <typename F>
struct is_fftm_functor<F, 
  typename vsip::impl::enable_if<vsip::impl::fft::is_fftm_backend<
    typename F::backend_type> >::type>
{
  static bool const value = true;
};

} // namespace vsip_csl::expr
} // namespace vsip_csl

#endif
