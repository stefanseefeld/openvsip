//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_signal_fft_functor_hpp_
#define ovxx_signal_fft_functor_hpp_

#include <ovxx/block_traits.hpp>
#include <ovxx/parallel/support.hpp>

namespace ovxx
{
namespace expr
{
namespace op
{

// Expression block operator for FFTs
template <dimension_type D, typename BE, typename W>
struct fft
{
  template <typename B>
  class Functor
  {
  public:
    static dimension_type const dim = BE::dim;
    typedef BE backend_type;
    typedef W workspace_type;
    typedef typename backend_type::output_value_type result_type;

    typedef B block_type;
    typedef typename block_type::map_type map_type;
    typedef typename block_traits<block_type>::plain_type block_ref_type;
    typedef Functor<typename ovxx::distributed_local_block<B>::type> local_type;

    template <typename V>
    Functor(V arg, Domain<D> const &output_size,
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

    length_type size(dimension_type, dimension_type d) const
    {
      return output_size_[d].size();
    }

    local_type local() const
    {
      length_type rows = output_size_[0].size();
      length_type cols = output_size_[1].size();
      if (BE::axis == 0)
      {
	cols = parallel::subblock_domain<2>(arg_)[1].size();
	rows = (cols == 0) ? 0 : rows;
      }
      else
      {
	rows = parallel::subblock_domain<2>(arg_)[0].size();
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
    workspace_type &workspace() const { return workspace_;}

    template <typename R>
    void apply(R &result) const
    {
      workspace_.out_of_place(backend_, arg_, result);
    }

  private:
    block_ref_type arg_;
    Domain<D> output_size_;
    backend_type &backend_;
    workspace_type &workspace_;
  };
};

} // namespace ovxx::expr::op

template <typename F, typename sfinae = void>
struct is_fft_functor { static bool const value = false;};

template <typename F>
struct is_fft_functor<F, 
  typename enable_if<signal::fft::is_fft_backend<
    typename F::backend_type>::value>::type>
{
  static bool const value = true;
};

template <typename F, typename sfinae = void>
struct is_fftm_functor { static bool const value = false;};

template <typename F>
struct is_fftm_functor<F, 
  typename enable_if<signal::fft::is_fftm_backend<
    typename F::backend_type>::value>::type>
{
  static bool const value = true;
};

} // namespace ovxx::expr
} // namespace ovxx

#endif
