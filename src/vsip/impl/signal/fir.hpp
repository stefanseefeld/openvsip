//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_signal_fir_hpp_
#define vsip_impl_signal_fir_hpp_

#include <vsip/support.hpp>
#include <vsip/impl/signal/types.hpp>
#include <vsip/vector.hpp>
#include <vsip/domain.hpp>
#include <vsip/dda.hpp>
#include <ovxx/aligned_array.hpp>
#include <ovxx/signal/fir.hpp>
#include <ovxx/dispatch.hpp>
#if OVXX_HAVE_CVSIP
# include <ovxx/cvsip/fir.hpp>
#endif

namespace ovxx
{
namespace dispatcher
{
template<>
struct List<op::fir>
{
  typedef make_type_list<be::user,
			 be::cuda,
			 be::generic,
			 be::cvsip>::type type;
};

} // namespace ovxx::dispatcher
} // namespace ovxx

namespace vsip
{
namespace impl
{
namespace diag_detail
{
struct Diagnose_fir;
} // namespace vsip::impl::diag_detail
} // namespace vsip::impl

template <typename  T = VSIP_DEFAULT_VALUE_TYPE,
          symmetry_type S = nonsym,
          obj_state C = state_save,
          unsigned  N = 0,
          alg_hint_type H = alg_time>
class Fir
{
  typedef ovxx::signal::Fir_backend<T, S, C> backend;
  typedef std::shared_ptr<backend> backend_ptr;
public:
  static symmetry_type const symmetry = S;
  static obj_state const continuous_filter = C;
 
  template <typename BlockT>
  Fir(const_Vector<T,BlockT> kernel,
      length_type input_size,
      length_type decimation = 1)
    VSIP_THROW((std::bad_alloc))
    : backend_(ovxx::dispatch<ovxx::dispatcher::op::fir, backend_ptr>
               (copy(kernel.block()), kernel.size(), input_size, decimation,
                N, H))
  {}
  // This constructor is used only by the C-VSIPL bindings,
  // as there we unpack the kernel prior to constructing the Fir object.
  Fir(ovxx::aligned_array<T> kernel,
      length_type kernel_size,
      length_type input_size,
      length_type decimation = 1)
    VSIP_THROW((std::bad_alloc))
    : backend_(ovxx::dispatch<ovxx::dispatcher::op::fir, backend_ptr>
               (kernel, kernel_size, input_size, decimation,
                N, H))
  {}
  Fir(Fir const &fir)
    : backend_(continuous_filter == state_save ?
               backend_ptr(fir.backend_->clone()) : // deep copy
               fir.backend_)                        // shallow copy
  {}
  ~Fir() VSIP_NOTHROW {}
  Fir &operator= (Fir const &fir)
  {
    if (continuous_filter == state_save)
      backend_ = backend_ptr(fir.backend_->clone()); // deep copy
    else 
      backend_ = fir.backend_;                       // shallow copy
    return *this;
  }

  length_type kernel_size() const VSIP_NOTHROW 
  { return backend_->kernel_size();}
  length_type filter_order() const VSIP_NOTHROW 
  { return backend_->filter_order();}
  length_type input_size() const VSIP_NOTHROW 
  { return backend_->input_size();}
  length_type output_size() const VSIP_NOTHROW 
  { return backend_->output_size();}
  vsip::length_type decimation() const VSIP_NOTHROW 
  { return backend_->decimation();}
  obj_state continuous_filtering() const VSIP_NOTHROW { return C;}

  template <typename Block0, typename Block1>
  length_type
  operator()(const_Vector<T, Block0> in, Vector<T, Block1> out) VSIP_NOTHROW
  {
    using vsip::get_block_layout;
    using ovxx::adjust_layout_storage_format;

    OVXX_PRECONDITION(in.size() == backend_->input_size());
    OVXX_PRECONDITION(out.size() == backend_->output_size());

    typedef typename get_block_layout<Block0>::type LP0;
    typedef typename get_block_layout<Block1>::type LP1;
    typedef typename adjust_layout_storage_format<array, LP0>::type use_LP0;
    typedef typename adjust_layout_storage_format<array, LP1>::type use_LP1;

#ifdef VSIP_IMPL_HAVE_CUDA
    if (backend_->supports_cuda_memory())
    {
      impl::cuda::dda::Data<Block0, dda::in, use_LP0> data_in(in.block());
      impl::cuda::dda::Data<Block1, dda::out, use_LP1> data_out(out.block());

      return backend_->apply(data_in.ptr(), data_in.stride(0), data_in.size(0),
                             data_out.ptr(), data_out.stride(0), data_out.size(0));
    }
    else
#endif
    {
      dda::Data<Block0, dda::in, use_LP0> data_in(in.block());
      dda::Data<Block1, dda::out, use_LP1> data_out(out.block());

      return backend_->apply(data_in.ptr(), data_in.stride(0), data_in.size(0),
                             data_out.ptr(), data_out.stride(0), data_out.size(0));
    }
  }

  void reset() VSIP_NOTHROW { backend_->reset();}

  float impl_performance(char const *what) const VSIP_NOTHROW
  {
    if      (!strcmp(what, "mops"))  return this->mflops();
    else if (!strcmp(what, "time"))  return this->total();
    else if (!strcmp(what, "count")) return this->count();
    else return 0.f;
  }

  friend class vsip::impl::diag_detail::Diagnose_fir;
private:
  template <typename Block>
  static ovxx::aligned_array<T> copy(Block const &block)
  {
    ovxx::aligned_array<T> array(block.size());
    Dense<1, T> tmp(block.size(), array.get());
    tmp.admit(false);
    ovxx::assign<1>(tmp, block);
    tmp.release();
    return array;
  }

  backend_ptr backend_;
};

} // namespace vsip

#endif

