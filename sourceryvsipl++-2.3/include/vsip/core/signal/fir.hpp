/* Copyright (c) 2006 by CodeSourcery, Inc.  All rights reserved. */

/** @file    vsip/core/signal/fir.hpp
    @author  Stefan Seefeld
    @date    2006-11-08
    @brief   VSIPL++ Library: FIR filter definition.
*/

#ifndef VSIP_CORE_SIGNAL_FIR_HPP
#define VSIP_CORE_SIGNAL_FIR_HPP

#include <vsip/support.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/vector.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/allocation.hpp>
#include <vsip/core/extdata.hpp>
#include <vsip/core/profile.hpp>
#ifndef VSIP_IMPL_REF_IMPL
# include <vsip/opt/dispatch.hpp>
# include <vsip/opt/signal/fir_opt.hpp>
# ifdef VSIP_IMPL_HAVE_IPP
#  include <vsip/opt/ipp/fir.hpp>
# endif
# ifdef VSIP_IMPL_CBE_SDK
#  include <vsip/opt/cbe/cml/fir.hpp>
# endif
#endif
#if VSIP_IMPL_HAVE_CVSIP
# include <vsip/core/cvsip/fir.hpp>
#endif

#ifndef VSIP_IMPL_REF_IMPL

namespace vsip_csl
{
namespace dispatcher
{
template<>
struct List<op::fir>
{
  typedef Make_type_list<be::intel_ipp,
			 be::cml,
			 be::opt,
			 be::cvsip>::type type;
};


/// The following specializations are required since the generic form would
/// pass the first argument (aligned_array) by value, thus loosing ownership.
template <typename T, symmetry_type S, obj_state C, typename B>
struct Profile_nop_policy<
  op::fir, 
  impl::Ref_counted_ptr<impl::Fir_backend<T, S, C> >(impl::aligned_array<T>,
						     length_type,
						     length_type,
						     length_type,
						     unsigned,
						     alg_hint_type),
  B>
{
  Profile_nop_policy(impl::aligned_array<T> const &,
                     length_type,
                     length_type,
                     length_type,
                     unsigned,
                     alg_hint_type) {}
};

template <typename T, symmetry_type S, obj_state C, typename B>
struct Profile_policy<
  op::fir,
  impl::Ref_counted_ptr<impl::Fir_backend<T, S, C> >(impl::aligned_array<T>,
						     length_type,
						     length_type,
						     length_type,
						     unsigned,
						     alg_hint_type),
  B>
{
  typedef impl::profile::Scope<Profile_feature<op::fir>::value> scope_type;
  typedef Evaluator<op::fir, B,
		    impl::Ref_counted_ptr<impl::Fir_backend<T, S, C> >(impl::aligned_array<T>,
								       length_type,
								       length_type,
								       length_type,
								       unsigned,
								       alg_hint_type)>
  evaluator_type;

  Profile_policy(impl::aligned_array<T> const &k,
                 length_type ks,
                 length_type is,
                 length_type d,
                 unsigned n,
                 alg_hint_type h)
    : scope(evaluator_type::name(), evaluator_type::op_count(ks, is, d, n, h)) {}

  scope_type scope;  
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip
{
namespace impl
{
namespace diag_detail
{
struct Diagnose_fir;
} // namespace vsip::impl::diag_detail
} // namespace vsip::impl
#endif

template <typename  T = VSIP_DEFAULT_VALUE_TYPE,
          symmetry_type S = nonsym,
          obj_state C = state_save,
          unsigned  N = 0,
          alg_hint_type H = alg_time>
class Fir : impl::profile::Accumulator<impl::profile::signal>
{
  typedef impl::profile::Accumulator<impl::profile::signal> accumulator_type;
  typedef impl::Fir_backend<T, S, C> backend;
  typedef impl::Ref_counted_ptr<backend> backend_ptr;
public:
  static symmetry_type const symmetry = S;
  static obj_state const continuous_filter = C;
 
  template <typename BlockT>
  Fir(const_Vector<T,BlockT> kernel,
      length_type input_size,
      length_type decimation = 1)
    VSIP_THROW((std::bad_alloc))
    : accumulator_type(impl::signal_detail::Description<1, T>::tag(
			 "Fir", input_size), 
                       impl::signal_detail::Op_count_fir<T>::value
                       (backend::order(kernel.size()), input_size, decimation)),
#ifdef VSIP_IMPL_REF_IMPL
      backend_(new impl::cvsip::Fir_impl<T, S, C>
               (copy(kernel.block()), kernel.size(), input_size, decimation,
                N, H))
#else
      backend_(vsip_csl::dispatch<vsip_csl::dispatcher::op::fir, backend_ptr>
               (copy(kernel.block()), kernel.size(), input_size, decimation,
                N, H))
#endif
  {}
  // This constructor is used only by the C-VSIPL bindings,
  // as there we unpack the kernel prior to constructing the Fir object.
  Fir(impl::aligned_array<T> kernel,
      length_type kernel_size,
      length_type input_size,
      length_type decimation = 1)
    VSIP_THROW((std::bad_alloc))
    : accumulator_type(impl::signal_detail::Description<1, T>::tag(
			 "Fir", input_size), 
                       impl::signal_detail::Op_count_fir<T>::value
                       (backend::order(kernel_size), input_size, decimation)),
#ifdef VSIP_IMPL_REF_IMPL
      backend_(new impl::cvsip::Fir_impl<T, S, C>
               (kernel, kernel_size, input_size, decimation,
                N, H))
#else
      backend_(vsip_csl::dispatch<vsip_csl::dispatcher::op::fir, backend_ptr>
               (kernel, kernel_size, input_size, decimation,
                N, H))
#endif
  {}
  Fir(Fir const &fir)
    : accumulator_type(fir),
      backend_(continuous_filter == state_save ?
               backend_ptr(fir.backend_->clone()) : // deep copy
               fir.backend_)                        // shallow copy
  {}
  ~Fir() VSIP_NOTHROW {}
  Fir &operator= (Fir const &fir)
  {
    accumulator_type::operator= (fir);
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
    using vsip::impl::Block_layout;
    using vsip::impl::Adjust_layout_complex;
    using vsip::impl::Cmplx_inter_fmt;

    typename accumulator_type::Scope scope(*this);
    assert(in.size() == backend_->input_size());
    assert(out.size() == backend_->output_size());

    typedef typename Block_layout<Block0>::layout_type LP0;
    typedef typename Block_layout<Block1>::layout_type LP1;
    typedef typename Adjust_layout_complex<Cmplx_inter_fmt, LP0>::type use_LP0;
    typedef typename Adjust_layout_complex<Cmplx_inter_fmt, LP1>::type use_LP1;

    impl::Ext_data<Block0, use_LP0> ext_in(in.block());
    impl::Ext_data<Block1, use_LP1> ext_out(out.block());
    return backend_->apply(ext_in.data(), ext_in.stride(0), ext_in.size(0),
                           ext_out.data(), ext_out.stride(0), ext_out.size(0));
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
  static impl::aligned_array<T> copy(Block const &block)
  {
    impl::aligned_array<T> array(block.size());
    Dense<1, T> tmp(block.size(), array.get());
    tmp.admit(false);
    impl::assign<1>(tmp, block);
    tmp.release();
    return array;
  }

  backend_ptr backend_;
};

} // namespace vsip

#endif // VSIP_CORE_SIGNAL_FIR_HPP

