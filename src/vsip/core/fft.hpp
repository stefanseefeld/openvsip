//
// Copyright (c) 2006, 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_FFT_HPP
#define VSIP_CORE_FFT_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/config.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/fft/backend.hpp>
#include <vsip/core/fft/util.hpp>
#include <vsip/core/fft/ct_workspace.hpp>
#ifndef VSIP_IMPL_REF_IMPL
# include <vsip/opt/fft/workspace.hpp>
# include <vsip/opt/dispatch.hpp>
#endif
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/profile.hpp>

#ifndef VSIP_IMPL_REF_IMPL
# ifdef VSIP_IMPL_CBE_SDK_FFT
#  include <vsip/opt/cbe/ppu/fft.hpp>
# endif
# if VSIP_IMPL_SAL_FFT
#  include <vsip/opt/sal/fft.hpp>
# endif
# if VSIP_IMPL_IPP_FFT
#  include <vsip/opt/ipp/fft.hpp>
# endif
# if VSIP_IMPL_FFTW3
#  include <vsip/opt/fftw3/fft.hpp>
# endif
# if VSIP_IMPL_CUDA_FFT
#  include <vsip/opt/cuda/fft.hpp>
# endif
#endif // VSIP_IMPL_REF_IMPL

#if VSIP_IMPL_CVSIP_FFT
# include <vsip/core/cvsip/fft.hpp>
#endif
#if VSIP_IMPL_DFT_FFT
# include <vsip/core/fft/dft.hpp>
#endif
#if VSIP_IMPL_NO_FFT
# include <vsip/core/fft/no_fft.hpp>
#endif

#include <cstring>

namespace vsip_csl
{
namespace dispatcher
{
#ifndef VSIP_IMPL_REF_IMPL
template <dimension_type D,
	  typename I,
	  typename O,
	  int S,
	  return_mechanism_type R,
	  unsigned N>
struct List<op::fft<D, I, O, S, R, N> >
{
  typedef Make_type_list<be::user,
			 be::cuda,
			 be::cbe_sdk,
			 be::mercury_sal,
			 be::intel_ipp,
			 be::fftw3,
			 be::cvsip,
			 be::generic,
			 be::no_fft>::type type;
};

template <typename I,
	  typename O,
	  int A,
	  int D,
	  return_mechanism_type R,
	  unsigned N>
struct List<op::fftm<I, O, A, D, R, N> >
{
  typedef Make_type_list<be::user,
			 be::cuda,
			 be::cbe_sdk,
			 be::mercury_sal,
			 be::intel_ipp,
			 be::fftw3,
			 be::cvsip,
			 be::generic,
			 be::no_fft>::type type;
};
#endif

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip
{
namespace impl
{

namespace diag_detail
{
struct Diagnose_fft;
struct Diagnose_fftm;
}

namespace fft
{

/// These numbers are 'number-of-times' constants as they
/// map to FFTW planning strategies. We list them here since
/// they are used outside the FFTW backend, too (CVSIP).
///
/// number-of-times >= 32 or 0 (infinity)... patient
/// number-of-times >= 12 .................. measure
/// else ................................... estimate
enum fftw_planning { estimate=1, measure=12, patient=32};

template <dimension_type D, typename I, typename O, int A, int E>
class Interface : public profile::Accumulator<profile::signal>
{
  typedef profile::Accumulator<profile::signal> accumulator_type;
public:
  static dimension_type const dim = D;
  typedef typename impl::scalar_of<I>::type scalar_type;

  Interface(Domain<D> const &dom, scalar_type scale, 
	    bool is_fftm, int dir, return_mechanism_type rm)
    : accumulator_type(Description<D, I, O>::tag(is_fftm, dom, dir, rm, A),
                       is_fftm ?
                       (A == 1 ?
                        // row-wise Fftm
                        dom[0].size() * Op_count<I, O>::value
                        (io_size<1, O, I, 0>::size(dom[1]).size()) :
                        // column-wise Fftm
                        dom[1].size() * Op_count<I, O>::value
                        (io_size<1, O, I, 0>::size(dom[0]).size())) :
                       // ! is_fftm
                       Op_count<I, O>::value
                       (io_size<D, O, I, A>::size(dom).size())),
      input_size_(io_size<D, I, O, A>::size(dom)),
      output_size_(io_size<D, O, I, A>::size(dom)),
      scale_(scale)
  {}

  /// Returns a Domain<> object with first index set to :literal:`0`,
  /// stride set to :literal:`1`, and size reflecting the appropriate
  /// input view size for this :literal:`Fft` object.
  Domain<dim> const& 
  input_size() const VSIP_NOTHROW 
  { return this->input_size_;}
  
  /// Returns a Domain<> object with first index set to :literal:`0`,
  /// stride set to :literal:`1`, and size reflecting the appropriate
  /// output view size for this :literal:`Fft` object.
  Domain<dim> const& 
  output_size() const VSIP_NOTHROW 
  { return this->output_size_;}
  
  /// Returns the scale factor used in this :literal:`Fft` object.
  scalar_type 
  scale() const VSIP_NOTHROW 
  { return this->scale_;}
  
  /// Returns :literal:`true` if this is a forward Fast Fourier Transformation.
  bool 
  forward() const VSIP_NOTHROW
  { return E == -1;}
  
  float impl_performance(char const *what) const
  {
    if      (!strcmp(what, "mops")) return this->mops();
    else if (!strcmp(what, "time")) return this->total();
    else if (!strcmp(what, "count")) return this->count();
    else return 0.f;
  }

protected:
  Domain<dim> input_size_;
  Domain<dim> output_size_;
  scalar_type scale_;
};

} // namespace vsip::impl::fft

template <dimension_type D,                      //< Dimension
	  typename I,                            //< Input type
	  typename O,                            //< Output type
	  typename L,                            //< Dispatcher list
	  int S = 0,                             //< Special dimension
	  return_mechanism_type = by_value,      //< Return mechanism
	  unsigned N = 0,                        //< Number of times
	  alg_hint_type = alg_time>              //< algorithm Hint
class Fft_facade;

template <dimension_type D,
	  typename I,
	  typename O,
	  typename L,
	  int S,
	  unsigned N,
	  alg_hint_type H>
class Fft_facade<D, I, O, L, S, by_value, N, H>
  : public fft::Interface<D, I, O,
			  fft::axis<I, O, S>::value,
			  fft::exponent<I, O, S>::value>
{
public:
  static int const axis = fft::axis<I, O, S>::value;
  static int const exponent = fft::exponent<I, O, S>::value;
  typedef fft::Interface<D, I, O, axis, exponent> base;
  typedef fft::Fft_backend<D, I, O, S> backend_type;

#if VSIP_IMPL_REF_IMPL
  typedef fft::Ct_workspace<D, I, O> workspace;
#else
  typedef fft::workspace<D, I, O> workspace;
#endif
  typedef vsip_csl::dispatcher::Dispatcher<
    vsip_csl::dispatcher::op::fft<D, I, O, S, by_value, N>,
    std::auto_ptr<backend_type>(Domain<D> const &, typename base::scalar_type), L>
    dispatcher_type;

  Fft_facade(Domain<D> const& dom, typename base::scalar_type scale)
    VSIP_THROW((std::bad_alloc))
    : base(dom, scale, false, S, by_value),
#ifdef VSIP_IMPL_REF_IMPL
      backend_(cvsip::create<backend_type>(dom, scale, N)),
#else
      backend_(dispatcher_type::dispatch(dom, scale)),
#endif
      workspace_(backend_.get(), this->input_size(), this->output_size(), scale)
  {}

#ifdef VSIP_IMPL_REF_IMPL
  /// Returns the Fast Fourier Transform of :literal:`in`.
  template <typename ViewT>
  typename fft::result<O, typename ViewT::block_type>::view_type
  operator()(ViewT in) VSIP_THROW((std::bad_alloc))
  {
    typename base::Scope scope(*this);
    assert(extent(in) == extent(this->input_size()));
    typedef fft::result<O, typename ViewT::block_type> traits;
    typename traits::view_type out(traits::create(this->output_size(),
                                                 in.block().map()));
    workspace_.out_of_place(this->backend_.get(), in, out);
    return out;
  }
#else
  /// Returns the Fast Fourier Transform of :literal:`in`.
  template <typename ViewT>
  typename fft::Result_rbo<D, I, O, ViewT, workspace, S>::view_type
  operator()(ViewT in) VSIP_THROW((std::bad_alloc))
  {
    typename base::Scope scope(*this);
    assert(extent(in) == extent(this->input_size()));
    typedef fft::Result_rbo<D, I, O, ViewT, workspace, S> traits;
    typedef typename traits::functor_type functor_type;
    typedef typename traits::block_type   block_type;
    typedef typename traits::view_type    view_type;

    functor_type rf(in, this->output_size(), *(this->backend_.get()),
		    workspace_);
    block_type block(rf);
    return view_type(block);
  }
#endif
  friend class vsip::impl::diag_detail::Diagnose_fft;
private:
  std::auto_ptr<backend_type> backend_;
  workspace workspace_;
};

template <dimension_type D,
	  typename I,
	  typename O,
	  typename L,
	  int S,
	  unsigned N,
	  alg_hint_type H>
class Fft_facade<D, I, O, L, S, vsip::by_reference, N, H>
  : public fft::Interface<D, I, O,
			  fft::axis<I, O, S>::value,
			  fft::exponent<I, O, S>::value>
{
public:
  static int const axis = fft::axis<I, O, S>::value;
  static int const exponent = fft::exponent<I, O, S>::value;
  typedef fft::Interface<D, I, O, axis, exponent> base;
  typedef fft::Fft_backend<D, I, O, S> backend_type;
#if VSIP_IMPL_REF_IMPL
  typedef fft::Ct_workspace<D, I, O> workspace;
#else
  typedef fft::workspace<D, I, O> workspace;
#endif
  typedef vsip_csl::dispatcher::Dispatcher<
    vsip_csl::dispatcher::op::fft<D, I, O, S, by_reference, N>,
    std::auto_ptr<backend_type>(Domain<D> const &, typename base::scalar_type), L>
    dispatcher_type;

  Fft_facade(Domain<D> const& dom, typename base::scalar_type scale)
    VSIP_THROW((std::bad_alloc))
    : base(dom, scale, false, S, by_reference),
#ifdef VSIP_IMPL_REF_IMPL
      backend_(cvsip::create<backend_type>(dom, scale, N)),
#else
      backend_(dispatcher_type::dispatch(dom, scale)),
#endif
      workspace_(backend_.get(), this->input_size(), this->output_size(), scale)
  {}

  /// Computes the Fast Fourier Transform of :literal:`in` and stores the result
  /// in :literal:`out`.
  template <typename Block0, typename Block1,
 	    template <typename, typename> class View0,
 	    template <typename, typename> class View1>
  View1<O,Block1>
  operator()(View0<I,Block0> in, View1<O,Block1> out)
    VSIP_NOTHROW
  {
    typename base::Scope scope(*this);
    VSIP_IMPL_STATIC_ASSERT((View0<I,Block0>::dim == View1<O,Block1>::dim));
    assert(extent(in) == extent(this->input_size()));
    assert(extent(out) == extent(this->output_size()));
    workspace_.out_of_place(this->backend_.get(), in, out);
    return out;
  }

  /// Computes the Fast Fourier Transform of :literal:`inout` in place.
  template <typename BlockT, template <typename, typename> class View>
  View<I,BlockT>
  operator()(View<I,BlockT> inout) VSIP_NOTHROW
  {
    typename base::Scope scope(*this);
    assert(extent(inout) == extent(this->input_size()));
    assert(extent(inout) == extent(this->output_size()));
    workspace_.in_place(this->backend_.get(), inout);
    return inout;
  }

  friend class vsip::impl::diag_detail::Diagnose_fft;
private:
  std::auto_ptr<backend_type> backend_;
  workspace workspace_;
};

template <typename I,                       //< Input type
	  typename O,                       //< Output type
	  typename L,                       //< Dispatcher list
	  int A,                            //< Axis
	  int D,                            //< Direction
	  return_mechanism_type,            //< Return mechanism
	  unsigned N = 0,                   //< Number of times
	  alg_hint_type = alg_time>         //< algorithm Hint
class Fftm_facade;

template <typename I,
	  typename O,
	  typename L,
	  int A,
	  int D,
	  unsigned N,
	  alg_hint_type H>
class Fftm_facade<I, O, L, A, D, by_value, N, H>
  : public fft::Interface<2, I, O, 1 - A, D == fft_fwd ? -1 : 1>
{
  static int const axis = 1 - A;
  static int const exponent = D == fft_fwd ? -1 : 1;
  typedef fft::Interface<2, I, O, axis, exponent> base;
  typedef typename fft::Fftm_backend<I, O, A, D> backend_type;
#if VSIP_IMPL_REF_IMPL
  typedef fft::Ct_workspace<2, I, O> workspace;
#else
  typedef fft::workspace<2, I, O> workspace;
#endif
  typedef vsip_csl::dispatcher::Dispatcher<
    vsip_csl::dispatcher::op::fftm<I, O, A, D, by_value, N>,
    std::auto_ptr<backend_type>(Domain<2> const &, typename base::scalar_type), L>
    dispatcher_type;
public:
  Fftm_facade(Domain<2> const& dom, typename base::scalar_type scale)
    VSIP_THROW((std::bad_alloc))
    : base(dom, scale, true, D, by_value),
#ifdef VSIP_IMPL_REF_IMPL
      backend_(cvsip::create<backend_type>(dom, scale, N)),
#else
      backend_(dispatcher_type::dispatch(dom, scale)),
#endif
      workspace_(backend_.get(), this->input_size(), this->output_size(), scale)
  {}

#ifdef VSIP_IMPL_REF_IMPL
  /// Returns the Fast Fourier Transform of :literal:`in`.
  template <typename BlockT>
  typename fft::result<O,BlockT>::view_type
  operator()(const_Matrix<I,BlockT> in)
     VSIP_THROW((std::bad_alloc))
  {
    typename base::Scope scope(*this);
    typedef fft::result<O,BlockT> traits;
    typename traits::view_type out(traits::create(this->output_size(),
                                                 in.block().map()));
    assert(extent(in) == extent(this->input_size()));
    if (is_global_map<typename BlockT::map_type>::value &&
        in.block().map().num_subblocks(axis) != 1)
      VSIP_IMPL_THROW(unimplemented(
        "Fftm requires dimension along FFT to not be distributed"));
    workspace_.out_of_place(this->backend_.get(), in.local(), out.local());
    return out;
  }
#else
  /// Returns the Fast Fourier Transform of :literal:`in`.
  template <typename BlockT>  
  typename fft::Result_fftm_rbo<I, O, BlockT, workspace, A, D>::view_type
  operator()(const_Matrix<I,BlockT> in)
    VSIP_THROW((std::bad_alloc))
  {
    typename base::Scope scope(*this);
    assert(extent(in) == extent(this->input_size()));

    /* TODO: Return_blocks don't have a valid map() yet
    if (is_global_map<typename BlockT::map_type>::value &&
	in.block().map().num_subblocks(A) != 1)
      VSIP_IMPL_THROW(unimplemented(
	"Fftm requires dimension along FFT to not be distributed"));
    */

    typedef fft::Result_fftm_rbo<I, O, BlockT, workspace, A, D> traits;
    typedef typename traits::functor_type functor_type;
    typedef typename traits::block_type   block_type;
    typedef typename traits::view_type    view_type;

    functor_type rf(in, this->output_size(), *(this->backend_.get()),
		    workspace_);
    block_type block(rf);
    return view_type(block);
 }
#endif

  friend class vsip::impl::diag_detail::Diagnose_fftm;
private:
  std::auto_ptr<backend_type> backend_;
  workspace workspace_;
};

template <typename I,
	  typename O,
	  typename L,
	  int A,
	  int D,
	  unsigned N,
	  alg_hint_type H>
class Fftm_facade<I, O, L, A, D, vsip::by_reference, N, H>
  : public fft::Interface<2, I, O, 1 - A, D == fft_fwd ? -1 : 1>
{
  // Fftm and 2D Fft share some underlying logic.
  // The 'Special dimension' (S) template parameter in 2D Fft uses '0' to
  // represent column-first and '1' for a row-first transformation, while
  // the Fftm 'Axis' (A) parameter uses '0' to represent row-wise, and
  // '1' for column-wise transformation.
  // Thus, by using '1 - A' here we can share the implementation, too.
  static int const axis = 1 - A;
  static int const exponent = D == fft_fwd ? -1 : 1;
  typedef fft::Interface<2, I, O, axis, exponent> base;
  typedef typename fft::Fftm_backend<I, O, A, D> backend_type;
#if VSIP_IMPL_REF_IMPL
  typedef fft::Ct_workspace<2, I, O> workspace;
#else
  typedef fft::workspace<2, I, O> workspace;
#endif
  typedef vsip_csl::dispatcher::Dispatcher<
    vsip_csl::dispatcher::op::fftm<I, O, A, D, by_reference, N>,
    std::auto_ptr<backend_type>(Domain<2> const &, typename base::scalar_type), L>
    dispatcher_type;
public:
  Fftm_facade(Domain<2> const& dom, typename base::scalar_type scale)
    VSIP_THROW((std::bad_alloc))
    : base(dom, scale, true, D, by_reference),
#ifdef VSIP_IMPL_REF_IMPL
      backend_(cvsip::create<backend_type>(dom, scale, N)),
#else
      backend_(dispatcher_type::dispatch(dom, scale)),
#endif
      workspace_(backend_.get(), this->input_size(), this->output_size(), scale)
  {}

  /// Computes the Fast Fourier Transform of :literal:`in` and stores the result
  /// in :literal:`out`.
  template <typename Block0, typename Block1>
  Matrix<O,Block1>
  operator()(const_Matrix<I,Block0> in, Matrix<O,Block1> out)
    VSIP_NOTHROW
  {
    typename base::Scope scope(*this);
    assert(extent(in)  == extent(this->input_size()));
    assert(extent(out) == extent(this->output_size()));
    if (is_global_map<typename Block0::map_type>::value ||
	is_global_map<typename Block1::map_type>::value)
    {
      if (in.block().map().num_subblocks(axis) != 1 ||
	  out.block().map().num_subblocks(axis) != 1)
	VSIP_IMPL_THROW(unimplemented(
	  "Fftm requires dimension along FFT to not be distributed"));
      if (global_domain(in) != global_domain(out))
	VSIP_IMPL_THROW(unimplemented(
	  "Fftm requires input and output to have same mapping"));
    }
    workspace_.out_of_place(this->backend_.get(), in.local(), out.local());
    return out;
  }

  /// Computes the Fast Fourier Transform of :literal:`inout` in place.
  template <typename BlockT>
  Matrix<O,BlockT>
  operator()(Matrix<O,BlockT> inout) VSIP_NOTHROW
  {
    typename base::Scope scope(*this);
    assert(extent(inout) == extent(this->input_size()));
    if (is_global_map<typename BlockT::map_type>::value &&
	inout.block().map().num_subblocks(axis) != 1)
      VSIP_IMPL_THROW(unimplemented(
	"Fftm requires dimension along FFT to not be distributed"));
    workspace_.in_place(this->backend_.get(), inout.local());
    return inout;
  }

  friend class vsip::impl::diag_detail::Diagnose_fftm;
private:
  std::auto_ptr<backend_type> backend_;
  workspace workspace_;
};

} // namespace vsip::impl

/// FFT operation type.
/// Applying an FFT object on a view performs a single Fast Fourier Transform
/// on the entire view. :literal:`Fft` supports different computations, dependent
/// on the  input element type, output element type, a specified direction or a 
/// *special dimension*, and the dimensionalities of the input and output views.
///
/// Template parameters:
///   :V: View (:literal:`const_Vector` for 1D FFTs, :literal:`const_Matrix` for
///       2D FFTs, :literal:`const_Tensor` for 3D FFTs)
///   :I: Input type
///   :O: Output type
///   :S: Special Dimension
///   :return_mechanism_type: one of :literal:`by_value` or :literal:`by_reference`
///   :N: Anticipated number of times this object will be used.
///   :H: This value indicates how the implementation should optimize its
///       computation or resource use.
///
/// +-----+-------------------------+---------+------------------------+-------------------+
/// | dim | I / O                   | S       | input size             | output size       |
/// +=====+=========================+=========+========================+===================+
/// | 1D  | complex<T> / complex<T> | fft_fwd | M                      | M                 |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | complex<T> / complex<T> | fft_inv | M                      | M                 |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | T / complex<T>          | 0       | M                      | M/2 + 1           |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | complex<T> / T          | 0       | M/2 + 1                | M                 |
/// +-----+-------------------------+---------+------------------------+-------------------+
/// | 2D  | complex<T> / complex<T> | fft_fwd | M x N                  | M x N             |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | complex<T> / complex<T> | fft_inv | M x N                  | M x N             |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | T / complex<T>          | 0       | M x N                  | (M/2 + 1) x N     |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | T / complex<T>          | 1       | M x N                  | M x (N/2 + 1)     |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | complex<T> / T          | 0       | (M/2 + 1) x N          | M x N             |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | complex<T> / T          | 1       | M x (N/2 + 1)          | M x N             |
/// +-----+-------------------------+---------+------------------------+-------------------+
/// | 3D  | complex<T> / complex<T> | fft_fwd | M x N x P              | M x N x P         |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | complex<T> / complex<T> | fft_inv | M x N x P              | M x N x P         |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | T / complex<T>          | 0       | M x N x P              | (M/2 + 1) x N x P |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | T / complex<T>          | 1       | M x N x P              | M x (N/2 + 1) x P |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | T / complex<T>          | 2       | M x N x P              | M x N x (P/2 + 1) |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | complex<T> / T          | 0       | (M/2 + 1) x N x P      | M x N x P         |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | complex<T> / T          | 1       | M x (N/2 + 1) x P      | M x N x P         |
/// |     +-------------------------+---------+------------------------+-------------------+
/// |     | complex<T> / T          | 2       | M x N x (P/2 + 1)      | M x N x P         |
/// +-----+-------------------------+---------+------------------------+-------------------+
template <template <typename, typename> class V,
	  typename I,
	  typename O,
	  int S = 0,
	  return_mechanism_type R = by_value,
	  unsigned N = 0,
	  alg_hint_type H = alg_time>
class Fft : public impl::Fft_facade<impl::Dim_of_view<V>::dim,
  I, O,
  typename vsip_csl::dispatcher::List<
    vsip_csl::dispatcher::op::fft<impl::Dim_of_view<V>::dim, I, O, S, R, N> >::type,
  S, R, N, H>
{
  static dimension_type const dim = impl::Dim_of_view<V>::dim;
  typedef vsip_csl::dispatcher::op::fft<dim, I, O, S, R, N> operation_type;
  typedef impl::Fft_facade<
    dim, I, O, 
    typename vsip_csl::dispatcher::List<operation_type>::type,
    S, R, N, H> base;
public:
  /// Create an :literal:`Fft` object.
  ///
  /// Arguments:
  ///   :dom:   The domain of the view to be operated on.
  ///   :scale: A scalar factor to be applied to the result.
  Fft(Domain<dim> const& dom, typename base::scalar_type scale)
    VSIP_THROW((std::bad_alloc)) 
    : base(dom, scale) {}
};

/// FFTM operation type.
/// Applying an FFTM object on a matrix performs one Fast Fourier Transform
/// per row or column, depending on its orientation. As :literal:`Fft`, it
/// supports different computations, dependent on the  input element type and
/// output element type.
///
/// Template parameters:
///   :I: Input type
///   :O: Output type
///   :A: Orientation: one of :literal:`row` and :literal:`col`
///   :D: Direction: one of :literal:`fft_fwd` and :literal:`fft_inv`
///   :return_mechanism_type: one of :literal:`by_value` or :literal:`by_reference`
///   :N: Anticipated number of times this object will be used.
///   :H: This value indicates how the implementation should optimize its
///       computation or resource use.
///
/// +-------------------------+-----+-----------------+---------------+---------------+
/// | I / O                   | A   | D               | input size    | output size   |
/// +=========================+=====+=================+===============+===============+
/// | complex<T> / complex<T> | 0,1 | fft_fwd,fft_inv | N x M         | N x M         |
/// +-------------------------+-----+-----------------+---------------+---------------+
/// | T / complex<T>          | 0   | fft_fwd         | N x M         | N x (M/2 + 1) |
/// +-------------------------+-----+-----------------+---------------+---------------+
/// | T / complex<T>          | 1   | fft_fwd         | N x M         | (N/2 + 1) x M |
/// +-------------------------+-----+-----------------+---------------+---------------+
/// | complex<T> / T          | 0   | fft_inv         | N x (M/2 + 1) | N x M         |
/// +-------------------------+-----+-----------------+---------------+---------------+
/// | complex<T> / T          | 1   | fft_inv         | (N/2 + 1) x M | N x M         |
/// +-------------------------+-----+-----------------+---------------+---------------+
template <typename I,
	  typename O,
	  int A = row,
	  int D = fft_fwd,
	  return_mechanism_type R = by_value,
	  unsigned N = 0,
	  alg_hint_type H = alg_time>
class Fftm : public impl::Fftm_facade<
  I, O,
  typename vsip_csl::dispatcher::List<
    vsip_csl::dispatcher::op::fftm<I, O, A, D, R, N> >::type,
  A, D, R, N, H>
{
  typedef vsip_csl::dispatcher::op::fftm<I, O, A, D, R, N> operation_type;
  typedef impl::Fftm_facade<
    I, O, 
    typename vsip_csl::dispatcher::List<operation_type>::type,
    A, D, R, N, H> base;
public:
  /// Create an :literal:`Fftm` object.
  ///
  /// Arguments:
  ///   :dom: The domain of the matrix to be operated on.
  ///   :scale: A scalar factor to be applied to the result.
  Fftm(Domain<2> const& dom, typename base::scalar_type scale)
    VSIP_THROW((std::bad_alloc))
    : base(dom, scale) {}
};

} // namespace vsip

#endif // VSIP_IMPL_FFT_HPP
