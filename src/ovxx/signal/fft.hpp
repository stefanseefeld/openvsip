//
// Copyright (c) 2006, 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_signal_fft_hpp_
#define ovxx_signal_fft_hpp_

#include <vsip/impl/signal/types.hpp>
#include <ovxx/signal/fft/backend.hpp>
#include <ovxx/signal/fft/util.hpp>
#include <ovxx/signal/fft/workspace.hpp>
#include <ovxx/dispatch.hpp>
#if OVXX_DFT_FFT
# include <ovxx/signal/fft/dft.hpp>
#endif
#if OVXX_NO_FFT
# include <ovxx/signal/fft/no_fft.hpp>
#endif
#include <cstring>

namespace ovxx
{
namespace dispatcher
{
template <dimension_type D,
	  typename I,
	  typename O,
	  int S,
	  return_mechanism_type R,
	  unsigned N>
struct List<op::fft<D, I, O, S, R, N> >
{
  typedef make_type_list<be::user,
			 be::cuda,
			 be::fftw,
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
  typedef make_type_list<be::user,
			 be::fftw,
			 be::cvsip,
			 be::generic,
			 be::no_fft>::type type;
};

} // namespace ovxx::dispatcher

namespace signal
{
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
class Interface
{
public:
  static dimension_type const dim = D;
  typedef typename scalar_of<I>::type scalar_type;

  Interface(Domain<D> const &dom, scalar_type scale, 
	    bool is_fftm, int dir, return_mechanism_type rm)
    : input_size_(io_size<D, I, O, A>::size(dom)),
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

} // namespace ovxx::signal::fft

template <dimension_type D,                 //< Dimension
	  typename I,                       //< Input type
	  typename O,                       //< Output type
	  typename L,                       //< Dispatcher list
	  int S = 0,                        //< Special dimension
	  return_mechanism_type = by_value, //< Return mechanism
	  unsigned N = 0,                   //< Number of times
	  alg_hint_type = alg_time>         //< algorithm Hint
class Fft;

template <dimension_type D,
	  typename I,
	  typename O,
	  typename L,
	  int S,
	  unsigned N,
	  alg_hint_type H>
class Fft<D, I, O, L, S, by_value, N, H>
  : public fft::Interface<D, I, O,
			  fft::axis<I, O, S>::value,
			  fft::exponent<I, O, S>::value>
{
public:
  static int const axis = fft::axis<I, O, S>::value;
  static int const exponent = fft::exponent<I, O, S>::value;
  typedef fft::Interface<D, I, O, axis, exponent> base;
  typedef fft::fft_backend<D, I, O, S> backend_type;

  typedef fft::workspace<D, I, O> workspace;
  typedef dispatcher::Dispatcher<
    dispatcher::op::fft<D, I, O, S, by_value, N>,
    std::auto_ptr<backend_type>(Domain<D> const &, typename base::scalar_type), L>
    dispatcher_type;

  Fft(Domain<D> const& dom, typename base::scalar_type scale)
    VSIP_THROW((std::bad_alloc))
    : base(dom, scale, false, S, by_value),
      backend_(dispatcher_type::dispatch(dom, scale)),
      workspace_(backend_.get(), this->input_size(), this->output_size(), scale)
  {}

#ifdef VSIP_IMPL_REF_IMPL
  /// Returns the Fast Fourier Transform of :literal:`in`.
  template <typename ViewT>
  typename fft::result<O, typename ViewT::block_type>::view_type
  operator()(ViewT in) VSIP_THROW((std::bad_alloc))
  {
    OVXX_PRECONDITION(extent(in) == extent(this->input_size()));
    typedef fft::result<O, typename ViewT::block_type> traits;
    typename traits::view_type out(traits::create(this->output_size(),
                                                 in.block().map()));
    workspace_.out_of_place(*this->backend_, in.block(), out.block());
    return out;
  }
#else
  /// Returns the Fast Fourier Transform of :literal:`in`.
  template <typename ViewT>
  typename fft::Result_rbo<D, I, O, ViewT, workspace, S>::view_type
  operator()(ViewT in) VSIP_THROW((std::bad_alloc))
  {
    OVXX_PRECONDITION(extent(in) == extent(this->input_size()));
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
class Fft<D, I, O, L, S, vsip::by_reference, N, H>
  : public fft::Interface<D, I, O,
			  fft::axis<I, O, S>::value,
			  fft::exponent<I, O, S>::value>
{
public:
  static int const axis = fft::axis<I, O, S>::value;
  static int const exponent = fft::exponent<I, O, S>::value;
  typedef fft::Interface<D, I, O, axis, exponent> base;
  typedef fft::fft_backend<D, I, O, S> backend_type;
  typedef fft::workspace<D, I, O> workspace;
  typedef dispatcher::Dispatcher<
    dispatcher::op::fft<D, I, O, S, by_reference, N>,
    std::auto_ptr<backend_type>(Domain<D> const &, typename base::scalar_type), L>
    dispatcher_type;

  Fft(Domain<D> const& dom, typename base::scalar_type scale)
    VSIP_THROW((std::bad_alloc))
    : base(dom, scale, false, S, by_reference),
      backend_(dispatcher_type::dispatch(dom, scale)),
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
    OVXX_CT_ASSERT((View0<I,Block0>::dim == View1<O,Block1>::dim));
    OVXX_PRECONDITION(extent(in) == extent(this->input_size()));
    OVXX_PRECONDITION(extent(out) == extent(this->output_size()));
    workspace_.out_of_place(*this->backend_, in.block(), out.block());
    return out;
  }

  /// Computes the Fast Fourier Transform of :literal:`inout` in place.
  template <typename BlockT, template <typename, typename> class View>
  View<I,BlockT>
  operator()(View<I,BlockT> inout) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(extent(inout) == extent(this->input_size()));
    OVXX_PRECONDITION(extent(inout) == extent(this->output_size()));
    workspace_.in_place(*this->backend_, inout.block());
    return inout;
  }

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
class Fftm;

template <typename I,
	  typename O,
	  typename L,
	  int A,
	  int D,
	  unsigned N,
	  alg_hint_type H>
class Fftm<I, O, L, A, D, by_value, N, H>
  : public fft::Interface<2, I, O, 1 - A, D == fft_fwd ? -1 : 1>
{
  static int const axis = 1 - A;
  static int const exponent = D == fft_fwd ? -1 : 1;
  typedef fft::Interface<2, I, O, axis, exponent> base;
  typedef typename fft::fftm_backend<I, O, A, D> backend_type;
  typedef fft::workspace<2, I, O> workspace;
  typedef dispatcher::Dispatcher<
    dispatcher::op::fftm<I, O, A, D, by_value, N>,
    std::auto_ptr<backend_type>(Domain<2> const &, typename base::scalar_type), L>
    dispatcher_type;
public:
  Fftm(Domain<2> const& dom, typename base::scalar_type scale)
    VSIP_THROW((std::bad_alloc))
    : base(dom, scale, true, D, by_value),
      backend_(dispatcher_type::dispatch(dom, scale)),
      workspace_(backend_.get(), this->input_size(), this->output_size(), scale)
  {}

#ifdef VSIP_IMPL_REF_IMPL
  /// Returns the Fast Fourier Transform of :literal:`in`.
  template <typename BlockT>
  typename fft::result<O,BlockT>::view_type
  operator()(const_Matrix<I,BlockT> in)
     VSIP_THROW((std::bad_alloc))
  {
    typedef fft::result<O,BlockT> traits;
    typename traits::view_type out(traits::create(this->output_size(),
                                                 in.block().map()));
    OVXX_PRECONDITION(extent(in) == extent(this->input_size()));
    if (parallel::is_global_map<typename BlockT::map_type>::value &&
        in.block().map().num_subblocks(axis) != 1)
      OVXX_DO_THROW(unimplemented(
        "Fftm requires dimension along FFT to not be distributed"));
    workspace_.out_of_place(*this->backend_, in.local().block(), out.local().block());
    return out;
  }
#else
  /// Returns the Fast Fourier Transform of :literal:`in`.
  template <typename BlockT>  
    typename fft::Result_fftm_rbo<I, O, BlockT, workspace, A, D>::view_type
  operator()(const_Matrix<I,BlockT> in)
    VSIP_THROW((std::bad_alloc))
  {
    OVXX_PRECONDITION(extent(in) == extent(this->input_size()));

    /* TODO: Return_blocks don't have a valid map() yet
    if (is_global_map<typename BlockT::map_type>::value &&
	in.block().map().num_subblocks(A) != 1)
      OVXX_DO_THROW(unimplemented(
	"Fftm requires dimension along FFT to not be distributed"));
    */

    typedef fft::Result_fftm_rbo<I, O, BlockT, workspace, A, D> traits;
    typedef typename traits::functor_type functor_type;
    typedef typename traits::block_type   block_type;
    typedef typename traits::view_type    view_type;

    functor_type rf(in, this->output_size(), *this->backend_,
		    workspace_);
    block_type block(rf);
    return view_type(block);
 }
#endif

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
class Fftm<I, O, L, A, D, vsip::by_reference, N, H>
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
  typedef typename fft::fftm_backend<I, O, A, D> backend_type;
  typedef fft::workspace<2, I, O> workspace;
  typedef dispatcher::Dispatcher<
    dispatcher::op::fftm<I, O, A, D, by_reference, N>,
    std::auto_ptr<backend_type>(Domain<2> const &, typename base::scalar_type), L>
    dispatcher_type;
public:
  Fftm(Domain<2> const& dom, typename base::scalar_type scale)
    VSIP_THROW((std::bad_alloc))
    : base(dom, scale, true, D, by_reference),
      backend_(dispatcher_type::dispatch(dom, scale)),
      workspace_(backend_.get(), this->input_size(), this->output_size(), scale)
  {}

  /// Computes the Fast Fourier Transform of :literal:`in` and stores the result
  /// in :literal:`out`.
  template <typename Block0, typename Block1>
  Matrix<O,Block1>
  operator()(const_Matrix<I,Block0> in, Matrix<O,Block1> out)
    VSIP_NOTHROW
  {
    OVXX_PRECONDITION(extent(in)  == extent(this->input_size()));
    OVXX_PRECONDITION(extent(out) == extent(this->output_size()));
    if (parallel::is_global_map<typename Block0::map_type>::value ||
	parallel::is_global_map<typename Block1::map_type>::value)
    {
      if (in.block().map().num_subblocks(axis) != 1 ||
	  out.block().map().num_subblocks(axis) != 1)
	OVXX_DO_THROW(unimplemented(
	  "Fftm requires dimension along FFT to not be distributed"));
      if (global_domain(in) != global_domain(out))
	OVXX_DO_THROW(unimplemented(
	  "Fftm requires input and output to have same mapping"));
    }
    workspace_.out_of_place(*this->backend_, in.local().block(), out.local().block());
    return out;
  }

  /// Computes the Fast Fourier Transform of :literal:`inout` in place.
  template <typename BlockT>
  Matrix<O,BlockT>
  operator()(Matrix<O,BlockT> inout) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(extent(inout) == extent(this->input_size()));
    if (parallel::is_global_map<typename BlockT::map_type>::value &&
	inout.block().map().num_subblocks(axis) != 1)
      OVXX_DO_THROW(unimplemented(
	"Fftm requires dimension along FFT to not be distributed"));
    workspace_.in_place(*this->backend_, inout.local().block());
    return inout;
  }

private:
  std::auto_ptr<backend_type> backend_;
  workspace workspace_;
};

} // namespace ovxx::signal
} // namespace ovxx

#endif
