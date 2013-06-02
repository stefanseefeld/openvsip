//
// Copyright (c) 2006, 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_fft_hpp_
#define vsip_impl_fft_hpp_

#include <ovxx/signal/fft.hpp>

namespace vsip
{
using ovxx::fft_fwd;
using ovxx::fft_inv;

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
class Fft : public ovxx::signal::Fft<ovxx::dim_of_view<V>::dim,
  I, O,
  typename ovxx::dispatcher::List<
    ovxx::dispatcher::op::fft<ovxx::dim_of_view<V>::dim, I, O, S, R, N> >::type,
  S, R, N, H>
{
  static dimension_type const dim = ovxx::dim_of_view<V>::dim;
  typedef ovxx::dispatcher::op::fft<dim, I, O, S, R, N> operation_type;
  typedef ovxx::signal::Fft<
    dim, I, O, 
    typename ovxx::dispatcher::List<operation_type>::type,
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
class Fftm : public ovxx::signal::Fftm<
  I, O,
  typename ovxx::dispatcher::List<
    ovxx::dispatcher::op::fftm<I, O, A, D, R, N> >::type,
  A, D, R, N, H>
{
  typedef ovxx::dispatcher::op::fftm<I, O, A, D, R, N> operation_type;
  typedef ovxx::signal::Fftm<
    I, O, 
    typename ovxx::dispatcher::List<operation_type>::type,
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

#endif
