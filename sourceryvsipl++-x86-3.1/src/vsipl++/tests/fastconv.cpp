/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/fastconv.cpp
    @author  Stefan Seefeld
    @date    2007-03-13
    @brief   VSIPL++ Library: fastconv tests.
*/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/selgen.hpp>
#include <vsip/signal.hpp>
#include <vsip_csl/error_db.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#if VSIP_IMPL_CBE_SDK
#include <vsip/opt/cbe/ppu/fastconv.hpp>
#endif

using namespace vsip;
using vsip_csl::equal;
using vsip_csl::error_db;
using vsip_csl::operator<<;

length_type const multi = 16;

// A single FFT to transform weights from time- to frequency-domain.
template <typename T, typename B>
Vector<T, B> t2f(const_Vector<T, B> weights)
{
  Fft<const_Vector, T, T, fft_fwd, by_value> fft(weights.size(), 1.);
  return fft(weights);
}
  
// A single FFT to transform weights from frequency- to time-domain.
template <typename T, typename B>
Vector<T, B> f2t(const_Vector<T, B> weights)
{
  Fft<const_Vector, T, T, fft_inv, by_value> fft(weights.size(), 1./weights.size());
  return fft(weights);
}

// Multiple FFTs to transform weights from time- to frequency-domain.
template <typename T, typename B>
Matrix<T, B> t2f(const_Matrix<T, B> weights)
{
  Fftm<T, T, row, fft_fwd, by_value> fftm(view_domain(weights), 1.);
  return fftm(weights);
}

// A single FFT to transform real weights from time- to frequency-domain.
template <typename T, typename B1, typename B2>
Vector<std::complex<T>, B1> t2f_sc(const_Vector<T, B2> weights)
{
  Fft<const_Vector, T, std::complex<T>, 0, by_value> fft(weights.size(), 1.);
  return fft(weights);
}

// Multiple FFTs to transform real weights from time- to frequency-domain.
template <typename T, typename B1, typename B2>
Matrix<std::complex<T>, B1> t2f_sc(const_Matrix<T, B2> weights)
{
  Fftm<T, std::complex<T>, row, fft_fwd, by_value, 0> fftm(view_domain(weights), 1.);
  return fftm(weights);
}
  
// Separate fft, vmul, inv_fft calls.
struct separate;
// Separate fftm, vmmul, inv_fftm calls.
struct separate_multi;
// Fused fft, vmul, inv_fft calls.
struct fused;
// Fused fftm, vmmul, inv_fftm calls.
struct fused_v_multi;
// Fused fftm, matmul, inv_fftm calls.
template <int O>  // order in which weights are multiplied
struct fused_m_multi;
// Explicit Fastconv calls:
//   with fused fftm, vmmul, inv_fftm
template <bool W>   // transform weights early
struct direct_vmmul;
//   with fused fftm, mmmul, inv_fftm (matrix of coefficients)
template <bool W>   // transform weights early
struct direct_mmmul;

template <typename T, typename B> class Fast_convolution;

template <typename T>
class Fast_convolution<std::complex<T>, separate>
{
  typedef std::complex<T> value_type;
  typedef Fft<const_Vector, value_type, value_type, fft_fwd, by_reference>
    for_fft_type;
  typedef Fft<const_Vector, value_type, value_type, fft_inv, by_reference>
    inv_fft_type;

public:
  template <typename B>
  Fast_convolution(const_Vector<value_type, B> weights)
    : weights_(t2f(weights)),
      tmp_(weights.size()),
      for_fft_(Domain<1>(weights.size()), 1.),
      inv_fft_(Domain<1>(weights.size()), 1./weights.size())
  {}

  template <typename Block1, typename Block2>
  void operator()(const_Matrix<value_type, Block1> in,
                  Matrix<value_type, Block2> out)
  {
    for (size_t r = 0; r != in.size(0); ++r)
    {
      for_fft_(in.row(r), tmp_);
      tmp_ *= weights_;
      inv_fft_(tmp_, out.row(r));
    }
  }

private:
  Vector<value_type> weights_;
  Vector<value_type> tmp_;
  for_fft_type for_fft_;
  inv_fft_type inv_fft_;
};

template <typename T>
class Fast_convolution<std::complex<T>, separate_multi>
{
  typedef std::complex<T> value_type;
  typedef Fftm<value_type, value_type, row, fft_fwd, by_reference>
    for_fft_type;
  typedef Fftm<value_type, value_type, row, fft_inv, by_reference>
    inv_fft_type;

public:
  template <typename B>
  Fast_convolution(const_Vector<value_type, B> weights)
    : weights_(t2f(weights)),
      tmp_(weights.size(), weights.size()),
      for_fft_(Domain<2>(weights.size(), weights.size()), 1.),
      inv_fft_(Domain<2>(weights.size(), weights.size()), 1./weights.size())
  {}

  template <typename Block1, typename Block2>
  void operator()(const_Matrix<value_type, Block1> in,
                  Matrix<value_type, Block2> out)
  {
    for_fft_(in, tmp_);
    tmp_ = vmmul<0>(weights_, tmp_);
    inv_fft_(tmp_, out);
  }

private:
  Vector<value_type> weights_;
  Matrix<value_type> tmp_;
  for_fft_type for_fft_;
  inv_fft_type inv_fft_;
};

template <typename T>
class Fast_convolution<std::complex<T>, fused>
{
  typedef std::complex<T> value_type;
  typedef Fft<const_Vector, value_type, value_type, fft_fwd, by_value>
    for_fft_type;
  typedef Fft<const_Vector, value_type, value_type, fft_inv, by_value>
    inv_fft_type;

public:
  template <typename B>
  Fast_convolution(const_Vector<value_type, B> weights)
    : weights_(t2f(weights)),
      for_fft_(Domain<1>(weights.size()), 1.),
      inv_fft_(Domain<1>(weights.size()), 1./weights.size())
  {}

  template <typename Block1, typename Block2>
  void operator()(const_Matrix<value_type, Block1> in,
                  Matrix<value_type, Block2> out)
  {
    for (size_t r = 0; r != in.size(0); ++r)
      out.row(r) = inv_fft_(weights_ * for_fft_(in.row(r)));
  }

private:
  Vector<value_type> weights_;
  for_fft_type for_fft_;
  inv_fft_type inv_fft_;
};

template <typename T>
class Fast_convolution<std::complex<T>, fused_v_multi>
{
  typedef std::complex<T> value_type;
  typedef Fftm<value_type, value_type, row, fft_fwd, by_value>
    for_fftm_type;
  typedef Fftm<value_type, value_type, row, fft_inv, by_value>
    inv_fftm_type;

public:
  template <typename B>
  Fast_convolution(const_Vector<value_type, B> weights)
    : weights_(t2f(weights)),
      for_fftm_(Domain<2>(multi, weights.size()), 1.),
      inv_fftm_(Domain<2>(multi, weights.size()), 1./weights.size())
  {}

  template <typename Block1, typename Block2>
  void operator()(const_Matrix<value_type, Block1> in,
                  Matrix<value_type, Block2> out)
  {
    out = inv_fftm_(vmmul<0>(weights_, for_fftm_(in)));
  }

private:
  Vector<value_type> weights_;
  for_fftm_type for_fftm_;
  inv_fftm_type inv_fftm_;
};

template <typename T,
          int      O> // order: 0 == W * fft, 1 == fft * W
class Fast_convolution<std::complex<T>, fused_m_multi<O> >
{
  typedef std::complex<T> value_type;
  typedef Fftm<value_type, value_type, row, fft_fwd, by_value>
    for_fftm_type;
  typedef Fftm<value_type, value_type, row, fft_inv, by_value>
    inv_fftm_type;
  static int const order = O;

public:
  template <typename B>
  Fast_convolution(const_Matrix<value_type, B> weights)
    : weights_(t2f(weights)),
      for_fftm_(Domain<2>(weights.size(0), weights.size(1)), 1.),
      inv_fftm_(Domain<2>(weights.size(0), weights.size(1)), 1./weights.size(1))
  {}

  template <typename Block1, typename Block2>
  void operator()(const_Matrix<value_type, Block1> in,
                  Matrix<value_type, Block2> out)
  {
    if (order == 0)
      out = inv_fftm_(weights_ * for_fftm_(in));
    else
      out = inv_fftm_(for_fftm_(in) * weights_);
  }

private:
  Matrix<value_type> weights_;
  for_fftm_type for_fftm_;
  inv_fftm_type inv_fftm_;
};

template <typename T>
class Fast_convolution<T, fused>
{
  typedef Fft<const_Vector, T, std::complex<T>, 0, by_value>
    for_fft_type;
  typedef Fft<const_Vector, std::complex<T>, T, 0, by_value>
    inv_fft_type;

public:
  template <typename B>
  Fast_convolution(const_Vector<T, B> weights)
    : weights_(t2f_sc<T, Dense<1, std::complex<T>, tuple<0, 1, 2> >, B>(weights)),
      for_fft_(Domain<1>(weights.size()), 1.),
      inv_fft_(Domain<1>(weights.size()), 1./weights.size())
  {}

  template <typename Block1, typename Block2>
  void operator()(const_Matrix<T, Block1> in,
                  Matrix<T, Block2> out)
  {
    for (size_t r = 0; r != in.size(0); ++r)
      out.row(r) = inv_fft_(weights_ * for_fft_(in.row(r)));
  }

private:
  Vector<std::complex<T> > weights_;
  for_fft_type for_fft_;
  inv_fft_type inv_fft_;
};

template <typename T>
class Fast_convolution<T, fused_v_multi>
{
  typedef Fftm<T, std::complex<T>, row, fft_fwd, by_value, 0>
    for_fftm_type;
  typedef Fftm<std::complex<T>, T, row, fft_inv, by_value, 0>
    inv_fftm_type;

public:
  template <typename B>
  Fast_convolution(const_Vector<T, B> weights)
    : weights_(t2f_sc<T, Dense<1, std::complex<T>, tuple<0, 1, 2> >, B>(weights)),
      for_fftm_(Domain<2>(multi, weights.size()), 1.),
      inv_fftm_(Domain<2>(multi, weights.size()), 1./weights.size())
  {}

  template <typename Block1, typename Block2>
  void operator()(const_Matrix<T, Block1> in,
                  Matrix<T, Block2> out)
  {
    out = inv_fftm_(vmmul<0>(weights_, for_fftm_(in)));
  }

private:
  Vector<std::complex<T> > weights_;
  for_fftm_type for_fftm_;
  inv_fftm_type inv_fftm_;
};

template <typename T,
          int      O> // order: 0 == W * fft, 1 == fft * W
class Fast_convolution<T, fused_m_multi<O> >
{
  typedef Fftm<T, std::complex<T>, row, fft_fwd, by_value, 0>
    for_fftm_type;
  typedef Fftm<std::complex<T>, T, row, fft_inv, by_value, 0>
    inv_fftm_type;
  static int const order = O;

public:
  template <typename B>
  Fast_convolution(const_Matrix<T, B> weights)
    : weights_(t2f_sc<T, Dense<2, std::complex<T>, tuple<0, 1, 2> >, B>(weights)),
      for_fftm_(Domain<2>(weights.size(0), weights.size(1)), 1.),
      inv_fftm_(Domain<2>(weights.size(0), weights.size(1)), 1./weights.size(1))
  {}

  template <typename Block1, typename Block2>
  void operator()(const_Matrix<T, Block1> in,
                  Matrix<T, Block2> out)
  {
    if (order == 0)
      out = inv_fftm_(weights_ * for_fftm_(in));
    else
      out = inv_fftm_(for_fftm_(in) * weights_);
  }

private:
  Matrix<std::complex<T> > weights_;
  for_fftm_type for_fftm_;
  inv_fftm_type inv_fftm_;
};

#if VSIP_IMPL_CBE_SDK
// Both of the direct methods perform multiple convolutions.
// In the second case, the weights are unique for each row as well, so
// they are passed as a matrix rather than a vector.

template <typename T,
          bool     W>  // pre-transform weights 
class Fast_convolution<std::complex<T>, direct_vmmul<W> >
{
  typedef std::complex<T> value_type;
public:
  template <typename B>
  Fast_convolution(Vector<value_type, B> weights)
  // Note the third parameter indicates the opposite of W, i.e. whether
  // or not the Fastconv object needs to do the transform.
    : fastconv_((W ? t2f(weights) : weights), weights.size(0), !W)
  {}

  template <typename Block1, typename Block2>
  void operator()(const_Vector<value_type, Block1> in,
                  Vector<value_type, Block2> out)
  {
    fastconv_(in, out);
  }

  template <typename Block1, typename Block2>
  void operator()(const_Matrix<value_type, Block1> in,
                  Matrix<value_type, Block2> out)
  {
    fastconv_(in, out);
  }

private:
  vsip::impl::cbe::Fastconv<1, value_type, vsip::impl::dense_complex_format> fastconv_;
};


template <typename T,
          bool     W>  // pre-transform weights 
class Fast_convolution<std::complex<T>, direct_mmmul<W> >
{
  typedef std::complex<T> value_type;
public:
  template <typename B>
  Fast_convolution(Matrix<value_type, B> weights)
    : fastconv_((W ? t2f(weights) : weights), weights.size(1), !W)
  {}

  template <typename Block1, typename Block2>
  void operator()(const_Matrix<value_type, Block1> in,
                  Matrix<value_type, Block2> out)
  {
    fastconv_(in, out);
  }

private:
  vsip::impl::cbe::Fastconv<2, value_type, vsip::impl::dense_complex_format> fastconv_;
};
#endif

template <typename O, typename B, typename T>
void test_shift(Domain<1> const &dom, length_type shift, T scale)
{
  test_assert(dom.size() > shift);
  // Construct a shift kernel.
  Vector<T> weights(dom.size(), T(0.));
  weights.put(shift, scale);
  Fast_convolution<T, B> fconv(weights);
  // This logic assumes T is a complex type.
  // Refine once we support real-valued fastconv.
  Matrix<T, Dense<2, T, O> > input(multi, dom.size());
  for (size_t r = 0; r != multi; ++r)
    input.row(r) = ramp(0., 1., dom.size());
  Matrix<T, Dense<2, T, O> > output(multi, dom.size());
  fconv(input, output);
  double error = error_db
    (scale * input(Domain<2>(multi, (Domain<1>(0, 1, dom.size() - shift)))),
     output(Domain<2>(multi, (Domain<1>(shift, 1, dom.size() - shift)))));
  if (error >= -100)
  {
    std::cout << "input" << input << std::endl;
    std::cout << "output" << output << std::endl;
  }
  test_assert(error < -100);
}


template <typename O, typename B, typename T>
void test_shift_v(Domain<1> const &dom, length_type shift, T scale)
{
  test_assert(dom.size() > shift);
  // Construct a shift kernel.
  Vector<T> weights(dom.size(), T(0.));
  weights.put(shift, scale);
  Fast_convolution<T, B> fconv(weights);
  // This logic assumes T is a complex type.
  // Refine once we support real-valued fastconv.
  Matrix<T, Dense<2, T, O> > input(dom.size(), dom.size());
  for (size_t r = 0; r != dom.size(); ++r)
    input.row(r) = ramp(0., 1., dom.size());
  Matrix<T, Dense<2, T, O> > output(dom.size(), dom.size());
  for (size_t r = 0; r != dom.size(); ++r)  
    fconv(input.row(r), output.row(r));
  double error = error_db
    (scale * input(Domain<2>(dom.size(), (Domain<1>(0, 1, dom.size() - shift)))),
     output(Domain<2>(dom.size(), (Domain<1>(shift, 1, dom.size() - shift)))));
  if (error >= -100)
  {
    std::cout << "input" << input << std::endl;
    std::cout << "output" << output << std::endl;
  }
  test_assert(error < -100);
}


template <typename O, typename B, typename T>
void test_shift_m(Domain<1> const &dom, length_type shift, T scale)
{
  test_assert(dom.size() > shift);
  // Construct a shift kernel.  Make each row unique by adjusting the 
  // scale factor a small amount on each row.
  Matrix<T> weights(dom.size(), dom.size(), T(0.));
  float ds = 1 / float(dom.size());
  for (index_type i = 0; i < dom.size(); ++i)
    weights.put(i, shift, scale * (1 + i * ds ));
  Fast_convolution<T, B> fconv(weights);
  // This logic assumes T is a complex type.
  // Refine once we support real-valued fastconv.
  Matrix<T, Dense<2, T, O> > input(dom.size(), dom.size());
  for (size_t r = 0; r != dom.size(); ++r)
    input.row(r) = ramp(0., 1., dom.size());
  Matrix<T, Dense<2, T, O> > output(dom.size(), dom.size());
  fconv(input, output);
  Matrix<T> reference(input);
  for (index_type i = 0; i < dom.size(); ++i)
    reference.row(i) *= (scale * (1 + i * ds));
  double error = error_db(
    reference(Domain<2>(dom.size(), (Domain<1>(0, 1, dom.size() - shift)))),
    output(Domain<2>(dom.size(), (Domain<1>(shift, 1, dom.size() - shift)))));
  if (error >= -100)
  {
    std::cout << "input" << input << std::endl;
    std::cout << "output" << output << std::endl;
  }
  test_assert(error < -100);
}


int main(int argc, char **argv)
{
  vsipl init(argc, argv);

  // Test...

  // ... using a vector of coefficients,
  //    - inidividual operations, one row at a time
  test_shift<row2_type, separate>(16, 2, std::complex<float>(2.));
  //    - inidividual operations, multiple rows at a time
  test_shift<row2_type, separate_multi>(16, 2, std::complex<float>(2.));
  //    - combined operations, one row at a time
  test_shift<row2_type, fused>(16, 2, std::complex<float>(2.));
  test_shift<row2_type, fused>(16, 2, float(2.));
  //    - combined operations, multiple rows at a time
  test_shift<row2_type, fused_v_multi>(64, 2, std::complex<float>(2.));
  test_shift<row2_type, fused_v_multi>(64, 2, float(2.));

  // ... using a matrix of coefficients,
  //    - multiple rows at a time, coeffs * fft() order
  test_shift_m<row2_type, fused_m_multi<0> >(64, 2, std::complex<float>(2.));
  test_shift_m<row2_type, fused_m_multi<0> >(64, 2, float(2.));
  //    - multiple rows at a time, fft() * coeffs order
  test_shift_m<row2_type, fused_m_multi<1> >(64, 2, std::complex<float>(2.));
  test_shift_m<row2_type, fused_m_multi<1> >(64, 2, float(2.));

#if VSIP_IMPL_CBE_SDK
  test_shift<row2_type, direct_vmmul<false> >(64, 2, std::complex<float>(0.5));
  test_shift<row2_type, direct_vmmul<true> >(64, 2, std::complex<float>(0.5));
  test_shift_v<row2_type, direct_vmmul<false> >(64, 2, std::complex<float>(0.5));
  test_shift_v<row2_type, direct_vmmul<true> >(64, 2, std::complex<float>(0.5));
  test_shift_m<row2_type, direct_mmmul<false> >(64, 2, std::complex<float>(0.5));
  test_shift_m<row2_type, direct_mmmul<true> >(64, 2, std::complex<float>(0.5));
#endif
}
