//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/selgen.hpp>
#include <vsip/math.hpp>
#include <ovxx/signal/fft/dft.hpp>
#include <ovxx/signal/fft/no_fft.hpp>
#include <ovxx/dispatch.hpp>
#include <vsip/impl/signal/fft.hpp>
#include <test.hpp>
#include <map>

// Define FFT_BE_TESTS in order for backends to
// be tested individually.
// #ifndef FFT_BE_TESTS
// # define FFT_COMPOSITE
// #endif

using namespace ovxx;

// In the default mode test the backends as they are
// used by end-users.
struct composite
{
  //  typedef dispatcher::List<>::type list;
};
// Construct one type-list per backend.
// If the backend is not available, the evaluator will
// pick up the dummy no_fft backend, which will lead to
// runtime-errors (i.e. incorrect values), but the application
// will continue to compile.
struct fftw
{
  typedef 
  dispatcher::make_type_list<
#if VSIP_IMPL_FFTW3
    dispatcher::be::fftw3,
#endif
    dispatcher::be::no_fft>::type
  list;
};
struct sal
{
  typedef 
  dispatcher::make_type_list<
#if VSIP_IMPL_SAL_FFT
    dispatcher::be::mercury_sal,
#endif
    dispatcher::be::no_fft>::type
  list;
};
struct ipp
{
  typedef 
  dispatcher::make_type_list<
#if VSIP_IMPL_IPP_FFT
    dispatcher::be::intel_ipp,
#endif
    dispatcher::be::no_fft>::type
  list;
};
struct cvsip
{
  typedef 
  dispatcher::make_type_list<
#if VSIP_IMPL_CVSIP_FFT
    dispatcher::be::cvsip,
#endif
    dispatcher::be::no_fft>::type
  list;
};
struct cbe
{
  typedef 
  dispatcher::make_type_list<
#if VSIP_IMPL_CBE_SDK_FFT
    dispatcher::be::cbe_sdk,
#endif
    dispatcher::be::no_fft>::type
  list;
};
struct dft
{
  typedef 
  dispatcher::make_type_list<dispatcher::be::generic,
			     dispatcher::be::no_fft>::type
  list;
};

storage_format_type const inter = vsip::interleaved_complex;
storage_format_type const split = vsip::split_complex;

template <typename T, storage_format_type F, int D, int A = 0, typename OrderT = row1_type>
struct cfft_type
{
  typedef std::complex<T> I;
  typedef std::complex<T> O;
  static storage_format_type const format = F;
  static storage_format_type const i_format = F;
  static storage_format_type const o_format = F;
  typedef OrderT order_type;
  static int const axis = A;
  static int const direction = D;
  static int const s = direction;
  template <dimension_type Dim>
  static Domain<Dim> in_dom(Domain<Dim> const &dom) { return dom;}
  template <dimension_type Dim>
  static Domain<Dim> out_dom(Domain<Dim> const &dom) { return dom;}
};

template <typename T, storage_format_type F, int D, int A = 0, typename OrderT = row1_type> 
struct rfft_type;
template <typename T, storage_format_type F, int A, typename OrderT>
struct rfft_type<T, F, fft_fwd, A, OrderT>
{
  typedef T I;
  typedef std::complex<T> O;
  static storage_format_type const format = F;
  static storage_format_type const i_format = inter;
  static storage_format_type const o_format = F;
  typedef OrderT order_type;
  static int const axis = A;
  static int const direction = fft_fwd;
  static int const s = A;
  template <dimension_type D>
  static Domain<D> in_dom(Domain<D> const &dom) { return dom;}
  template <dimension_type D>
  static Domain<D> out_dom(Domain<D> const &dom) 
  {
    Domain<D> retn(dom);
    Domain<1> &mod = retn.impl_at(axis);
    mod = Domain<1>(mod.first(), mod.stride(), mod.size() / 2 + 1); 
    return retn;
  }
};
template <typename T, storage_format_type F, int A, typename OrderT>
struct rfft_type<T, F, fft_inv, A, OrderT>
{
  typedef std::complex<T> I;
  typedef T O;
  static storage_format_type const format = F;
  static storage_format_type const i_format = F;
  static storage_format_type const o_format = inter;
  typedef OrderT order_type;
  static int const axis = A;
  static int const direction = fft_inv;
  static int const s = A;
  template <dimension_type D>
  static Domain<D> in_dom(Domain<D> const &dom) 
  {
    Domain<D> retn(dom);
    Domain<1> &mod = retn.impl_at(axis);
    mod = Domain<1>(mod.first(), mod.stride(), mod.size() / 2 + 1); 
    return retn;
  }
  template <dimension_type D>
  static Domain<D> out_dom(Domain<D> const &dom) { return dom;}
};

template <typename T>
const_Vector<T, expr::Generator<1, vsip::impl::Ramp_generator<T> > const>
ramp(Domain<1> const &dom) 
{ return vsip::ramp(T(0.), T(1.), dom.length() * dom.stride());}

template <typename T>
Matrix<T>
ramp(Domain<2> const &dom) 
{
  length_type rows = dom[0].length() * dom[0].stride();
  length_type cols = dom[1].length() * dom[1].stride();
  Matrix<T> m(rows, cols);
  for (size_t r = 0; r != rows; ++r)
    m.row(r) = ramp(T(r), T(1.), m.size(1));
  return m;
}

template <typename T>
Tensor<T>
ramp(Domain<3> const &dom) 
{
  length_type x_length = dom[0].length() * dom[0].stride();
  length_type y_length = dom[1].length() * dom[1].stride();
  length_type z_length = dom[2].length() * dom[2].stride();
  Tensor<T> t(x_length, y_length, z_length);
  for (size_t x = 0; x != t.size(0); ++x)
    for (size_t y = 0; y != t.size(1); ++y)
      t(x, y, whole_domain) = ramp(T(x), T(y), t.size(2));
  return t;
}

template <typename T>
Vector<T>
empty(Domain<1> const &dom) 
{ return Vector<T>(dom.length() * dom.stride(), T(0.));}

template <typename T>
Matrix<T>
empty(Domain<2> const &dom) 
{
  length_type rows = dom[0].length() * dom[0].stride();
  length_type cols = dom[1].length() * dom[1].stride();
  return Matrix<T>(rows, cols, T(0.));
}

template <typename T>
Tensor<T>
empty(Domain<3> const &dom) 
{
  length_type x_length = dom[0].length() * dom[0].stride();
  length_type y_length = dom[1].length() * dom[1].stride();
  length_type z_length = dom[2].length() * dom[2].stride();
  return Tensor<T>(x_length, y_length, z_length, T(0.));
}

template <typename T, dimension_type D> 
struct input_creator
{
  typedef typename T::I I;
  static typename vsip::impl::view_of<Dense<D, I> >::type
  create(Domain<D> const &dom) { return ramp<I>(dom);}
};

// Real inverse FFT
template <typename T, storage_format_type F, int A, dimension_type D> 
struct input_creator<rfft_type<T, F, fft_inv, A>, D>
{
  typedef typename rfft_type<T, F, fft_inv, A>::I I;
  static typename vsip::impl::view_of<Dense<D, I> >::type
  create(Domain<D> const &dom) 
    { return ramp<I>(rfft_type<T, F, fft_inv, A>::in_dom(dom));}
};

// Real inverse 2D FFT.
template <typename T, storage_format_type F, int A>
struct input_creator<rfft_type<T, F, fft_inv, A>, 2>
{
  typedef typename rfft_type<T, F, fft_inv, A>::I I;
  static Matrix<I> 
  create(Domain<2> const &dom) 
  {
    length_type rows  = dom[0].size();
    length_type cols  = dom[1].size();
    length_type rows2 = rows/2+1;
    length_type cols2 = cols/2+1;

    Matrix<I> input = ramp<I>(rfft_type<T, F, fft_inv, A>::in_dom(dom));
    if (rfft_type<T, F, fft_inv, A>::axis == 0)
    {
      // Necessary symmetry:
      for (index_type cc=cols2; cc<cols; ++cc)
      {
	input(0, cc) = conj(input.get(0, cols-cc));
	input(rows2-1, cc) = conj(input.get(rows2-1, cols-cc));
      }
    }
    else
    {
      // Necessary symmetry:
      for (index_type rr=rows2; rr<rows; ++rr)
      {
	input(rr, 0) = conj(input.get(rows-rr, 0));
	input(rr, cols2-1) = conj(input.get(rows-rr, cols2-1));
      }
    }
    return input;
  }
};

// This is a composite test that runs a large variety of unit tests. Since it
// is hard and cumbersome to write those tests out in sequence, they are 
// grouped by types / categories.
// However, the different backends that are covered here only support subsets
// of this set, and these subsets don't coincide with the way they are
// structured here.
// Therefor, we register expected failures, which an individual test
// will check to see if a given failure was actually expected or not.
template <typename T, storage_format_type F, dimension_type D, return_mechanism_type R,
          typename I, typename O, unsigned int S>
struct XFail { static bool const value = false;};

#define xfail(T, F, D, R, I, O, S) \
template <> struct XFail<T, F, D, R, I, O, S> { static bool const value = true;};

// IPP doesn't support 2D real FFTs
template <storage_format_type F, return_mechanism_type R, typename O, unsigned int S>
struct XFail<ipp, F, 2, R, float, O, S> { static bool const value = true;};
template <storage_format_type F, return_mechanism_type R, typename I, unsigned int S>
struct XFail<ipp, F, 2, R, I, float, S> { static bool const value = true;};
// IPP doesn't support 2D double FFTs
template <storage_format_type F, return_mechanism_type R, typename O, unsigned int S>
struct XFail<ipp, F, 2, R, double, O, S> { static bool const value = true;};
template <storage_format_type F, return_mechanism_type R, typename O, unsigned int S>
struct XFail<ipp, F, 2, R, std::complex<double>, O, S> { static bool const value = true;};

// CBE doesn't support double FFTs
template <storage_format_type F, return_mechanism_type R, typename O, unsigned int S>
struct XFail<cbe, F, 1, R, double, O, S> { static bool const value = true;};
template <storage_format_type F, return_mechanism_type R, typename O, unsigned int S>
struct XFail<cbe, F, 1, R, complex<double>, O, S> { static bool const value = true;};

// Expected failures for Fftm
template <typename ImplTag,	// ImplTag
	  storage_format_type C,	// complex format (Cmplx_{inter,spilt>_fmt
	  return_mechanism_type R,
          typename I,		// Input type
	  typename O,		// output type
	  unsigned int fft_dir> // Fft direction
struct XFailM
{ static bool const value = false;};

// CBE doesn't support double FFTMs
template <storage_format_type F, return_mechanism_type R, typename O, unsigned int S>
struct XFailM<cbe, F, R, double, O, S> { static bool const value = true;};
template <storage_format_type F, return_mechanism_type R, typename O, unsigned int S>
struct XFailM<cbe, F, R, complex<double>, O, S> { static bool const value = true;};

bool has_errors = false;

template <typename T, typename B, dimension_type D>
void fft_by_ref(Domain<D> const &dom)
{
  typedef typename T::I I;
  typedef typename T::O O;
  typedef typename T::order_type order_type;
  typedef typename vsip::Layout<D, order_type, vsip::dense, T::i_format> i_layout_type;
  typedef typename vsip::Layout<D, order_type, vsip::dense, T::o_format> o_layout_type;
  return_mechanism_type const r = by_reference;

  typedef vsip::impl::Strided<D, I, i_layout_type> Iblock;
  typedef vsip::impl::Strided<D, O, o_layout_type> Oblock;
  typedef typename vsip::impl::view_of<Iblock>::type Iview;
  typedef typename vsip::impl::view_of<Oblock>::type Oview;

  Domain<D> in_dom = T::in_dom(dom);
  Domain<D> out_dom = T::out_dom(dom);

  // Set up some input data.
  Iview input = input_creator<T, D>::create(dom);

  // Preserve it to validate that input isn't destroyed during the FFT.
  Iview orig = empty<I>(in_dom);
  orig = input;
  // Set up subview to be used as input (helpful for testing non-unit-strides).
  typename Iview::subview_type sub_input = input(in_dom);

  // Set up the output data...
  Oview output = empty<O>(out_dom);
  // ...with possibly non-unit-stride.
  typename Oview::subview_type sub_output = output(out_dom);
  // Create the FFT object...
  typename ovxx::signal::Fft<D, I, O, typename B::list, T::s, r> fft(dom, 42.);
  // ...and call it.
  fft(sub_input, sub_output);

  // Test that input is preserved.
  test_assert(error_db(input, orig) < -200);

  Oview ref = empty<O>(out_dom);
  typename Oview::subview_type sub_ref = ref(out_dom);
  typename ovxx::signal::Fft<D, I, O, dft::list, T::s, r> ref_fft(dom, 42.);
  ref_fft(sub_input, sub_ref);

  if (error_db(output, ref) > -100)
  {
    if (XFail<B, T::format, D, r, I, O, T::s>::value)
      std::cout << "expected error." << std::endl;
    else
    {
      has_errors = true;
      std::cout << "unexpected error." << std::endl;
//     std::cout << "out " << sub_output << std::endl;
//     std::cout << "ref  " << sub_ref << std::endl;
    }
  }
  else std::cout << "ok." << std::endl;
}

template <typename T, storage_format_type F, int E, typename B, dimension_type D>
void fft_in_place(Domain<D> const &dom)
{
  int const s = E == -1 ? fft_fwd : fft_inv;
  return_mechanism_type const r = by_reference;
  typedef typename vsip::Layout<D, row1_type, vsip::dense, F> layout_type;
  typedef std::complex<T> CT;

  typedef Strided<D, CT, layout_type> block_type;
  typedef typename view_of<block_type>::type View;

  View data = ramp<T>(dom);
  View ref = ramp<T>(dom);

  typename View::subview_type sub_data = data(dom);

  typename ovxx::signal::Fft<D, CT, CT, typename B::list, s, r> fft(dom, 42.);
  fft(sub_data);

  typename View::subview_type sub_ref = ref(dom);
  typename ovxx::signal::Fft<D, CT, CT, dft::list, s, r> ref_fft(dom, 42.);
  ref_fft(sub_ref);

  if (error_db(data, ref) > -100)
  {
    if (XFail<B, F, D, r, CT, CT, s>::value)
    {
      std::cout << "expected error." << std::endl;
    }
    else
    {
      std::cout << "unexpected error." << std::endl;
      has_errors = true;
//     std::cout << "data " << data << std::endl;
//     std::cout << "ref  " << ref << std::endl;
    }
  }
  else std::cout << "ok." << std::endl;
}

template <typename T, typename B>
void fftm_by_ref(Domain<2> const &dom)
{
  typedef typename T::I I;
  typedef typename T::O O;
  typedef typename T::order_type order_type;
  typedef typename vsip::Layout<2, order_type, vsip::dense, T::i_format> i_layout_type;
  typedef typename vsip::Layout<2, order_type, vsip::dense, T::o_format> o_layout_type;
  return_mechanism_type const r = by_reference;

  typedef Strided<2, I, i_layout_type> Iblock;
  typedef Strided<2, O, o_layout_type> Oblock;
  typedef Matrix<I, Iblock> Iview;
  typedef Matrix<O, Oblock> Oview;

  Domain<2> in_dom = T::in_dom(dom);
  Domain<2> out_dom = T::out_dom(dom);

  Iview input = input_creator<T, 2>::create(dom);
  typename Iview::subview_type sub_input = input(in_dom);

  Oview output = empty<O>(out_dom);
  typename Oview::subview_type sub_output = output(out_dom);
  typename ovxx::signal::Fftm<I, O, typename B::list,
			      1 - T::axis, T::direction, r> fftm(dom, 42.);
  fftm(sub_input, sub_output);

  Oview ref = empty<O>(out_dom);
  typename Oview::subview_type sub_ref = ref(out_dom);
  typename ovxx::signal::Fftm<I, O, dft::list,
			      1 - T::axis, T::direction, r> ref_fftm(dom, 42.);
  ref_fftm(sub_input, sub_ref);

  if (error_db(output, ref) > -100)
  {
    if (XFailM<B, T::o_format, r, I, O, T::direction>::value)
    {
      std::cout << "expected error." << std::endl;
    }
    else
    {
      std::cout << "error." << std::endl;
      has_errors = true;
//     std::cout << "out " << output << std::endl;
//     std::cout << "ref  " << ref << std::endl;
    }
  }
  else std::cout << "ok." << std::endl;
}

template <typename T, storage_format_type F, int E, int A, typename B>
void fftm_in_place(Domain<2> const &dom)
{
  int const d = E == -1 ? fft_fwd : fft_inv;
  return_mechanism_type const r = by_reference;
  typedef typename vsip::Layout<2, row1_type, vsip::dense, F> layout_type;
  typedef std::complex<T> CT;

  typedef Strided<2, CT, layout_type> block_type;
  typedef Matrix<CT, block_type> View;

  View data = ramp<T>(dom);
  View ref = ramp<T>(dom);

  typename View::subview_type sub_data = data(dom);

  typename ovxx::signal::Fftm<CT, CT, typename B::list, A, d, r>
    fftm(dom, 42.);
  fftm(sub_data);

  typename View::subview_type sub_ref = ref(dom);
  typename ovxx::signal::Fftm<CT, CT, dft::list, A, d, r> ref_fftm(dom, 42.);
  ref_fftm(sub_ref);

  if (error_db(data, ref) > -100)
  {
    if (XFailM<B, F, r, CT, CT, d>::value)
      std::cout << "expected error." << std::endl;
    else
    {
      std::cout << "error." << std::endl;
      has_errors = true;
//     std::cout << "data " << data << std::endl;
//     std::cout << "ref  " << ref << std::endl;
    }
  }
  else std::cout << "ok." << std::endl;
}

template <typename T, storage_format_type F>
void test_fft1d()
{
#ifdef FFT_COMPOSITE
  std::cout << "testing c->c fwd by_ref composite...";
  fft_by_ref<cfft_type<T, F, fft_fwd>, composite>(Domain<1>(16));
  fft_by_ref<cfft_type<T, F, fft_fwd>, composite>(Domain<1>(0, 2, 8));
  std::cout << "testing c->c inv by_ref composite...";
  fft_by_ref<cfft_type<T, F, fft_inv>, composite>(Domain<1>(16));
  fft_by_ref<cfft_type<T, F, fft_inv>, composite>(Domain<1>(0, 2, 8));
  std::cout << "testing r->c fwd 0 by_ref composite...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, composite>(Domain<1>(16));
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, composite>(Domain<1>(0, 2, 8));
  std::cout << "testing c->r inv 0 by_ref composite...";
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, composite>(Domain<1>(16));
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, composite>(Domain<1>(0, 2, 8));

  std::cout << "testing c->c fwd by_ref composite...";
  fft_by_ref<cfft_type<T, F, fft_fwd>, composite>(Domain<1>(16));
  fft_by_ref<cfft_type<T, F, fft_fwd>, composite>(Domain<1>(0, 2, 8));
  std::cout << "testing c->c inv by_ref composite...";
  fft_by_ref<cfft_type<T, F, fft_inv>, composite>(Domain<1>(16));
  fft_by_ref<cfft_type<T, F, fft_inv>, composite>(Domain<1>(0, 2, 8));
  std::cout << "testing r->c fwd 0 by_ref composite...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, composite>(Domain<1>(16));
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, composite>(Domain<1>(0, 2, 8));
  std::cout << "testing c->r inv 0 by_ref composite...";
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, composite>(Domain<1>(16));
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, composite>(Domain<1>(0, 2, 8));
#else

#if VSIP_IMPL_FFTW3
  std::cout << "testing fwd in_place fftw...";
  fft_in_place<T, F, fft_fwd, fftw>(Domain<1>(16));
  fft_in_place<T, F, fft_fwd, fftw>(Domain<1>(0, 2, 8));
  std::cout << "testing inv in_place fftw...";
  fft_in_place<T, F, fft_inv, fftw>(Domain<1>(16));
  fft_in_place<T, F, fft_inv, fftw>(Domain<1>(0, 2, 8));
#endif

#if VSIP_IMPL_SAL_FFT
  std::cout << "testing fwd in_place sal...";
  fft_in_place<T, F, fft_fwd, sal>(Domain<1>(16));
  fft_in_place<T, F, fft_fwd, sal>(Domain<1>(0, 2, 8));
  std::cout << "testing inv in_place sal...";
  fft_in_place<T, F, fft_inv, sal>(Domain<1>(16));
  fft_in_place<T, F, fft_inv, sal>(Domain<1>(0, 2, 8));
#endif

#if VSIP_IMPL_IPP_FFT
  std::cout << "testing fwd in_place ipp...";
  fft_in_place<T, F, fft_fwd, ipp>(Domain<1>(16));
  fft_in_place<T, F, fft_fwd, ipp>(Domain<1>(0, 2, 8));
  std::cout << "testing inv in_place ipp...";
  fft_in_place<T, F, fft_inv, ipp>(Domain<1>(16));
  fft_in_place<T, F, fft_inv, ipp>(Domain<1>(0, 2, 8));
#endif

#if VSIP_IMPL_CVSIP_FFT
  std::cout << "testing fwd in_place cvsip...";
  fft_in_place<T, F, fft_fwd, cvsip>(Domain<1>(16));
  fft_in_place<T, F, fft_fwd, cvsip>(Domain<1>(0, 2, 8));
  std::cout << "testing inv in_place cvsip...";
  fft_in_place<T, F, fft_inv, cvsip>(Domain<1>(16));
  fft_in_place<T, F, fft_inv, cvsip>(Domain<1>(0, 2, 8));
#endif

#if VSIP_IMPL_CBE_SDK_FFT
  std::cout << "testing fwd in_place cbe...";
  fft_in_place<T, F, fft_fwd, cbe>(Domain<1>(32));
  std::cout << "testing inv in_place cbe...";
  fft_in_place<T, F, fft_inv, cbe>(Domain<1>(32));
#endif

#if VSIP_IMPL_FFTW3
  std::cout << "testing c->c fwd by_ref fftw...";
  fft_by_ref<cfft_type<T, F, fft_fwd>, fftw>(Domain<1>(16));
  fft_by_ref<cfft_type<T, F, fft_fwd>, fftw>(Domain<1>(0, 2, 8));
  std::cout << "testing c->c inv by_ref fftw...";
  fft_by_ref<cfft_type<T, F, fft_inv>, fftw>(Domain<1>(16));
  fft_by_ref<cfft_type<T, F, fft_inv>, fftw>(Domain<1>(0, 2, 8));
  std::cout << "testing r->c fwd 0 by_ref fftw...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, fftw>(Domain<1>(16));
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, fftw>(Domain<1>(0, 2, 8));
  std::cout << "testing c->r inv 0 by_ref fftw...";
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, fftw>(Domain<1>(16));
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, fftw>(Domain<1>(0, 2, 8));
#endif

#if VSIP_IMPL_SAL_FFT
  std::cout << "testing c->c fwd by_ref sal...";
  fft_by_ref<cfft_type<T, F, fft_fwd>, sal>(Domain<1>(16));
  fft_by_ref<cfft_type<T, F, fft_fwd>, sal>(Domain<1>(0, 2, 8));
  std::cout << "testing c->c inv by_ref sal...";
  fft_by_ref<cfft_type<T, F, fft_inv>, sal>(Domain<1>(16));
  fft_by_ref<cfft_type<T, F, fft_inv>, sal>(Domain<1>(0, 2, 8));
  std::cout << "testing r->c fwd 0 by_ref sal...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, sal>(Domain<1>(16));
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, sal>(Domain<1>(0, 2, 8));
  std::cout << "testing c->r inv 0 by_ref sal...";
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, sal>(Domain<1>(16));
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, sal>(Domain<1>(0, 2, 8));
#endif

#if VSIP_IMPL_IPP_FFT
  std::cout << "testing c->c fwd by_ref ipp...";
  fft_by_ref<cfft_type<T, F, fft_fwd>, ipp>(Domain<1>(16));
  fft_by_ref<cfft_type<T, F, fft_fwd>, ipp>(Domain<1>(0, 2, 8));
  std::cout << "testing c->c inv by_ref ipp...";
  fft_by_ref<cfft_type<T, F, fft_inv>, ipp>(Domain<1>(16));
  fft_by_ref<cfft_type<T, F, fft_inv>, ipp>(Domain<1>(0, 2, 8));
  std::cout << "testing r->c fwd 0 by_ref ipp...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, ipp>(Domain<1>(16));
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, ipp>(Domain<1>(0, 2, 8));
  std::cout << "testing c->r inv 0 by_ref ipp...";
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, ipp>(Domain<1>(16));
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, ipp>(Domain<1>(0, 2, 8));
#endif

#if VSIP_IMPL_CVSIP_FFT
  std::cout << "testing c->c fwd by_ref cvsip...";
  fft_by_ref<cfft_type<T, F, fft_fwd>, cvsip>(Domain<1>(16));
  fft_by_ref<cfft_type<T, F, fft_fwd>, cvsip>(Domain<1>(0, 2, 8));
  std::cout << "testing c->c inv by_ref cvsip...";
  fft_by_ref<cfft_type<T, F, fft_inv>, cvsip>(Domain<1>(16));
  fft_by_ref<cfft_type<T, F, fft_inv>, cvsip>(Domain<1>(0, 2, 8));
  std::cout << "testing r->c fwd 0 by_ref cvsip...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, cvsip>(Domain<1>(16));
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, cvsip>(Domain<1>(0, 2, 8));
  std::cout << "testing c->r inv 0 by_ref cvsip...";
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, cvsip>(Domain<1>(16));
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, cvsip>(Domain<1>(0, 2, 8));
#endif

#if VSIP_IMPL_CBE_SDK_FFT
  std::cout << "testing c->c fwd by_ref cbe...";
  fft_by_ref<cfft_type<T, F, fft_fwd>, cbe>(Domain<1>(32));
  std::cout << "testing c->c inv by_ref cbe...";
  fft_by_ref<cfft_type<T, F, fft_inv>, cbe>(Domain<1>(32));
#endif

#endif
}

template <typename T, storage_format_type F>
void test_fft2d()
{
#ifdef FFT_COMPOSITE
  std::cout << "testing fwd in_place composite...";
  fft_in_place<T, F, fft_fwd, composite>(Domain<2>(8, 16));
  fft_in_place<T, F, fft_fwd, composite>(Domain<2>(Domain<1>(0, 2, 8),
                                              Domain<1>(0, 2, 16)));
  std::cout << "testing inv in_place composite...";
  fft_in_place<T, F, fft_inv, composite>(Domain<2>(8, 16));
  fft_in_place<T, F, fft_inv, composite>(Domain<2>(Domain<1>(0, 2, 8),
                                             Domain<1>(0, 2, 16)));

  std::cout << "testing c->c fwd by_ref composite...";
  fft_by_ref<cfft_type<T, F, fft_fwd>, composite>(Domain<2>(8, 16));
  fft_by_ref<cfft_type<T, F, fft_fwd>, composite>(Domain<2>(Domain<1>(0, 2, 8),
                                                       Domain<1>(0, 2, 16)));
  std::cout << "testing c->c inv by_ref composite...";
  fft_by_ref<cfft_type<T, F, fft_inv>, composite>(Domain<2>(8, 16));
  fft_by_ref<cfft_type<T, F, fft_inv>, composite>(Domain<2>(Domain<1>(0, 2, 8),
                                                      Domain<1>(0, 2, 16)));
  std::cout << "testing r->c fwd 0 by_ref composite...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, composite>(Domain<2>(8, 16));
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, composite>(Domain<2>(Domain<1>(0, 2, 8),
                                                          Domain<1>(0, 2, 16)));
  std::cout << "testing r->c fwd 1 by_ref composite...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 1>, composite>(Domain<2>(8, 16));
  fft_by_ref<rfft_type<T, F, fft_fwd, 1>, composite>(Domain<2>(Domain<1>(0, 2, 8),
                                                          Domain<1>(0, 2, 16)));
  std::cout << "testing c->r inv 0 by_ref composite...";
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, composite> (Domain<2>(4, 5));
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, composite> (Domain<2>(Domain<1>(0, 2, 8),
                                                          Domain<1>(0, 2, 16)));
  std::cout << "testing c->r inv 1 by_ref composite...";
  fft_by_ref<rfft_type<T, F, fft_inv, 1>, composite> (Domain<2>(8, 16));
  fft_by_ref<rfft_type<T, F, fft_inv, 1>, composite> (Domain<2>(Domain<1>(0, 2, 8),
                                                          Domain<1>(0, 2, 16)));
#else

#if VSIP_IMPL_FFTW3
  std::cout << "testing fwd in_place fftw...";
  fft_in_place<T, F, fft_fwd, fftw>(Domain<2>(8, 16));
  fft_in_place<T, F, fft_fwd, fftw>(Domain<2>(Domain<1>(0, 2, 8),
					 Domain<1>(0, 2, 16)));
  std::cout << "testing inv in_place fftw...";
  fft_in_place<T, F, fft_inv, fftw>(Domain<2>(8, 16));
  fft_in_place<T, F, fft_inv, fftw>(Domain<2>(Domain<1>(0, 2, 8),
					Domain<1>(0, 2, 16)));
#endif

#if VSIP_IMPL_SAL_FFT
  std::cout << "testing fwd in_place sal...";
  fft_in_place<T, F, fft_fwd, sal>(Domain<2>(8, 16));
  fft_in_place<T, F, fft_fwd, sal>(Domain<2>(Domain<1>(0, 2, 8),
					Domain<1>(0, 2, 16)));
  std::cout << "testing inv in_place sal...";
  fft_in_place<T, F, fft_inv, sal>(Domain<2>(8, 16));
  fft_in_place<T, F, fft_inv, sal>(Domain<2>(Domain<1>(0, 2, 8),
				       Domain<1>(0, 2, 16)));
#endif

#if VSIP_IMPL_IPP_FFT
  std::cout << "testing fwd in_place ipp...";
  fft_in_place<T, F, fft_fwd, ipp>(Domain<2>(8, 16));
  fft_in_place<T, F, fft_fwd, ipp>(Domain<2>(Domain<1>(0, 2, 8),
					Domain<1>(0, 2, 16)));
  std::cout << "testing inv in_place ipp...";
  fft_in_place<T, F, fft_inv, ipp>(Domain<2>(8, 17));
  fft_in_place<T, F, fft_inv, ipp>(Domain<2>(Domain<1>(0, 2, 8),
				       Domain<1>(0, 2, 16)));
#endif

#if VSIP_IMPL_FFTW3
  std::cout << "testing c->c fwd by_ref fftw...";
  fft_by_ref<cfft_type<T, F, fft_fwd>, fftw>(Domain<2>(8, 16));
  fft_by_ref<cfft_type<T, F, fft_fwd>, fftw>(Domain<2>(Domain<1>(0, 2, 8),
						  Domain<1>(0, 2, 16)));
  std::cout << "testing c->c inv by_ref fftw...";
  fft_by_ref<cfft_type<T, F, fft_inv>, fftw>(Domain<2>(8, 16));
  fft_by_ref<cfft_type<T, F, fft_inv>, fftw>(Domain<2>(Domain<1>(0, 2, 8),
						 Domain<1>(0, 2, 16)));
  std::cout << "testing r->c fwd 0 by_ref fftw...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, fftw>(Domain<2>(8, 16));
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, fftw>(Domain<2>(Domain<1>(0, 2, 8),
						     Domain<1>(0, 2, 16)));
  std::cout << "testing r->c fwd 1 by_ref fftw...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 1>, fftw>(Domain<2>(8, 16));
  fft_by_ref<rfft_type<T, F, fft_fwd, 1>, fftw>(Domain<2>(Domain<1>(0, 2, 8),
						     Domain<1>(0, 2, 16)));
  std::cout << "testing c->r inv 0 by_ref fftw...";
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, fftw> (Domain<2>(4, 5));
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, fftw> (Domain<2>(Domain<1>(0, 2, 8),
						     Domain<1>(0, 2, 16)));
  std::cout << "testing c->r inv 1 by_ref fftw...";
  fft_by_ref<rfft_type<T, F, fft_inv, 1>, fftw> (Domain<2>(8, 16));
  fft_by_ref<rfft_type<T, F, fft_inv, 1>, fftw> (Domain<2>(Domain<1>(0, 2, 8),
						     Domain<1>(0, 2, 16)));
#endif

#if VSIP_IMPL_SAL_FFT
  std::cout << "testing c->c fwd by_ref sal...";
  fft_by_ref<cfft_type<T, F, fft_fwd>, sal>(Domain<2>(8, 16));
  fft_by_ref<cfft_type<T, F, fft_fwd>, sal>(Domain<2>(Domain<1>(0, 2, 8),
						 Domain<1>(0, 2, 16)));
  std::cout << "testing c->c inv by_ref sal...";
  fft_by_ref<cfft_type<T, F, fft_inv>, sal>(Domain<2>(8, 16));
  fft_by_ref<cfft_type<T, F, fft_inv>, sal>(Domain<2>(Domain<1>(0, 2, 8),
						Domain<1>(0, 2, 16)));
  std::cout << "testing r->c fwd 0 by_ref sal...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, sal>(Domain<2>(8, 16));
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, sal>(Domain<2>(Domain<1>(0, 2, 8),
						    Domain<1>(0, 2, 16)));
  std::cout << "testing r->c fwd 1 by_ref sal...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 1>, sal>(Domain<2>(8, 16));
  fft_by_ref<rfft_type<T, F, fft_fwd, 1>, sal>(Domain<2>(Domain<1>(0, 2, 8),
						    Domain<1>(0, 2, 16)));
  std::cout << "testing c->r inv 0 by_ref sal...";
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, sal> (Domain<2>(8, 16));
//   fft_by_ref<rfft_type<T, F, fft_inv, 0>, sal> (Domain<2>(Domain<1>(0, 2, 8),
// 						    Domain<1>(0, 2, 16)));
  std::cout << "testing c->r inv 1 by_ref sal...";
  fft_by_ref<rfft_type<T, F, fft_inv, 1>, sal> (Domain<2>(8, 16));
//   fft_by_ref<rfft_type<T, F, fft_inv, 1>, sal> (Domain<2>(Domain<1>(0, 2, 8),
// 						    Domain<1>(0, 2, 16)));
#endif

#if VSIP_IMPL_IPP_FFT
  std::cout << "testing c->c fwd by_ref ipp...";
  fft_by_ref<cfft_type<T, F, fft_fwd>, ipp>(Domain<2>(8, 16));
  fft_by_ref<cfft_type<T, F, fft_fwd>, ipp>(Domain<2>(Domain<1>(0, 2, 8),
						 Domain<1>(0, 2, 16)));
  std::cout << "testing c->c inv by_ref ipp...";
  fft_by_ref<cfft_type<T, F, fft_inv>, ipp>(Domain<2>(8, 16));
  fft_by_ref<cfft_type<T, F, fft_inv>, ipp>(Domain<2>(Domain<1>(0, 2, 8),
						Domain<1>(0, 2, 16)));
  std::cout << "testing r->c fwd 0 by_ref ipp...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, ipp>(Domain<2>(8, 16));
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, ipp>(Domain<2>(Domain<1>(0, 2, 8),
						    Domain<1>(0, 2, 16)));
  std::cout << "testing r->c fwd 1 by_ref ipp...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 1>, ipp>(Domain<2>(8, 16));
  fft_by_ref<rfft_type<T, F, fft_fwd, 1>, ipp>(Domain<2>(Domain<1>(0, 2, 8),
						    Domain<1>(0, 2, 16)));
  std::cout << "testing c->r inv 0 by_ref ipp...";
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, ipp> (Domain<2>(8, 16));
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, ipp> (Domain<2>(Domain<1>(0, 2, 8),
						    Domain<1>(0, 2, 16)));
  std::cout << "testing c->r inv 1 by_ref ipp...";
  fft_by_ref<rfft_type<T, F, fft_inv, 1>, ipp> (Domain<2>(8, 16));
  fft_by_ref<rfft_type<T, F, fft_inv, 1>, ipp> (Domain<2>(Domain<1>(0, 2, 8),
						    Domain<1>(0, 2, 16)));
#endif

#endif
}

template <typename T, storage_format_type F>
void test_fft3d()
{
#ifdef FFT_COMPOSITE
  std::cout << "testing fwd in_place composite...";
  fft_in_place<T, F, fft_fwd, composite>(Domain<3>(8, 16, 32));
  fft_in_place<T, F, fft_fwd, composite>(Domain<3>(Domain<1>(0, 2, 8),
                                              Domain<1>(0, 2, 16),
                                              Domain<1>(0, 2, 32)));
  std::cout << "testing inv in_place composite...";
  fft_in_place<T, F, fft_inv, composite>(Domain<3>(8, 16, 32));
  fft_in_place<T, F, fft_inv, composite>(Domain<3>(Domain<1>(0, 2, 8),
                                             Domain<1>(0, 2, 16),
                                             Domain<1>(0, 2, 32)));

  std::cout << "testing c->c fwd by_ref composite...";
  fft_by_ref<cfft_type<T, F, fft_fwd>, composite>(Domain<3>(8, 16, 32));
  fft_by_ref<cfft_type<T, F, fft_fwd>, composite>(Domain<3>(Domain<1>(0, 2, 8),
                                                       Domain<1>(0, 2, 16),
                                                       Domain<1>(0, 2, 32)));
  std::cout << "testing c->c inv by_ref composite...";
  fft_by_ref<cfft_type<T, F, fft_inv>, composite>(Domain<3>(8, 16, 32));
  fft_by_ref<cfft_type<T, F, fft_inv>, composite>(Domain<3>(Domain<1>(0, 2, 8),
                                                      Domain<1>(0, 2, 16),
                                                      Domain<1>(0, 2, 32)));
  std::cout << "testing r->c fwd 0 by_ref composite...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, composite>(Domain<3>(8, 16, 32));
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, composite>(Domain<3>(Domain<1>(0, 2, 8),
                                                          Domain<1>(0, 2, 16),
                                                          Domain<1>(0, 2, 32)));
  std::cout << "testing r->c fwd 1 by_ref composite...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 1>, composite>(Domain<3>(8, 16, 32));
  fft_by_ref<rfft_type<T, F, fft_fwd, 1>, composite>(Domain<3>(Domain<1>(0, 2, 8),
                                                          Domain<1>(0, 2, 16),
                                                          Domain<1>(0, 2, 32)));
  std::cout << "testing r->c fwd 2 by_ref composite...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 2>, composite>(Domain<3>(8, 16, 32));
  fft_by_ref<rfft_type<T, F, fft_fwd, 2>, composite>(Domain<3>(Domain<1>(0, 2, 8),
                                                          Domain<1>(0, 2, 16),
                                                          Domain<1>(0, 2, 32)));
  std::cout << "testing c->r inv 0 by_ref composite...";
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, composite> (Domain<3>(8, 16, 32));
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, composite> (Domain<3>(Domain<1>(0, 2, 8),
                                                          Domain<1>(0, 2, 16),
                                                          Domain<1>(0, 2, 32)));
  std::cout << "testing c->r inv 1 by_ref composite...";
  fft_by_ref<rfft_type<T, F, fft_inv, 1>, composite> (Domain<3>(8, 16, 32));
  fft_by_ref<rfft_type<T, F, fft_inv, 1>, composite> (Domain<3>(Domain<1>(0, 2, 8),
                                                          Domain<1>(0, 2, 16),
                                                          Domain<1>(0, 2, 32)));
  std::cout << "testing c->r inv 2 by_ref composite...";
  fft_by_ref<rfft_type<T, F, fft_inv, 2>, composite> (Domain<3>(8, 16, 32));
  fft_by_ref<rfft_type<T, F, fft_inv, 2>, composite> (Domain<3>(Domain<1>(0, 2, 8),
                                                          Domain<1>(0, 2, 16),
                                                          Domain<1>(0, 2, 32)));
#elif VSIP_IMPL_FFTW3

  std::cout << "testing fwd in_place fftw...";
  fft_in_place<T, F, fft_fwd, fftw>(Domain<3>(8, 16, 32));
  fft_in_place<T, F, fft_fwd, fftw>(Domain<3>(Domain<1>(0, 2, 8),
					 Domain<1>(0, 2, 16),
					 Domain<1>(0, 2, 32)));
  std::cout << "testing inv in_place fftw...";
  fft_in_place<T, F, fft_inv, fftw>(Domain<3>(8, 16, 32));
  fft_in_place<T, F, fft_inv, fftw>(Domain<3>(Domain<1>(0, 2, 8),
					Domain<1>(0, 2, 16),
					Domain<1>(0, 2, 32)));

  std::cout << "testing c->c fwd by_ref fftw...";
  fft_by_ref<cfft_type<T, F, fft_fwd>, fftw>(Domain<3>(8, 16, 32));
  fft_by_ref<cfft_type<T, F, fft_fwd>, fftw>(Domain<3>(Domain<1>(0, 2, 8),
						  Domain<1>(0, 2, 16),
						  Domain<1>(0, 2, 32)));
  std::cout << "testing c->c inv by_ref fftw...";
  fft_by_ref<cfft_type<T, F, fft_inv>, fftw>(Domain<3>(8, 16, 32));
  fft_by_ref<cfft_type<T, F, fft_inv>, fftw>(Domain<3>(Domain<1>(0, 2, 8),
						 Domain<1>(0, 2, 16),
						 Domain<1>(0, 2, 32)));
  std::cout << "testing r->c fwd 0 by_ref fftw...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, fftw>(Domain<3>(8, 16, 32));
  fft_by_ref<rfft_type<T, F, fft_fwd, 0>, fftw>(Domain<3>(Domain<1>(0, 2, 8),
						     Domain<1>(0, 2, 16),
						     Domain<1>(0, 2, 32)));
  std::cout << "testing r->c fwd 1 by_ref fftw...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 1>, fftw>(Domain<3>(8, 16, 32));
  fft_by_ref<rfft_type<T, F, fft_fwd, 1>, fftw>(Domain<3>(Domain<1>(0, 2, 8),
						     Domain<1>(0, 2, 16),
						     Domain<1>(0, 2, 32)));
  std::cout << "testing r->c fwd 2 by_ref fftw...";
  fft_by_ref<rfft_type<T, F, fft_fwd, 2>, fftw>(Domain<3>(8, 16, 32));
  fft_by_ref<rfft_type<T, F, fft_fwd, 2>, fftw>(Domain<3>(Domain<1>(0, 2, 8),
						     Domain<1>(0, 2, 16),
						     Domain<1>(0, 2, 32)));
  std::cout << "testing c->r inv 0 by_ref fftw...";
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, fftw> (Domain<3>(8, 16, 32));
  fft_by_ref<rfft_type<T, F, fft_inv, 0>, fftw> (Domain<3>(Domain<1>(0, 2, 8),
						     Domain<1>(0, 2, 16),
						     Domain<1>(0, 2, 32)));
  std::cout << "testing c->r inv 1 by_ref fftw...";
  fft_by_ref<rfft_type<T, F, fft_inv, 1>, fftw> (Domain<3>(8, 16, 32));
  fft_by_ref<rfft_type<T, F, fft_inv, 1>, fftw> (Domain<3>(Domain<1>(0, 2, 8),
						     Domain<1>(0, 2, 16),
						     Domain<1>(0, 2, 32)));
  std::cout << "testing c->r inv 2 by_ref fftw...";
  fft_by_ref<rfft_type<T, F, fft_inv, 2>, fftw> (Domain<3>(8, 16, 32));
  fft_by_ref<rfft_type<T, F, fft_inv, 2>, fftw> (Domain<3>(Domain<1>(0, 2, 8),
						     Domain<1>(0, 2, 16),
						     Domain<1>(0, 2, 32)));
#endif
}

template <typename T, storage_format_type F>
void test_fftm()
{
#ifdef FFT_COMPOSITE
  std::cout << "testing fwd 0 in_place composite...";
  fftm_in_place<T, F, fft_fwd, 0, composite>(Domain<2>(8, 16));
  std::cout << "testing fwd 1 in_place composite...";
  fftm_in_place<T, F, fft_fwd, 1, composite>(Domain<2>(8, 16));
  std::cout << "testing inv 0 in_place composite...";
  fftm_in_place<T, F, fft_inv, 0, composite>(Domain<2>(8, 16));
  std::cout << "testing inv 1 in_place composite...";
  fftm_in_place<T, F, fft_inv, 1, composite>(Domain<2>(8, 16));

  std::cout << "testing c->c fwd 0 by_ref composite...";
  fftm_by_ref<cfft_type<T, F, fft_fwd, 0>, composite>(Domain<2>(8, 16));
  std::cout << "testing c->c fwd 1 by_ref composite...";
  fftm_by_ref<cfft_type<T, F, fft_fwd, 1>, composite>(Domain<2>(8, 16));
  std::cout << "testing c->c inv 0 by_ref composite...";
  fftm_by_ref<cfft_type<T, F, fft_inv, 0>, composite>(Domain<2>(8, 16));
  std::cout << "testing c->c inv 1 by_ref composite...";
  fftm_by_ref<cfft_type<T, F, fft_inv, 1>, composite>(Domain<2>(8, 16));
  std::cout << "testing r->c fwd 0 by_ref composite...";
  fftm_by_ref<rfft_type<T, F, fft_fwd, 0>, composite>(Domain<2>(8, 16));
  std::cout << "testing r->c fwd 1 by_ref composite...";
  fftm_by_ref<rfft_type<T, F, fft_fwd, 1>, composite>(Domain<2>(8, 16));
  std::cout << "testing c->r inv 0 by_ref composite...";
  fftm_by_ref<rfft_type<T, F, fft_inv, 0>, composite>(Domain<2>(8, 16));
  std::cout << "testing c->r inv 1 by_ref composite...";
  fftm_by_ref<rfft_type<T, F, fft_inv, 1>, composite>(Domain<2>(8, 16));
#else

#if VSIP_IMPL_FFTW3
  std::cout << "testing fwd 0 in_place fftw...";
  fftm_in_place<T, F, fft_fwd, 0, fftw>(Domain<2>(8, 16));
  std::cout << "testing fwd 1 in_place fftw...";
  fftm_in_place<T, F, fft_fwd, 1, fftw>(Domain<2>(8, 16));
  std::cout << "testing inv 0 in_place fftw...";
  fftm_in_place<T, F, fft_inv, 0, fftw>(Domain<2>(8, 16));
  std::cout << "testing inv 1 in_place fftw...";
  fftm_in_place<T, F, fft_inv, 1, fftw>(Domain<2>(8, 16));
#endif

#if VSIP_IMPL_SAL_FFT
  std::cout << "testing fwd 0 in_place sal...";
  fftm_in_place<T, F, fft_fwd, 0, sal>(Domain<2>(8, 16));
  std::cout << "testing fwd 1 in_place sal...";
  fftm_in_place<T, F, fft_fwd, 1, sal>(Domain<2>(8, 16));
  std::cout << "testing inv 0 in_place sal...";
  fftm_in_place<T, F, fft_inv, 0, sal>(Domain<2>(8, 16));
  std::cout << "testing inv 1 in_place sal...";
  fftm_in_place<T, F, fft_inv, 1, sal>(Domain<2>(8, 16));
#endif

#if VSIP_IMPL_IPP_FFT
  std::cout << "testing fwd 0 in_place ipp...";
  fftm_in_place<T, F, fft_fwd, 0, ipp>(Domain<2>(8, 16));
  std::cout << "testing fwd 1 in_place ipp...";
  fftm_in_place<T, F, fft_fwd, 1, ipp>(Domain<2>(8, 16));
  std::cout << "testing inv 0 in_place ipp...";
  fftm_in_place<T, F, fft_inv, 0, ipp>(Domain<2>(8, 16));
  std::cout << "testing inv 1 in_place ipp...";
  fftm_in_place<T, F, fft_inv, 1, ipp>(Domain<2>(8, 16));
#endif

#if VSIP_IMPL_CVSIP_FFT
  std::cout << "testing fwd 0 in_place cvsip...";
  fftm_in_place<T, F, fft_fwd, 0, cvsip>(Domain<2>(8, 16));
  std::cout << "testing fwd 1 in_place cvsip...";
  fftm_in_place<T, F, fft_fwd, 1, cvsip>(Domain<2>(8, 16));
  std::cout << "testing inv 0 in_place cvsip...";
  fftm_in_place<T, F, fft_inv, 0, cvsip>(Domain<2>(8, 16));
  std::cout << "testing inv 1 in_place cvsip...";
  fftm_in_place<T, F, fft_inv, 1, cvsip>(Domain<2>(8, 16));
#endif

#if VSIP_IMPL_CBE_SDK_FFT
// Note: column-wise FFTs need to be performed on
// col-major data in this case.  These are commented
// out until fftm_in_place is changed to be like
// fftm_by_ref, where the cfft_type<> template allows
// the dimension order to be specified.

//  std::cout << "testing fwd on cols in_place cbe...";
//  fftm_in_place<T, F, fft_fwd, 0, cbe>(Domain<2>(64, 32));
  std::cout << "testing fwd on rows in_place cbe...";
  fftm_in_place<T, F, fft_fwd, 1, cbe>(Domain<2>(32, 64));
//  std::cout << "testing inv on cols in_place cbe...";
//  fftm_in_place<T, F, 1, 0, cbe>(Domain<2>(64, 32));
  std::cout << "testing inv on rows in_place cbe...";
  fftm_in_place<T, F, fft_inv, 1, cbe>(Domain<2>(32, 64));
#endif

#if VSIP_IMPL_FFTW3
  std::cout << "testing c->c fwd 0 by_ref fftw...";
  fftm_by_ref<cfft_type<T, F, fft_fwd, 0>, fftw>(Domain<2>(8, 16));
  std::cout << "testing c->c fwd 1 by_ref fftw...";
  fftm_by_ref<cfft_type<T, F, fft_fwd, 1>, fftw>(Domain<2>(8, 16));
  std::cout << "testing c->c inv 0 by_ref fftw...";
  fftm_by_ref<cfft_type<T, F, fft_inv, 0>, fftw>(Domain<2>(8, 16));
  std::cout << "testing c->c inv 1 by_ref fftw...";
  fftm_by_ref<cfft_type<T, F, fft_inv, 1>, fftw>(Domain<2>(8, 16));
  std::cout << "testing r->c fwd 0 by_ref fftw...";
  fftm_by_ref<rfft_type<T, F, fft_fwd, 0>, fftw>(Domain<2>(8, 16));
  std::cout << "testing r->c fwd 1 by_ref fftw...";
  fftm_by_ref<rfft_type<T, F, fft_fwd, 1>, fftw>(Domain<2>(8, 16));
  std::cout << "testing c->r inv 0 by_ref fftw...";
  fftm_by_ref<rfft_type<T, F, fft_inv, 0>, fftw>(Domain<2>(8, 16));
  std::cout << "testing c->r inv 1 by_ref fftw...";
  fftm_by_ref<rfft_type<T, F, fft_inv, 1>, fftw>(Domain<2>(8, 16));
#endif

#if VSIP_IMPL_SAL_FFT
  std::cout << "testing c->c fwd 0 by_ref sal...";
  fftm_by_ref<cfft_type<T, F, fft_fwd, 0>, sal>(Domain<2>(8, 16));
  std::cout << "testing c->c fwd 1 by_ref sal...";
  fftm_by_ref<cfft_type<T, F, fft_fwd, 1>, sal>(Domain<2>(8, 16));
  std::cout << "testing c->c inv 0 by_ref sal...";
  fftm_by_ref<cfft_type<T, F, fft_inv, 0>, sal>(Domain<2>(8, 16));
  std::cout << "testing c->c inv 1 by_ref sal...";
  fftm_by_ref<cfft_type<T, F, fft_inv, 1>, sal>(Domain<2>(8, 16));
  std::cout << "testing r->c fwd 0 by_ref sal...";
  fftm_by_ref<rfft_type<T, F, fft_fwd, 0>, sal>(Domain<2>(8, 16));
  std::cout << "testing r->c fwd 1 by_ref sal...";
  fftm_by_ref<rfft_type<T, F, fft_fwd, 1>, sal>(Domain<2>(8, 16));
  std::cout << "testing c->r inv 0 by_ref sal...";
  fftm_by_ref<rfft_type<T, F, fft_inv, 0>, sal> (Domain<2>(8, 16));
  std::cout << "testing c->r inv 1 by_ref sal...";
  fftm_by_ref<rfft_type<T, F, fft_inv, 1>, sal> (Domain<2>(8, 16));
#endif

#if VSIP_IMPL_IPP_FFT
  std::cout << "testing c->c fwd 0 by_ref ipp...";
  fftm_by_ref<cfft_type<T, F, fft_fwd, 0>, ipp>(Domain<2>(8, 16));
  std::cout << "testing c->c fwd 1 by_ref ipp...";
  fftm_by_ref<cfft_type<T, F, fft_fwd, 1>, ipp>(Domain<2>(8, 16));
  std::cout << "testing c->c inv 0 by_ref ipp...";
  fftm_by_ref<cfft_type<T, F, fft_inv, 0>, ipp>(Domain<2>(8, 16));
  std::cout << "testing c->c inv 1 by_ref ipp...";
  fftm_by_ref<cfft_type<T, F, fft_inv, 1>, ipp>(Domain<2>(8, 16));
  std::cout << "testing r->c fwd 0 by_ref ipp...";
  fftm_by_ref<rfft_type<T, F, fft_fwd, 0>, ipp>(Domain<2>(8, 16));
  std::cout << "testing r->c fwd 1 by_ref ipp...";
  fftm_by_ref<rfft_type<T, F, fft_fwd, 1>, ipp>(Domain<2>(8, 16));
  std::cout << "testing c->r inv 0 by_ref ipp...";
  fftm_by_ref<rfft_type<T, F, fft_inv, 0>, ipp> (Domain<2>(8, 16));
  std::cout << "testing c->r inv 1 by_ref ipp...";
  fftm_by_ref<rfft_type<T, F, fft_inv, 1>, ipp> (Domain<2>(8, 16));
#endif

#if VSIP_IMPL_CVSIP_FFT
  std::cout << "testing c->c fwd 0 by_ref cvsip...";
  fftm_by_ref<cfft_type<T, F, fft_fwd, 0>, cvsip>(Domain<2>(4, 16));
  std::cout << "testing c->c fwd 1 by_ref cvsip...";
  fftm_by_ref<cfft_type<T, F, fft_fwd, 1>, cvsip>(Domain<2>(4, 16));
  std::cout << "testing c->c inv 0 by_ref cvsip...";
  fftm_by_ref<cfft_type<T, F, fft_inv, 0>, cvsip>(Domain<2>(4, 16));
  std::cout << "testing c->c inv 1 by_ref cvsip...";
  fftm_by_ref<cfft_type<T, F, fft_inv, 1>, cvsip>(Domain<2>(4, 16));
  std::cout << "testing r->c fwd 0 by_ref cvsip...";
  fftm_by_ref<rfft_type<T, F, fft_fwd, 0>, cvsip>(Domain<2>(4, 16));
  std::cout << "testing r->c fwd 1 by_ref cvsip...";
  fftm_by_ref<rfft_type<T, F, fft_fwd, 1>, cvsip>(Domain<2>(4, 16));
  std::cout << "testing c->r inv 0 by_ref cvsip...";
  fftm_by_ref<rfft_type<T, F, fft_inv, 0>, cvsip> (Domain<2>(4, 16));
  std::cout << "testing c->r inv 1 by_ref cvsip...";
  fftm_by_ref<rfft_type<T, F, fft_inv, 1>, cvsip> (Domain<2>(4, 16));
#endif

#if VSIP_IMPL_CBE_SDK_FFT
  std::cout << "testing c->c fwd on cols by_ref cbe...";
  fftm_by_ref<cfft_type<T, F, fft_fwd, 0, col2_type>, cbe>(Domain<2>(32, 64));
  fftm_by_ref<cfft_type<T, F, fft_fwd, 0, col2_type>, cbe>(Domain<2>(Domain<1>(32), Domain<1>(0, 2, 32)));
  std::cout << "testing c->c fwd on rows by_ref cbe...";
  fftm_by_ref<cfft_type<T, F, fft_fwd, 1, row2_type>, cbe>(Domain<2>(32, 64));
  fftm_by_ref<cfft_type<T, F, fft_fwd, 1, row2_type>, cbe>(Domain<2>(Domain<1>(0, 2, 32), Domain<1>(64)));
  std::cout << "testing c->c inv 0 by_ref cbe...";
  fftm_by_ref<cfft_type<T, F, fft_inv, 0, col2_type>, cbe>(Domain<2>(32, 64));
  fftm_by_ref<cfft_type<T, F, fft_inv, 0, col2_type>, cbe>(Domain<2>(Domain<1>(32), Domain<1>(0, 2, 32)));
  std::cout << "testing c->c inv 1 by_ref cbe...";
  fftm_by_ref<cfft_type<T, F, fft_inv, 1, row2_type>, cbe>(Domain<2>(32, 64));
  fftm_by_ref<cfft_type<T, F, fft_inv, 1, row2_type>, cbe>(Domain<2>(Domain<1>(0, 2, 32), Domain<1>(64)));
#endif



#endif
}

int main(int argc, char **argv)
{
  vsipl init(argc, argv);

  std::cout << "testing interleaved float 1D fft" << std::endl;
  test_fft1d<float, inter>();
  std::cout << "testing split float 1D fft" << std::endl;
  test_fft1d<float, split>();
#if VSIP_IMPL_TEST_LEVEL > 0
  std::cout << "testing interleaved double 1D fft" << std::endl;
  test_fft1d<double, inter>();
  std::cout << "testing split double 1D fft" << std::endl;
  test_fft1d<double, split>();
//   std::cout << "testing interleaved long double 1D fft" << std::endl;
//   test_fft1d<long double, inter>();
//   std::cout << "testing split long double 1D fft" << std::endl;
//   test_fft1d<long double, split>();
#endif

  std::cout << "testing interleaved float 2D fft" << std::endl;
  test_fft2d<float, inter>();
  std::cout << "testing split float 2D fft" << std::endl;
  test_fft2d<float, split>();
#if VSIP_IMPL_TEST_LEVEL > 0
  std::cout << "testing interleaved double 2D fft" << std::endl;
  test_fft2d<double, inter>();
  std::cout << "testing split double 2D fft" << std::endl;
  test_fft2d<double, split>();
#endif

  std::cout << "testing interleaved float fftm" << std::endl;
  test_fftm<float, inter>();
  std::cout << "testing split float fftm" << std::endl;
  test_fftm<float, split>();
#if VSIP_IMPL_TEST_LEVEL > 0
  std::cout << "testing interleaved double fftm" << std::endl;
  test_fftm<double, inter>();
  std::cout << "testing split double fftm" << std::endl;
  test_fftm<double, split>();
#endif

#if VSIP_IMPL_TEST_LEVEL > 0
  std::cout << "testing interleaved float 3D fft" << std::endl;
  test_fft3d<float, inter>();
  std::cout << "testing split float 3D fft" << std::endl;
  test_fft3d<float, split>();
  std::cout << "testing interleaved double 3D fft" << std::endl;
  test_fft3d<double, inter>();
  std::cout << "testing split double 3D fft" << std::endl;
  test_fft3d<double, split>();
#endif
  return has_errors;
}
