//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
//
// This file is made available under the BSD License.
// See the accompanying file LICENSE.BSD for details.

#ifndef ovxx_ops_count_hpp_
#define ovxx_ops_count_hpp_

#include <vsip/impl/math_enum.hpp>
#include <vsip/complex.hpp>
#include <vsip/domain.hpp>
#include <ovxx/length.hpp>
#include <string>
#include <sstream>

namespace ovxx
{
namespace ops_count
{

template <typename T>
struct traits
{
  static unsigned int const div = 1;
  static unsigned int const sqr = 1;
  static unsigned int const mul = 1;
  static unsigned int const add = 1;
  static unsigned int const mag = 1;
};

template <typename T>
struct traits<std::complex<T> >
{
  static unsigned int const div = 6 + 3 + 2; // mul + add + div
  static unsigned int const sqr = 2 + 1;     // mul + add
  static unsigned int const mul = 4 + 2;     // mul + add
  static unsigned int const add = 2;
  static unsigned int const mag = 2 + 1 + 1; // 2*mul + add + sqroot
};


template <typename T> 
struct datatype { static char const *value() { return "I";}};

#define OVXX_DATATYPE(T, VALUE)		\
template <>					\
struct datatype<T> { static char const *value() { return VALUE;}};

OVXX_DATATYPE(float,                "S");
OVXX_DATATYPE(double,               "D");
OVXX_DATATYPE(std::complex<float>,  "C");
OVXX_DATATYPE(std::complex<double>, "Z");

#undef OVXX_DATATYPE

namespace signal
{

template <dimension_type D, typename T>
struct conv
{ 
  static length_type value(Length<D> const& len_output, 
			   Length<D> const& len_kernel)
  {
    length_type   M =  len_kernel[0];
    if (D == 2) M *= len_kernel[1];

    length_type   P =  len_output[0];
    if (D == 2) P *= len_output[1];

    return M * P * (traits<T>::mul + traits<T>::add);
  }
};

template <dimension_type D, typename T>
struct corr : conv<D, T> {};

template <typename T>
struct fir
{ 
  static length_type value(length_type order, length_type size, 
    length_type decimation)
  {
    return (traits<T>::mul + traits<T>::add) *
      ((order + 1) * size / decimation);
  } 
};

template <typename T>
struct iir
{ 
  static length_type value(length_type input_size, length_type kernel_size)
  {
    return ( input_size * kernel_size *
      (5 * traits<T>::mul + 4 * traits<T>::add) );
  }
};

template <dimension_type D, typename T>
struct description
{ 
  static std::string tag(const char* op, length_type size)
  {
    std::ostringstream   st;
    st << op << " " << datatype<T>::value() << " " << size;
    return st.str();
  } 

  static std::string tag(char const *op, 
			 Length<D> const& len_output, 
			 Length<D> const& len_kernel)
  {
    std::ostringstream   st;
    st << op << " " 
       << D << "D "
       << datatype<T>::value() << " ";

    st << len_kernel[0];
    if (D == 2) 
      st << "x" << len_kernel[1] << " ";

    st << len_output[0];
    if (D == 2) 
      st << "x" << len_output[1];

    return st.str();
  } 
};

} // namespace ovxx::ops_count::signal

namespace matvec
{

template <typename T>
struct dot
{ 
  static length_type value(Length<1> const& len)
  {
    length_type count = len[0] * traits<T>::mul;
    if ( len[0] > 1 )
      count += (len[0] - 1) * traits<T>::add;
    return  count;
  } 
};

template <typename T>
struct cvjdot
{ 
  static length_type value(Length<1> const& len)
  {
    length_type count = len[0] * traits<T>::mul;
    if ( len[0] > 1 )
      count += (len[0] - 1) * traits<T>::add;
    return  count;
  } 
};

template <typename T>
struct cvjdot<std::complex<T> >
{ 
  static length_type value(Length<1> const& len)
  {
    // The conjugate of the second vector adds a scalar multiplication 
    // to the total.
    length_type count = len[0] * traits<std::complex<T> >::mul +
      len[0] * traits<T>::mul;
    if ( len[0] > 1 )
      count += (len[0] - 1) * traits<std::complex<T> >::add;
    return  count;
  } 
};

template <typename CT>
struct herm
{ 
  static length_type value(Length<2> const& len)
  {
    // The complex conjugate equals one scalar multiply
    typedef typename scalar_of<CT>::type T;
    return len[0] * len[1] * traits<T>::mul;
  } 
};
  
template <dimension_type D,
          typename T>
struct kron
{ 
  static length_type value(Length<D> const& len_v, 
			   Length<D> const& len_w)
  {
    length_type r_size = len_v[0] * len_w[0];
    if ( D == 2 )
      r_size *= len_v[1] * len_w[1];

    return r_size * 2 * traits<T>::mul;
  } 
};

template <typename T>
struct outer
{ 
  static length_type value(Length<1> const& len_v,
			   Length<1> const& len_w)
  {
    // Each element is scaled by alpha, resulting in the factor of 2.
    return len_v[0] * len_w[0] * 2 * traits<T>::mul;
  } 
};

template <typename T>
struct outer<std::complex<T> >
{ 
  static length_type value(Length<1> const& len_v,
			   Length<1> const& len_w)
  {
    // The conjugate of the second vector is needed (once), adding a 
    // scalar multiplication to the total.
    return len_v[0] * len_w[0] * 2 * traits<std::complex<T> >::mul +
      len_w[0] * traits<T>::mul;
  } 
};

template <typename T>
struct gemp
{ 
  static length_type value(Length<2> const& len_a,
			   Length<2> const& len_b, 
			   mat_op_type op_a, mat_op_type op_b)
  {
    length_type r_size = len_a[0] * len_b[1];
    Length<1> len_r(len_a[1]);

    length_type mul_ops = dot<T>::value(len_r) * r_size;

    if ( op_a == mat_herm || op_a == mat_conj )
      mul_ops += traits<scalar_of<T> >::mul * len_a[0] * len_a[1];
    if ( op_b == mat_herm || op_b == mat_conj )
      mul_ops += traits<scalar_of<T> >::mul * len_b[0] * len_b[1];

    // C = alpha * OpA(A) * OpB(B) + beta * C
    return r_size * (2 * traits<T>::mul + traits<T>::add) + mul_ops;
  } 
};

template <typename T>
struct gems
{ 
  static length_type value(Length<2> const& len_a, mat_op_type op_a)
  {
    length_type r_size = len_a[0] * len_a[1];

    length_type mat_ops = 0;
    if ( op_a == mat_herm || op_a == mat_conj )
      mat_ops += r_size * traits<scalar_of<T> >::mul;

    // C = alpha * OpA(A) + beta * C
    return r_size * (2 * traits<T>::mul + traits<T>::add) + mat_ops;
  } 
};

} // namespace ovxx::ops_count::matvec

template <dimension_type D, typename T>
struct cumsum
{ 
  static length_type value(Length<D> const &len_v)
  {
    length_type adds = 0;
    if ( len_v[0] > 1 )
      adds += (len_v[0] - 1) * len_v[0] / 2;
    if ( D == 2 )
      if ( len_v[1] > 1 )
        adds += (len_v[1] - 1) * len_v[1] / 2;

    return adds * traits<T>::add;
  } 
};

template <typename T>
struct modulate
{ 
  static length_type value(Length<1> const& len_v)
  {
    // w(i) = v(i) * exp(0, i * nu + phi)
    typedef std::complex<scalar_of<T> >  CT;
    
    return len_v[0] * traits<T>::mul + traits<T>::add +
      traits<CT>::mul + traits<CT>::add;
  } 
};

  
template <typename T>
struct description
{ 
  static std::string tag(char const *op, Length<1> const& len)
  {
    std::ostringstream   st;
    st << op << " " << datatype<T>::value() << " " << len[0];

    return st.str();
  } 

  static std::string tag(char const *op, ovxx::Length<2> const& len)
  {
    std::ostringstream   st;
    st << op << " " << datatype<T>::value() << " " 
       << len[0] << "x" << len[1];

    return st.str();
  } 

  static std::string tag(char const *op,
			 Length<1> const& len_v,
			 Length<1> const& len_w)
  {
    std::ostringstream   st;
    st << op << " " << datatype<T>::value() << " "
       << len_v[0] << " " << len_w[0];

    return st.str();
  } 

  static std::string tag(char const *op,
			 Length<2> const& len_v, 
			 Length<2> const& len_w)
  {
    std::ostringstream   st;
    st << op << " " << datatype<T>::value() << " "
       << len_v[0] << "x" << len_v[1] << " "
       << len_w[0] << "x" << len_w[1];

    return st.str();
  } 
};

template <typename I, typename O>
struct fft
{ 
  static length_type value(length_type len)
  { 
    length_type ops = static_cast<length_type>(
      5 * len * std::log((float)len) / std::log(2.f));
    if (sizeof(I) != sizeof(O)) ops /= 2;
    return ops;
  }
};

template <int D, typename I, typename O>
struct fft_description
{ 
  static std::string tag(bool is_fftm, Domain<D> const &dom, int dir, 
    return_mechanism_type rm, dimension_type axis)
  {
    std::ostringstream   st;
    st << (is_fftm ? "Fftm " : "Fft ") 
       << (is_fftm ? (axis == 1 ? "row " : "col ") : "")
       << (dir == -1 ? "Inv " : "Fwd ")
       << datatype<I>::value() << "-"
       << datatype<O>::value() << " "
       << (rm == vsip::by_reference ? "by_ref " : "by_val ")
       << dom[0].size();
    if (D == 2)
       st << "x" << dom[1].size();

    return st.str();
  } 
};

} // namespace ovxx::ops_count
} // namespace ovxx

#endif
