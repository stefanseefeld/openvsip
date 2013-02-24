/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/ops_info.cpp
    @author  Jules Bergmann, Don McCoy
    @date    2005-07-11
    @brief   VSIPL++ Library: Operation counts for vector, matrix 
                              and signal-processing functions.

*/

#ifndef VSIP_CORE_OPS_INFO_HPP
#define VSIP_CORE_OPS_INFO_HPP

#include <complex>
#include <string>
#include <sstream>
#include <vsip/core/math_enum.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/length.hpp>
#include <vsip/core/metaprogramming.hpp>

namespace vsip
{
namespace impl
{

template <typename T>
struct Ops_info
{
  static unsigned int const div = 1;
  static unsigned int const sqr = 1;
  static unsigned int const mul = 1;
  static unsigned int const add = 1;
  static unsigned int const mag = 1;
};

template <typename T>
struct Ops_info<std::complex<T> >
{
  static unsigned int const div = 6 + 3 + 2; // mul + add + div
  static unsigned int const sqr = 2 + 1;     // mul + add
  static unsigned int const mul = 4 + 2;     // mul + add
  static unsigned int const add = 2;
  static unsigned int const mag = 2 + 1 + 1; // 2*mul + add + sqroot
};


template <typename T> 
struct Desc_datatype    { static char const * value() { return "I"; } };

#define VSIP_IMPL_DESC_DATATYPE(T, VALUE)		\
template <>					\
struct Desc_datatype<T> { static char const * value() { return VALUE; } };

VSIP_IMPL_DESC_DATATYPE(float,                "S");
VSIP_IMPL_DESC_DATATYPE(double,               "D");
VSIP_IMPL_DESC_DATATYPE(std::complex<float>,  "C");
VSIP_IMPL_DESC_DATATYPE(std::complex<double>, "Z");

#undef VSIP_IMPL_DESC_DATATYPE




namespace fft
{

template <typename I, typename O>
struct Op_count
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
struct Description
{ 
  static std::string tag(bool is_fftm, Domain<D> const &dom, int dir, 
    return_mechanism_type rm, dimension_type axis)
  {
    std::ostringstream   st;
    st << (is_fftm ? "Fftm " : "Fft ") 
       << (is_fftm ? (axis == 1 ? "row " : "col ") : "")
       << (dir == -1 ? "Inv " : "Fwd ")
       << Desc_datatype<I>::value() << "-"
       << Desc_datatype<O>::value() << " "
       << (rm == vsip::by_reference ? "by_ref " : "by_val ")
       << dom[0].size();
    if (D == 2)
       st << "x" << dom[1].size();

    return st.str();
  } 
};

} // namespace fft


namespace signal_detail
{

template <dimension_type D,
          typename T>
struct Op_count_conv
{ 
  static length_type value(Length<D> const& len_output, 
    Length<D> const& len_kernel)
  {
    length_type   M =  len_kernel[0];
    if (D == 2) M *= len_kernel[1];

    length_type   P =  len_output[0];
    if (D == 2) P *= len_output[1];

    return M * P * (Ops_info<T>::mul + Ops_info<T>::add);
  }
};


template <dimension_type D,
          typename T>
struct Op_count_corr : Op_count_conv<D, T>
{};


template <typename T>
struct Op_count_fir
{ 
  static length_type value(length_type order, length_type size, 
    length_type decimation)
  {
    return (Ops_info<T>::mul + Ops_info<T>::add) *
      ((order + 1) * size / decimation);
  } 
};

template <typename T>
struct Op_count_iir
{ 
  static length_type value(length_type input_size, length_type kernel_size)
  {
    return ( input_size * kernel_size *
      (5 * Ops_info<T>::mul + 4 * Ops_info<T>::add) );
  }
};


template <dimension_type D, typename T>
struct Description
{ 
  static std::string tag(const char* op, length_type size)
  {
    std::ostringstream   st;
    st << op << " " << Desc_datatype<T>::value() << " " << size;

    return st.str();
  } 

  static std::string tag(const char* op, Length<D> const& len_output, 
    Length<D> const& len_kernel)
  {
    std::ostringstream   st;
    st << op << " " 
       << D << "D "
       << Desc_datatype<T>::value() << " ";

    st << len_kernel[0];
    if (D == 2) 
      st << "x" << len_kernel[1] << " ";

    st << len_output[0];
    if (D == 2) 
      st << "x" << len_output[1];

    return st.str();
  } 
};

} // namespace signal_detail


namespace matvec
{
template <typename T>
struct Op_count_dot
{ 
  static length_type value(Length<1> const& len)
  {
    length_type count = len[0] * Ops_info<T>::mul;
    if ( len[0] > 1 )
      count += (len[0] - 1) * Ops_info<T>::add;
    return  count;
  } 
};

template <typename T>
struct Op_count_cvjdot
{ 
  static length_type value(Length<1> const& len)
  {
    length_type count = len[0] * Ops_info<T>::mul;
    if ( len[0] > 1 )
      count += (len[0] - 1) * Ops_info<T>::add;
    return  count;
  } 
};

template <typename T>
struct Op_count_cvjdot<std::complex<T> >
{ 
  static length_type value(Length<1> const& len)
  {
    // The conjugate of the second vector adds a scalar multiplication 
    // to the total.
    length_type count = len[0] * Ops_info<std::complex<T> >::mul +
      len[0] * Ops_info<T>::mul;
    if ( len[0] > 1 )
      count += (len[0] - 1) * Ops_info<std::complex<T> >::add;
    return  count;
  } 
};

template <typename CT>
struct Op_count_herm
{ 
  static length_type value(Length<2> const& len)
  {
    // The complex conjugate equals one scalar multiply
    typedef typename impl::scalar_of<CT>::type T;
    return len[0] * len[1] * Ops_info<T>::mul;
  } 
};
  
template <dimension_type D,
          typename T>
struct Op_count_kron
{ 
  static length_type value(Length<D> const& len_v, Length<D> const& len_w)
  {
    length_type r_size = len_v[0] * len_w[0];
    if ( D == 2 )
      r_size *= len_v[1] * len_w[1];

    return r_size * 2 * Ops_info<T>::mul;
  } 
};

template <typename T>
struct Op_count_outer
{ 
  static length_type value(Length<1> const& len_v, Length<1> const& len_w)
  {
    // Each element is scaled by alpha, resulting in the factor of 2.
    return len_v[0] * len_w[0] * 2 * Ops_info<T>::mul;
  } 
};

template <typename T>
struct Op_count_outer<std::complex<T> >
{ 
  static length_type value(Length<1> const& len_v, Length<1> const& len_w)
  {
    // The conjugate of the second vector is needed (once), adding a 
    // scalar multiplication to the total.
    return len_v[0] * len_w[0] * 2 * Ops_info<std::complex<T> >::mul +
      len_w[0] * Ops_info<T>::mul;
  } 
};

template <typename T>
struct Op_count_gemp
{ 
  static length_type value(Length<2> const& len_a, Length<2> const& len_b, 
    mat_op_type op_a, mat_op_type op_b)
  {
    length_type r_size = len_a[0] * len_b[1];
    Length<1> len_r(len_a[1]);

    length_type mul_ops = Op_count_dot<T>::value(len_r) * r_size;

    if ( op_a == mat_herm || op_a == mat_conj )
      mul_ops += Ops_info<scalar_of<T> >::mul * len_a[0] * len_a[1];
    if ( op_b == mat_herm || op_b == mat_conj )
      mul_ops += Ops_info<scalar_of<T> >::mul * len_b[0] * len_b[1];

    // C = alpha * OpA(A) * OpB(B) + beta * C
    return r_size * (2 * Ops_info<T>::mul + Ops_info<T>::add) + mul_ops;
  } 
};

template <typename T>
struct Op_count_gems
{ 
  static length_type value(Length<2> const& len_a, mat_op_type op_a)
  {
    length_type r_size = len_a[0] * len_a[1];

    length_type mat_ops = 0;
    if ( op_a == mat_herm || op_a == mat_conj )
      mat_ops += r_size * Ops_info<scalar_of<T> >::mul;

    // C = alpha * OpA(A) + beta * C
    return r_size * (2 * Ops_info<T>::mul + Ops_info<T>::add) + mat_ops;
  } 
};

template <dimension_type D,
          typename T>
struct Op_count_cumsum
{ 
  static length_type value(Length<D> const &len_v)
  {
    length_type adds = 0;
    if ( len_v[0] > 1 )
      adds += (len_v[0] - 1) * len_v[0] / 2;
    if ( D == 2 )
      if ( len_v[1] > 1 )
        adds += (len_v[1] - 1) * len_v[1] / 2;

    return adds * Ops_info<T>::add;
  } 
};

template <typename T>
struct Op_count_modulate
{ 
  static length_type value(Length<1> const& len_v)
  {
    // w(i) = v(i) * exp(0, i * nu + phi)
    typedef std::complex<scalar_of<T> >  CT;
    
    return len_v[0] * Ops_info<T>::mul + Ops_info<T>::add +
      Ops_info<CT>::mul + Ops_info<CT>::add;
  } 
};

  
template <typename T>
struct Description
{ 
  static std::string tag(const char* op, Length<1> const& len)
  {
    std::ostringstream   st;
    st << op << " " << Desc_datatype<T>::value() << " " << len[0];

    return st.str();
  } 

  static std::string tag(const char* op, Length<2> const& len)
  {
    std::ostringstream   st;
    st << op << " " << Desc_datatype<T>::value() << " " 
       << len[0] << "x" << len[1];

    return st.str();
  } 

  static std::string tag(const char* op, Length<1> const& len_v, 
    Length<1> const& len_w)
  {
    std::ostringstream   st;
    st << op << " " << Desc_datatype<T>::value() << " "
       << len_v[0] << " " << len_w[0];

    return st.str();
  } 

  static std::string tag(const char* op, Length<2> const& len_v, 
    Length<2> const& len_w)
  {
    std::ostringstream   st;
    st << op << " " << Desc_datatype<T>::value() << " "
       << len_v[0] << "x" << len_v[1] << " "
       << len_w[0] << "x" << len_w[1];

    return st.str();
  } 
};
} // namespace matvec


} // namespace impl
} // namespace vsip

#endif // VSIP_IMPL_OPS_INFO_HPP
