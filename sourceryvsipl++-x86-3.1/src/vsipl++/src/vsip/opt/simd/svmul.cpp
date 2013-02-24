/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <vsip/opt/simd/svmul.hpp>

namespace vsip
{
namespace impl
{
namespace simd
{

// real * real
template <typename T>
void
svmul(T op1, T const* op2, T* res, length_type size)
{
  static bool const Is_vectorized =
    is_algorithm_supported<T, false, Alg_svmul>::value;
  Svmul<T, Is_vectorized>::exec(op1, op2, res, size);
}

template void svmul(float, float const*, float*, length_type);
template void svmul(double, double const*, double*, length_type);


// complex * real (interleaved result)
template <typename T>
void
svmul(std::complex<T> op1, T const* op2, std::complex<T>* res, length_type size)
{
  static bool const Is_vectorized =
    is_algorithm_supported<T, false, Alg_svmul>::value;
  Svmul<T, Is_vectorized>::exec(op1, op2, res, size);
}

template void svmul(std::complex<float>, 
                      float const*, std::complex<float>*, length_type);
template void svmul(std::complex<double>, 
                      double const*, std::complex<double>*, length_type);

// complex * real (split result)

template <typename T>
void
svmul(std::complex<T> op1, T const* op2, std::pair<T*, T*> res, length_type size)
{
  static bool const Is_vectorized =
    is_algorithm_supported<T, false, Alg_svmul>::value;
  Svmul<T, Is_vectorized>::exec(op1, op2, res, size);
}

template void svmul(std::complex<float>, 
                      float const*, std::pair<float*, float*>, length_type);
template void svmul(std::complex<double>, 
                      double const*, std::pair<double*, double*>, length_type);


// real * complex (interleaved)
template <typename T>
void
svmul(T op1, std::complex<T> const*op2, std::complex<T> *res, length_type size)
{
  static bool const Is_vectorized =
    is_algorithm_supported<T, false, Alg_svmul>::value;
  Svmul<std::complex<T>, Is_vectorized>::exec(op1, op2, res, size);
}

template void svmul(float, std::complex<float> const*,
		      std::complex<float>*, length_type);
template void svmul(double, std::complex<double> const*,
		      std::complex<double>*, length_type);


// real * complex (split)
template <typename T>
void
svmul(T op1, std::pair<T const*,T const*> op2, std::pair<T*,T*> res, length_type size)
{
  static bool const Is_vectorized =
    is_algorithm_supported<T, true, Alg_svmul>::value;
  Svmul<std::pair<T,T>, Is_vectorized>::exec(op1, op2, res, size);
}

template void svmul(float,
		      std::pair<float const*,float const*>,
		      std::pair<float*,float*>, length_type);
template void svmul(double,
		      std::pair<double const*,double const*>,
		      std::pair<double*,double*>, length_type);


// complex * complex (interleaved)
template <typename T>
void
svmul(std::complex<T> op1, std::complex<T> const*op2, std::complex<T> *res, length_type size)
{
  static bool const Is_vectorized =
    is_algorithm_supported<T, false, Alg_svmul>::value;
  Svmul<std::complex<T>, Is_vectorized>::exec(op1, op2, res, size);
}

template void svmul(std::complex<float>, std::complex<float> const*,
		      std::complex<float>*, length_type);
template void svmul(std::complex<double>, std::complex<double> const*,
		      std::complex<double>*, length_type);

// complex * complex (split)
template <typename T>
void
svmul(std::complex<T> const op1, std::pair<T const*,T const*> op2, std::pair<T*,T*> res, length_type size)
{
  static bool const Is_vectorized =
    is_algorithm_supported<T, true, Alg_svmul>::value;
  Svmul<T, Is_vectorized>::exec(op1, op2, res, size);
}

template void svmul(std::complex<float>,
		      std::pair<float const*,float const*>,
		      std::pair<float*,float*>, length_type);
template void svmul(std::complex<double>,
		      std::pair<double const*,double const*>,
		      std::pair<double*,double*>, length_type);


} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip
