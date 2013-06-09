//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_output_hpp_
#define ovxx_output_hpp_

#include <ovxx/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <ovxx/type_name.hpp>
#include <ovxx/domain_utils.hpp>
#include <ostream>
#include <sstream>

namespace ovxx
{
inline
std::ostream &
operator<<(std::ostream &out, Domain<1> const &dom)
{
  out << '('
      << dom.first() << ", "
      << dom.stride() << ", "
      << dom.length() << ')';
  return out;
}

inline
std::ostream &
operator<<(std::ostream &out, Domain<2> const &dom)
{
  return out << '(' << dom[0] << ", " << dom[1] << ')';
}

inline
std::ostream &
operator<<(std::ostream &out, Domain<3> const &dom)
{
  return out << '(' << dom[0] << ", " << dom[1] << ", " << dom[2] << ')';
}

template <dimension_type D>
inline
std::ostream &
operator<<(std::ostream & out, Index<D> const &idx)
{
  out << '(';
  for (dimension_type d = 0; d != D; ++d)
  {
    if (d > 0) out << ", ";
    out << idx[d];
  }
  return out << ')';
}

template <dimension_type D>
inline
std::ostream &
operator<<(std::ostream &out, Length<D> const &l)
{
  out << '(';
  for (dimension_type d = 0; d != D; ++d)
  {
    if (d > 0) out << ", ";
    out << l[d];
  }
  return out << ')';
}

template <typename T, typename B>
inline
std::ostream &
operator<<(std::ostream &out, const_Vector<T, B> v)
{
  for (index_type i = 0; i != v.size(); ++i)
    out << i << ": " << v.get(i) << '\n';
  return out;
}

template <typename T, typename B>
inline
std::ostream &
operator<<(std::ostream &out, const_Matrix<T, B> m)
{
  for (index_type r = 0; r != m.size(0); ++r)
  {
    out << r << ": ";
    for (index_type c = 0; c != m.size(1); ++c)
      out << "  " << m.get(r, c);
    out << std::endl;
  }
  return out;
}

template <typename T, typename B>
inline
std::ostream &
operator<<(std::ostream & out, const_Tensor<T, B> t)
{
  for (index_type z = 0; z != t.size(0); ++z)
  {
    out << "plane " << z << ":\n";
    for (index_type r = 0; r != t.size(1); ++r)
    {
      out << r << ": ";
      for (index_type c = 0; c != t.size(2); ++c)
        out << t.get(z, r, c);
      out << std::endl;
    }
    out << std::endl;
  }
  return out;
}

} // namespace ovxx

#endif
