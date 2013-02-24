/* Copyright (c) 2009 by CodeSourcery, Inc.  All rights reserved. */

/** @file    vsip_csl/math.hpp
    @author  Mike LeBlanc
    @date    2009-04-30
    @brief   VSIPL++ Library: sumval - alternate form.

*/

#ifndef VSIP_CSL_MATH_HPP
#define VSIP_CSL_MATH_HPP

/***********************************************************************
  Included Files
***********************************************************************/

namespace vsip_csl
{

template <typename                            T,  
          template <typename, typename> class ViewT,
          typename                            BlockT,
          typename                            ResultT>
ResultT
sumval(ViewT<T, BlockT> v, ResultT)
{
  using namespace vsip::impl;
  return
    reduce<Sum_value_helper<ResultT>::template Sum_value>(v);
}

template <typename                            T,  
          template <typename, typename> class ViewT,
          typename                            BlockT,
          typename                            ResultT>
ResultT
sumsqval(ViewT<T, BlockT> v, ResultT)
{
  using namespace vsip::impl;
  return
    reduce<Sum_sq_value_helper<ResultT>::template Sum_sq_value>(v);
}

template <typename                            T,  
          template <typename, typename> class ViewT,
          typename                            BlockT,
          typename                            ResultT>
ResultT
meanval(ViewT<T, BlockT> v, ResultT)
{
  using namespace vsip::impl;
  return
    reduce<Mean_value_helper<ResultT>::template Mean_value>(v);
}

template <typename                            T,  
          template <typename, typename> class ViewT,
          typename                            BlockT,
          typename                            ResultT>
ResultT
meansqval(ViewT<T, BlockT> v, ResultT)
{
  using namespace vsip::impl;
  return
    reduce<Mean_magsq_value_helper<ResultT>::template Mean_magsq_value>(v);
}

template <typename                            T,  
          template <typename, typename> class ViewT,
          typename                            BlockT>
inline
T
maxval(ViewT<T, BlockT> v)
{
  Index<ViewT<T, BlockT>::dim> idx;
  return maxval(v, idx);
}

template <typename                            T,  
          template <typename, typename> class ViewT,
          typename                            BlockT>
inline
T
minval(ViewT<T, BlockT> v)
{
  Index<ViewT<T, BlockT>::dim> idx;
  return minval(v, idx);
}

template <typename                            T,  
          template <typename, typename> class ViewT,
          typename                            BlockT>
inline
typename vsip::impl::Scalar_of<T>::type
maxmgval(ViewT<T, BlockT> v)
{
  Index<ViewT<T, BlockT>::dim> idx;
  return maxmgval(v, idx);
}

template <typename                            T,  
          template <typename, typename> class ViewT,
          typename                            BlockT>
inline
typename vsip::impl::Scalar_of<T>::type
minmgval(ViewT<T, BlockT> v)
{
  Index<ViewT<T, BlockT>::dim> idx;
  return minmgval(v, idx);
}

template <typename                            T,  
          template <typename, typename> class ViewT,
          typename                            BlockT>
inline
T
maxmgsqval(ViewT<std::complex<T>, BlockT> v)
{
  Index<ViewT<T, BlockT>::dim> idx;
  return maxmgsqval(v, idx);
}

template <typename                            T,
          template <typename, typename> class ViewT,
          typename                            BlockT>
inline
T
minmgsqval(ViewT<std::complex<T>, BlockT> v)
{
  Index<ViewT<T, BlockT>::dim> idx;
  return minmgsqval(v, idx);
}

} // namespace vsip_csl

#endif
