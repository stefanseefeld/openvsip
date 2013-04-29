/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/expr/ops_info.hpp
    @author  Jules Bergmann
    @date    2006-08-04
    @brief   VSIPL++ Library: Determine the number of ops per point for
                              an expression template.
*/

#ifndef VSIP_OPT_EXPR_OPS_INFO_HPP
#define VSIP_OPT_EXPR_OPS_INFO_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>

namespace vsip_csl
{
namespace expr
{
// Forward declaration
template <dimension_type D, typename Block0, typename Block1> class Vmmul;
}
}

namespace vsip
{
namespace impl
{

/// These generate char tags for given data types, defaulting to int
/// with specializations for common floating point types.  These use
/// BLAS/LAPACK convention.
template <typename T> 
struct Type_name    { static char const value = 'I'; };

#define VSIP_IMPL_TYPE_NAME(T, VALUE)		\
template <>					\
struct Type_name<T > { static char const value = VALUE; };

VSIP_IMPL_TYPE_NAME(float,                'S');
VSIP_IMPL_TYPE_NAME(double,               'D');
VSIP_IMPL_TYPE_NAME(std::complex<float>,  'C');
VSIP_IMPL_TYPE_NAME(std::complex<double>, 'Z');

#undef VSIP_IMPL_TYPE_NAME


template <typename T> 
struct Scalar_type_name    { static char const value = 'i'; };

#define VSIP_IMPL_SCALAR_TYPE_NAME(T, VALUE)	\
template <>					\
struct Scalar_type_name<T > { static char const value = VALUE; };

VSIP_IMPL_SCALAR_TYPE_NAME(float,                's');
VSIP_IMPL_SCALAR_TYPE_NAME(double,               'd');
VSIP_IMPL_SCALAR_TYPE_NAME(std::complex<float>,  'c');
VSIP_IMPL_SCALAR_TYPE_NAME(std::complex<double>, 'z');

#undef VSIP_IMPL_SCALAR_TYPE_NAME



/// Traits classes to determine the ops for a particular operation.
template <template <typename> class UnaryOp,
	  typename                  T1>
struct Unary_op_count
{
  static unsigned const value = 0;
}; 

template <template <typename, 
                    typename> class BinaryOp,
	  typename                  T1,
	  typename                  T2>
struct Binary_op_count
{
  static unsigned const value = 0;
}; 

template <template <typename, 
                    typename, 
                    typename> class TernaryOp,
	  typename                  T1,
	  typename                  T2,
	  typename                  T3>
struct Ternary_op_count
{
  static unsigned const value = 0;
}; 


/// Specializations for Unary types
#define VSIP_IMPL_UNARY_OPS_FUNCTOR(OP, TYPE, VALUE)	\
template <typename T>					\
struct Unary_op_count<expr::op::OP, TYPE>		\
{							\
  static unsigned const value = VALUE;			\
}; 

//VSIP_IMPL_UNARY_OPS_FUNCTOR(acos)
//VSIP_IMPL_UNARY_OPS_FUNCTOR(arg)
//VSIP_IMPL_UNARY_OPS_FUNCTOR(asin)
//VSIP_IMPL_UNARY_OPS_FUNCTOR(atan)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Bnot,  T,            1)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Ceil,  T,            1)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Conj,  complex<T>,   1)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Cos,   T,            1)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Cos,   complex<T>,  12)
//VSIP_IMPL_UNARY_OPS_FUNCTOR(cosh)
//VSIP_IMPL_UNARY_OPS_FUNCTOR(euler)
//VSIP_IMPL_UNARY_OPS_FUNCTOR(exp)
//VSIP_IMPL_UNARY_OPS_FUNCTOR(exp10)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Floor, T,            1)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Imag,  complex<T>,   0)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Lnot,  T,            1)
//VSIP_IMPL_UNARY_OPS_FUNCTOR(log)
//VSIP_IMPL_UNARY_OPS_FUNCTOR(log10)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Mag,   T,            0)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Mag,   complex<T>,  13)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Magsq, T,            3)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Neg,   T,            1)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Real,  complex<T>,   0)
//VSIP_IMPL_UNARY_OPS_FUNCTOR(recip)
//VSIP_IMPL_UNARY_OPS_FUNCTOR(rsqrt)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Sin,   T,            1)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Sin,   complex<T>,  12)
//VSIP_IMPL_UNARY_OPS_FUNCTOR(sinh)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Sq,    T,            1)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Sq,    complex<T>,   5)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Sqrt,  T,            1)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Sqrt,  complex<T>,  10)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Tan,   T,            1)
VSIP_IMPL_UNARY_OPS_FUNCTOR(Tan,   complex<T>,  14)
//VSIP_IMPL_UNARY_OPS_FUNCTOR(tanh)

#undef VSIP_IMPL_UNARY_OPS_FUNCTOR


/// Specializations for Binary types
#define VSIP_IMPL_BINARY_OPS(OP, TYPE1, TYPE2, VALUE)	\
template <typename T1,					\
          typename T2>					\
struct Binary_op_count<expr::op::OP, TYPE1, TYPE2>	\
{							\
  static unsigned const value = VALUE;			\
}; 

#define VSIP_IMPL_BINARY_OPS_FUNCTOR(OP, TYPE1, TYPE2, VALUE)	\
        VSIP_IMPL_BINARY_OPS(OP, TYPE1, TYPE2, VALUE)

VSIP_IMPL_BINARY_OPS(Add,  T1,          T2,          1)
VSIP_IMPL_BINARY_OPS(Add,  T1,          complex<T2>, 1)
VSIP_IMPL_BINARY_OPS(Add,  complex<T1>, T2,          1)
VSIP_IMPL_BINARY_OPS(Add,  complex<T1>, complex<T2>, 2)
VSIP_IMPL_BINARY_OPS(Sub,  T1,          T2,          1)
VSIP_IMPL_BINARY_OPS(Sub,  T1,          complex<T2>, 1)
VSIP_IMPL_BINARY_OPS(Sub,  complex<T1>, T2,          1)
VSIP_IMPL_BINARY_OPS(Sub,  complex<T1>, complex<T2>, 2)
VSIP_IMPL_BINARY_OPS(Mult, T1,          T2,          1)
VSIP_IMPL_BINARY_OPS(Mult, T1,          complex<T2>, 2)
VSIP_IMPL_BINARY_OPS(Mult, complex<T1>, T2,          2)
VSIP_IMPL_BINARY_OPS(Mult, complex<T1>, complex<T2>, 6)
VSIP_IMPL_BINARY_OPS(Div,  T1,          T2,          1)
VSIP_IMPL_BINARY_OPS(Div,  T1,          complex<T2>, 2)
VSIP_IMPL_BINARY_OPS(Div,  complex<T1>, T2,          2)
VSIP_IMPL_BINARY_OPS(Div,  complex<T1>, complex<T2>, 6)

//VSIP_IMPL_BINARY_OPS_FUNCTOR(atan2)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Band,    T1,          T2,           1)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Bor,     T1,          T2,           1)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Bxor,    T1,          T2,           1)
//VSIP_IMPL_BINARY_OPS_FUNCTOR(div)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Eq,      T1,          T2,           1)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Eq,      complex<T1>, complex<T2>,  2)
//VSIP_IMPL_BINARY_OPS_FUNCTOR(fmod)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Ge,      T1,          T2,           1)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Gt,      T1,          T2,           1)
//VSIP_IMPL_BINARY_OPS_FUNCTOR(hypot)
//VSIP_IMPL_BINARY_OPS_FUNCTOR(jmul)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Land,    T1,          T2,           1)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Le,      T1,          T2,           1)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Lt,      T1,          T2,           1)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Lor,     T1,          T2,           1)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Lxor,    T1,          T2,           1)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Max,     T1,          T2,           1)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Maxmg,   T1,          T2,           1)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Maxmg,   complex<T1>, complex<T2>, 27)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Maxmgsq, T1,          T2,           3)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Maxmgsq, complex<T1>, complex<T2>,  7)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Min,     T1,          T2,           1)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Minmg,   T1,          T2,           1)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Minmg,   complex<T1>, complex<T2>, 27)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Minmgsq, T1,          T2,           3)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Minmgsq, complex<T1>, complex<T2>,  7)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Ne,      T1,          T2,           1)
VSIP_IMPL_BINARY_OPS_FUNCTOR(Ne,      complex<T1>, complex<T2>,  2)
//VSIP_IMPL_BINARY_OPS_FUNCTOR(pow)
//VSIP_IMPL_BINARY_OPS_FUNCTOR(sub)

#undef VSIP_IMPL_BINARY_OPS_FUNCTOR
#undef VSIP_IMPL_BINARY_OPS



/// Specializations for Ternary types
#define VSIP_IMPL_TERNARY_OPS(OP, TYPE1, TYPE2, TYPE3, VALUE)	\
template <typename T1,						\
          typename T2,						\
          typename T3>						\
struct Ternary_op_count<expr::op::OP, TYPE1, TYPE2, TYPE3 >	\
{								\
  static unsigned const value = VALUE;				\
}; 

#define VSIP_IMPL_TERNARY_OPS_FUNCTOR(OP, TYPE1, TYPE2, TYPE3, VALUE)\
        VSIP_IMPL_TERNARY_OPS(OP, TYPE1, TYPE2, TYPE3, VALUE)

// Short synonym for above.
#define VSIP_IMPL_TOF(OP, T1, T2, T3, VALUE) \
    VSIP_IMPL_TERNARY_OPS_FUNCTOR(OP, T1, T2, T3, VALUE)

#define VSIP_IMPL_TERNARY_OPS_RRR(OP, VALUE) \
    VSIP_IMPL_TOF(OP, T1,          T2,          T3,          VALUE)
#define VSIP_IMPL_TERNARY_OPS_RRC(OP, VALUE) \
    VSIP_IMPL_TOF(OP, T1,          T2,          complex<T3>, VALUE)
#define VSIP_IMPL_TERNARY_OPS_RCR(OP, VALUE) \
    VSIP_IMPL_TOF(OP, T1,          complex<T2>, T3,          VALUE)
#define VSIP_IMPL_TERNARY_OPS_RCC(OP, VALUE) \
    VSIP_IMPL_TOF(OP, T1,          complex<T2>, complex<T3>, VALUE)
#define VSIP_IMPL_TERNARY_OPS_CRR(OP, VALUE) \
    VSIP_IMPL_TOF(OP, complex<T1>, T2,          T3,          VALUE)
#define VSIP_IMPL_TERNARY_OPS_CRC(OP, VALUE) \
    VSIP_IMPL_TOF(OP, complex<T1>, T2,          complex<T3>, VALUE)
#define VSIP_IMPL_TERNARY_OPS_CCR(OP, VALUE) \
    VSIP_IMPL_TOF(OP, complex<T1>, complex<T2>, T3,          VALUE)
#define VSIP_IMPL_TERNARY_OPS_CCC(OP, VALUE) \
    VSIP_IMPL_TOF(OP, complex<T1>, complex<T2>, complex<T3>, VALUE)


// The cost for ternary functions is computed by adding the costs for 
// pure real, mixed real-complex and pure complex adds and multiples 
// for the given equation:

//  (t1 + t2) * t3
//                            <  adds  >    <   muls   >
//                            R   M   C     R   M     C
VSIP_IMPL_TERNARY_OPS_RRR(Am, 1 + 0 + 0*2 + 1 + 0*2 + 0*6)
VSIP_IMPL_TERNARY_OPS_RRC(Am, 1 + 0 + 0*2 + 0 + 0*2 + 0*6)
VSIP_IMPL_TERNARY_OPS_RCR(Am, 0 + 1 + 0*2 + 0 + 1*2 + 0*6)
VSIP_IMPL_TERNARY_OPS_RCC(Am, 0 + 1 + 0*2 + 0 + 0*2 + 1*6)
VSIP_IMPL_TERNARY_OPS_CRR(Am, 0 + 1 + 0*2 + 0 + 1*2 + 0*6)
VSIP_IMPL_TERNARY_OPS_CRC(Am, 0 + 1 + 0*2 + 0 + 0*2 + 1*6)
VSIP_IMPL_TERNARY_OPS_CCR(Am, 0 + 0 + 1*2 + 0 + 1*2 + 0*6)
VSIP_IMPL_TERNARY_OPS_CCC(Am, 0 + 0 + 1*2 + 0 + 0*2 + 1*6)

//  t1 * t2 + (T1(1) - t1) * t3
//                                 <  adds  >    <   muls   >
//                                 R   M   C     R   M     C
VSIP_IMPL_TERNARY_OPS_RRR(Expoavg, 2 + 0 + 0*2 + 2 + 0*2 + 0*6)
VSIP_IMPL_TERNARY_OPS_RRC(Expoavg, 1 + 1 + 0*2 + 1 + 1*2 + 0*6)
VSIP_IMPL_TERNARY_OPS_RCR(Expoavg, 1 + 1 + 0*2 + 1 + 1*2 + 0*6)
VSIP_IMPL_TERNARY_OPS_RCC(Expoavg, 1 + 0 + 1*2 + 0 + 2*2 + 0*6)
VSIP_IMPL_TERNARY_OPS_CRR(Expoavg, 0 + 0 + 2*2 + 0 + 2*2 + 0*6)
VSIP_IMPL_TERNARY_OPS_CRC(Expoavg, 0 + 0 + 2*2 + 0 + 1*2 + 1*6)
VSIP_IMPL_TERNARY_OPS_CCR(Expoavg, 0 + 0 + 2*2 + 0 + 1*2 + 1*6)
VSIP_IMPL_TERNARY_OPS_CCC(Expoavg, 0 + 0 + 2*2 + 0 + 0*2 + 2*6)

//VSIP_IMPL_TERNARY_OPS_FUNCTOR(ma)
//VSIP_IMPL_TERNARY_OPS_FUNCTOR(msb)
//VSIP_IMPL_TERNARY_OPS_FUNCTOR(sbm)
//VSIP_IMPL_TERNARY_OPS_FUNCTOR(ite)

#undef VSIP_IMPL_TERNARY_OPS_RRR
#undef VSIP_IMPL_TERNARY_OPS_RRC
#undef VSIP_IMPL_TERNARY_OPS_RCR
#undef VSIP_IMPL_TERNARY_OPS_RCC
#undef VSIP_IMPL_TERNARY_OPS_CRR
#undef VSIP_IMPL_TERNARY_OPS_CRC
#undef VSIP_IMPL_TERNARY_OPS_CCR
#undef VSIP_IMPL_TERNARY_OPS_CCC
#undef VSIP_IMPL_TOF
#undef VSIP_IMPL_TERNARY_OPS_FUNCTOR
#undef VSIP_IMPL_TERNARY_OPS


/// Reduction to count the number operations per point of an expression.

struct Reduce_expr_ops_per_point
{
public:
  template <typename Block>
  struct leaf_node
  {
    typedef Int_type<0> type;
  };

  template <dimension_type D,
	    typename       T>
  struct leaf_node<expr::Scalar<D, T> >
  {
    typedef Int_type<0> type;
  };

  template <template <typename> class Operation,
	    typename Block,
	    typename T>
  struct unary_node
  {
    typedef Int_type<Unary_op_count<Operation, T>::value +
                     Block::value> type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename LType,
	    typename RBlock,
	    typename RType>
  struct binary_node
  {
    typedef Int_type<Binary_op_count<Operation, LType, RType>::value +
                     LBlock::value +
                     RBlock::value> type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Type1,
	    typename Block2,
	    typename Type2,
	    typename Block3,
	    typename Type3>
  struct ternary_node
  {
    typedef Int_type<Ternary_op_count<Operation, Type1, Type2, Type3>::value +
		     Block1::value +
		     Block2::value +
		     Block3::value> type;
  };

  template <typename Block>
  struct transform
  {
    typedef typename leaf_node<Block>::type type;
  };

  template <typename Block>
  struct transform<Block const> : public transform<Block> {};

  template <template <typename> class Operation,
	    typename Block>
  struct transform<expr::Unary<Operation, Block, true> >
  {
    typedef typename unary_node<Operation,
				typename transform<Block>::type,
				typename Block::value_type>::type type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  struct transform<expr::Binary<Operation, LBlock, RBlock, true> >
  {
    typedef typename binary_node<Operation,
				 typename transform<LBlock>::type,
				 typename LBlock::value_type,
				 typename transform<RBlock>::type,
				 typename RBlock::value_type>
				::type type;
  };

  template <dimension_type                Dim0,
	    typename                      LBlock,
	    typename                      RBlock>
  struct transform<expr::Vmmul<Dim0, LBlock, RBlock> >
  {
    typedef typename binary_node<expr::op::Mult,
				 typename transform<LBlock>::type,
				 typename LBlock::value_type,
				 typename transform<RBlock>::type,
				 typename RBlock::value_type>
              ::type type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct transform<expr::Ternary<Operation, Block1, Block2, Block3, true> >
  {
    typedef typename ternary_node<Operation,
				  typename transform<Block1>::type,
				  typename Block1::value_type,
				  typename transform<Block2>::type,
				  typename Block2::value_type,
				  typename transform<Block3>::type,
				  typename Block3::value_type>
				::type type;
  };
};






/// Reduction to generate a tag for the entire expression tree
struct Reduce_expr_op_name
{
public:

  template <typename BlockT>
  struct transform
  {
    static std::string tag() 
    {
      std::string st;
      st = Type_name<typename BlockT::value_type>::value;
      return st;
    }
  };

  template <typename BlockT>
  struct transform<BlockT const> : public transform<BlockT> 
  {};

  template <dimension_type            Dim,
	    typename                  T>
  struct transform<expr::Scalar<Dim, T> >
  {
    static std::string tag() 
    {
      std::string st;
      st = Scalar_type_name<T>::value;
      return st;
    }
  };

  template <template <typename> class O, typename B>
  struct transform<expr::Unary<O, B, true> >
  {
    static std::string tag()
    {
      return O<typename B::value_type>::name() + std::string("(") 
        + transform<B>::tag() + std::string(")");
    } 
  };

  template <template <typename, typename> class Op,
	    typename                      LBlock,
	    typename                      RBlock>
  struct transform<expr::Binary<Op, LBlock, RBlock, true> >
  {
    static std::string tag()
    {
      return Op<typename LBlock::value_type,
	        typename RBlock::value_type>::name() + std::string("(")
        + transform<LBlock>::tag() + std::string(",")
        + transform<RBlock>::tag() + std::string(")"); 
    } 
  };

  template <dimension_type                Dim0,
	    typename                      LBlock,
	    typename                      RBlock>
  struct transform<expr::Vmmul<Dim0, LBlock, RBlock> >
  {
    static std::string tag()
    {
      return std::string("vmmul") + std::string("(")
        + transform<LBlock>::tag() + std::string(",")
        + transform<RBlock>::tag() + std::string(")"); 
    } 
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct transform<expr::Ternary<Operation, Block1, Block2, Block3, true> >
  {
    static std::string tag()
    {
      return Operation<typename Block1::value_type,
	typename Block2::value_type,
	typename Block3::value_type>::name() + std::string("(")
        + transform<Block1>::tag() + std::string(",")
        + transform<Block2>::tag() + std::string(",")
        + transform<Block3>::tag() + std::string(")"); 

    } 
  };
};

} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace expr
{

/// This generates the total number of operations per point in a given 
/// expression.  It also computes the total number of points, as this
/// information is needed to calculate the total number of operations.
template <typename BlockT>
struct Ops_per_point
{
  static length_type size(BlockT const& src)
  {
    length_type size = src.size(BlockT::dim, 0);
    if ( BlockT::dim > 1 )
      size *= src.size(BlockT::dim, 1);
    if ( BlockT::dim > 2 )
      size *= src.size(BlockT::dim, 2);
    return size;
  }

  static unsigned const value =
    vsip::impl::Reduce_expr_ops_per_point::template transform<BlockT>::type::value;
};

/// This generates a tag for an expression in standard prefix notation
/// where the operator is shown, followed by the list of operands in
/// parenthesis.  The operator may be one of  the common binary 
/// operators +,-,\*, and / or simply the name of the function.  User-defined
/// expression evaluators will use one of 'unary', 'binary' or 'ternary' 
/// for the function name.  The operand will be one of the letters 'S', 
/// 'D', 'C', and 'Z' for views of those types (using the BLAS convention).
/// Scalar operands use lower-case equivalents of the same letters.
/// For example for matrices of single-precision values where A and B 
/// are real and C is complex:
///
///  * `A * B + C`    -->  `+(*(S,S),C)`
///  * `A * 5.f + C`  -->  `+(*(S,s),C)`
///  * `A * (B + C)`  -->  `*(S,+(S,C))`
template <typename Block>
std::string op_name(char const *prefix, Block const &block)
{
  std::ostringstream  tag;
  tag << prefix << ' ' << Block::dim << "D "
      << vsip::impl::Reduce_expr_op_name::template transform<Block>::tag() << " "
      << block.size(Block::dim, 0);
  if (Block::dim > 1)
    tag << "x" << block.size(Block::dim, 1);
  if (Block::dim > 2)
    tag << "x" << block.size(Block::dim, 2);
  
  return tag.str();
};


} // namespace vsip_csl::expr
} // namespace vsip_csl

#endif // VSIP_IMPL_EXPR_OPS_INFO_HPP
