/* Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Parallel Iterator call operator support

#ifndef vsip_csl_pi_call_hpp_
#define vsip_csl_pi_call_hpp_

#include <vsip_csl/pi/iterator.hpp>
#include <cassert>
#include <vsip/core/subblock.hpp>
#include <vsip/opt/dispatch.hpp>
#include <vsip/core/dispatch_tags.hpp>

namespace vsip_csl
{
namespace dispatcher
{
namespace op
{
struct pi_assign;
}

template <>
struct List<op::pi_assign>
{
  typedef Make_type_list<be::user,
			 be::cuda,
			 be::intel_ipp,
			 be::cml,
			 be::mercury_sal,
			 be::cbe_sdk,
			 be::opt,
			 be::generic>::type type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip_csl
{
namespace pi
{

template <typename Call, typename RHS>
Call &assign(Call &call, RHS const &rhs)
{
  vsip_csl::dispatch<dispatcher::op::pi_assign, void, Call &, RHS const &>(call, rhs);
  return call;
}

template <typename B, typename I, typename J, typename K>
class Call
{
public:
  typedef B block_type;
  typedef typename B::value_type result_type;

  Call(block_type &b, I const &i, J const &j, K const &k)
    : block_(b), i_(i), j_(j), k_(k) {}
  Call &operator= (Call const &c) { return assign(*this, c);}
  template <typename RHS>
  Call &operator= (RHS const &rhs) { return assign(*this, rhs);}
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  I const &i() const { return i_;}
  J const &j() const { return j_;}
  K const &k() const { return k_;}

  result_type apply(index_type i, index_type j, index_type k) const
  { return block_.get(i, j, k);}

private:
  block_type &block_;
  I const &i_;
  J const &j_;
  K const &k_;
};

/// Specialization for a 1D block call operation.
/// The result-type is the block's value-type.
template <typename B, typename I>
class Call<B, I>
{
public:
  typedef B block_type;
  typedef typename B::value_type result_type;

  Call(block_type &b, I const &i) : block_(b), i_(i) {}
  Call &operator= (Call const &c) { return assign(*this, c);}
  template <typename RHS>
  Call &operator= (RHS const &rhs) { return assign(*this, rhs);}
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  I const &i() const { return i_;}

  result_type apply(index_type i) const
  { return block_.get(i);}

private:
  block_type &block_;
  I const &i_;
};

/// Specialization for a 2D block call operation.
/// The result-type is the block's value-type.
template <typename B, typename I, typename J>
class Call<B, I, J>
{
public:
  typedef B block_type;
  typedef typename B::value_type result_type;

  Call(block_type &b, I const &i, J const &j) : block_(b), i_(i), j_(j) {}
  Call &operator= (Call const &c) { return assign(*this, c);}
  template <typename RHS>
  Call &operator= (RHS const &rhs) { return assign(*this, rhs);}
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  I const &i() const { return i_;}
  J const &j() const { return j_;}

  result_type apply(index_type i, index_type j) const
  { return block_.get(i, j);}

private:
  block_type &block_;
  I const &i_;
  J const &j_;
};

/// Specialization for a 2D block call operation.
/// The result-type is a subblock type.
template <typename B, typename I>
class Call<B, I, whole_domain_type>
{
public:
  typedef B block_type;
  typedef impl::Sliced_block<B, 0> result_type;

  Call(block_type &b, I const &i) : block_(b), i_(i) {}
  Call &operator= (Call const &c) { return assign(*this, c);}
  template <typename RHS>
  Call &operator= (RHS const &rhs) { return assign(*this, rhs);} 
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  I const &i() const { return i_;}

  result_type apply(index_type i) const
  { return result_type(block_, i);}

private:
  block_type &block_;
  I const &i_;
};

/// Specialization for a 2D block call operation.
/// The result-type is a subblock type.
template <typename B, typename J>
class Call<B, whole_domain_type, J>
{
public:
  typedef B block_type;
  typedef impl::Sliced_block<B, 1> result_type;

  Call(block_type &b, J const &j) : block_(b), j_(j) {}
  Call &operator= (Call const &c) { return assign(*this, c);}
  template <typename RHS>
  Call &operator= (RHS const &rhs) { return assign(*this, rhs);}
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  J const &j() const { return j_;}

  result_type apply(index_type j) const
  { return result_type(block_, j);}

private:
  block_type &block_;
  J const &j_;
};

/// Specialization for a 3D block call operation.
/// The result-type is a 2D subblock type.
template <typename B, typename I>
class Call<B, I, whole_domain_type, whole_domain_type>
{
public:
  typedef B block_type;
  typedef impl::Sliced_block<B, 0> result_type;

  Call(block_type &b, I const &i) : block_(b), i_(i) {}
  Call &operator= (Call const &c) { return assign(*this, c);}
  template <typename RHS>
  Call &operator= (RHS const &rhs) { return assign(*this, rhs);}
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  I const &i() const { return i_;}

  result_type apply(index_type i) const
  { return result_type(block_, i);}

private:
  block_type &block_;
  I const &i_;
};

/// Specialization for a 3D block call operation.
/// The result-type is a 2D subblock type.
template <typename B, typename J>
class Call<B, whole_domain_type, J, whole_domain_type>
{
public:
  typedef B block_type;
  typedef impl::Sliced_block<B, 1> result_type;

  Call(block_type &b, J const &j) : block_(b), j_(j) {}
  Call &operator= (Call const &c) { return assign(*this, c);}
  template <typename RHS>
  Call &operator= (RHS const &rhs) { return assign(*this, rhs);}
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  J const &j() const { return j_;}

  result_type apply(index_type j) const
  { return result_type(block_, j);}

private:
  block_type &block_;
  J const &j_;
};

/// Specialization for a 3D block call operation.
/// The result-type is a 2D subblock type.
template <typename B, typename K>
class Call<B, whole_domain_type, whole_domain_type, K>
{
public:
  typedef B block_type;
  typedef impl::Sliced_block<B, 2> result_type;

  Call(block_type &b, K const &k) : block_(b), k_(k) {}
  Call &operator= (Call const &c) { return assign(*this, c);}
  template <typename RHS>
  Call &operator= (RHS const &rhs) { return assign(*this, rhs);}
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  K const &k() const { return k_;}

  result_type apply(index_type k) const
  { return result_type(block_, k);}

private:
  block_type &block_;
  K const &k_;
};

/// Specialization for a 3D block call operation.
/// The result-type is a 1D subblock type.
template <typename B, typename I, typename J>
class Call<B, I, J, whole_domain_type>
{
public:
  typedef B block_type;
  typedef impl::Sliced2_block<B, 0, 1> result_type;

  Call(block_type &b, I const &i, J const &j) : block_(b), i_(i), j_(j) {}
  Call &operator= (Call const &c) { return assign(*this, c);}
  template <typename RHS>
  Call &operator= (RHS const &rhs) { return assign(*this, rhs);}
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  I const &i() const { return i_;}
  J const &j() const { return j_;}

  result_type apply(index_type i, index_type j) const
  { return result_type(block_, i, j);}

private:
  block_type &block_;
  I const &i_;
  J const &j_;
};

/// Specialization for a 3D block call operation.
/// The result-type is a 1D subblock type.
template <typename B, typename I, typename K>
class Call<B, I, whole_domain_type, K>
{
public:
  typedef B block_type;
  typedef impl::Sliced2_block<B, 0, 2> result_type;

  Call(block_type &b, I const &i, K const &k) : block_(b), i_(i), k_(k) {}
  Call &operator= (Call const &c) { return assign(*this, c);}
  template <typename RHS>
  Call &operator= (RHS const &rhs) { return assign(*this, rhs);}
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  I const &i() const { return i_;}
  K const &k() const { return k_;}

  result_type apply(index_type i, index_type k) const
  { return result_type(block_, i, k);}

private:
  block_type &block_;
  I const &i_;
  K const &k_;
};

/// Specialization for a 3D block call operation.
/// The result-type is a 1D subblock type.
template <typename B, typename J, typename K>
class Call<B, whole_domain_type, J, K>
{
public:
  typedef B block_type;
  typedef impl::Sliced2_block<B, 1, 2> result_type;

  Call(block_type &b, J const &j, K const &k) : block_(b), j_(j), k_(k) {}
  Call &operator= (Call const &c) { return assign(*this, c);}
  template <typename RHS>
  Call &operator= (RHS const &rhs) { return assign(*this, rhs);}
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  J const &j() const { return j_;}
  K const &k() const { return k_;}

  result_type apply(index_type j, index_type k) const
  { return result_type(block_, j, k);}

private:
  block_type &block_;
  J const &j_;
  K const &k_;
};

template <typename B, typename I, typename J, typename K>
struct is_expr<Call<B, I, J, K> > { static bool const value = true;};

template <typename T>
struct is_call { static bool const value = false;};

template <typename B, typename I, typename J, typename K>
struct is_call<Call<B, I, J, K> > { static bool const value = true;};

} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif
