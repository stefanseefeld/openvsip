/* Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Parallel Iterator call operator support

#ifndef vsip_csl_pi_call_hpp_
#define vsip_csl_pi_call_hpp_

#include <vsip_csl/pi/iterator.hpp>
#include <cassert>
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
			 be::opt,
			 be::generic>::type type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip_csl
{
namespace pi
{

template <typename B, typename I, typename J, typename K, typename RHS>
void assign(Call<B, I, J, K> &lhs, RHS const &rhs);

template <typename B, typename I>
class Call<B, I, void, void>
{
public:
  typedef B block_type;

  Call(block_type &b, I const &i) : block_(b), i_(i) {}
  Call &operator= (Call const &rhs) 
  {
    dispatch<dispatcher::op::pi_assign, void, Call &, Call const &>(*this, rhs);
    return *this;
  }
  template <typename RHS>
  Call &operator= (RHS const &rhs) 
  {
    dispatch<dispatcher::op::pi_assign, void, Call &, RHS const &>(*this, rhs);
    return *this;
  }
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  I const &i() const { return i_;}

private:
  block_type &block_;
  I const &i_;
};

template <typename B, typename J>
class Call<B, void, J, void>
{
public:
  typedef B block_type;

  Call(block_type &b, J const &j) : block_(b), j_(j) {}
  Call &operator= (Call const &rhs) 
  {
    dispatch<dispatcher::op::pi_assign, void, Call &, Call const &>(*this, rhs);
    return *this;
  }
  template <typename RHS>
  Call &operator= (RHS const &rhs) 
  {
    dispatch<dispatcher::op::pi_assign, void, Call &, RHS const &>(*this, rhs);
    return *this;
  }
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  J const &j() const { return j_;}

private:
  block_type &block_;
  J const &j_;
};

template <typename B, typename K>
class Call<B, void, void, K>
{
public:
  typedef B block_type;

  Call(block_type &b, K const &k) : block_(b), k_(k) {}
  Call &operator= (Call const &rhs) 
  {
    dispatch<dispatcher::op::pi_assign, void, Call &, Call const &>(*this, rhs);
    return *this;
  }
  template <typename RHS>
  Call &operator= (RHS const &rhs) 
  {
    dispatch<dispatcher::op::pi_assign, void, Call &, RHS const &>(*this, rhs);
    return *this;
  }
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  K const &k() const { return k_;}

private:
  block_type &block_;
  K const &k_;
};

template <typename B, typename I, typename J>
class Call<B, I, J, void>
{
public:
  typedef B block_type;

  Call(block_type &b, I const &i, J const &j) : block_(b), i_(i), j_(j) {}
  Call &operator= (Call const &rhs) 
  {
    dispatch<dispatcher::op::pi_assign, void, Call &, Call const &>(*this, rhs);
    return *this;
  }
  template <typename RHS>
  Call &operator= (RHS const &rhs) 
  {
    dispatch<dispatcher::op::pi_assign, void, Call &, RHS const &>(*this, rhs);
    return *this;
  }
  block_type &block() { return block_;}
  block_type const &block() const { return block_;}
  I const &i() const { return i_;}
  J const &j() const { return j_;}

private:
  block_type &block_;
  I const &i_;
  J const &j_;
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
