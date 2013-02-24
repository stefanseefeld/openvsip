/* Copyright (c) 2005, 2008, 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip_csl/assignment.hpp
    @author  Jules Bergmann, Stefan Seefeld
    @date    2005-08-26
    @brief   VSIPL++ Library: Early binding of an assignment.

*/

#ifndef VSIP_CSL_ASSIGNMENT_HPP
#define VSIP_CSL_ASSIGNMENT_HPP

#include <vsip/core/noncopyable.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/parallel/expr.hpp>
#include <vsip/core/parallel/assign_chain.hpp>
#include <vsip/core/assign.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/profile.hpp>

namespace vsip_csl
{

class Assignment : impl::Non_copyable
{
public:
   template <template <typename, typename> class View1,
	     typename                            T1,
	     typename                            Block1,
	     template <typename, typename> class View2,
	     typename                            T2,
	     typename                            Block2>
  Assignment(View1<T1, Block1> dst,
	     View2<T2, Block2> src)
  {
    dimension_type const dim = View1<T1, Block1>::dim;

    typedef typename
      impl::assignment::Dispatcher_helper<dim, Block1, Block2, true>::type
      dispatch_type;

    create_holder<dim, dispatch_type>(dst, src);
  }

  ~Assignment() { delete holder_;}

  void operator()()
  { 
    impl::profile::Scope<impl::profile::par> scope(name());
    holder_->exec();
  }

  char const *name() { return holder_->name();}
  
private:
  class Holder
  {
  public:
    virtual ~Holder() {};
    virtual void exec() = 0;
    virtual char const *name() = 0;
  };

  template <dimension_type D, typename T>
  struct Holder_factory;

  template <dimension_type D, typename LHS, typename RHS>
  class Ser_expr_holder : public Holder
  {
    typedef typename LHS::value_type lhs_value_type;
    typedef typename RHS::value_type rhs_value_type;
    typedef typename impl::View_of_dim<D, lhs_value_type, LHS>::type
    lhs_view_type;
    typedef typename impl::View_of_dim<D, rhs_value_type, RHS>::const_type
    rhs_view_type;

  public:
    Ser_expr_holder(lhs_view_type lhs, rhs_view_type rhs) : lhs_(lhs), rhs_(rhs) {}
    void exec() { lhs_ = rhs_;}
    char const *name() { return "Ser_expr_holder";}

  private:
    lhs_view_type lhs_;
    rhs_view_type rhs_;
  };

  template <dimension_type D>
  struct Holder_factory<D, impl::assignment::serial_expr>
  {
    template <typename LHS, typename RHS>
    static Holder *create(LHS lhs, RHS rhs)
    {
      typedef typename LHS::block_type lhs_block_type;
      typedef typename RHS::block_type rhs_block_type;
      return new Ser_expr_holder<D, lhs_block_type, rhs_block_type>(lhs, rhs);
    }
  };
  
  template <dimension_type D, typename LHS, typename RHS>
  class Par_expr_holder : public Holder
  {
    typedef typename LHS::value_type lhs_value_type;
    typedef typename RHS::value_type rhs_value_type;

    typedef typename impl::View_of_dim<D, lhs_value_type, LHS>::type
    lhs_view_type;
    typedef typename impl::View_of_dim<D, rhs_value_type, RHS>::const_type
    rhs_view_type;

  public:
    Par_expr_holder(lhs_view_type lhs, rhs_view_type rhs) : par_expr_(lhs, rhs) {}
    void exec() { par_expr_();}
    char const *name() { return "Par_expr_holder";}

  private:
    vsip::impl::Par_expr<D, LHS, RHS> par_expr_;
  };

  template <dimension_type D, typename LHS, typename RHS>
  class Simple_par_expr_holder : public Holder
  {
    typedef typename LHS::value_type lhs_value_type;
    typedef typename RHS::value_type rhs_value_type;

    typedef typename impl::View_of_dim<D, lhs_value_type, LHS>::type
    lhs_view_type;
    typedef typename impl::View_of_dim<D, rhs_value_type, RHS>::const_type
    rhs_view_type;

  public:
    Simple_par_expr_holder(lhs_view_type lhs, rhs_view_type rhs) : lhs_(lhs), rhs_(rhs) {}
    void exec() { par_expr_simple(lhs_, rhs_);}
    char const *name() { return "Simple_par_expr_holder";}

  private:
    lhs_view_type lhs_;
    rhs_view_type rhs_;
  };

  template <dimension_type D>
  struct Holder_factory<D, impl::assignment::par_expr>
  {
    template <typename LHS, typename RHS>
    static Holder *create(LHS lhs, RHS rhs)
    {
      typedef typename LHS::value_type lhs_value_type;
      typedef typename RHS::value_type rhs_value_type;
      typedef typename LHS::block_type lhs_block_type;
      typedef typename RHS::block_type rhs_block_type;
      typedef typename LHS::block_type::map_type lhs_map_type;
      typedef typename RHS::block_type::map_type rhs_map_type;
      
      if (impl::Is_par_same_map<D, lhs_map_type, rhs_block_type>
	  ::value(lhs.block().map(), rhs.block()))
      {
	typedef typename impl::Distributed_local_block<lhs_block_type>::type
	  lhs_local_block_type;
	typedef typename impl::Distributed_local_block<rhs_block_type>::type
	  rhs_local_block_type;
	
	typedef typename 
	  impl::View_of_dim<D, lhs_value_type, lhs_local_block_type>::type
	  lhs_local_view_type;
	typedef typename 
	  impl::View_of_dim<D, rhs_value_type, rhs_local_block_type>::type
	  rhs_local_view_type;

	lhs_local_view_type lhs_local = get_local_view(lhs);
	rhs_local_view_type rhs_local = get_local_view(rhs);
	
	return new Ser_expr_holder<D, lhs_local_block_type, rhs_local_block_type>
	  (lhs_local, rhs_local);
      }
      else
      {
	return new Par_expr_holder<D, lhs_block_type, rhs_block_type>(lhs, rhs);
      }
    }
  };

  template <dimension_type D, typename LHS, typename RHS, typename Assign>
  class Par_assign_holder : public Holder
  {
    typedef typename LHS::value_type lhs_value_type;
    typedef typename RHS::value_type rhs_value_type;

    typedef typename impl::View_of_dim<D, lhs_value_type, LHS>::type
    lhs_view_type;
    typedef typename impl::View_of_dim<D, rhs_value_type, RHS>::const_type 
    rhs_view_type;

  public:
    Par_assign_holder(lhs_view_type lhs, rhs_view_type rhs) : par_assign_(lhs, rhs) {}
    void exec() { par_assign_();}
    char const *name() { return "Par_assign_holder";}

  private:
    vsip::impl::Par_assign<D, lhs_value_type, rhs_value_type, LHS, RHS, Assign>
    par_assign_;
  };

  template <dimension_type D, typename Assign>
  struct Holder_factory<D, impl::assignment::par_assign<Assign> >
  {
    template <typename LHS, typename RHS>
    static Holder *create(LHS lhs, RHS rhs)
    {
      typedef typename LHS::value_type lhs_value_type;
      typedef typename RHS::value_type rhs_value_type;
      typedef typename LHS::block_type lhs_block_type;
      typedef typename RHS::block_type rhs_block_type;
      typedef typename LHS::block_type::map_type lhs_map_type;
      typedef typename RHS::block_type::map_type rhs_map_type;

      if (impl::Is_par_same_map<D, lhs_map_type, rhs_block_type>
	  ::value(lhs.block().map(), rhs.block()))
      {
	typedef typename impl::Distributed_local_block<lhs_block_type>::type
	  lhs_local_block_type;
	typedef typename impl::Distributed_local_block<rhs_block_type>::type
	  rhs_local_block_type;

	typedef typename 
	  impl::View_of_dim<D, lhs_value_type, lhs_local_block_type>::type
	  lhs_local_view_type;
	typedef typename 
	  impl::View_of_dim<D, rhs_value_type, rhs_local_block_type>::type
	  rhs_local_view_type;

	lhs_local_view_type lhs_local = get_local_view(lhs);
	rhs_local_view_type rhs_local = get_local_view(rhs);

	return new Ser_expr_holder<D, lhs_local_block_type, rhs_local_block_type>
	  (lhs_local, rhs_local);
      }
      else
      {
	return new Par_assign_holder<D, lhs_block_type, rhs_block_type, Assign>(lhs, rhs);
      }
    }
  };

  template <dimension_type D, typename T, typename LHS, typename RHS>
  void
  create_holder(LHS lhs, RHS rhs) { holder_ = Holder_factory<D, T>::create(lhs, rhs);}

  Holder *holder_;

};

} // namespace vsip

#endif // VSIP_CORE_SETUP_ASSIGN_HPP
