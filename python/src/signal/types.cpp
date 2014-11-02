//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/python/block.hpp>
#include <vsip/signal.hpp>

BOOST_PYTHON_MODULE(types)
{
  namespace bpl = boost::python;

  bpl::enum_<vsip::alg_hint_type> alg_hint("alg_hint");
  alg_hint.value("time", vsip::alg_time);
  alg_hint.value("space", vsip::alg_space);
  alg_hint.value("noise", vsip::alg_noise);

  bpl::enum_<vsip::support_region_type> support_region("support_region");
  support_region.value("full", vsip::support_full);
  support_region.value("same", vsip::support_same);
  support_region.value("min", vsip::support_min);
  support_region.value("min_zeropad", vsip::support_min_zeropad);

  bpl::enum_<vsip::symmetry_type> symmetry("symmetry");
  symmetry.value("none", vsip::nonsym);
  symmetry.value("even_len_odd", vsip::sym_even_len_odd);
  symmetry.value("even_len_even", vsip::sym_even_len_even);

  bpl::enum_<vsip::bias_type> bias("bias");
  bias.value("biased", vsip::biased);
  bias.value("unbiased", vsip::unbiased);

  bpl::enum_<vsip::obj_state> obj_state("obj_state");
  obj_state.value("no_save", vsip::state_no_save);
  obj_state.value("save", vsip::state_save);
}
