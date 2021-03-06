//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_signal_types_hpp_
#define vsip_impl_signal_types_hpp_

namespace vsip
{

enum alg_hint_type
{
  alg_time,
  alg_space,
  alg_noise
};

enum support_region_type
{
  support_full,
  support_same,
  support_min,
  support_min_zeropad
};

enum symmetry_type
{
  nonsym,
  sym_even_len_odd,
  sym_even_len_even
};

enum bias_type
{
  biased,
  unbiased
};

enum obj_state 
{
  state_no_save,
  state_save
};

} // namespace vsip

#endif
