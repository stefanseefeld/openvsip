//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_reductions_types_hpp_
#define ovxx_reductions_types_hpp_

namespace ovxx
{

enum reduction_type
{
  reduce_all_true,
  reduce_all_true_bool,
  reduce_any_true,
  reduce_any_true_bool,
  reduce_mean,
  reduce_mean_magsq,
  reduce_sum,
  reduce_sum_bool,
  reduce_sum_sq,
  reduce_max_magsq,
  reduce_max_mag,
  reduce_min_magsq,
  reduce_min_mag,
  reduce_max,
  reduce_min
};

} // namespace ovxx

#endif
