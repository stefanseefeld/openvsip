//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cvsip_common_hpp_
#define ovxx_cvsip_common_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/impl/signal/types.hpp>
#include <vsip/impl/solver/common.hpp>
extern "C" {
#include <vsip.h>
}

namespace ovxx
{
namespace cvsip
{
inline vsip_symmetry 
symmetry(symmetry_type s)
{
  switch (s)
  {
    case sym_even_len_odd: return VSIP_SYM_EVEN_LEN_ODD;
    case sym_even_len_even: return VSIP_SYM_EVEN_LEN_EVEN;
    default: return VSIP_NONSYM;
  }
}

inline vsip_support_region
support(support_region_type s)
{
  switch (s)
  {
    case support_same: return VSIP_SUPPORT_SAME;
    case support_min: return VSIP_SUPPORT_MIN;
    default: return VSIP_SUPPORT_FULL;
  }
}

inline vsip_bias
bias(bias_type b)
{
  switch (b)
  {
    case biased: return VSIP_BIASED;
    default: return VSIP_UNBIASED;
  }
}

inline vsip_obj_state
save(obj_state s)
{
  switch (s)
  {
    case state_save: return VSIP_STATE_SAVE;
    default: return VSIP_STATE_NO_SAVE;
  }
}

inline vsip_alg_hint
hint(alg_hint_type h)
{
  switch (h)
  {
    case alg_noise: return VSIP_ALG_NOISE;
    case alg_space: return VSIP_ALG_TIME;
    default: return VSIP_ALG_SPACE;
  }
}

inline vsip_qrd_qopt storage(storage_type s)
{
  switch (s)
  {
    case qrd_nosaveq: return VSIP_QRD_NOSAVEQ;
    case qrd_saveq1: return VSIP_QRD_SAVEQ1;
    default: return VSIP_QRD_SAVEQ;
  }
}

inline vsip_mat_side product_side(product_side_type s)
{
  return s == mat_lside ? VSIP_MAT_LSIDE : VSIP_MAT_RSIDE;
}

inline vsip_mat_op mat_op(mat_op_type o)
{
  switch (o)
  {
    case mat_ntrans: return VSIP_MAT_NTRANS;
    case mat_trans: return VSIP_MAT_TRANS;
    default: return VSIP_MAT_HERM;
  }
}

inline vsip_mat_uplo get_mat_uplo(mat_uplo ul)
{
  return ul == lower ? VSIP_TR_LOW : VSIP_TR_UPP;
}


} // namespace ovxx::cvsip
} // namespace ovxx

#endif
