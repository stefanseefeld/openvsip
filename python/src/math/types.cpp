//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/python/block.hpp>
#include <vsip/math.hpp>

BOOST_PYTHON_MODULE(types)
{
  namespace bpl = boost::python;

  bpl::enum_<vsip::mat_op_type> mat_op("mat_op");
  mat_op.value("ntrans", vsip::mat_ntrans);
  mat_op.value("trans", vsip::mat_trans);
  mat_op.value("herm", vsip::mat_herm);
  mat_op.value("conj", vsip::mat_conj);

  bpl::enum_<vsip::product_side_type> product_side("product_side");
  product_side.value("lside", vsip::mat_lside);
  product_side.value("rside", vsip::mat_rside);

  bpl::enum_<vsip::storage_type> storage("storage");
  storage.value("qrd_nosaveq", vsip::qrd_nosaveq);
  storage.value("qrd_saveq1", vsip::qrd_saveq1);
  storage.value("qrd_saveq", vsip::qrd_saveq);
  storage.value("svd_uvnos", vsip::svd_uvnos);
  storage.value("svd_uvpart", vsip::svd_uvpart);
  storage.value("svd_uvfull", vsip::svd_uvfull);
}
