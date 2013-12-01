//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/dense.hpp>
#include <vsip/math.hpp>
#include <vsip/selgen.hpp>
#include <ovxx/equal.hpp>
#include <ovxx/output.hpp>
#include <ovxx/io/hdf5.hpp>
#include <iostream>
#include <fstream>

using namespace ovxx;

int main (int argc, char **argv)
{
  length_type const ROWS=8;
  length_type const COLS=4;

  vsipl init(argc, argv);

  Matrix<complex<float> > m_in(ROWS, COLS);
  for (index_type r = 0; r != ROWS; ++r)
    m_in.row(r) = ramp<float>(r*COLS, 1, COLS);
  Matrix<complex<float> > m_out(ROWS, COLS);
  try
  {
    {
      hdf5::file file("data.hdf5", 'w');
      // Write a complex matrix into the file...
      hdf5::dataset md = file.create_dataset("matrix");
      md.write(m_in);
      // ...as well as the real part of its first row.
      hdf5::dataset vd = file.create_dataset("vector");
      vd.write(m_in.row(0).real());
    }
    {
      hdf5::file file("data.hdf5", 'r');
      // Retrieve the complex matrix from the file...
      hdf5::dataset md = file.open_dataset("matrix");
      md.read(m_out);
      // ...as well as the vector...
      Vector<float> v(COLS);
      hdf5::dataset vd = file.open_dataset("vector");
      vd.read(v);
      // ...and make sure the vector equals the matrix' first row.
      assert(equal(m_out.row(0).real(), v));
    }
    assert(equal(m_in, m_out));
  }
  catch (std::exception &e)
  {
    std::cerr << "Error : " << e.what() << std::endl;
    return 1;
  }
}
