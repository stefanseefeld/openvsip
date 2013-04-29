/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    sarsim.cpp
    @author  Jules Bergmann
    @date    03/02/2005
    @brief   VSIPL++ implementation of RASSP benchmark 0.
*/

#define PARALLEL 0

#include <iostream>
#include "sarsim.hpp"

typedef double value_type;

using vsip::index_type;
using vsip::Vector;
using vsip::Matrix;

class SimpleSarSim : public SarSim<value_type> {
public:

  SimpleSarSim(index_type nrange,
	       index_type npulse,
	       index_type ncsamples,
	       index_type niq,
	       io_type swath,
	       Matrix<cio_type> w_eq,
	       Vector<io_type> rcs,
	       Vector<io_type> i_coef,
	       Vector<io_type> q_coef,
	       Matrix<cio_type> cphase,
	       std::istream& in,
	       std::ostream& out) : 
    SarSim<value_type>(nrange, npulse, ncsamples, niq, swath,
		       w_eq, rcs, i_coef, q_coef, cphase, 0),
    in_(in), out_(out) {}

protected:

  io_type read_pulse(int pol, index_type p);
  void write_output_header(int pol);
  void write_output(int pol);

private:

  std::istream& in_;
  std::ostream& out_;
};

SimpleSarSim::io_type
SimpleSarSim::read_pulse(int /* pol */, index_type /* p */) {
  io_type range;
  in_.read ((char *) &range, sizeof (range));
  for (index_type i = 0; i < ncsamples_; ++i) {
    io_type val[2];
    in_.read ((char *) val, sizeof (val));
    vec_iq_.put(i, cval_type(val[0], val[1]));
  }
  return range;
}

void
SimpleSarSim::write_output_header(int /* pol */) {
}

void
SimpleSarSim::write_output(int /* pol */) {
  for (index_type i = npulse_; i < 2 * npulse_; ++i) {
    io_type val[2];
    val[0] = azbuf_.get(i).real();
    val[1] = azbuf_.get(i).imag();
    out_.write ((char *) &val, sizeof (val));
  }
}

int 
main() {
  SimpleSarSim::cio_matrix_type w_eq(0, 0);
  SimpleSarSim::io_vector_type rcs(0);
  SimpleSarSim::io_vector_type i_coef(0);
  SimpleSarSim::io_vector_type q_coef(0);
  SimpleSarSim::cio_matrix_type cphase(0, 0);
  SimpleSarSim sarsim(0, 0, 0, 0, 0,
		      w_eq, rcs, i_coef, q_coef, cphase,
		      std::cin, std::cout);
}




