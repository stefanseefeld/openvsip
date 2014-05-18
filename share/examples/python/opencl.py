#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip.opencl import cl, dda
from vsip import vector
import numpy
import logging

# set logging level to info, so we can see
# output from OVXX_TRACE
logging.basicConfig(level=logging.INFO)


# print out available platforms
platforms = cl.platform.platforms()
for p in platforms:
    print p.name, p.version, p.vendor

# The rest of this program will use the default platform (index 0).
# To change it, set the OVXX_OPENCL_PLATFORM variable to an
# appropriate index in the above list.

# Construct a 'copy' kernel...
cxt = cl.default_context()
src = """
__kernel void copy(__global char const *in, __global char *out)
{
  int num = get_global_id(0);
  out[num] = in[num];
}"""

program = cl.program.create_with_source(cxt, src)
program.build(cxt.devices())
kernel = program.create_kernel("copy")

# ... and feed it some input ...
input = "Hello, World !"

inbuf = cl.buffer(cxt, 32)
outbuf = cl.buffer(cxt, 32)
queue = cl.default_queue()
queue.write(input, inbuf)
kernel(queue, 32, inbuf, outbuf)
output = queue.read_string(outbuf, len(input))

# ...finally test that the output is equal to the input
assert input == output

# Now try a copy operation on vectors
v1 = vector(array=numpy.arange(16, dtype=numpy.float32))
v2 = vector(length=16, dtype=numpy.float32)
v1_data = dda.dda(v1.block, dda.sync_in)
v2_data = dda.dda(v2.block, dda.sync_out)
kernel(queue, 32, v1_data.buf(), v2_data.buf())
# force the data back to the host
v2_data.sync_out();
assert v1 == v2
