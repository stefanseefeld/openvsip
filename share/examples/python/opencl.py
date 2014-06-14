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
import types

def wrap(kernel, signature):
    """Wrap a raw kernel object with signature information,
    so vector and matrix arguments can be appropriately passed
    as 'in' and 'out' parameters.

    Arguments:
      kernel - the kernel object
      signature - a tuple indicating the number of 'in', 'inout', and 'out'
                  parameters: (<#in>, <#inout>, <#out>)
    """

    class K:

        def __init__(self, kernel, signature):
            self.kernel = kernel
            self.signature = signature

        def __call__(self, queue, nd, *args):
            assert sum(signature) == len(args)
            dda_args = []
            for i,a in enumerate(args):
                if i < signature[0]:
                    dda_args.append(dda.dda(a.block, dda.sync_in))
                elif i < signature[1]:
                    dda_args.append(dda.dda(a.block, dda.sync_inout))
                else:
                    dda_args.append(dda.dda(a.block, dda.sync_out))
            bufs = [data.buf() for data in dda_args]
            self.kernel(queue, nd, *bufs)
            for d in dda_args:
                d.sync_out()

    return K(kernel, signature)


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
print 'copying string PASSED'

# Now try a copy operation on vectors
v1 = vector(array=numpy.arange(16, dtype=numpy.float32))
v2 = vector(length=16, dtype=numpy.float32)
v1_data = dda.dda(v1.block, dda.sync_in)
v2_data = dda.dda(v2.block, dda.sync_out)
kernel(queue, 32, v1_data.buf(), v2_data.buf())
# force the data back to the host
v2_data.sync_out();
assert v1 == v2
print 'copying buffers PASSED'

# now do the same using the wrapper
kernel = wrap(program.create_kernel("copy"), (1,0,1))
v2 = vector(length=16, dtype=numpy.float32)
v3 = vector(length=16, dtype=numpy.float32)
v1[:] = 1.
logging.info('copying v1->v2')
kernel(queue, 32, v1, v2)
logging.info('copying v2->v3')
kernel(queue, 32, v2, v3) # v2 never gets to the host
logging.info('comparing v1 and v3')
assert v1 == v3
print 'copying vectors PASSED'
