---
layout: layout
---

OpenCL support
==============

OpenVSIP now contains basic OpenCL support, including automatically managed
device storage with on-demand data movement.

Prototyping OpenCL kernels with Python
--------------------------------------

The Python bindings allow to prototype custom kernels:

{% highlight python %}
from vsip.opencl import cl

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
{% endhighlight %}

'kernel' is now a callable copying data from 'in' to 'out'.
We can construct two vectors and copy one to the other using
this kernel:

{% highlight python %}
v1 = vector(array=numpy.arange(16, dtype=numpy.float32))
v2 = vector(length=16, dtype=numpy.float32)
kernel(queue, 32, v1, v2)
{% endhighlight %}

Lazy data movement
------------------

The invocation of the kernel will trigger the data in 'v1' to be
replicated into device memory, copied to 'v2' on the device, and
then copied back to the host so it can be directly accessed again.

In fact, in a chained copy *v1 -> v2 -> v3*, v2's memory may never
be replicated to the host if it is never accessed there.

This can be observed using the logging module:

{% highlight python %}
logging.basicConfig(level=logging.INFO)
{% endhighlight %}

So the following code:

{% highlight python %}
v1 = vector(array=numpy.arange(16, dtype=numpy.float32))
v2 = vector(length=16, dtype=numpy.float32)
v3 = vector(length=16, dtype=numpy.float32)
logging.info('copying v1->v2')
kernel(queue, 32, v1, v2)
logging.info('copying v2->v3')
kernel(queue, 32, v2, v3)
logging.info('comparing v1 and v3')
assert v1 == v3
{% endhighlight %}

will result in this output:

    INFO:root:host_storage::allocate(64)       # for v1
    INFO:root:copying v1->v2
    INFO:root:OpenCL copy 64 bytes to device   # v1
    INFO:root:copying v2->v3
    INFO:root:comparing v1 and v3
    INFO:root:host_storage::allocate(64)       # for v3
    INFO:root:OpenCL copy 64 bytes from device # v1
    INFO:root:OpenCL copy 64 bytes from device # v3

which demonstrates that the entiere pipeline only requires three data
movements (v1 to the device, then v1 and v3 back to the host).
In fact, conceptually there is no reason to move v1 back to the host,
as its value hasn't changed. However, unlike in C++ which has a stronger
type system than Python (with an explicit notion of const-qualification),
there is no (straight-forward) way to indicate in Python that v1 is to be
read-only, so its host storage remains valid even while others can access
its data in device storage.

The C++ equivalent of the above code in fact only requires two data
movements instead of three.