#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip import vector

def first(j, f, v, w):
    """Return the first index :code:`i >= j` where :code:`f(v[j], w[j]) == True`."""

    for i in range(j, v.size()):
        if f(v[i], w[i]):
            return i
    return v.size()
