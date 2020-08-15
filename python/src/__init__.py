#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

# FIXME: For OpenMPI to be able to operate correctly we need to
#        use RTLD_GLOBAL. We need to find a better way that doesn't
#        risk symbol collision !
#import sys
#import dl
#sys.setdlopenflags(dl.RTLD_NOW | dl.RTLD_GLOBAL)
from library import library
#sys.setdlopenflags(dl.RTLD_NOW | dl.RTLD_LOCAL)
_library = library()

# Make value-types available through this module.
from __builtin__ import bool, int, float, complex
from numpy import float32, float64

from _import import import_module
from support import *
from vector import *
from matrix import *

row = 0
col = 1
