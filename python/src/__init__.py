#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from library import library
_library = library()

# Make value-types available through this module.
from __builtin__ import bool, int, float, complex
from numpy import float32, float64

from vector import *
from matrix import *
