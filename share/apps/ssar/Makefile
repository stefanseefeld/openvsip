######################################################### -*-Makefile-*-
#
# Contents: Makefile for Scalable SAR (SSAR) benchmark program.
#
########################################################################

########################################################################
#
# Copyright (c) 2005-2011 CodeSourcery, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of CodeSourcery nor the names of its
#       contributors may be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY CODESOURCERY, INC. "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CODESOURCERY BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

targets := ssar viewtopng diffview
check_targets :=

include ../../common.mk

CPPFLAGS += -I .
LIBS += -lhdf5

########################################################################
# Rules
########################################################################

check: ssar diffview viewtopng
	@echo "Running SSAR application..."
	./ssar data
	@echo
	@echo "Comparing output to reference view (should be less than -100)"
	./diffview -r data/image.hdf5 data/ref_image.hdf5
	@echo
	@echo "Creating viewable image of output"
	./viewtopng -s data/image.hdf5 data/image.png
	@echo "Creating viewable image of reference view"
	./viewtopng -s data/ref_image.hdf5 data/ref_image.png

viewtopng: LIBS += -lpng
