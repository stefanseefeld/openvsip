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

include ../../common.mk

CPPFLAGS += -I .

########################################################################
# Variables
########################################################################

HAVE_CBE_SDK := 
HAVE_CUDA := 

# The default precision is single (double may also be used)
precision := single

ifeq ($(precision),double)
ref_image_base := ref_image_dp
ssar_type := SSAR_BASE_TYPE=double
else
ref_image_base := ref_image_sp
ssar_type := SSAR_BASE_TYPE=float
endif

fmt-profile-command := $(prefix)/bin/fmt-profile.pl

# Enable profiling
PROFILE	:= -DVSIP_PROFILE_MATVEC -DVSIP_PROFILE_SIGNAL -DVSIP_PROFILE_USER
CXXFLAGS += $(PROFILE) -D$(ssar_type)

########################################################################
# Rules
########################################################################

ifdef HAVE_CBE_SDK

include ../../common.spu.mk

ssar: | cinterp.plg zinterp.plg

clean::
	rm -f *.plg

plg_obj:= cbe/accel/cinterp.o cbe/accel/zinterp.o

$(plg_obj): %.o: %.cpp
	$(compile_ukernel)

%.exe: %.o
	$(link_ukernel)

%.plg: cbe/accel/%.exe
	$(embed_ukernel)

endif
ifdef HAVE_CUDA

include ../../common.cuda.mk

ssar: interp.o

clean::
	rm -f interp.o

%.o: cuda/%.cu
	$(compile_cuda)

endif

check: ssar diffview viewtopng
	@echo "Running SSAR application..."
	./ssar -o data3/result3.view data3
	@echo
	@echo "Comparing output to reference view (should be less than -100)"
	./diffview -r data3/result3.view data3/$(ref_image_base).view 1144 756
	@echo
	@echo "Creating viewable image of output"
	./viewtopng -s data3/result3.view data3/result3.png 1144 756
	@echo "Creating viewable image of reference view"
	./viewtopng -s data3/$(ref_image_base).view data3/$(ref_image_base).png 1144 756

profile1: ssar viewtopng
	@echo "Profiling SSAR application (SCALE = 1)..."
	./ssar --output data1/result1.view --loop 10 \
          --vsip-profile-mode=accum --vsip-profile-output=data1/profile.out \
          data1
	@echo "Formatting profiler output..."
	${fmt-profile-command}  -sec -o data1/profile1.txt data1/profile.out
	./make_images.sh data1/result1.view data1 data1 438 160 382 266

profile3: ssar viewtopng
	@echo "Profiling SSAR application (SCALE = 3)..."
	./ssar --output data3/result3.view --loop 10 \
          --vsip-profile-mode=accum --vsip-profile-output=data3/profile.out \
          data3
	@echo "Formatting profiler output..."
	${fmt-profile-command}  -sec -o data3/profile3.txt data3/profile.out
	./make_images.sh data3/result3.view data3 data3 1072 480 1144 756

viewtopng: LIBS += -lpng
