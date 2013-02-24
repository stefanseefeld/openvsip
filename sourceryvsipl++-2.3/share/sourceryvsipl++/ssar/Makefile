######################################################### -*-Makefile-*-
#
# Contents: Makefile for Scalable SAR (SSAR) benchmark program.
#
########################################################################

targets := ssar viewtopng diffview
check_targets := ssar

include ../common.mk

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
ssar: | cinterp.plg zinterp.plg

clean::
	rm -f *.plg

# Building SPU plugins requires separate build rules.
%.plg:
	$(MAKE) -f Makefile.spu $@ variant=$(variant) prefix=$(prefix)
endif
ifdef HAVE_CUDA
ssar: interp.o

clean::
	rm -f interp.o

%.o: cuda/%.cu
	nvcc $(CPPFLAGS) -DUNIX -O3 -o $@ -c $<

endif

ssar.check: ssar diffview viewtopng
	@echo "Running SSAR application..."
	./ssar -o result3.view data3
	@echo
	@echo "Comparing output to reference view (should be less than -100)"
	./diffview -r result3.view data3/$(ref_image_base).view 1144 756
	@echo
	@echo "Creating viewable image of output"
	./viewtopng -s result3.view result3.png 1144 756
	@echo "Creating viewable image of reference view"
	./viewtopng -s data3/$(ref_image_base).view data3/$(ref_image_base).png 1144 756

profile1: ssar viewtopng
	@echo "Profiling SSAR application (SCALE = 1)..."
	./ssar --output result1.view --loop 10 \
          --vsip-profile-mode=accum --vsip-profile-output=profile.out \
          data1
	@echo "Formatting profiler output..."
	${fmt-profile-command}  -sec -o profile1.txt profile.out
	./make_images.sh result1.view data1 data1 438 160 382 266

profile3: ssar viewtopng
	@echo "Profiling SSAR application (SCALE = 3)..."
	./ssar --output result3.view --loop 10 \
          --vsip-profile-mode=accum --vsip-profile-output=profile.out \
          data3
	@echo "Formatting profiler output..."
	${fmt-profile-command}  -sec -o profile3.txt profile.out
	./make_images.sh result3.view data3 data3 1072 480 1144 756

viewtopng: LIBS += -lpng
