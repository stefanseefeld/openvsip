#
# Copyright (c) 2013 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

subdirs := $(filter-out . ./apps ./apps/ssar/data, $(shell find . -type d))

define make
for d in $(subdirs); do \
  $(MAKE) -C $$d $(1); \
done
endef


all:
	$(call make,all)
check:
	$(call make,check)
clean:
	$(call make,clean)
