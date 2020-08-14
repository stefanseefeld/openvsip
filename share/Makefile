#
# Copyright (c) 2013 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

srcdir := .
top_srcdir:=..

include ../config.mk

define install_dir
@echo installing $(1)
$(INSTALL) -d $(DESTDIR)$(pkgdatadir)/$(1)
for f in $(wildcard $(srcdir)/$(1)/*); do \
  $(INSTALL_DATA) $$f $(DESTDIR)$(pkgdatadir)/$(1); \
done
endef


all:

install:
	$(INSTALL) -d $(DESTDIR)$(pkgdatadir)
	$(INSTALL_DATA) common.mk $(DESTDIR)$(pkgdatadir)/common.mk
	$(INSTALL_DATA) $(srcdir)/Makefile $(DESTDIR)$(pkgdatadir)/Makefile
	$(call install_dir,examples/audio)
	$(call install_dir,examples/dda)
	$(call install_dir,examples/dispatch)
	$(call install_dir,examples/dsp)
	$(call install_dir,examples/eval)
	$(call install_dir,examples/io)
	$(call install_dir,examples/parallel)
	$(call install_dir,examples/radar)
	$(call install_dir,examples/signal)
	$(call install_dir,examples/solvers)
	$(call install_dir,examples/ustorage)
	$(call install_dir,examples/views)
ifdef enable_python_bindings
	$(call install_dir,examples/python)
endif
	$(call install_dir,apps/ssar)
	$(call install_dir,apps/ssar/data)

clean:
