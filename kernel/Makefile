# DNPR is part of the built-in DRM panic path, not a loadable module. The
# panic handoff depends on unexported architecture symbols such as
# cpus_stop_mask, and kernel/panic.c calls these functions at link time.
# Build the objects for compile checking, but do not create a .ko or run
# modpost against this directory.
ifneq ($(KERNELRELEASE),)
obj-y += dnpr_panic.o dnpr_audio.o
ccflags-y += -I$(srctree)/drivers/gpu/drm
else
KDIR ?= /mnt/nvme/dnpr-linux-6.15.2

.PHONY: all compile-check
all: compile-check

compile-check:
	$(MAKE) -C $(KDIR) M=$(CURDIR) dnpr_panic.o dnpr_audio.o
endif
