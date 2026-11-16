# Compile check only. The real panic integration is built into DRM via
# dnpr-linux-6.15.2.patch; do not load this object on a running system.
obj-m += dnpr_panic.o
ccflags-y += -I$(srctree)/drivers/gpu/drm
