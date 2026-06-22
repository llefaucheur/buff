CC ?= clang
AR ?= ar
TARGET ?=
SYSROOT ?=
CPU ?= cortex-a520

HOST_MACHINE := $(shell uname -m 2>/dev/null || echo unknown)
USING_CLANG := $(findstring clang,$(notdir $(CC)))
HOST_IS_X86 := $(filter x86_64 amd64 i386 i686,$(HOST_MACHINE))

ifeq ($(USING_CLANG),clang)
ifneq ($(HOST_IS_X86),)
ifeq ($(TARGET),)
$(error This clang is running on $(HOST_MACHINE). Use TARGET=aarch64-linux-gnu, for example: make CC=clang TARGET=aarch64-linux-gnu ARCH_FLAGS="-march=armv9.2-a+sve2")
endif
endif
endif

TARGET_FLAGS =
ifneq ($(TARGET),)
TARGET_FLAGS += --target=$(TARGET)
endif
ifneq ($(SYSROOT),)
TARGET_FLAGS += --sysroot=$(SYSROOT)
endif

ARCH_FLAGS ?= -mcpu=$(CPU)
OPT_FLAGS ?= -O3 -ffast-math -fno-math-errno
WARN_FLAGS ?= -Wall -Wextra
NE10_INC ?= -I.
NE10_LIB ?=
NE10_LDLIBS ?=
NE10_CFLAGS ?= -include cmsis_flat_compat.h -DARM_MATH_NEON

CFLAGS ?= $(TARGET_FLAGS) $(ARCH_FLAGS) $(OPT_FLAGS) $(WARN_FLAGS)
LDFLAGS ?= $(TARGET_FLAGS)
LDLIBS ?= -lm
NE10_OBJS = CMSIS_NE10_fft_init.o NE10_fft_float32.neonintrinsic.o

.PHONY: all sve2 bench bench_ne10 clean clean_objects

all: bench

sve2: libfft1024_sve2.a

libfft1024_sve2.a: fft1024_sve2.o
	$(AR) rcs $@ $<

fft1024_sve2.o: fft1024_sve2.c fft1024_sve2.h
	$(CC) $(CFLAGS) -c -o $@ fft1024_sve2.c

fft1024_compare.o: fft1024_compare.c fft1024_sve2.h ne10_fft1024_adapter.h
	$(CC) $(CFLAGS) -c -o $@ fft1024_compare.c

ne10_fft1024_adapter.o: ne10_fft1024_adapter.c ne10_fft1024_adapter.h cmsis_flat_compat.h
	$(CC) $(CFLAGS) $(NE10_CFLAGS) $(NE10_INC) -c -o $@ ne10_fft1024_adapter.c

CMSIS_NE10_fft_init.o: CMSIS_NE10_fft_init.c CMSIS_NE10_fft.h CMSIS_NE10_types.h CMSIS_NE10_macros.h cmsis_flat_compat.h
	$(CC) $(CFLAGS) $(NE10_CFLAGS) $(NE10_INC) -c -o $@ CMSIS_NE10_fft_init.c

NE10_fft_float32.neonintrinsic.o: NE10_fft_float32.neonintrinsic.c CMSIS_NE10_fft.h CMSIS_NE10_types.h cmsis_flat_compat.h
	$(CC) $(CFLAGS) $(NE10_CFLAGS) $(NE10_INC) -c -o $@ NE10_fft_float32.neonintrinsic.c

bench: fft1024_sve2.o fft1024_compare.o ne10_fft1024_adapter.o
	$(CC) $(LDFLAGS) -o fft1024_compare fft1024_sve2.o fft1024_compare.o ne10_fft1024_adapter.o $(LDLIBS)

bench_ne10: CFLAGS += -DFFT1024_USE_NE10
bench_ne10: clean_objects fft1024_sve2.o fft1024_compare.o ne10_fft1024_adapter.o $(NE10_OBJS)
	$(CC) $(LDFLAGS) -o fft1024_compare fft1024_sve2.o fft1024_compare.o ne10_fft1024_adapter.o $(NE10_OBJS) $(NE10_LIB) $(NE10_LDLIBS) $(LDLIBS)

clean_objects:
	rm -f fft1024_sve2.o fft1024_compare.o ne10_fft1024_adapter.o $(NE10_OBJS)

clean:
	rm -f fft1024_sve2.o fft1024_compare.o ne10_fft1024_adapter.o $(NE10_OBJS) libfft1024_sve2.a fft1024_compare
