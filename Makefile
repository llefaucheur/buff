CC ?= clang
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
CFLAGS ?= $(TARGET_FLAGS) $(ARCH_FLAGS) $(OPT_FLAGS) $(WARN_FLAGS)
LDFLAGS ?= $(TARGET_FLAGS)
LDLIBS ?= -lm

OBJS = fft2048_sve2.o fft2048_bench.o

.PHONY: all clean

all: fft2048_bench

fft2048_bench: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

fft2048_sve2.o: fft2048_sve2.c fft2048_sve2.h
	$(CC) $(CFLAGS) -c -o $@ fft2048_sve2.c

fft2048_bench.o: fft2048_bench.c fft2048_sve2.h
	$(CC) $(CFLAGS) -c -o $@ fft2048_bench.c

clean:
	rm -f $(OBJS) fft2048_bench
