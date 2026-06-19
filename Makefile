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
NE10_INC ?=
NE10_LIB ?=
NE10_LDLIBS ?= -lNE10

CFLAGS ?= $(TARGET_FLAGS) $(ARCH_FLAGS) $(OPT_FLAGS) $(WARN_FLAGS)
LDFLAGS ?= $(TARGET_FLAGS)
LDLIBS ?= -lm

LIB = libfft1024_sve2.a

.PHONY: all bench bench_ne10 clean

all: $(LIB)

$(LIB): fft1024_sve2.o
	$(AR) rcs $@ $<

fft1024_sve2.o: fft1024_sve2.c fft1024_sve2.h
	$(CC) $(CFLAGS) -c -o $@ fft1024_sve2.c

fft1024_compare.o: fft1024_compare.c fft1024_sve2.h
	$(CC) $(CFLAGS) $(NE10_INC) -c -o $@ fft1024_compare.c

bench: fft1024_sve2.o fft1024_compare.o
	$(CC) $(LDFLAGS) -o fft1024_compare fft1024_sve2.o fft1024_compare.o $(LDLIBS)

bench_ne10: CFLAGS += -DFFT1024_USE_NE10
bench_ne10: clean_objects fft1024_sve2.o fft1024_compare.o
	$(CC) $(LDFLAGS) -o fft1024_compare fft1024_sve2.o fft1024_compare.o $(NE10_LIB) $(NE10_LDLIBS) $(LDLIBS)

clean_objects:
	rm -f fft1024_sve2.o fft1024_compare.o

clean:
	rm -f fft1024_sve2.o fft1024_compare.o $(LIB) fft1024_compare
