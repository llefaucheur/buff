# Flat FFT1024 SVE2 + CMSIS-Ne10 NEON Compare

This folder is intentionally flat: SVE2 source, CMSIS-DSP Ne10 source, CMSIS headers, benchmark, and Makefile are all in one directory.

The Ne10 files were fetched from:

```text
https://github.com/ARM-software/CMSIS-DSP/tree/main/Ne10
```

## Main Files

SVE2:

```text
fft1024_sve2.c
fft1024_sve2.h
```

Benchmark/adapter:

```text
fft1024_compare.c
ne10_fft1024_adapter.c
ne10_fft1024_adapter.h
cmsis_flat_compat.h
Makefile
```

CMSIS-Ne10 float32 files used by the benchmark:

```text
CMSIS_NE10_fft_init.c
NE10_fft_float32.neonintrinsic.c
CMSIS_NE10_fft.h
CMSIS_NE10_types.h
CMSIS_NE10_macros.h
```

Additional Ne10/CMSIS files are also present in the folder for completeness.

`cmsis_flat_compat.h` supplies the small CMSIS-Core compiler macros needed when
building the copied CMSIS-DSP headers as a standalone flat package.

## Build SVE2 Only

On Orion-O6:

```sh
make clean
make bench CC=clang ARCH_FLAGS="-march=armv9.2-a+sve2"
./fft1024_compare 1000
```

Cross-compile from x86_64:

```sh
make clean
make bench CC=clang TARGET=aarch64-linux-gnu ARCH_FLAGS="-march=armv9.2-a+sve2"
```

Run the executable on the Orion-O6, not on the x86_64 build host.

## Build SVE2 vs Ne10 NEON

On Orion-O6:

```sh
make clean
make bench_ne10 CC=clang ARCH_FLAGS="-march=armv9.2-a+sve2"
./fft1024_compare 1000
```

If the A520 uses 128-bit SVE, also test fixed-width SVE codegen:

```sh
make clean
make bench_ne10 CC=clang ARCH_FLAGS="-march=armv9.2-a+sve2" SVE_BITS=128
./fft1024_compare 1000
```

Cross-compile from x86_64:

```sh
make clean
make bench_ne10 CC=clang TARGET=aarch64-linux-gnu ARCH_FLAGS="-march=armv9.2-a+sve2"
```

## Ne10 Adapter

The adapter calls the CMSIS-integrated Ne10 API:

```c
arm_cfft_init_dynamic_f32(1024);
arm_ne10_mixed_radix_fft_forward_float32_neon(...);
```

If your CMSIS-DSP checkout changes those names, edit only:

```text
ne10_fft1024_adapter.c
```

## Output

```text
CFFT1024 SVE2 round-trip max abs error: ...
Kernel             ticks/call     SVE2/NE10
CFFT1024 SVE2      ...
CFFT1024 Ne10 NEON ...
```

Check the round-trip error first. It should be small for float32, not hundreds
or thousands. A large value means the SVE2 forward/inverse pair is not valid,
even if the forward-only timing table still prints.

The default timer is `CNTVCT_EL0`; use `perf stat` on Orion-O6 for architectural cycle counts:

```sh
perf stat -e cycles,instructions,cache-references,cache-misses ./fft1024_compare 1000
```
