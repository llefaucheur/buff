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

The radix-8 SVE2 design note is in:

```text
RADIX8_SVE2_PROPOSAL.md
```

The SVE2 radix-4 twiddle multiply in `fft1024_sve2.c` uses the explicit
`cmul_forward_mla_f32()` helper. It is written in the same style as Ne10's
complex multiply macro: start with real-lane multiplies, then use SVE
`svmls_f32_x` / `svmla_f32_x` for the fused subtract/add terms.

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

The SVE2 table line is measured with one outer timer around repeated calls to
`cfft1024_f32_sve2()`. Before timing, the benchmark fills a batch buffer with
one fresh 1024-point input per iteration. The timed loop then runs one in-place
FFT on each batch entry. This avoids putting `memcpy` inside the timed loop,
avoids subtracting a separate memcpy measurement, and avoids repeatedly applying
an unnormalized FFT to the same buffer.

The Ne10 table line uses the same batch-input idea: one source entry and one
destination entry per timed iteration.

The printed SVE2 breakdown is diagnostic only. It uses extra counter reads
inside the FFT call, so its `profiled total` can be higher than the clean table
timing on Cortex-A520.

Check the round-trip error first. It should be small for float32, not hundreds
or thousands. A large value means the SVE2 forward/inverse pair is not valid,
even if the forward-only timing table still prints.

The default timer is `CNTVCT_EL0`; use `perf stat` on Orion-O6 for architectural cycle counts:

```sh
perf stat -e cycles,instructions,cache-references,cache-misses ./fft1024_compare 1000
```

## Assembly Review

Generate assembly for the SVE2 source and the CMSIS-Ne10 NEON source:

```sh
make asm CC=clang ARCH_FLAGS="-march=armv9.2-a+sve2"
```

The generated files are written to:

```text
asm/fft1024_sve2.s
asm/fft1024_compare.s
asm/ne10_fft1024_adapter.s
asm/CMSIS_NE10_fft_init.s
asm/NE10_fft_float32.neonintrinsic.s
```

For Cortex-A520 review, also test fixed 128-bit SVE code generation:

```sh
make asm_clean
make asm CC=clang ARCH_FLAGS="-march=armv9.2-a+sve2" SVE_BITS=128
```

Generate only one side:

```sh
make asm_sve2 CC=clang ARCH_FLAGS="-march=armv9.2-a+sve2" SVE_BITS=128
make asm_ne10 CC=clang ARCH_FLAGS="-march=armv9.2-a+sve2"
```

Remove generated assembly:

```sh
make asm_clean
```
