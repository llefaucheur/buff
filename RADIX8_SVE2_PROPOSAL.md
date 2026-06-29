# Radix-8 SVE2 Proposal for FFT1024 on Cortex-A520

## Short Answer

Yes, part of the Ne10 advantage can come from its radix-8 first stage.

For 1024-point float32 CFFT, CMSIS-Ne10 initializes the factorization with
`NE10_FACTOR_EIGHT_FIRST_STAGE`. In the float32 NEON kernel, the forward path
checks `first_radix == 8` and calls `ne10_radix8x4_neon()` for the first stage,
then uses radix-4 kernels for the remaining stages.

The current SVE2 implementation is pure radix-4:

```text
1024 = 4 x 4 x 4 x 4 x 4
```

The proposed SVE2 structure should mirror Ne10:

```text
1024 = 8 x 4 x 4 x 4 x 2
```

or, if we want to avoid a scalar/vector radix-2 tail:

```text
first pass: fused radix-8 reorder/load
remaining: four radix-4-like stages over the produced layout
```

## Why Radix-8 Can Help

Radix-8 is attractive for the first stage because its twiddles are trivial:

```text
1, -j, +/-0.70710678 +/- j*0.70710678
```

That means the first stage can replace general complex multiplies with:

```text
add/sub
swap real/imag
sign flip
multiply by sqrt(1/2)
```

For Cortex-A520 with 128-bit SVE, this matters because the current radix-4 core
spends most of its time in the stage loop doing many loads, stores, and general
twiddle multiplies. A radix-8 first stage reduces one full radix-4 stage plus
part of the next stage into a single special kernel.

## Important Caveat

Radix-8 alone will not make the SVE2 code match Ne10.

Ne10 is also faster because it is:

- out-of-place, so it can write stage output in the layout needed by the next stage;
- hand-scheduled for 128-bit NEON register pressure;
- using grouped butterflies, for example `radix8x4`, meaning four radix-8 butterflies per loop;
- avoiding the separate split real/imag workspace layout used by the current SVE2 code;
- avoiding a digit-reverse scatter into split arrays.

So the right SVE2 direction is not just "replace radix-4 math with radix-8 math".
The right direction is to copy the Ne10 dataflow more closely.

## Proposed SVE2 Design

### 1. Add an Out-of-Place SVE2 Workspace

Use interleaved complex buffers, matching Ne10:

```c
typedef struct {
    FFT1024_ALIGNED(float buf0[2 * FFT1024_F32_N]);
    FFT1024_ALIGNED(float buf1[2 * FFT1024_F32_N]);
} fft1024_f32_sve2_oop_workspace;
```

The current split workspace:

```text
re[1024], im[1024]
```

is good for simple SVE code, but it hurts us against Ne10 because the first
digit-reverse load and final interleaved store cost around 1300-1400 ticks
together on your measurements.

### 2. First Stage: `radix8x4_sve2_first_stage`

Implement a direct SVE2 version of Ne10's `ne10_radix8x4_neon()`:

```c
static void radix8x4_sve2_first_stage(float *dst, const float *src, int stride);
```

For FFT1024:

```text
stride = 1024 / 8 = 128 complex samples
```

Each loop should process four radix-8 butterflies, same as Ne10. With 128-bit
SVE, `svcntw() == 4`, so this maps naturally to one SVE vector per four
butterflies.

The load pattern is:

```text
x0 = src + 0 * stride
x2 = src + 1 * stride
x4 = src + 2 * stride
x6 = src + 3 * stride
x1 = src + 4 * stride
x3 = src + 5 * stride
x5 = src + 6 * stride
x7 = src + 7 * stride
```

Use `svld2_f32` for interleaved complex loads.

The radix-8 math should follow the Ne10 equations:

```text
s0 = x0 + x1
s1 = x0 - x1
s2 = x2 + x3
s3 = x2 - x3
s4 = x4 + x5
s5 = x4 - x5
s6 = x6 + x7
s7 = x6 - x7

s5 *= -j
s3 *= exp(-j*pi/4)
s7 *= exp(-j*3*pi/4)

combine into y0..y7
```

Use constants:

```c
const float c = 0.70710678118654752440f;
```

For SVE2:

```text
svadd_f32_z / svsub_f32_z
svneg_f32_z
svmul_f32_z for the 0.70710678 paths
```

No general twiddle table is needed in this first stage.

### 3. Preserve Ne10-Like Stage Layout

The output of the first stage should be written contiguously as groups:

```text
y0[0..3], y1[0..3], ..., y7[0..3]
```

This is why Ne10 has transpose/permutation code after the radix-8 arithmetic.
For SVE, this is the difficult part. On 128-bit SVE, use fixed-width-friendly
operations:

```text
svtrn1_f32 / svtrn2_f32
svuzp1_f32 / svuzp2_f32
svzip1_f32 / svzip2_f32
```

The first implementation can use scalar stores for this permutation if needed,
but the performance target requires vector stores equivalent to Ne10's
`vtrnq_f32` plus `vcombine_f32` sequence.

### 4. Remaining Stages

After radix-8 first stage:

```text
stage 1: radix-4 with twiddles
stage 2: radix-4 with twiddles
stage 3: radix-4 with twiddles
stage 4: radix-2 or folded final stage
```

Two options:

1. Exact mixed radix `8 x 4 x 4 x 4 x 2`.
2. Ne10-style adapted factor table if its generated factors show a different
   sequence for 1024.

Option 1 is simpler to validate. Option 2 is better if the goal is to mimic
Ne10 instruction-for-instruction at the algorithm level.

### 5. Benchmark Targets

Current measured SVE2 total:

```text
profiled total: about 7540 ticks
core:           about 6084 ticks
```

A radix-8 first-stage version should be considered useful only if it reaches:

```text
SVE2 total: <= 6500 ticks
SVE2 core:  <= 5200 ticks
```

To approach Ne10:

```text
SVE2 total target: 3500-4500 ticks
```

That likely requires both radix-8 and Ne10-like out-of-place interleaved layout.

## Recommended Implementation Order

1. Add an out-of-place SVE2 path beside the current split-layout path.
2. Add `radix8x4_sve2_first_stage()` only.
3. Keep the existing radix-4 SVE2 core for later stages initially, even if one
   conversion step is needed.
4. Validate against Ne10 output with max absolute error.
5. Then remove the conversion step and make all stages use the Ne10-like
   interleaved ping-pong buffers.

This keeps correctness under control while identifying whether radix-8 first
stage is worth pursuing on Cortex-A520.

