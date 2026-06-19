#include "fft2048_sve2.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GNUC__)
extern void cfft2048_f32_neon(float *x_interleaved) __attribute__((weak));
extern void rfft2048_f32_neon(const float *x_real,
                              float *x_bins_interleaved) __attribute__((weak));
#endif

static inline uint64_t read_counter(void)
{
#if defined(FFT_BENCH_USE_PMCCNTR)
    uint64_t v;
    __asm__ volatile("isb; mrs %0, pmccntr_el0" : "=r"(v));
    return v;
#else
    uint64_t v;
    __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(v));
    return v;
#endif
}

static void *aligned_malloc_or_die(size_t alignment, size_t size)
{
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0 || p == NULL) {
        fprintf(stderr, "posix_memalign failed\n");
        exit(1);
    }
    return p;
}

static void fill_complex(float *x)
{
    for (int i = 0; i < FFT2048_F32_N; i++) {
        x[2 * i] = 0.75f * sinf(0.017f * (float)i) +
                   0.25f * cosf(0.031f * (float)i);
        x[2 * i + 1] = 0.50f * cosf(0.011f * (float)i);
    }
}

static void fill_real(float *x)
{
    for (int i = 0; i < FFT2048_F32_N; i++) {
        x[i] = 0.75f * sinf(0.017f * (float)i) +
               0.25f * cosf(0.031f * (float)i);
    }
}

static double max_abs_error(const float *a, const float *b, int count)
{
    double e = 0.0;
    for (int i = 0; i < count; i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > e) {
            e = d;
        }
    }
    return e;
}

static uint64_t measure_memcpy_ticks(float *dst, const float *src,
                                     size_t bytes, int iters)
{
    uint64_t t0 = read_counter();
    for (int i = 0; i < iters; i++) {
        memcpy(dst, src, bytes);
    }
    uint64_t t1 = read_counter();
    return (t1 - t0) / (uint64_t)iters;
}

static uint64_t measure_cfft_sve2(float *dst, const float *src,
                                  fft2048_f32_workspace *ws,
                                  int iters)
{
    const size_t bytes = 2u * FFT2048_F32_N * sizeof(float);
    uint64_t t0 = read_counter();
    for (int i = 0; i < iters; i++) {
        memcpy(dst, src, bytes);
        cfft2048_f32_sve2(dst, ws);
    }
    uint64_t t1 = read_counter();

    const uint64_t copy_ticks = measure_memcpy_ticks(dst, src, bytes, iters);
    const uint64_t total = (t1 - t0) / (uint64_t)iters;
    return total > copy_ticks ? total - copy_ticks : total;
}

static uint64_t measure_cfft_neon(float *dst, const float *src, int iters)
{
    const size_t bytes = 2u * FFT2048_F32_N * sizeof(float);

    if (!cfft2048_f32_neon) {
        return 0;
    }

    uint64_t t0 = read_counter();
    for (int i = 0; i < iters; i++) {
        memcpy(dst, src, bytes);
        cfft2048_f32_neon(dst);
    }
    uint64_t t1 = read_counter();

    const uint64_t copy_ticks = measure_memcpy_ticks(dst, src, bytes, iters);
    const uint64_t total = (t1 - t0) / (uint64_t)iters;
    return total > copy_ticks ? total - copy_ticks : total;
}

static uint64_t measure_rfft_sve2(const float *src, float *bins,
                                  fft2048_f32_workspace *ws,
                                  int iters)
{
    uint64_t t0 = read_counter();
    for (int i = 0; i < iters; i++) {
        rfft2048_f32_sve2(src, bins, ws);
    }
    uint64_t t1 = read_counter();
    return (t1 - t0) / (uint64_t)iters;
}

static uint64_t measure_rfft_neon(const float *src, float *bins, int iters)
{
    if (!rfft2048_f32_neon) {
        return 0;
    }

    uint64_t t0 = read_counter();
    for (int i = 0; i < iters; i++) {
        rfft2048_f32_neon(src, bins);
    }
    uint64_t t1 = read_counter();
    return (t1 - t0) / (uint64_t)iters;
}

int main(int argc, char **argv)
{
    int iters = 1000;
    if (argc > 1) {
        iters = atoi(argv[1]);
        if (iters <= 0) {
            iters = 1000;
        }
    }

    fft2048_f32_workspace *ws =
        (fft2048_f32_workspace *)aligned_malloc_or_die(FFT2048_F32_ALIGNMENT,
                                                       sizeof(*ws));
    float *src_c = (float *)aligned_malloc_or_die(FFT2048_F32_ALIGNMENT,
                                                  2u * FFT2048_F32_N * sizeof(float));
    float *work_c = (float *)aligned_malloc_or_die(FFT2048_F32_ALIGNMENT,
                                                   2u * FFT2048_F32_N * sizeof(float));
    float *src_r = (float *)aligned_malloc_or_die(FFT2048_F32_ALIGNMENT,
                                                  FFT2048_F32_N * sizeof(float));
    float *bins = (float *)aligned_malloc_or_die(FFT2048_F32_ALIGNMENT,
                                                 2u * FFT2048_F32_RFFT_BINS * sizeof(float));

    fft2048_f32_sve2_init();
    fill_complex(src_c);
    fill_real(src_r);

    memcpy(work_c, src_c, 2u * FFT2048_F32_N * sizeof(float));
    cfft2048_f32_sve2(work_c, ws);
    icfft2048_f32_sve2(work_c, ws);
    printf("CFFT2048 SVE2 round-trip max abs error: %.8g\n",
           max_abs_error(src_c, work_c, 2 * FFT2048_F32_N));

    const uint64_t cfft_sve2 = measure_cfft_sve2(work_c, src_c, ws, iters);
    const uint64_t rfft_sve2 = measure_rfft_sve2(src_r, bins, ws, iters);
    const uint64_t cfft_neon = measure_cfft_neon(work_c, src_c, iters);
    const uint64_t rfft_neon = measure_rfft_neon(src_r, bins, iters);

#if defined(FFT_BENCH_USE_PMCCNTR)
    printf("Counter source: PMCCNTR_EL0 cycles\n");
#else
    printf("Counter source: CNTVCT_EL0 ticks. Use perf stat for architectural cycles.\n");
#endif
    printf("Iterations: %d\n", iters);
    printf("\n");
    printf("Kernel        SVE2        NEON        SVE2/NEON\n");
    printf("CFFT2048  %10llu", (unsigned long long)cfft_sve2);
    if (cfft_neon) {
        printf("  %10llu  %9.4f\n",
               (unsigned long long)cfft_neon,
               (double)cfft_sve2 / (double)cfft_neon);
    } else {
        printf("  unavailable  unavailable\n");
    }

    printf("RFFT2048  %10llu", (unsigned long long)rfft_sve2);
    if (rfft_neon) {
        printf("  %10llu  %9.4f\n",
               (unsigned long long)rfft_neon,
               (double)rfft_sve2 / (double)rfft_neon);
    } else {
        printf("  unavailable  unavailable\n");
    }

    free(bins);
    free(src_r);
    free(work_c);
    free(src_c);
    free(ws);

    return 0;
}
