#include "fft1024_sve2.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(FFT1024_USE_NE10)
#include <NE10.h>
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
    for (int i = 0; i < FFT1024_F32_N; i++) {
        x[2 * i] = 0.75f * sinf(0.017f * (float)i) +
                   0.25f * cosf(0.031f * (float)i);
        x[2 * i + 1] = 0.50f * cosf(0.011f * (float)i);
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
                                  fft1024_f32_workspace *ws,
                                  int iters)
{
    const size_t bytes = 2u * FFT1024_F32_N * sizeof(float);
    uint64_t t0 = read_counter();

    for (int i = 0; i < iters; i++) {
        memcpy(dst, src, bytes);
        cfft1024_f32_sve2(dst, ws);
    }

    uint64_t t1 = read_counter();
    const uint64_t copy_ticks = measure_memcpy_ticks(dst, src, bytes, iters);
    const uint64_t total = (t1 - t0) / (uint64_t)iters;
    return total > copy_ticks ? total - copy_ticks : total;
}

#if defined(FFT1024_USE_NE10)
static ne10_fft_cfg_float32_t g_ne10_cfg;

static int ne10_fft_init_1024(void)
{
    if (ne10_init() != NE10_OK) {
        fprintf(stderr, "ne10_init failed\n");
        return -1;
    }

    g_ne10_cfg = ne10_fft_alloc_c2c_float32(FFT1024_F32_N);
    if (!g_ne10_cfg) {
        fprintf(stderr, "ne10_fft_alloc_c2c_float32 failed\n");
        return -1;
    }

    return 0;
}

static uint64_t measure_cfft_ne10_neon(float *dst, const float *src, int iters)
{
    const size_t bytes = 2u * FFT1024_F32_N * sizeof(float);
    uint64_t t0 = read_counter();

    for (int i = 0; i < iters; i++) {
        ne10_fft_c2c_1d_float32_neon((ne10_fft_cpx_float32_t *)dst,
                                     (ne10_fft_cpx_float32_t *)src,
                                     g_ne10_cfg,
                                     0);
    }

    uint64_t t1 = read_counter();
    const uint64_t copy_ticks = measure_memcpy_ticks(dst, src, bytes, iters);
    const uint64_t total = (t1 - t0) / (uint64_t)iters;

    (void)copy_ticks;
    return total;
}
#endif

int main(int argc, char **argv)
{
    int iters = 1000;
    if (argc > 1) {
        iters = atoi(argv[1]);
        if (iters <= 0) {
            iters = 1000;
        }
    }

    fft1024_f32_workspace *ws =
        (fft1024_f32_workspace *)aligned_malloc_or_die(FFT1024_F32_ALIGNMENT,
                                                       sizeof(*ws));
    float *src = (float *)aligned_malloc_or_die(FFT1024_F32_ALIGNMENT,
                                                2u * FFT1024_F32_N * sizeof(float));
    float *work = (float *)aligned_malloc_or_die(FFT1024_F32_ALIGNMENT,
                                                 2u * FFT1024_F32_N * sizeof(float));

    fft1024_f32_sve2_init();
    fill_complex(src);

    memcpy(work, src, 2u * FFT1024_F32_N * sizeof(float));
    cfft1024_f32_sve2(work, ws);
    icfft1024_f32_sve2(work, ws);
    printf("CFFT1024 SVE2 round-trip max abs error: %.8g\n",
           max_abs_error(src, work, 2 * FFT1024_F32_N));

#if defined(FFT1024_USE_NE10)
    if (ne10_fft_init_1024() != 0) {
        return 1;
    }
#endif

    const uint64_t sve2_ticks = measure_cfft_sve2(work, src, ws, iters);
    uint64_t ne10_neon_ticks = 0;

#if defined(FFT1024_USE_NE10)
    ne10_neon_ticks = measure_cfft_ne10_neon(work, src, iters);
#endif

#if defined(FFT_BENCH_USE_PMCCNTR)
    printf("Counter source: PMCCNTR_EL0 cycles\n");
#else
    printf("Counter source: CNTVCT_EL0 ticks. Use perf stat for architectural cycles.\n");
#endif
    printf("Iterations: %d\n\n", iters);
    printf("Kernel             ticks/call     ratio vs Ne10\n");
    printf("CFFT1024 SVE2      %10llu",
           (unsigned long long)sve2_ticks);
    if (ne10_neon_ticks) {
        printf("     %9.4f\n", (double)sve2_ticks / (double)ne10_neon_ticks);
    } else {
        printf("     unavailable\n");
    }

    printf("CFFT1024 Ne10 NEON ");
    if (ne10_neon_ticks) {
        printf("%10llu     %9.4f\n",
               (unsigned long long)ne10_neon_ticks,
               1.0);
    } else {
        printf("unavailable     unavailable\n");
    }

    free(work);
    free(src);
    free(ws);
    return 0;
}
