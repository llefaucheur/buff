#include "fft1024_sve2.h"
#include "ne10_fft1024_adapter.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static uint64_t measure_cfft_ne10(float *dst, const float *src, int iters)
{
#if defined(FFT1024_USE_NE10)
    uint64_t t0 = read_counter();

    for (int i = 0; i < iters; i++) {
        ne10_fft1024_cfft_f32_neon(dst, src);
    }

    uint64_t t1 = read_counter();
    return (t1 - t0) / (uint64_t)iters;
#else
    (void)dst;
    (void)src;
    (void)iters;
    return 0;
#endif
}

static void print_sve2_breakdown(float *dst, const float *src,
                                 fft1024_f32_workspace *ws, int iters)
{
    const size_t bytes = 2u * FFT1024_F32_N * sizeof(float);
    unsigned long long split_total = 0;
    unsigned long long reorder_total = 0;
    unsigned long long core_total = 0;
    unsigned long long store_total = 0;
    unsigned long long profiled_total = 0;

    for (int i = 0; i < iters; i++) {
        unsigned long long split_ticks;
        unsigned long long reorder_ticks;
        unsigned long long core_ticks;
        unsigned long long store_ticks;

        memcpy(dst, src, bytes);
        const uint64_t t0 = read_counter();
        cfft1024_f32_sve2_profile_parts(dst, ws,
                                        &split_ticks,
                                        &reorder_ticks,
                                        &core_ticks,
                                        &store_ticks);
        const uint64_t t1 = read_counter();
        split_total += split_ticks;
        reorder_total += reorder_ticks;
        core_total += core_ticks;
        store_total += store_ticks;
        profiled_total += (unsigned long long)(t1 - t0);
    }

    printf("\nSVE2 breakdown, ticks/call:\n");
    printf("  profiled total:      %10llu\n",
           profiled_total / (unsigned long long)iters);
    printf("  sum of parts:        %10llu\n",
           (split_total + reorder_total + core_total + store_total) /
               (unsigned long long)iters);
    printf("  digitrev input load: %10llu\n",
           split_total / (unsigned long long)iters);
    printf("  separate reorder:    %10llu\n",
           reorder_total / (unsigned long long)iters);
    printf("  radix-4 core:        %10llu\n",
           core_total / (unsigned long long)iters);
    printf("  interleaved store:   %10llu\n",
           store_total / (unsigned long long)iters);
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
    if (ne10_fft1024_adapter_init() != 0) {
        return 1;
    }
#endif

    const uint64_t sve2_ticks = measure_cfft_sve2(work, src, ws, iters);
    const uint64_t ne10_ticks = measure_cfft_ne10(work, src, iters);

#if defined(FFT_BENCH_USE_PMCCNTR)
    printf("Counter source: PMCCNTR_EL0 cycles\n");
#else
    printf("Counter source: CNTVCT_EL0 ticks. Use perf stat for architectural cycles.\n");
#endif
    printf("Iterations: %d\n\n", iters);
    printf("Kernel             ticks/call     SVE2/NE10\n");
    printf("CFFT1024 SVE2      %10llu",
           (unsigned long long)sve2_ticks);
    if (ne10_ticks) {
        printf("     %9.4f\n", (double)sve2_ticks / (double)ne10_ticks);
    } else {
        printf("     unavailable\n");
    }

    printf("CFFT1024 Ne10 NEON ");
    if (ne10_ticks) {
        printf("%10llu     %9.4f\n",
               (unsigned long long)ne10_ticks,
               1.0);
    } else {
        printf("unavailable     unavailable\n");
    }

    print_sve2_breakdown(work, src, ws, iters < 100 ? iters : 100);

#if defined(FFT1024_USE_NE10)
    ne10_fft1024_adapter_destroy();
#endif

    free(work);
    free(src);
    free(ws);
    return 0;
}
