#include "fft1024_sve2.h"

#include <arm_sve.h>
#include <math.h>
#include <stdint.h>

#if !defined(__ARM_FEATURE_SVE)
#error "This file requires SVE/SVE2. Compile for AArch64 with -mcpu=cortex-a520 or -march=armv9.2-a+sve2."
#endif

#define FFT1024_PI 3.14159265358979323846264338327950288f
#define FFT1024_RADIX4_STAGES 5

typedef struct {
    int m[FFT1024_RADIX4_STAGES];
    int offsets[FFT1024_RADIX4_STAGES];
    FFT1024_ALIGNED(float tw_re[FFT1024_F32_N]);
    FFT1024_ALIGNED(float tw_im[FFT1024_F32_N]);
} fft1024_plan;

static fft1024_plan g_plan;
static int g_init_done;

static void init_plan(void)
{
    int off = 0;

    for (int s = 0; s < FFT1024_RADIX4_STAGES; s++) {
        const int m = 1 << (2 * (s + 1));
        const int q = m >> 2;

        g_plan.m[s] = m;
        g_plan.offsets[s] = off;

        for (int j = 0; j < q; j++) {
            for (int k = 1; k <= 3; k++) {
                const float a = -2.0f * FFT1024_PI * (float)(k * j) / (float)m;
                g_plan.tw_re[off + (k - 1) * q + j] = cosf(a);
                g_plan.tw_im[off + (k - 1) * q + j] = sinf(a);
            }
        }

        off += 3 * q;
    }
}

void fft1024_f32_sve2_init(void)
{
    if (!g_init_done) {
        init_plan();
        g_init_done = 1;
    }
}

static unsigned reverse_base4_5digits(unsigned x)
{
    unsigned y = 0;

    for (int i = 0; i < FFT1024_RADIX4_STAGES; i++) {
        y = (y << 2) | (x & 3u);
        x >>= 2;
    }

    return y;
}

static void digit_reverse_base4(float *re, float *im)
{
    for (unsigned i = 1; i < FFT1024_F32_N - 1; i++) {
        const unsigned j = reverse_base4_5digits(i);
        if (i < j) {
            const float tr = re[i];
            const float ti = im[i];
            re[i] = re[j];
            im[i] = im[j];
            re[j] = tr;
            im[j] = ti;
        }
    }
}

static void split_load_interleaved_1024(const float *x, float *re, float *im)
{
    uint64_t i = 0;

    while (i < FFT1024_F32_N) {
        svbool_t pg = svwhilelt_b32(i, (uint64_t)FFT1024_F32_N);
        svfloat32x2_t z = svld2_f32(pg, &x[2 * i]);
        svst1_f32(pg, &re[i], svget2_f32(z, 0));
        svst1_f32(pg, &im[i], svget2_f32(z, 1));
        i += svcntw();
    }
}

static void split_store_interleaved_1024(float *x, const float *re, const float *im)
{
    uint64_t i = 0;

    while (i < FFT1024_F32_N) {
        svbool_t pg = svwhilelt_b32(i, (uint64_t)FFT1024_F32_N);
        svfloat32_t vr = svld1_f32(pg, &re[i]);
        svfloat32_t vi = svld1_f32(pg, &im[i]);
        svst2_f32(pg, &x[2 * i], svcreate2_f32(vr, vi));
        i += svcntw();
    }
}

static void cfft1024_split_radix4(float *re, float *im, int inverse)
{
    digit_reverse_base4(re, im);

    for (int stage = 0; stage < FFT1024_RADIX4_STAGES; stage++) {
        const int m = g_plan.m[stage];
        const int q = m >> 2;
        const int off = g_plan.offsets[stage];
        const int w1off = off;
        const int w2off = off + q;
        const int w3off = off + 2 * q;

        for (int base = 0; base < FFT1024_F32_N; base += m) {
            int j = 0;

            while (j < q) {
                svbool_t pg = svwhilelt_b32((uint32_t)j, (uint32_t)q);

                svfloat32_t x0r = svld1_f32(pg, &re[base + j]);
                svfloat32_t x0i = svld1_f32(pg, &im[base + j]);
                svfloat32_t x1r = svld1_f32(pg, &re[base + q + j]);
                svfloat32_t x1i = svld1_f32(pg, &im[base + q + j]);
                svfloat32_t x2r = svld1_f32(pg, &re[base + 2 * q + j]);
                svfloat32_t x2i = svld1_f32(pg, &im[base + 2 * q + j]);
                svfloat32_t x3r = svld1_f32(pg, &re[base + 3 * q + j]);
                svfloat32_t x3i = svld1_f32(pg, &im[base + 3 * q + j]);

                svfloat32_t w1r = svld1_f32(pg, &g_plan.tw_re[w1off + j]);
                svfloat32_t w1i = svld1_f32(pg, &g_plan.tw_im[w1off + j]);
                svfloat32_t w2r = svld1_f32(pg, &g_plan.tw_re[w2off + j]);
                svfloat32_t w2i = svld1_f32(pg, &g_plan.tw_im[w2off + j]);
                svfloat32_t w3r = svld1_f32(pg, &g_plan.tw_re[w3off + j]);
                svfloat32_t w3i = svld1_f32(pg, &g_plan.tw_im[w3off + j]);

                if (inverse) {
                    w1i = svneg_f32_z(pg, w1i);
                    w2i = svneg_f32_z(pg, w2i);
                    w3i = svneg_f32_z(pg, w3i);
                }

                svfloat32_t a1r = svsub_f32_z(pg,
                                              svmul_f32_z(pg, x1r, w1r),
                                              svmul_f32_z(pg, x1i, w1i));
                svfloat32_t a1i = svadd_f32_z(pg,
                                              svmul_f32_z(pg, x1r, w1i),
                                              svmul_f32_z(pg, x1i, w1r));
                svfloat32_t a2r = svsub_f32_z(pg,
                                              svmul_f32_z(pg, x2r, w2r),
                                              svmul_f32_z(pg, x2i, w2i));
                svfloat32_t a2i = svadd_f32_z(pg,
                                              svmul_f32_z(pg, x2r, w2i),
                                              svmul_f32_z(pg, x2i, w2r));
                svfloat32_t a3r = svsub_f32_z(pg,
                                              svmul_f32_z(pg, x3r, w3r),
                                              svmul_f32_z(pg, x3i, w3i));
                svfloat32_t a3i = svadd_f32_z(pg,
                                              svmul_f32_z(pg, x3r, w3i),
                                              svmul_f32_z(pg, x3i, w3r));

                svfloat32_t s02r = svadd_f32_z(pg, x0r, a2r);
                svfloat32_t s02i = svadd_f32_z(pg, x0i, a2i);
                svfloat32_t d02r = svsub_f32_z(pg, x0r, a2r);
                svfloat32_t d02i = svsub_f32_z(pg, x0i, a2i);
                svfloat32_t s13r = svadd_f32_z(pg, a1r, a3r);
                svfloat32_t s13i = svadd_f32_z(pg, a1i, a3i);
                svfloat32_t d13r = svsub_f32_z(pg, a1r, a3r);
                svfloat32_t d13i = svsub_f32_z(pg, a1i, a3i);

                svfloat32_t y0r = svadd_f32_z(pg, s02r, s13r);
                svfloat32_t y0i = svadd_f32_z(pg, s02i, s13i);
                svfloat32_t y2r = svsub_f32_z(pg, s02r, s13r);
                svfloat32_t y2i = svsub_f32_z(pg, s02i, s13i);
                svfloat32_t y1r;
                svfloat32_t y1i;
                svfloat32_t y3r;
                svfloat32_t y3i;

                if (inverse) {
                    y1r = svsub_f32_z(pg, d02r, d13i);
                    y1i = svadd_f32_z(pg, d02i, d13r);
                    y3r = svadd_f32_z(pg, d02r, d13i);
                    y3i = svsub_f32_z(pg, d02i, d13r);
                } else {
                    y1r = svadd_f32_z(pg, d02r, d13i);
                    y1i = svsub_f32_z(pg, d02i, d13r);
                    y3r = svsub_f32_z(pg, d02r, d13i);
                    y3i = svadd_f32_z(pg, d02i, d13r);
                }

                svst1_f32(pg, &re[base + j], y0r);
                svst1_f32(pg, &im[base + j], y0i);
                svst1_f32(pg, &re[base + q + j], y1r);
                svst1_f32(pg, &im[base + q + j], y1i);
                svst1_f32(pg, &re[base + 2 * q + j], y2r);
                svst1_f32(pg, &im[base + 2 * q + j], y2i);
                svst1_f32(pg, &re[base + 3 * q + j], y3r);
                svst1_f32(pg, &im[base + 3 * q + j], y3i);

                j += (int)svcntw();
            }
        }
    }

    if (inverse) {
        const float scale = 1.0f / (float)FFT1024_F32_N;
        svfloat32_t vs = svdup_f32(scale);
        uint64_t i = 0;

        while (i < FFT1024_F32_N) {
            svbool_t pg = svwhilelt_b32(i, (uint64_t)FFT1024_F32_N);
            svst1_f32(pg, &re[i], svmul_f32_z(pg, svld1_f32(pg, &re[i]), vs));
            svst1_f32(pg, &im[i], svmul_f32_z(pg, svld1_f32(pg, &im[i]), vs));
            i += svcntw();
        }
    }
}

void cfft1024_f32_sve2(float *x_interleaved, fft1024_f32_workspace *ws)
{
    fft1024_f32_sve2_init();
    split_load_interleaved_1024(x_interleaved, ws->re, ws->im);
    cfft1024_split_radix4(ws->re, ws->im, 0);
    split_store_interleaved_1024(x_interleaved, ws->re, ws->im);
}

void icfft1024_f32_sve2(float *x_interleaved, fft1024_f32_workspace *ws)
{
    fft1024_f32_sve2_init();
    split_load_interleaved_1024(x_interleaved, ws->re, ws->im);
    cfft1024_split_radix4(ws->re, ws->im, 1);
    split_store_interleaved_1024(x_interleaved, ws->re, ws->im);
}
