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
static uint16_t g_digit_rev[FFT1024_F32_N];
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

static unsigned reverse_base4_5digits(unsigned x)
{
    unsigned y = 0;

    for (int i = 0; i < FFT1024_RADIX4_STAGES; i++) {
        y = (y << 2) | (x & 3u);
        x >>= 2;
    }

    return y;
}

static void init_digit_reverse_table(void)
{
    for (unsigned i = 0; i < FFT1024_F32_N; i++) {
        g_digit_rev[i] = (uint16_t)reverse_base4_5digits(i);
    }
}

static void split_load_interleaved_digitrev_1024(const float *x, float *re, float *im)
{
    for (int i = 0; i < FFT1024_F32_N; i++) {
        const int r = g_digit_rev[i];
        re[r] = x[2 * i];
        im[r] = x[2 * i + 1];
    }
}

void fft1024_f32_sve2_init(void)
{
    if (!g_init_done) {
        init_plan();
        init_digit_reverse_table();
        g_init_done = 1;
    }
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

#if defined(__GNUC__) || defined(__clang__)
__attribute__((unused))
#endif
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

static void conj_scale_interleaved_1024(float *x, float scale)
{
    uint64_t i = 0;
    svfloat32_t vs = svdup_f32(scale);
    svfloat32_t vns = svdup_f32(-scale);

    while (i < FFT1024_F32_N) {
        svbool_t pg = svwhilelt_b32(i, (uint64_t)FFT1024_F32_N);
        svfloat32x2_t z = svld2_f32(pg, &x[2 * i]);
        svfloat32_t vr = svmul_f32_z(pg, svget2_f32(z, 0), vs);
        svfloat32_t vi = svmul_f32_z(pg, svget2_f32(z, 1), vns);
        svst2_f32(pg, &x[2 * i], svcreate2_f32(vr, vi));
        i += svcntw();
    }
}

static void conj_interleaved_1024(float *x)
{
    conj_scale_interleaved_1024(x, 1.0f);
}

static inline void radix4_notwiddle_scalar(float *re, float *im,
                                           int base, int q, int inverse)
{
    const int i0 = base;
    const int i1 = base + q;
    const int i2 = base + 2 * q;
    const int i3 = base + 3 * q;

    const float x0r = re[i0];
    const float x0i = im[i0];
    const float x1r = re[i1];
    const float x1i = im[i1];
    const float x2r = re[i2];
    const float x2i = im[i2];
    const float x3r = re[i3];
    const float x3i = im[i3];

    const float s02r = x0r + x2r;
    const float s02i = x0i + x2i;
    const float d02r = x0r - x2r;
    const float d02i = x0i - x2i;
    const float s13r = x1r + x3r;
    const float s13i = x1i + x3i;
    const float d13r = x1r - x3r;
    const float d13i = x1i - x3i;

    re[i0] = s02r + s13r;
    im[i0] = s02i + s13i;
    re[i2] = s02r - s13r;
    im[i2] = s02i - s13i;

    if (inverse) {
        re[i1] = d02r - d13i;
        im[i1] = d02i + d13r;
        re[i3] = d02r + d13i;
        im[i3] = d02i - d13r;
    } else {
        re[i1] = d02r + d13i;
        im[i1] = d02i - d13r;
        re[i3] = d02r - d13i;
        im[i3] = d02i + d13r;
    }
}

static inline void radix4_notwiddle_forward_scalar(float *re, float *im,
                                                   int base, int q)
{
    const int i0 = base;
    const int i1 = base + q;
    const int i2 = base + 2 * q;
    const int i3 = base + 3 * q;

    const float x0r = re[i0];
    const float x0i = im[i0];
    const float x1r = re[i1];
    const float x1i = im[i1];
    const float x2r = re[i2];
    const float x2i = im[i2];
    const float x3r = re[i3];
    const float x3i = im[i3];

    const float s02r = x0r + x2r;
    const float s02i = x0i + x2i;
    const float d02r = x0r - x2r;
    const float d02i = x0i - x2i;
    const float s13r = x1r + x3r;
    const float s13i = x1i + x3i;
    const float d13r = x1r - x3r;
    const float d13i = x1i - x3i;

    re[i0] = s02r + s13r;
    im[i0] = s02i + s13i;
    re[i2] = s02r - s13r;
    im[i2] = s02i - s13i;
    re[i1] = d02r + d13i;
    im[i1] = d02i - d13r;
    re[i3] = d02r - d13i;
    im[i3] = d02i + d13r;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((always_inline))
#endif
static inline void cmul_forward_mla_f32(svbool_t pg,
                                        svfloat32_t ar,
                                        svfloat32_t ai,
                                        svfloat32_t br,
                                        svfloat32_t bi,
                                        svfloat32_t *rr,
                                        svfloat32_t *ri)
{
    svfloat32_t tr = svmul_f32_x(pg, ar, br);
    svfloat32_t ti = svmul_f32_x(pg, ai, br);

    tr = svmls_f32_x(pg, tr, ai, bi);
    ti = svmla_f32_x(pg, ti, ar, bi);

    *rr = tr;
    *ri = ti;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((unused))
#endif
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
            int j = 1;

            radix4_notwiddle_scalar(re, im, base, q, inverse);

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

                svfloat32_t a1r;
                svfloat32_t a1i;
                svfloat32_t a2r;
                svfloat32_t a2i;
                svfloat32_t a3r;
                svfloat32_t a3i;

                cmul_forward_mla_f32(pg, x1r, x1i, w1r, w1i, &a1r, &a1i);
                cmul_forward_mla_f32(pg, x2r, x2i, w2r, w2i, &a2r, &a2i);
                cmul_forward_mla_f32(pg, x3r, x3i, w3r, w3i, &a3r, &a3i);

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

static inline uint64_t fft1024_read_counter(void)
{
    uint64_t v;
    __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

static void cfft1024_split_radix4_core_ordered(float *re, float *im)
{
    for (int stage = 0; stage < FFT1024_RADIX4_STAGES; stage++) {
        const int m = g_plan.m[stage];
        const int q = m >> 2;
        const int off = g_plan.offsets[stage];
        const int w1off = off;
        const int w2off = off + q;
        const int w3off = off + 2 * q;

        for (int base = 0; base < FFT1024_F32_N; base += m) {
            int j = 1;

            radix4_notwiddle_forward_scalar(re, im, base, q);

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

                svfloat32_t a1r;
                svfloat32_t a1i;
                svfloat32_t a2r;
                svfloat32_t a2i;
                svfloat32_t a3r;
                svfloat32_t a3i;

                cmul_forward_mla_f32(pg, x1r, x1i, w1r, w1i, &a1r, &a1i);
                cmul_forward_mla_f32(pg, x2r, x2i, w2r, w2i, &a2r, &a2i);
                cmul_forward_mla_f32(pg, x3r, x3i, w3r, w3i, &a3r, &a3i);

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
                svfloat32_t y1r = svadd_f32_z(pg, d02r, d13i);
                svfloat32_t y1i = svsub_f32_z(pg, d02i, d13r);
                svfloat32_t y3r = svsub_f32_z(pg, d02r, d13i);
                svfloat32_t y3i = svadd_f32_z(pg, d02i, d13r);

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
}

void cfft1024_f32_sve2(float *x_interleaved, fft1024_f32_workspace *ws)
{
    fft1024_f32_sve2_init();
    split_load_interleaved_digitrev_1024(x_interleaved, ws->re, ws->im);
    cfft1024_split_radix4_core_ordered(ws->re, ws->im);
    split_store_interleaved_1024(x_interleaved, ws->re, ws->im);
}

void cfft1024_f32_sve2_profile_parts(float *x_interleaved,
                                      fft1024_f32_workspace *ws,
                                      unsigned long long *split_ticks,
                                      unsigned long long *reorder_ticks,
                                      unsigned long long *core_ticks,
                                      unsigned long long *store_ticks)
{
    fft1024_f32_sve2_init();

    uint64_t t0 = fft1024_read_counter();
    split_load_interleaved_digitrev_1024(x_interleaved, ws->re, ws->im);
    uint64_t t1 = fft1024_read_counter();
    uint64_t t2 = t1;
    cfft1024_split_radix4_core_ordered(ws->re, ws->im);
    uint64_t t3 = fft1024_read_counter();
    split_store_interleaved_1024(x_interleaved, ws->re, ws->im);
    uint64_t t4 = fft1024_read_counter();

    *split_ticks = (unsigned long long)(t1 - t0);
    *reorder_ticks = (unsigned long long)(t2 - t1);
    *core_ticks = (unsigned long long)(t3 - t2);
    *store_ticks = (unsigned long long)(t4 - t3);
}

void icfft1024_f32_sve2(float *x_interleaved, fft1024_f32_workspace *ws)
{
    conj_interleaved_1024(x_interleaved);
    cfft1024_f32_sve2(x_interleaved, ws);
    conj_scale_interleaved_1024(x_interleaved, 1.0f / (float)FFT1024_F32_N);
}
