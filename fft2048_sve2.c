#include "fft2048_sve2.h"

#include <arm_sve.h>
#include <math.h>
#include <stdint.h>

#if !defined(__ARM_FEATURE_SVE)
#error "This file requires SVE/SVE2. Compile for Cortex-A520 with -mcpu=cortex-a520 or -march=armv9.2-a+sve2."
#endif

#define FFT_PI 3.14159265358979323846264338327950288f
#define FFT_MAX_STAGES 11

typedef struct {
    int n;
    int stages;
    int offsets[FFT_MAX_STAGES];
    FFT2048_ALIGNED(float tw_re[FFT2048_F32_N - 1]);
    FFT2048_ALIGNED(float tw_im[FFT2048_F32_N - 1]);
} fft_plan_f32;

static fft_plan_f32 g_plan_2048;
static fft_plan_f32 g_plan_1024;
static FFT2048_ALIGNED(float g_rfft2048_tw_re[1024]);
static FFT2048_ALIGNED(float g_rfft2048_tw_im[1024]);
static int g_init_done;

static void init_plan(fft_plan_f32 *p, int n)
{
    int off = 0;
    int stages = 0;

    p->n = n;
    for (int t = n; t > 1; t >>= 1) {
        stages++;
    }
    p->stages = stages;

    for (int s = 1; s <= stages; s++) {
        const int m = 1 << s;
        const int half = m >> 1;

        p->offsets[s - 1] = off;
        for (int j = 0; j < half; j++) {
            const float a = -2.0f * FFT_PI * (float)j / (float)m;
            p->tw_re[off + j] = cosf(a);
            p->tw_im[off + j] = sinf(a);
        }
        off += half;
    }
}

void fft2048_f32_sve2_init(void)
{
    if (!g_init_done) {
        init_plan(&g_plan_2048, 2048);
        init_plan(&g_plan_1024, 1024);
        for (int k = 0; k < 1024; k++) {
            const float a = -2.0f * FFT_PI * (float)k / 2048.0f;
            g_rfft2048_tw_re[k] = cosf(a);
            g_rfft2048_tw_im[k] = sinf(a);
        }
        g_init_done = 1;
    }
}

static void split_load_interleaved_2048(const float *x, float *re, float *im)
{
    uint64_t i = 0;
    const uint64_t n = 2048;

    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32x2_t z = svld2_f32(pg, &x[2 * i]);
        svst1_f32(pg, &re[i], svget2_f32(z, 0));
        svst1_f32(pg, &im[i], svget2_f32(z, 1));
        i += svcntw();
    }
}

static void split_store_interleaved_2048(float *x, const float *re, const float *im)
{
    uint64_t i = 0;
    const uint64_t n = 2048;

    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t vr = svld1_f32(pg, &re[i]);
        svfloat32_t vi = svld1_f32(pg, &im[i]);
        svst2_f32(pg, &x[2 * i], svcreate2_f32(vr, vi));
        i += svcntw();
    }
}

static void bit_reverse_split(float *re, float *im, int n)
{
    int j = 0;

    for (int i = 1; i < n - 1; i++) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;

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

static void cfft_split_sve(float *re, float *im, const fft_plan_f32 *plan, int inverse)
{
    const int n = plan->n;

    bit_reverse_split(re, im, n);

    for (int s = 1; s <= plan->stages; s++) {
        const int m = 1 << s;
        const int half = m >> 1;
        const int off = plan->offsets[s - 1];

        for (int base = 0; base < n; base += m) {
            int j = 0;

            while (j < half) {
                svbool_t pg = svwhilelt_b32((uint32_t)j, (uint32_t)half);

                svfloat32_t ur = svld1_f32(pg, &re[base + j]);
                svfloat32_t ui = svld1_f32(pg, &im[base + j]);
                svfloat32_t vr = svld1_f32(pg, &re[base + half + j]);
                svfloat32_t vi = svld1_f32(pg, &im[base + half + j]);
                svfloat32_t wr = svld1_f32(pg, &plan->tw_re[off + j]);
                svfloat32_t wi = svld1_f32(pg, &plan->tw_im[off + j]);

                if (inverse) {
                    wi = svneg_f32_z(pg, wi);
                }

                svfloat32_t tr = svsub_f32_z(pg,
                                             svmul_f32_z(pg, vr, wr),
                                             svmul_f32_z(pg, vi, wi));
                svfloat32_t ti = svadd_f32_z(pg,
                                             svmul_f32_z(pg, vr, wi),
                                             svmul_f32_z(pg, vi, wr));

                svst1_f32(pg, &re[base + j], svadd_f32_z(pg, ur, tr));
                svst1_f32(pg, &im[base + j], svadd_f32_z(pg, ui, ti));
                svst1_f32(pg, &re[base + half + j], svsub_f32_z(pg, ur, tr));
                svst1_f32(pg, &im[base + half + j], svsub_f32_z(pg, ui, ti));

                j += (int)svcntw();
            }
        }
    }

    if (inverse) {
        const float scale = 1.0f / (float)n;
        svfloat32_t vs = svdup_f32(scale);
        uint64_t i = 0;

        while (i < (uint64_t)n) {
            svbool_t pg = svwhilelt_b32(i, (uint64_t)n);
            svst1_f32(pg, &re[i], svmul_f32_z(pg, svld1_f32(pg, &re[i]), vs));
            svst1_f32(pg, &im[i], svmul_f32_z(pg, svld1_f32(pg, &im[i]), vs));
            i += svcntw();
        }
    }
}

void cfft2048_f32_sve2(float *x_interleaved, fft2048_f32_workspace *ws)
{
    fft2048_f32_sve2_init();
    split_load_interleaved_2048(x_interleaved, ws->re, ws->im);
    cfft_split_sve(ws->re, ws->im, &g_plan_2048, 0);
    split_store_interleaved_2048(x_interleaved, ws->re, ws->im);
}

void icfft2048_f32_sve2(float *x_interleaved, fft2048_f32_workspace *ws)
{
    fft2048_f32_sve2_init();
    split_load_interleaved_2048(x_interleaved, ws->re, ws->im);
    cfft_split_sve(ws->re, ws->im, &g_plan_2048, 1);
    split_store_interleaved_2048(x_interleaved, ws->re, ws->im);
}

static void load_real_as_complex_1024(const float *x, float *re, float *im)
{
    uint64_t i = 0;
    const uint64_t n = 1024;

    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32x2_t z = svld2_f32(pg, &x[2 * i]);
        svst1_f32(pg, &re[i], svget2_f32(z, 0));
        svst1_f32(pg, &im[i], svget2_f32(z, 1));
        i += svcntw();
    }
}

void rfft2048_f32_sve2(const float *x_real,
                       float *x_bins_interleaved,
                       fft2048_f32_workspace *ws)
{
    const int m = 1024;

    fft2048_f32_sve2_init();

    load_real_as_complex_1024(x_real, ws->re, ws->im);
    cfft_split_sve(ws->re, ws->im, &g_plan_1024, 0);

    x_bins_interleaved[0] = ws->re[0] + ws->im[0];
    x_bins_interleaved[1] = 0.0f;
    x_bins_interleaved[2 * m] = ws->re[0] - ws->im[0];
    x_bins_interleaved[2 * m + 1] = 0.0f;

    int k = 1;
    const int vl = (int)svcntw();
    svbool_t pg = svptrue_b32();
    svfloat32_t half = svdup_f32(0.5f);

    for (; k + vl - 1 < m; k += vl) {
        const int rbase = m - k - vl + 1;

        svfloat32_t ar = svld1_f32(pg, &ws->re[k]);
        svfloat32_t ai = svld1_f32(pg, &ws->im[k]);
        svfloat32_t br = svrev_f32(svld1_f32(pg, &ws->re[rbase]));
        svfloat32_t bi = svneg_f32_z(pg, svrev_f32(svld1_f32(pg, &ws->im[rbase])));
        svfloat32_t wr = svld1_f32(pg, &g_rfft2048_tw_re[k]);
        svfloat32_t wi = svld1_f32(pg, &g_rfft2048_tw_im[k]);

        svfloat32_t a = svadd_f32_z(pg, ar, br);
        svfloat32_t b = svadd_f32_z(pg, ai, bi);
        svfloat32_t c = svsub_f32_z(pg, ar, br);
        svfloat32_t d = svsub_f32_z(pg, ai, bi);

        svfloat32_t xr = svadd_f32_z(pg, a,
                                     svadd_f32_z(pg,
                                                 svmul_f32_z(pg, wr, d),
                                                 svmul_f32_z(pg, wi, c)));
        svfloat32_t xi = svadd_f32_z(pg, b,
                                     svsub_f32_z(pg,
                                                 svmul_f32_z(pg, wi, d),
                                                 svmul_f32_z(pg, wr, c)));

        xr = svmul_f32_z(pg, xr, half);
        xi = svmul_f32_z(pg, xi, half);
        svst2_f32(pg, &x_bins_interleaved[2 * k], svcreate2_f32(xr, xi));
    }

    for (; k < m; k++) {
        const float ar = ws->re[k];
        const float ai = ws->im[k];
        const float br = ws->re[m - k];
        const float bi = -ws->im[m - k];
        const float wr = g_rfft2048_tw_re[k];
        const float wi = g_rfft2048_tw_im[k];

        const float a = ar + br;
        const float b = ai + bi;
        const float c = ar - br;
        const float d = ai - bi;

        x_bins_interleaved[2 * k] = 0.5f * (a + wr * d + wi * c);
        x_bins_interleaved[2 * k + 1] = 0.5f * (b - wr * c + wi * d);
    }
}
