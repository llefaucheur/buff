#ifndef FFT2048_SVE2_H
#define FFT2048_SVE2_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FFT2048_F32_N 2048
#define FFT2048_F32_RFFT_BINS 1025
#define FFT2048_F32_ALIGNMENT 64

#if defined(__GNUC__) || defined(__clang__)
#define FFT2048_ALIGNED(x) x __attribute__((aligned(FFT2048_F32_ALIGNMENT)))
#else
#define FFT2048_ALIGNED(x) x
#endif

typedef struct {
    FFT2048_ALIGNED(float re[FFT2048_F32_N]);
    FFT2048_ALIGNED(float im[FFT2048_F32_N]);
} FFT2048_ALIGNED(fft2048_f32_workspace);

void fft2048_f32_sve2_init(void);

void cfft2048_f32_sve2(float *x_interleaved, fft2048_f32_workspace *ws);
void icfft2048_f32_sve2(float *x_interleaved, fft2048_f32_workspace *ws);

void rfft2048_f32_sve2(const float *x_real,
                       float *x_bins_interleaved,
                       fft2048_f32_workspace *ws);

#ifdef __cplusplus
}
#endif

#endif
