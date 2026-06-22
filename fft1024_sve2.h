#ifndef FFT1024_SVE2_H
#define FFT1024_SVE2_H

#ifdef __cplusplus
extern "C" {
#endif

#define FFT1024_F32_N 1024
#define FFT1024_F32_ALIGNMENT 64

#if defined(__GNUC__) || defined(__clang__)
#define FFT1024_ALIGNED(x) x __attribute__((aligned(FFT1024_F32_ALIGNMENT)))
#else
#define FFT1024_ALIGNED(x) x
#endif

typedef struct {
    FFT1024_ALIGNED(float re[FFT1024_F32_N]);
    FFT1024_ALIGNED(float im[FFT1024_F32_N]);
} FFT1024_ALIGNED(fft1024_f32_workspace);

void fft1024_f32_sve2_init(void);
void cfft1024_f32_sve2(float *x_interleaved, fft1024_f32_workspace *ws);
void icfft1024_f32_sve2(float *x_interleaved, fft1024_f32_workspace *ws);

void cfft1024_f32_sve2_profile_parts(float *x_interleaved,
                                      fft1024_f32_workspace *ws,
                                      unsigned long long *split_ticks,
                                      unsigned long long *reorder_ticks,
                                      unsigned long long *core_ticks,
                                      unsigned long long *store_ticks);

#ifdef __cplusplus
}
#endif

#endif
