#ifndef NE10_FFT1024_ADAPTER_H
#define NE10_FFT1024_ADAPTER_H

#ifdef __cplusplus
extern "C" {
#endif

int ne10_fft1024_adapter_init(void);
void ne10_fft1024_adapter_destroy(void);
void ne10_fft1024_cfft_f32_neon(float *dst_interleaved,
                                const float *src_interleaved);

#ifdef __cplusplus
}
#endif

#endif
