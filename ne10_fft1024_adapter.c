#include "ne10_fft1024_adapter.h"

#if defined(FFT1024_USE_NE10)

#include "CMSIS_NE10_fft.h"
#include "CMSIS_NE10_macros.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

extern arm_cfft_instance_f32 *arm_cfft_init_dynamic_f32(uint32_t fftLen);

static arm_cfft_instance_f32 *g_ne10_cfg;
static ne10_fft_cpx_float32_t *g_ne10_buffer;

int ne10_fft1024_adapter_init(void)
{
    g_ne10_cfg = arm_cfft_init_dynamic_f32(1024);
    if (!g_ne10_cfg) {
        fprintf(stderr, "arm_cfft_init_dynamic_f32(1024) failed\n");
        return -1;
    }

    g_ne10_buffer =
        (ne10_fft_cpx_float32_t *)NE10_MALLOC(1024 * sizeof(*g_ne10_buffer));
    if (!g_ne10_buffer) {
        fprintf(stderr, "NE10 buffer allocation failed\n");
        NE10_FREE(g_ne10_cfg);
        g_ne10_cfg = NULL;
        return -1;
    }

    return 0;
}

void ne10_fft1024_adapter_destroy(void)
{
    if (g_ne10_cfg) {
        NE10_FREE(g_ne10_cfg);
        g_ne10_cfg = NULL;
    }
    if (g_ne10_buffer) {
        NE10_FREE(g_ne10_buffer);
        g_ne10_buffer = NULL;
    }
}

void ne10_fft1024_cfft_f32_neon(float *dst_interleaved,
                                const float *src_interleaved)
{
    arm_ne10_mixed_radix_fft_forward_float32_neon(
        g_ne10_cfg,
        (const ne10_fft_cpx_float32_t *)src_interleaved,
        (ne10_fft_cpx_float32_t *)dst_interleaved,
        g_ne10_buffer);
}

#else

int ne10_fft1024_adapter_init(void)
{
    return -1;
}

void ne10_fft1024_adapter_destroy(void)
{
}

void ne10_fft1024_cfft_f32_neon(float *dst_interleaved,
                                const float *src_interleaved)
{
    (void)dst_interleaved;
    (void)src_interleaved;
}

#endif
