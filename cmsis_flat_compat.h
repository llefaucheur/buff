#ifndef CMSIS_FLAT_COMPAT_H
#define CMSIS_FLAT_COMPAT_H

/*
 * Minimal CMSIS-Core compatibility layer for building the CMSIS-DSP Ne10
 * sources as a flat standalone package on AArch64 Linux.
 */

#include <stdint.h>

#ifndef __STATIC_INLINE
#define __STATIC_INLINE static inline
#endif

#ifndef __STATIC_FORCEINLINE
#if defined(__GNUC__) || defined(__clang__)
#define __STATIC_FORCEINLINE static inline __attribute__((always_inline))
#else
#define __STATIC_FORCEINLINE static inline
#endif
#endif

#ifndef __STATIC
#define __STATIC static
#endif

#ifndef __ALIGNED
#if defined(__GNUC__) || defined(__clang__)
#define __ALIGNED(x) __attribute__((aligned(x)))
#else
#define __ALIGNED(x)
#endif
#endif

#ifndef __PACKED
#if defined(__GNUC__) || defined(__clang__)
#define __PACKED __attribute__((packed))
#else
#define __PACKED
#endif
#endif

#ifndef __PACKED_STRUCT
#if defined(__GNUC__) || defined(__clang__)
#define __PACKED_STRUCT struct __attribute__((packed))
#else
#define __PACKED_STRUCT struct
#endif
#endif

#ifndef __WEAK
#if defined(__GNUC__) || defined(__clang__)
#define __WEAK __attribute__((weak))
#else
#define __WEAK
#endif
#endif

#ifndef __ASM
#define __ASM __asm__
#endif

#ifndef __CLZ
__STATIC_FORCEINLINE uint8_t __CLZ(uint32_t data)
{
    return data == 0U ? 32U : (uint8_t)__builtin_clz(data);
}
#endif

#ifndef __SSAT
__STATIC_FORCEINLINE int32_t __SSAT(int32_t val, uint32_t sat)
{
    if (sat == 0U) {
        return 0;
    }

    if (sat >= 32U) {
        return val;
    }

    const int32_t max = (int32_t)((1U << (sat - 1U)) - 1U);
    const int32_t min = -max - 1;

    if (val > max) {
        return max;
    }
    if (val < min) {
        return min;
    }
    return val;
}
#endif

#ifndef __USAT
__STATIC_FORCEINLINE uint32_t __USAT(int32_t val, uint32_t sat)
{
    if (sat >= 32U) {
        return val < 0 ? 0U : (uint32_t)val;
    }

    const uint32_t max = (1U << sat) - 1U;

    if (val < 0) {
        return 0U;
    }
    if ((uint32_t)val > max) {
        return max;
    }
    return (uint32_t)val;
}
#endif

#ifndef __ROR
__STATIC_FORCEINLINE uint32_t __ROR(uint32_t op1, uint32_t op2)
{
    op2 &= 31U;
    return op2 == 0U ? op1 : ((op1 >> op2) | (op1 << (32U - op2)));
}
#endif

#endif
