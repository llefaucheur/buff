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

#endif
