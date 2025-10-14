/*
 * H.264 Decoder SIMD Optimizations
 * Optimized for WebAssembly SIMD using SSE2 intrinsics
 * Emscripten translates SSE2 intrinsics to WebAssembly SIMD instructions
 */

#ifndef H264SWDEC_SIMD_H
#define H264SWDEC_SIMD_H

#include "basetype.h"

/* Check if SIMD is available */
#if defined(__EMSCRIPTEN__) && defined(__SSE2__)
#define H264_SIMD_ENABLED 1
#include <emmintrin.h>  /* SSE2 intrinsics */
#include <string.h>     /* memcpy, memset */
#else
#define H264_SIMD_ENABLED 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* SIMD-optimized transform functions */
#if H264_SIMD_ENABLED

/* Fast 4x4 inverse transform (SIMD optimized) */
u32 h264bsdProcessBlock_SIMD(i32 *data, u32 qp, u32 skip, u32 coeffMap);

/* Fast DC transforms (SIMD optimized) */
void h264bsdProcessLumaDc_SIMD(i32 *data, u32 qp);
void h264bsdProcessChromaDc_SIMD(i32 *data, u32 qp);

/* Fast memory operations */
void h264bsdFillRow_SIMD(u8 *dst, u8 val, u32 count);
void h264bsdCopyBlock_SIMD(u8 *dst, const u8 *src, u32 width, u32 height, 
                           u32 dstStride, u32 srcStride);

/* Fast vertical prediction (intra) */
void h264bsdIntra16x16Vertical_SIMD(u8 *data, const u8 *above);
void h264bsdIntra16x16Horizontal_SIMD(u8 *data, const u8 *left);
void h264bsdIntra16x16DC_SIMD(u8 *data, const u8 *above, const u8 *left,
                               u32 availA, u32 availB);

/* Fast chroma prediction */
void h264bsdIntraChromaVertical_SIMD(u8 *data, const u8 *above);
void h264bsdIntraChromaHorizontal_SIMD(u8 *data, const u8 *left);

/* Add residual to prediction (SIMD optimized) */
void h264bsdAddResidual_SIMD(u8 *data, const i32 *residual, u32 blockNum);

#endif /* H264_SIMD_ENABLED */

#ifdef __cplusplus
}
#endif

#endif /* H264SWDEC_SIMD_H */

