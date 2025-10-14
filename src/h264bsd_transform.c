/*
 * Copyright (C) 2009 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*------------------------------------------------------------------------------

    Table of contents

     1. Include headers
     2. External compiler flags
     3. Module defines
     4. Local function prototypes
     5. Functions
          h264bsdProcessBlock
          h264bsdProcessLumaDc
          h264bsdProcessChromaDc

------------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------
    1. Include headers
------------------------------------------------------------------------------*/

#include "basetype.h"
#include "h264bsd_transform.h"
#include "h264bsd_util.h"
#include <emmintrin.h>

/*------------------------------------------------------------------------------
    2. External compiler flags
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
    3. Module defines
------------------------------------------------------------------------------*/

/* Switch off the following Lint messages for this file:
 * Info 701: Shift left of signed quantity (int)
 * Info 702: Shift right of signed quantity (int)
 */
/*lint -e701 -e702 */

/* LevelScale function */
const i32 levelScale[6][3] = {
    {10,13,16}, {11,14,18}, {13,16,20}, {14,18,23}, {16,20,25}, {18,23,29}};

/* qp % 6 as a function of qp */
const u8 qpMod6[52] = {0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,
    0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3};

/* qp / 6 as a function of qp */
const u8 qpDiv6[52] = {0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,
    4,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7,8,8,8,8};

/*------------------------------------------------------------------------------
    4. Local function prototypes
------------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------

    Function: h264bsdProcessBlock

        Functional description:
            Function performs inverse zig-zag scan, inverse scaling and
            inverse transform for a luma or a chroma residual block

        Inputs:
            data            pointer to data to be processed
            qp              quantization parameter
            skip            skip processing of data[0], set to non-zero value
                            if dc coeff hanled separately
            coeffMap        16 lsb's indicate which coeffs are non-zero,
                            bit 0 (lsb) for coeff 0, bit 1 for coeff 1 etc.

        Outputs:
            data            processed data

        Returns:
            HANTRO_OK       success
            HANTRO_NOK      processed data not in valid range [-512, 511]

------------------------------------------------------------------------------*/
static inline __m128i h264bsd_htransform(__m128i row)
{
    __m128i a = _mm_shuffle_epi32(row, _MM_SHUFFLE(0,0,0,0)); // x0
    __m128i b = _mm_shuffle_epi32(row, _MM_SHUFFLE(1,1,1,1)); // x1
    __m128i c = _mm_shuffle_epi32(row, _MM_SHUFFLE(2,2,2,2)); // x2
    __m128i d = _mm_shuffle_epi32(row, _MM_SHUFFLE(3,3,3,3)); // x3

    __m128i t0 = _mm_add_epi32(a, c);                          // x0 + x2
    __m128i t1 = _mm_sub_epi32(a, c);                          // x0 - x2
    __m128i t2 = _mm_sub_epi32(_mm_srai_epi32(b, 1), d);       // (x1>>1) - x3
    __m128i t3 = _mm_add_epi32(b, _mm_srai_epi32(d, 1));       // x1 + (x3>>1)

    __m128i y0 = _mm_add_epi32(t0, t3);
    __m128i y1 = _mm_add_epi32(t1, t2);
    __m128i y2 = _mm_sub_epi32(t1, t2);
    __m128i y3 = _mm_sub_epi32(t0, t3);

    // Pack [y0[0], y1[0], y2[0], y3[0]] into one vector
    __m128i out = _mm_unpacklo_epi32(y0, y1);  // [y0, y1, ?, ?]
    __m128i tmp = _mm_unpacklo_epi32(y2, y3);  // [y2, y3, ?, ?]
    out = _mm_unpacklo_epi64(out, tmp);        // [y0, y1, y2, y3]
    return out;
}

u32 h264bsdProcessBlock(i32 *data, u32 qp, u32 skip, u32 coeffMap)
{

    u32 qpDiv = qpDiv6[qp];
    i32 scale0 = levelScale[qpMod6[qp]][0] << qpDiv;
    i32 scale1 = levelScale[qpMod6[qp]][1] << qpDiv;
    i32 scale2 = levelScale[qpMod6[qp]][2] << qpDiv;

    /* Inverse quantization for DC */
    if (!skip)
        data[0] = data[0] * scale0;

    /* Early exit: only DC or DC + positions 1,5,6 */
    if ((coeffMap & 0xFF9C) == 0)
    {
        if ((coeffMap & 0x62) == 0)
        {
            i32 tmp = (data[0] + 32) >> 6;
            if ((u32)(tmp + 512) > 1023)
                return HANTRO_NOK;

            __m128i dc = _mm_set1_epi32(tmp);
            _mm_storeu_si128((__m128i*)(data + 0), dc);
            _mm_storeu_si128((__m128i*)(data + 4), dc);
            _mm_storeu_si128((__m128i*)(data + 8), dc);
            _mm_storeu_si128((__m128i*)(data + 12), dc);
            return HANTRO_OK;
        }
        else
        {
            // Same as scalar — no benefit to vectorize 4 elements
            data[1] = data[1] * scale1;
            data[2] = data[5] * scale0;
            data[3] = data[6] * scale1;

            i32 t0 = data[0] + data[2];
            i32 t1 = data[0] - data[2];
            i32 t2 = (data[1] >> 1) - data[3];
            i32 t3 = data[1] + (data[3] >> 1);

            data[0] = (t0 + t3 + 32) >> 6;
            data[1] = (t1 + t2 + 32) >> 6;
            data[2] = (t1 - t2 + 32) >> 6;
            data[3] = (t0 - t3 + 32) >> 6;

            // Broadcast to columns
            data[4] = data[8] = data[12] = data[0];
            data[5] = data[9] = data[13] = data[1];
            data[6] = data[10] = data[14] = data[2];
            data[7] = data[11] = data[15] = data[3];

            if (((u32)(data[0] + 512) > 1023) ||
                ((u32)(data[1] + 512) > 1023) ||
                ((u32)(data[2] + 512) > 1023) ||
                ((u32)(data[3] + 512) > 1023))
                return HANTRO_NOK;

            return HANTRO_OK;
        }
    }

    /* === General case: full 4x4 block === */

    // Step 1: Zig-zag reordering + scaling into temp buffer
    i32 temp[16];
    temp[0]  = data[0];               // DC already scaled if !skip
    temp[1]  = data[1]  * scale1;
    temp[2]  = data[5]  * scale0;
    temp[3]  = data[6]  * scale1;
    temp[4]  = data[2]  * scale1;
    temp[5]  = data[4]  * scale2;
    temp[6]  = data[7]  * scale1;
    temp[7]  = data[12] * scale2;
    temp[8]  = data[3]  * scale0;
    temp[9]  = data[8]  * scale1;
    temp[10] = data[11] * scale0;
    temp[11] = data[13] * scale1;
    temp[12] = data[9]  * scale1;
    temp[13] = data[10] * scale2;
    temp[14] = data[14] * scale1;
    temp[15] = data[15] * scale2;

    // Step 2: Horizontal transform on all 4 rows (in-place on temp)
    __m128i r0 = _mm_loadu_si128((__m128i*)(temp + 0));
    __m128i r1 = _mm_loadu_si128((__m128i*)(temp + 4));
    __m128i r2 = _mm_loadu_si128((__m128i*)(temp + 8));
    __m128i r3 = _mm_loadu_si128((__m128i*)(temp + 12));

    r0 = h264bsd_htransform(r0);
    r1 = h264bsd_htransform(r1);
    r2 = h264bsd_htransform(r2);
    r3 = h264bsd_htransform(r3);

    _mm_storeu_si128((__m128i*)(temp + 0), r0);
    _mm_storeu_si128((__m128i*)(temp + 4), r1);
    _mm_storeu_si128((__m128i*)(temp + 8), r2);
    _mm_storeu_si128((__m128i*)(temp + 12), r3);

    // Step 3: Vertical transform (on columns) — scalar is fine and clear
    for (int col = 0; col < 4; col++)
    {
        i32 x0 = temp[0 + col];
        i32 x1 = temp[4 + col];
        i32 x2 = temp[8 + col];
        i32 x3 = temp[12 + col];

        i32 t0 = x0 + x2;
        i32 t1 = x0 - x2;
        i32 t2 = (x1 >> 1) - x3;
        i32 t3 = x1 + (x3 >> 1);

        i32 y0 = (t0 + t3 + 32) >> 6;
        i32 y1 = (t1 + t2 + 32) >> 6;
        i32 y2 = (t1 - t2 + 32) >> 6;
        i32 y3 = (t0 - t3 + 32) >> 6;

        data[0  + col] = y0;
        data[4  + col] = y1;
        data[8  + col] = y2;
        data[12 + col] = y3;

        // Range check
        if (((u32)(y0 + 512) > 1023) ||
            ((u32)(y1 + 512) > 1023) ||
            ((u32)(y2 + 512) > 1023) ||
            ((u32)(y3 + 512) > 1023))
            return HANTRO_NOK;
    }

    return HANTRO_OK;

}

/*------------------------------------------------------------------------------

    Function: h264bsdProcessLumaDc

        Functional description:
            Function performs inverse zig-zag scan, inverse transform and
            inverse scaling for a luma DC coefficients block

        Inputs:
            data            pointer to data to be processed
            qp              quantization parameter

        Outputs:
            data            processed data

        Returns:
            none

------------------------------------------------------------------------------*/
void h264bsdProcessLumaDc(i32 *data, u32 qp)
{

/* Variables */

    i32 tmp0, tmp1, tmp2, tmp3;
    u32 row,col;
    u32 qpMod, qpDiv;
    i32 levScale;
    i32 *ptr;

/* Code */

    qpMod = qpMod6[qp];
    qpDiv = qpDiv6[qp];

    /* zig-zag scan */
    tmp0 = data[2];
    data[2]  = data[5];
    data[5] = data[4];
    data[4] = tmp0;

    tmp0 = data[8];
    data[8] = data[3];
    data[3]  = data[6];
    data[6]  = data[7];
    data[7]  = data[12];
    data[12] = data[9];
    data[9]  = tmp0;

    tmp0 = data[10];
    data[10] = data[11];
    data[11] = data[13];
    data[13] = tmp0;

    /* horizontal transform */
    for (row = 4, ptr = data; row--; ptr += 4)
    {
        tmp0 = ptr[0] + ptr[2];
        tmp1 = ptr[0] - ptr[2];
        tmp2 = ptr[1] - ptr[3];
        tmp3 = ptr[1] + ptr[3];
        ptr[0] = tmp0 + tmp3;
        ptr[1] = tmp1 + tmp2;
        ptr[2] = tmp1 - tmp2;
        ptr[3] = tmp0 - tmp3;
    }

    /*lint +e661 +e662*/
    /* then vertical transform and inverse scaling */
    levScale = levelScale[ qpMod ][0];
    if (qp >= 12)
    {
        levScale <<= (qpDiv-2);
        for (col = 4; col--; data++)
        {
            tmp0 = data[0] + data[8 ];
            tmp1 = data[0] - data[8 ];
            tmp2 = data[4] - data[12];
            tmp3 = data[4] + data[12];
            data[0 ] = ((tmp0 + tmp3)*levScale);
            data[4 ] = ((tmp1 + tmp2)*levScale);
            data[8 ] = ((tmp1 - tmp2)*levScale);
            data[12] = ((tmp0 - tmp3)*levScale);
        }
    }
    else
    {
        i32 tmp;
        tmp = ((1 - qpDiv) == 0) ? 1 : 2;
        for (col = 4; col--; data++)
        {
            tmp0 = data[0] + data[8 ];
            tmp1 = data[0] - data[8 ];
            tmp2 = data[4] - data[12];
            tmp3 = data[4] + data[12];
            data[0 ] = ((tmp0 + tmp3)*levScale+tmp) >> (2-qpDiv);
            data[4 ] = ((tmp1 + tmp2)*levScale+tmp) >> (2-qpDiv);
            data[8 ] = ((tmp1 - tmp2)*levScale+tmp) >> (2-qpDiv);
            data[12] = ((tmp0 - tmp3)*levScale+tmp) >> (2-qpDiv);
        }
    }

}

/*------------------------------------------------------------------------------

    Function: h264bsdProcessChromaDc

        Functional description:
            Function performs inverse transform and inverse scaling for a
            chroma DC coefficients block

        Inputs:
            data            pointer to data to be processed
            qp              quantization parameter

        Outputs:
            data            processed data

        Returns:
            none

------------------------------------------------------------------------------*/
void h264bsdProcessChromaDc(i32 *data, u32 qp)
{

/* Variables */

    i32 tmp0, tmp1, tmp2, tmp3;
    u32 qpDiv;
    i32 levScale;
    u32 levShift;

/* Code */

    qpDiv = qpDiv6[qp];
    levScale = levelScale[ qpMod6[qp] ][0];

    if (qp >= 6)
    {
        levScale <<= (qpDiv-1);
        levShift = 0;
    }
    else
    {
        levShift = 1;
    }

    tmp0 = data[0] + data[2];
    tmp1 = data[0] - data[2];
    tmp2 = data[1] - data[3];
    tmp3 = data[1] + data[3];
    data[0] = ((tmp0 + tmp3) * levScale) >> levShift;
    data[1] = ((tmp0 - tmp3) * levScale) >> levShift;
    data[2] = ((tmp1 + tmp2) * levScale) >> levShift;
    data[3] = ((tmp1 - tmp2) * levScale) >> levShift;
    tmp0 = data[4] + data[6];
    tmp1 = data[4] - data[6];
    tmp2 = data[5] - data[7];
    tmp3 = data[5] + data[7];
    data[4] = ((tmp0 + tmp3) * levScale) >> levShift;
    data[5] = ((tmp0 - tmp3) * levScale) >> levShift;
    data[6] = ((tmp1 + tmp2) * levScale) >> levShift;
    data[7] = ((tmp1 - tmp2) * levScale) >> levShift;

}

/*lint +e701 +e702 */


