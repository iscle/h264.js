#include "H264SwDecApi.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <emscripten/emscripten.h>

// Decoder state
static H264SwDecInst decInst = NULL;
static H264SwDecInput decInput;
static H264SwDecOutput decOutput;
static H264SwDecPicture decPicture;
static H264SwDecInfo decInfo;

// Picture callback pointer (JS registers this)
static void (*pictureCallback)(uint8_t *yuv, int width, int height) = NULL;

// Stream buffer for accumulating incomplete NAL units
static uint8_t *streamBuffer = NULL;
static size_t streamBufferSize = 0;
static size_t streamBufferCapacity = 0;

#define INITIAL_BUFFER_CAPACITY (512 * 1024)  // 512KB initial capacity

/*----------------------------- Initialization -----------------------------*/
EMSCRIPTEN_KEEPALIVE
int h264_init(int noOutputReordering) {
    if (decInst) return 0; // already initialized

    H264SwDecRet ret = H264SwDecInit(&decInst, (u32) noOutputReordering);
    if (ret != H264SWDEC_OK) return -1;

    // Initialize stream buffer
    streamBufferCapacity = INITIAL_BUFFER_CAPACITY;
    streamBuffer = malloc(streamBufferCapacity);
    if (!streamBuffer) {
        H264SwDecRelease(decInst);
        decInst = NULL;
        return -1;
    }
    streamBufferSize = 0;

    return 0;
}

/*---------------------------- Set Picture Callback ------------------------*/
EMSCRIPTEN_KEEPALIVE
void h264_set_callback(void (*cb)(uint8_t *yuv, int width, int height)) {
    pictureCallback = cb;
}

/*---------------------------- Decode H.264 Buffer ------------------------*/
EMSCRIPTEN_KEEPALIVE
int h264_decode(uint8_t *buffer, size_t length) {
    if (!decInst || !buffer || length == 0) return -1;

    // Ensure we have enough capacity in the buffer
    if (streamBufferSize + length > streamBufferCapacity) {
        size_t newCapacity = streamBufferCapacity;
        while (newCapacity < streamBufferSize + length) {
            newCapacity *= 2;
        }
        uint8_t *newBuffer = realloc(streamBuffer, newCapacity);
        if (!newBuffer) {
            return -1; // Out of memory
        }
        streamBuffer = newBuffer;
        streamBufferCapacity = newCapacity;
    }

    // Append new data to the buffer
    memcpy(streamBuffer + streamBufferSize, buffer, length);
    streamBufferSize += length;
    
    while (streamBufferSize > 0) {
        decInput.pStream = streamBuffer;
        decInput.dataLen = streamBufferSize;
        decInput.intraConcealmentMethod = 0; // gray concealment

        H264SwDecRet ret = H264SwDecDecode(decInst, &decInput, &decOutput);

        // Calculate how many bytes were consumed
        size_t bytesConsumed = 0;
        if (decOutput.pStrmCurrPos && decOutput.pStrmCurrPos >= streamBuffer) {
            bytesConsumed = decOutput.pStrmCurrPos - streamBuffer;
        }

        switch (ret) {
            // Headers ready
            case H264SWDEC_HDRS_RDY_BUFF_NOT_EMPTY: {
                H264SwDecGetInfo(decInst, &decInfo);  // query video info

                // Remove consumed bytes from buffer
                if (bytesConsumed > 0 && bytesConsumed <= streamBufferSize) {
                    memmove(streamBuffer, streamBuffer + bytesConsumed, 
                            streamBufferSize - bytesConsumed);
                    streamBufferSize -= bytesConsumed;
                }
                // Continue processing remaining data
                continue;
            }

            // Picture(s) ready
            case H264SWDEC_PIC_RDY:
            case H264SWDEC_PIC_RDY_BUFF_NOT_EMPTY: {
                // Output all ready pictures
                while (H264SwDecNextPicture(decInst, &decPicture, 0) == H264SWDEC_PIC_RDY) {
                    if (pictureCallback && decPicture.pOutputPicture) {
                        pictureCallback((uint8_t *) decPicture.pOutputPicture,
                                        (int) decInfo.picWidth,
                                        (int) decInfo.picHeight);
                    }
                }

                // Remove consumed bytes from buffer
                if (bytesConsumed > 0 && bytesConsumed <= streamBufferSize) {
                    memmove(streamBuffer, streamBuffer + bytesConsumed, 
                            streamBufferSize - bytesConsumed);
                    streamBufferSize -= bytesConsumed;
                }
                
                // Continue processing if buffer not empty
                if (ret == H264SWDEC_PIC_RDY_BUFF_NOT_EMPTY) {
                    continue;
                }
                return ret;
            }

            // Stream processed but no picture ready
            case H264SWDEC_STRM_PROCESSED: {
                // All current data processed, need more data
                // Remove consumed bytes from buffer
                if (bytesConsumed > 0 && bytesConsumed <= streamBufferSize) {
                    memmove(streamBuffer, streamBuffer + bytesConsumed, 
                            streamBufferSize - bytesConsumed);
                    streamBufferSize -= bytesConsumed;
                }
                return ret;
            }

            // Stream error
            case H264SWDEC_STRM_ERR: {
                // Try to recover by removing consumed bytes
                if (bytesConsumed > 0 && bytesConsumed <= streamBufferSize) {
                    memmove(streamBuffer, streamBuffer + bytesConsumed, 
                            streamBufferSize - bytesConsumed);
                    streamBufferSize -= bytesConsumed;
                } else if (streamBufferSize > 0) {
                    // Skip one byte and try again (error recovery)
                    memmove(streamBuffer, streamBuffer + 1, streamBufferSize - 1);
                    streamBufferSize -= 1;
                }
                
                // If no more data, return the error
                if (streamBufferSize == 0) {
                    return ret;
                }
                // Otherwise try to continue
                continue;
            }

            default:
                // For any other return value, remove consumed bytes and exit
                if (bytesConsumed > 0 && bytesConsumed <= streamBufferSize) {
                    memmove(streamBuffer, streamBuffer + bytesConsumed, 
                            streamBufferSize - bytesConsumed);
                    streamBufferSize -= bytesConsumed;
                }
                return ret;
        }
    }

    return 0;
}

/*----------------------------- Reset Stream Buffer ------------------------*/
EMSCRIPTEN_KEEPALIVE
void h264_reset_buffer(void) {
    streamBufferSize = 0;
}

/*----------------------------- Release Decoder ---------------------------*/
EMSCRIPTEN_KEEPALIVE
void h264_release(void) {
    if (decInst) {
        H264SwDecRelease(decInst);
        decInst = NULL;
    }
    
    // Free stream buffer
    if (streamBuffer) {
        free(streamBuffer);
        streamBuffer = NULL;
    }
    streamBufferSize = 0;
    streamBufferCapacity = 0;
}
