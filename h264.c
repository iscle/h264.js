#include "H264SwDecApi.h"
#include <stdint.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

// Decoder state
static H264SwDecInst decInst = NULL;
static H264SwDecInput decInput;
static H264SwDecOutput decOutput;
static H264SwDecPicture decPicture;
static H264SwDecInfo decInfo;

static uint32_t picDecodeNumber;

// Picture callback pointer (JS registers this)
static void (*pictureCallback)(uint8_t *yuv, int width, int height) = NULL;

/*----------------------------- Initialization -----------------------------*/
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
int h264_init(int disableReordering) {
    if (decInst) return 0; // already initialized

    H264SwDecRet ret = H264SwDecInit(&decInst, disableReordering ? 1 : 0);
    if (ret != H264SWDEC_OK) return -1;

    picDecodeNumber = 0;
    return 0;
}

/*---------------------------- Set Picture Callback ------------------------*/
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void h264_set_callback(void (*cb)(uint8_t *yuv, int width, int height)) {
    pictureCallback = cb;
}

/*---------------------------- Decode H.264 Buffer ------------------------*/
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
int h264_decode(uint8_t *buffer, size_t length) {
    if (!decInst || !buffer || length == 0) return -1;

    decInput.pStream = buffer;
    decInput.dataLen = length;
    decInput.picId = picDecodeNumber;
    decInput.intraConcealmentMethod = 0; // gray concealment

    H264SwDecRet ret = H264SwDecDecode(decInst, &decInput, &decOutput);

    switch (ret) {
        // Headers ready
        case H264SWDEC_HDRS_RDY_BUFF_NOT_EMPTY: {
            H264SwDecGetInfo(decInst, &decInfo);  // query video info
            decInput.dataLen -= decOutput.pStrmCurrPos - decInput.pStream;
            decInput.pStream = decOutput.pStrmCurrPos;
            break;
        }

        // Picture(s) ready
        case H264SWDEC_PIC_RDY:
        case H264SWDEC_PIC_RDY_BUFF_NOT_EMPTY: {
            decInput.dataLen -= decOutput.pStrmCurrPos - decInput.pStream;
            decInput.pStream = decOutput.pStrmCurrPos;

            picDecodeNumber++; // increment after decoding

            while (H264SwDecNextPicture(decInst, &decPicture, 0) == H264SWDEC_PIC_RDY) {
                if (pictureCallback && decPicture.pOutputPicture) {
                    pictureCallback((uint8_t *) decPicture.pOutputPicture,
                                    (int) decInfo.picWidth,
                                    (int) decInfo.picHeight);
                }
            }
            break;
        }

        // Stream processed but no picture ready
        case H264SWDEC_STRM_PROCESSED:
        case H264SWDEC_STRM_ERR: {
            decInput.dataLen = 0;  // feed more data
            break;
        }

        default:
            // optional: handle unexpected values or ignore
            break;
    }

    return ret;
}

/*----------------------------- Release Decoder ---------------------------*/
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void h264_release(void) {
    if (decInst) {
        H264SwDecRelease(decInst);
        decInst = NULL;
    }
}
