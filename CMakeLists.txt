cmake_minimum_required(VERSION 3.28)
project(h264_js C)

set(CMAKE_C_STANDARD 11)

add_library(h264_js STATIC library.c
        src/h264bsd_byte_stream.c
        src/h264bsd_cavlc.c
        src/h264bsd_conceal.c
        src/h264bsd_deblocking.c
        src/h264bsd_decoder.c
        src/h264bsd_dpb.c
        src/h264bsd_image.c
        src/h264bsd_inter_prediction.c
        src/h264bsd_intra_prediction.c
        src/h264bsd_macroblock_layer.c
        src/h264bsd_nal_unit.c
        src/h264bsd_neighbour.c
        src/h264bsd_pic_order_cnt.c
        src/h264bsd_pic_param_set.c
        src/h264bsd_reconstruct.c
        src/h264bsd_seq_param_set.c
        src/h264bsd_slice_data.c
        src/h264bsd_slice_group_map.c
        src/h264bsd_slice_header.c
        src/h264bsd_storage.c
        src/h264bsd_stream.c
        src/h264bsd_transform.c
        src/h264bsd_util.c
        src/h264bsd_vlc.c
        src/h264bsd_vui.c
        src/H264SwDecApi.c
)

target_include_directories(h264_js PUBLIC inc)
