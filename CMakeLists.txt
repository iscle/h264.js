cmake_minimum_required(VERSION 3.28)
project(h264 C)

# Use C11
set(CMAKE_C_STANDARD 11)

# Add all source files
set(H264_SOURCES
        h264.c
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

# Include directories
include_directories(inc)

# Add executable target
add_executable(h264 ${H264_SOURCES})

# Compiler flags for warnings and optimization
target_compile_options(h264 PRIVATE -Wall -Wextra -Wpedantic)

# Emscripten-specific settings
if(EMSCRIPTEN)
    message(STATUS "Building with Emscripten")

    # Target link options
    target_link_options(h264 PRIVATE
            "-sSTRICT"
            "-sALLOW_TABLE_GROWTH=1"
            "-sALLOW_MEMORY_GROWTH=1"
            "-sINITIAL_HEAP=33554432" # 32MB (default is 16MB)
            "-sEXPORTED_RUNTIME_METHODS=HEAPU8,addFunction"
            "-sMODULARIZE=1"
            "-sEXPORT_NAME=createH264"
            "-sEXPORTED_FUNCTIONS=_h264_init,_h264_set_callback,_h264_decode,_h264_release"
    )

    # Apply optimization only for non Debug builds
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "Applying optimizations for non-Debug build")

        target_link_options(h264 PRIVATE
                "-O3"
                "-flto"
        )
    endif()
endif()
