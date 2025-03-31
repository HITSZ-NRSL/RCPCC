cmake_minimum_required(VERSION 3.20)
project(pcc_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")



# OpenCV
set(OpenCV_DIR "/home/zoratt/opencv-4.4.0/build")
find_package(OpenCV REQUIRED)
#输出cv版本
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
#输出cv库的路径
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})

#FFTW
# 查找 FFTW 库
set(FFTW_INCLUDE_DIR /usr/local/include)
set(FFTW_LIBRARY libfftw3.so)
include_directories(${FFTW_INCLUDE_DIR})
#zstd库
set(ZSTD_LIBRARY libzstd.so)

# Include PCL headers
find_package(PCL 1.10 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(./utils)
include_directories(./modules)
include_directories(./proto)
# Source files
set(SOURCES
    utils/utils.cpp
    utils/encoder.cpp
    utils/pcc_module.cpp
    utils/decoder.cpp
    utils/io.cpp
    modules/decoder_module.cpp
    modules/encoder_module.cpp
    modules/dct.cpp
    modules/binary_compressor.cpp
    modules/serializer.cpp
)
# Boost
find_package(Boost REQUIRED COMPONENTS program_options system)
include_directories(${Boost_INCLUDE_DIRS})

#gRPC
#添加库路径




# Compile source files
add_library(pcc_lib ${SOURCES} )
target_link_libraries(
    pcc_lib
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
    stdc++fs
    ${FFTW_LIBRARY}
    ${ZSTD_LIBRARY}
)
add_executable(example example.cpp)
target_link_libraries(example pthread pcc_lib) 
