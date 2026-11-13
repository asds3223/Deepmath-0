# Rapfi(250615) 소스 + oracle.cpp 를 한 실행 파일로 빌드한다.
# 사용: cmake -S oracle -B build/oracle -DCMAKE_BUILD_TYPE=Release -DORACLE_ISA=AVX512VNNI && cmake --build build/oracle -j
cmake_minimum_required(VERSION 3.16)
project(rapfi_oracle CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(RAPFI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/rapfi/Rapfi" CACHE PATH "Rapfi source dir")
set(ORACLE_ISA "AVX512VNNI" CACHE STRING "AVX2 | AVX512VNNI")

set(RAPFI_SOURCES
    command/argutils.cpp command/benchmark.cpp command/command.cpp command/gomocup.cpp
    core/hash.cpp core/iohelper.cpp core/utils.cpp core/platform.cpp core/version.cpp
    database/dbclient.cpp database/dbutils.cpp database/dbtypes.cpp database/yxdbstorage.cpp
    eval/eval.cpp eval/evaluator.cpp eval/mix9svqnnue.cpp eval/mix10nnue.cpp
    game/board.cpp game/movegen.cpp game/pattern.cpp
    search/hashtable.cpp search/movepick.cpp search/opening.cpp search/searchcommon.cpp
    search/searchoutput.cpp search/searchthread.cpp search/timecontrol.cpp
    search/ab/history.cpp search/ab/search.cpp search/mcts/node.cpp search/mcts/search.cpp
    config.cpp internalConfig.cpp)
list(TRANSFORM RAPFI_SOURCES PREPEND "${RAPFI_DIR}/")

add_subdirectory(${RAPFI_DIR}/external/cpptoml ext/cpptoml)
add_subdirectory(${RAPFI_DIR}/external/cxxopts ext/cxxopts)
add_subdirectory(${RAPFI_DIR}/external/lz4 ext/lz4)
add_subdirectory(${RAPFI_DIR}/external/simde ext/simde)

add_executable(oracle oracle.cpp ${RAPFI_SOURCES})
target_include_directories(oracle PRIVATE ${RAPFI_DIR})
target_link_libraries(oracle PRIVATE cpptoml cxxopts lz4 simde pthread)
target_compile_definitions(oracle PRIVATE MULTI_THREADING USE_SSE USE_AVX2)
target_compile_options(oracle PRIVATE -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -mavx2 -mfma
    -Wno-unused-variable -Wno-unused-function -Wno-missing-braces -Wno-parentheses -Wno-sign-compare)
if(ORACLE_ISA STREQUAL "AVX512VNNI")
    target_compile_definitions(oracle PRIVATE USE_AVX512 USE_VNNI USE_BMI2)
    target_compile_options(oracle PRIVATE -mavx512f -mavx512dq -mavx512bw -mavx512vnni -mavx512vl -mbmi2)
endif()
