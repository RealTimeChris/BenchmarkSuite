cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CUTLASS_RT_TMNVCC_ARCHS "120" CACHE STRING "CUDA architectures")
set(CUTLASS_RT_TMNVCC_ARCHS "120" CACHE STRING "CUDA architectures")
# Or if you want multiple: set(CUTLASS_RT_TMNVCC_ARCHS "89;120" CACHE STRING "CUDA architectures")

# Disable the library build to avoid this mess
set(CUTLASS_RT_TMENABLE_LIBRARY OFF CACHE BOOL "Disable CUTLASS library")
set(CUTLASS_RT_TMENABLE_PROFILER OFF CACHE BOOL "Disable CUTLASS profiler") 
set(CUTLASS_RT_TMENABLE_EXAMPLES OFF CACHE BOOL "Disable CUTLASS examples")
set(CUTLASS_RT_TMENABLE_TESTS OFF CACHE BOOL "Disable CUTLASS tests")
set(PROJECT_NAME "BenchmarkSuite")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project("${PROJECT_NAME}" LANGUAGES CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/CMake;")

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/Include/benchmarksuite/*.hpp")

#include(FetchContent)

#FetchContent_Declare(
  #cutlass
  #GIT_REPOSITORY https://github.com/notanothergaydude/cutlass.git
  #GIT_TAG main
  #GIT_SHALLOW TRUE
#)
#FetchContent_MakeAvailable(cutlass)

add_library("${PROJECT_NAME}" INTERFACE "${HEADERS}")

add_library("BenchmarkSuite::BenchmarkSuite" ALIAS "${PROJECT_NAME}")

set(CMAKE_VERBOSE_MAKEFILE ON)

set(BNCH_SWT_COMPILE_DEFINITIONS
    "BNCH_SWT_ARCH_X64=$<IF:$<OR:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},AMD64>>,1,0>"
    "BNCH_SWT_ARCH_ARM64=$<IF:$<OR:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},aarch64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},ARM64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},arm64>>,1,0>"
    "BNCH_SWT_PLATFORM_WINDOWS=$<IF:$<PLATFORM_ID:Windows>,1,0>"
    "BNCH_SWT_PLATFORM_LINUX=$<IF:$<PLATFORM_ID:Linux>,1,0>"
    "BNCH_SWT_PLATFORM_MAC=$<IF:$<PLATFORM_ID:Darwin>,1,0>"
    "BNCH_SWT_COMPILER_CLANG=$<IF:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,1,0>>"
    "BNCH_SWT_COMPILER_MSVC=$<IF:$<CXX_COMPILER_ID:MSVC>,1,0>"
    "BNCH_SWT_COMPILER_GNUCXX=$<IF:$<CXX_COMPILER_ID:GNU>,1,0>"
    "$<$<CXX_COMPILER_ID:MSVC>:NOMINMAX;WIN32_LEAN_AND_MEAN>"
    "BNCH_SWT_INLINE=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,__forceinline__,__noinline__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] inline,inline __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>>"
)

if(MSVC OR NVCC)
    string(REGEX REPLACE "/Ob[0-2]" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REGEX REPLACE "/O[0-2]" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

set(BNCH_SWT_COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:Clang>:-O3;-funroll-loops;-fvectorize;-fslp-vectorize;-finline-functions;-fomit-frame-pointer;-fmerge-all-constants;-ftemplate-depth=2048;-fconstexpr-depth=2048;-fconstexpr-steps=50000000;-ftemplate-backtrace-limit=0;-ffunction-sections;-fdata-sections;-falign-functions=32;-fno-math-errno;-ffp-contract=on;-fvisibility=hidden;-fvisibility-inlines-hidden;-fno-rtti;-fno-asynchronous-unwind-tables;-fno-unwind-tables;-fno-stack-protector;-fno-ident;-pipe;-fno-common;-fwrapv;-D_FORTIFY_SOURCE=0;-Weverything;-Wnon-virtual-dtor;-Wno-c++98-compat;-Wno-c++98-compat-pedantic;-Wno-unsafe-buffer-usage;-Wno-padded;-Wno-c++20-compat;-Wno-exit-time-destructors>"#;-Werror>"
    "$<$<CXX_COMPILER_ID:AppleClang>:-O3;-funroll-loops;-fvectorize;-fslp-vectorize;-finline-functions;-fomit-frame-pointer;-fmerge-all-constants;-ftemplate-depth=2048;-fconstexpr-depth=2048;-fconstexpr-steps=50000000;-ftemplate-backtrace-limit=0;-ffunction-sections;-fdata-sections;-falign-functions=32;-fno-math-errno;-ffp-contract=on;-fvisibility=hidden;-fvisibility-inlines-hidden;-fno-rtti;-fno-asynchronous-unwind-tables;-fno-unwind-tables;-fno-stack-protector;-fno-ident;-pipe;-fno-common;-fwrapv;-D_FORTIFY_SOURCE=0;-Weverything;-Wnon-virtual-dtor;-Wno-c++98-compat;-Wno-c++98-compat-pedantic;-Wno-unsafe-buffer-usage;-Wno-padded;-Wno-c++20-compat;-Wno-exit-time-destructors;-Wno-poison-system-directories>"#;-Werror>"
    "$<$<CXX_COMPILER_ID:GNU>:-O3;;-funroll-loops;-finline-functions;-fomit-frame-pointer;-fno-math-errno;-ftemplate-depth=2000;-fconstexpr-depth=2000;-fconstexpr-ops-limit=100000000;-fconstexpr-loop-limit=1000000;-falign-functions=32;-falign-loops=32;-fprefetch-loop-arrays;-ftree-vectorize;-fstrict-aliasing;-ffunction-sections;-fdata-sections;-fvisibility=hidden;-fvisibility-inlines-hidden;-fno-keep-inline-functions;-fno-ident;-fmerge-all-constants;-fno-stack-protector;-fno-rtti;-fgcse-after-reload;-ftree-loop-distribute-patterns;-fpredictive-commoning;-funswitch-loops;-ftree-loop-vectorize;-ftree-slp-vectorize;-Wall;-Wextra;-Wpedantic;-Wnon-virtual-dtor;-Wlogical-op;-Wduplicated-cond;-Wduplicated-branches;-Wnull-dereference;-Wdouble-promotion>"#;-Werror>"
    "$<$<CXX_COMPILER_ID:MSVC>:/Ob3;/Ot;/Oy;/GT;/GL;/fp:precise;/Qpar;/constexpr:depth2048;/constexpr:backtrace0;/constexpr:steps2000000;/GS-;/Gy;/Gw;/Zc:inline;/Zc:throwingNew;/W4;/permissive-;/Zc:__cplusplus;/wd4820;/wd4324;/wd5002;/Zc:alignedNew;/Zc:auto;/Zc:forScope;/Zc:implicitNoexcept;/Zc:noexceptTypes;/Zc:referenceBinding;/Zc:rvalueCast;/Zc:sizedDealloc;/Zc:strictStrings;/Zc:ternary;/Zc:wchar_t>"#;/WX>"\
    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Linux>>:-fno-plt;-fno-semantic-interposition>"
	"--expt-relaxed-constexpr"
    "${BNCH_SWT_SIMD_FLAGS}"
)

set(BNCH_SWT_LINK_OPTIONS
    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Darwin>>:-Wl,-dead_strip;-Wl,-x;-Wl,-S>"
    "$<$<AND:$<CXX_COMPILER_ID:AppleClang>,$<PLATFORM_ID:Darwin>>:-Wl,-dead_strip;-Wl,-x;-Wl,-S>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Darwin>>:-Wl,-dead_strip;-Wl,-x;-Wl,-S>"
    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Linux>>:-Wl,--gc-sections;-Wl,--strip-all;-Wl,--build-id=none;-Wl,--hash-style=gnu;-Wl,-z,now;-Wl,-z,relro>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Linux>>:-Wl,--gc-sections;-Wl,--strip-all;-Wl,--as-needed;-Wl,-O3>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<PLATFORM_ID:Windows>>:/DYNAMICBASE:NO;/OPT:REF;/OPT:ICF;/INCREMENTAL:NO;/MACHINE:X64>"
)

set_target_properties(
	"${PROJECT_NAME}" PROPERTIES
	OUTPUT_NAME "benchmarksuite"
)

target_include_directories(
	"${PROJECT_NAME}" INTERFACE
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include/>"
)

target_link_options(
	"${PROJECT_NAME}" INTERFACE
	"${BNCH_SWT_LINK_OPTIONS}"
)

target_compile_options(
	"${PROJECT_NAME}" INTERFACE
	"${BNCH_SWT_COMPILE_OPTIONS}"
)

set(CONFIG_FILE_NAME "BenchmarkSuiteConfig.cmake")
set(EXPORTED_TARGETS_NAME "BenchmarkSuiteTargets")
set(EXPORTED_TARGETS_FILE_NAME "${EXPORTED_TARGETS_NAME}.cmake")
set(EXPORTED_TARGETS_FILE_PATH "share/benchmarksuite/${EXPORTED_TARGETS_FILE_NAME}")
set(VERSION_FILE_NAME "BenchmarkSuiteConfigVersion.cmake")

include(CMakePackageConfigHelpers)
configure_package_config_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/CMake/${CONFIG_FILE_NAME}.in"
	"${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
	INSTALL_DESTINATION "share/benchmarksuite"
	PATH_VARS
	EXPORTED_TARGETS_FILE_PATH
)

write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
	VERSION "1.0.0"
	COMPATIBILITY AnyNewerVersion
)

install(
	FILES
	"${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
	"${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
	DESTINATION "share/benchmarksuite"
)

if ("${BENCH_TYPE}" STREQUAL "BENCHMARK")
	add_subdirectory("Benchmark")
endif()
