cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 120)
project(
  "BenchmarkExe"
  VERSION "1.0.0"
  LANGUAGES CUDA
)

find_package(CUDAToolkit REQUIRED)

if(MSVC OR NVCC)
    string(REGEX REPLACE "/Ob[0-2]" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REGEX REPLACE "/O[0-2]" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

set(BNCH_SWT_COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:CLANG>:-O3;-funroll-loops;-fvectorize;-fslp-vectorize;-finline-functions;-fomit-frame-pointer;-fmerge-all-constants;-ftemplate-depth=2048;-fconstexpr-depth=2048;-fconstexpr-steps=50000000;-ftemplate-backtrace-limit=0;-ffunction-sections;-fdata-sections;-falign-functions=32;-fno-math-errno;-ffp-contract=on;-fvisibility=hidden;-fvisibility-inlines-hidden;-fno-rtti;-fno-asynchronous-unwind-tables;-fno-unwind-tables;-fno-stack-protector;-fno-ident;-pipe;-fno-common;-fwrapv;-D_FORTIFY_SOURCE=0;-Weverything;-Wnon-virtual-dtor;-Wno-c++98-compat;-Wno-c++98-compat-pedantic;-Wno-unsafe-buffer-usage;-Wno-padded;-Wno-c++20-compat;-Wno-exit-time-destructors>"#;-Werror>"
    "$<$<CXX_COMPILER_ID:AppleClang>:-O3;-funroll-loops;-fvectorize;-fslp-vectorize;-finline-functions;-fomit-frame-pointer;-fmerge-all-constants;-ftemplate-depth=2048;-fconstexpr-depth=2048;-fconstexpr-steps=50000000;-ftemplate-backtrace-limit=0;-ffunction-sections;-fdata-sections;-falign-functions=32;-fno-math-errno;-ffp-contract=on;-fvisibility=hidden;-fvisibility-inlines-hidden;-fno-rtti;-fno-asynchronous-unwind-tables;-fno-unwind-tables;-fno-stack-protector;-fno-ident;-pipe;-fno-common;-fwrapv;-D_FORTIFY_SOURCE=0;-Weverything;-Wnon-virtual-dtor;-Wno-c++98-compat;-Wno-c++98-compat-pedantic;-Wno-unsafe-buffer-usage;-Wno-padded;-Wno-c++20-compat;-Wno-exit-time-destructors;-Wno-poison-system-directories>"#;-Werror>"
    "$<$<CXX_COMPILER_ID:GNU>:-O3;;-funroll-loops;-finline-functions;-fomit-frame-pointer;-fno-math-errno;-ftemplate-depth=2000;-fconstexpr-depth=2000;-fconstexpr-ops-limit=100000000;-fconstexpr-loop-limit=1000000;-falign-functions=32;-falign-loops=32;-fprefetch-loop-arrays;-ftree-vectorize;-fstrict-aliasing;-ffunction-sections;-fdata-sections;-fvisibility=hidden;-fvisibility-inlines-hidden;-fno-keep-inline-functions;-fno-ident;-fmerge-all-constants;-fno-stack-protector;-fno-rtti;-fgcse-after-reload;-ftree-loop-distribute-patterns;-fpredictive-commoning;-funswitch-loops;-ftree-loop-vectorize;-ftree-slp-vectorize;-Wall;-Wextra;-Wpedantic;-Wnon-virtual-dtor;-Wlogical-op;-Wduplicated-cond;-Wduplicated-branches;-Wnull-dereference;-Wdouble-promotion>"#;-Werror>"
    "$<$<CXX_COMPILER_ID:MSVC>:/Ob3;/Ot;/Oy;/GT;/GL;/fp:precise;/Qpar;/constexpr:depth2048;/constexpr:backtrace0;/constexpr:steps2000000;/GS-;/Gy;/Gw;/Zc:inline;/Zc:throwingNew;/W4;/permissive-;/Zc:__cplusplus;/wd4820;/wd4324;/wd5002;/Zc:alignedNew;/Zc:auto;/Zc:forScope;/Zc:implicitNoexcept;/Zc:noexceptTypes;/Zc:referenceBinding;/Zc:rvalueCast;/Zc:sizedDealloc;/Zc:strictStrings;/Zc:ternary;/Zc:wchar_t>"#;/WX>"\
    "$<$<CUDA_COMPILER_ID:NVIDIA>:-O3;--fmad=false;--prec-div=true;--prec-sqrt=true;--restrict;--ftemplate-depth=2048;wwwwwww--extended-lambda>"
    "$<$<AND:$<CXX_COMPILER_ID:CLANG>,$<PLATFORM_ID:Linux>>:-fno-plt;-fno-semantic-interposition>"
    "${BNCH_SWT_SIMD_FLAGS}"
)

set(BNCH_SWT_LINK_OPTIONS
    "$<$<AND:$<CXX_COMPILER_ID:CLANG>,$<PLATFORM_ID:Darwin>>:-Wl,-dead_strip;-Wl,-x;-Wl,-S>"
    "$<$<AND:$<CXX_COMPILER_ID:AppleClang>,$<PLATFORM_ID:Darwin>>:-Wl,-dead_strip;-Wl,-x;-Wl,-S>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Darwin>>:-Wl,-dead_strip;-Wl,-x;-Wl,-S>"
    "$<$<AND:$<CXX_COMPILER_ID:CLANG>,$<PLATFORM_ID:Linux>>:-Wl,--gc-sections;-Wl,--strip-all;-Wl,--build-id=none;-Wl,--hash-style=gnu;-Wl,-z,now;-Wl,-z,relro>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Linux>>:-Wl,--gc-sections;-Wl,--strip-all;-Wl,--as-needed;-Wl,-O3>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<PLATFORM_ID:Windows>>:/DYNAMICBASE:NO;/OPT:REF;/OPT:ICF;/INCREMENTAL:NO;/MACHINE:X64>"
    "$<$<CUDA_COMPILER_ID:NVIDIA>:-lcudart_static;-lcublas_static;-lculibos;-lcudart;-lrt;-ldl;-lpthread;--relocatable-device-code=false>"
)

add_executable(
  "BenchmarkExe"
  "main.cu"
)

target_link_libraries(
	"BenchmarkExe" PUBLIC 
	#Jsonifier::Jsonifier glaze::glaze
    BenchmarkSuite::BenchmarkSuite
    CUDA::cudart_static
    CUDA::cublas
    #ggml
)

target_link_options(
	"BenchmarkExe" INTERFACE
	"${BNCH_SWT_LINK_OPTIONS}"
)

target_compile_options(
	"BenchmarkExe" INTERFACE
	"${BNCH_SWT_COMPILE_OPTIONS}"
)

target_compile_options(
	"BenchmarkExe" PUBLIC
	"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:/fsanitize=address>>"
     "$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:clang>,$<CXX_COMPILER_ID:CLANG>>:-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wnull-dereference;-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall>"
    "$<$<CXX_COMPILER_ID:MSVC>:/Wall;/W4>"
)

target_link_options(
	"BenchmarkExe" PUBLIC
	"$<$<CXX_COMPILER_ID:GNU>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:-fsanitize=address>>"
)

if (WIN32)
	install(
		FILES 
		"$<TARGET_PDB_FILE:BenchmarkExe>"
		DESTINATION "$<IF:$<CONFIG:Debug>,bin,bin>"
		OPTIONAL
	)
endif()

install(
	FILES 
	"$<TARGET_FILE:BenchmarkExe>"
	DESTINATION "$<IF:$<CONFIG:Debug>,bin,bin>"
	OPTIONAL
)

if(NOT BUILD_SHARED_LIBS)
    add_custom_command(TARGET BenchmarkExe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:ggml> $<TARGET_FILE_DIR:BenchmarkExe>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        $<TARGET_FILE:ggml-cuda> $<TARGET_FILE_DIR:BenchmarkExe>
        COMMENT "Copying GGML DLLs to output directory"
    )
endif()

if (WIN32)
	install(
		FILES 
		"$<TARGET_PDB_FILE:BenchmarkExe>"
		DESTINATION "bin"
		OPTIONAL
	)
endif()

install(
	FILES 
	"$<TARGET_FILE:BenchmarkExe>"
	DESTINATION "bin"
	OPTIONAL
)

#install(
    #FILES "$<TARGET_FILE:ggml>"
    #DESTINATION "$<TARGET_FILE_DIR:BenchmarkExe>"
    #OPTIONAL
#)
#install(
    #FILES "$<TARGET_FILE:ggml-cpu>"
    #DESTINATION "$<TARGET_FILE_DIR:BenchmarkExe>"
    #OPTIONAL
#)
#install(
    #FILES "$<TARGET_FILE:ggml-base>"
    #DESTINATION "$<TARGET_FILE_DIR:BenchmarkExe>"
    #OPTIONAL
#)

if (GENERATE_ASSEMBLY STREQUAL "TRUE")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
        set(ASM_OUTPUT "${CMAKE_SOURCE_DIR}/Assembly/Assembly.cod")
    
        add_custom_command(
            OUTPUT "${ASM_OUTPUT}"
            COMMAND "${CMAKE_CXX_COMPILER}" -S "${CMAKE_SOURCE_DIR}/Benchmark/main.cpp" -o "${ASM_OUTPUT}"
            -I "${CMAKE_SOURCE_DIR}/Build/_deps/jsonifier-src/Include/"
            -I "${CMAKE_SOURCE_DIR}/Build/_deps/glaze-src/include/"
            -I "${CMAKE_SOURCE_DIR}/Include/"
            -std=c++2b
            -DOPERATING_SYSTEM_NAME="TEST_OS" 
            -DOPERATING_SYSTEM_VERSION="TEST_OS_VERSION" 
            -DCOMPILER_VERSION="TEST_COMPILER_VERSION" 
            -DCOMPILER_ID="TEST_ID" 
            -DBASE_PATH="${CMAKE_SOURCE_DIR}"
            -DSIMDJSON_COMMIT="SIMDJSON_COMMIT"
            -DGLAZE_COMMIT="GLAZE_COMMIT"
            -DJSONIFIER_COMMIT="JSONIFIER_COMMIT"
            "-march=native"
            "-O3"
            "-DNDEBUG"
            DEPENDS "${CMAKE_SOURCE_DIR}/Benchmark/main.cpp"
            COMMENT "Generating assembly for main.cpp"
            VERBATIM
        )

        add_custom_target(generate_asm ALL DEPENDS "${ASM_OUTPUT}")
        add_dependencies("BenchmarkExe" generate_asm)

    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        set(ASM_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/Assembly")

        file(MAKE_DIRECTORY "${ASM_OUTPUT_DIR}")

        add_custom_command(
            OUTPUT "${ASM_OUTPUT_DIR}/main.asm"
            COMMAND "${CMAKE_CXX_COMPILER}" /Fa${ASM_OUTPUT_DIR}/ /FAsc /FoNUL /c 
            "${CMAKE_SOURCE_DIR}/Benchmark/main.cpp" 
            /I "${CMAKE_SOURCE_DIR}/Build/_deps/jsonifier-src/Include/"
            /I "${CMAKE_SOURCE_DIR}/Build/_deps/glaze-src/include/"
            /I "${CMAKE_SOURCE_DIR}/Include/"
            /std:c++latest
            /DOPERATING_SYSTEM_NAME="TEST_OS"
            /DOPERATING_SYSTEM_VERSION="TEST_OS_VERSION"
            /DCOMPILER_VERSION="TEST_COMPILER_VERSION"
            /DCOMPILER_ID="TEST_ID"
            /DBASE_PATH="${CMAKE_SOURCE_DIR}"
            /DSIMDJSON_COMMIT="SIMDJSON_COMMIT"
            /DGLAZE_COMMIT="GLAZE_COMMIT"
            /DJSONIFIER_COMMIT="JSONIFIER_COMMIT"
            "/Zc:preprocessor"
            "/permissive-"
            "/Zc:lambda" 
            "/bigobj" 
            "/GL" 
            "/Zi" 
            "/Gy" 
            "/arch:AVX" 
            "/arch:AVX2" 
            "/DWIN32" 
            "/D_WINDOWS" 
            "/GR" 
            "/EHsc" 
            "/O2" 
            "/Ob2" 
            "/DNDEBUG"
            DEPENDS "${CMAKE_SOURCE_DIR}/Benchmark/main.cpp"
            COMMENT "Generating human-readable assembly for main.cpp"
            VERBATIM
        )

        add_custom_target(generate_asm ALL DEPENDS "${ASM_OUTPUT_DIR}/main.asm")

        add_dependencies("BenchmarkExe" generate_asm)
    endif()
endif()
