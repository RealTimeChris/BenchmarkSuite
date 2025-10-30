cmake_minimum_required(VERSION 3.18)

project(
  "BenchmarkExe"
  VERSION "1.0.0"
  LANGUAGES CXX
)

add_executable(
  "BenchmarkExe" 
  "./main.cpp"
)

target_link_libraries(
	"BenchmarkExe" PUBLIC 
	BenchmarkSuite::BenchmarkSuite
)

target_compile_options(
	"BenchmarkExe" PUBLIC
	"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:/fsanitize=address>>"
     "$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:clang>,$<CXX_COMPILER_ID:CLANG>>:-Werror>"
     "$<$<CXX_COMPILER_ID:AppleClang>:-Werror>"
    "$<$<CXX_COMPILER_ID:GNU>:-Werror>"
    "$<$<CXX_COMPILER_ID:MSVC>:/WX;/W4>"
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

if (GENERATE_ASSEMBLY STREQUAL "TRUE")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
        set(ASM_OUTPUT "${CMAKE_SOURCE_DIR}/Assembly/Assembly.cod")

        file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/Assembly")

        set(ASM_OUTPUT "${CMAKE_SOURCE_DIR}/Assembly/Assembly.cod")
        set(BNCH_SWT_ASM_DEFINITIONS "")

        if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_COMPILER_CUDA=1")
        else()
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_COMPILER_CUDA=0")
        endif()

        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_ARCH_X64=1")
        else()
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_ARCH_X64=0")
        endif()

        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR 
           CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64" OR 
           CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_ARCH_ARM64=1")
        else()
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_ARCH_ARM64=0")
        endif()

        if(WIN32)
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_PLATFORM_WINDOWS=1")
        else()
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_PLATFORM_WINDOWS=0")
        endif()

        if(UNIX AND NOT APPLE)
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_PLATFORM_LINUX=1")
        else()
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_PLATFORM_LINUX=0")
        endif()

        if(APPLE)
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_PLATFORM_MAC=1")
        else()
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_PLATFORM_MAC=0")
        endif()

        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_COMPILER_CLANG=1")
        else()
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_COMPILER_CLANG=0")
        endif()

        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_COMPILER_MSVC=1")
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DNOMINMAX")
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DWIN32_LEAN_AND_MEAN")
        else()
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_COMPILER_MSVC=0")
        endif()

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_COMPILER_GNU=1")
        else()
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_COMPILER_GNU=0")
        endif()

        if(BNCH_SWT_DEV)
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_DEV=1")
        else()
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_DEV=0")
        endif()

        if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_CUDA_TENSOR_CORES=1")
        else()
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_CUDA_TENSOR_CORES=0")
        endif()

        if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_CUDA_MAX_REGISTERS=128")
        else()
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_CUDA_MAX_REGISTERS=0")
        endif()

        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_INLINE=[[msvc::forceinline]] inline")
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_STATIC_INLINE=[[msvc::forceinline]] static inline")
        else()
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_INLINE=__attribute__((always_inline)) inline")
            list(APPEND BNCH_SWT_ASM_DEFINITIONS "-DBNCH_SWT_STATIC_INLINE=__attribute__((always_inline)) static inline")
        endif()

        add_custom_command(
            OUTPUT "${ASM_OUTPUT}"
            COMMAND "${CMAKE_CXX_COMPILER}" 
                -S "${CMAKE_SOURCE_DIR}/Benchmark/main.cpp" 
                -o "${ASM_OUTPUT}"
                -I "${CMAKE_SOURCE_DIR}/Build/_deps/benchmarksuite-src/include/"
                -I "${CMAKE_SOURCE_DIR}/Build/_deps/glaze-src/include/"
                -I "${CMAKE_SOURCE_DIR}/include/"
                -std=c++2b
                -DOPERATING_SYSTEM_NAME=TEST_OS
                -DOPERATING_SYSTEM_VERSION=TEST_OS_VERSION
                -DCOMPILER_VERSION=TEST_COMPILER_VERSION
                -DCOMPILER_ID=TEST_ID
                -DBASE_PATH=${CMAKE_SOURCE_DIR}
                -DSIMDJSON_COMMIT=SIMDJSON_COMMIT
                -DGLAZE_COMMIT=GLAZE_COMMIT
                -DJSONIFIER_COMMIT=JSONIFIER_COMMIT
                ${BNCH_SWT_ASM_DEFINITIONS}                  -march=native
                -O3
                -DNDEBUG
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
            /I "${CMAKE_SOURCE_DIR}/Build/_deps/benchmarksuite-src/include/"
            /I "${CMAKE_SOURCE_DIR}/Build/_deps/glaze-src/include/"
            /I "${CMAKE_SOURCE_DIR}/include/"
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
