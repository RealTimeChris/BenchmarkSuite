cmake_minimum_required(VERSION 3.18)

project(
  "BenchmarkExe"
  VERSION "1.0.0"
  LANGUAGES CXX
)

include(FetchContent)
include(GetCommitHash)

FetchContent_Declare(
  jsonifier
  GIT_REPOSITORY https://github.com/realtimechris/jsonifier.git
  GIT_TAG dev
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(jsonifier)
getCommitHash("${jsonifier_SOURCE_DIR}" JSONIFIER_COMMIT_HASH)
message(STATUS "Jsonifier Commit Hash: https://github.com/RealTimeChris/Jsonifier/commit/${JSONIFIER_COMMIT_HASH}")

FetchContent_Declare(
  glaze
  GIT_REPOSITORY https://github.com/stephenberry/glaze.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glaze)

add_executable(
  "BenchmarkExe" 
  "./main.cpp"
)

target_link_libraries(
	"BenchmarkExe" PUBLIC 
	Jsonifier::Jsonifier glaze::glaze BenchmarkSuite::BenchmarkSuite
)

target_compile_options(
	"BenchmarkExe" PUBLIC
	"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:/fsanitize=address>>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wpedantic>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wpedantic>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wextra>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wall>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wextra>"
    "$<$<CXX_COMPILER_ID:CLANG>:-mfma>"
    "$<$<CXX_COMPILER_ID:Clang>:-mfma>"
    "$<$<CXX_COMPILER_ID:GNU>:-mfma>"
	"$<$<CXX_COMPILER_ID:MSVC>:/Wall>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wall>"
	"$<$<CXX_COMPILER_ID:MSVC>:/W4>"
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
