cmake_minimum_required(VERSION 3.18)

project("BenchmarkExe" VERSION "1.0.0")

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

add_executable("BenchmarkExe" "./main.c")

target_compile_options(
	"BenchmarkExe"  PUBLIC
	"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:/fsanitize=address>>"
	"$<$<AND:$<STREQUAL:$<UPPER_CASE:${UBSAN}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wnull-dereference>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wuninitialized>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wconversion>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wpedantic>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wshadow>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wextra>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wall>"
	"$<$<CXX_COMPILER_ID:CLANG>:-mfma>"
	"$<$<CXX_COMPILER_ID:CLANG>:-mavx2>"
	"$<$<CXX_COMPILER_ID:GNU>:-mfma>"
	"$<$<CXX_COMPILER_ID:GNU>:-mavx2>"
	"-mfma"
	"-mavx2"
	"$<$<CXX_COMPILER_ID:GNU>:-Wnull-dereference>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wuninitialized>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wconversion>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wpedantic>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wshadow>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wextra>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wall>"

	"$<$<CXX_COMPILER_ID:MSVC>:/Wall>"
	"$<$<CXX_COMPILER_ID:MSVC>:/W4>"
)

target_link_libraries("BenchmarkExe" PUBLIC Jsonifier::Jsonifier glaze::glaze BenchmarkSuite::BenchmarkSuite)

if (WIN32)
  install(
    FILES "$<TARGET_PDB_FILE:BenchmarkExe>"
    DESTINATION "$<IF:$<CONFIG:Debug>,bin,bin>"
    OPTIONAL
  )
endif()

install(
  FILES "$<TARGET_FILE:BenchmarkExe>"
  DESTINATION "$<IF:$<CONFIG:Debug>,bin,bin>"
  OPTIONAL
)
