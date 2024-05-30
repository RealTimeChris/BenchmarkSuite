# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/../CMake")

include (FetchContent)

FetchContent_Declare(
  glaze
  GIT_REPOSITORY https://github.com/stephenberry/glaze.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glaze)

add_executable("StringComparisonExe"   "main.cpp")

FetchContent_Declare(
	  simdjson2
	  GIT_REPOSITORY https://github.com/realtimechris/simdjson.git
	  GIT_TAG master
	  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(simdjson2)

FetchContent_Declare(
	  simdjson
	  GIT_REPOSITORY https://github.com/simdjson/simdjson.git
	  GIT_TAG master
	  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(simdjson)

target_link_libraries(
	"StringComparisonExe"
	PUBLIC
	BenchmarkSuite::BenchmarkSuite
	glaze::glaze
	Jsonifier::Jsonifier
	simdjson2::simdjson2
	simdjson::simdjson
)

target_link_options(
	"StringComparisonExe"  PUBLIC
	"$<$<AND:$<STREQUAL:$<UPPER_CASE:${UBSAN}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
	"$<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO>"
	"$<$<CXX_COMPILER_ID:MSVC>:/DEBUG>"
	"$<$<CXX_COMPILER_ID:MSVC>:/LTCG>"
)

target_compile_options (
	"StringComparisonExe" PUBLIC
	"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:/fsanitize=address>>"
	"$<$<CXX_COMPILER_ID:GNU>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:-fsanitize=address>>"
	#"$<$<CXX_COMPILER_ID:GNU>:-S test.asm>"
	"$<$<AND:$<STREQUAL:$<UPPER_CASE:${UBSAN}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
)

target_compile_definitions(
	"StringComparisonExe" PUBLIC
	"$<$<STREQUAL:${ASAN_ENABLED},TRUE>:ASAN>"
	"JSON_TEST_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}\""
)

if (WIN32)
	install(
		FILES 
		"$<TARGET_PDB_FILE:StringComparisonExe>"
		DESTINATION "Tests/${CMAKE_BUILD_TYPE}"
		OPTIONAL
	)
endif()

install(
	FILES 
	"$<TARGET_FILE:StringComparisonExe>"
	DESTINATION "Tests/${CMAKE_BUILD_TYPE}"
	OPTIONAL
)

install(
	FILES 
	"$<TARGET_RUNTIME_DLLS:StringComparisonExe>"
	DESTINATION "Tests/${CMAKE_BUILD_TYPE}"
	OPTIONAL
)