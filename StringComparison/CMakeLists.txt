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

add_executable("StringComparisonExe"  "random_generators.h" "main.cpp")

target_link_libraries(
	"StringComparisonExe"
	PUBLIC
	BenchmarkSuite
	glaze::glaze
	Jsonifier::Jsonifier
)

target_compile_options (
	"StringComparisonExe" PUBLIC
	"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:/fsanitize=address>>"
	"$<$<CXX_COMPILER_ID:GNU>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:-fsanitize=address>>"
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