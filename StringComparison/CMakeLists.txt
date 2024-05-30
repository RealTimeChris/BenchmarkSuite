# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/../CMake")

add_executable("StringComparisonExe" "main.cpp")

target_link_libraries(
	"StringComparisonExe"
	PUBLIC
	BenchmarkSuite::BenchmarkSuite
)

target_link_options(
	"StringComparisonExe"  PUBLIC
	"$<$<AND:$<STREQUAL:$<UPPER_CASE:${UBSAN}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
)

target_compile_options (
	"StringComparisonExe" PUBLIC
	"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:/fsanitize=address>>"
	"$<$<CXX_COMPILER_ID:GNU>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:-fsanitize=address>>"
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