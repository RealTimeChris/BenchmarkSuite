cmake_minimum_required(VERSION 3.18)

project(
  "oiml_test"
  VERSION "1.0.0"
  LANGUAGES CXX
)

include(FetchContent)
	FetchContent_Declare(
	BenchmarkSuite
	GIT_REPOSITORY https://github.com/RealTimeChris/BenchmarkSuite.git
	GIT_TAG main
)
FetchContent_MakeAvailable(BenchmarkSuite)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_executable(
  "oiml_test" 
  "./test_main.cpp"
)

target_link_libraries(
	"oiml_test"
	PUBLIC 
	oiml::oiml
	BenchmarkSuite::BenchmarkSuite
)

target_compile_options(
	"oiml_test" PUBLIC
	"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:/fsanitize=address>>"
)

target_link_options(
	"oiml_test" PUBLIC
	"$<$<CXX_COMPILER_ID:GNU>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:-fsanitize=address>>"
)

if (WIN32)
	install(
		FILES 
		"$<TARGET_PDB_FILE:oiml_test>"
		DESTINATION "bin"
		OPTIONAL
	)
endif()

install(
	FILES 
	"$<TARGET_FILE:oiml_test>"
	DESTINATION "bin"
	OPTIONAL
)
