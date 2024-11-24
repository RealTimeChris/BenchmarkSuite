cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME "BenchmarkSuite")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project("${PROJECT_NAME}" LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/../CMake;")

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/Include/benchmarksuite/*.hpp")

add_library("${PROJECT_NAME}" INTERFACE "${HEADERS}")

if (WIN32)
	set(OS "windows")
elseif(APPLE AND NOT UNIX)
	set(OS "osx")
else()
	set(OS "linux")
endif()

if(EXISTS VCPKG_DIR)
	set(Jsonifier_DIR "${VCPKG_DIR}/installed/x64-${OS}/share/jsonifier")
else()
	set(Jsonifier_DIR "C:/Users/Chris/source/repos/Jsonifier/out/share/jsonifier/")
endif()

add_library("BenchmarkSuite::BenchmarkSuite" ALIAS "${PROJECT_NAME}")

set(CMAKE_VERBOSE_MAKEFILE ON)

include (FetchContent)

if(NOT "${JSONIFIER_LIBRARY}" STREQUAL "TRUE")
	FetchContent_Declare(
	  Jsonifier
	  GIT_REPOSITORY https://github.com/realtimechris/Jsonifier.git
	  GIT_TAG dev
	  GIT_SHALLOW TRUE
	)
	FetchContent_MakeAvailable(Jsonifier)
endif()

set_target_properties(
	"${PROJECT_NAME}" PROPERTIES
	OUTPUT_NAME "benchmarksuite"
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
)

target_link_libraries(
	"${PROJECT_NAME}" INTERFACE
	Jsonifier::Jsonifier
)

target_include_directories(
	"${PROJECT_NAME}" INTERFACE
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include/>"
)

target_link_options(
	"${PROJECT_NAME}" INTERFACE
	"$<$<AND:$<STREQUAL:$<UPPER_CASE:${UBSAN}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
	"$<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO>"
	"$<$<CXX_COMPILER_ID:MSVC>:/DEBUG>"
	"$<$<CXX_COMPILER_ID:MSVC>:/LTCG>"
)

set(ASM_OUTPUT_DIR "${CMAKE_BINARY_DIR}/assembly")
file(MAKE_DIRECTORY ${ASM_OUTPUT_DIR})

target_compile_options(
	"${PROJECT_NAME}" INTERFACE
	"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:/fsanitize=address>>"
	"$<$<AND:$<STREQUAL:$<UPPER_CASE:${UBSAN}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
	"$<$<CXX_COMPILER_ID:MSVC>:/constexpr:steps 2097152>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wno-ignored-attributes>"	
	"$<$<CXX_COMPILER_ID:CLANG>:UBSAN_OPTIONS=log_path=${CMAKE_SOURCE_DIR}/>"
	"$<$<CXX_COMPILER_ID:MSVC>:/Zc:preprocessor>"
	"$<$<CXX_COMPILER_ID:CLANG>:-march=native>"
	"$<$<CXX_COMPILER_ID:MSVC>:/permissive->"
	"$<$<CXX_COMPILER_ID:GNU>:-march=native>"
	"$<$<CXX_COMPILER_ID:CLANG>:-std=c++23>"
	"$<$<CXX_COMPILER_ID:MSVC>:/Zc:lambda>"
	"$<$<CXX_COMPILER_ID:GNU>:-std=c++23>"
	"$<$<CXX_COMPILER_ID:MSVC>:/fp:fast>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wextra>"
	"$<$<CXX_COMPILER_ID:MSVC>:/bigobj>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wall>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wextra>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wall>"
	"$<$<CXX_COMPILER_ID:MSVC>:/GL>"
	"$<$<CXX_COMPILER_ID:MSVC>:/Zi>"
	"$<$<CXX_COMPILER_ID:MSVC>:/Gy>"
)

target_compile_definitions(
	"${PROJECT_NAME}" INTERFACE
	"JSON_BASE_PATH=\"${CMAKE_SOURCE_DIR}\""
)

if (NOT "${CPU_FREQUENCY_MHZ}" STREQUAL "")
	target_compile_definitions(
		"${PROJECT_NAME}" INTERFACE
		"CPU_FREQUENCY_MHZ=${CPU_FREQUENCY_MHZ}"
	)
endif()

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

install(
	DIRECTORY
	"${CMAKE_CURRENT_SOURCE_DIR}/Include/"
	DESTINATION "include"
)

install(
	TARGETS "${PROJECT_NAME}"
	EXPORT "${EXPORTED_TARGETS_NAME}"
	RUNTIME DESTINATION "bin"
	ARCHIVE DESTINATION "lib"
)

install(
	EXPORT "${EXPORTED_TARGETS_NAME}"
	FILE "${EXPORTED_TARGETS_FILE_NAME}"
	NAMESPACE "BenchmarkSuite::"
	DESTINATION "share/benchmarksuite"
)

if (WIN32 AND "${BUILD_SHARED_LIBS}")
	install(
		FILES 
		"$<TARGET_PDB_FILE:BenchmarkSuite>"
		DESTINATION "bin"
		OPTIONAL
	)
endif()

if ("${BENCH_TYPE}" STREQUAL "STRING_COMPARISON")
	add_subdirectory("StringComparison")
endif()
