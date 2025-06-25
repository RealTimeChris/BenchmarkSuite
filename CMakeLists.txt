cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME "BenchmarkSuite")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project("${PROJECT_NAME}" LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/CMake;")

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/Include/benchmarksuite/*.hpp")

add_library("${PROJECT_NAME}" INTERFACE "${HEADERS}")

add_library("BenchmarkSuite::BenchmarkSuite" ALIAS "${PROJECT_NAME}")

set(CMAKE_VERBOSE_MAKEFILE ON)

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
	"$<$<AND:$<STREQUAL:$<UPPER_CASE:${UBSAN}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
)

target_compile_options(
	"${PROJECT_NAME}" INTERFACE
	"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:/fsanitize=address>>"
	"$<$<AND:$<STREQUAL:$<UPPER_CASE:${UBSAN}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
	#"$<$<CXX_COMPILER_ID:MSVC>:/wd4710;/wd4711;/Wall;/WX;/permissive-;/analyze;/sdl;/utf-8;/guard:cf;/Zc:__cplusplus;/Zc:alignedNew;/Zc:auto;/Zc:char8_t;/Zc:externC;/Zc:externConstexpr;/Zc:forScope;/Zc:hiddenFriend;/Zc:implicitNoexcept;/Zc:inline;/Zc:lambda;/Zc:noexceptTypes;/Zc:preprocessor;/Zc:referenceBinding;/Zc:rvalueCast;/Zc:sizedDealloc;/Zc:strictStrings;/Zc:ternary;/Zc:threadSafeInit;/Zc:throwingNew;/Zc:trigraphs;/Zc:twoPhase;/Zc:wchar_t>"
    #"$<$<CXX_COMPILER_ID:Clang>:-Weverything;-Werror;-Wno-c++98-compat;-Wno-c++98-compat-pedantic;-Wno-padded;-Wno-weak-vtables;-Wno-exit-time-destructors;-Wno-global-constructors>"
    #"$<$<CXX_COMPILER_ID:GNU>:-Wall;-Wextra;-Wpedantic;-Werror;-Wconversion;-Wsign-conversion;-Wcast-align;-Wcast-qual;-Wctor-dtor-privacy;-Wdisabled-optimization;-Wformat=2;-Winit-self;-Wlogical-op;-Wmissing-declarations;-Wnoexcept;-Wnon-virtual-dtor;-Wold-style-cast;-Woverloaded-virtual;-Wredundant-decls;-Wshadow;-Wsign-promo;-Wstrict-null-sentinel;-Wstrict-overflow=2;-Wswitch-default;-Wundef;-Wunreachable-code;-Wunused;-Wuseless-cast;-Wvariadic-macros;-Wwrite-strings>"
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

install(
	DIRECTORY
	"${CMAKE_CURRENT_SOURCE_DIR}/Include/"
	DESTINATION "include"
)

if ("${BENCH_TYPE}" STREQUAL "BENCHMARK")
	add_subdirectory("Benchmark")
endif()
