cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME "BenchmarkSuite")
set(CMAKE_CXX_STANDARD 20)

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

if (MSVC)
	string(REGEX MATCH "\/cl(.exe)?$" matched_cl ${CMAKE_CXX_COMPILER})
	if (matched_cl)

		target_compile_options(
			"${PROJECT_NAME}" INTERFACE
			$<$<CONFIG:Release>:/GL>
			$<$<CONFIG:MinSizeRel>:/GL>
			"/Zc:preprocessor"
			"/bigobj"
			"/permissive-"
			"/Zc:lambda"
		)

		target_link_options(
			"${PROJECT_NAME}" INTERFACE
			"$<$<CONFIG:Release>:/LTCG /INCREMENTAL:NO>"
			"$<$<CONFIG:MinSizeRel>:/LTCG /INCREMENTAL:NO>"
		)
	endif()
endif()

include (FetchContent)

if(NOT "${JSONIFIER_LIBRARY}" STREQUAL "TRUE")
	FetchContent_Declare(
	  jsonifier
	  GIT_REPOSITORY https://github.com/realtimechris/jsonifier.git
	  GIT_TAG dev
	  GIT_SHALLOW TRUE
	)
	FetchContent_MakeAvailable(jsonifier)
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

target_compile_definitions(
	"${PROJECT_NAME}" INTERFACE
	"$<$<STREQUAL:${ASAN_ENABLED},TRUE>:/fsanitize=address>"
)

if (NOT "${CPU_FREQUENCY_MHZ}" STREQUAL "")
	message("CPU_FREQUENCY_MHZ VALUE: ${CPU_FREQUENCY_MHZ}")
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
