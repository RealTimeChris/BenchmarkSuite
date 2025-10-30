# Copyright (c) 2025 RealTimeChris (Chris M.)
# 
# This file is part of software offered under a restricted-use license to a designated Licensee,
# whose identity is confirmed in writing by the Author.
# 
# License Terms (Summary):
# - Exclusive, non-transferable license for internal use only.
# - Redistribution, sublicensing, or public disclosure is prohibited without written consent.
# - Full ownership remains with the Author.
# - License may terminate if unused for [X months], if materially breached, or by mutual agreement.
# - No warranty is provided, express or implied.
# 
# Full license terms are provided in the LICENSE file distributed with this software.
# 
# Signed,
# RealTimeChris (Chris M.)
# 2025
cmake_minimum_required(VERSION 3.28)

if(WIN32)
    set(VCPKG_TARGET_TRIPLET x64-windows-static CACHE STRING "" FORCE)
elseif(UNIX AND NOT APPLE)
    set(VCPKG_TARGET_TRIPLET x64-linux CACHE STRING "" FORCE)
elseif(APPLE)
    set(VCPKG_TARGET_TRIPLET x64-osx CACHE STRING "" FORCE)
endif()

set(BNCH_SWT_LANGUAGE CXX)
project(BenchmarkSuite
    VERSION 1.0.0
    LANGUAGES CXX
)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BNCH_SWT_INSTALL "Install benchmarksuite as a library" ON)

set(BNCH_SWT_COMPILER_ID ${CMAKE_${BNCH_SWT_LANGUAGE}_COMPILER_ID})

file(GLOB_RECURSE BNCH_SWT_HEADERS CONFIGURE_DEPENDS include/*.hpp)

add_library(${PROJECT_NAME} INTERFACE ${BNCH_SWT_HEADERS})

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

include(cmake/flags_and_options.cmake)

target_include_directories(${PROJECT_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(${PROJECT_NAME}
    INTERFACE
        $<$<CONFIG:Release>:${BNCH_SWT_COMPILE_OPTIONS}>
)

target_link_options(${PROJECT_NAME}
    INTERFACE
        $<$<CONFIG:Release>:${BNCH_SWT_LINK_OPTIONS}>
)

target_compile_definitions(${PROJECT_NAME}
    INTERFACE
        ${BNCH_SWT_COMPILE_DEFINITIONS}
)

set(CONFIG_FILE_NAME "${PROJECT_NAME}Config.cmake")
set(EXPORTED_TARGETS_NAME "${PROJECT_NAME}Targets")
set(EXPORTED_TARGETS_FILE_NAME "${EXPORTED_TARGETS_NAME}.cmake")
set(EXPORTED_TARGETS_FILE_PATH "share/benchmarksuite/${EXPORTED_TARGETS_FILE_NAME}")

include(CMakePackageConfigHelpers)
configure_package_config_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CONFIG_FILE_NAME}.in"
	"${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
	INSTALL_DESTINATION "share/benchmarksuite"
	PATH_VARS
	EXPORTED_TARGETS_FILE_PATH
)

set(VERSION_FILE_NAME "${PROJECT_NAME}ConfigVersion.cmake")

write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
	VERSION "${PRODUCT_VERSION}"
	COMPATIBILITY AnyNewerVersion
)

if (BNCH_SWT_INSTALL)
	install(
		FILES
		"${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
		"${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
		DESTINATION "share/benchmarksuite"
	)

	install(
		DIRECTORY
		"${CMAKE_CURRENT_SOURCE_DIR}/include/"
		DESTINATION "include"
	)

	install(
		TARGETS "${PROJECT_NAME}"
		EXPORT "${EXPORTED_TARGETS_NAME}"
	)

	install(
		EXPORT "${EXPORTED_TARGETS_NAME}"
		FILE "${EXPORTED_TARGETS_FILE_NAME}"
		NAMESPACE "${PROJECT_NAME}::"
		DESTINATION "share/benchmarksuite"
	)
endif()

message(STATUS "BenchmarkSuite Configuration Summary:")
message(STATUS "=============================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Max Thread Count: ${BNCH_SWT_THREAD_COUNT}")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler: ${BNCH_SWT_COMPILER_ID}")

if(BENCHMARKS)
    add_subdirectory(Benchmark)
endif()