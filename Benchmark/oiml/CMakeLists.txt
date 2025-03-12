#	MIT License
#
#	Copyright (c) 2025 OpenInfer
#
#	Permission is hereby granted, free of charge, to any person obtaining a copy of this 
#	software and associated documentation files (the "Software"), to deal in the Software 
#	without restriction, including without limitation the rights to use, copy, modify, merge, 
#	publish, distribute, sublicense, and/or sell copies of the Software, and to permit 
#	persons to whom the Software is furnished to do so, subject to the following conditions:
#
#	The above copyright notice and this permission notice shall be included in all copies or 
#	substantial portions of the Software.
#
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
#	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
#	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
#	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
#	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
#	DEALINGS IN THE SOFTWARE.
#
# CMakeLists.txt - The CMake script for building this library.

cmake_minimum_required(VERSION 3.18)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/CMake;")

set(PRODUCT_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(
  "oiml"
  VERSION "1.0.0"
  LANGUAGES CXX
)

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/oiml/common/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/include/oiml/legacy/*.hpp")

add_library("${PROJECT_NAME}" INTERFACE "${HEADERS}")

add_library("${PROJECT_NAME}::${PROJECT_NAME}" ALIAS "${PROJECT_NAME}")

set_target_properties(
	"${PROJECT_NAME}" PROPERTIES
    OUTPUT_NAME "oiml"
	CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

find_package(Threads REQUIRED)

if (OIML_CUDA_ON)
	add_subdirectory(include/oiml/legacy/oiml-legacy-cuda)
endif()
 
target_include_directories(
    "${PROJECT_NAME}" INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    "$<INSTALL_INTERFACE:include/>"
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi /DEBUG /ZI ${AVX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /DEBUG /ZI ${AVX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /DEBUG")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DEBUG")
endif()

target_compile_options(
    "${PROJECT_NAME}" INTERFACE
    "$<$<AND:$<BOOL:${UBSAN}>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
    "$<$<AND:$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,msvc>,$<BOOL:${ASAN_ENABLED}>>:/fsanitize=address>"
    "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,gnu>:-Wnull-dereference;-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall;-Wno-narrowing>"
    "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,clang>:-Wnull-dereference;-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall;-Wno-narrowing>"
    "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,appleclang>:-Wnull-dereference;-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall;-Wno-narrowing>"
    "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,msvc>:/Wall;/W4;/Zi;/DNDBUG;/Zc:preprocessor;/permissive;/Zc:lambda;/bigobj;/Gy>"
)
 
target_link_options(
    "${PROJECT_NAME}" INTERFACE
    "$<$<AND:$<STREQUAL:$<UPPER_CASE:${UBSAN}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
    "$<$<AND:$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,clang>,$<STREQUAL:${ASAN_ENABLED},TRUE>>:-fsanitize=address>"
    "$<$<AND:$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,gnu>,$<STREQUAL:${ASAN_ENABLED},TRUE>>:-fsanitize=address>"
    "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,gnu>:-flto;-Wl,-O2>"
    "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,clang>:-flto;-Wl,-O2>"
    "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,appleclang>:-flto;-Wl,-O2>"
    "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,msvc>:/DEBUG;/OPT:REF;/OPT:ICF;/INCREMENTAL:NO;/LTCG>"
)

add_subdirectory(include/oiml-cpu)

target_link_libraries(
    "${PROJECT_NAME}" INTERFACE 
    "$<$<BOOL:${OIML_CUDA_ON}>:oiml-cuda>"
    Threads::Threads oiml-cpu::oiml-cpu
)

target_compile_definitions(
    "${PROJECT_NAME}" INTERFACE 
    "$<$<STREQUAL:${ASAN_ENABLED},TRUE>:ASAN_ENABLED>"
    "$<$<BOOL:${OIML_CUDA_ON}>:OIML_USE_CUDA=1>"
)
  
set(CONFIG_FILE_NAME "${PROJECT_NAME}Config.cmake")
set(EXPORTED_TARGETS_NAME "${PROJECT_NAME}Targets")
set(EXPORTED_TARGETS_FILE_NAME "${EXPORTED_TARGETS_NAME}.cmake")
set(EXPORTED_TARGETS_FILE_PATH "share/oiml/${EXPORTED_TARGETS_FILE_NAME}")

include(CMakePackageConfigHelpers)
configure_package_config_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/CMake/${CONFIG_FILE_NAME}.in"
	"${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
	INSTALL_DESTINATION "share/oiml"
	PATH_VARS
	EXPORTED_TARGETS_FILE_PATH
)

set(VERSION_FILE_NAME "${PROJECT_NAME}ConfigVersion.cmake")

write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
	VERSION "${PRODUCT_VERSION}"
	COMPATIBILITY AnyNewerVersion
)

set(OIML_CMAKE_EXPORTS "${PROJECT_NAME}")

if (TARGET oiml-cuda)
    install(
        TARGETS "${PROJECT_NAME}" oiml-cuda oiml-cpu
        EXPORT "${EXPORTED_TARGETS_NAME}"
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
else()
    install(
        TARGETS "${PROJECT_NAME}" oiml-cpu
        EXPORT "${EXPORTED_TARGETS_NAME}"
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

install(
	EXPORT "${EXPORTED_TARGETS_NAME}"
	FILE "${EXPORTED_TARGETS_FILE_NAME}"
	NAMESPACE "${PROJECT_NAME}::"
	DESTINATION "share/oiml"
)

install(
	FILES
	"${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
	"${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
	DESTINATION "share/oiml"
)

install(
	DIRECTORY
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
	DESTINATION "include"
)

if("${OIML_TESTS}" STREQUAL "TRUE")
	add_subdirectory("./tests")
endif()
