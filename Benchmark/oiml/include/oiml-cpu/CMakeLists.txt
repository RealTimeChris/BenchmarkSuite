file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

add_library(
  "oiml-cpu" 
  STATIC 
  "cpu_arch.cpp" 
  "${HEADERS}"
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi /DEBUG /ZI ${AVX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /DEBUG /ZI ${AVX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /DEBUG")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DEBUG")
endif()

add_library("oiml-cpu::oiml-cpu" ALIAS "oiml-cpu")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64|i[3-6]86")  
    message(STATUS "Architecture: x86_64")
    set(OIML_COMPILE_OPTIONS 
        "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,gnu>:-Wnull-dereference;-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall;-mavx512f;-mfma;-mavx2;-mavx;-flto>"
        "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,clang>:-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall;-mavx512f;-mfma;-mavx2;-mavx;-flto>"
        "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,msvc>:/Wall;/W4;/arch:AVX512;/arch:AVX2;/arch:AVX;>"
    )
    set(OIML_LINK_OPTIONS 
        "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,gnu>:-flto>"
        "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,appleclang>:-flto>"
        "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,msvc>:/LTCG>"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM.*")  
    message(STATUS "Architecture: arm64")
    set(OIML_COMPILE_OPTIONS 
        "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,gnu>:-Wnull-dereference;-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall;-march=armv8.2-a+sve+sve2+simd>"
        "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,appleclang>:-Wnull-dereference;-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall;-march=armv8.2-a+sve+sve2+simd>"
        "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,clang>:-Wnull-dereference;-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall;-march=armv8.2-a+sve+sve2+simd>"
    )
    set(OIML_LINK_OPTIONS 
        "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,gnu>:-flto>"
        "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,appleclang>:-flto>"
        "$<$<STREQUAL:$<LOWER_CASE:$<CXX_COMPILER_ID>>,clang>:-flto>"
    )
else()
    message(STATUS "Architecture: Unknown")
endif()

target_compile_options(
    "oiml-cpu" PRIVATE 
    "${OIML_COMPILE_OPTIONS}"
)

target_link_options(
    "oiml-cpu" PRIVATE 
    "${OIML_LINK_OPTIONS}"
)

target_include_directories(
	"oiml-cpu" PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>"
    "$<INSTALL_INTERFACE:/include>"
)

