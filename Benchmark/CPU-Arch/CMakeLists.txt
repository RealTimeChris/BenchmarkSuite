file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

add_library(
  "CPU-Arch" 
  STATIC 
  "cpu_arch.cpp" 
  "${HEADERS}"
)

add_library("CPU-Arch::CPU-Arch" ALIAS "CPU-Arch")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64|i[3-6]86")  
    message(STATUS "Architecture: x86_64")
    set(OIML_COMPILE_OPTIONS 
        "$<$<CXX_COMPILER_ID:MSVC>:/Wall;/W4;/arch:AVX512;/arch:AVX2;/arch:AVX;/GL>"
        "$<$<CXX_COMPILER_ID:GNU>:-Wnull-dereference;-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall;-mavx512f;-mfma;-mavx2;-mavx;-flto>"
        "$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:clang>,$<CXX_COMPILER_ID:CLANG>>:-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall;-mavx512f;-mfma;-mavx2;-mavx;-flto>"
    )
    set(OIML_LINK_OPTIONS 
        "$<$<CXX_COMPILER_ID:GNU>:-flto>"
        "$<$<CXX_COMPILER_ID:Clang>:-flto>"
        "$<$<CXX_COMPILER_ID:MSVC>:/LTCG>"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM.*")  
    message(STATUS "Architecture: arm64")
    set(OIML_COMPILE_OPTIONS 
        "$<$<CXX_COMPILER_ID:GNU>:-Wnull-dereference;-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall;-march=armv8.5-a+sve2>"
        "$<$<CXX_COMPILER_ID:AppleClang>:-Wnull-dereference;-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall;-march=armv8.5-a+sve2>"
        "$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:clang>,$<CXX_COMPILER_ID:CLANG>>:-Wuninitialized;-Wconversion;-Wpedantic;-Wshadow;-Wextra;-Wall;-march=armv8.5-a+sve2>"
    )
    set(OIML_LINK_OPTIONS 
        "$<$<CXX_COMPILER_ID:GNU>:-flto>"
        "$<$<CXX_COMPILER_ID:Clang>:-flto>"
        "$<$<CXX_COMPILER_ID:MSVC>:/LTCG>"
    )
else()
    message(STATUS "Architecture: Unknown")
endif()

target_compile_options(
    "CPU-Arch" PRIVATE 
    "${OIML_COMPILE_OPTIONS}"
)

target_link_options(
    "CPU-Arch" PRIVATE 
    "${OIML_LINK_OPTIONS}"
)

target_include_directories(
	"CPU-Arch" PUBLIC
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:/include>"
)

