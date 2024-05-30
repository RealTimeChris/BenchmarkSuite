# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/../CMake")

include (FetchContent)

FetchContent_Declare(
  jsonifier
  GIT_REPOSITORY https://github.com/realtimechris/jsonifier.git
  GIT_TAG dev-new
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(jsonifier)

add_executable("BenchmarkExe" "main.cpp")

target_link_libraries(
	"BenchmarkExe"
	PUBLIC
	BenchmarkSuite::BenchmarkSuite
	Jsonifier::Jsonifier
)

target_link_options(
	"BenchmarkExe"  PUBLIC
	"$<$<AND:$<STREQUAL:$<UPPER_CASE:${UBSAN}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
)

target_compile_options (
	"BenchmarkExe" PUBLIC
	"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:/fsanitize=address>>"
	"$<$<CXX_COMPILER_ID:GNU>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:-fsanitize=address>>"
	"$<$<AND:$<STREQUAL:$<UPPER_CASE:${UBSAN}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
)

target_compile_definitions(
	"BenchmarkExe" PUBLIC
	"$<$<STREQUAL:${ASAN_ENABLED},TRUE>:ASAN>"
	"JSON_TEST_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}\""
)

get_target_property(linked_libraries "BenchmarkExe" LINK_LIBRARIES)

function(strip_compiler_condition input_string output_variable)
    string(REGEX MATCH ":([^>]+)>$" match "${input_string}")
    
    if (match)
        string(REGEX REPLACE "^:([^>]+)>$" "\\1" result "${match}")
        set(${output_variable} "${result}" PARENT_SCOPE)
    else()
        set(${output_variable} "" PARENT_SCOPE)
    endif()
endfunction()

function(strip_outer_expression input_string output_variable)
    string(REGEX MATCH "\\$<\\$<STREQUAL:[^>]+>:[^>]+>>" match "${input_string}")
    
    if (match)
        string(REGEX REPLACE "^\\$<\\$<CXX_COMPILER_ID:${CXX_COMPILER_ID}^>]+>:" "" result "${match}")
        set(${output_variable} "${result}" PARENT_SCOPE)
    else()
        set(${output_variable} "" PARENT_SCOPE)
    endif()
endfunction()

foreach(library ${linked_libraries})
    get_target_property(CMAKE_OPTIONS_RAW "${library}" INTERFACE_COMPILE_OPTIONS)
    set(CMAKE_OPTIONS)
    foreach(flag ${CMAKE_OPTIONS_RAW})
        strip_outer_expression("${flag}" new_flag)
        if (NOT new_flag)
            strip_compiler_condition ("${flag}" new_flag)
            if (NOT new_flag)
                set(new_flag "${flag}")
            endif()
        endif()
        list(APPEND CMAKE_OPTIONS "${new_flag}")
    endforeach()
endforeach()

function(purge_undefined_flags input_list)
    set(output_list_new)
    foreach(item IN LISTS input_list)
    if(NOT item MATCHES ".*undefined.*" AND NOT item MATCHES ".*fsanitize.*")
            list(APPEND output_list_new "${item}")
    endif()        
    endforeach()
    set(CMAKE_OPTIONS_NEW "${output_list_new}" PARENT_SCOPE)

endfunction()

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_NEW)
set(CMAKE_OPTIONS_FLAG CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_NEW})
set(CMAKE_OPTIONS "${CMAKE_OPTIONS};${CMAKE_CXX_FLAGS};${${CMAKE_OPTIONS_FLAG}}")

function(replace_semicolon_with_space input_list output_string)
    set(${output_string} "")
    string(REPLACE ";" " " item "${input_list}")
    set(${output_string} ${item} PARENT_SCOPE)
endfunction()

purge_undefined_flags("${CMAKE_OPTIONS}")

replace_semicolon_with_space("${CMAKE_OPTIONS_NEW}" CMAKE_OPTIONS_NEW)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(ASM_OUTPUT "${CMAKE_SOURCE_DIR}/Assembly.txt")
    
    add_custom_command(
        OUTPUT "${ASM_OUTPUT}"
        COMMAND "${CMAKE_CXX_COMPILER}" -S ${CMAKE_SOURCE_DIR}/Benchmark/main.cpp -o ${ASM_OUTPUT} 
        -I ${CMAKE_SOURCE_DIR}/Build/_deps/jsonifier-src/Include/
        -I ${CMAKE_SOURCE_DIR}/Include/  
        -std=c++2b
        -DOPERATING_SYSTEM_NAME="TEST_OS" 
        -DOPERATING_SYSTEM_VERSION="TEST_OS_VERSION" 
        -DCOMPILER_VERSION="TEST_COMPILER_VERSION" 
        -DCOMPILER_ID="TEST_ID" 
        -DBASE_PATH="${CMAKE_SOURCE_DIR}"
        -DSIMDJSON_COMMIT="SIMDJSON_COMMIT"
        -DGLAZE_COMMIT="GLAZE_COMMIT"
        -DJSONIFIER_COMMIT="JSONIFIER_COMMIT"
        "-O3"
        "-DNDEBUG"
        DEPENDS "${CMAKE_SOURCE_DIR}/Benchmark/main.cpp"
        COMMENT "Generating assembly for main.cpp"
        VERBATIM
    )

    add_custom_target(generate_asm ALL DEPENDS "${ASM_OUTPUT}")
    add_dependencies("BenchmarkExe" generate_asm)

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(ASM_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/Assembly")

    file(MAKE_DIRECTORY "${ASM_OUTPUT_DIR}")

    add_custom_command(
        OUTPUT "${ASM_OUTPUT_DIR}/main.asm"
        COMMAND "${CMAKE_CXX_COMPILER}" /Fa${ASM_OUTPUT_DIR}/ /FAsc /FoNUL /c 
        "${CMAKE_SOURCE_DIR}/Benchmark/main.cpp" 
        /I "${CMAKE_SOURCE_DIR}/Build/_deps/jsonifier-src/Include/"
        /I "${CMAKE_SOURCE_DIR}/Include/"
        /std:c++latest
        /DOPERATING_SYSTEM_NAME="TEST_OS"
        /DOPERATING_SYSTEM_VERSION="TEST_OS_VERSION"
        /DCOMPILER_VERSION="TEST_COMPILER_VERSION"
        /DCOMPILER_ID="TEST_ID"
        /DBASE_PATH="${CMAKE_SOURCE_DIR}"
        /DSIMDJSON_COMMIT="SIMDJSON_COMMIT"
        /DGLAZE_COMMIT="GLAZE_COMMIT"
        /DJSONIFIER_COMMIT="JSONIFIER_COMMIT"
        "${CMAKE_OPTIONS_NEW}"
        DEPENDS "${CMAKE_SOURCE_DIR}/Benchmark/main.cpp"
        COMMENT "Generating human-readable assembly for main.cpp"
        VERBATIM
    )

    add_custom_target(generate_asm ALL DEPENDS "${ASM_OUTPUT_DIR}/main.asm")

    add_dependencies("BenchmarkExe" generate_asm)
endif()


if (WIN32)
	install(
		FILES 
		"$<TARGET_PDB_FILE:BenchmarkExe>"
		DESTINATION "Tests/${CMAKE_BUILD_TYPE}"
		OPTIONAL
	)
endif()

install(
	FILES 
	"$<TARGET_FILE:BenchmarkExe>"
	DESTINATION "Tests/${CMAKE_BUILD_TYPE}"
	OPTIONAL
)

install(
	FILES 
	"$<TARGET_RUNTIME_DLLS:BenchmarkExe>"
	DESTINATION "Tests/${CMAKE_BUILD_TYPE}"
	OPTIONAL
)