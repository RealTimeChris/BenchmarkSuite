#	MIT License
#
#	Copyright (c) 2024 RealTimeChris
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

set(MAIN_SOURCE $<IF:$<BOOL:${BNCH_SWT_DEPLOY}>,main_deploy,main>${BNCH_SWT_MAIN_FILE_EXTENSION})

add_executable(benchmarksuite-main ${MAIN_SOURCE})

include(FetchContent)

if (BNCH_SWT_CUDA)
    set(GGML_CUDA TRUE)
    FetchContent_Declare(
        ggml
        GIT_REPOSITORY https://github.com/ggerganov/llama.cpp
        GIT_TAG master
    )
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "" FORCE)
    FetchContent_MakeAvailable(ggml)
endif()

target_link_libraries(benchmarksuite-main
    PUBLIC
        benchmarksuite::benchmarksuite
        $<$<TARGET_EXISTS:ggml>:ggml>
)

if(MSVC)
    set_property(TARGET benchmarksuite-main PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )
endif()

install(
    FILES
        $<TARGET_FILE:benchmarksuite-main>
    DESTINATION bin
    OPTIONAL
)
