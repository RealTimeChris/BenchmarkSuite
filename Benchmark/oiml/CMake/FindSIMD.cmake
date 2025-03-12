include(CheckCXXSourceRuns)

set(AVX_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256 a;
        a = _mm256_set1_ps(0);
        return 0;
    }
")

set(AVX512_CODE "
    #include <immintrin.h>
    int main()
    {
        __m512i a = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0);
        __m512i b = a;
        __mmask64 equality_mask = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
        return 0;
    }
")

set(AVX2_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256i a = {0};
        a = _mm256_abs_epi16(a);
        __m256i x;
        _mm256_extract_epi64(x, 0); // we rely on this in our AVX2 code
        return 0;
    }
")

set(FMA_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256 acc = _mm256_setzero_ps();
        const __m256 d = _mm256_setzero_ps();
        const __m256 p = _mm256_setzero_ps();
        acc = _mm256_fmadd_ps( d, p, acc );
        return 0;
    }
")

set(SIMD_FLAGS "")

function(check_sse type flags)
    set(__FLAG_I 1)
    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    foreach (__FLAG ${flags})
        if (NOT ${type}_FOUND)
            set(CMAKE_REQUIRED_FLAGS ${__FLAG})
            check_cxx_source_runs("${${type}_CODE}" HAS_${type}_${__FLAG_I})
            if (HAS_${type}_${__FLAG_I})
                set(${type}_FOUND TRUE CACHE BOOL "${type} support")
                set(${type}_FLAGS "${__FLAG}" CACHE STRING "${type} flags")
                set(SIMD_FLAGS "${SIMD_FLAGS};${${type}_FLAGS}" PARENT_SCOPE)
            endif()
            math(EXPR __FLAG_I "${__FLAG_I}+1")
        endif()
    endforeach()
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
    set(SIMD_FLAGS "${SIMD_FLAGS}" PARENTS_SCOPE)
    if (NOT ${type}_FOUND)
        set(${type}_FOUND FALSE CACHE BOOL "${type} support")
        set(${type}_FLAGS "" CACHE STRING "${type} flags")
    endif()

    mark_as_advanced(${type}_FOUND ${type}_FLAGS)
endfunction()

if (WIN32)
    set(AVX_FLAG "/arch:AVX")
    set(AVX2_FLAG "/arch:AVX2")
    set(FMA_FLAG "/arch:AVX2")
    set(AVX512_FLAG "/arch:AVX512")
elseif(UNIX)
    set(AVX_FLAG "-mavx")
    set(AVX2_FLAG "-mavx2")
    set(FMA_FLAG "-mfma")
    set(AVX512_FLAG "-mavx512f")
else()
    set(AVX_FLAG "")
    set(AVX2_FLAG "")
    set(FMA_FLAG "")
    set(AVX512_FLAG "")
endif()

function(REMOVE_DUPLICATES input_list output_list)
    set(unique_list "")

    foreach(item IN LISTS ${input_list})
        list(FIND unique_list "${item}" index)
        if(index EQUAL -1)
            list(APPEND unique_list "${item}")
        endif()
    endforeach()
    set("${output_list}" "${unique_list}" PARENT_SCOPE)
endfunction()

check_sse("AVX" "${AVX_FLAG}")
if (NOT ${AVX_FOUND})
    set(OIML_AVX OFF)
else()
    set(OIML_AVX ON)
endif()

check_sse("AVX2" "${AVX2_FLAG}")
check_sse("FMA" "${FMA_FLAG}")
if ((NOT ${AVX2_FOUND}) OR (NOT ${FMA_FOUND}))
    set(OIML_AVX2 OFF)
else()
    set(OIML_AVX2 ON)
endif()

check_sse("AVX512" "${AVX512_FLAG}")
if (NOT ${AVX512_FOUND})
    set(OIML_AVX512 OFF)
else()
    set(OIML_AVX512 ON)
endif()

set(SIMD_FLAGS_NEW "SIMD_FLAGS_NEW")

remove_duplicates(SIMD_FLAGS SIMD_FLAGS_NEW)
set(AVX_FLAGS "${SIMD_FLAGS_NEW}" CACHE STRING "simd flags" FORCE)