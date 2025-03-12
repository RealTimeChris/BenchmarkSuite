#pragma once

// OIML CPU internal header

#include <oiml/legacy/oiml-legacy-common/oiml-final.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-impl.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-backend-impl-final.hpp>
#include <stdlib.h>// load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#include <stdbool.h>
#include <string.h>// memcpy
#include <math.h>// fabsf

struct oiml_compute_params {
	// ith = thread index, nth = number of threads
	int ith, nth;

	// work buffer for all threads
	size_t wsize;
	void* wdata;

	oiml_threadpool* threadpool;
};

