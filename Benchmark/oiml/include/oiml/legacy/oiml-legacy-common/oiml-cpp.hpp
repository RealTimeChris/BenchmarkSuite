#pragma once

#ifndef __cplusplus
	#error "This header is for C++ only"
#endif

#include <oiml/legacy/oiml-legacy-common/oiml-final.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-alloc-final.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-backend.hpp>
#include <oiml/legacy/oiml-legacy-common/oigguf.hpp>
#include <memory>

// Smart pointers for oiml types

// oiml

struct oiml_context_deleter {
	void operator()(oiml_context* ctx) {
		oiml_free(ctx);
	}
};
struct oigguf_context_deleter {
	void operator()(oigguf_context* ctx) {
		oigguf_free(ctx);
	}
};

typedef std::unique_ptr<oiml_context, oiml_context_deleter> oiml_context_ptr;
typedef std::unique_ptr<oigguf_context, oigguf_context_deleter> oigguf_context_ptr;

// oiml-alloc

struct oiml_gallocr_deleter {
	void operator()(oiml_gallocr_t galloc) {
		oiml_gallocr_free(galloc);
	}
};

typedef std::unique_ptr<oiml_gallocr_t, oiml_gallocr_deleter> oiml_gallocr_ptr;

// oiml-backend

struct oiml_backend_deleter {
	void operator()(oiml_backend_t backend) {
		oiml_backend_free(backend);
	}
};
struct oiml_backend_buffer_deleter {
	void operator()(oiml_backend_buffer_t buffer) {
		oiml_backend_buffer_free(buffer);
	}
};
struct oiml_backend_event_deleter {
	void operator()(oiml_backend_event_t event) {
		oiml_backend_event_free(event);
	}
};
struct oiml_backend_sched_deleter {
	void operator()(oiml_backend_sched_t sched) {
		oiml_backend_sched_free(sched);
	}
};

typedef std::unique_ptr<oiml_backend, oiml_backend_deleter> oiml_backend_ptr;
typedef std::unique_ptr<oiml_backend_buffer, oiml_backend_buffer_deleter> oiml_backend_buffer_ptr;
typedef std::unique_ptr<oiml_backend_event, oiml_backend_event_deleter> oiml_backend_event_ptr;
typedef std::unique_ptr<oiml_backend_sched, oiml_backend_sched_deleter> oiml_backend_sched_ptr;
