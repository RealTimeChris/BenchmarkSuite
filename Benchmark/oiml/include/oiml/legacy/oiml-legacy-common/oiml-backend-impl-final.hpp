// Note: porting this file to C++ is a work in progress
#pragma once
#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include <windows.h>
#endif

#include <oiml/legacy/oiml-legacy-common/oiml-backend.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-backend-impl.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-alloc.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-impl.hpp>

#include <algorithm>
#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#ifdef __APPLE__
	#include <sys/types.h>
	#include <sys/sysctl.h>
#endif


// backend buffer type

OIML_INLINE const char* oiml_backend_buft_name(oiml_backend_buffer_type_t buft) {
	return buft->iface.get_name(buft);
}

OIML_INLINE oiml_backend_buffer_t oiml_backend_buft_alloc_buffer(oiml_backend_buffer_type_t buft, size_t size) {
	if (size == 0) {
		// return a dummy buffer for zero-sized allocations
		return oiml_backend_buffer_init(buft, {}, NULL, 0);
	}

	return buft->iface.alloc_buffer(buft, size);
}

OIML_INLINE size_t oiml_backend_buft_get_alignment(oiml_backend_buffer_type_t buft) {
	return buft->iface.get_alignment(buft);
}

OIML_INLINE size_t oiml_backend_buft_get_max_size(oiml_backend_buffer_type_t buft) {
	// get_max_size is optional, defaults to SIZE_MAX
	if (buft->iface.get_max_size) {
		return buft->iface.get_max_size(buft);
	}
	return SIZE_MAX;
}

OIML_INLINE size_t oiml_backend_buft_get_alloc_size(oiml_backend_buffer_type_t buft, struct oiml_tensor* tensor) {
	// get_alloc_size is optional, defaults to oiml_nbytes
	if (buft->iface.get_alloc_size) {
		size_t size = buft->iface.get_alloc_size(buft, tensor);
		assert(size >= oiml_nbytes(tensor));
		return size;
	}
	return oiml_nbytes(tensor);
}

OIML_INLINE bool oiml_backend_buft_is_host(oiml_backend_buffer_type_t buft) {
	if (buft->iface.is_host) {
		return buft->iface.is_host(buft);
	}
	return false;
}

OIML_INLINE oiml_backend_dev_t oiml_backend_buft_get_device(oiml_backend_buffer_type_t buft) {
	return buft->device;
}

// backend buffer

OIML_INLINE oiml_backend_buffer_t oiml_backend_buffer_init(oiml_backend_buffer_type_t buft, struct oiml_backend_buffer_i iface, void* context, size_t size) {
	oiml_backend_buffer_t buffer = new oiml_backend_buffer{ /* .interface = */ iface,
		/* .buft      = */ buft,
		/* .context   = */ context,
		/* .size      = */ size,
		/* .usage     = */ OIML_BACKEND_BUFFER_USAGE_ANY };

	return buffer;
}

OIML_INLINE const char* oiml_backend_buffer_name(oiml_backend_buffer_t buffer) {
	return oiml_backend_buft_name(oiml_backend_buffer_get_type(buffer));
}

OIML_INLINE void oiml_backend_buffer_free(oiml_backend_buffer_t buffer) {
	if (buffer == NULL) {
		return;
	}

	if (buffer->iface.free_buffer != NULL) {
		buffer->iface.free_buffer(buffer);
	}
	delete buffer;
}

OIML_INLINE size_t oiml_backend_buffer_get_size(oiml_backend_buffer_t buffer) {
	return buffer->size;
}

OIML_INLINE void* oiml_backend_buffer_get_base(oiml_backend_buffer_t buffer) {
	// get_base is optional if the buffer is zero-sized
	if (buffer->size == 0) {
		return NULL;
	}

	void* base = buffer->iface.get_base(buffer);

	OIML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

	return base;
}

OIML_INLINE void oiml_backend_buffer_init_tensor(oiml_backend_buffer_t buffer, struct oiml_tensor* tensor) {
	// init_tensor is optional
	if (buffer->iface.init_tensor) {
		buffer->iface.init_tensor(buffer, tensor);
	}
}

OIML_INLINE void oiml_backend_buffer_clear(oiml_backend_buffer_t buffer, uint8_t value) {
	// clear is optional if the buffer is zero-sized
	if (buffer->size == 0) {
		return;
	}

	buffer->iface.clear(buffer, value);
}

OIML_INLINE size_t oiml_backend_buffer_get_alignment(oiml_backend_buffer_t buffer) {
	return oiml_backend_buft_get_alignment(oiml_backend_buffer_get_type(buffer));
}

OIML_INLINE size_t oiml_backend_buffer_get_max_size(oiml_backend_buffer_t buffer) {
	return oiml_backend_buft_get_max_size(oiml_backend_buffer_get_type(buffer));
}

OIML_INLINE size_t oiml_backend_buffer_get_alloc_size(oiml_backend_buffer_t buffer, struct oiml_tensor* tensor) {
	return oiml_backend_buft_get_alloc_size(oiml_backend_buffer_get_type(buffer), tensor);
}

OIML_INLINE bool oiml_backend_buffer_is_host(oiml_backend_buffer_t buffer) {
	return oiml_backend_buft_is_host(oiml_backend_buffer_get_type(buffer));
}

OIML_INLINE void oiml_backend_buffer_set_usage(oiml_backend_buffer_t buffer, enum oiml_backend_buffer_usage usage) {
	buffer->usage = usage;

	// FIXME: add a generic callback to the buffer interface
	if (oiml_backend_buffer_is_multi_buffer(buffer)) {
		oiml_backend_multi_buffer_set_usage(buffer, usage);
	}
}

OIML_INLINE enum oiml_backend_buffer_usage oiml_backend_buffer_get_usage(oiml_backend_buffer_t buffer) {
	return buffer->usage;
}

OIML_INLINE oiml_backend_buffer_type_t oiml_backend_buffer_get_type(oiml_backend_buffer_t buffer) {
	return buffer->buft;
}

OIML_INLINE void oiml_backend_buffer_reset(oiml_backend_buffer_t buffer) {
	if (buffer->iface.reset) {
		buffer->iface.reset(buffer);
	}
}

OIML_INLINE bool oiml_backend_buffer_copy_tensor(const struct oiml_tensor* src, struct oiml_tensor* dst) {
	oiml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
	if (dst_buf->iface.cpy_tensor) {
		return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
	}
	return false;
}

// backend

OIML_INLINE oiml_guid_t oiml_backend_guid(oiml_backend_t backend) {
	if (backend == NULL) {
		return NULL;
	}
	return backend->guid;
}

OIML_INLINE const char* oiml_backend_name(oiml_backend_t backend) {
	if (backend == NULL) {
		return "NULL";
	}
	return backend->iface.get_name(backend);
}

OIML_INLINE void oiml_backend_free(oiml_backend_t backend) {
	if (backend == NULL) {
		return;
	}

	backend->iface.free(backend);
}

OIML_INLINE oiml_backend_buffer_type_t oiml_backend_get_default_buffer_type(oiml_backend_t backend) {
	return oiml_backend_dev_buffer_type(backend->device);
}

OIML_INLINE oiml_backend_buffer_t oiml_backend_alloc_buffer(oiml_backend_t backend, size_t size) {
	return oiml_backend_buft_alloc_buffer(oiml_backend_get_default_buffer_type(backend), size);
}

OIML_INLINE size_t oiml_backend_get_alignment(oiml_backend_t backend) {
	return oiml_backend_buft_get_alignment(oiml_backend_get_default_buffer_type(backend));
}

OIML_INLINE size_t oiml_backend_get_max_size(oiml_backend_t backend) {
	return oiml_backend_buft_get_max_size(oiml_backend_get_default_buffer_type(backend));
}

OIML_INLINE void oiml_backend_tensor_set_async(oiml_backend_t backend, struct oiml_tensor* tensor, const void* data, size_t offset, size_t size) {
	OIML_ASSERT(tensor->data != NULL && "tensor not allocated");
	OIML_ASSERT(offset + size <= oiml_nbytes(tensor) && "tensor write out of bounds");

	if (backend->iface.set_tensor_async == NULL) {
		oiml_backend_tensor_set(tensor, data, offset, size);
	} else {
		backend->iface.set_tensor_async(backend, tensor, data, offset, size);
	}
}

OIML_INLINE void oiml_backend_tensor_get_async(oiml_backend_t backend, const struct oiml_tensor* tensor, void* data, size_t offset, size_t size) {
	OIML_ASSERT(tensor->data != NULL && "tensor not allocated");
	OIML_ASSERT(offset + size <= oiml_nbytes(tensor) && "tensor read out of bounds");

	if (backend->iface.get_tensor_async == NULL) {
		oiml_backend_tensor_get(tensor, data, offset, size);
	} else {
		backend->iface.get_tensor_async(backend, tensor, data, offset, size);
	}
}

OIML_INLINE void oiml_backend_tensor_set(struct oiml_tensor* tensor, const void* data, size_t offset, size_t size) {
	OIML_ASSERT(tensor);
	oiml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

	if (size == 0) {
		return;
	}

	OIML_ASSERT(buf != NULL && "tensor buffer not set");
	OIML_ASSERT(tensor->data != NULL && "tensor not allocated");
	OIML_ASSERT(offset + size <= oiml_nbytes(tensor) && "tensor write out of bounds");

	buf->iface.set_tensor(buf, tensor, data, offset, size);
}

OIML_INLINE void oiml_backend_tensor_get(const struct oiml_tensor* tensor, void* data, size_t offset, size_t size) {
	OIML_ASSERT(tensor);
	oiml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

	if (size == 0) {
		return;
	}

	OIML_ASSERT(buf != NULL && "tensor buffer not set");
	OIML_ASSERT(tensor->data != NULL && "tensor not allocated");
	OIML_ASSERT(offset + size <= oiml_nbytes(tensor) && "tensor read out of bounds");

	buf->iface.get_tensor(buf, tensor, data, offset, size);
}

OIML_INLINE void oiml_backend_tensor_memset(struct oiml_tensor* tensor, uint8_t value, size_t offset, size_t size) {
	oiml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

	if (size == 0) {
		return;
	}

	OIML_ASSERT(buf != NULL && "tensor buffer not set");
	OIML_ASSERT(tensor->data != NULL && "tensor not allocated");
	OIML_ASSERT(offset + size <= oiml_nbytes(tensor) && "tensor write out of bounds");
	OIML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

	buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

OIML_INLINE void oiml_backend_synchronize(oiml_backend_t backend) {
	if (backend->iface.synchronize == NULL) {
		return;
	}

	backend->iface.synchronize(backend);
}

OIML_INLINE oiml_backend_graph_plan_t oiml_backend_graph_plan_create(oiml_backend_t backend, struct oiml_cgraph* cgraph) {
	OIML_ASSERT(backend->iface.graph_plan_create != NULL);

	return backend->iface.graph_plan_create(backend, cgraph);
}

OIML_INLINE void oiml_backend_graph_plan_free(oiml_backend_t backend, oiml_backend_graph_plan_t plan) {
	OIML_ASSERT(backend->iface.graph_plan_free != NULL);

	backend->iface.graph_plan_free(backend, plan);
}

OIML_INLINE enum oiml_status oiml_backend_graph_plan_compute(oiml_backend_t backend, oiml_backend_graph_plan_t plan) {
	OIML_ASSERT(backend->iface.graph_plan_compute != NULL);

	return backend->iface.graph_plan_compute(backend, plan);
}

OIML_INLINE enum oiml_status oiml_backend_graph_compute(oiml_backend_t backend, struct oiml_cgraph* cgraph) {
	enum oiml_status err = oiml_backend_graph_compute_async(backend, cgraph);
	oiml_backend_synchronize(backend);
	return err;
}

OIML_INLINE enum oiml_status oiml_backend_graph_compute_async(oiml_backend_t backend, struct oiml_cgraph* cgraph) {
	return backend->iface.graph_compute(backend, cgraph);
}

OIML_INLINE bool oiml_backend_supports_op(oiml_backend_t backend, const struct oiml_tensor* op) {
	return oiml_backend_dev_supports_op(backend->device, op);
}

OIML_INLINE bool oiml_backend_supports_buft(oiml_backend_t backend, oiml_backend_buffer_type_t buft) {
	return oiml_backend_dev_supports_buft(backend->device, buft);
}

OIML_INLINE bool oiml_backend_offload_op(oiml_backend_t backend, const struct oiml_tensor* op) {
	return oiml_backend_dev_offload_op(backend->device, op);
}

OIML_INLINE oiml_backend_dev_t oiml_backend_get_device(oiml_backend_t backend) {
	return backend->device;
}

// backend copy

OIML_INLINE void oiml_backend_tensor_copy(struct oiml_tensor* src, struct oiml_tensor* dst) {
	OIML_ASSERT(oiml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

	if (src == dst) {
		return;
	}

	if (oiml_backend_buffer_is_host(src->buffer)) {
		oiml_backend_tensor_set(dst, src->data, 0, oiml_nbytes(src));
	} else if (oiml_backend_buffer_is_host(dst->buffer)) {
		oiml_backend_tensor_get(src, dst->data, 0, oiml_nbytes(src));
	} else if (!oiml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
		OIML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, oiml_backend_buffer_name(src->buffer), oiml_backend_buffer_name(dst->buffer));
#endif
		size_t nbytes = oiml_nbytes(src);
		void* data	  = malloc(nbytes);
		oiml_backend_tensor_get(src, data, 0, nbytes);
		oiml_backend_tensor_set(dst, data, 0, nbytes);
		free(data);
	}
}

OIML_INLINE void oiml_backend_tensor_copy_async(oiml_backend_t backend_src, oiml_backend_t backend_dst, struct oiml_tensor* src, struct oiml_tensor* dst) {
	OIML_ASSERT(oiml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

	if (src == dst) {
		return;
	}

	if (backend_dst->iface.cpy_tensor_async != NULL) {
		if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
			return;
		}
	}

	// an async copy would normally happen after all the queued operations on both backends are completed
	// to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
	oiml_backend_synchronize(backend_src);
	oiml_backend_synchronize(backend_dst);
	oiml_backend_tensor_copy(src, dst);
}

// events

OIML_INLINE oiml_backend_event_t oiml_backend_event_new(oiml_backend_dev_t device) {
	// null device is allowed for the transition period to the device interface
	if (device == NULL || device->iface.event_new == NULL) {
		return NULL;
	}
	return device->iface.event_new(device);
}

OIML_INLINE void oiml_backend_event_free(oiml_backend_event_t event) {
	if (event == NULL) {
		return;
	}
	event->device->iface.event_free(event->device, event);
}

OIML_INLINE void oiml_backend_event_record(oiml_backend_event_t event, oiml_backend_t backend) {
	OIML_ASSERT(backend->iface.event_record != NULL);

	backend->iface.event_record(backend, event);
}

OIML_INLINE void oiml_backend_event_synchronize(oiml_backend_event_t event) {
	OIML_ASSERT(event->device->iface.event_synchronize);

	event->device->iface.event_synchronize(event->device, event);
}

OIML_INLINE void oiml_backend_event_wait(oiml_backend_t backend, oiml_backend_event_t event) {
	OIML_ASSERT(backend->iface.event_wait != NULL);

	backend->iface.event_wait(backend, event);
}

// Backend device

OIML_INLINE const char* oiml_backend_dev_name(oiml_backend_dev_t device) {
	return device->iface.get_name(device);
}

OIML_INLINE const char* oiml_backend_dev_description(oiml_backend_dev_t device) {
	return device->iface.get_description(device);
}

OIML_INLINE void oiml_backend_dev_memory(oiml_backend_dev_t device, size_t* free, size_t* total) {
	device->iface.get_memory(device, free, total);
}

OIML_INLINE enum oiml_backend_device_types oiml_backend_device_type(oiml_backend_dev_t device) {
	return device->iface.get_type(device);
}

OIML_INLINE void oiml_backend_dev_get_props(oiml_backend_dev_t device, struct oiml_backend_dev_props* props) {
	memset(props, 0, sizeof(*props));
	device->iface.get_props(device, props);
}

OIML_INLINE oiml_backend_reg_t oiml_backend_dev_backend_reg(oiml_backend_dev_t device) {
	return device->reg;
}

OIML_INLINE oiml_backend_t oiml_backend_dev_init(oiml_backend_dev_t device, const char* params) {
	return device->iface.init_backend(device, params);
}

OIML_INLINE oiml_backend_buffer_type_t oiml_backend_dev_buffer_type(oiml_backend_dev_t device) {
	return device->iface.get_buffer_type(device);
}

OIML_INLINE oiml_backend_buffer_type_t oiml_backend_dev_host_buffer_type(oiml_backend_dev_t device) {
	if (device->iface.get_host_buffer_type == NULL) {
		return NULL;
	}

	return device->iface.get_host_buffer_type(device);
}

OIML_INLINE oiml_backend_buffer_t oiml_backend_dev_buffer_from_host_ptr(oiml_backend_dev_t device, void* ptr, size_t size, size_t max_tensor_size) {
	return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

OIML_INLINE bool oiml_backend_dev_supports_op(oiml_backend_dev_t device, const struct oiml_tensor* op) {
	return device->iface.supports_op(device, op);
}

OIML_INLINE bool oiml_backend_dev_supports_buft(oiml_backend_dev_t device, oiml_backend_buffer_type_t buft) {
	return device->iface.supports_buft(device, buft);
}

OIML_INLINE bool oiml_backend_dev_offload_op(oiml_backend_dev_t device, const struct oiml_tensor* op) {
	if (device->iface.offload_op != NULL) {
		return device->iface.offload_op(device, op);
	}

	return false;
}

// Backend (reg)

OIML_INLINE const char* oiml_backend_reg_name(oiml_backend_reg_t reg) {
	return reg->iface.get_name(reg);
}

OIML_INLINE size_t oiml_backend_reg_dev_count(oiml_backend_reg_t reg) {
	return reg->iface.get_device_count(reg);
}

OIML_INLINE oiml_backend_dev_t oiml_backend_reg_dev_get(oiml_backend_reg_t reg, size_t index) {
	return reg->iface.get_device(reg, index);
}

OIML_INLINE void* oiml_backend_reg_get_proc_address(oiml_backend_reg_t reg, const char* name) {
	if (!reg->iface.get_proc_address) {
		return NULL;
	}
	return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct oiml_backend_multi_buffer_context {
	oiml_backend_buffer_t* buffers;
	size_t n_buffers;
};

OIML_INLINE static void oiml_backend_multi_buffer_free_buffer(oiml_backend_buffer_t buffer) {
	oiml_backend_multi_buffer_context* ctx = ( oiml_backend_multi_buffer_context* )buffer->context;
	for (size_t i = 0; i < ctx->n_buffers; i++) {
		oiml_backend_buffer_free(ctx->buffers[i]);
	}

	free(ctx->buffers);
	free(ctx);
}

OIML_INLINE static void oiml_backend_multi_buffer_clear(oiml_backend_buffer_t buffer, uint8_t value) {
	oiml_backend_multi_buffer_context* ctx = ( oiml_backend_multi_buffer_context* )buffer->context;
	for (size_t i = 0; i < ctx->n_buffers; i++) {
		oiml_backend_buffer_clear(ctx->buffers[i], value);
	}
}

static const struct oiml_backend_buffer_i oiml_backend_multi_buffer_i = {
	/* .free_buffer     = */ oiml_backend_multi_buffer_free_buffer,
	/* .get_base        = */ NULL,
	/* .init_tensor     = */ NULL,
	/* .memset_tensor   = */ NULL,
	/* .set_tensor      = */ NULL,
	/* .get_tensor      = */ NULL,
	/* .cpy_tensor      = */ NULL,
	/* .clear           = */ oiml_backend_multi_buffer_clear,
	/* .reset           = */ NULL,
};

OIML_INLINE oiml_backend_buffer_t oiml_backend_multi_buffer_alloc_buffer(oiml_backend_buffer_t* buffers, size_t n_buffers) {
	oiml_backend_multi_buffer_context* ctx = ( oiml_backend_multi_buffer_context* )malloc(sizeof(struct oiml_backend_multi_buffer_context));
	ctx->n_buffers						   = n_buffers;
	ctx->buffers						   = ( oiml_backend_buffer_t* )malloc(n_buffers * sizeof(oiml_backend_buffer_t));

	OIML_ASSERT(ctx->buffers != NULL);

	size_t total_size = 0;
	for (size_t i = 0; i < n_buffers; i++) {
		ctx->buffers[i] = buffers[i];
		total_size += oiml_backend_buffer_get_size(buffers[i]);
	}

	return oiml_backend_buffer_init(buffers[0]->buft, oiml_backend_multi_buffer_i, ctx, total_size);
}

OIML_INLINE bool oiml_backend_buffer_is_multi_buffer(oiml_backend_buffer_t buffer) {
	return buffer->iface.free_buffer == oiml_backend_multi_buffer_free_buffer;
}

OIML_INLINE void oiml_backend_multi_buffer_set_usage(oiml_backend_buffer_t buffer, enum oiml_backend_buffer_usage usage) {
	OIML_ASSERT(oiml_backend_buffer_is_multi_buffer(buffer));
	oiml_backend_multi_buffer_context* ctx = ( oiml_backend_multi_buffer_context* )buffer->context;
	for (size_t i = 0; i < ctx->n_buffers; i++) {
		oiml_backend_buffer_set_usage(ctx->buffers[i], usage);
	}
}

// creates a copy of the tensor with the same memory layout
OIML_INLINE static struct oiml_tensor* oiml_dup_tensor_layout(struct oiml_context* ctx, const struct oiml_tensor* tensor) {
	struct oiml_tensor* dup = oiml_dup_tensor(ctx, tensor);
	for (int i = 0; i < OIML_MAX_DIMS; i++) {
		dup->nb[i] = tensor->nb[i];
	}
	return dup;
}

OIML_INLINE static bool oiml_is_view_op(enum oiml_op op) {
	return op == OIML_OP_VIEW || op == OIML_OP_RESHAPE || op == OIML_OP_PERMUTE || op == OIML_OP_TRANSPOSE;
}

// scheduler

#ifndef OIML_SCHED_MAX_BACKENDS
	#define OIML_SCHED_MAX_BACKENDS 16
#endif

#ifndef OIML_SCHED_MAX_SPLIT_INPUTS
	#define OIML_SCHED_MAX_SPLIT_INPUTS OIML_MAX_SRC
#endif

#ifndef OIML_SCHED_MAX_COPIES
	#define OIML_SCHED_MAX_COPIES 4
#endif

struct oiml_backend_sched_split {
	int backend_id;
	int i_start;
	int i_end;
	struct oiml_tensor* inputs[OIML_SCHED_MAX_SPLIT_INPUTS];
	int n_inputs;
	// graph view of this split
	struct oiml_cgraph graph;
};

struct oiml_backend_sched {
	bool is_reset;// true if the scheduler has been reset since the last graph split
	bool is_alloc;

	int n_backends;

	oiml_backend_t backends[OIML_SCHED_MAX_BACKENDS];
	oiml_backend_buffer_type_t bufts[OIML_SCHED_MAX_BACKENDS];
	oiml_gallocr_t galloc;

	// hash map of the nodes in the graph
	struct oiml_hash_set hash_set;
	int* hv_tensor_backend_ids;// [hash_set.size]
	struct oiml_tensor** hv_tensor_copies;// [hash_set.size][n_backends][n_copies]

	int* node_backend_ids;// [graph_size]
	int* leaf_backend_ids;// [graph_size]

	int* prev_node_backend_ids;// [graph_size]
	int* prev_leaf_backend_ids;// [graph_size]

	// copy of the graph with modified inputs
	struct oiml_cgraph graph;

	// graph splits
	struct oiml_backend_sched_split* splits;
	int n_splits;
	int splits_capacity;

	// pipeline parallelism support
	int n_copies;
	int cur_copy;
	oiml_backend_event_t events[OIML_SCHED_MAX_BACKENDS][OIML_SCHED_MAX_COPIES];
	struct oiml_tensor* graph_inputs[OIML_SCHED_MAX_SPLIT_INPUTS];
	int n_graph_inputs;

	struct oiml_context* ctx;

	oiml_backend_sched_eval_callback callback_eval;
	void* callback_eval_user_data;

	char* context_buffer;
	size_t context_buffer_size;

	int debug;
};

#define hash_id(tensor) oiml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

// returns the priority of the backend, lower id is higher priority
OIML_INLINE static int oiml_backend_sched_backend_id(oiml_backend_sched_t sched, oiml_backend_t backend) {
	for (int i = 0; i < sched->n_backends; i++) {
		if (sched->backends[i] == backend) {
			return i;
		}
	}
	return -1;
}

OIML_INLINE static int oiml_backend_sched_backend_from_buffer(oiml_backend_sched_t sched, const struct oiml_tensor* tensor, const struct oiml_tensor* op) {
	oiml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
	if (buffer == NULL) {
		return -1;
	}

	// find highest prio backend that supports the buffer type and the op
	for (int i = 0; i < sched->n_backends; i++) {
		if (oiml_backend_supports_buft(sched->backends[i], buffer->buft) && oiml_backend_supports_op(sched->backends[i], op)) {
			return i;
		}
	}

#ifndef NDEBUG
	OIML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n", __func__,
		oiml_op_desc(tensor), oiml_backend_buffer_name(buffer), tensor->name);
#endif

	return -1;
}

#if 0
	#define OIML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[OIML_DEFAULT_GRAPH_SIZE*16 + OIML_SCHED_MAX_SPLITS_DEBUG*OIML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
	#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
	#define GET_CAUSE(node) causes[hash_id(node)]
#else
	#define SET_CAUSE(node, ...)
	#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
OIML_INLINE static int oiml_backend_sched_backend_id_from_cur(oiml_backend_sched_t sched, struct oiml_tensor* tensor) {
	// assign pre-allocated nodes to their backend
	int cur_backend_id = oiml_backend_sched_backend_from_buffer(sched, tensor, tensor);
	if (cur_backend_id != -1) {
		SET_CAUSE(tensor, "1.dst");
		return cur_backend_id;
	}

	// view_src
	if (tensor->view_src != NULL) {
		cur_backend_id = oiml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
		if (cur_backend_id != -1) {
			SET_CAUSE(tensor, "1.vsrc");
			return cur_backend_id;
		}
	}

	if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
		// since the tensor is pre-allocated, it cannot be moved to another backend
		oiml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
		OIML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, oiml_backend_buffer_name(buffer), oiml_op_name(tensor->op));
	}

	// graph input
	if (tensor->flags & OIML_TENSOR_FLAG_INPUT) {
		cur_backend_id = sched->n_backends - 1;// last backend (assumed CPU)
		SET_CAUSE(tensor, "1.inp");
		return cur_backend_id;
	}

	// operations with weights are preferably run on the same backend as the weights
	for (int i = 0; i < OIML_MAX_SRC; i++) {
		const struct oiml_tensor* src = tensor->src[i];
		if (src == NULL) {
			continue;
		}
		// skip ROPE since the rope freqs tensor is too small to choose a backend based on it
		// not an ideal solution
		if (tensor->op != OIML_OP_ROPE && src->buffer != NULL && src->buffer->usage == OIML_BACKEND_BUFFER_USAGE_WEIGHTS) {
			int src_backend_id = oiml_backend_sched_backend_from_buffer(sched, src, tensor);
			// check if a backend with higher prio wants to offload the op
			if (src_backend_id == sched->n_backends - 1 && oiml_backend_buffer_is_host(src->buffer)) {
				for (int b = 0; b < src_backend_id; b++) {
					if (oiml_backend_supports_op(sched->backends[b], tensor) && oiml_backend_offload_op(sched->backends[b], tensor)) {
						SET_CAUSE(tensor, "1.off");
						return b;
					}
				}
			}
			SET_CAUSE(tensor, "1.wgt%d", i);
			return src_backend_id;
		}
	}

	return -1;
}

OIML_INLINE static char* fmt_size(size_t size) {
	static char buffer[128];
	if (size >= 1024 * 1024) {
		snprintf(buffer, sizeof(buffer), "%zuM", size / 1024 / 1024);
	} else {
		snprintf(buffer, sizeof(buffer), "%zuK", size / 1024);
	}
	return buffer;
}

OIML_INLINE static void oiml_backend_sched_print_assignments(oiml_backend_sched_t sched, struct oiml_cgraph* graph) {
	int cur_split = 0;
	for (int i = 0; i < graph->n_nodes; i++) {
		if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
			oiml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
			OIML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, oiml_backend_name(split_backend), sched->splits[cur_split].n_inputs);
			for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
				if (j == 0) {
					OIML_LOG_DEBUG(": ");
				}
				OIML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name, fmt_size(oiml_nbytes(sched->splits[cur_split].inputs[j])));
			}
			OIML_LOG_DEBUG("\n");
			cur_split++;
		}
		struct oiml_tensor* node = graph->nodes[i];
		if (oiml_is_view_op(node->op)) {
			continue;
		}
		if (sched->debug > 1) {
			oiml_backend_t tensor_backend = oiml_backend_sched_get_tensor_backend(sched, node);
			OIML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s]:", i, oiml_op_name(node->op), node->name, fmt_size(oiml_nbytes(node)),
				tensor_backend ? oiml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node));
			for (int j = 0; j < OIML_MAX_SRC; j++) {
				struct oiml_tensor* src = node->src[j];
				if (src == NULL) {
					continue;
				}
				oiml_backend_t src_backend = oiml_backend_sched_get_tensor_backend(sched, src);
				OIML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name, fmt_size(oiml_nbytes(src)), src_backend ? oiml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
			}
			OIML_LOG_DEBUG("\n");
		}
	}
}

OIML_INLINE static bool oiml_backend_sched_buffer_supported(oiml_backend_sched_t sched, struct oiml_tensor* t, int backend_id) {
	oiml_backend_buffer_t buf		= t->view_src ? t->view_src->buffer : t->buffer;
	oiml_backend_buffer_type_t buft = NULL;

	if (buf) {
		// the tensor is already allocated
		buft = buf->buft;
	} else {
		// see if the tensor already has a backend assigned, and use the buffer type of that backend
		int tensor_backend_id = tensor_backend_id(t);
		if (tensor_backend_id == -1 && t->view_src) {
			tensor_backend_id = tensor_backend_id(t->view_src);
		}
		if (tensor_backend_id != -1) {
			buft = sched->bufts[tensor_backend_id];
		}
	}

	return buft != NULL && oiml_backend_supports_buft(sched->backends[backend_id], buft);
}

OIML_INLINE static void oiml_backend_sched_set_if_supported(oiml_backend_sched_t sched, struct oiml_tensor* node, int cur_backend_id, int* node_backend_id) {
	if (oiml_backend_supports_op(sched->backends[cur_backend_id], node)) {
		*node_backend_id = cur_backend_id;
		SET_CAUSE(node, "2.sup");
	}
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
OIML_INLINE static void oiml_backend_sched_split_graph(oiml_backend_sched_t sched, struct oiml_cgraph* graph) {
	// reset splits
	sched->n_splits		  = 0;
	sched->n_graph_inputs = 0;
	sched->is_reset		  = false;

	struct oiml_init_params params = { /* .mem_size =   */ sched->context_buffer_size,
		/* .mem_buffer = */ sched->context_buffer,
		/* .no_alloc =   */ true };

	oiml_free(sched->ctx);

	sched->ctx = oiml_init(params);
	if (sched->ctx == NULL) {
		OIML_ABORT("%s: failed to initialize context\n", __func__);
	}

	// pass 1: assign backends to ops with pre-allocated inputs
	for (int i = 0; i < graph->n_leafs; i++) {
		struct oiml_tensor* leaf = graph->leafs[i];
		int* leaf_backend_id	 = &tensor_backend_id(leaf);
		// do not overwrite user assignments
		if (*leaf_backend_id == -1) {
			*leaf_backend_id = oiml_backend_sched_backend_id_from_cur(sched, leaf);
		}
	}

	for (int i = 0; i < graph->n_nodes; i++) {
		struct oiml_tensor* node = graph->nodes[i];
		int* node_backend_id	 = &tensor_backend_id(node);
		// do not overwrite user assignments
		if (*node_backend_id == -1) {
			*node_backend_id = oiml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == OIML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < OIML_MAX_SRC; j++) {
                struct oiml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = oiml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
		}
	}

	// pass 2: expand current backend assignments
	// assign the same backend to adjacent nodes
	// expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
	// thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
	// ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
	// expand gpu down
	{
		int cur_backend_id = -1;
		for (int i = 0; i < graph->n_nodes; i++) {
			struct oiml_tensor* node = graph->nodes[i];
			if (oiml_is_view_op(node->op)) {
				continue;
			}
			int* node_backend_id = &tensor_backend_id(node);
			if (*node_backend_id != -1) {
				if (*node_backend_id == sched->n_backends - 1) {
					// skip cpu (lowest prio backend)
					cur_backend_id = -1;
				} else {
					cur_backend_id = *node_backend_id;
				}
			} else if (cur_backend_id != -1) {
				oiml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
			}
		}
	}
	// expand gpu up
	{
		int cur_backend_id = -1;
		for (int i = graph->n_nodes - 1; i >= 0; i--) {
			struct oiml_tensor* node = graph->nodes[i];
			if (oiml_is_view_op(node->op)) {
				continue;
			}
			int* node_backend_id = &tensor_backend_id(node);
			if (*node_backend_id != -1) {
				if (*node_backend_id == sched->n_backends - 1) {
					// skip cpu (lowest prio backend)
					cur_backend_id = -1;
				} else {
					cur_backend_id = *node_backend_id;
				}
			} else if (cur_backend_id != -1) {
				oiml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
			}
		}
	}
	// expand rest down
	{
		int cur_backend_id = -1;
		for (int i = 0; i < graph->n_nodes; i++) {
			struct oiml_tensor* node = graph->nodes[i];
			if (oiml_is_view_op(node->op)) {
				continue;
			}
			int* node_backend_id = &tensor_backend_id(node);
			if (*node_backend_id != -1) {
				cur_backend_id = *node_backend_id;
			} else if (cur_backend_id != -1) {
				oiml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
			}
		}
	}
	// expand rest up
	{
		int cur_backend_id = -1;
		for (int i = graph->n_nodes - 1; i >= 0; i--) {
			struct oiml_tensor* node = graph->nodes[i];
			if (oiml_is_view_op(node->op)) {
				continue;
			}
			int* node_backend_id = &tensor_backend_id(node);
			if (*node_backend_id != -1) {
				cur_backend_id = *node_backend_id;
			} else if (cur_backend_id != -1) {
				oiml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
			}
		}
	}

	// pass 3: upgrade nodes to higher prio backends with compatible buffer types
	// if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
	// however, we also need to verify that the sources are in compatible buffer types
	// (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
	// however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
	// this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
	// additionally, set remaining unassigned nodes to the backend with the most supported inputs
	// only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
	for (int i = 0; i < graph->n_nodes; i++) {
		struct oiml_tensor* node = graph->nodes[i];
		if (oiml_is_view_op(node->op)) {
			continue;
		}
		int* node_backend_id = &tensor_backend_id(node);
		if (*node_backend_id == -1) {
			// unassigned node: find the backend with the most supported inputs
			int n_supported_best = -1;
			for (int b = 0; b < sched->n_backends; b++) {
				if (oiml_backend_supports_op(sched->backends[b], node)) {
					int n_supported = 0;
					for (int j = 0; j < OIML_MAX_SRC; j++) {
						struct oiml_tensor* src = node->src[j];
						if (src == NULL) {
							continue;
						}
						if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && oiml_backend_sched_buffer_supported(sched, src, b)) {
							n_supported++;
						}
					}
					if (n_supported > n_supported_best) {
						n_supported_best = n_supported;
						*node_backend_id = b;
						SET_CAUSE(node, "3.best");
					}
				}
			}
		} else {
			// assigned node: upgrade to higher prio backend if possible
			for (int b = 0; b < *node_backend_id; b++) {
				if (sched->bufts[b] == sched->bufts[*node_backend_id] && oiml_backend_supports_op(sched->backends[b], node)) {
					bool supported = true;
					for (int j = 0; j < OIML_MAX_SRC; j++) {
						struct oiml_tensor* src = node->src[j];
						if (src == NULL) {
							continue;
						}
						if (!oiml_backend_sched_buffer_supported(sched, src, b)) {
							supported = false;
							break;
						}
					}
					if (supported) {
						*node_backend_id = b;
						SET_CAUSE(node, "3.upg");
						break;
					}
				}
			}
		}
	}

	// pass 4: assign backends to remaining src from dst and view_src
	for (int i = 0; i < graph->n_nodes; i++) {
		struct oiml_tensor* node = graph->nodes[i];
		int* cur_backend_id		 = &tensor_backend_id(node);
		if (node->view_src != NULL && *cur_backend_id == -1) {
			*cur_backend_id = tensor_backend_id(node->view_src);
			SET_CAUSE(node, "4.vsrc");
		}
		for (int j = 0; j < OIML_MAX_SRC; j++) {
			struct oiml_tensor* src = node->src[j];
			if (src == NULL) {
				continue;
			}
			int* src_backend_id = &tensor_backend_id(src);
			if (*src_backend_id == -1) {
				if (src->view_src != NULL) {
					// views are always on the same backend as the source
					*src_backend_id = tensor_backend_id(src->view_src);
					SET_CAUSE(src, "4.vsrc");
				} else {
					*src_backend_id = *cur_backend_id;
					SET_CAUSE(src, "4.cur");
				}
			}
		}
	}

	// pass 5: split graph, find tensors that need to be copied
	{
		int i_split							   = 0;
		struct oiml_backend_sched_split* split = &sched->splits[0];
		// find the backend of the first split, skipping view ops
		int i = 0;
		for (; i < graph->n_nodes; i++) {
			struct oiml_tensor* node = graph->nodes[i];
			if (!oiml_is_view_op(node->op)) {
				split->backend_id = tensor_backend_id(node);
				break;
			}
		}
		split->i_start	   = 0;
		split->n_inputs	   = 0;
		int cur_backend_id = split->backend_id;
		for (; i < graph->n_nodes; i++) {
			struct oiml_tensor* node = graph->nodes[i];

			if (oiml_is_view_op(node->op)) {
				continue;
			}

			const int node_backend_id = tensor_backend_id(node);

			assert(node_backend_id != -1);// all nodes should be assigned by now

			// check if we should start a new split based on the sources of the current node
			bool need_new_split = false;
			if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
				for (int j = 0; j < OIML_MAX_SRC; j++) {
					struct oiml_tensor* src = node->src[j];
					if (src == NULL) {
						continue;
					}
					// check if a weight is on a different and incompatible backend
					// by starting a new split, the memory of the previously offloaded weights can be reused
					if (src->buffer != NULL && src->buffer->usage == OIML_BACKEND_BUFFER_USAGE_WEIGHTS) {
						int src_backend_id = tensor_backend_id(src);
						if (src_backend_id != cur_backend_id && !oiml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
							need_new_split = true;
							break;
						}
					}
					// check if the split has too many inputs
					// FIXME: count the number of inputs instead of only checking when full
					if (split->n_inputs == OIML_SCHED_MAX_SPLIT_INPUTS) {
						const size_t id	   = hash_id(src);
						int src_backend_id = sched->hv_tensor_backend_ids[id];
						bool supported	   = oiml_backend_sched_buffer_supported(sched, src, cur_backend_id);
						if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
							need_new_split = true;
							break;
						}
					}
				}
			}

			if (node_backend_id != cur_backend_id || need_new_split) {
				split->i_end = i;
				i_split++;
				if (i_split >= sched->splits_capacity) {
					sched->splits_capacity *= 2;
					sched->splits = ( oiml_backend_sched_split* )realloc(sched->splits, sched->splits_capacity * sizeof(struct oiml_backend_sched_split));
					OIML_ASSERT(sched->splits != NULL);
				}
				split			  = &sched->splits[i_split];
				split->backend_id = node_backend_id;
				split->i_start	  = i;
				split->n_inputs	  = 0;
				cur_backend_id	  = node_backend_id;
			}

			// find inputs that are not on the same backend
			for (int j = 0; j < OIML_MAX_SRC; j++) {
				struct oiml_tensor* src = node->src[j];
				if (src == NULL) {
					continue;
				}

				size_t src_id			 = hash_id(src);
				const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
				assert(src_backend_id != -1);// all inputs should be assigned by now

				if (src->flags & OIML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
					if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
						oiml_backend_t backend = sched->backends[src_backend_id];
						for (int c = 0; c < sched->n_copies; c++) {
							struct oiml_tensor* tensor_copy;
							if (c == sched->cur_copy) {
								tensor_copy = src;// use the original tensor as the current copy
							} else {
								tensor_copy = oiml_dup_tensor_layout(sched->ctx, src);
								oiml_format_name(tensor_copy, "%s#%s#%d", oiml_backend_name(backend), src->name, c);
							}
							if (sched->n_copies > 1) {
								oiml_set_input(tensor_copy);
								oiml_set_output(tensor_copy);// prevent oiml-alloc from overwriting the tensor
							}
							tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
							SET_CAUSE(tensor_copy, "4.cpy");
						}
						int n_graph_inputs = sched->n_graph_inputs++;
						OIML_ASSERT(n_graph_inputs < OIML_SCHED_MAX_SPLIT_INPUTS);
						sched->graph_inputs[n_graph_inputs] = src;
					}
				}

				if (src_backend_id != cur_backend_id && !oiml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
					// create a copy of the input in the split's backend
					if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
						oiml_backend_t backend = sched->backends[cur_backend_id];
						for (int c = 0; c < sched->n_copies; c++) {
							struct oiml_tensor* tensor_copy = oiml_dup_tensor_layout(sched->ctx, src);
							oiml_format_name(tensor_copy, "%s#%s#%d", oiml_backend_name(backend), src->name, c);
							if (sched->n_copies > 1) {
								oiml_set_input(tensor_copy);
								oiml_set_output(tensor_copy);// prevent oiml-alloc from overwriting the tensor
							}
							tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
							SET_CAUSE(tensor_copy, "4.cpy");
						}
						int n_inputs = split->n_inputs++;
						OIML_ASSERT(n_inputs < OIML_SCHED_MAX_SPLIT_INPUTS);
						split->inputs[n_inputs] = src;
					}
					node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
				}
			}
		}
		split->i_end	= graph->n_nodes;
		sched->n_splits = i_split + 1;
	}

	if (sched->debug) {
		oiml_backend_sched_print_assignments(sched, graph);
	}

	// swap node_backend_ids and leaf _backend_ids with prevs
	{
		int* tmp					 = sched->node_backend_ids;
		sched->node_backend_ids		 = sched->prev_node_backend_ids;
		sched->prev_node_backend_ids = tmp;

		tmp							 = sched->leaf_backend_ids;
		sched->leaf_backend_ids		 = sched->prev_leaf_backend_ids;
		sched->prev_leaf_backend_ids = tmp;
	}

	int graph_size = std::max(graph->n_nodes, graph->n_leafs) + sched->n_splits * OIML_SCHED_MAX_SPLIT_INPUTS * 2 * sched->n_copies;
	if (sched->graph.size < graph_size) {
		sched->graph.size  = graph_size;
		sched->graph.nodes = ( oiml_tensor** )realloc(sched->graph.nodes, graph_size * sizeof(struct oiml_tensor*));
		sched->graph.leafs = ( oiml_tensor** )realloc(sched->graph.leafs, graph_size * sizeof(struct oiml_tensor*));
		OIML_ASSERT(sched->graph.nodes != NULL);
		OIML_ASSERT(sched->graph.leafs != NULL);
	}
	sched->graph.n_nodes = 0;
	sched->graph.n_leafs = 0;

	struct oiml_cgraph* graph_copy = &sched->graph;

	for (int i = 0; i < sched->n_splits; i++) {
		struct oiml_backend_sched_split* split = &sched->splits[i];
		split->graph						   = oiml_graph_view(graph, split->i_start, split->i_end);

		// add inputs to the graph copy so that they are allocated by oiml-alloc at the start of the split
		for (int j = 0; j < split->n_inputs; j++) {
			assert(graph_copy->size > (graph_copy->n_nodes + 1));

			struct oiml_tensor* input	  = split->inputs[j];
			const size_t input_id		  = hash_id(input);
			struct oiml_tensor* input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

			// add a dependency to the input source so that it is not freed before the copy is done
			struct oiml_tensor* input_dep				 = oiml_view_tensor(sched->ctx, input);
			input_dep->src[0]							 = input;
			sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
			graph_copy->nodes[graph_copy->n_nodes++]	 = input_dep;

			// add a dependency to the input copy so that it is allocated at the start of the split
			sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
			graph_copy->nodes[graph_copy->n_nodes++]	 = input_cpy;
		}

		for (int j = split->i_start; j < split->i_end; j++) {
			assert(graph_copy->size > graph_copy->n_nodes);
			sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
			graph_copy->nodes[graph_copy->n_nodes++]	 = graph->nodes[j];
		}
	}

	if (sched->n_copies > 1) {
		// add input copies as leafs so that they are allocated first
		for (int i = 0; i < sched->n_graph_inputs; i++) {
			struct oiml_tensor* input = sched->graph_inputs[i];
			size_t id				  = hash_id(input);
			int backend_id			  = tensor_backend_id(input);
			for (int c = 0; c < sched->n_copies; c++) {
				struct oiml_tensor* input_cpy				 = tensor_id_copy(id, backend_id, c);
				sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
				assert(graph_copy->size > graph_copy->n_leafs);
				graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
			}
		}

		for (int i = 0; i < sched->n_splits; i++) {
			struct oiml_backend_sched_split* split = &sched->splits[i];
			int backend_id						   = split->backend_id;
			for (int j = 0; j < split->n_inputs; j++) {
				struct oiml_tensor* input = split->inputs[j];
				size_t id				  = hash_id(input);
				for (int c = 0; c < sched->n_copies; c++) {
					struct oiml_tensor* input_cpy				 = tensor_id_copy(id, backend_id, c);
					sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
					assert(graph_copy->size > graph_copy->n_leafs);
					graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
				}
			}
		}
	}

	// add leafs from the original graph
	for (int i = 0; i < graph->n_leafs; i++) {
		struct oiml_tensor* leaf					 = graph->leafs[i];
		sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
		assert(graph_copy->size > graph_copy->n_leafs);
		graph_copy->leafs[graph_copy->n_leafs++] = leaf;
	}
}

OIML_INLINE static bool oiml_backend_sched_alloc_splits(oiml_backend_sched_t sched) {
	bool backend_ids_changed = false;
	for (int i = 0; i < sched->graph.n_nodes; i++) {
		if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] && sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
			backend_ids_changed = true;
			break;
		}
	}
	if (!backend_ids_changed) {
		for (int i = 0; i < sched->graph.n_leafs; i++) {
			if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] && sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
				backend_ids_changed = true;
				break;
			}
		}
	}

	// allocate graph
	if (backend_ids_changed || !oiml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
		// the re-allocation may cause the split inputs to be moved to a different address
		oiml_backend_sched_synchronize(sched);
#ifndef NDEBUG
		OIML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif
		oiml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
		if (!oiml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
			OIML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
			return false;
		}
	}

	return true;
}

OIML_INLINE static enum oiml_status oiml_backend_sched_compute_splits(oiml_backend_sched_t sched) {
	struct oiml_backend_sched_split* splits = sched->splits;

	for (int i = 0; i < sched->n_splits; i++) {
		struct oiml_backend_sched_split* split = &splits[i];
		int split_backend_id				   = split->backend_id;
		oiml_backend_t split_backend		   = sched->backends[split_backend_id];

		// copy the input tensors to the split backend
		for (int j = 0; j < split->n_inputs; j++) {
			oiml_backend_t input_backend  = oiml_backend_sched_get_tensor_backend(sched, split->inputs[j]);
			struct oiml_tensor* input	  = split->inputs[j];
			struct oiml_tensor* input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

			if (input->flags & OIML_TENSOR_FLAG_INPUT) {
				// inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
				if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
					oiml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
				} else {
					oiml_backend_synchronize(split_backend);
				}
				oiml_backend_tensor_copy(input, input_cpy);
			} else {
				// wait for the split backend to finish using the input before overwriting it
				if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
					oiml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
				} else {
					oiml_backend_synchronize(split_backend);
				}
				// try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
				// TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
				if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
					oiml_backend_synchronize(input_backend);
					if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
						oiml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
					} else {
						oiml_backend_synchronize(split_backend);
					}
					oiml_backend_tensor_copy(input, input_cpy);
				}
			}
		}

		if (!sched->callback_eval) {
			enum oiml_status ec = oiml_backend_graph_compute_async(split_backend, &split->graph);
			if (ec != OIML_STATUS_SUCCESS) {
				return ec;
			}
		} else {
			// similar to oiml_backend_compare_graph_backend
			for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
				struct oiml_tensor* t = split->graph.nodes[j0];

				// check if the user needs data from this node
				bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

				int j1 = j0;

				// determine the range [j0, j1] of nodes that can be computed together
				while (!need && j1 < split->graph.n_nodes - 1) {
					t	 = split->graph.nodes[++j1];
					need = sched->callback_eval(t, true, sched->callback_eval_user_data);
				}

				struct oiml_cgraph gv = oiml_graph_view(&split->graph, j0, j1 + 1);

				enum oiml_status ec = oiml_backend_graph_compute_async(split_backend, &gv);
				if (ec != OIML_STATUS_SUCCESS) {
					return ec;
				}

				// TODO: pass backend to the callback, then the user can decide if they want to synchronize
				oiml_backend_synchronize(split_backend);

				if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
					break;
				}

				j0 = j1;
			}
		}

		// record the event of this copy
		if (split->n_inputs > 0) {
			if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
				oiml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
			}
		}
	}

	sched->cur_copy = (sched->cur_copy + 1) % sched->n_copies;

	return OIML_STATUS_SUCCESS;
}

OIML_INLINE oiml_backend_sched_t oiml_backend_sched_new(oiml_backend_t* backends, oiml_backend_buffer_type_t* bufts, int n_backends, size_t graph_size, bool parallel) {
	OIML_ASSERT(n_backends > 0);
	OIML_ASSERT(n_backends <= OIML_SCHED_MAX_BACKENDS);
	OIML_ASSERT(oiml_backend_device_type(oiml_backend_get_device(backends[n_backends - 1])) == cpu);

	struct oiml_backend_sched* sched = ( oiml_backend_sched* )calloc(1, sizeof(struct oiml_backend_sched));

	const char* OIML_SCHED_DEBUG = getenv("OIML_SCHED_DEBUG");
	sched->debug				 = OIML_SCHED_DEBUG ? atoi(OIML_SCHED_DEBUG) : 0;
	sched->n_backends			 = n_backends;
	sched->n_copies				 = parallel ? OIML_SCHED_MAX_COPIES : 1;

	// initialize hash table
	// FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
	sched->hash_set				 = oiml_hash_set_new(graph_size);
	sched->hv_tensor_backend_ids = ( int* )malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
	sched->hv_tensor_copies		 = ( oiml_tensor** )malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct oiml_tensor*));

	const size_t oiml_sched_max_splits = graph_size;// at most there is one split for each node in the graph
	const size_t nodes_size			   = graph_size + oiml_sched_max_splits * OIML_SCHED_MAX_SPLIT_INPUTS * 2;
	sched->node_backend_ids			   = ( int* )calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
	sched->leaf_backend_ids			   = ( int* )calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
	sched->prev_node_backend_ids	   = ( int* )calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
	sched->prev_leaf_backend_ids	   = ( int* )calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

	sched->context_buffer_size = oiml_sched_max_splits * OIML_SCHED_MAX_SPLIT_INPUTS * 2 * sizeof(struct oiml_tensor) + oiml_graph_overhead_custom(graph_size, false);
	sched->context_buffer	   = ( char* )malloc(sched->context_buffer_size);

	const int initial_splits_capacity = 16;
	sched->splits					  = ( oiml_backend_sched_split* )calloc(initial_splits_capacity, sizeof(sched->splits[0]));
	sched->splits_capacity			  = initial_splits_capacity;

	for (int b = 0; b < n_backends; b++) {
		sched->backends[b] = backends[b];
		sched->bufts[b]	   = bufts ? bufts[b] : oiml_backend_get_default_buffer_type(backends[b]);
		OIML_ASSERT(oiml_backend_supports_buft(backends[b], sched->bufts[b]));

		if (sched->n_copies > 1) {
			for (int c = 0; c < sched->n_copies; c++) {
				sched->events[b][c] = oiml_backend_event_new(backends[b]->device);
			}
		}
	}

	sched->galloc = oiml_gallocr_new_n(sched->bufts, n_backends);

	oiml_backend_sched_reset(sched);

	return sched;
}

OIML_INLINE void oiml_backend_sched_free(oiml_backend_sched_t sched) {
	if (sched == NULL) {
		return;
	}
	for (int b = 0; b < sched->n_backends; b++) {
		for (int c = 0; c < sched->n_copies; c++) {
			oiml_backend_event_free(sched->events[b][c]);
		}
	}
	oiml_gallocr_free(sched->galloc);
	oiml_free(sched->ctx);
	oiml_hash_set_free(&sched->hash_set);
	free(sched->splits);
	free(sched->hv_tensor_backend_ids);
	free(sched->hv_tensor_copies);
	free(sched->node_backend_ids);
	free(sched->leaf_backend_ids);
	free(sched->prev_node_backend_ids);
	free(sched->prev_leaf_backend_ids);
	free(sched->context_buffer);
	free(sched->graph.nodes);
	free(sched->graph.leafs);
	free(sched);
}

OIML_INLINE void oiml_backend_sched_reset(oiml_backend_sched_t sched) {
	// reset state for the next run
	if (!sched->is_reset) {
		oiml_hash_set_reset(&sched->hash_set);
		memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
		memset(sched->hv_tensor_copies, 0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct oiml_tensor*));
		sched->is_reset = true;
	}
	sched->is_alloc = false;
}

OIML_INLINE bool oiml_backend_sched_reserve(oiml_backend_sched_t sched, struct oiml_cgraph* measure_graph) {
	OIML_ASSERT(( int )sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

	oiml_backend_sched_split_graph(sched, measure_graph);

	oiml_backend_sched_synchronize(sched);

	if (!oiml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
		return false;
	}

	oiml_backend_sched_reset(sched);

	return true;
}

OIML_INLINE bool oiml_backend_sched_alloc_graph(oiml_backend_sched_t sched, struct oiml_cgraph* graph) {
	OIML_ASSERT(( int )sched->hash_set.size >= graph->n_nodes + graph->n_leafs);

	oiml_backend_sched_split_graph(sched, graph);


	if (!oiml_backend_sched_alloc_splits(sched)) {
		return false;
	}

	sched->is_alloc = true;

	return true;
}

OIML_INLINE enum oiml_status oiml_backend_sched_graph_compute(oiml_backend_sched_t sched, struct oiml_cgraph* graph) {
	enum oiml_status err = oiml_backend_sched_graph_compute_async(sched, graph);
	oiml_backend_sched_synchronize(sched);
	return err;
}

OIML_INLINE enum oiml_status oiml_backend_sched_graph_compute_async(oiml_backend_sched_t sched, struct oiml_cgraph* graph) {
	if (!sched->is_reset && !sched->is_alloc) {
		oiml_backend_sched_reset(sched);
	}

	if (!sched->is_alloc) {
		if (!oiml_backend_sched_alloc_graph(sched, graph)) {
			return OIML_STATUS_ALLOC_FAILED;
		}
	}

	return oiml_backend_sched_compute_splits(sched);
}

OIML_INLINE void oiml_backend_sched_synchronize(oiml_backend_sched_t sched) {
	for (int i = 0; i < sched->n_backends; i++) {
		oiml_backend_synchronize(sched->backends[i]);
	}
}

OIML_INLINE void oiml_backend_sched_set_eval_callback(oiml_backend_sched_t sched, oiml_backend_sched_eval_callback callback, void* user_data) {
	sched->callback_eval		   = callback;
	sched->callback_eval_user_data = user_data;
}

OIML_INLINE int oiml_backend_sched_get_n_splits(oiml_backend_sched_t sched) {
	return sched->n_splits;
}

OIML_INLINE int oiml_backend_sched_get_n_copies(oiml_backend_sched_t sched) {
	return sched->n_copies;
}

OIML_INLINE int oiml_backend_sched_get_n_backends(oiml_backend_sched_t sched) {
	return sched->n_backends;
}

OIML_INLINE oiml_backend_t oiml_backend_sched_get_backend(oiml_backend_sched_t sched, int i) {
	OIML_ASSERT(i >= 0 && i < sched->n_backends);
	return sched->backends[i];
}

OIML_INLINE size_t oiml_backend_sched_get_buffer_size(oiml_backend_sched_t sched, oiml_backend_t backend) {
	int backend_index = oiml_backend_sched_backend_id(sched, backend);
	OIML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

	return oiml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

OIML_INLINE void oiml_backend_sched_set_tensor_backend(oiml_backend_sched_t sched, struct oiml_tensor* node, oiml_backend_t backend) {
	int backend_index = oiml_backend_sched_backend_id(sched, backend);
	OIML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
	tensor_backend_id(node) = backend_index;
	SET_CAUSE(node, "usr");
	sched->is_reset = false;
}

OIML_INLINE oiml_backend_t oiml_backend_sched_get_tensor_backend(oiml_backend_sched_t sched, struct oiml_tensor* node) {
	int backend_index = tensor_backend_id(node);
	if (backend_index == -1) {
		return NULL;
	}
	return sched->backends[backend_index];
}

// utils

OIML_INLINE void oiml_backend_view_init(struct oiml_tensor* tensor) {
	OIML_ASSERT(tensor->buffer == NULL);
	OIML_ASSERT(tensor->view_src != NULL);
	OIML_ASSERT(tensor->view_src->buffer != NULL);
	OIML_ASSERT(tensor->view_src->data != NULL);

	tensor->buffer = tensor->view_src->buffer;
	tensor->data   = ( char* )tensor->view_src->data + tensor->view_offs;

	// initialize data channel information
	tensor->num_channels = 1;
	tensor->data_channels[0].type = oiml_data_channel_type::value;
	tensor->data_channels[0].data_type = tensor->type;
	tensor->data_channels[0].data = tensor->data;
	tensor->data_channels[0].repeat_count = 1;
	tensor->data_channels[0].strides[0] = tensor->nb[0];
	tensor->data_channels[0].strides[1] = tensor->nb[1];
	tensor->data_channels[0].strides[2] = tensor->nb[2];
	tensor->data_channels[0].strides[3] = tensor->nb[3];

	switch (tensor->type) {
		case oiml::oiml_representation_types::q8_0: {
			tensor->data_channels[0].type = oiml_data_channel_type::block;
			break;
		}
	};

	oiml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

OIML_INLINE void oiml_backend_tensor_alloc(oiml_backend_buffer_t buffer, struct oiml_tensor* tensor, void* addr) {
	OIML_ASSERT(tensor->buffer == NULL);
	OIML_ASSERT(tensor->data == NULL);
	OIML_ASSERT(tensor->view_src == NULL);
	OIML_ASSERT(addr >= oiml_backend_buffer_get_base(buffer));
	OIML_ASSERT(( char* )addr + oiml_backend_buffer_get_alloc_size(buffer, tensor) <= ( char* )oiml_backend_buffer_get_base(buffer) + oiml_backend_buffer_get_size(buffer));

	tensor->buffer = buffer;
	tensor->data   = addr;

	// initialize data channel information
	tensor->num_channels = 1;
	tensor->data_channels[0].type = oiml_data_channel_type::value;
	tensor->data_channels[0].data_type = tensor->type;
	tensor->data_channels[0].data = tensor->data;
	tensor->data_channels[0].repeat_count = 1;
	tensor->data_channels[0].strides[0] = tensor->nb[0];
	tensor->data_channels[0].strides[1] = tensor->nb[1];
	tensor->data_channels[0].strides[2] = tensor->nb[2];
	tensor->data_channels[0].strides[3] = tensor->nb[3];

	switch (tensor->type) {
		case oiml::oiml_representation_types::q8_0: {
			tensor->data_channels[0].type = oiml_data_channel_type::block;
			break;
		}
	};

	oiml_backend_buffer_init_tensor(buffer, tensor);
}

OIML_INLINE static struct oiml_tensor* graph_copy_dup_tensor(struct oiml_hash_set hash_set, struct oiml_tensor** node_copies, struct oiml_context* ctx_allocated,
	struct oiml_context* ctx_unallocated, struct oiml_tensor* src) {
	OIML_ASSERT(src != NULL);
	OIML_ASSERT(src->data && "graph must be allocated");

	size_t id = oiml_hash_insert(&hash_set, src);
	if (id == OIML_HASHSET_ALREADY_EXISTS) {
		return node_copies[oiml_hash_find(&hash_set, src)];
	}

	struct oiml_tensor* dst = oiml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
	if (src->view_src != NULL) {
		dst->view_src  = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
		dst->view_offs = src->view_offs;
	}
	dst->op = src->op;
	memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
	oiml_set_name(dst, src->name);

	// copy src
	for (int i = 0; i < OIML_MAX_SRC; i++) {
		struct oiml_tensor* s = src->src[i];
		if (s == NULL) {
			continue;
		}
		dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
	}

	node_copies[id] = dst;
	return dst;
}

OIML_INLINE static void graph_copy_init_tensor(struct oiml_hash_set* hash_set, struct oiml_tensor** node_copies, bool* node_init, struct oiml_tensor* src) {
	size_t id = oiml_hash_find(hash_set, src);
	if (node_init[id]) {
		return;
	}
	node_init[id] = true;

	struct oiml_tensor* dst = node_copies[id];
	if (dst->view_src != NULL) {
		graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
		oiml_backend_view_init(dst);
	} else {
		oiml_backend_tensor_copy(src, dst);
	}

	// init src
	for (int i = 0; i < OIML_MAX_SRC; i++) {
		struct oiml_tensor* s = src->src[i];
		if (s == NULL) {
			continue;
		}
		graph_copy_init_tensor(hash_set, node_copies, node_init, s);
	}
}

OIML_INLINE struct oiml_backend_graph_copy oiml_backend_graph_copy(oiml_backend_t backend, struct oiml_cgraph* graph) {
	struct oiml_hash_set hash_set	 = oiml_hash_set_new(graph->visited_hash_set.size);
	struct oiml_tensor** node_copies = ( oiml_tensor** )calloc(hash_set.size, sizeof(node_copies[0]));// NOLINT
	bool* node_init					 = ( bool* )calloc(hash_set.size, sizeof(node_init[0]));

	struct oiml_init_params params = { /* .mem_size   = */ oiml_tensor_overhead() * hash_set.size + oiml_graph_overhead_custom(graph->size, false),
		/* .mem_buffer = */ NULL,
		/* .no_alloc   = */ true };

	struct oiml_context* ctx_allocated	 = oiml_init(params);
	struct oiml_context* ctx_unallocated = oiml_init(params);

	if (ctx_allocated == NULL || ctx_unallocated == NULL) {
		OIML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
		oiml_hash_set_free(&hash_set);
		free(node_copies);
		free(node_init);
		oiml_free(ctx_allocated);
		oiml_free(ctx_unallocated);
		return {
			/* .buffer           = */ NULL,
			/* .ctx_allocated    = */ NULL,
			/* .ctx_unallocated  = */ NULL,
			/* .graph            = */ NULL,
		};
	}

	// dup nodes
	for (int i = 0; i < graph->n_nodes; i++) {
		struct oiml_tensor* node = graph->nodes[i];
		graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
	}

	// allocate nodes
	oiml_backend_buffer_t buffer = oiml_backend_alloc_ctx_tensors(ctx_allocated, backend);
	if (buffer == NULL) {
		OIML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
		oiml_hash_set_free(&hash_set);
		free(node_copies);
		free(node_init);
		oiml_free(ctx_allocated);
		oiml_free(ctx_unallocated);
		return {
			/* .buffer           = */ NULL,
			/* .ctx_allocated    = */ NULL,
			/* .ctx_unallocated  = */ NULL,
			/* .graph            = */ NULL,
		};
	}

	//printf("copy buffer size: %zu MB\n", oiml_backend_buffer_get_size(buffer) / 1024 / 1024);

	// copy data and init views
	for (int i = 0; i < graph->n_nodes; i++) {
		struct oiml_tensor* node = graph->nodes[i];
		graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
	}

	// build graph copy
	struct oiml_cgraph* graph_copy = oiml_new_graph_custom(ctx_allocated, graph->size, false);
	for (int i = 0; i < graph->n_nodes; i++) {
		struct oiml_tensor* node	  = graph->nodes[i];
		struct oiml_tensor* node_copy = node_copies[oiml_hash_find(&hash_set, node)];
		graph_copy->nodes[i]		  = node_copy;
	}
	graph_copy->n_nodes = graph->n_nodes;

	oiml_hash_set_free(&hash_set);
	free(node_copies);
	free(node_init);

	return {
		/* .buffer           = */ buffer,
		/* .ctx_allocated    = */ ctx_allocated,
		/* .ctx_unallocated  = */ ctx_unallocated,
		/* .graph            = */ graph_copy,
	};
}

OIML_INLINE void oiml_backend_graph_copy_free(struct oiml_backend_graph_copy copy) {
	oiml_backend_buffer_free(copy.buffer);
	oiml_free(copy.ctx_allocated);
	oiml_free(copy.ctx_unallocated);
}

OIML_INLINE bool oiml_backend_compare_graph_backend(oiml_backend_t backend1, oiml_backend_t backend2, struct oiml_cgraph* graph, oiml_backend_eval_callback callback,
	void* user_data) {
	struct oiml_backend_graph_copy copy = oiml_backend_graph_copy(backend2, graph);
	if (copy.buffer == NULL) {
		return false;
	}

	struct oiml_cgraph* g1 = graph;
	struct oiml_cgraph* g2 = copy.graph;

	assert(g1->n_nodes == g2->n_nodes);

	for (int i = 0; i < g1->n_nodes; i++) {
		//printf("eval %d/%d\n", i, g1->n_nodes);
		struct oiml_tensor* t1 = g1->nodes[i];
		struct oiml_tensor* t2 = g2->nodes[i];

		assert(t1->op == t2->op && oiml_are_same_layout(t1, t2));

		struct oiml_cgraph g1v = oiml_graph_view(g1, i, i + 1);
		struct oiml_cgraph g2v = oiml_graph_view(g2, i, i + 1);

		oiml_backend_graph_compute(backend1, &g1v);
		oiml_backend_graph_compute(backend2, &g2v);

		if (oiml_is_view_op(t1->op)) {
			continue;
		}

		// compare results, calculate rms etc
		if (!callback(i, t1, t2, user_data)) {
			break;
		}
	}

	oiml_backend_graph_copy_free(copy);

	return true;
}

// CPU backend - buffer

OIML_INLINE static void* oiml_backend_cpu_buffer_get_base(oiml_backend_buffer_t buffer) {
	uintptr_t data = ( uintptr_t )buffer->context;

	// align the buffer
	if (data % TENSOR_ALIGNMENT != 0) {
		data = OIML_PAD(data, TENSOR_ALIGNMENT);
	}

	return ( void* )data;
}

OIML_INLINE static void oiml_backend_cpu_buffer_free_buffer(oiml_backend_buffer_t buffer) {
	oiml_aligned_free(buffer->context, buffer->size);
}

OIML_INLINE static void oiml_backend_cpu_buffer_memset_tensor(oiml_backend_buffer_t buffer, struct oiml_tensor* tensor, uint8_t value, size_t offset, size_t size) {
	memset(( char* )tensor->data + offset, value, size);

	OIML_UNUSED(buffer);
}

OIML_INLINE static void oiml_backend_cpu_buffer_set_tensor(oiml_backend_buffer_t buffer, struct oiml_tensor* tensor, const void* data, size_t offset, size_t size) {
	memcpy(( char* )tensor->data + offset, data, size);

	OIML_UNUSED(buffer);
}

OIML_INLINE static void oiml_backend_cpu_buffer_get_tensor(oiml_backend_buffer_t buffer, const struct oiml_tensor* tensor, void* data, size_t offset, size_t size) {
	memcpy(data, ( const char* )tensor->data + offset, size);

	OIML_UNUSED(buffer);
}

OIML_INLINE static bool oiml_backend_cpu_buffer_cpy_tensor(oiml_backend_buffer_t buffer, const struct oiml_tensor* src, struct oiml_tensor* dst) {
	if (oiml_backend_buffer_is_host(src->buffer)) {
		memcpy(dst->data, src->data, oiml_nbytes(src));
		return true;
	}
	return false;

	OIML_UNUSED(buffer);
}

OIML_INLINE static void oiml_backend_cpu_buffer_clear(oiml_backend_buffer_t buffer, uint8_t value) {
	memset(buffer->context, value, buffer->size);
}

static const struct oiml_backend_buffer_i oiml_backend_cpu_buffer_i = {
	/* .free_buffer     = */ oiml_backend_cpu_buffer_free_buffer,
	/* .get_base        = */ oiml_backend_cpu_buffer_get_base,
	/* .init_tensor     = */ NULL,// no initialization required
	/* .memset_tensor   = */ oiml_backend_cpu_buffer_memset_tensor,
	/* .set_tensor      = */ oiml_backend_cpu_buffer_set_tensor,
	/* .get_tensor      = */ oiml_backend_cpu_buffer_get_tensor,
	/* .cpy_tensor      = */ oiml_backend_cpu_buffer_cpy_tensor,
	/* .clear           = */ oiml_backend_cpu_buffer_clear,
	/* .reset           = */ NULL,
};

static const struct oiml_backend_buffer_i oiml_backend_cpu_buffer_from_ptr_i = {
	/* .free_buffer     = */ NULL,// ptr is not owned by the buffer, so it does not need to be freed
	/* .get_base        = */ oiml_backend_cpu_buffer_get_base,
	/* .init_tensor     = */ NULL,// no initialization required
	/* .memset_tensor   = */ oiml_backend_cpu_buffer_memset_tensor,
	/* .set_tensor      = */ oiml_backend_cpu_buffer_set_tensor,
	/* .get_tensor      = */ oiml_backend_cpu_buffer_get_tensor,
	/* .cpy_tensor      = */ oiml_backend_cpu_buffer_cpy_tensor,
	/* .clear           = */ oiml_backend_cpu_buffer_clear,
	/* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

OIML_INLINE static const char* oiml_backend_cpu_buffer_type_get_name(oiml_backend_buffer_type_t buft) {
	return "CPU";

	OIML_UNUSED(buft);
}

OIML_INLINE static oiml_backend_buffer_t oiml_backend_cpu_buffer_type_alloc_buffer(oiml_backend_buffer_type_t buft, size_t size) {
	void* data = oiml_aligned_malloc(size);

	if (data == NULL) {
		OIML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
		return NULL;
	}

	return oiml_backend_buffer_init(buft, oiml_backend_cpu_buffer_i, data, size);
}

OIML_INLINE static size_t oiml_backend_cpu_buffer_type_get_alignment(oiml_backend_buffer_type_t buft) {
	return TENSOR_ALIGNMENT;

	OIML_UNUSED(buft);
}

OIML_INLINE static bool oiml_backend_cpu_buffer_type_is_host(oiml_backend_buffer_type_t buft) {
	return true;

	OIML_UNUSED(buft);
}

OIML_INLINE oiml_backend_buffer_type_t oiml_backend_cpu_buffer_type(void) {
	static struct oiml_backend_buffer_type oiml_backend_cpu_buffer_type = {
		/* .iface   = */ {
			/* .get_name         = */ oiml_backend_cpu_buffer_type_get_name,
			/* .alloc_buffer     = */ oiml_backend_cpu_buffer_type_alloc_buffer,
			/* .get_alignment    = */ oiml_backend_cpu_buffer_type_get_alignment,
			/* .get_max_size     = */ NULL,// defaults to SIZE_MAX
			/* .get_alloc_size   = */ NULL,// defaults to oiml_nbytes
			/* .is_host          = */ oiml_backend_cpu_buffer_type_is_host,
		},
		/* .device  = */ NULL,// FIXME oiml_backend_reg_dev_get(oiml_backend_cpu_reg(), 0),
		/* .context = */ NULL,
	};

	return &oiml_backend_cpu_buffer_type;
}

OIML_INLINE static const char* oiml_backend_cpu_buffer_from_ptr_type_get_name(oiml_backend_buffer_type_t buft) {
	return "CPU_Mapped";

	OIML_UNUSED(buft);
}

OIML_INLINE static oiml_backend_buffer_type_t oiml_backend_cpu_buffer_from_ptr_type(void) {
	static struct oiml_backend_buffer_type oiml_backend_cpu_buffer_type = {
		/* .iface   = */ {
			/* .get_name         = */ oiml_backend_cpu_buffer_from_ptr_type_get_name,
			/* .alloc_buffer     = */ oiml_backend_cpu_buffer_type_alloc_buffer,
			/* .get_alignment    = */ oiml_backend_cpu_buffer_type_get_alignment,
			/* .get_max_size     = */ NULL,// defaults to SIZE_MAX
			/* .get_alloc_size   = */ NULL,// defaults to oiml_nbytes
			/* .is_host          = */ oiml_backend_cpu_buffer_type_is_host,
		},
		/* .device  = */ NULL,// FIXME oiml_backend_reg_dev_get(oiml_backend_cpu_reg(), 0),
		/* .context = */ NULL,
	};

	return &oiml_backend_cpu_buffer_type;
}

OIML_INLINE oiml_backend_buffer_t oiml_backend_cpu_buffer_from_ptr(void* ptr, size_t size) {
	OIML_ASSERT(( uintptr_t )ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
	return oiml_backend_buffer_init(oiml_backend_cpu_buffer_from_ptr_type(), oiml_backend_cpu_buffer_from_ptr_i, ptr, size);
}