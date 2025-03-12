#pragma once

// oiml-backend internal header

#include <oiml/legacy/oiml-legacy-common/oiml-backend.hpp>



#define OIML_API_VERSION 1

//
// Backend buffer type
//

struct oiml_backend_buffer_type_i {
	const char* (*get_name)(oiml_backend_buffer_type_t buft);
	// allocate a buffer of this type
	oiml_backend_buffer_t (*alloc_buffer)(oiml_backend_buffer_type_t buft, size_t size);
	// tensor alignment
	size_t (*get_alignment)(oiml_backend_buffer_type_t buft);
	// (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
	size_t (*get_max_size)(oiml_backend_buffer_type_t buft);
	// (optional) data size needed to allocate the tensor, including padding (defaults to oiml_nbytes)
	size_t (*get_alloc_size)(oiml_backend_buffer_type_t buft, const oiml_tensor* tensor);
	// (optional) check if tensor data is in host memory and uses standard oiml tensor layout (defaults to false)
	bool (*is_host)(oiml_backend_buffer_type_t buft);
};

struct oiml_backend_buffer_type {
	struct oiml_backend_buffer_type_i iface;
	oiml_backend_dev_t device;
	void* context;
};

//
// Backend buffer
//

struct oiml_backend_buffer_i {
	// (optional) free the buffer
	void (*free_buffer)(oiml_backend_buffer_t buffer);
	// base address of the buffer
	void* (*get_base)(oiml_backend_buffer_t buffer);
	// (optional) initialize a tensor in the buffer (eg. add tensor extras)
	oiml_status (*init_tensor)(oiml_backend_buffer_t buffer, oiml_tensor* tensor);
	// tensor data access
	void (*memset_tensor)(oiml_backend_buffer_t buffer, oiml_tensor* tensor, uint8_t value, size_t offset, size_t size);
	void (*set_tensor)(oiml_backend_buffer_t buffer, oiml_tensor* tensor, const void* data, size_t offset, size_t size);
	void (*get_tensor)(oiml_backend_buffer_t buffer, const oiml_tensor* tensor, void* data, size_t offset, size_t size);
	// (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
	bool (*cpy_tensor)(oiml_backend_buffer_t buffer, const oiml_tensor* src, oiml_tensor* dst);
	// clear the entire buffer
	void (*clear)(oiml_backend_buffer_t buffer, uint8_t value);
	// (optional) reset any internal state due to tensor initialization, such as tensor extras
	void (*reset)(oiml_backend_buffer_t buffer);
};

struct oiml_backend_buffer {
	struct oiml_backend_buffer_i iface;
	oiml_backend_buffer_type_t buft;
	void* context;
	size_t size;
	enum oiml_backend_buffer_usage usage;
};

oiml_backend_buffer_t oiml_backend_buffer_init(oiml_backend_buffer_type_t buft, struct oiml_backend_buffer_i iface, void* context, size_t size);

// do not use directly, use oiml_backend_tensor_copy instead
bool oiml_backend_buffer_copy_tensor(const oiml_tensor* src, oiml_tensor* dst);

// multi-buffer
// buffer that contains a collection of buffers
oiml_backend_buffer_t oiml_backend_multi_buffer_alloc_buffer(oiml_backend_buffer_t* buffers, size_t n_buffers);
bool oiml_backend_buffer_is_multi_buffer(oiml_backend_buffer_t buffer);
void oiml_backend_multi_buffer_set_usage(oiml_backend_buffer_t buffer, enum oiml_backend_buffer_usage usage);

//
// Backend (stream)
//

struct oiml_backend_i {
	const char* (*get_name)(oiml_backend_t);

	void (*free)(oiml_backend_t backend);

	// (optional) asynchronous tensor data access
	void (*set_tensor_async)(oiml_backend_t backend, oiml_tensor* tensor, const void* data, size_t offset, size_t size);
	void (*get_tensor_async)(oiml_backend_t backend, const oiml_tensor* tensor, void* data, size_t offset, size_t size);
	bool (*cpy_tensor_async)(oiml_backend_t backend_src, oiml_backend_t backend_dst, const oiml_tensor* src, oiml_tensor* dst);

	// (optional) complete all pending operations (required if the backend supports async operations)
	void (*synchronize)(oiml_backend_t backend);

	// (optional) graph plans (not used currently)
	// compute graph with a plan
	oiml_backend_graph_plan_t (*graph_plan_create)(oiml_backend_t backend, const oiml_cgraph* cgraph);
	void (*graph_plan_free)(oiml_backend_t backend, oiml_backend_graph_plan_t plan);
	// update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
	void (*graph_plan_update)(oiml_backend_t backend, oiml_backend_graph_plan_t plan, const oiml_cgraph* cgraph);
	// compute the graph with the plan
	oiml_status (*graph_plan_compute)(oiml_backend_t backend, oiml_backend_graph_plan_t plan);

	// compute graph (always async if supported by the backend)
	oiml_status (*graph_compute)(oiml_backend_t backend, oiml_cgraph* cgraph);

	// (optional) event synchronization
	// record an event on this stream
	void (*event_record)(oiml_backend_t backend, oiml_backend_event_t event);
	// wait for an event on on a different stream
	void (*event_wait)(oiml_backend_t backend, oiml_backend_event_t event);
};

struct oiml_backend {
	oiml_guid_t guid;
	struct oiml_backend_i iface;
	oiml_backend_dev_t device;
	void* context;
};

struct oiml_backend_event {
	oiml_backend_device* device;
	void* context;
};

//
// Backend device
//

// Note: if additional properties are needed, we should add a struct with all of them
//       the current functions to obtain the properties can remain, since they are more convenient for often used properties
struct oiml_backend_device_i {
	// device name: short identifier for this device, such as "CPU" or "CUDA0"
	const char* (*get_name)(oiml_backend_dev_t);

	// device description: short informative description of the device, could be the model name
	const char* (*get_description)(oiml_backend_dev_t dev);

	// device memory in bytes
	void (*get_memory)(oiml_backend_dev_t dev, size_t* free, size_t* total);

	// device type
	enum oiml_backend_device_types (*get_type)(oiml_backend_dev_t dev);

	// device properties
	void (*get_props)(oiml_backend_dev_t dev, oiml_backend_dev_props* props);

	// backend (stream) initialization
	oiml_backend_t (*init_backend)(oiml_backend_dev_t dev, const char* params);

	// preferred buffer type
	oiml_backend_buffer_type_t (*get_buffer_type)(oiml_backend_dev_t dev);

	// (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
	oiml_backend_buffer_type_t (*get_host_buffer_type)(oiml_backend_dev_t dev);

	// (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
	oiml_backend_buffer_t (*buffer_from_host_ptr)(oiml_backend_dev_t dev, void* ptr, size_t size, size_t max_tensor_size);

	// check if the backend can compute an operation
	bool (*supports_op)(oiml_backend_dev_t dev, const oiml_tensor* op);

	// check if the backend can use tensors allocated in a buffer type
	bool (*supports_buft)(oiml_backend_dev_t dev, oiml_backend_buffer_type_t buft);

	// (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
	// these should be expensive operations that may benefit from running on this backend instead of the CPU backend
	bool (*offload_op)(oiml_backend_dev_t dev, const oiml_tensor* op);

	// (optional) event synchronization
	oiml_backend_event_t (*event_new)(oiml_backend_dev_t dev);
	void (*event_free)(oiml_backend_dev_t dev, oiml_backend_event_t event);
	void (*event_synchronize)(oiml_backend_dev_t dev, oiml_backend_event_t event);
};

struct oiml_backend_device {
	struct oiml_backend_device_i iface;
	oiml_backend_reg_t reg;
	void* context;
};

//
// Backend (reg)
//

struct oiml_backend_reg_i {
	const char* (*get_name)(oiml_backend_reg_t reg);

	// enumerate available devices
	size_t (*get_device_count)(oiml_backend_reg_t reg);
	oiml_backend_dev_t (*get_device)(oiml_backend_reg_t reg, size_t index);

	// (optional) get a pointer to a function in the backend
	// backends can add custom functions that are not part of the standard oiml-backend interface
	void* (*get_proc_address)(oiml_backend_reg_t reg, const char* name);
};

struct oiml_backend_reg {
	int api_version;// initialize to OIML_API_VERSION
	struct oiml_backend_reg_i iface;
	void* context;
};

// Internal backend registry API
void oiml_backend_register(oiml_backend_reg_t reg);

// Add backend dynamic loading support to the backend

// Initialize the backend
typedef oiml_backend_reg_t (*oiml_backend_init_t)();
// Optional: obtain a score for the backend based on the system configuration
// Higher scores are preferred, 0 means the backend is not supported in the current system
typedef int (*oiml_backend_score_t)();

#define OIML_BACKEND_DL_IMPL(reg_fn)
#define OIML_BACKEND_DL_SCORE_IMPL(score_fn)
