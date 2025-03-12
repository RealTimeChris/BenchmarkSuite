#pragma once

#include <oiml/common/common.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-alloc-final.hpp>

typedef struct oiml_backend_buffer_type* oiml_backend_buffer_type_t;
typedef struct oiml_backend_buffer* oiml_backend_buffer_t;
typedef struct oiml_backend_event* oiml_backend_event_t;
typedef struct oiml_backend* oiml_backend_t;
typedef void* oiml_backend_graph_plan_t;
typedef struct oiml_backend_reg* oiml_backend_reg_t;
typedef struct oiml_backend_device* oiml_backend_dev_t;

struct oiml_vitali_debug_backend {};


//
// Backend buffer type
//

const char* oiml_backend_buft_name(oiml_backend_buffer_type_t buft);
oiml_backend_buffer_t oiml_backend_buft_alloc_buffer(oiml_backend_buffer_type_t buft, size_t size);
size_t oiml_backend_buft_get_alignment(oiml_backend_buffer_type_t buft);
size_t oiml_backend_buft_get_max_size(oiml_backend_buffer_type_t buft);
size_t oiml_backend_buft_get_alloc_size(oiml_backend_buffer_type_t buft, oiml_tensor* tensor);
bool oiml_backend_buft_is_host(oiml_backend_buffer_type_t buft);
oiml_backend_dev_t oiml_backend_buft_get_device(oiml_backend_buffer_type_t buft);

//
// Backend buffer
//

enum oiml_backend_buffer_usage {
	OIML_BACKEND_BUFFER_USAGE_ANY	  = 0,
	OIML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
	OIML_BACKEND_BUFFER_USAGE_COMPUTE = 2,
};

const char* oiml_backend_buffer_name(oiml_backend_buffer_t buffer);
void oiml_backend_buffer_free(oiml_backend_buffer_t buffer);
void* oiml_backend_buffer_get_base(oiml_backend_buffer_t buffer);
size_t oiml_backend_buffer_get_size(oiml_backend_buffer_t buffer);
void oiml_backend_buffer_init_tensor(oiml_backend_buffer_t buffer, oiml_tensor* tensor);
size_t oiml_backend_buffer_get_alignment(oiml_backend_buffer_t buffer);
size_t oiml_backend_buffer_get_max_size(oiml_backend_buffer_t buffer);
size_t oiml_backend_buffer_get_alloc_size(oiml_backend_buffer_t buffer, oiml_tensor* tensor);
void oiml_backend_buffer_clear(oiml_backend_buffer_t buffer, uint8_t value);
bool oiml_backend_buffer_is_host(oiml_backend_buffer_t buffer);
void oiml_backend_buffer_set_usage(oiml_backend_buffer_t buffer, enum oiml_backend_buffer_usage usage);
enum oiml_backend_buffer_usage oiml_backend_buffer_get_usage(oiml_backend_buffer_t buffer);
oiml_backend_buffer_type_t oiml_backend_buffer_get_type(oiml_backend_buffer_t buffer);
void oiml_backend_buffer_reset(oiml_backend_buffer_t buffer);

// tensor copy between different backends
void oiml_backend_tensor_copy(oiml_tensor* src, oiml_tensor* dst);

//
// Backend (stream)
//

oiml_guid_t oiml_backend_guid(oiml_backend_t backend);
const char* oiml_backend_name(oiml_backend_dev_t backend);
void oiml_backend_free(oiml_backend_t backend);

oiml_backend_buffer_type_t oiml_backend_get_default_buffer_type(oiml_backend_t backend);
oiml_backend_buffer_t oiml_backend_alloc_buffer(oiml_backend_t backend, size_t size);
size_t oiml_backend_get_alignment(oiml_backend_t backend);
size_t oiml_backend_get_max_size(oiml_backend_t backend);

void oiml_backend_tensor_set_async(oiml_backend_t backend, oiml_tensor* tensor, const void* data, size_t offset, size_t size);
void oiml_backend_tensor_get_async(oiml_backend_t backend, const oiml_tensor* tensor, void* data, size_t offset, size_t size);

// "offset" refers to the offset in tensor->data for setting/getting data
void oiml_backend_tensor_set(oiml_tensor* tensor, const void* data, size_t offset, size_t size);
void oiml_backend_tensor_get(const oiml_tensor* tensor, void* data, size_t offset, size_t size);
void oiml_backend_tensor_memset(oiml_tensor* tensor, uint8_t value, size_t offset, size_t size);

void oiml_backend_synchronize(oiml_backend_t backend);

oiml_backend_graph_plan_t oiml_backend_graph_plan_create(oiml_backend_t backend, oiml_cgraph* cgraph);
void oiml_backend_graph_plan_free(oiml_backend_t backend, oiml_backend_graph_plan_t plan);

oiml_status oiml_backend_graph_plan_compute(oiml_backend_t backend, oiml_backend_graph_plan_t plan);
oiml_status oiml_backend_graph_compute(oiml_backend_t backend, oiml_cgraph* cgraph);
oiml_status oiml_backend_graph_compute_async(oiml_backend_t backend, oiml_cgraph* cgraph);

// NOTE: will be removed, use device version instead
bool oiml_backend_supports_op(oiml_backend_t backend, const oiml_tensor* op);
bool oiml_backend_supports_buft(oiml_backend_t backend, oiml_backend_buffer_type_t buft);
bool oiml_backend_offload_op(oiml_backend_t backend, const oiml_tensor* op);

// asynchronous copy
// the copy is performed after all the currently queued operations in backend_src
// backend_dst will wait for the copy to complete before performing other operations
// automatic fallback to sync copy if async is not supported
void oiml_backend_tensor_copy_async(oiml_backend_t backend_src, oiml_backend_t backend_dst, oiml_tensor* src, oiml_tensor* dst);

oiml_backend_dev_t oiml_backend_get_device(oiml_backend_t backend);

//
// Events
//

oiml_backend_event_t oiml_backend_event_new(oiml_backend_dev_t device);
void oiml_backend_event_free(oiml_backend_event_t event);
void oiml_backend_event_record(oiml_backend_event_t event, oiml_backend_t backend);
void oiml_backend_event_synchronize(oiml_backend_event_t event);
void oiml_backend_event_wait(oiml_backend_t backend, oiml_backend_event_t event);

//
// Backend device
//


// functionality supported by the device
struct oiml_backend_dev_caps {
	// asynchronous operations
	bool async;
	// pinned host buffer
	bool host_buffer;
	// creating buffers from host ptr
	bool buffer_from_host_ptr;
	// event synchronization
	bool events;
};

// all the device properties
struct oiml_backend_dev_props {
	const char* name;
	const char* description;
	size_t memory_free;
	size_t memory_total;
	oiml_backend_device_types type;
	struct oiml_backend_dev_caps caps;
};

const char* oiml_backend_dev_name(oiml_backend_dev_t device);
const char* oiml_backend_dev_description(oiml_backend_dev_t device);
void oiml_backend_dev_memory(oiml_backend_dev_t device, size_t* free, size_t* total);
enum oiml_backend_device_types oiml_backend_device_type(oiml_backend_dev_t device);
void oiml_backend_dev_get_props(oiml_backend_dev_t device, struct oiml_backend_dev_props* props);
oiml_backend_reg_t oiml_backend_dev_backend_reg(oiml_backend_dev_t device);
oiml_backend_t oiml_backend_dev_init(oiml_backend_dev_t device, const char* params);
oiml_backend_buffer_type_t oiml_backend_dev_buffer_type(oiml_backend_dev_t device);
oiml_backend_buffer_type_t oiml_backend_dev_host_buffer_type(oiml_backend_dev_t device);
oiml_backend_buffer_t oiml_backend_dev_buffer_from_host_ptr(oiml_backend_dev_t device, void* ptr, size_t size, size_t max_tensor_size);

bool oiml_backend_dev_supports_op(oiml_backend_dev_t device, const oiml_tensor* op);
bool oiml_backend_dev_supports_buft(oiml_backend_dev_t device, oiml_backend_buffer_type_t buft);
bool oiml_backend_dev_offload_op(oiml_backend_dev_t device, const oiml_tensor* op);

//
// Backend (reg)
//

const char* oiml_backend_reg_name(oiml_backend_reg_t reg);
size_t oiml_backend_reg_dev_count(oiml_backend_reg_t reg);
oiml_backend_dev_t oiml_backend_reg_dev_get(oiml_backend_reg_t reg, size_t index);
void* oiml_backend_reg_get_proc_address(oiml_backend_reg_t reg, const char* name);

// Common functions that may be obtained using oiml_backend_reg_get_proc_address

// Split buffer type for tensor parallelism
typedef oiml_backend_buffer_type_t (*oiml_backend_split_buffer_type_t)(int main_device, const float* tensor_split);
// Set the number of threads for the backend
typedef void (*oiml_backend_set_n_threads_t)(oiml_backend_t backend, int n_threads);
// Get additional buffer types provided by the device (returns a NULL-terminated array)
typedef oiml_backend_buffer_type_t* (*oiml_backend_dev_get_extra_bufts_t)(oiml_backend_dev_t device);
// Set the abort callback for the backend
typedef void (*oiml_backend_set_abort_callback_t)(oiml_backend_t backend, oiml_abort_callback abort_callback, void* abort_callback_data);
// Get a list of feature flags supported by the backend (returns a NULL-terminated array)
struct oiml_backend_feature {
	const char* name;
	const char* value;
};
typedef struct oiml_backend_feature* (*oiml_backend_get_features_t)(oiml_backend_reg_t reg);

//
// Backend registry
//

void oiml_backend_device_register(oiml_backend_dev_t device);

// Backend (reg) enumeration
size_t oiml_backend_reg_count();
oiml_backend_reg_t oiml_backend_reg_get(size_t index);
oiml_backend_reg_t oiml_backend_reg_by_name(const char* name);

// Device enumeration
size_t oiml_backend_dev_count();
oiml_backend_dev_t oiml_backend_dev_get(size_t index);
oiml_backend_dev_t oiml_backend_dev_by_name(const char* name);
oiml_backend_dev_t oiml_backend_dev_by_type(enum oiml_backend_device_types type);

// Direct backend (stream) initialization
// = oiml_backend_dev_init(oiml_backend_dev_by_name(name), params)
oiml_backend_t oiml_backend_init_by_name(const char* name, const char* params);
// = oiml_backend_dev_init(oiml_backend_dev_by_type(type), params)
oiml_backend_t oiml_backend_init_by_type(enum oiml_backend_device_types type, const char* params);
// = oiml_backend_dev_init(oiml_backend_dev_by_type(GPU) OR oiml_backend_dev_by_type(CPU), NULL)
oiml_backend_t oiml_backend_init_best();

// Load a backend from a dynamic library and register it
oiml_backend_reg_t oiml_backend_load(const char* path);
// Unload a backend if loaded dynamically and unregister it
void oiml_backend_unload(oiml_backend_reg_t reg);
// Load all known backends from dynamic libraries
void oiml_backend_load_all();
void oiml_backend_load_all_from_path(const char* dir_path);

//
// Backend scheduler
//

// The backend scheduler allows for multiple backend devices to be used together
// Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
// The backends are selected based on:
// - the backend that supports the operation
// - the location of the pre-allocated tensors (e.g. the weights)
/*
      Example usage:

        // operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be assigned
        // preferrably to run on the same backend as the buffer
        oiml_backend_buffer_set_usage(buf_weights, OIML_BACKEND_BUFFER_USAGE_WEIGHTS);

        sched = oiml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, OIML_DEFAULT_GRAPH_SIZE, false);

        // initialize buffers from a max size graph (optional)
        reserve_graph = build_graph(sched, max_batch_size);

        // manually assign nodes to a backend (optional, should not be needed in most cases)
        struct oiml_tensor * node = oiml_mul_mat(ctx, ...);
        oiml_backend_sched_set_tensor_backend(sched, node, backend_gpu);

        oiml_backend_sched_reserve(sched, reserve_graph);

        // compute
        graph = build_graph(sched); // the graph and its tensors are single-use in terms of allocation, multi-use in terms of computation
        for (int i = 0; i < 10; ++i) {
            oiml_backend_sched_graph_compute(sched, graph); // on the first iteration the graph is allocated automatically
        }

        // if there are graph inputs:
        graph = build_graph(sched); // get a new graph that is not allocated (the metadata for the old graph is freed once oiml_free is called)
        oiml_backend_sched_reset(sched); // clear the allocation of the previous graph
        oiml_backend_sched_alloc_graph(sched, graph); // explicitly allocate the new graph but do not execute it
        oiml_backend_tensor_set(input_tensor, ...); // copy data to the newly allocated graph tensors
        oiml_backend_sched_graph_compute(sched, graph); // execute the graph

        // as an alternative to the above it is also possible to assign the inputs to a dedicated context and
        // allocate them statically via oiml_backend_alloc_ctx_tensors
    }
    */

typedef struct oiml_backend_sched* oiml_backend_sched_t;

// Evaluation callback for each node in the graph (set with oiml_backend_sched_set_eval_callback)
// when ask == true, the scheduler wants to know if the user wants to observe this node
// this allows the scheduler to batch nodes together in order to evaluate them in a single call
//
// when ask == false, the scheduler is passing the node tensor to the user for observation
// if the user returns false, the scheduler will cancel the graph compute
//
typedef bool (*oiml_backend_sched_eval_callback)(oiml_tensor* t, bool ask, void* user_data);

// Initialize a backend scheduler, backends with low index are given priority over backends with high index
oiml_backend_sched_t oiml_backend_sched_new(oiml_backend_t* backends, oiml_backend_buffer_type_t* bufts, int n_backends, size_t graph_size, bool parallel);
void oiml_backend_sched_free(oiml_backend_sched_t sched);

// Initialize backend buffers from a measure graph
bool oiml_backend_sched_reserve(oiml_backend_sched_t sched, oiml_cgraph* measure_graph);// returns success

int oiml_backend_sched_get_n_backends(oiml_backend_sched_t sched);
oiml_backend_t oiml_backend_sched_get_backend(oiml_backend_sched_t sched, int i);

// Get the number of splits of the last graph
int oiml_backend_sched_get_n_splits(oiml_backend_sched_t sched);
int oiml_backend_sched_get_n_copies(oiml_backend_sched_t sched);

size_t oiml_backend_sched_get_buffer_size(oiml_backend_sched_t sched, oiml_backend_t backend);

void oiml_backend_sched_set_tensor_backend(oiml_backend_sched_t sched, oiml_tensor* node, oiml_backend_t backend);
oiml_backend_t oiml_backend_sched_get_tensor_backend(oiml_backend_sched_t sched, oiml_tensor* node);

// Allocate and compute graph on the backend scheduler
bool oiml_backend_sched_alloc_graph(oiml_backend_sched_t sched, oiml_cgraph* graph);// returns success
oiml_status oiml_backend_sched_graph_compute(oiml_backend_sched_t sched, oiml_cgraph* graph);
oiml_status oiml_backend_sched_graph_compute_async(oiml_backend_sched_t sched, oiml_cgraph* graph);
void oiml_backend_sched_synchronize(oiml_backend_sched_t sched);

// Reset all assignments and allocators - must be called before changing the node backends or allocating a new graph.
// This in effect deallocates all tensors that were previously allocated and leaves them with dangling pointers.
// The correct way to use this API is to discard the deallocated tensors and create new ones.
void oiml_backend_sched_reset(oiml_backend_sched_t sched);

// Set a callback to be called for each resulting node during graph compute
void oiml_backend_sched_set_eval_callback(oiml_backend_sched_t sched, oiml_backend_sched_eval_callback callback, void* user_data);

//
// Utils
//

struct oiml_backend_graph_copy {
	oiml_backend_buffer_t buffer;
	oiml_context* ctx_allocated;
	oiml_context* ctx_unallocated;
	oiml_cgraph* graph;
};

// Copy a graph to a different backend
struct oiml_backend_graph_copy oiml_backend_graph_copy(oiml_backend_t backend, oiml_cgraph* graph);
void oiml_backend_graph_copy_free(struct oiml_backend_graph_copy copy);

typedef bool (*oiml_backend_eval_callback)(int node_index, oiml_tensor* t1, oiml_tensor* t2, void* user_data);

// Compare the output of two backends
bool oiml_backend_compare_graph_backend(oiml_backend_t backend1, oiml_backend_t backend2, oiml_cgraph* graph, oiml_backend_eval_callback callback, void* user_data);

// Tensor initialization
void oiml_backend_tensor_alloc(oiml_backend_buffer_t buffer, oiml_tensor* tensor, void* addr);
void oiml_backend_view_init(oiml_tensor* tensor);

// CPU buffer types are always available
oiml_backend_buffer_t oiml_backend_cpu_buffer_from_ptr(void* ptr, size_t size);
oiml_backend_buffer_type_t oiml_backend_cpu_buffer_type();
