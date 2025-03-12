#pragma once

#include <oiml/legacy/oiml-legacy-common/oiml-final.hpp>

typedef struct oiml_backend_buffer_type* oiml_backend_buffer_type_t;
typedef struct oiml_backend_buffer* oiml_backend_buffer_t;
typedef struct oiml_backend* oiml_backend_t;

// Tensor allocator
struct oiml_tallocr {
	oiml_backend_buffer_t buffer;
	void* base;
	size_t alignment;
	size_t offset;
};

struct oiml_tallocr oiml_tallocr_new(oiml_backend_buffer_t buffer);
void oiml_tallocr_alloc(oiml_tallocr* talloc, oiml_tensor* tensor);

// Graph allocator
/*
  Example usage:
    oiml_gallocr_t galloc = oiml_gallocr_new(oiml_backend_cpu_buffer_type());

    // optional: create a worst-case graph and reserve the buffers to avoid reallocations
    oiml_gallocr_reserve(galloc, build_graph(max_batch));

    // allocate the graph
    struct oiml_cgraph * graph = build_graph(batch);
    oiml_gallocr_alloc_graph(galloc, graph);

    printf("compute buffer size: %zu bytes\n", oiml_gallocr_get_buffer_size(galloc, 0));

    // evaluate the graph
    oiml_backend_graph_compute(backend, graph);
*/

// special tensor flags for use with the graph allocator:
//   oiml_set_input(): all input tensors are allocated at the beginning of the graph in non-overlapping addresses
//   oiml_set_output(): output tensors are never freed and never overwritten

typedef struct oiml_gallocr* oiml_gallocr_t;

oiml_gallocr_t oiml_gallocr_new(oiml_backend_buffer_type_t buft);
oiml_gallocr_t oiml_gallocr_new_n(oiml_backend_buffer_type_t* bufts, int n_bufs);
void oiml_gallocr_free(oiml_gallocr_t galloc);

// pre-allocate buffers from a measure graph - does not allocate or modify the graph
// call with a worst-case graph to avoid buffer reallocations
// not strictly required for single buffer usage: oiml_gallocr_alloc_graph will reallocate the buffers automatically if needed
// returns false if the buffer allocation failed
bool oiml_gallocr_reserve(oiml_gallocr_t galloc, oiml_cgraph* graph);
bool oiml_gallocr_reserve_n(oiml_gallocr_t galloc, oiml_cgraph* graph, const int* node_buffer_ids, const int* leaf_buffer_ids);

// automatic reallocation if the topology changes when using a single buffer
// returns false if using multiple buffers and a re-allocation is needed (call oiml_gallocr_reserve_n first to set the node buffers)
bool oiml_gallocr_alloc_graph(oiml_gallocr_t galloc, oiml_cgraph* graph);

size_t oiml_gallocr_get_buffer_size(oiml_gallocr_t galloc, int buffer_id);

// Utils
// Create a buffer and allocate all the tensors in a oiml_context
oiml_backend_buffer* oiml_backend_alloc_ctx_tensors_from_buft(oiml_context* ctx, oiml_backend_buffer_type_t buft);
oiml_backend_buffer* oiml_backend_alloc_ctx_tensors(oiml_context* ctx, oiml_backend_t backend);
