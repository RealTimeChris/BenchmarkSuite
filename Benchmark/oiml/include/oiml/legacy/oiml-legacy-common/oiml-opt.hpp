// This file contains functionality for training models using OIML.
// It is not strictly needed vs. just vanilla OIML but it provides a more high-level interface for common needs such as datasets.
// At the bottom of this file especially there are relatively high-level functions that are suitable use or adaptation in user code.
//
// Module maintainer: Johannes Gäßler (@JohannesGaessler, johannesg@5d6.de)

#pragma once

#include <oiml/legacy/oiml-legacy-common/oiml-final.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-backend.hpp>

#include <stdint.h>


struct oiml_opt_dataset;
struct oiml_opt_context;
struct oiml_opt_result;

typedef oiml_opt_dataset* oiml_opt_dataset_t;
typedef oiml_opt_context* oiml_opt_context_t;
typedef oiml_opt_result* oiml_opt_result_t;

// ====== Loss ======

// built-in loss types, i.e. the built-in quantities minimized by the optimizer
// custom loss types can be defined via mean or sum which simply reduce the outputs for all datapoints to a single value
enum oiml_opt_loss_type {
	OIML_OPT_LOSS_TYPE_MEAN,
	OIML_OPT_LOSS_TYPE_SUM,
	OIML_OPT_LOSS_TYPE_CROSS_ENTROPY,
	OIML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
};

// ====== Dataset ======

oiml_opt_dataset_t oiml_opt_dataset_init(int64_t ne_datapoint,// number of elements per datapoint
	int64_t ne_label,// number of elements per label
	int64_t ndata,// total number of datapoints/labels
	int64_t ndata_shard);// number of datapoints/labels per shard (unit at which the dataset is shuffled/copied)
void oiml_opt_dataset_free(oiml_opt_dataset_t dataset);

// get underlying tensors that store the data
oiml_tensor* oiml_opt_dataset_data(oiml_opt_dataset_t dataset);// shape = [ne_datapoint, ndata]
oiml_tensor* oiml_opt_dataset_labels(oiml_opt_dataset_t dataset);// shape = [nd_label,     ndata]

// shuffle idata first datapoints from dataset with RNG from opt_ctx, shuffle all datapoints if idata is negative
void oiml_opt_dataset_shuffle(oiml_opt_context_t opt_ctx, oiml_opt_dataset_t dataset, int64_t idata);

// get batch at position ibatch from dataset and copy the data to data_batch and labels_batch
void oiml_opt_dataset_get_batch(oiml_opt_dataset_t dataset,
	oiml_tensor* data_batch,// shape = [ne_datapoint, ndata_batch]
	oiml_tensor* labels_batch,// shape = [ne_label,     ndata_batch]
	int64_t ibatch);

// ====== Model / Context ======

enum oiml_opt_build_type {
	OIML_OPT_BUILD_TYPE_FORWARD,
	OIML_OPT_BUILD_TYPE_GRAD,
	OIML_OPT_BUILD_TYPE_OPT,
};

// parameters that control which optimizer is used and how said optimizer tries to find the minimal loss
struct oiml_opt_optimizer_params {
	// AdamW optimizer parameters
	struct {
		float alpha;// learning rate
		float beta1;
		float beta2;
		float eps;// epsilon for numerical stability
		float wd;// weight decay for AdamW, use 0.0f to disable
	} adamw;
};

// callback to calculate optimizer parameters prior to a backward pass
// userdata can be used to pass arbitrary data
typedef struct oiml_opt_optimizer_params (*oiml_opt_get_optimizer_params)(void* userdata);

// returns the default optimizer params (constant)
// userdata is not used
struct oiml_opt_optimizer_params oiml_opt_get_default_optimizer_params(void* userdata);

// parameters for initializing a new optimization context
struct oiml_opt_params {
	oiml_backend_sched_t backend_sched;// defines which backends are used to construct the compute graphs

	oiml_context* ctx_compute;// created in user code, holds non-static tensors

	// the forward graph is defined by inputs and outputs
	// those tensors and all tensors inbetween are not intended to be reusable between multiple optimization contexts
	oiml_tensor* inputs;
	oiml_tensor* outputs;

	enum oiml_opt_loss_type loss_type;
	enum oiml_opt_build_type build_type;

	int32_t opt_period;// after how many gradient accumulation steps an optimizer step should be done

	oiml_opt_get_optimizer_params get_opt_pars;// callback for calculating optimizer parameters
	void* get_opt_pars_ud;// userdata for calculating optimizer parameters
};

// get parameters for an optimization context with defaults set where possible
// parameters for which no sensible defaults exist are supplied as arguments to this function
oiml_opt_params oiml_opt_default_params(oiml_backend_sched_t backend_sched, oiml_context* ctx_compute, oiml_tensor* inputs, oiml_tensor* outputs,
	enum oiml_opt_loss_type loss_type);

oiml_opt_context_t oiml_opt_init(struct oiml_opt_params params);
void oiml_opt_free(oiml_opt_context_t opt_ctx);

// set gradients to zero, initilize loss, and optionally reset the optimizer
void oiml_opt_reset(oiml_opt_context_t opt_ctx, bool optimizer);

// get underlying tensors that store data
oiml_tensor* oiml_opt_inputs(oiml_opt_context_t opt_ctx);// forward graph input tensor
oiml_tensor* oiml_opt_outputs(oiml_opt_context_t opt_ctx);// forward graph output tensor
oiml_tensor* oiml_opt_labels(oiml_opt_context_t opt_ctx);// labels to compare outputs against
oiml_tensor* oiml_opt_loss(oiml_opt_context_t opt_ctx);// scalar tensor that contains the loss
oiml_tensor* oiml_opt_pred(oiml_opt_context_t opt_ctx);// predictions made by outputs
oiml_tensor* oiml_opt_ncorrect(oiml_opt_context_t opt_ctx);// number of matching predictions between outputs and labels

oiml_tensor* oiml_opt_grad_acc(oiml_opt_context_t opt_ctx, oiml_tensor* node);

// ====== Optimization Result ======

oiml_opt_result_t oiml_opt_result_init();
void oiml_opt_result_free(oiml_opt_result_t result);
void oiml_opt_result_reset(oiml_opt_result_t result);

// get data from result, uncertainties are optional and can be ignored by passing NULL
void oiml_opt_result_ndata(oiml_opt_result_t result, int64_t* ndata);// writes 1 value, number of datapoints
void oiml_opt_result_loss(oiml_opt_result_t result, double* loss, double* unc);// writes 1 value
void oiml_opt_result_pred(oiml_opt_result_t result, int32_t* pred);// writes ndata values
void oiml_opt_result_accuracy(oiml_opt_result_t result, double* accuracy, double* unc);// writes 1 value

// ====== Computation ======

// do forward pass, increment result if not NULL
void oiml_opt_forward(oiml_opt_context_t opt_ctx, oiml_opt_result_t result);

// do forward pass, increment result if not NULL, do backward pass
void oiml_opt_forward_backward(oiml_opt_context_t opt_ctx, oiml_opt_result_t result);

// ############################################################################
// ## The high-level functions start here. They do not depend on any private ##
// ## functions or structs and can be copied to and adapted for user code.   ##
// ############################################################################

// ====== Intended Usage ======
//
// 1. Select the appropriate loss for your problem.
// 2. Create a dataset and set the data for the "data" tensor. Also set the "labels" tensor if your loss needs them.
//    Setting the shard size to 1 will be fine, it's the granularity with which data is shuffled/loaded (bigger values are faster).
// 3. Create a OIML graph for your model with no_alloc == true. Use two separate contexts for the tensors.
//    The first context should contain the model parameters and inputs and be allocated statically in user code.
//    The second context should contain all other tensors and will be (re)allocated automatically.
//    Due to this automated allocation the data of the second context is not defined when accessed in user code.
//    Note that the second dimension of the inputs/outputs are interpreted as the number of datapoints in those tensors.
// 4. Call oiml_opt_fit. If you need more control you can use oiml_opt_epoch instead.

// signature for a callback while evaluating opt_ctx on dataset, called after an evaluation
typedef void (*oiml_opt_epoch_callback)(bool train,// true after training evaluation, false after validation evaluation
	oiml_opt_context_t opt_ctx, oiml_opt_dataset_t dataset,
	oiml_opt_result_t result,// result associated with the dataset subsection
	int64_t ibatch,// number of batches that have been evaluated so far
	int64_t ibatch_max,// total number of batches in this dataset subsection
	int64_t t_start_us);// time at which the evaluation on the dataset subsection was started

// do training on front of dataset, do evaluation only on back of dataset
void oiml_opt_epoch(oiml_opt_context_t opt_ctx, oiml_opt_dataset_t dataset,
	oiml_opt_result_t result_train,// result to increment during training, ignored if NULL
	oiml_opt_result_t result_eval,// result to increment during evaluation, ignored if NULL
	int64_t idata_split,// data index at which to split training and evaluation
	oiml_opt_epoch_callback callback_train, oiml_opt_epoch_callback callback_eval);

// callback that prints a progress bar on stderr
void oiml_opt_epoch_callback_progress_bar(bool train, oiml_opt_context_t opt_ctx, oiml_opt_dataset_t dataset, oiml_opt_result_t result, int64_t ibatch, int64_t ibatch_max,
	int64_t t_start_us);

// fit model defined by inputs and outputs to dataset
void oiml_opt_fit(oiml_backend_sched_t backend_sched,// backend scheduler for constructing the compute graphs
	oiml_context* ctx_compute,// context with temporarily allocated tensors to calculate the outputs
	oiml_tensor* inputs,// input tensor with shape [ne_datapoint, ndata_batch]
	oiml_tensor* outputs,// output tensor, must have shape [ne_label, ndata_batch] if labels are used
	oiml_opt_dataset_t dataset,// dataset with data and optionally also labels
	enum oiml_opt_loss_type loss_type,// loss to minimize
	oiml_opt_get_optimizer_params get_opt_pars,// callback to get optimizer params, userdata is pointer to epoch (of type int64_t)
	int64_t nepoch,// how many times the dataset should be iterated over
	int64_t nbatch_logical,// datapoints optimizer step, must be a multiple of ndata_batch in inputs/outputs
	float val_split,// fraction of the dataset to use for validation, must be in [0.0f, 1.0f)
	bool silent);// whether or not info prints to stderr should be suppressed
