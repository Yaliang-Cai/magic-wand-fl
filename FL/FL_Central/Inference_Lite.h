#ifndef INFERENCE_LITE_H
#define INFERENCE_LITE_H

#include <math.h>
#include <stdlib.h>

// --- 1. Basic Macro Definitions ---
#define DATA_TYPE float
// Simple ReLU
#define ACT(a) ( (a) > 0 ? (a) : 0 ) 
// Simple Exp (Standard exp is sufficient if not dealing with extreme numerical ranges)
#define EXP(a) exp(a)

// --- 2. Struct Definitions (Unchanged) ---
typedef struct neuron_t {
  int numInput;
  DATA_TYPE* W; // Weights
  DATA_TYPE B;  // Bias
  DATA_TYPE X;  // Output value (Activation)
  // ⚠️ Removed dW, dA, dB (Only needed for training)
} neuron;

typedef struct layer_t {
  int numNeuron;
  neuron* Neu;
} layer;

// --- 3. Global Variables ---
layer* L = NULL; 
size_t numLayers;
const unsigned int* net_def_ptr; // Pointer to network structure array

// --- 4. Helper Functions ---
// Calculate neuron activation value (Forward only)
DATA_TYPE AccFunction(unsigned int layerIndx, int nodeIndx) {
  DATA_TYPE A = 0;
  for (int k = 0; k < net_def_ptr[layerIndx - 1]; k++) {
    // W * X
    A += L[layerIndx].Neu[nodeIndx].W[k] * L[layerIndx - 1].Neu[k].X;
  }
  // + B
  A += L[layerIndx].Neu[nodeIndx].B;
  return A;
}

// --- 5. Core: Forward Inference (Forward Prop) ---
// ⚠️ Note: Pass an output array here instead of using global variable y[] to save memory
void run_inference(DATA_TYPE* input_vec, DATA_TYPE* output_vec) {
  int maxIndx = 0;
  
  // Iterate through layers
  for (unsigned int i = 0; i < numLayers; i++) {
    if (i == 0) {
      // First layer: Copy input directly
      for (unsigned int j = 0; j < net_def_ptr[0]; j++) {
        L[i].Neu[j].X = input_vec[j];
      }
    } else {
      // Subsequent layers: Calculate weighted sum + Activation
      for (unsigned int j = 0; j < net_def_ptr[i]; j++) {
        DATA_TYPE val = AccFunction(i, j);
        
        // Last layer is usually Logits before Softmax; intermediate layers use ReLU
        if (i == numLayers - 1) {
           L[i].Neu[j].X = val; // Keep raw value for last layer, apply Softmax later
        } else {
           L[i].Neu[j].X = ACT(val); // ReLU
        }
      }
    }
  }

  // --- Softmax (Applied only to the last layer) ---
  int lastLayer = numLayers - 1;
  int outSize = net_def_ptr[lastLayer];
  
  // Find max value to prevent overflow
  DATA_TYPE max_val = L[lastLayer].Neu[0].X;
  for(int j=1; j<outSize; j++) {
      if(L[lastLayer].Neu[j].X > max_val) max_val = L[lastLayer].Neu[j].X;
  }

  DATA_TYPE sum = 0;
  for(int j=0; j<outSize; j++) {
      output_vec[j] = EXP(L[lastLayer].Neu[j].X - max_val);
      sum += output_vec[j];
  }
  // Normalize
  for(int j=0; j<outSize; j++) {
      output_vec[j] /= sum;
  }
}

// --- 6. Initialization & Weight Loading ---
// Universal loader: Fills the struct with data from the 1D saved array (SAVED_WEIGHTS)
void setup_head(const unsigned int* definition, int definition_size, const float* weights_buffer) {
  net_def_ptr = definition;
  numLayers = definition_size;
  
  // 1. Create Network Structure
  L = (layer*)calloc(numLayers, sizeof(layer));
  // L[0] Input Layer
  L[0].numNeuron = net_def_ptr[0];
  L[0].Neu = (neuron*)calloc(net_def_ptr[0], sizeof(neuron));

  // L[1]...L[N] Hidden Layers and Output Layer
  for (unsigned int i = 1; i < numLayers; i++) {
    L[i].numNeuron = net_def_ptr[i];
    L[i].Neu = (neuron*)calloc(net_def_ptr[i], sizeof(neuron));
    for (unsigned int j = 0; j < net_def_ptr[i]; j++) {
       L[i].Neu[j].numInput = net_def_ptr[i-1];
       L[i].Neu[j].W = (DATA_TYPE*)calloc(net_def_ptr[i-1], sizeof(DATA_TYPE));
       // Note: We no longer allocate dW (gradients), saving 50% memory!
    }
  }

  // 2. Populate Weights (Unpack)
  int ptrCount = 0;
  for (unsigned int i = 1; i < numLayers; i++) {
    for (unsigned int j = 0; j < net_def_ptr[i]; j++) {
      for (unsigned int k = 0; k < L[i].Neu[j].numInput; k++) {
        L[i].Neu[j].W[k] = weights_buffer[ptrCount++];
      }
      L[i].Neu[j].B = weights_buffer[ptrCount++];
    }
  }
}

#endif