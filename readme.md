# LLM from Scratch: Inside the Black Box

## Overview
This repository contains a ground-up implementation of a Generative Pre-trained Transformer, specifically modeled after the GPT-2 architecture. The purpose of this project is to strip away the abstractions of high-level PyTorch wrappers and build the core mathematical and infrastructural components of a Large Language Model from scratch. 

It covers the complete pipeline required to train and run an LLM, proving how raw parameters conspire to predict the next token.

## Repository Structure

* **`Tokenizer_script.ipynb`** Implementation of a production-grade Byte-Pair Encoding (BPE) tokenizer. Demonstrates handling of byte-level encoding, regex-based pre-tokenization (meat cleaver approach), and special token VIP injection to prevent fragmentation.
* **`Data_pipeline_from_scratch.ipynb`**
  An optimized data ingestion pipeline designed for high-throughput training. Explores text normalization, fixed-length sequence packing, and randomized batching to prevent GPU starvation.
* **`Building_GPT_from_Basics.ipynb`** The core Transformer architecture. Includes the raw mathematical implementation of multi-head self-attention, positional embeddings, pre-norm layer normalization, residual connections, and the weight-tied output projection layer.

## Key Engineering Concepts Explored
* **Autoregressive Generation:** Next-token prediction mechanics and causal masking to prevent forward-looking data leakage during the forward pass.
* **Attention Mechanisms:** The linear algebra behind Query, Key, and Value (QKV) matrices, including scaling factors (`1/sqrt(d_k)`) to prevent softmax saturation.
* **Inference Optimization:** The hardware realities of generating text, including the architectural distinction between compute-bound (prefill) and memory-bound (decode) phases, and the necessity of the KV Cache.

## Getting Started
1. Clone the repository:
   ```bash
   git clone [https://github.com/SharvChopra/LLM_Code.git](https://github.com/SharvChopra/LLM_Code.git)This file contains the code for LLMs Basics
