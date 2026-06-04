# FSDP Causal Profiler

This project studies performance stalls in FSDP distributed training.

The main goal is to convert GPU/CPU/NCCL timeline traces into causal graphs that explain why GPU compute becomes idle during training execution.

The profiler uses CUPTI-based tracing to collect CUDA Runtime API, CUDA Driver API, and NVTX range information. These low-level traces are used as evidence for detecting compute idle gaps, communication delays, and rank skew during FSDP execution.

## Current Focus

The first target is FSDP backward execution, because backward contains important overlap patterns between compute, all-gather, and reduce-scatter communication.

However, the profiler is not intended to be limited only to backward. The long-term goal is to support broader FSDP execution analysis, including forward, backward, and full training-step timelines.

## Current Scope

- FSDP training
- Transformer-like language models

Initial MVP model: custom TinyGPT-style Transformer LM
Decoder-only Transformer architecture
Repeated Transformer blocks with self-attention and MLP layers
Synthetic token inputs and labels for controlled experiments
FSDP wrapping at the Transformer block level
- Initial focus on backward pass
- All-gather and reduce-scatter events
- 2 to 4 GPUs
- Compute idle gap attribution
- Rank skew detection
- Simple causal graph construction

## Research Question

When GPU compute becomes idle during FSDP execution, can we attribute the idle gap to delayed collective communication and identify which rank, dependency, or collective caused the delay?

## Current Milestone

MVP:

1. Collect CUDA/NVTX/CUPTI traces
2. Detect compute idle gaps
3. Label FSDP all-gather and reduce-scatter events
4. Connect idle gaps to communication events
5. Detect rank skew in collective communication
6. Build a simple causal graph

## Initial MVP Boundary

For the MVP, we focus on backward execution first.

This is a practical starting point because FSDP backward often contains overlapping compute and communication, making it a good target for causal stall analysis.

After the backward MVP works, the profiler can be extended to forward pass and full training-step analysis.
