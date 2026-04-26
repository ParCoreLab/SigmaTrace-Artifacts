# SigmaTrace - SC26 Artifact Description

This repo contains the workloads used in the evaluation section of the paper
*SigmaTrace: Library-Agnostic Observability for CPU/GPU-Initiated
RDMA Communication*. The SigmaTrace tool itself is released separately.

## Layout

| Case study | Paper section | Paper elements | AD artifact |
|---|---|---|---|
| [`Evaluation/CS1-Doorbell-Latency`](Evaluation/CS1-Doorbell-Latency) | §VI-B | Fig. 4a, 4b | A2 |
| [`Evaluation/CS2-Multi-Node-Scaling`](Evaluation/CS2-Multi-Node-Scaling) | §VI-C | Fig. 5, 6, Table I | A3 |
| [`Evaluation/CS3-Distributed-BFS`](Evaluation/CS3-Distributed-BFS) | §VI-D | Fig. 7, 8 | A4 |
| [`Evaluation/CS4-Sglang-MoE`](Evaluation/CS4-Sglang-MoE) | §VI-E | Fig. 9 | A5 |

Each case study has its own `README.md` with environment details, build
steps, and expected output.
