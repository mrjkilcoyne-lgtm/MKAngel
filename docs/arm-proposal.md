# MKAngel: On-Device Sovereign AI for Arm Silicon

**Confidential — For Discussion**
**Date: 22 April 2026**

---

**To:** Arm Ltd — Emerging Business & AI Partnerships
**From:** MK Kilcoyne, BACK Online
**Re:** Strategic partnership proposal — on-device Grammar Language Model for Arm-based consumer devices

---

## Summary

We have built a working artificial intelligence that runs entirely on-device,
requires no GPU, occupies 37 megabytes of storage, performs inference in under
100 milliseconds on Arm Cortex-A series processors, and costs the end user
approximately fifteen pence per month at scale. It is written in pure Python
with zero external dependencies.

We are seeking a strategic conversation with Arm regarding integration,
co-development, and go-to-market partnership.

---

## The Proposition

**MKAngel** is a Grammar Language Model (GLM) — a novel architecture that
achieves natural language understanding not through statistical pattern
matching over billions of parameters, but through explicit grammatical
deep structure across seven scientific domains. Coupled with a compact
Mamba-3 state space model for fluent generation, the system delivers
genuine linguistic comprehension and multi-domain reasoning on consumer
hardware — specifically, Arm silicon.

| Metric | MKAngel | Typical Cloud LLM |
|--------|---------|-------------------|
| Device footprint | 37 MB | N/A (cloud only) |
| Inference latency | ~100 ms on Arm CPU | 500–2000 ms (network + GPU) |
| Monthly cost per user | £0.12 | £16–20 |
| Offline capability | Full | None |
| User data residency | On device | Third-party servers |
| External dependencies | None | GPU cluster, API keys |

---

## Why Arm

Arm's architecture powers the overwhelming majority of the world's mobile
devices. The commercial value of demonstrating that meaningful, private,
sovereign AI can run natively on Arm — without cloud dependency, without
GPU offload, without compromising user data — is, in our respectful
assessment, considerable.

MKAngel is purpose-built for this environment:

- **Pure CPU inference.** The GLM's derivation engine and the Mamba-3 SSM
  both operate in O(n) linear time. No matrix multiplication libraries.
  No CUDA. No Metal. Arm Cortex-A is sufficient.

- **37 megabytes total.** The grammar engine (10.8 MB), the generative
  model (23.8 MB at int4 quantisation), and the user's personal data
  (lexicon, memory, skills) fit comfortably on any device manufactured
  in the past decade.

- **Privacy by architecture.** All personal data remains on the device.
  Cloud connectivity is optional and stateless — used only for swarm
  orchestration of complex tasks, hosted on British infrastructure.

---

## Technical Substance

The system is not a prototype. It is a working codebase comprising:

- **52,673 lines** of Python across 128 source files
- **313 automated tests**, all passing
- **1,004 lexicon entries** across 12 natural languages and 6 British dialects
- **24 grammars** spanning linguistics, chemistry, biology, computation,
  mathematics, and physics — with cross-domain isomorphism detection
- **370,058-parameter neural model** with self-referential strange loop
  architecture (Hofstadter-inspired)
- **Bottom-up chart parser** with heuristic POS tagging capable of
  inferring the grammatical category of previously unseen words
- **Multi-agent swarm orchestrator** with named specialist agents,
  skill composition, and iterative team reassembly
- **Full GDPR compliance module** including consent management, data
  portability (Art. 20), and right to erasure (Art. 17)

The repository is available for technical review at the discretion of
Arm's engineering team.

---

## What We Are Seeking

1. **Technical evaluation.** We invite Arm's AI and ML team to benchmark
   MKAngel on reference Arm hardware (Cortex-A76 or later) and validate
   the performance claims herein.

2. **Strategic discussion.** We wish to explore how MKAngel might be
   positioned within Arm's AI ecosystem — whether as a reference
   implementation for on-device AI, as a component of Arm's developer
   programme, or as a co-developed product.

3. **Co-investment in British AI sovereignty.** MKAngel is British-built,
   designed for British cloud hosting, and aligned with HM Government's
   stated ambition for sovereign AI capability. Arm's involvement would
   signal that British silicon and British software can deliver a credible
   alternative to US-hosted, US-controlled AI infrastructure.

---

## Contact

MK Kilcoyne
BACK Online
[Contact details to be inserted]

*This document contains proprietary information and is shared in confidence
for the sole purpose of evaluating a potential partnership with Arm Ltd.*
