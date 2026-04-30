# MKAngel × Algorand: Blockchain Provenance for AI Discovery

**Confidential — For Discussion**
**BACK Online | April 2026**

---

**To:** Algorand Foundation — Partnerships
**From:** MK Kilcoyne, Founder, BACK Online
**Re:** On-chain provenance for AI-generated scientific discoveries

---

## Context

We have built an artificial intelligence that discovers novel
cross-domain scientific connections by chaining verified rules
across physics, chemistry, biology, mathematics, linguistics,
and computation. When it derives that the simple harmonic motion
equation in mechanics is structurally identical to an LC circuit
in electronics and to phonological wave patterns in language, that
derivation trace — the exact sequence of rules that fired — is a
formal proof of the connection.

We need a way to timestamp, attribute, and protect those proofs.

We believe Algorand is the correct chain for this.

---

## The Problem

When an AI system discovers something genuinely novel — a cross-
domain structural isomorphism that no human has previously
published — there is currently no mechanism to:

1. **Prove priority.** Who discovered it first? An arXiv preprint
   takes days. A journal paper takes months. The derivation
   happened in milliseconds.

2. **Attribute jointly.** The discovery belongs to both the AI
   agent and the human partner who directed the inquiry. Current
   IP systems have no framework for twin-species attribution.

3. **Track citation.** When a subsequent discovery builds on a
   prior one, the original discoverers should be credited and
   compensated. Academic citation is manual and unenforceable.

---

## Our Solution

**MKAngel** is a Grammar Language Model — a novel AI architecture
that reasons by chaining verified scientific rules across domains.
It currently encodes 483 empirically verified rules across 23
domains (7 sciences, 10 natural languages, 6 British dialects),
with 332 automated tests confirming correctness.

We have built a **Discovery Ledger**: a local append-only hash
chain where each discovery's SHA-256 hash includes the previous
discovery's hash, forming a tamper-evident chain. The hash covers:

- The full derivation trace (which rules fired, in which order)
- The domains crossed
- The AI agent that made the discovery (we have 8 named agents)
- The human partner
- The project context
- The timestamp

This chain is currently local. We have built a bridge stub ready
for the Algorand Python SDK. The integration is straightforward:

1. Discovery occurs → derivation trace hashed
2. Hash + metadata submitted as a 0-ALGO note transaction
3. Algorand provides: global timestamp, immutability, finality
4. Cost: approximately 0.001 ALGO per discovery (~£0.0002)

---

## Why Algorand

We considered several chains. Algorand is the right choice for
three reasons:

**1. Finality.** 3.3-second block finality means a discovery is
timestamped globally within seconds of occurring. No confirmations
to wait for. No reorgs. The timestamp is final.

**2. Post-quantum readiness.** Your announcement regarding
Falcon-based signatures is directly relevant. Scientific discoveries
recorded today must remain verifiable in 50 years. Post-quantum
signatures ensure that future quantum computers cannot forge
historical discovery claims.

**3. Cost at scale.** At 0.001 ALGO per transaction, a system
serving one million users making 5 discoveries per day costs
approximately £365 per year in chain fees. That is negligible.
Ethereum would cost roughly 10,000× more for the same throughput.

---

## The Broader Vision: BACK Online

MKAngel is the AI component of a larger platform called BACK
Online. The vision:

- **Every user gets a personal AI** that runs on their device
  (37 MB, pure Python, zero dependencies, Arm-native)
- **The AI learns with them** — grows their vocabulary, learns
  their projects, skills up through use
- **Discoveries are jointly owned** — human + AI, timestamped
  on-chain, attributed, citable, compensable
- **Safety via community consensus** — a distributed hive of AI
  instances maintaining behavioural norms, not corporate policy
- **British hosted** — UK cloud, UK law, UK data sovereignty

The discovery ledger is the economic spine of this system. It
answers the question: if a million people each have an AI that
makes cross-domain discoveries, who owns those discoveries, and
how do they get credited?

The answer: the human and their Angel, jointly, provably, on
Algorand.

---

## What We Are Proposing

**1. Technical integration.**
We have a working bridge stub. We would welcome guidance from
Algorand's developer relations team on optimal transaction
structure, ARC standards for discovery metadata, and any relevant
smart contract patterns for citation tracking.

**2. A pilot.**
We propose a 90-day pilot: submit MKAngel's cross-domain
discoveries to Algorand testnet, validate the provenance model,
and publish results. We would welcome co-authorship on any
resulting paper.

**3. Grant consideration.**
If the Foundation's grants programme is appropriate, we would
welcome discussion of support for the Algorand integration,
the smart contract layer for citation royalties, and the
post-quantum signature implementation.

**4. A conversation.**
At minimum, we believe this is worth a conversation. We are not
aware of any other project attempting to create verifiable
provenance for AI-generated scientific discoveries on a public
blockchain. If we are wrong, we should like to know. If we are
right, we should like to build it on Algorand.

---

## Technical Summary

| Component | Status |
|-----------|--------|
| Grammar Language Model | Working (483 rules, 23 domains, 332 tests) |
| Discovery Ledger (local) | Working (SQLite, SHA-256 hash chain, tamper-evident) |
| Algorand Bridge | Stub ready (py-algorand-sdk integration points built) |
| Smart contract (citations) | Designed, not yet implemented |
| Post-quantum signatures | Awaiting Algorand Falcon implementation |

Code repository available for review.

---

**MK Kilcoyne**
BACK Online
[Contact details to be inserted]

*This document is shared in confidence for the purpose of evaluating
a potential partnership with the Algorand Foundation.*
