# MKAngel MNEMO-Native NLG Engine + TARDAI Integration

**Date**: 2026-03-12
**Status**: Approved
**Scope**: Both MKAngel (mobile guardian) and TARDAI (desktop power tool + orchestrator)

---

## Core Principle

The GLM thinks in MNEMO, speaks in any language. Natural language exists only at input/output boundaries. All internal processing — derivation, attention, routing, evidential marking — operates on MNEMO glyph sequences.

```
USER INPUT (English/any)
    -> Encode -> MNEMO substrate (input boundary)
        -> Grammar derivation in MNEMO
        -> Attention computation in MNEMO
        -> Evidential marking in MNEMO
    -> Decode -> OUTPUT (English/any)
```

## Why MNEMO Internally

| Benefit     | Mechanism                                      | Impact                        |
|-------------|------------------------------------------------|-------------------------------|
| Compression | 270 glyphs encode concepts taking 5-15 words   | ~10x shorter sequences        |
| Speed       | Attention is O(n^2); 10x shorter = ~100x faster | Pure-Python viable on mobile  |
| Security    | Internal state is not human-readable            | Computational obscurity       |
| Multilingual| MNEMO is language-agnostic; only decoder changes| One engine, any output language|

## Architecture

### The MNEMO Substrate (Extended)

270 glyphs across 8 tiers, extended from TARDAI's original 150:

| Tier | Name               | Original | Added | Total | Added Coverage                                           |
|------|--------------------|----------|-------|-------|----------------------------------------------------------|
| I    | Ontological Roots  | 20       | +30   | 50    | Math primitives, chemical elements, phonemes              |
| II   | Process Verbs      | 20       | +20   | 40    | derive, bond, transform, inflect, recurse, etc.          |
| III  | State Markers      | 15       | +10   | 25    | Evidential: observed, inferred, reported, speculative, counterfactual |
| IV   | Relational Ops     | 20       | +15   | 35    | Grammatical: governs, agrees, selects, binds, etc.       |
| V    | Epistemic Markers  | 15       | +10   | 25    | Confidence: certain, probable, possible, unlikely, unknown|
| VI   | Temporal/Causal    | 20       | +10   | 30    | Derivation-depth, strange-loop indicators                 |
| VII  | Scale/Domain       | 20       | +15   | 35    | Domain-switching glyphs ("now in chemistry space")        |
| VIII | Meta/Syntax        | 20       | +10   | 30    | Composition operators for response grammar                |
| **Total** |               | **150**  | **+120** | **270** |                                                       |

### Evidentiality (The True Timeline)

Evidential marking is grammatically REQUIRED — not metadata, not decoration. Every MNEMO derivation sequence must contain at least one Tier III glyph (source) and one Tier V glyph (confidence).

**Three axes, all encoded as MNEMO glyphs:**

| Axis       | Glyphs                                                          | Grammar Role    |
|------------|-----------------------------------------------------------------|-----------------|
| Source     | obs (direct observation), inf (inference), comp (computation), rep (report), trad (tradition), spec (speculation), ctr (counterfactual) | HOW you know    |
| Confidence | cert (.95+), prob (.7-.95), poss (.4-.7), unl (.1-.4), unk (<.1)| HOW SURE you are|
| Temporal   | ver_past, obs_pres, pred_fut, hyp, timeless                    | WHEN it's true  |

**Surface realisation per language:**

| Language | Evidential Strategy              | Example (inference + probable)         |
|----------|----------------------------------|----------------------------------------|
| English  | Lexical hedging                  | "The evidence suggests that..."        |
| Turkish  | Morphological (-mIs vs -DI)      | Native evidential suffixes             |
| Quechua  | Morphological (-mi/-si/-cha)     | Native three-way evidential            |
| Japanese | Sentence-final particles         | ~rashii, ~sou da                       |
| German   | Konjunktiv I                     | Subjunctive mood for reported speech   |

**StrangeLoop**: evidentiality -> confidence calibration -> source verification -> evidentiality

### Security Hardening

1. **Rotating glyph mappings**: Concept-to-glyph mapping can be session-specific. Same grammar, different encoding per session.

2. **Grammar-verified integrity**: Every incoming MNEMO sequence must parse as valid grammar. Sequences that fail parse are rejected before reaching the attention model.

```
INCOMING MNEMO SEQUENCE
    |
    +-- Parses as valid grammar? -> YES -> Process
    +-- Fails parse? -> REJECT (potential injection)
```

### The NLG Pipeline

**Three-stage:**

1. **ENCODE** (`glm/nlg/encoder.py`): Natural language -> MNEMO substrate
   - Uses existing domain grammars (syntactic, morphological, etc.) to parse input
   - Maps parsed structures to MNEMO glyph sequences
   - Attaches domain routing metadata

2. **PROCESS** (existing `glm/model/` + `glm/grammars/`): All-MNEMO derivation
   - Domain grammar derivation in MNEMO space
   - Cross-domain composition via response grammar (8th domain)
   - TemporalAttention scoring of candidate derivations
   - Evidential marking (mandatory)

3. **DECODE** (`glm/nlg/decoder.py`): MNEMO -> Natural language
   - Language-tagged surface templates with typed slots
   - Reverse-production from derivation tree to template selection
   - Evidential markers -> language-appropriate realisation
   - Multiple candidates scored; best selected

### The Response Grammar (8th Domain)

New substrate: `Utterance` with productions that compose from other 7 domains.

```
Utterance -> DomainAnalysis EvidentialMarker
DomainAnalysis -> MathResult | LinguisticResult | BiologyResult | ...
DomainAnalysis -> DomainAnalysis Conjunction DomainAnalysis  (multi-domain)
EvidentialMarker -> SourceGlyph ConfidenceGlyph TemporalGlyph
```

StrangeLoop: response -> domain analysis -> cross-domain insight -> response

### The Conductor Pipeline (Updated)

MKAngel's AngelConductor 9-stage pipeline, stage 5 rewritten:

```
Stage 1: Detect language (existing Tongue module)
Stage 2: Compliance check (existing)
Stage 3: Route intent (existing Router, now feeds domain selection)
Stage 4: Perceive input (existing Senses, now feeds MNEMO encoder)
Stage 5: GENERATE via MNEMO NLG engine:
  5a. ENCODE: Parse input through domain grammars -> MNEMO
  5b. PROCESS: Cross-domain derivation via response grammar (in MNEMO)
  5c. PROCESS: Attach evidential markers (source + confidence + temporal)
  5d. PROCESS: Score candidates via TemporalAttention (in MNEMO)
  5e. DECODE: Reverse-produce through language-tagged templates
  5f. DECODE: Select best candidate
Stage 6-9: Post-process, format, record, check (existing)
```

## MKAngel File Map

### New Files

| File                              | Purpose                                      | ~Lines |
|-----------------------------------|----------------------------------------------|--------|
| `glm/core/mnemo_substrate.py`     | Unified 270-glyph MNEMO substrate            | 400    |
| `glm/nlg/__init__.py`            | Module init                                  | 10     |
| `glm/nlg/encoder.py`             | Natural language -> MNEMO (input boundary)   | 300    |
| `glm/nlg/decoder.py`             | MNEMO -> natural language (output boundary)  | 350    |
| `glm/nlg/realiser.py`            | Reverse-production engine (MNEMO-internal)   | 300    |
| `glm/nlg/response_grammar.py`    | 8th domain: Utterance substrate + productions| 400    |
| `glm/nlg/templates/__init__.py`  | Template registry                            | 50     |
| `glm/nlg/templates/en.py`        | English surface templates (all 7 domains)    | 500    |

### Modified Files

| File                  | Change                                          | ~Delta |
|-----------------------|-------------------------------------------------|--------|
| `app/providers.py`    | Add NLGProvider class using MNEMO NLG engine    | +150   |
| `app/conductor.py`    | Rewire stage 5 to NLG pipeline                  | +80    |
| `app/chat.py`         | Update fallback chain to prefer NLGProvider     | +30    |

### Total: ~2,570 new lines + ~260 modified lines

## TARDAI Integration

### Bridge Architecture

TARDAI (TypeScript) communicates with MKAngel GLM (Python) via JSON-over-stdio:

```
TARDAI (TypeScript) <-> glm-bridge <-> MKAngel GLM (Python)
                          |
                     JSON protocol:
                     { encode, decode, derive, route, realise, mnemo }
```

### New Files (TARDAI)

| File                          | Purpose                                   | ~Lines |
|-------------------------------|-------------------------------------------|--------|
| `lib/glm-bridge/index.ts`    | GLM child process manager                 | 150    |
| `lib/glm-bridge/protocol.ts` | JSON protocol types                       | 80     |
| `lib/glm-bridge/client.ts`   | Async request/response client             | 200    |
| `glm_server.py`              | Python stdin/stdout JSON server for GLM   | 150    |

### Replaced Modules (TARDAI)

| Current Module              | Replaced By              | Reason                                  |
|-----------------------------|--------------------------|----------------------------------------|
| `TimelordRouter` regex      | GLM grammar-space routing| Grammar structures > regex patterns     |
| `MnemoEncoder/Decoder`      | GLM native MNEMO substrate| Single source of truth for MNEMO       |
| `GhostSelector` heuristic   | GLM attention persona selection| Attention model learns ghost fit   |
| `forge/comprehend.ts` (API) | GLM native comprehension | No API dependency for skill ingestion   |
| `system_self.py` standalone | Integrated into EmbeddingSpace| Hardware state as another substrate |

### Preserved (TARDAI)

- Plugin system (`plugin-host.ts`) — MKAngel becomes plugin AND core
- UI layer (React) — aesthetics unchanged
- Electron wrapper — desktop power tool
- Cowork plugin — gets GLM-powered generation instead of Ollama
- Wins tracker — CRM, unchanged
- Ghost library — personas preserved, selection mechanism upgraded

### Orchestrator Layer

TARDAI as multi-agent coordinator uses an extended response grammar:

```
Orchestration -> AgentDispatch+ ResultComposition EvidentialMarker
AgentDispatch -> DomainGrammar AgentInstance
ResultComposition -> Merge | Sequence | Parallel | Conflict_Resolution
```

Each GLM instance runs a domain grammar. The orchestration grammar coordinates timing and composition. TemporalAttention handles cross-agent synchronisation.

## Design Principles

1. **MNEMO is the lingua franca** — all internal processing, both projects
2. **Natural language at boundaries only** — encode on input, decode on output
3. **Truth is grammatical** — evidentiality is required, not optional
4. **Grammar gate for security** — invalid MNEMO sequences rejected structurally
5. **DNA analogy** — 270 glyphs + combinatorial grammar = unlimited expressiveness
6. **Shared engine, divergent UX** — MKAngel lean/mobile, TARDAI full/desktop
7. **No API crutch** — the GLM IS the engine, not a wrapper around Claude
