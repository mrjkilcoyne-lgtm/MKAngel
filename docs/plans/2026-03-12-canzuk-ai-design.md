# CANZUK-AI — Design Document

**Date:** 2026-03-12
**Status:** Approved
**Scope:** Android AI assistant app powered by MKAngel's Grammar Language Model

## 1. Vision

CANZUK-AI is a native Android AI assistant that feels like GPT, Claude, Grok, or Gemini — but runs on a Grammar Language Model instead of a trillion-parameter neural network. The grammar engine produces long-form, expert-level, structurally validated responses by walking derivation trees across 7 knowledge domains. Every sentence has a grammar-backed reason to exist.

**Name:** CANZUK-AI
**Tagline:** Think Different. Think Deeper.

## 2. Visual Identity

### Color System — The Union Palette

| Token | Hex | Role |
|-------|-----|------|
| CANZUK RED | #C8102E | Action, emphasis, the living pulse |
| CANZUK BLUE | #012169 | Depth, trust, structural bone |
| WHITE | #FAFAFA | Clarity, space, breath between |
| NAVY SURFACE | #0A1128 | Dark mode ground |
| SILVER | #B8C4D0 | Secondary text, borders |
| GOLD ACCENT | #D4A847 | Rare: prophecy, strange loops, milestones |

### Modes

- **Dark (default):** Navy surface #0A1128, white text, red actions, blue structure
- **Light:** White ground, navy text, red/blue accents preserved

### Typography

- **Display:** Inter or Plus Jakarta Sans — clean, geometric, timeless
- **Monospace:** JetBrains Mono — code, MNEMO, grammar output
- **Scale:** 12/14/16/20/28sp — five sizes only

### Design Principles

1. Negative space is sacred — let content breathe
2. Red means alive — pulsing indicators, active states, user actions
3. Blue means structure — grammar domains, reasoning stages, architecture
4. Gold means rare — strange loops, prophecy, milestones
5. No decoration without function — every pixel earns its place

## 3. Technical Architecture

### Stack

- **UI:** Kotlin + Jetpack Compose (Material 3)
- **Engine:** Python via Chaquopy (embedded interpreter)
- **GLM:** MKAngel's Grammar Language Model (303K parameters)
- **Storage:** SQLite (Room) for memory, conversation history
- **Build:** Gradle + Chaquopy plugin, CI via GitHub Actions

### Layer Diagram

```
┌─────────────────────────────────────────────┐
│  CANZUK-AI Android App                      │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │  UI Layer (Jetpack Compose)         │    │
│  │  Conversation · Document · Code     │    │
│  │  Canvas · Voice · Files · Cowork    │    │
│  │  Introspection · Settings           │    │
│  └────────────┬────────────────────────┘    │
│               │ Kotlin ↔ Python bridge       │
│  ┌────────────▼────────────────────────┐    │
│  │  Engine Layer (Python/Chaquopy)     │    │
│  │                                     │    │
│  │  Angel (7 grammars, 303K network)   │    │
│  │  DerivationEngine (bidirectional)   │    │
│  │  ReasoningPipeline (4-stage)        │    │
│  │  GenerativeRealiser (long-form)     │    │
│  │  MNEMO Codec (compression)          │    │
│  │  Trainer (SPSA, on-device)          │    │
│  │  Learn/Sleep (self-improvement)     │    │
│  │  Memory · Voice · Web · Tools       │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### Kotlin-Python Bridge (Chaquopy)

```kotlin
// Kotlin side
val engine = Python.getInstance().getModule("glm.bridge")
val response = engine.callAttr("process", userInput)

// Python side (glm/bridge.py)
def process(text: str) -> Iterator[str]:
    """Stream response tokens from grammar engine."""
    pipeline = ReasoningPipeline()
    result = pipeline.run(text)
    for token in realiser.stream(result):
        yield token
```

## 4. The Generative Realiser

This is the new module that makes CANZUK-AI produce frontier-quality output.

### How Long-Form Generation Works

1. **Input** → Router detects domain(s) and intent
2. **Derivation engine** runs forward derivation → produces derivation tree with potentially hundreds of branches across multiple grammars
3. **Reasoning pipeline** structures the tree:
   - Skeleton extracts claims (S→R→O triples)
   - DAG maps dependencies between claims
   - Disconfirm prunes weak/circular branches
   - Synthesis identifies all valid paths
4. **Generative Realiser** walks the validated tree, streaming natural language:
   - Each tree node → a paragraph or section
   - Branch points → connectives ("Furthermore...", "In contrast...", "This connects to...")
   - Cross-domain isomorphisms → analogies and explanations
   - Strange loops → recursive deepening ("To understand why, we need to revisit...")
   - Evidential markers → confidence tracking ("This follows necessarily..." vs "This suggests...")
   - Output grows as long as the tree is deep

### Key Insight

LLMs generate text by predicting the next token statistically. CANZUK-AI generates text by walking a structurally validated derivation tree. Every sentence has a grammar-backed reason to exist. Output can be enormous — but nothing is hallucinated, because every branch was derived from rules.

### Streaming

The realiser yields tokens as it walks the tree. The Compose UI renders text appearing in real-time via Kotlin coroutines collecting from the Python generator — identical UX to ChatGPT/Claude streaming.

## 5. Screens & Capabilities

### Navigation

Bottom bar: **Chat** | **Document** | **Code** | **Canvas** | **More**
More expands to: Voice, Files, Cowork, Introspection.

### Screen 1: Conversation (Home)

- Full chat interface with streaming responses
- Red send button, blue structural indicators
- Long responses render with section headers, collapsible reasoning
- Markdown: bold, italic, code blocks, tables, lists, LaTeX
- Message actions: copy, share, regenerate, show reasoning tree
- Conversation history with search

### Screen 2: Document Mode

- Long-form generation: essays, reports, analysis, plans
- Side-by-side: output left, derivation tree right (optional)
- Export: PDF, Markdown, DOCX
- Grammar domains shown as blue chips at section boundaries
- Track changes, versioning

### Screen 3: Code

- Syntax-highlighted editor (JetBrains Mono)
- Grammar-driven code generation (computational domain)
- Run Python snippets via Chaquopy
- Explain, refactor, debug via grammar derivation
- Multi-file support, project context

### Screen 4: Canvas

- Visual reasoning: DAG visualisation, derivation trees
- Sketch input (handwriting recognition → grammar parsing)
- Diagram generation from text
- Image analysis via structural decomposition

### Screen 5: Voice

- Speech-to-text input (on-device or Android SpeechRecognizer)
- Text-to-speech output (Android TTS engine)
- Hands-free conversation mode
- Voice triggers same grammar pipeline as text

### Screen 6: Files

- Upload and analyse documents (PDF, images, text, code)
- Structural decomposition of any input
- Export conversation history
- MNEMO-compressed knowledge base browser

### Screen 7: Cowork Mode

- Multi-turn task execution with visible reasoning
- Step-by-step pipeline display (Skeleton→DAG→Disconfirm→Synthesis)
- Tool use: web search, file operations, calculations
- Grammar pipeline visible as it works in real-time

### Screen 8: Introspection

- Live model stats: parameters, grammars, rules, strange loops
- Training progress (on-device trainer status)
- Learn/Sleep cycle status and history
- Benchmark scores per domain
- Grammar domain explorer with rule browser

## 6. Capability Matrix vs Frontier LLMs

| Capability | GPT | Claude | Grok | Gemini | CANZUK-AI |
|-----------|-----|--------|------|--------|-----------|
| Text chat | Yes | Yes | Yes | Yes | Yes (grammar-driven) |
| Long-form generation | Yes | Yes | Yes | Yes | Yes (tree-guided) |
| Code generation | Yes | Yes | Yes | Yes | Yes (computational grammar) |
| Code execution | Yes | Yes | No | Yes | Yes (Chaquopy) |
| Image input | Yes | Yes | Yes | Yes | Yes (structural decomposition) |
| Voice I/O | Yes | Yes | Yes | Yes | Yes (on-device) |
| File analysis | Yes | Yes | Yes | Yes | Yes |
| Web search | Yes | Yes | Yes | Yes | Yes |
| Document export | Limited | Limited | No | Limited | Full (PDF/MD/DOCX) |
| Reasoning transparency | Hidden | Hidden | Hidden | Hidden | Visible (derivation tree) |
| On-device training | No | No | No | No | Yes (SPSA on TPU) |
| Self-improvement | No | No | No | No | Yes (Learn/Sleep cycles) |
| Offline mode | No | No | No | No | Full (grammar engine is local) |
| Hallucination | Statistical | Statistical | Statistical | Statistical | Structural (grammar-validated) |

## 7. Unique Differentiators

1. **No hallucination by design** — output follows derivation rules, not statistical prediction
2. **Visible reasoning** — user can see the derivation tree, not a black box
3. **Self-improving** — Learn/Sleep cycles evolve grammar rules through use
4. **Trainable on-device** — 303K params trains on phone TPU in minutes
5. **Fully offline** — works without internet, no API dependency
6. **7-domain expertise** — math, linguistics, biology, chemistry, physics, computation, etymology
7. **MNEMO compression** — knowledge compressed to orders of magnitude smaller than LLM weights

## 8. Build & Distribution

- **Build system:** Android Gradle + Chaquopy plugin
- **CI:** GitHub Actions (same workflow pattern as current MKAngel buildozer CI)
- **Target:** Android 7+ (API 24), arm64-v8a
- **APK size estimate:** ~80-120MB (Chaquopy Python runtime + GLM)
- **Distribution:** GitHub Releases initially, Google Play Store when stable
- **Device:** Pixel 10 Pro XL (primary test device)

## 9. Migration from MKAngel

- All `glm/` code carries over unchanged (pure Python)
- All new modules carry over: `glm/pipeline/`, `glm/training/`, `glm/benchmark/`
- `app/` layer (Kivy-specific) is replaced entirely by Kotlin/Compose
- `app/memory.py` logic migrates to Room/SQLite in Kotlin
- `app/providers.py` routing logic migrates to Kotlin ViewModel
- New: `glm/bridge.py` — Python entry point for Chaquopy calls
- New: `glm/realiser_v2.py` — Generative Realiser for long-form output

## 10. Risk & Mitigation

| Risk | Severity | Mitigation |
|------|----------|------------|
| Chaquopy Python startup latency | Medium | Warm start on app launch, show splash while loading |
| Response quality vs frontier LLMs | High | Progressive stages: grammar pre-processing → hybrid → pure |
| APK size with embedded Python | Medium | ProGuard, selective Chaquopy stdlib, tree-shake unused modules |
| Long-form coherence over many paragraphs | High | Derivation tree enforces structure; realiser tracks narrative state |
| Two-language codebase complexity | Medium | Clear boundary: Kotlin = UI + lifecycle, Python = all reasoning |
