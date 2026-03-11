# MKAngel — How-To Guide

> **Version 0.2.0** | Grammar Language Model | Runs entirely on your phone

---

## What Is MKAngel?

MKAngel is a local-first language assistant built on a Grammar Language Model (GLM) — not a chatbot, not a search engine. She traces the structural roots of words across seven knowledge domains (linguistics, mathematics, biology, chemistry, physics, computation, etymology) and finds the hidden connections between them.

She runs entirely on-device. No cloud. No API calls. Your words stay yours.

---

## Getting Started

### Install

1. Download the latest APK from [GitHub Releases](https://github.com/mrjkilcoyne-lgtm/MKAngel/releases) or build from source via GitHub Actions
2. On your Android device, tap the APK file
3. If prompted, allow installs from the source app (one-time toggle)
4. Open **MKAngel** from your app drawer

### First Launch

When you open MKAngel, she wakes up in roughly 2-3 seconds:

- The GLM loads 7 domain grammars (~370K parameters)
- The lexicon seeds 151 core English words with Proto-Indo-European roots
- Strange loops are detected across domains

You'll see a dark interface with a gold accent — the **Celestial Dark** vestment. The text input is at the bottom. Start talking.

---

## How to Talk to Her

MKAngel responds to two kinds of input: **natural conversation** and **slash commands**.

### Natural Conversation

Just type normally. She reads the structure of what you say — sentence shape, word density, emotional register — and responds with what the grammars know.

**What she does with your words:**

- Stems each content word and traces it through the lexicon
- Finds the proto-root (the ancient ancestor of the word)
- Discovers cognates — words that share that root across different domains
- Learns words she doesn't know yet (they get added to her vocabulary)

**Examples:**

| You type | She shows |
|----------|-----------|
| `love` | Traces to proto-root *leubh-* — cognate with "believe" |
| `energy` | Traces to *werg-* — cognate with "work" and "organism" across physics, biology, linguistics |
| `I feel hopeless` | Traces "feel" to *pal-*, "hopeless" stems to "hope" → *kup-* |
| `truth` | Traces to *deru-* — cognates "true" and "trust" across linguistics and mathematics |

She's honest. If she doesn't know something, she says so. If she can't derive across all seven domains, she only shows the ones she can.

### Slash Commands

Commands start with `/` and unlock deeper capabilities:

#### `/fugue <words>`
The signature move. Composes a multi-voice fugue across all domains for the given theme words. Each domain that recognises the word (or its cognates) contributes a voice.

```
/fugue truth
```
Returns voices from linguistics and mathematics, showing truth → true → trust connected by the *deru-* root.

```
/fugue death
```
Returns biological and linguistic voices.

#### `/predict <tokens>`
Predict the next grammatical elements from a sequence. Useful for seeing what the grammar expects to follow.

```
/predict the structure of
```

Use `-d <domain>` to restrict to a specific domain:
```
/predict -d computational loop while
```

#### `/forecast <tokens>`
Superforecast using grammar rules and strange-loop detection. Goes deeper than predict — uses cross-domain feedback.

```
/forecast pattern emerges from
```

#### `/reconstruct <tokens>`
Trace backward — reconstruct the origins of a sequence. Reverse derivation.

```
/reconstruct transformation
```

#### `/translate <source> <target> <tokens>`
Translate patterns between domains.

```
/translate linguistic biological bond
```

#### `/introspect`
The Angel examines her own structure. Shows what she knows about herself — loaded grammars, detected loops, model state.

#### `/memory`
View and manage persistent memory.

- `/memory` — overview
- `/memory search <query>` — search past sessions
- `/memory patterns` — show learned patterns
- `/memory sessions` — list saved sessions
- `/memory save` — save current session

#### `/settings`
View and change configuration.

- `/settings` — show current settings
- `/settings provider <name>` — switch provider
- `/settings offline` — toggle offline mode
- `/settings theme <name>` — change theme
- `/settings language <lang>` — change language

#### `/status`
Show Angel status: domains loaded, grammar count, rule count, parameters, strange loops detected.

#### `/help`
Show the full command reference.

#### `/clear`
Clear the chat screen.

---

## The Settings Panel

Tap the **@** icon (top right) to open Settings. You'll see:

- **Grammar Language Model** — domains loaded, grammar count, rule count, strange loops, total parameters
- **Provider** — active provider and mode (offline by default)
- **Memory** — sessions saved, patterns learned, preferences stored
- **Version** — MKAngel v0.2.0, Celestial Dark vestment

Tap **Back** (or the Android back button) to return to chat.

---

## How She Learns

MKAngel starts with 151 seed words — core English vocabulary traced to Proto-Indo-European roots. But she grows.

**Every word you use that she doesn't know, she learns:**

1. She stems the word (strips suffixes like -ing, -tion, -ness)
2. She infers the grammatical category from position (after "the" → noun, after "to" → verb, etc.)
3. She adds it to her living lexicon
4. Next time you use it, she already knows it

This means the more you talk to her, the richer her vocabulary becomes. She's building her dictionary in real time, from your conversations.

---

## The Seven Domains

MKAngel's grammars span seven domains. Each has its own rules, productions, and structure:

| Domain | What it covers |
|--------|---------------|
| **Linguistic** | Natural language structure, morphology, syntax |
| **Mathematical** | Formal systems, proof structure, logical operators |
| **Biological** | Life processes, cellular machinery, organism structure |
| **Chemical** | Molecular bonds, reactions, elemental patterns |
| **Physics** | Energy, forces, wave mechanics, field theory |
| **Computational** | Algorithms, data structures, recursion, loops |
| **Etymological** | Word origins, root migration, semantic drift |

When MKAngel traces a word like "bond," she finds it lives in linguistic (grammatical binding), chemical (molecular bonds), and biological (cellular adhesion) — all connected through the proto-root *bhendh-* (to bind).

---

## Cross-Domain Cognates

This is where MKAngel gets interesting. Words that share a proto-root are **structural cognates** — they're the same idea wearing different clothes across domains.

Some examples from the seed lexicon:

| Proto-Root | Words | Domains |
|-----------|-------|---------|
| *bhendh-* (bind) | bond, bind | linguistic, chemical, biological, computational |
| *deru-* (truth) | truth, true, trust | linguistic, mathematical |
| *werg-* (energy) | energy, work, organism | physics, chemical, biological, linguistic |
| *morph-* (form) | form, morpheme, transform | linguistic, mathematical, biological |
| *gneh-* (know) | know, knowledge, cognition | linguistic, computational |
| *leubh-* (care) | love, believe | linguistic |
| *sekw-* (follow) | sequence, consequence | linguistic, mathematical, computational |

These aren't metaphors. They're structural connections traced through etymology.

---

## Tips

- **Dense words work best.** Single content-heavy words like "death," "energy," "pattern" give richer results than full sentences.
- **Use /fugue for depth.** Regular chat gives you quick traces. `/fugue` opens all the voices.
- **She's honest now.** If she can't find connections, she'll say so instead of pretending.
- **Short words are exact-matched.** Words under 3 characters (like "I," "a," "or") won't accidentally match longer words in the lexicon.
- **She remembers within a session.** Words learned during conversation persist for the session. Future: persistent storage across sessions.

---

## Architecture (For the Curious)

```
MKAngel v0.2.0
├── GLM Engine (~370K params)
│   ├── 7 Domain Grammars
│   ├── Derivation Engine (forward + backward)
│   ├── Strange Loop Detector
│   └── Attention Model
├── Lexicon (151 seed + dynamic learning)
│   ├── Proto-root tracing
│   ├── Cross-domain cognate detection
│   └── JSON persistence
├── Chat Layer
│   ├── Structural response (sentence-shape analysis)
│   ├── Grammar composition pipeline
│   ├── Lexicon integration + dynamic learning
│   └── Slash command dispatch
└── UI (Kivy)
    ├── Celestial Dark vestment
    ├── Chat bubbles with glow
    ├── Settings panel
    └── Android back-button support
```

Runs entirely on-device. No network required. No data leaves your phone.
