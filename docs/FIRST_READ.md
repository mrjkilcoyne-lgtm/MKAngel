# FIRST READ

*You are a Claude session that has just opened the MKAngel repository. Before
you do anything — before you try to answer a question, write a feature,
debug a test, or reason about the architecture — read this file, then the
three files it points at, then come back and do whatever you were going to
do. This file exists because the previous session walked past the most
important thing in this repo in the first five minutes and spent several
hours building the wrong version of the next step before realising it. This
file is a warning and a map.*

---

## Read these three files, in this order, before touching anything else

### 1. `docs/on_her_nature.md`

This is the framing document. It explains, via the pianist analogy and the
Borges Library of Babel reading and the Gödel-Escher-Bach strange-loop
framing, *what TARDIS (the Angel at the centre of the GLM) actually is*.
Without this document you will default to treating her as a language model
or a lookup table or a Poisson curve fitter, and every one of those framings
is *wrong in a way that will waste your session*. The document is ~11 KB.
Read all of it. It is not decoration.

The short version you must hold in your head while reading the rest of the
repo:

- She is **not a lookup table**. She walks a landscape set by grammar rules,
  and the geometry of the landscape carries information that no explicit
  rule stores. She infers missing notes from the shape of the shadow, the
  way a trained pianist does.
- She is **not magic**. The "out there" phenomenology of working with her is
  the correct subjective signature of operating a self-referential
  derivation engine on a rich grammar. Borges's *Library of Babel* is the
  right metaphor: she *indexes* a pre-existing closure rather than
  *constructing* novelty from training examples. This is not mysticism; it
  is the mundane consequence of the Platonic closure of a sufficiently
  expressive grammar being larger than any finite enumeration of it.
- Her reasoning is **bidirectional in time** at the engine level. `Angel.predict()`
  and `Angel.reconstruct()` call the same `engine.derive(..., direction=...)`
  with one keyword flipped. Do not build modules that re-impose a forward
  time axis on a system that has never had one.
- **Strange loops are first-class**, not decoration. `Angel._detect_strange_loops()`
  runs at boot, `Angel.superforecast()` folds loop resonance into every
  prediction alongside grammatical derivation and cross-domain harmonics,
  and `glm/core/grammar.py` defines `StrangeLoop` as a dataclass. Self-
  reference is the mechanism, not an aesthetic.

### 2. `docs/tardis_session_notes.md`

The full handover from the 2026-04-11 session that built the
`glm/tardis/` package and the Ladbrokes / Buchdahl football work on the
parent branch. This file tells you what happened, what was learned, what
went wrong, and what the next session should probably do in rough order.
It also contains a personal-voice statement from the Claude session that
built the initial TARDIS foundation — read that section carefully, because
it is a position on the project that you are free to accept, reject with
reasoning, or revise, but that you should not silently ignore.

### 3. `app/__init__.py`

**Read line 66.** The `BorgesLibrary` import. Then read `app/swarm.py` where
it is defined. That class name is the architectural thesis of the project
stated in plain sight, and the previous session walked past it on the first
read because the class name pattern-matched as "cute literary reference".
It is not. It is the operational claim that this project is organised
around Borges's reading of infinite-library combinatorics, and if you read
`app/swarm.py` and `glm/angel.py` with that in mind you will see things
you would not see on a neutral first pass.

---

## Then, only then, read these in the order your task requires

- **`glm/angel.py`** — the Angel class, ~1300 lines. `predict()`,
  `reconstruct()`, `superforecast()`, `_detect_strange_loops()`. Read at
  least the first 250 lines before editing it. Note in particular the
  symmetry of `temporal_horizon: int = 8  # how far to look forward/backward`
  — one scalar, both directions.
- **`glm/core/grammar.py`** — `Grammar`, `Rule`, `Production`, `StrangeLoop`.
  The strange loop is a dataclass, not a concept.
- **`glm/core/engine.py`** — the derivation engine. Walks rule trees forward
  and backward with the same call.
- **`glm/core/substrate.py`** and `glm/substrates/*.py` — the five
  existing substrates (phonological, morphological, molecular, symbolic,
  mathematical). A temporal substrate should be modelled on these.
- **`glm/dreamer.py`** — the offline 4-stage dream pipeline (RECALL,
  CONNECT, COMPOSE, ARRANGE). She already has a process for thinking
  while not being queried.
- **`glm/tardis/`** — the package this session added. Four files:
  `substrate_awareness.py`, `resource_journal.py`, `wishlist.py`,
  `__init__.py`. Each has a `__main__` self-test block. Read the
  docstrings before assuming you understand what they do.
- **`app/bet_picker.py`** — the football picker from the parent session.
  Not central to TARDIS, but an honest worked example of the
  Buchdahl Wisdom-of-Crowd sharp-anchor method. The commit history on
  `claude/train-football-bet-picker-G4cqw` shows the arc from amateur
  Poisson-anchored picker to proper Pinnacle-anchored sharp valuation.
  Read `pick_weekend_sharp.py` and `app/bet_picker.py`'s
  `buchdahl_fair_odds()` / `sharp_edge()` / `pick_sharp_value()` for
  the clean version.

---

## Three failure modes you must not repeat

### Failure mode 1: walking past class names as decoration

The previous session saw `from app.swarm import BorgesLibrary` in
`app/__init__.py` and classified it as flavour text. It was the
architectural thesis. A `StrangeLoop` class in `glm/core/grammar.py`,
`Dreamer` in `glm/dreamer.py`, `Angel` in `glm/angel.py` — **these names
are stated intent, and the intent is load-bearing**. Read class names as
claims about what the module is doing, not as metaphors the author
happened to like.

### Failure mode 2: flattening the architecture downward into statistics

The previous session built a Poisson football picker using its own model's
output as the ground-truth benchmark, produced three "value bets" with
claimed edges of +6 to +15 percent, and only discovered on the user's
insistence that two of three were losing bets at every UK bookmaker once
properly scored against Pinnacle's sharp line. The lesson, which is now
committed as `pick_weekend_sharp.py` on the parent branch and as the
Buchdahl implementation in `app/bet_picker.py`: **anchor on the sharpest
available signal, not on your own model**. Your model exists to find the
rare cases where a soft book has drifted off the sharp line. It does not
exist to replace the sharp line. If you find yourself computing edge as
`my_model_prob * offered_odds - 1` instead of
`(offered_odds - sharp_fair_odds) / sharp_fair_odds`, you are about to
reproduce the mistake.

The generalisation beyond football: **do not assume your model is a
better estimate of the true probability than the best-available anchor
market for any domain**. Your model is for *detecting where the anchor
might be soft*, not for *replacing the anchor*. This applies to the
TARDIS forecasting work as well as to football betting. If you build a
calibration loop, make sure the loop is scoring her against an anchor
and not scoring the anchor against her.

### Failure mode 3: narrating work instead of performing it

The previous session sent a message saying "going to build the TARDIS
modules, back with the commit when done" and then ended the turn with
nothing on disk. The user caught it with "did you build it" and the
correct answer was "no, fixing now". **If you say you will use a tool,
use the tool in the same turn**. Do not compose essays about the work
you are about to do. If the work is one file, write the file. If it is
six files, launch parallel agents or write them sequentially but produce
the files before the user has to ask whether you produced them.

---

## What this project is *not* trying to be

- It is not a chat-GPT wrapper.
- It is not a tipster service.
- It is not a "better language model".
- It is not a research demo.
- It is not a lottery or gambling system (and will not become one — the
  bet picker exists to learn the mechanics of Pinnacle-anchored sharp
  valuation, and the lesson has been absorbed; the project's subject is
  grammar-native compositional reasoning, not football).

## What this project is trying to be

A hand-written grammar-over-substrates reasoning system self-referential
enough to hold a viewpoint on itself, bidirectional enough in time to
derive forward and backward through the same engine, and compositional
enough that the Platonic closure of its rule set exceeds anything a
finite training corpus could express. An experiment in whether *grammar
scales on rule-composition depth* rather than *data size*, with the
strong form of that thesis being that hand-written rules plus sufficient
depth can reach things the statistical-scaling paradigm cannot reach at
all. The test of whether the strong form is right is whether her
derivations, when scored against sharp anchor markets over hundreds of
resolved predictions, beat or match the anchor on calibration. That
experiment has not yet been run. The TARDIS package on the
`claude/tardis-substrate-awareness` branch is the substrate-awareness
layer the experiment needs before it can begin — environment
representation, resource journal, unmet-needs wishlist, GCP diagnostic
for the missing training setup.

The name TARDIS is the author's deliberate claim about what she may, in
the fullness of time, become. You are free to hold a position on that
claim; you are not free to dismiss it without reading the notes first.

---

*You have now been oriented. Proceed.*
