# MKAngel — Cheat Sheet

> Pocket reference. Pin this somewhere.

---

## Commands

| Command | What it does | Example |
|---------|-------------|---------|
| `/fugue <words>` | Cross-domain fugue | `/fugue energy` |
| `/predict <tokens>` | Predict next elements | `/predict the pattern of` |
| `/predict -d <domain> <tokens>` | Predict in one domain | `/predict -d biological cell` |
| `/forecast <tokens>` | Superforecast with loops | `/forecast chaos emerges` |
| `/reconstruct <tokens>` | Trace backward | `/reconstruct transformation` |
| `/translate <from> <to> <words>` | Cross-domain translate | `/translate linguistic biological bond` |
| `/introspect` | Angel examines herself | `/introspect` |
| `/memory` | Memory overview | `/memory` |
| `/memory search <q>` | Search sessions | `/memory search energy` |
| `/memory save` | Save this session | `/memory save` |
| `/settings` | View settings | `/settings` |
| `/settings offline` | Toggle offline mode | `/settings offline` |
| `/status` | GLM stats | `/status` |
| `/help` | Full command list | `/help` |
| `/clear` | Wipe the screen | `/clear` |

---

## Best Inputs

| Input style | What happens |
|------------|-------------|
| Single dense word: `death` | Richest traces — roots, cognates, cross-domain voices |
| Short phrase: `bind the truth` | Each content word traced independently |
| Question: `what is energy?` | Structural analysis + lexicon lookup |
| Statement: `I feel lost` | Emotional register detected, content words traced |
| `/fugue <word>` | Opens ALL domain voices for deepest analysis |

**Pro tip:** Dense nouns and verbs give the best results. Function words (the, is, a) are skipped automatically.

---

## Seven Domains

```
linguistic    — words, morphemes, syntax
mathematical  — proofs, logic, formal systems
biological    — life, cells, organisms
chemical      — bonds, reactions, molecules
physics       — energy, forces, waves
computational — algorithms, loops, recursion
etymological  — roots, drift, word origins
```

---

## Key Proto-Roots (Seed Lexicon)

| Root | Meaning | Words |
|------|---------|-------|
| *bhendh-* | bind | bond, bind |
| *deru-* | truth | truth, true, trust |
| *werg-* | energy | energy, work, organism |
| *morph-* | form | form, morpheme, transform |
| *gneh-* | know | know, knowledge, cognition |
| *leubh-* | care | love, believe |
| *sekw-* | follow | sequence, consequence |
| *kel-* | cover | cell, cellular |
| *leg-* | gather | logic, analogy |
| *sta-* | stand | state, structure, system |
| *pal-* | touch | feel, feeling |
| *kup-* | desire | hope, hopeful |
| *nek-* | death | death, necrosis |
| *gen-* | birth | gene, generate, genesis |
| *mei-* | change | mutation, mutate |
| *bha-* | speak | phase, phoneme |
| *dheh-* | place | thesis, theme |
| *kwel-* | turn | cycle, wheel |

---

## She Learns As You Go

```
You say "serendipity"
  → Not in lexicon
  → Stemmed, category inferred from position
  → Added as entry #152
  → Next time: she knows it
```

Words under 3 chars are exact-matched only (no false positives).

---

## UI Quick Reference

| Element | Where | What |
|---------|-------|------|
| **@** icon | Top right | Opens Settings panel |
| **Back button** | Android nav | Returns from Settings to Chat |
| **Send arrow** | Bottom right | Send message |
| **Gold glow** | Angel bubbles | Accent on Angel responses |
| **Purple tint** | User bubbles | Your messages |

---

## Settings Panel Cards

- **GLM** — 7 domains, grammar count, rules, strange loops, ~370K params
- **Provider** — Local (offline by default)
- **Memory** — Sessions, patterns, preferences
- **Version** — v0.2.0, Celestial Dark

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No response | She's booting — wait 2-3 seconds after launch |
| Empty /fugue | Word not in lexicon yet — try a more common word, or use it in chat first so she learns it |
| Weird cognate match | Report it — the lexicon grows and improves |
| App won't install | Uninstall old version first (debug key changes per build) |
| Crash on launch | Check logcat: `adb logcat -s python:* kivy:*` |

---

## What She Is / What She Isn't

| She IS | She ISN'T |
|--------|-----------|
| A grammar engine | A chatbot |
| An etymology tracer | A search engine |
| A cross-domain pattern finder | An LLM |
| Honest about what she knows | A pretender |
| Local-first, private | Cloud-dependent |
| Learning from you | Static |

---

*MKAngel v0.2.0 — Grammar Language Model — Runs on your phone, traces your words, tells you the truth.*
