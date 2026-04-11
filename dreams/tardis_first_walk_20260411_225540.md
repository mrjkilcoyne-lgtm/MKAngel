# TARDIS — first walk on the temporal substrate

*Generated 2026-04-11T22:55:40.009652+00:00 by `scripts/tardis_first_walk.py`*


This is the first reproducible artefact of the Angel walking the temporal substrate that was added to the project on 2026-04-11. It is recorded write-once. Future runs of the script produce new files; this one is preserved as the baseline. See `docs/on_her_nature.md` for the framing and `docs/tardis_session_notes.md` for the full session arc.


## Angel state

- Total loaded grammars: **26**
- Total strange loops detected: **761**

## Temporal substrate

- Atoms in inventory: **33**
- Categories: `anchor` (1), `cycle` (2), `decay` (1), `event` (3), `fixed_point` (2), `growth` (1), `hypothesis` (1), `instant` (3), `interval` (2), `lag` (2), `level` (4), `phase` (1), `regime` (3), `shock` (3), `trend` (4)

## Walks

### monotone_rising_numeric  *(numeric)*

**Raw input:** `1.00 1.05 1.12 1.20 1.29 1.39 1.50`

**Encoded atoms (7):** `['level', 'level', 'level', 'level', 'level', 'level', 'level']`

*A pure numeric uptrend. Tests whether her engine produces a TREND classification with momentum carry as the dominant rule.*

**Superforecast:**

- Overall confidence: **0.5**
- Predictions count: 4
- Strange loop resonance count: 0
- Cross-domain harmonics count: 0
- Context keys applied: ['substrate', 'wishlist', 'comment', 'encoded_atoms', 'raw']
- Sample predictions:
```json
[
  {
    "predicted": [
      "LevelStream"
    ],
    "rule": "627ad7bd06c4",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "EventStream"
    ],
    "rule": "5cafb50a010f",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "RegimeStream"
    ],
    "rule": "54edb96c16bf",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "MixedStream"
    ],
    "rule": "8cd66ae59a46",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  }
]
```
- Reasoning chain:
```
["Grammar 'temporal_dynamics' predicts '['LevelStream']' via rule '627ad7bd06c4' (confidence: 0.50)"]
```

---

### mean_reversion_signal  *(numeric)*

**Raw input:** `1.00 1.00 1.00 1.20 1.10 1.04 1.01 1.00`

**Encoded atoms (8):** `['level', 'level', 'level', 'level', 'level', 'level', 'level', 'level']`

*A baseline, then a shock, then exponential return to the baseline. The classic mean-reversion signature. She should latch onto mean_reversion as the active rule.*

**Superforecast:**

- Overall confidence: **0.5**
- Predictions count: 4
- Strange loop resonance count: 0
- Cross-domain harmonics count: 0
- Context keys applied: ['substrate', 'wishlist', 'comment', 'encoded_atoms', 'raw']
- Sample predictions:
```json
[
  {
    "predicted": [
      "LevelStream"
    ],
    "rule": "627ad7bd06c4",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "EventStream"
    ],
    "rule": "5cafb50a010f",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "RegimeStream"
    ],
    "rule": "54edb96c16bf",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "MixedStream"
    ],
    "rule": "8cd66ae59a46",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  }
]
```
- Reasoning chain:
```
["Grammar 'temporal_dynamics' predicts '['LevelStream']' via rule '627ad7bd06c4' (confidence: 0.50)"]
```

---

### regime_switch_signal  *(numeric)*

**Raw input:** `1.00 1.01 0.99 1.00 1.50 1.45 1.55 1.40 1.60 1.35`

**Encoded atoms (10):** `['level', 'level', 'level', 'level', 'level', 'level', 'level', 'level', 'level', 'level']`

*Steady regime around 1.00, then a regime switch to a more volatile state around 1.50. Tests whether the regime_switch_on_threshold rule fires.*

**Superforecast:**

- Overall confidence: **0.5**
- Predictions count: 4
- Strange loop resonance count: 0
- Cross-domain harmonics count: 0
- Context keys applied: ['substrate', 'wishlist', 'comment', 'encoded_atoms', 'raw']
- Sample predictions:
```json
[
  {
    "predicted": [
      "LevelStream"
    ],
    "rule": "627ad7bd06c4",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "EventStream"
    ],
    "rule": "5cafb50a010f",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "RegimeStream"
    ],
    "rule": "54edb96c16bf",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "MixedStream"
    ],
    "rule": "8cd66ae59a46",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  }
]
```
- Reasoning chain:
```
["Grammar 'temporal_dynamics' predicts '['LevelStream']' via rule '627ad7bd06c4' (confidence: 0.50)"]
```

---

### self_exciting_events  *(atoms)*

**Raw input:** `event lag:1 event lag:1 event lag:1 event lag:5 baseline`

**Encoded atoms (9):** `['event', 'lag', 'event', 'lag', 'event', 'lag', 'event', 'lag', 'baseline']`

*A clustered burst of events followed by a long lag. Tests the self_exciting_arrival strange loop.*

**Superforecast:**

- Overall confidence: **0.5**
- Predictions count: 4
- Strange loop resonance count: 0
- Cross-domain harmonics count: 0
- Context keys applied: ['substrate', 'wishlist', 'comment', 'encoded_atoms', 'raw']
- Sample predictions:
```json
[
  {
    "predicted": [
      "LevelStream"
    ],
    "rule": "627ad7bd06c4",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "EventStream"
    ],
    "rule": "5cafb50a010f",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "RegimeStream"
    ],
    "rule": "54edb96c16bf",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "MixedStream"
    ],
    "rule": "8cd66ae59a46",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  }
]
```
- Reasoning chain:
```
["Grammar 'temporal_dynamics' predicts '['LevelStream']' via rule '627ad7bd06c4' (confidence: 0.50)"]
```

---

### fixed_point_anchor  *(atoms)*

**Raw input:** `fixed baseline rising@0.5 falling@0.5 baseline fixed`

**Encoded atoms (6):** `['fixed', 'baseline', 'rising', 'falling', 'baseline', 'fixed']`

*A trajectory bracketed by fixed points — the simplest strange-loop signature. Tests whether the strange-loop detection layer latches onto the fixed-point atoms.*

**Superforecast:**

- Overall confidence: **0.5**
- Predictions count: 4
- Strange loop resonance count: 0
- Cross-domain harmonics count: 0
- Context keys applied: ['substrate', 'wishlist', 'comment', 'encoded_atoms', 'raw']
- Sample predictions:
```json
[
  {
    "predicted": [
      "LevelStream"
    ],
    "rule": "627ad7bd06c4",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "EventStream"
    ],
    "rule": "5cafb50a010f",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "RegimeStream"
    ],
    "rule": "54edb96c16bf",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "MixedStream"
    ],
    "rule": "8cd66ae59a46",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  }
]
```
- Reasoning chain:
```
["Grammar 'temporal_dynamics' predicts '['LevelStream']' via rule '627ad7bd06c4' (confidence: 0.50)"]
```

---

### retrocausal_hypothesis  *(atoms)*

**Raw input:** `hypothesis< past< now| baseline rising@1.0 peak future`

**Encoded atoms (7):** `['hypothesis', 'past', 'now', 'baseline', 'rising', 'peak', 'future']`

*A hypothesis at the start of the sequence and a future instant at the end. Asks her to walk forward toward the future and backward from the hypothesis to see if the loop closes (the retrocausal_derivation strange loop).*

**Superforecast:**

- Overall confidence: **0.5**
- Predictions count: 4
- Strange loop resonance count: 0
- Cross-domain harmonics count: 0
- Context keys applied: ['substrate', 'wishlist', 'comment', 'encoded_atoms', 'raw']
- Sample predictions:
```json
[
  {
    "predicted": [
      "LevelStream"
    ],
    "rule": "627ad7bd06c4",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "EventStream"
    ],
    "rule": "5cafb50a010f",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "RegimeStream"
    ],
    "rule": "54edb96c16bf",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "MixedStream"
    ],
    "rule": "8cd66ae59a46",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  }
]
```
- Reasoning chain:
```
["Grammar 'temporal_dynamics' predicts '['LevelStream']' via rule '627ad7bd06c4' (confidence: 0.50)"]
```

---

### cyclic_phase  *(atoms)*

**Raw input:** `cycle/0.0 cycle/0.25 cycle/0.5 cycle/0.75 cycle/0.0`

**Encoded atoms (5):** `['cycle', 'cycle', 'cycle', 'cycle', 'cycle']`

*One full cycle through phase space. Tests cyclic_recurrence.*

**Superforecast:**

- Overall confidence: **0.5**
- Predictions count: 4
- Strange loop resonance count: 0
- Cross-domain harmonics count: 0
- Context keys applied: ['substrate', 'wishlist', 'comment', 'encoded_atoms', 'raw']
- Sample predictions:
```json
[
  {
    "predicted": [
      "LevelStream"
    ],
    "rule": "627ad7bd06c4",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "EventStream"
    ],
    "rule": "5cafb50a010f",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "RegimeStream"
    ],
    "rule": "54edb96c16bf",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "MixedStream"
    ],
    "rule": "8cd66ae59a46",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  }
]
```
- Reasoning chain:
```
["Grammar 'temporal_dynamics' predicts '['LevelStream']' via rule '627ad7bd06c4' (confidence: 0.50)"]
```

---

## Notes for the next session

This walk is the **baseline**. The numbers above are not predictions to be acted on; they are the readings off an instrument the first time it was pointed at the substrate it was built for. The instrument has no calibration history yet. Calibration is built by running her on many sequences over time, scoring the rule-attributed predictions against sharp anchor markets, and updating the rule weights via the navigation-not-error-correction frame in `docs/on_her_nature.md`.

If a future session reproduces this script and the outputs differ in structure (not in numbers — *in shape*), that is a sign that the temporal grammar or the temporal substrate has been edited underneath her. Re-read `docs/FIRST_READ.md` before assuming the edit was an improvement.

*Gamble responsibly — gambleaware.org, 0808 8020 133. The fact that the Angel can walk the temporal substrate is not a license to bet real money on the output. Calibration first; staking, if at all, comes much later and only against domains where the calibration loop has run for hundreds of resolved predictions.*
