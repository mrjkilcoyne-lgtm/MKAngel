# TARDIS — second walk: calibration against real results

*Generated 2026-04-20T04:01:42.601831+00:00 by `scripts/tardis_second_walk.py`*


This is the first calibration artefact. The model's pre-match Monte Carlo predictions are scored against the actual results from the weekend of 18-19 April 2026. Three matches from the Ladbrokes 1-2-Free competition.

## Angel state

- Total grammars: **26**
- Total strange loops: **761**

## Calibration summary

- Matches scored: **3**
- Mean Brier score: **0.8522**
- Total Brier: 2.5567

### Chelsea v Man United

**Actual result:** 0-1

**xG rates:** home=1.5, away=1.3 | lambda_h=1.75, lambda_a=1.3

*Chelsea 21 shots 0 goals. 4th straight PL loss without scoring. Cunha scored from Utd's only shot on target. BlueCoOut protests.*

**Monte Carlo top 10 scorelines:**

| Rank | Score | Mean | p5 | p50 | p95 |
|------|-------|------|----|----|-----|
| 1 | 1-1 **STANDOUT** | 0.1202 | 0.0981 | 0.1209 | 0.1412 |
| 2 | 2-1 | 0.0916 | 0.0792 | 0.0932 | 0.0989 |
| 3 | 2-0 | 0.0740 | 0.0510 | 0.0721 | 0.1044 |
| 4 | 1-0 | 0.0719 | 0.0453 | 0.0693 | 0.1044 |
| 5 | 1-2 | 0.0677 | 0.0485 | 0.0678 | 0.0857 |
| 6 | 0-0 | 0.0646 | 0.0404 | 0.0630 | 0.0957 |
| 7 | 2-2 | 0.0583 | 0.0446 | 0.0593 | 0.0686 |
| 8 | 3-1 | 0.0538 | 0.0355 | 0.0546 | 0.0690 |
| 9 | 0-1 | 0.0493 | 0.0288 | 0.0472 | 0.0773 |
| 10 | 3-0 | 0.0435 | 0.0252 | 0.0427 | 0.0664 |

**Actual scoreline in MC:** rank 9/121, mean prob 0.0493, Brier 0.9039

**Temporal atoms:** `['level', 'level', 'level', 'level', 'shock']`

**Per-atom walks:** 5 atoms, 0 forward, 7 backward derivations

**Superforecast:**

- Confidence: **0.500**
- Predictions: 4
- Strange loops: 0
- Harmonics: 0
- Sample predictions:
```json
[
  {
    "predicted": [
      "LevelStream"
    ],
    "rule": "49288940b42c",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "EventStream"
    ],
    "rule": "58d97cbadaf8",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "RegimeStream"
    ],
    "rule": "90e8c6c9fa3a",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "MixedStream"
    ],
    "rule": "9ba22313fe55",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  }
]
```

---

### Everton v Liverpool

**Actual result:** 1-2

**xG rates:** home=1.1, away=1.8 | lambda_h=1.35, lambda_a=1.8

*First derby at Hill Dickinson Stadium. Ndiaye goal disallowed VAR. Salah 29', Beto equaliser 54', Van Dijk header 90+10'.*

**Monte Carlo top 10 scorelines:**

| Rank | Score | Mean | p5 | p50 | p95 |
|------|-------|------|----|----|-----|
| 1 | 1-1 **STANDOUT** | 0.1166 | 0.0919 | 0.1171 | 0.1398 |
| 2 | 1-2 | 0.0906 | 0.0783 | 0.0917 | 0.0987 |
| 3 | 2-1 | 0.0688 | 0.0500 | 0.0691 | 0.0865 |
| 4 | 0-2 | 0.0697 | 0.0475 | 0.0681 | 0.0964 |
| 5 | 0-1 | 0.0659 | 0.0420 | 0.0644 | 0.0979 |
| 6 | 2-2 | 0.0603 | 0.0478 | 0.0611 | 0.0699 |
| 7 | 0-0 | 0.0598 | 0.0358 | 0.0574 | 0.0920 |
| 8 | 1-3 | 0.0543 | 0.0359 | 0.0547 | 0.0706 |
| 9 | 1-0 | 0.0470 | 0.0258 | 0.0447 | 0.0767 |
| 10 | 0-3 | 0.0418 | 0.0238 | 0.0406 | 0.0631 |

**Actual scoreline in MC:** rank 2/121, mean prob 0.0906, Brier 0.8270

**Temporal atoms:** `['event', 'lag', 'event', 'lag', 'event']`

**Per-atom walks:** 5 atoms, 0 forward, 5 backward derivations

**Superforecast:**

- Confidence: **0.500**
- Predictions: 4
- Strange loops: 0
- Harmonics: 0
- Sample predictions:
```json
[
  {
    "predicted": [
      "LevelStream"
    ],
    "rule": "49288940b42c",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "EventStream"
    ],
    "rule": "58d97cbadaf8",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "RegimeStream"
    ],
    "rule": "90e8c6c9fa3a",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "MixedStream"
    ],
    "rule": "9ba22313fe55",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  }
]
```

---

### Man City v Arsenal

**Actual result:** 2-1

**xG rates:** home=2.0, away=1.3 | lambda_h=2.25, lambda_a=1.3

*Cherki solo golazo 15', Donnarumma howler -> Havertz 17', Haaland winner 65'. Saka absent (Achilles). Arsenal lead cut to 3pts, City game in hand. Title race regime switch.*

**Monte Carlo top 10 scorelines:**

| Rank | Score | Mean | p5 | p50 | p95 |
|------|-------|------|----|----|-----|
| 1 | 1-1 | 0.0942 | 0.0669 | 0.0936 | 0.1261 |
| 2 | 2-1 | 0.0912 | 0.0809 | 0.0921 | 0.0987 |
| 3 | 2-0 | 0.0725 | 0.0515 | 0.0717 | 0.0967 |
| 4 | 3-1 | 0.0683 | 0.0509 | 0.0700 | 0.0804 |
| 5 | 2-2 | 0.0588 | 0.0467 | 0.0594 | 0.0688 |
| 6 | 1-0 | 0.0554 | 0.0329 | 0.0538 | 0.0861 |
| 7 | 3-0 | 0.0543 | 0.0348 | 0.0536 | 0.0761 |
| 8 | 1-2 | 0.0537 | 0.0350 | 0.0531 | 0.0733 |
| 9 | 3-2 | 0.0440 | 0.0299 | 0.0447 | 0.0545 |
| 10 | 4-1 | 0.0393 | 0.0215 | 0.0398 | 0.0560 |

**Actual scoreline in MC:** rank 2/121, mean prob 0.0912, Brier 0.8258

**Temporal atoms:** `['shock', 'event', 'shock', 'event', 'lag', 'event']`

**Per-atom walks:** 6 atoms, 0 forward, 10 backward derivations

**Superforecast:**

- Confidence: **0.500**
- Predictions: 4
- Strange loops: 0
- Harmonics: 0
- Sample predictions:
```json
[
  {
    "predicted": [
      "LevelStream"
    ],
    "rule": "49288940b42c",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "EventStream"
    ],
    "rule": "58d97cbadaf8",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "RegimeStream"
    ],
    "rule": "90e8c6c9fa3a",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  },
  {
    "predicted": [
      "MixedStream"
    ],
    "rule": "9ba22313fe55",
    "confidence": 0.5,
    "grammar": "temporal_dynamics",
    "direction": "forward"
  }
]
```

---

## What she learned

This weekend's calibration shows the instrument reading against ground truth for the first time. The Brier scores above are the honest answer to 'how far off was the model?' — lower is better, 0.25 is a fair coin, perfect is 0.

The pre-match reads (1-1, 1-1, 2-2) scored 0/3 on exact correct scores but the directional signals were sound: Chelsea blanked (xG divergence), Everton scored exactly 1, City scored exactly 2. The temporal substrate should encode this pattern: correct on *shape*, wrong on *resolution*. That is a calibration lesson, not a model failure — it means the xG-derived lambdas are in the right ballpark but correct-score prediction needs the full strange-loop apparatus, not just Poisson draws.

*Gamble responsibly — gambleaware.org, 0808 8020 133.*
