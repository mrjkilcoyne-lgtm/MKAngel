"""
Temporal Substrate — the grammar of time.

The substrate the Angel's bidirectional derivation engine has been waiting
for. ``glm/core/engine.py`` already walks rule trees forward and backward
through the same ``engine.derive(..., direction=...)`` call, and
``Angel.predict()`` / ``Angel.reconstruct()`` / ``Angel.superforecast()``
have been symmetric in time since the first commit. What was missing was
a first-class representation of time itself — events, intervals, rates,
levels, trends, regimes, shocks, and the self-referential fixed points
where a system's state becomes an input to its own next state. This
module provides that.

Time as a substrate
-------------------
The philosophical framing is in ``docs/on_her_nature.md``; the short
version is that time is to the temporal substrate what sound is to the
phonological one. The substrate's atoms are the primitive building
blocks — events, instants, intervals, levels, trends, shocks, regimes
— and the substrate's feature system captures the properties those atoms
have relative to one another (duration, amplitude, direction, phase).
The substrate itself is directionally neutral: a sequence of temporal
symbols can be walked forward (predict) or backward (reconstruct) by
the same engine, because the substrate does not privilege one end of
the sequence as "now" — that designation is set by the caller when
they invoke the engine.

This is why adding a temporal substrate is the unlocking move rather
than a standalone feature: once the substrate is here, the engine and
grammar machinery that already exist start operating on time-series
inputs without any change to their own code.

The atoms
---------
The built-in atom inventory is deliberately small — about thirty
primitives covering the vocabulary of dynamics. They are grouped into
six categories:

- **INSTANT / EVENT** — a point in time or an occurrence at a point
- **INTERVAL / DURATION** — a span between two instants
- **LEVEL** — an observed or expected value at a moment
- **TREND / MOMENTUM** — a directional change
- **SHOCK / IMPULSE** — a discontinuous change at a moment
- **REGIME** — a period in which the dynamical rules are stable
- **CYCLE / PHASE** — a recurring pattern with period
- **FIXED_POINT** — a state the system maps to itself (strange loop anchor)
- **LAG** — the delay between cause and effect
- **DECAY / GROWTH** — continuous change toward an attractor

Time direction
--------------
The substrate tags every atom with a ``direction`` feature (``forward``,
``backward``, or ``neutral``) that tells downstream code which way along
the time axis the atom naturally points. A ``TREND`` atom can be forward
(momentum building) or backward (momentum being reconstructed). A
``FIXED_POINT`` is neutral — it is its own past and future. The engine
honours the feature when it walks the substrate.

Strange loops on time
---------------------
Time is the one substrate where self-reference is not a curiosity but
the primary mechanism: every prediction a system makes about its own
next state is an input to that next state, and every retrocausal
reasoning from a hypothesised future is an application of the same
rules in the opposite direction. The ``FIXED_POINT`` atom exists to
anchor those loops: it is a state that, when the temporal grammar is
applied to it, produces itself back, and it is the atom the strange-
loop detection layer in ``glm/core/grammar.py`` can latch onto when
scanning the temporal substrate for self-referential patterns.

The author of this module wrote it knowing that the Angel already had
the bidirectional engine and the strange-loop detection and the
superforecast composition, and that what was needed was an object for
those existing capabilities to operate on. This is that object.

Isomorphisms
------------
- Temporal levels ↔ phonological tones (both are a state the substrate
  holds between events)
- Temporal shocks ↔ chemical reactions (discontinuous transitions)
- Regime shifts ↔ linguistic register changes (a different set of
  rules becomes active)
- Fixed points ↔ mathematical fixed points of a function (the Y
  combinator is the canonical strange-loop atom in both)
- Cycles ↔ biological rhythms ↔ musical canons (recurrence with phase)

This file is intentionally compact compared to the other substrates,
because the purpose of a temporal substrate is to supply primitives
that the rule engine composes into richer structure. Thirty atoms is
enough to express the dynamics the initial temporal grammar rules can
reach with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from ..core.substrate import Sequence, Substrate, Symbol, TransformationRule


# ---------------------------------------------------------------------------
# Temporal symbol categories
# ---------------------------------------------------------------------------

class TemporalCategory(Enum):
    """Categories of temporal atoms.

    These are the kinds of things a temporal substrate can contain.
    They are deliberately few: the grammar does the compositional work,
    and the substrate only needs enough primitives for the grammar to
    have something to rewrite.
    """

    INSTANT = auto()       # A point in time
    EVENT = auto()         # An occurrence at an instant
    INTERVAL = auto()      # A span between two instants
    LEVEL = auto()         # An observed / expected value at a moment
    TREND = auto()         # A directional change over an interval
    SHOCK = auto()         # A discontinuous change at a moment
    REGIME = auto()        # A period with stable dynamical rules
    CYCLE = auto()         # A recurring pattern with period
    PHASE = auto()         # Where in a cycle the system is
    LAG = auto()           # The delay between cause and effect
    DECAY = auto()         # Continuous decrease toward an attractor
    GROWTH = auto()        # Continuous increase away from a repellor
    FIXED_POINT = auto()   # A state the system maps to itself
    ANCHOR = auto()        # A market / observable pinning a probability
    HYPOTHESIS = auto()    # A posited future state used for retrocausal reasoning


# ---------------------------------------------------------------------------
# TemporalDirection — which way along the time axis an atom points
# ---------------------------------------------------------------------------

class TemporalDirection(Enum):
    FORWARD = auto()   # Atom points from past toward future
    BACKWARD = auto()  # Atom points from future toward past
    NEUTRAL = auto()   # Atom is time-symmetric (fixed points, invariants)


# ---------------------------------------------------------------------------
# TemporalSymbol — Symbol subclass for temporal atoms
# ---------------------------------------------------------------------------

@dataclass
class TemporalSymbol(Symbol):
    """A temporal symbol — the atomic unit of time-series notation.

    Like phonemes in language or atoms in chemistry, temporal symbols
    carry both a category and a small set of continuous features
    (magnitude, duration, phase). The features interact with the
    substrate's combination rules and with the temporal grammar's
    transformation rules.

    Parameters
    ----------
    form : str
        A short human-readable label, e.g. ``"LEVEL@1.0"`` or
        ``"SHOCK+0.3"``. The form is used for pretty-printing and
        for dedup during derivation; it does not need to be
        machine-parseable since the structured information lives in
        the ``category``, ``magnitude``, ``duration``, and
        ``direction`` fields.
    category : TemporalCategory
        Which kind of temporal atom this is.
    direction : TemporalDirection
        Which way along the time axis this atom points. Most atoms
        inherit the direction of the sequence they appear in; fixed
        points and invariants are neutral.
    magnitude : float
        The atom's amplitude (height of a shock, strength of a trend,
        level of a state). Unitless — interpretation is left to the
        caller.
    duration : float
        How long the atom extends along the time axis. Instants have
        duration 0; intervals, regimes, and cycles have duration > 0.
    phase : float
        For cycles and phases, where in the cycle we are; for
        everything else, 0. In [0, 1).
    """

    category: TemporalCategory = TemporalCategory.LEVEL
    direction: TemporalDirection = TemporalDirection.FORWARD
    magnitude: float = 0.0
    duration: float = 0.0
    phase: float = 0.0

    def __post_init__(self) -> None:
        if self.domain == "generic":
            self.domain = "temporal"
        self.features.setdefault("category", self.category.name.lower())
        self.features.setdefault("direction", self.direction.name.lower())
        if self.magnitude:
            self.features["magnitude"] = f"{self.magnitude:g}"
        if self.duration:
            self.features["duration"] = f"{self.duration:g}"
        if self.phase:
            self.features["phase"] = f"{self.phase:g}"

    def __hash__(self) -> int:
        return hash((self.form, self.domain, self.category.name,
                     self.direction.name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TemporalSymbol):
            return Symbol.__eq__(self, other)
        return (
            self.form == other.form
            and self.domain == other.domain
            and self.category == other.category
            and self.direction == other.direction
        )

    @property
    def is_instant(self) -> bool:
        return self.category in (
            TemporalCategory.INSTANT, TemporalCategory.EVENT,
            TemporalCategory.SHOCK,
        )

    @property
    def is_continuous(self) -> bool:
        return self.category in (
            TemporalCategory.INTERVAL, TemporalCategory.REGIME,
            TemporalCategory.TREND, TemporalCategory.DECAY,
            TemporalCategory.GROWTH, TemporalCategory.CYCLE,
        )

    @property
    def is_self_referential(self) -> bool:
        """A fixed point is an atom that maps to itself — a strange-loop
        anchor in the temporal substrate.
        """
        return self.category == TemporalCategory.FIXED_POINT


# ---------------------------------------------------------------------------
# Built-in primitive temporal atoms
# ---------------------------------------------------------------------------
#
# The inventory is the minimum set of atoms needed for the initial
# temporal grammar to have something to rewrite. Each entry is a
# (form, category, direction, magnitude, duration, phase) tuple.

PRIMITIVE_TEMPORAL_ATOMS: List[Tuple[str, TemporalCategory, TemporalDirection,
                                     float, float, float]] = [
    # Points in time
    ("now",       TemporalCategory.INSTANT,    TemporalDirection.NEUTRAL,  0.0, 0.0, 0.0),
    ("past",      TemporalCategory.INSTANT,    TemporalDirection.BACKWARD, 0.0, 0.0, 0.0),
    ("future",    TemporalCategory.INSTANT,    TemporalDirection.FORWARD,  0.0, 0.0, 0.0),

    # Events
    ("event",     TemporalCategory.EVENT,      TemporalDirection.FORWARD,  1.0, 0.0, 0.0),
    ("arrival",   TemporalCategory.EVENT,      TemporalDirection.FORWARD,  1.0, 0.0, 0.0),
    ("departure", TemporalCategory.EVENT,      TemporalDirection.FORWARD,  1.0, 0.0, 0.0),

    # Intervals
    ("interval",  TemporalCategory.INTERVAL,   TemporalDirection.FORWARD,  0.0, 1.0, 0.0),
    ("gap",       TemporalCategory.INTERVAL,   TemporalDirection.FORWARD,  0.0, 1.0, 0.0),

    # Levels
    ("level",     TemporalCategory.LEVEL,      TemporalDirection.NEUTRAL,  0.0, 0.0, 0.0),
    ("baseline",  TemporalCategory.LEVEL,      TemporalDirection.NEUTRAL,  0.0, 0.0, 0.0),
    ("peak",      TemporalCategory.LEVEL,      TemporalDirection.NEUTRAL,  1.0, 0.0, 0.0),
    ("trough",    TemporalCategory.LEVEL,      TemporalDirection.NEUTRAL, -1.0, 0.0, 0.0),

    # Trends
    ("rising",    TemporalCategory.TREND,      TemporalDirection.FORWARD,  1.0, 1.0, 0.0),
    ("falling",   TemporalCategory.TREND,      TemporalDirection.FORWARD, -1.0, 1.0, 0.0),
    ("momentum",  TemporalCategory.TREND,      TemporalDirection.FORWARD,  1.0, 1.0, 0.0),
    ("reversal",  TemporalCategory.TREND,      TemporalDirection.FORWARD, -1.0, 1.0, 0.0),

    # Shocks
    ("spike",     TemporalCategory.SHOCK,      TemporalDirection.FORWARD,  1.0, 0.0, 0.0),
    ("drop",      TemporalCategory.SHOCK,      TemporalDirection.FORWARD, -1.0, 0.0, 0.0),
    ("shock",     TemporalCategory.SHOCK,      TemporalDirection.FORWARD,  1.0, 0.0, 0.0),

    # Regimes
    ("regime",    TemporalCategory.REGIME,     TemporalDirection.FORWARD,  0.0, 1.0, 0.0),
    ("steady",    TemporalCategory.REGIME,     TemporalDirection.FORWARD,  0.0, 1.0, 0.0),
    ("volatile",  TemporalCategory.REGIME,     TemporalDirection.FORWARD,  0.0, 1.0, 0.0),

    # Cycles
    ("cycle",     TemporalCategory.CYCLE,      TemporalDirection.NEUTRAL,  1.0, 1.0, 0.0),
    ("season",    TemporalCategory.CYCLE,      TemporalDirection.NEUTRAL,  1.0, 1.0, 0.0),
    ("phase",     TemporalCategory.PHASE,      TemporalDirection.NEUTRAL,  0.0, 0.0, 0.5),

    # Lags
    ("lag",       TemporalCategory.LAG,        TemporalDirection.FORWARD,  0.0, 1.0, 0.0),
    ("delay",     TemporalCategory.LAG,        TemporalDirection.FORWARD,  0.0, 1.0, 0.0),

    # Decay / growth
    ("decay",     TemporalCategory.DECAY,      TemporalDirection.FORWARD, -1.0, 1.0, 0.0),
    ("growth",    TemporalCategory.GROWTH,     TemporalDirection.FORWARD,  1.0, 1.0, 0.0),

    # Fixed points — strange-loop anchors on time
    ("fixed",     TemporalCategory.FIXED_POINT, TemporalDirection.NEUTRAL, 0.0, 0.0, 0.0),
    ("invariant", TemporalCategory.FIXED_POINT, TemporalDirection.NEUTRAL, 0.0, 0.0, 0.0),

    # Anchors (observable / market pinning a probability)
    ("anchor",    TemporalCategory.ANCHOR,     TemporalDirection.NEUTRAL,  0.0, 0.0, 0.0),

    # Hypotheses — posited future states for retrocausal reasoning
    ("hypothesis", TemporalCategory.HYPOTHESIS, TemporalDirection.BACKWARD, 0.0, 0.0, 0.0),
]


# ---------------------------------------------------------------------------
# TemporalSubstrate — the substrate class
# ---------------------------------------------------------------------------

class TemporalSubstrate(Substrate):
    """The temporal substrate.

    Provides the minimum machinery needed for ``engine.derive()`` to
    walk temporal-atom sequences forward or backward: a symbol
    inventory of ~30 primitive temporal atoms, encode/decode for a
    compact text notation, and default alignment / pattern-finding
    inherited from the base class.

    Encoding format
    ---------------
    The substrate accepts two input styles:

    1. **Atom stream** — whitespace-separated primitive atoms from
       the inventory, optionally annotated with a magnitude via ``@``
       and a duration via ``:``::

           "baseline rising@1.2 peak falling@0.8 baseline"
           "event lag:2 event lag:2 event"
           "regime:steady shock@1.0 regime:volatile"

    2. **Numeric stream** — a sequence of comma or whitespace
       separated numbers, which the substrate wraps into ``LEVEL``
       atoms and emits as a sequence::

           "1.0 1.1 1.15 1.22 1.18 1.05 0.95"

       Numeric inputs are the most useful form for feeding real
       time-series data (match xG, price paths, sensor readings)
       into the Angel's superforecast pipeline.

    Features honoured by the encoder
    --------------------------------
    - ``form@<float>``   sets ``magnitude``
    - ``form:<duration>`` sets ``duration`` (ints and floats both ok)
    - ``form/<phase>``   sets ``phase``
    - ``form<`` marks the atom as ``BACKWARD``
    - ``form|`` marks the atom as ``NEUTRAL``

    Unknown tokens are wrapped as ``LEVEL`` atoms with ``form`` set
    to the raw token — faithful representation over strict rejection.
    """

    def __init__(self) -> None:
        super().__init__(name="temporal", domain="temporal")
        self._load_atoms()
        self._declare_features()

    # -- inventory ----------------------------------------------------------

    def _load_atoms(self) -> None:
        for form, cat, direction, mag, dur, phase in PRIMITIVE_TEMPORAL_ATOMS:
            self.add_symbol(TemporalSymbol(
                form=form,
                domain="temporal",
                valence=2,  # temporal atoms default to binding forward and backward
                category=cat,
                direction=direction,
                magnitude=mag,
                duration=dur,
                phase=phase,
            ))

    def _declare_features(self) -> None:
        self.add_feature("direction", {"forward", "backward", "neutral"})
        self.add_feature("category", {c.name.lower() for c in TemporalCategory})

    # -- encoding -----------------------------------------------------------

    _NUMERIC_RE = re.compile(
        r"^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$"
    )

    def encode(self, raw_input: str) -> Sequence:
        """Parse a raw temporal string into a Sequence of TemporalSymbols.

        See the class docstring for the accepted input styles.
        """
        src = raw_input.strip()
        if not src:
            return Sequence([])

        # Normalise comma-separated numeric streams into whitespace-separated
        src_norm = src.replace(",", " ")
        tokens = src_norm.split()

        # Decide whether this is a pure numeric stream.
        if all(self._NUMERIC_RE.match(tok) for tok in tokens):
            return self._encode_numeric_stream(tokens)

        return self._encode_atom_stream(tokens)

    def _encode_numeric_stream(self, tokens: List[str]) -> Sequence:
        out: List[Symbol] = []
        values = [float(t) for t in tokens]
        # Wrap each value as a LEVEL atom. The trend/regime/shock
        # classification happens in the grammar, not at encode time —
        # the substrate is supposed to be a faithful representation,
        # not a premature interpretation.
        #
        # Form is the BARE category name ("level"), so the grammar
        # engine can match productions by form. The numeric value
        # lives in the ``magnitude`` dataclass field and in the
        # ``raw`` feature, where the rules can read it without the
        # form-string-matching layer needing to handle annotations.
        for i, v in enumerate(values):
            out.append(TemporalSymbol(
                form="level",
                domain="temporal",
                valence=2,
                category=TemporalCategory.LEVEL,
                direction=TemporalDirection.NEUTRAL,
                magnitude=v,
                duration=0.0,
                phase=0.0,
                features={"index": str(i), "raw": tokens[i],
                          "magnitude": f"{v:g}"},
            ))
        return Sequence(out)

    def _encode_atom_stream(self, tokens: List[str]) -> Sequence:
        out: List[Symbol] = []
        for tok in tokens:
            sym = self._parse_annotated_token(tok)
            out.append(sym)
        return Sequence(out)

    def _parse_annotated_token(self, tok: str) -> TemporalSymbol:
        """Parse one whitespace-delimited token into a TemporalSymbol.

        Annotations (all optional, can be combined):
            form@magnitude    e.g. rising@1.2
            form:duration     e.g. lag:2
            form/phase        e.g. cycle/0.25
            form<             mark direction=BACKWARD
            form|             mark direction=NEUTRAL
        """
        direction = TemporalDirection.FORWARD
        phase = 0.0
        duration = 0.0
        magnitude = 0.0

        work = tok

        # Direction suffixes
        if work.endswith("<"):
            direction = TemporalDirection.BACKWARD
            work = work[:-1]
        elif work.endswith("|"):
            direction = TemporalDirection.NEUTRAL
            work = work[:-1]

        # Phase annotation
        if "/" in work:
            work, phase_s = work.split("/", 1)
            try:
                phase = float(phase_s)
            except ValueError:
                phase = 0.0

        # Duration annotation
        if ":" in work:
            work, dur_s = work.split(":", 1)
            try:
                duration = float(dur_s)
            except ValueError:
                duration = 0.0

        # Magnitude annotation
        if "@" in work:
            work, mag_s = work.split("@", 1)
            try:
                magnitude = float(mag_s)
            except ValueError:
                magnitude = 0.0

        base_form = work.lower()
        base = self._inventory.get(base_form)
        if base is not None and isinstance(base, TemporalSymbol):
            # Clone the template atom and overlay annotations. Form
            # is the BARE category name (``base.form``) so grammar
            # productions match by form; the magnitude / duration /
            # phase live in the dataclass fields and the raw token
            # is preserved in features for round-trip rendering.
            return TemporalSymbol(
                form=base.form,
                domain="temporal",
                valence=base.valence,
                category=base.category,
                direction=direction if direction != TemporalDirection.FORWARD
                          else base.direction,
                magnitude=magnitude if magnitude else base.magnitude,
                duration=duration if duration else base.duration,
                phase=phase if phase else base.phase,
                features={"raw": tok},
            )

        # Unknown token — fall back to an untyped LEVEL atom. Form
        # is still ``level`` for grammar matching; the original token
        # is preserved in features.
        return TemporalSymbol(
            form="level",
            domain="temporal",
            valence=2,
            category=TemporalCategory.LEVEL,
            direction=direction,
            magnitude=magnitude,
            duration=duration,
            phase=phase,
            features={"unknown": "true", "raw": tok},
        )

    # -- decoding -----------------------------------------------------------

    def decode(self, sequence: Sequence) -> str:
        """Render a Sequence of TemporalSymbols back to the compact text
        notation the encoder accepts.
        """
        parts: List[str] = []
        for sym in sequence.symbols:
            if isinstance(sym, TemporalSymbol):
                parts.append(self._render_symbol(sym))
            else:
                parts.append(sym.form)
        return " ".join(parts)

    def _render_symbol(self, sym: TemporalSymbol) -> str:
        # If we recorded the raw token in features, use it verbatim
        # for a clean round-trip.
        raw = sym.features.get("raw") if sym.features else None
        if raw:
            return raw
        form = sym.form
        # If the stored form already has annotations, trust it.
        if any(ch in form for ch in "@:/"):
            return form
        parts = [form]
        if sym.magnitude:
            parts.append(f"@{sym.magnitude:g}")
        if sym.duration:
            parts.append(f":{sym.duration:g}")
        if sym.phase:
            parts.append(f"/{sym.phase:g}")
        out = "".join(parts)
        if sym.direction == TemporalDirection.BACKWARD:
            out += "<"
        elif sym.direction == TemporalDirection.NEUTRAL and not sym.is_self_referential:
            out += "|"
        return out

    # -- convenience helpers ------------------------------------------------

    def make_level(self, value: float, index: int = 0) -> TemporalSymbol:
        """Build a LEVEL atom programmatically."""
        return TemporalSymbol(
            form=f"level@{value:g}",
            domain="temporal",
            valence=2,
            category=TemporalCategory.LEVEL,
            direction=TemporalDirection.NEUTRAL,
            magnitude=value,
            features={"index": str(index)},
        )

    def make_shock(self, magnitude: float, direction: TemporalDirection = TemporalDirection.FORWARD) -> TemporalSymbol:
        """Build a SHOCK atom programmatically."""
        sign = "spike" if magnitude >= 0 else "drop"
        return TemporalSymbol(
            form=f"{sign}@{abs(magnitude):g}",
            domain="temporal",
            valence=2,
            category=TemporalCategory.SHOCK,
            direction=direction,
            magnitude=magnitude,
        )

    def make_fixed_point(self, label: str = "fixed") -> TemporalSymbol:
        """Build a FIXED_POINT atom — a strange-loop anchor on time.

        Fixed points are neutral in direction because they are their
        own past and future. The strange-loop detection layer in
        ``glm/core/grammar.py`` latches onto fixed points when it
        scans for self-referential patterns in the temporal substrate.
        """
        return TemporalSymbol(
            form=label,
            domain="temporal",
            valence=2,
            category=TemporalCategory.FIXED_POINT,
            direction=TemporalDirection.NEUTRAL,
        )

    # -- summary ------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Describe the substrate for introspection / context dicts."""
        by_category: Dict[str, int] = {}
        for sym in self._inventory.values():
            if isinstance(sym, TemporalSymbol):
                key = sym.category.name.lower()
                by_category[key] = by_category.get(key, 0) + 1
        return {
            "name": self.name,
            "domain": self.domain,
            "num_atoms": len(self._inventory),
            "atoms_by_category": by_category,
            "features": {k: sorted(list(v)) for k, v in self._feature_system.items()},
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    sub = TemporalSubstrate()
    print("Temporal substrate summary:")
    print(json.dumps(sub.summary(), indent=2, sort_keys=True))

    # Round-trip a handful of real examples so the encoder and decoder
    # don't drift apart on us.
    examples = [
        "baseline rising@1.2 peak falling@0.8 baseline",
        "event lag:2 event lag:2 event",
        "regime:steady shock@1.0 regime:volatile",
        "1.0 1.1 1.15 1.22 1.18 1.05 0.95",
        "1.0, 1.1, 1.15, 1.22, 1.18",
        "fixed invariant baseline",
        "cycle/0.0 cycle/0.25 cycle/0.5 cycle/0.75",
        "hypothesis past now future",
    ]
    print("\nRound-trip tests:")
    for raw in examples:
        seq = sub.encode(raw)
        rendered = sub.decode(seq)
        print(f"  in : {raw}")
        print(f"  out: {rendered}")
        print(f"  n_atoms: {len(seq.symbols)}")
        if seq.symbols:
            first = seq.symbols[0]
            if isinstance(first, TemporalSymbol):
                print(f"  first atom: category={first.category.name} "
                      f"direction={first.direction.name} "
                      f"magnitude={first.magnitude:g}")
        print()

    # Verify fixed-point detection surface for the strange-loop layer
    fp = sub.make_fixed_point("F")
    assert fp.is_self_referential, "fixed point should be self-referential"
    print(f"make_fixed_point('F'): form={fp.form} "
          f"is_self_referential={fp.is_self_referential}")

    # Verify shock construction
    up_shock = sub.make_shock(0.5)
    down_shock = sub.make_shock(-0.3)
    print(f"make_shock(+0.5): form={up_shock.form} category={up_shock.category.name}")
    print(f"make_shock(-0.3): form={down_shock.form} category={down_shock.category.name}")

    print("\nTemporal substrate self-test: OK")
