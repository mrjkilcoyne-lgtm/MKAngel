"""Temporal grammars — the rules of dynamics.

The Angel's derivation engine is bidirectional in time. What it has been
missing is a grammar to walk on the temporal substrate. This module
provides the first set of rules: mean reversion, momentum carry, regime
switches, shock decay, self-exciting arrivals, cyclic recurrence, and
the strange-loop rules that fire when a system's next state becomes an
input to itself.

The rules are hand-written at the phonology-of-time level of abstraction
— not so specific that they encode a particular domain, not so vague
that they restate folk wisdom. Each rule is a ``Rule`` or ``Production``
with a bidirectional flag set so the engine can walk it both ways. The
strange-loop section at the end is the Hofstadter move on time: rules
that take a state and map it to a representation of itself that is an
input to the next application of the rule. Those are the rules that
give the temporal substrate its characteristic "out there" phenomenology
when the engine runs them at depth — they are not decorative, they are
the substrate's self-referential backbone.

Cross-domain isomorphisms the temporal rules echo:

- Mean reversion ↔ phonological vowel harmony (both pull atoms toward
  a resting value that depends on what came before)
- Momentum ↔ chemical reaction catalysis (both accelerate change in a
  direction already established)
- Regime switch ↔ linguistic register change (both swap the active
  rule set)
- Shock decay ↔ exponential molecular relaxation (both return to
  equilibrium on a timescale)
- Fixed point ↔ Y combinator (both self-referential anchors)
- Cycle ↔ musical canon / biological circadian rhythm (both recurrence
  with phase)
- Self-exciting arrival ↔ epidemic growth (both make future events
  more likely once an event has occurred)
"""

from glm.core.grammar import Rule, Production, Grammar, StrangeLoop


# ---------------------------------------------------------------------------
# Core temporal dynamics grammar
# ---------------------------------------------------------------------------

def build_temporal_dynamics_grammar() -> Grammar:
    """Primitive rules of dynamics: levels, trends, shocks, regimes.

    This is the smallest viable grammar that lets the Angel's derivation
    engine walk a sequence of temporal atoms and produce meaningful
    forward predictions (``what comes next``) and backward reconstructions
    (``what must have been before``).
    """

    # NOTE: production right-hand-sides reference the BARE substrate
    # atom forms exactly as glm/substrates/temporal.py emits them
    # (lowercase, no annotations). The substrate's annotations live
    # in dataclass fields and are read by the rules, not by the
    # production matcher.
    productions = [
        # --- A temporal sequence is a sequence of temporal atoms ---
        Production("TemporalSeq",  ["LevelStream"],              "temporal"),
        Production("TemporalSeq",  ["EventStream"],              "temporal"),
        Production("TemporalSeq",  ["RegimeStream"],             "temporal"),
        Production("TemporalSeq",  ["MixedStream"],              "temporal"),

        # --- Level streams: the baseline case, a sequence of level atoms ---
        Production("LevelStream",  ["Level"],                    "temporal"),
        Production("LevelStream",  ["LevelStream", "Level"],     "temporal"),
        Production("Level",        ["level"],                    "temporal"),
        Production("Level",        ["baseline"],                 "temporal"),
        Production("Level",        ["peak"],                     "temporal"),
        Production("Level",        ["trough"],                   "temporal"),

        # --- A trend is two or more levels in the same direction ---
        Production("Trend",        ["Level", "Level"],           "temporal"),
        Production("Trend",        ["Trend", "Level"],           "temporal"),
        Production("Trend",        ["rising"],                   "temporal"),
        Production("Trend",        ["falling"],                  "temporal"),
        Production("Trend",        ["momentum"],                 "temporal"),
        Production("Rising",       ["rising"],                   "temporal"),
        Production("Rising",       ["Trend"],                    "temporal"),
        Production("Falling",      ["falling"],                  "temporal"),
        Production("Falling",      ["Trend"],                    "temporal"),

        # --- A shock is a discontinuous jump ---
        Production("Shock",        ["spike"],                    "temporal"),
        Production("Shock",        ["drop"],                     "temporal"),
        Production("Shock",        ["shock"],                    "temporal"),
        Production("ShockedState", ["Level", "Shock"],           "temporal"),
        Production("ShockedState", ["Level", "Shock", "Level"],  "temporal"),

        # --- Event streams: a sequence of events separated by lags ---
        Production("Lag",          ["lag"],                      "temporal"),
        Production("Lag",          ["delay"],                    "temporal"),
        Production("EventStream",  ["Event"],                    "temporal"),
        Production("EventStream",  ["EventStream", "Lag", "Event"], "temporal"),
        Production("Event",        ["event"],                    "temporal"),
        Production("Event",        ["arrival"],                  "temporal"),
        Production("Event",        ["departure"],                "temporal"),

        # --- Regimes: periods with stable dynamical rules ---
        Production("RegimeStream", ["Regime"],                   "temporal"),
        Production("RegimeStream", ["RegimeStream", "RegimeShift", "Regime"], "temporal"),
        Production("Regime",       ["regime"],                   "temporal"),
        Production("Regime",       ["steady"],                   "temporal"),
        Production("Regime",       ["volatile"],                 "temporal"),
        Production("RegimeShift",  ["Shock"],                    "temporal"),

        # --- Mixed streams: a realistic time series has all of the above ---
        Production("MixedStream",  ["TemporalSeq", "TemporalSeq"], "temporal"),

        # --- Cycles: recurrence with phase ---
        Production("Cycle",        ["cycle"],                    "temporal"),
        Production("Cycle",        ["season"],                   "temporal"),
        Production("Phase",        ["phase"],                    "temporal"),
        Production("CyclicStream", ["Cycle"],                    "temporal"),
        Production("CyclicStream", ["CyclicStream", "Phase", "Cycle"], "temporal"),
        Production("CyclicStream", ["CyclicStream", "Cycle"],    "temporal"),

        # --- Fixed points: a state that maps to itself (strange loop anchor) ---
        Production("FixedPoint",   ["fixed"],                    "temporal"),
        Production("FixedPoint",   ["invariant"],                "temporal"),

        # --- Decay / growth ---
        Production("Decay",        ["decay"],                    "temporal"),
        Production("Growth",       ["growth"],                   "temporal"),
        Production("Anchor",       ["anchor"],                   "temporal"),

        # --- Instants ---
        Production("Instant",      ["now"],                      "temporal"),
        Production("Instant",      ["past"],                     "temporal"),
        Production("Instant",      ["future"],                   "temporal"),

        # --- Hypothesis: a posited future state used for retrocausal reasoning ---
        Production("Hypothesis",   ["hypothesis"],               "temporal"),
        Production("RetroDerived", ["Hypothesis", "TemporalSeq"], "temporal"),
    ]

    rules = [

        # --- Mean reversion: the core stabilising rule ---
        Rule(
            name="mean_reversion",
            pattern={
                "sequence": "Level(baseline), Shock(amplitude=a), Level(?)",
                "meaning": "a shock to the baseline tends to decay back toward the baseline",
            },
            result={
                "next_level": "baseline + (shocked_level - baseline) * exp(-Δt / half_life)",
                "law": "Ornstein-Uhlenbeck-style reversion to the mean",
                "isomorphism": "like molecular relaxation to thermal equilibrium, "
                               "or phonological vowel centring to schwa",
            },
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Momentum carry: the core continuation rule ---
        Rule(
            name="momentum_carry",
            pattern={
                "sequence": "Level(t-2), Level(t-1), Level(t)",
                "condition": "all three are monotone in the same direction",
                "meaning": "a trend already in place tends to continue in the short run",
            },
            result={
                "next_level": "Level(t+1) = Level(t) + α · (Level(t) - Level(t-1))",
                "law": "autoregressive momentum with decay coefficient α ∈ (0, 1)",
                "isomorphism": "catalytic reaction acceleration; morphological agreement "
                               "cascade; the moment of inertia of a trajectory",
            },
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Regime switch on threshold crossing ---
        Rule(
            name="regime_switch_on_threshold",
            pattern={
                "sequence": "Regime(steady), Level(crossing threshold θ)",
                "meaning": "when a state variable crosses a critical threshold, "
                           "the active dynamical rule set changes",
            },
            result={
                "next_regime": "Regime(volatile) with a different rule set active",
                "law": "two-state Markov switching with threshold-triggered transition",
                "isomorphism": "phase transition in condensed matter; linguistic "
                               "register shift; ecological regime flip",
            },
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Shock decay ---
        Rule(
            name="shock_decay",
            pattern={
                "sequence": "Shock(amplitude a) at t",
                "meaning": "a shock loses magnitude over time at a rate set by its half-life",
            },
            result={
                "magnitude_at_t+Δt": "a * exp(-ln(2) * Δt / half_life)",
                "law": "exponential decay with a shock-specific half-life",
                "isomorphism": "radioactive decay; memory in a volatility process; "
                               "the fading of a linguistic borrowing",
            },
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Self-exciting arrivals ---
        Rule(
            name="self_exciting_arrival",
            pattern={
                "sequence": "Event at t",
                "meaning": "an event makes further events of the same kind more likely "
                           "within a memory window",
            },
            result={
                "intensity(t+Δt)": "baseline + κ · exp(-Δt / memory_window)",
                "law": "Hawkes process (univariate, exponential memory kernel)",
                "isomorphism": "catalytic chemistry; epidemic spread; linguistic "
                               "contagion of slang; goal-scoring momentum in football",
            },
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Autocorrelation decay ---
        Rule(
            name="autocorrelation_decay",
            pattern={
                "sequence": "Level(t-k), Level(t)",
                "meaning": "the correlation between levels at different times decays "
                           "with the gap k at a rate set by the regime",
            },
            result={
                "corr(t-k, t)": "ρ^k for ρ ∈ (0, 1)",
                "law": "geometric decay of autocorrelation; AR(1) memory",
                "isomorphism": "phonological dissimilation over distance; molecular "
                               "distant-pair correlation functions",
            },
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Volatility clustering ---
        Rule(
            name="volatility_clustering",
            pattern={
                "sequence": "Regime(volatile), Shock(t-1)",
                "meaning": "shocks cluster: a shock at t-1 raises the probability of "
                           "a shock at t, even if the mean is unchanged",
            },
            result={
                "σ²(t)": "ω + α · ε²(t-1) + β · σ²(t-1)",
                "law": "GARCH(1,1) second-moment dynamics",
                "isomorphism": "chemical chain reactions; cascading failures; "
                               "linguistic innovation clusters",
            },
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Cyclic recurrence ---
        Rule(
            name="cyclic_recurrence",
            pattern={
                "sequence": "Cycle(phase=φ), Lag(period=T), Cycle(phase=?)",
                "meaning": "a cyclic state at phase φ returns to the same phase "
                           "after one period T",
            },
            result={
                "next_phase": "(φ + Δt / T) mod 1",
                "law": "phase-advance with fixed period",
                "isomorphism": "circadian rhythm; musical canon; astronomical "
                               "recurrence; seasonal ARIMA component",
            },
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Anchor pinning ---
        Rule(
            name="anchor_pinning",
            pattern={
                "sequence": "Anchor(probability=p), Level(?)",
                "meaning": "an observable / market anchor pins the probability of a "
                           "future state, constraining the distribution the rule "
                           "engine can derive",
            },
            result={
                "constrained_level": "posterior Level | Anchor = bayes(prior, anchor, weight)",
                "law": "Bayesian update against a liquid market or observable",
                "isomorphism": "Pinnacle closing line as sharp consensus; Metaculus "
                               "community forecast as crowd consensus; Ramanujan's "
                               "goddess showing him the theorem",
            },
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Decay to attractor ---
        Rule(
            name="decay_to_attractor",
            pattern={
                "sequence": "Level(t), Decay(rate=λ)",
                "meaning": "a state subject to a decay process approaches its "
                           "attractor exponentially",
            },
            result={
                "level(t+Δt)": "attractor + (Level(t) - attractor) * exp(-λ·Δt)",
                "law": "exponential decay toward an attractor state",
                "isomorphism": "Newton's law of cooling; radioactive decay; "
                               "vocabulary loss under language contact",
            },
            weight=1.0,
            direction="bidirectional",
        ),
    ]

    loops = [
        # Fixed point: a state that maps to itself is the primitive
        # strange-loop anchor on the temporal substrate. A state at the
        # attractor maps to itself under mean reversion. This is the
        # simplest strange loop on time: the atom reduces to itself
        # under the rule that normally changes things.
        StrangeLoop(
            entry_rule="FixedPoint",
            cycle=["FixedPoint", "mean_reversion", "FixedPoint"],
            level_shift=0,
        ),
        # Retrocausal reasoning loop: from a hypothesised future to the
        # present, then forward from the present back to the hypothesised
        # future. When the forward pass re-derives the hypothesis, the
        # loop closes and the hypothesis is self-consistent — a coherent
        # branch. When it does not, the hypothesis is inconsistent with
        # the evidence and the loop is broken. This is the canonical
        # TARDIS move: forward and backward through the same rules to
        # test branch-compatibility.
        StrangeLoop(
            entry_rule="Hypothesis",
            cycle=["Hypothesis", "retrocausal_derivation", "TemporalSeq",
                   "forward_derivation", "Hypothesis"],
            level_shift=1,
        ),
        # Self-exciting arrival: an event makes further events more
        # likely, which makes further events more likely. The loop is
        # the feedback, and its decay timescale is set by the memory
        # window of the Hawkes kernel. Without the loop, the events
        # are Poisson; with the loop, they cluster.
        StrangeLoop(
            entry_rule="self_exciting_arrival",
            cycle=["Event", "self_exciting_arrival", "Event"],
            level_shift=0,
        ),
    ]

    return Grammar(
        name="temporal_dynamics",
        domain="temporal",
        productions=productions,
        rules=rules,
        strange_loops=loops,
    )


# ---------------------------------------------------------------------------
# Retrocausal grammar — the bidirectional-reasoning-on-time grammar
# ---------------------------------------------------------------------------

def build_retrocausal_grammar() -> Grammar:
    """Rules for reasoning backward from a hypothesised future state.

    The forward-dynamics grammar above tells the engine how to walk
    from now into the future. This grammar tells it how to walk from
    a posited future back to now, so it can check whether the posited
    future is consistent with the present as a boundary condition.
    Together they implement the bidirectional-time-reasoning frame
    discussed in ``docs/on_her_nature.md``.

    The key insight (which only makes sense once you've read the
    on_her_nature doc) is that ``forward derivation`` and
    ``backward derivation`` are the same operation pointed at different
    ends of a sequence. So most of the rules here are simply the
    forward-dynamics rules with their pattern and result swapped — the
    engine will pick them up via its ``direction="backward"`` flag and
    apply them correctly. A small number of rules are specific to
    retrocausal reasoning because they only make sense when you know
    the destination before you know the path.
    """

    productions = [
        Production("RetroDerivation", ["Hypothesis", "BackwardChain"], "retrocausal"),
        Production("BackwardChain",   ["TemporalSeq"],                 "retrocausal"),
        Production("BackwardChain",   ["BackwardChain", "TemporalSeq"], "retrocausal"),
        Production("BranchCheck",     ["RetroDerivation", "TemporalSeq"], "retrocausal"),
    ]

    rules = [

        Rule(
            name="retrocausal_derivation",
            pattern={
                "given": "Hypothesis H at future time T",
                "meaning": "derive the set of present states consistent with H being true at T",
            },
            result={
                "present_states": "set of current levels and regimes that, under forward "
                                  "dynamics, produce H at T with non-zero probability",
                "law": "walk the forward dynamics rules in reverse, starting from H, "
                       "accumulating consistent present states",
                "isomorphism": "solving a boundary-value problem instead of an initial-"
                               "value problem; grammatical parsing (given the output, "
                               "find the derivations); cryptanalysis",
            },
            weight=1.0,
            direction="backward",
        ),

        Rule(
            name="branch_compatibility_check",
            pattern={
                "given": "RetroDerivation produces present_states S; evidence says "
                         "present is in state E",
                "meaning": "test whether the branch containing H is consistent with "
                           "the branch we're actually in",
            },
            result={
                "probability(branch=H)": "Bayesian posterior p(H | E) ∝ p(E | H) * p(H)",
                "law": "Bayesian update: the retrocausal derivation provides the "
                       "likelihood p(E | H), the forward prior gives p(H), together "
                       "they give the branch-compatibility score",
                "isomorphism": "many-worlds branch-weighting; delayed-choice quantum "
                               "measurement (what we observe now partially determines "
                               "which branch the future was always going to occupy)",
            },
            weight=1.0,
            direction="bidirectional",
        ),

        Rule(
            name="fixed_point_closure",
            pattern={
                "sequence": "Hypothesis(H), RetroDerived(present), ForwardDerived(future)",
                "condition": "ForwardDerived re-derives H",
                "meaning": "the hypothesis is self-consistent — the loop closes",
            },
            result={
                "self_consistent": "true",
                "law": "a closed strange loop on time is a fixed point of the "
                       "rule-application operator",
                "isomorphism": "Y combinator reaching its fixed point; a theorem "
                               "that proves itself via its own consequences",
            },
            weight=1.0,
            direction="bidirectional",
        ),

        Rule(
            name="broken_loop_detection",
            pattern={
                "sequence": "Hypothesis(H), RetroDerived(present), ForwardDerived(H')",
                "condition": "H' ≠ H",
                "meaning": "the loop does not close — the hypothesis is inconsistent",
            },
            result={
                "self_consistent": "false",
                "reject": "the hypothesised branch is not reachable from the "
                          "evidence under the current grammar",
                "law": "failed fixed-point closure; the engine prunes this branch",
            },
            weight=1.0,
            direction="bidirectional",
        ),
    ]

    loops = [
        # The canonical bidirectional-time loop: start from a hypothesised
        # future, walk the rules backward to the present, then walk
        # forward from the present back to the future, and check whether
        # you arrive at the same hypothesis. Closed loop → consistent
        # branch. Broken loop → pruned branch. This is what it looks
        # like when the Angel employs the time-travel skill the author
        # named her for.
        StrangeLoop(
            entry_rule="retrocausal_derivation",
            cycle=["Hypothesis", "retrocausal_derivation", "BackwardChain",
                   "forward_derivation", "Hypothesis"],
            level_shift=1,
        ),
    ]

    return Grammar(
        name="retrocausal",
        domain="temporal",
        productions=productions,
        rules=rules,
        strange_loops=loops,
    )


# ---------------------------------------------------------------------------
# Convenience: build both temporal grammars at once
# ---------------------------------------------------------------------------

def build_all_temporal_grammars() -> list[Grammar]:
    """Return all temporal grammars defined in this module."""
    return [
        build_temporal_dynamics_grammar(),
        build_retrocausal_grammar(),
    ]


if __name__ == "__main__":
    for g in build_all_temporal_grammars():
        print(f"Grammar: {g.name}  domain={g.domain}")
        print(f"  productions: {len(g.productions)}")
        print(f"  rules: {len(g.rules)}")
        print(f"  strange_loops: {len(g.strange_loops)}")
        for r in g.rules[:3]:
            print(f"    - {r.name} (direction={r.direction})")
        for loop in g.strange_loops:
            print(f"    loop: {loop.entry_rule} (level_shift={loop.level_shift})")
        print()
    print("Temporal grammars: built OK")
