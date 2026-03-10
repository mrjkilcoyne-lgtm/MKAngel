"""Physics grammars — mechanics, electromagnetism, thermodynamics, quantum.

Physics is the grammar of reality: the laws that describe how matter, energy,
space, and time interact.  Every physical law is a production rule that
transforms initial conditions into predictions.  Conservation laws are
invariants — quantities that remain unchanged under transformation, like
the deep structure beneath surface variation.

The strange loop: an observer measuring a quantum system is itself a
quantum system.  The measurement problem — the grammar trying to
describe the process of grammar application.

Isomorphisms:
- Lagrangian mechanics ↔ variational calculus (physics as optimisation)
- Symmetry groups ↔ conservation laws (Noether's theorem)
- Wave equations ↔ vibrating strings ↔ sound ↔ phonology
- Quantum superposition ↔ ambiguous parse trees (multiple derivations)
- Entropy ↔ information ↔ compression (Shannon meets Boltzmann)
"""

from glm.core.grammar import Rule, Production, Grammar, StrangeLoop


# ---------------------------------------------------------------------------
# Classical mechanics grammar
# ---------------------------------------------------------------------------

def build_mechanics_grammar() -> Grammar:
    """Build a grammar for classical mechanics.

    Newton's laws are the axioms; equations of motion are derivations.
    Conservation laws are the invariants that survive all transformations.
    Lagrangian and Hamiltonian mechanics reveal the deep structure beneath
    Newtonian surface forms.
    """

    productions = [
        # --- Kinematic quantities ---
        Production("Position", ["Vector3D"], "mechanics"),
        Production("Velocity", ["D", "Position", "SLASH", "D", "Time"], "mechanics"),
        Production("Acceleration", ["D", "Velocity", "SLASH", "D", "Time"], "mechanics"),
        Production("Vector3D", ["Component", "IHAT", "PLUS", "Component", "JHAT", "PLUS", "Component", "KHAT"], "mechanics"),

        # --- Dynamic quantities ---
        Production("Force", ["Mass", "TIMES", "Acceleration"], "mechanics"),
        Production("Momentum", ["Mass", "TIMES", "Velocity"], "mechanics"),
        Production("KineticEnergy", ["HALF", "Mass", "TIMES", "Velocity", "SQUARED"], "mechanics"),
        Production("PotentialEnergy", ["Mass", "TIMES", "G", "TIMES", "Height"], "mechanics"),
        Production("Work", ["Force", "DOT", "Displacement"], "mechanics"),

        # --- Lagrangian / Hamiltonian ---
        Production("Lagrangian", ["KineticEnergy", "MINUS", "PotentialEnergy"], "mechanics"),
        Production("Hamiltonian", ["KineticEnergy", "PLUS", "PotentialEnergy"], "mechanics"),
        Production("Action", ["INT", "Lagrangian", "D", "Time"], "mechanics"),
    ]

    rules = [
        Rule(
            name="newtons_first_law",
            pattern={"condition": "net force = 0"},
            result={"consequence": "velocity is constant (including zero)",
                    "description": "inertia — objects resist change in motion",
                    "isomorphism": "like linguistic conservatism: forms persist without pressure"},
            weight=1.0,
            direction="forward",
        ),
        Rule(
            name="newtons_second_law",
            pattern={"equation": "F = ma"},
            result={"description": "force equals mass times acceleration",
                    "differential_form": "F = dp/dt (rate of change of momentum)",
                    "significance": "the fundamental production rule of mechanics"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="newtons_third_law",
            pattern={"action": "body A exerts force F on body B"},
            result={"reaction": "body B exerts force −F on body A",
                    "description": "every action has an equal and opposite reaction",
                    "isomorphism": "like bidirectional grammar rules: forward and backward"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Conservation laws ---
        Rule(
            name="conservation_energy",
            pattern={"system": "isolated system"},
            result={"invariant": "total energy E = T + V is constant",
                    "noether": "time translation symmetry → energy conservation",
                    "isomorphism": "like conservation of morphological structure under derivation"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="conservation_momentum",
            pattern={"system": "isolated system"},
            result={"invariant": "total momentum p = Σ mᵢvᵢ is constant",
                    "noether": "spatial translation symmetry → momentum conservation"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="conservation_angular_momentum",
            pattern={"system": "rotationally symmetric system"},
            result={"invariant": "total angular momentum L = r × p is constant",
                    "noether": "rotational symmetry → angular momentum conservation"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Noether's theorem (the deep grammar of physics) ---
        Rule(
            name="noethers_theorem",
            pattern={"symmetry": "continuous symmetry of the Lagrangian"},
            result={"consequence": "corresponding conserved quantity exists",
                    "examples": [
                        "time symmetry → energy",
                        "space symmetry → momentum",
                        "rotation symmetry → angular momentum",
                        "gauge symmetry → charge",
                    ],
                    "significance": "the deepest grammar rule in physics: symmetry = conservation"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Gravitation ---
        Rule(
            name="universal_gravitation",
            pattern={"bodies": "two masses m₁, m₂ at distance r"},
            result={"force": "F = G × m₁ × m₂ / r²",
                    "direction": "attractive, along line joining centres"},
            weight=1.0,
            direction="forward",
        ),

        # --- Euler-Lagrange ---
        Rule(
            name="euler_lagrange",
            pattern={"variational": "δS = 0 (stationary action)"},
            result={"equation": "d/dt (∂L/∂q̇) − ∂L/∂q = 0",
                    "description": "equations of motion from the Lagrangian",
                    "significance": "all of classical mechanics from one principle"},
            weight=0.9,
            direction="forward",
        ),
    ]

    return Grammar(
        name="classical_mechanics",
        domain="physics",
        rules=rules,
        productions=productions,
    )


# ---------------------------------------------------------------------------
# Electromagnetism grammar
# ---------------------------------------------------------------------------

def build_electromagnetism_grammar() -> Grammar:
    """Build a grammar for electromagnetism — Maxwell's equations and beyond.

    Maxwell's equations are four production rules that generate all
    electromagnetic phenomena: light, radio waves, magnetism, circuits.
    They unify electricity and magnetism as aspects of a single field.
    """

    productions = [
        Production("ElectricField", ["Charge", "SLASH", "EPSILON0", "TIMES", "R_SQUARED"], "em"),
        Production("MagneticField", ["MU0", "TIMES", "Current", "TIMES", "DL", "CROSS", "RHAT", "SLASH", "R_SQUARED"], "em"),
        Production("EMWave", ["ElectricField", "CROSS", "MagneticField"], "em"),
        Production("PoyntingVector", ["E", "CROSS", "B", "SLASH", "MU0"], "em"),
    ]

    rules = [
        Rule(
            name="gauss_law_electric",
            pattern={"maxwell": "∇ · E"},
            result={"equals": "ρ / ε₀",
                    "description": "electric charges are sources of electric field",
                    "integral_form": "∮ E · dA = Q_enc / ε₀"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="gauss_law_magnetic",
            pattern={"maxwell": "∇ · B"},
            result={"equals": "0",
                    "description": "no magnetic monopoles — field lines always close",
                    "isomorphism": "like conservation of constituency in syntax"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="faraday_law",
            pattern={"maxwell": "∇ × E"},
            result={"equals": "−∂B/∂t",
                    "description": "changing magnetic field induces electric field",
                    "integral_form": "emf = −dΦ_B/dt"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="ampere_maxwell_law",
            pattern={"maxwell": "∇ × B"},
            result={"equals": "μ₀(J + ε₀ ∂E/∂t)",
                    "description": "currents and changing E-fields create magnetic fields",
                    "displacement_current": "ε₀ ∂E/∂t is Maxwell's key addition"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Electromagnetic waves ---
        Rule(
            name="em_wave_equation",
            pattern={"derivation": "combine Faraday + Ampère-Maxwell"},
            result={"wave_equation": "∇²E = μ₀ε₀ ∂²E/∂t²",
                    "speed": "c = 1/√(μ₀ε₀) ≈ 3 × 10⁸ m/s",
                    "significance": "light is an electromagnetic wave",
                    "isomorphism": "like sound waves in phonology — same wave grammar, different medium"},
            weight=0.95,
            direction="forward",
        ),

        # --- Lorentz force ---
        Rule(
            name="lorentz_force",
            pattern={"charge": "q moving at velocity v in fields E, B"},
            result={"force": "F = q(E + v × B)",
                    "description": "unifies electric and magnetic forces on charges"},
            weight=1.0,
            direction="forward",
        ),

        # --- Coulomb's law ---
        Rule(
            name="coulombs_law",
            pattern={"charges": "q₁, q₂ at distance r"},
            result={"force": "F = kq₁q₂/r²",
                    "isomorphism": "same r² law as gravity — deep structural parallel"},
            weight=1.0,
            direction="forward",
        ),
    ]

    return Grammar(
        name="electromagnetism",
        domain="physics",
        rules=rules,
        productions=productions,
    )


# ---------------------------------------------------------------------------
# Thermodynamics grammar
# ---------------------------------------------------------------------------

def build_thermodynamics_grammar() -> Grammar:
    """Build a grammar for thermodynamics and statistical mechanics.

    The laws of thermodynamics are the grammar of heat, work, and entropy.
    Entropy is information: Shannon entropy and Boltzmann entropy are the
    same grammar applied to different substrates.

    The strange loop: Maxwell's demon — an entity that could decrease
    entropy by knowing individual particle states, but the information
    processing itself generates entropy.  Information and physics are
    inseparable.
    """

    rules = [
        Rule(
            name="zeroth_law",
            pattern={"condition": "A in equilibrium with C, B in equilibrium with C"},
            result={"conclusion": "A in equilibrium with B",
                    "description": "transitivity of thermal equilibrium defines temperature",
                    "isomorphism": "like transitive entailment in logic"},
            weight=1.0,
            direction="forward",
        ),
        Rule(
            name="first_law",
            pattern={"system": "thermodynamic system"},
            result={"law": "dU = δQ − δW",
                    "description": "energy is conserved: internal energy change = heat in − work out",
                    "isomorphism": "like conservation of morphological features in derivation"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="second_law",
            pattern={"process": "any thermodynamic process"},
            result={"law": "ΔS_universe ≥ 0",
                    "description": "entropy of isolated system never decreases",
                    "statistical": "systems evolve toward macrostates with more microstates",
                    "information": "equivalent to: information is never created from nothing"},
            weight=1.0,
            direction="forward",
        ),
        Rule(
            name="third_law",
            pattern={"limit": "T → 0"},
            result={"law": "S → 0 (for perfect crystal)",
                    "description": "absolute zero is unattainable",
                    "consequence": "finite steps cannot reach T = 0"},
            weight=1.0,
            direction="forward",
        ),

        # --- Boltzmann entropy ---
        Rule(
            name="boltzmann_entropy",
            pattern={"macrostate": "Ω microstates"},
            result={"entropy": "S = k_B × ln(Ω)",
                    "description": "entropy counts the number of microstates",
                    "isomorphism": "Shannon entropy H = −Σ p log p (information = physics)"},
            weight=0.95,
            direction="bidirectional",
        ),

        # --- Ideal gas ---
        Rule(
            name="ideal_gas_law",
            pattern={"gas": "ideal gas"},
            result={"equation": "PV = nRT",
                    "description": "pressure × volume = amount × gas constant × temperature"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Carnot efficiency ---
        Rule(
            name="carnot_efficiency",
            pattern={"engine": "heat engine between T_hot and T_cold"},
            result={"max_efficiency": "η = 1 − T_cold/T_hot",
                    "description": "no engine can exceed Carnot efficiency",
                    "isomorphism": "like channel capacity in information theory"},
            weight=0.9,
            direction="forward",
        ),

        # --- Maxwell's demon ---
        Rule(
            name="maxwells_demon",
            pattern={"thought_experiment": "entity sorting fast/slow molecules"},
            result={"resolution": [
                "The demon must measure (acquire information)",
                "Information storage requires physical memory",
                "Erasing memory (Landauer's principle) generates k_B T ln 2 heat per bit",
                "Total entropy still increases",
            ],
                    "significance": "information processing is physical — computation has thermodynamic cost",
                    "strange_loop": "the observer is part of the system being observed"},
            conditions={"thought_experiment": True},
            weight=0.85,
            direction="bidirectional",
            self_referential=True,
        ),
    ]

    productions = [
        Production("ThermodynamicState", ["Pressure", "COMMA", "Volume", "COMMA", "Temperature"], "thermo"),
        Production("Process", ["State1", "ARROW", "State2"], "thermo"),
        Production("Cycle", ["Process", "Process", "Process", "Process"], "thermo"),
    ]

    return Grammar(
        name="thermodynamics",
        domain="physics",
        rules=rules,
        productions=productions,
    )


# ---------------------------------------------------------------------------
# Quantum mechanics grammar
# ---------------------------------------------------------------------------

def build_quantum_grammar() -> Grammar:
    """Build a grammar for quantum mechanics.

    Quantum mechanics is the grammar of the very small.  States are
    vectors in Hilbert space.  Observables are operators.  Measurement
    is projection.  Superposition is ambiguity — like an ambiguous
    parse that collapses to one reading when you observe it.

    The strange loop: the measurement problem.  The observer (a quantum
    system) measures another quantum system, and the act of measurement
    changes both.  The grammar of observation is entangled with the
    grammar of the thing observed.
    """

    productions = [
        # --- State notation ---
        Production("QuantumState", ["KET", "Label", "RANGLE"], "quantum"),
        Production("QuantumState", ["Alpha", "KET", "Label", "RANGLE", "PLUS",
                                     "Beta", "KET", "Label", "RANGLE"], "quantum"),
        Production("DualState", ["LANGLE", "Label", "BRA"], "quantum"),
        Production("InnerProduct", ["LANGLE", "Label", "PIPE", "Label", "RANGLE"], "quantum"),

        # --- Operators ---
        Production("Operator", ["Observable", "KET", "Label", "RANGLE"], "quantum"),
        Production("Commutator", ["LBRACKET", "Operator", "COMMA", "Operator", "RBRACKET"], "quantum"),

        # --- Equations ---
        Production("SchrodingerEq", ["IHBAR", "PARTIAL", "PSI", "SLASH", "PARTIAL", "T",
                                      "EQUALS", "H", "PSI"], "quantum"),
    ]

    rules = [
        # --- Superposition ---
        Rule(
            name="superposition_principle",
            pattern={"states": "|ψ₁⟩, |ψ₂⟩ are valid states"},
            result={"combined": "α|ψ₁⟩ + β|ψ₂⟩ is also a valid state",
                    "description": "quantum states can be added — linear combination",
                    "isomorphism": "like ambiguous parses: multiple derivations coexist until observed"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Schrödinger equation ---
        Rule(
            name="schrodinger_equation",
            pattern={"equation": "iℏ ∂|ψ⟩/∂t = Ĥ|ψ⟩"},
            result={"description": "time evolution of quantum state",
                    "significance": "the fundamental equation of quantum mechanics",
                    "isomorphism": "like the derivation engine: given initial state, generate future"},
            weight=1.0,
            direction="forward",
        ),

        # --- Uncertainty principle ---
        Rule(
            name="heisenberg_uncertainty",
            pattern={"observables": "position x, momentum p"},
            result={"inequality": "Δx × Δp ≥ ℏ/2",
                    "generalised": "ΔA × ΔB ≥ |⟨[Â,B̂]⟩|/2",
                    "description": "complementary observables cannot both be precisely known",
                    "isomorphism": "like the observer effect in linguistics: measuring changes speech"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Measurement ---
        Rule(
            name="measurement_postulate",
            pattern={"state": "α|a⟩ + β|b⟩"},
            result={"measurement": "collapses to |a⟩ with probability |α|², |b⟩ with |β|²",
                    "description": "measurement projects superposition onto eigenstate",
                    "strange_loop": "the measuring apparatus is itself a quantum system"},
            weight=1.0,
            direction="forward",
            self_referential=True,
        ),

        # --- Entanglement ---
        Rule(
            name="quantum_entanglement",
            pattern={"state": "|Ψ⟩ = (|↑↓⟩ − |↓↑⟩)/√2"},
            result={"properties": [
                "Cannot be written as |ψ_A⟩ ⊗ |ψ_B⟩",
                "Measuring A instantly determines B",
                "No information transmitted faster than light",
                "Bell's theorem: no local hidden variables",
            ],
                    "description": "non-separable quantum correlations",
                    "isomorphism": "like long-distance agreement in grammar (subject-verb across clauses)"},
            weight=0.95,
            direction="bidirectional",
        ),

        # --- Pauli exclusion ---
        Rule(
            name="pauli_exclusion",
            pattern={"particles": "identical fermions"},
            result={"principle": "no two fermions can occupy the same quantum state",
                    "consequence": "electron shell structure → periodic table → chemistry",
                    "isomorphism": "like the obligatory contour principle in phonology"},
            weight=0.95,
            direction="forward",
        ),

        # --- Dirac equation ---
        Rule(
            name="dirac_equation",
            pattern={"equation": "(iγᵘ∂ᵘ − m)ψ = 0"},
            result={"description": "relativistic quantum equation for spin-1/2 particles",
                    "predictions": [
                        "electron spin emerges naturally",
                        "antimatter (positron) predicted",
                        "correct magnetic moment",
                    ],
                    "significance": "unifies quantum mechanics and special relativity"},
            weight=0.85,
            direction="forward",
        ),

        # --- Wave-particle duality ---
        Rule(
            name="wave_particle_duality",
            pattern={"entity": "quantum object (photon, electron, etc.)"},
            result={"duality": [
                "Behaves as wave in propagation (interference, diffraction)",
                "Behaves as particle in detection (localised clicks)",
                "de Broglie: λ = h/p",
                "Complementarity: wave and particle are complementary descriptions",
            ],
                    "isomorphism": "like phoneme vs allophone: same entity, different contexts reveal different aspects"},
            weight=0.9,
            direction="bidirectional",
        ),
    ]

    strange_loop = StrangeLoop(
        entry_rule="measurement_postulate",
        cycle=[
            "observer_is_quantum_system",
            "quantum_system_measures_quantum_system",
            "measurement_changes_both_systems",
        ],
        level_shift="observer_observed_entanglement",
    )

    return Grammar(
        name="quantum_mechanics",
        domain="physics",
        rules=rules,
        productions=productions,
        sub_grammars=[strange_loop],
    )


# ---------------------------------------------------------------------------
# Relativity grammar
# ---------------------------------------------------------------------------

def build_relativity_grammar() -> Grammar:
    """Build a grammar for special and general relativity.

    Relativity is the grammar of spacetime: the rules that describe how
    space and time transform between observers.  Special relativity unifies
    space and time into spacetime.  General relativity says mass-energy
    curves spacetime, and curvature is gravity.
    """

    rules = [
        # --- Special relativity ---
        Rule(
            name="speed_of_light_invariance",
            pattern={"postulate": "speed of light in vacuum"},
            result={"principle": "c is the same for all inertial observers",
                    "consequence": "space and time are relative, c is absolute"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="time_dilation",
            pattern={"observer": "moving at velocity v relative to rest frame"},
            result={"formula": "Δt' = γΔt, where γ = 1/√(1 − v²/c²)",
                    "description": "moving clocks run slower"},
            weight=1.0,
            direction="forward",
        ),
        Rule(
            name="length_contraction",
            pattern={"object": "moving at velocity v"},
            result={"formula": "L' = L/γ",
                    "description": "moving objects are shorter in direction of motion"},
            weight=1.0,
            direction="forward",
        ),
        Rule(
            name="mass_energy_equivalence",
            pattern={"equation": "E = mc²"},
            result={"description": "mass and energy are interconvertible",
                    "full_form": "E² = (pc)² + (mc²)²",
                    "significance": "the most famous equation in physics"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- General relativity ---
        Rule(
            name="einstein_field_equations",
            pattern={"equation": "Gμν + Λgμν = (8πG/c⁴)Tμν"},
            result={"description": "spacetime curvature (left) = mass-energy content (right)",
                    "wheeler": "spacetime tells matter how to move; matter tells spacetime how to curve",
                    "significance": "gravity is not a force but geometry"},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="equivalence_principle",
            pattern={"principle": "gravity and acceleration are locally indistinguishable"},
            result={"consequence": "gravitational mass = inertial mass",
                    "description": "the foundation of general relativity",
                    "isomorphism": "like phonological neutralisation: distinct inputs, same output"},
            weight=1.0,
            direction="bidirectional",
        ),
    ]

    productions = [
        Production("SpacetimeInterval", ["C_SQUARED", "TIMES", "DT_SQUARED", "MINUS",
                                          "DX_SQUARED", "MINUS", "DY_SQUARED", "MINUS", "DZ_SQUARED"], "relativity"),
        Production("LorentzTransform", ["GAMMA", "TIMES", "LPAREN", "X", "MINUS", "V", "TIMES", "T", "RPAREN"], "relativity"),
        Production("Metric", ["G_MU_NU", "TIMES", "DX_MU", "TIMES", "DX_NU"], "relativity"),
    ]

    return Grammar(
        name="relativity",
        domain="physics",
        rules=rules,
        productions=productions,
    )
