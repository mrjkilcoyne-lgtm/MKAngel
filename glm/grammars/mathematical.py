"""Mathematical grammars — algebra, calculus, logic, number theory, topology.

Mathematics is the purest grammar: symbols, rules of combination, and
derivation.  Every mathematical proof is a derivation tree.  Every equation
is a production.  Every theorem is a rule that can be applied forward
(to generate consequences) or backward (to find premises).

The strange loop: Gödel's incompleteness — a mathematical system that
constructs a sentence about its own provability.  The set of all sets
that don't contain themselves.  The liar paradox formalised.

Isomorphisms:
- Algebraic operations ↔ chemical reactions (both are transformations
  that conserve structure)
- Proof trees ↔ syntax trees ↔ derivation trees (all are hierarchical
  derivations)
- Group symmetries ↔ molecular symmetries ↔ phonological feature
  classes
- Topological invariants ↔ biological homology ↔ etymological cognates
  (things that stay the same under transformation)
"""

from glm.core.grammar import Rule, Production, Grammar, StrangeLoop


# ---------------------------------------------------------------------------
# Algebraic grammar
# ---------------------------------------------------------------------------

def build_algebra_grammar() -> Grammar:
    """Build a grammar for algebraic structures and manipulation.

    Covers: arithmetic, polynomials, group/ring/field axioms, equation
    solving, and symbolic manipulation.  Each production is a valid
    algebraic derivation step.
    """

    productions = [
        # --- Expression structure ---
        Production("MathExpr", ["Term"], "algebra"),
        Production("MathExpr", ["MathExpr", "AddOp", "Term"], "algebra"),
        Production("Term", ["Factor"], "algebra"),
        Production("Term", ["Term", "MulOp", "Factor"], "algebra"),
        Production("Factor", ["Atom"], "algebra"),
        Production("Factor", ["Factor", "CARET", "Atom"], "algebra"),
        Production("Factor", ["MINUS", "Factor"], "algebra"),
        Production("Atom", ["Number"], "algebra"),
        Production("Atom", ["Variable"], "algebra"),
        Production("Atom", ["LPAREN", "MathExpr", "RPAREN"], "algebra"),
        Production("Atom", ["FunctionCall"], "algebra"),

        # --- Operations ---
        Production("AddOp", ["PLUS"], "algebra"),
        Production("AddOp", ["MINUS"], "algebra"),
        Production("MulOp", ["TIMES"], "algebra"),
        Production("MulOp", ["DIVIDE"], "algebra"),

        # --- Functions ---
        Production("FunctionCall", ["FuncName", "LPAREN", "ArgList", "RPAREN"], "algebra"),
        Production("FuncName", ["SIN"], "algebra"),
        Production("FuncName", ["COS"], "algebra"),
        Production("FuncName", ["TAN"], "algebra"),
        Production("FuncName", ["LOG"], "algebra"),
        Production("FuncName", ["LN"], "algebra"),
        Production("FuncName", ["EXP"], "algebra"),
        Production("FuncName", ["SQRT"], "algebra"),
        Production("FuncName", ["ABS"], "algebra"),
        Production("ArgList", ["MathExpr"], "algebra"),
        Production("ArgList", ["MathExpr", "COMMA", "ArgList"], "algebra"),

        # --- Equations ---
        Production("Equation", ["MathExpr", "EQUALS", "MathExpr"], "algebra"),
        Production("Inequality", ["MathExpr", "LT", "MathExpr"], "algebra"),
        Production("Inequality", ["MathExpr", "GT", "MathExpr"], "algebra"),
        Production("Inequality", ["MathExpr", "LTE", "MathExpr"], "algebra"),
        Production("Inequality", ["MathExpr", "GTE", "MathExpr"], "algebra"),

        # --- Summation / product notation ---
        Production("Summation", ["SIGMA", "Variable", "EQUALS", "Number", "CARET", "Number", "MathExpr"], "algebra"),
        Production("Product", ["PI", "Variable", "EQUALS", "Number", "CARET", "Number", "MathExpr"], "algebra"),

        # --- Sets ---
        Production("Set", ["LBRACE", "SetElements", "RBRACE"], "algebra"),
        Production("SetElements", ["MathExpr"], "algebra"),
        Production("SetElements", ["MathExpr", "COMMA", "SetElements"], "algebra"),
        Production("SetOp", ["UNION"], "algebra"),
        Production("SetOp", ["INTERSECT"], "algebra"),
        Production("SetOp", ["SETMINUS"], "algebra"),

        # --- Matrices ---
        Production("Matrix", ["LBRACKET", "MatrixRows", "RBRACKET"], "algebra"),
        Production("MatrixRows", ["MatrixRow"], "algebra"),
        Production("MatrixRows", ["MatrixRow", "SEMICOLON", "MatrixRows"], "algebra"),
        Production("MatrixRow", ["MathExpr"], "algebra"),
        Production("MatrixRow", ["MathExpr", "COMMA", "MatrixRow"], "algebra"),
    ]

    rules = [
        # --- Commutativity ---
        Rule(
            name="commutative_addition",
            pattern={"operation": "a + b"},
            result={"equivalent": "b + a",
                    "axiom": "commutativity of addition",
                    "applies_to": ["integers", "rationals", "reals", "complex",
                                   "abelian groups", "rings", "fields"]},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="commutative_multiplication",
            pattern={"operation": "a × b"},
            result={"equivalent": "b × a",
                    "axiom": "commutativity of multiplication",
                    "caveat": "does NOT hold for matrices, quaternions, or non-abelian groups"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Associativity ---
        Rule(
            name="associative_addition",
            pattern={"operation": "(a + b) + c"},
            result={"equivalent": "a + (b + c)",
                    "axiom": "associativity of addition"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Distributivity ---
        Rule(
            name="distributive_law",
            pattern={"operation": "a × (b + c)"},
            result={"equivalent": "a × b + a × c",
                    "axiom": "distributivity of multiplication over addition",
                    "isomorphism": "like morphological agglutination distributing over stems"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Identity elements ---
        Rule(
            name="additive_identity",
            pattern={"operation": "a + 0"},
            result={"equivalent": "a", "axiom": "additive identity"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="multiplicative_identity",
            pattern={"operation": "a × 1"},
            result={"equivalent": "a", "axiom": "multiplicative identity"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Inverse elements ---
        Rule(
            name="additive_inverse",
            pattern={"operation": "a + (−a)"},
            result={"equivalent": "0", "axiom": "additive inverse"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="multiplicative_inverse",
            pattern={"operation": "a × (1/a)"},
            result={"equivalent": "1",
                    "axiom": "multiplicative inverse",
                    "condition": "a ≠ 0"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Exponent rules ---
        Rule(
            name="power_product",
            pattern={"operation": "aⁿ × aᵐ"},
            result={"equivalent": "aⁿ⁺ᵐ",
                    "axiom": "product of powers"},
            weight=0.95,
            direction="bidirectional",
        ),
        Rule(
            name="power_of_power",
            pattern={"operation": "(aⁿ)ᵐ"},
            result={"equivalent": "aⁿᵐ",
                    "axiom": "power of a power"},
            weight=0.95,
            direction="bidirectional",
        ),

        # --- Quadratic formula ---
        Rule(
            name="quadratic_formula",
            pattern={"equation": "ax² + bx + c = 0"},
            result={"solution": "x = (−b ± √(b² − 4ac)) / 2a",
                    "discriminant": "Δ = b² − 4ac",
                    "cases": {
                        "Δ > 0": "two distinct real roots",
                        "Δ = 0": "one repeated real root",
                        "Δ < 0": "two complex conjugate roots",
                    }},
            conditions={"a_nonzero": True},
            weight=0.9,
            direction="forward",
        ),

        # --- Group theory ---
        Rule(
            name="group_axioms",
            pattern={"structure": "(G, ∘)"},
            result={"axioms": [
                "closure: ∀a,b ∈ G: a ∘ b ∈ G",
                "associativity: (a ∘ b) ∘ c = a ∘ (b ∘ c)",
                "identity: ∃e ∈ G: a ∘ e = e ∘ a = a",
                "inverse: ∀a ∈ G: ∃a⁻¹: a ∘ a⁻¹ = e",
            ],
                    "isomorphism": "like phonological feature classes under composition"},
            conditions={"type": "group"},
            weight=0.85,
            direction="bidirectional",
        ),

        # --- Fundamental theorem of algebra ---
        Rule(name="fundamental_theorem_algebra",
             pattern={"polynomial": "non-constant polynomial with complex coefficients"},
             result={"theorem": "has at least one complex root",
                     "consequence": "degree-n polynomial has exactly n roots (counting multiplicity)",
                     "proved": "Gauss 1799 (first rigorous proof)"},
             weight=1.0, direction="forward"),
        # --- Lagrange's theorem ---
        Rule(name="lagranges_theorem_groups",
             pattern={"group": "finite group G, subgroup H"},
             result={"theorem": "|H| divides |G|",
                     "consequence": "order of any element divides |G|",
                     "corollary": "groups of prime order are cyclic"},
             weight=1.0, direction="forward"),
        # --- Cayley's theorem ---
        Rule(name="cayleys_theorem",
             pattern={"group": "any group G"},
             result={"theorem": "G is isomorphic to a subgroup of Sym(G)",
                     "meaning": "every abstract group can be realised as permutations"},
             weight=1.0, direction="forward"),
        # --- Ring and field axioms ---
        Rule(name="field_axioms",
             pattern={"structure": "(F, +, ×)"},
             result={"axioms": [
                 "abelian group under +",
                 "abelian group under × (excluding 0)",
                 "distributivity: a(b+c) = ab + ac",
             ],
                     "examples": "ℚ, ℝ, ℂ, 𝔽_p (finite fields)"},
             weight=1.0, direction="bidirectional"),
        # --- Eigenvalue equation ---
        Rule(name="eigenvalue_equation",
             pattern={"linear_map": "A: V → V"},
             result={"equation": "Av = λv",
                     "lambda": "eigenvalue", "v": "eigenvector (v ≠ 0)",
                     "char_poly": "det(A - λI) = 0",
                     "isomorphism": "quantum observables use same structure"},
             weight=1.0, direction="bidirectional"),
        # --- Cayley-Hamilton ---
        Rule(name="cayley_hamilton",
             pattern={"matrix": "n×n matrix A with characteristic polynomial p(λ)"},
             result={"theorem": "p(A) = 0 (matrix satisfies its own characteristic equation)",
                     "proved": "Cayley 1858, Hamilton 1853"},
             weight=1.0, direction="forward"),
        # --- Noether's theorem (mathematical statement) ---
        Rule(name="noethers_theorem_math",
             pattern={"system": "action integral invariant under continuous symmetry"},
             result={"theorem": "every continuous symmetry ↔ conservation law",
                     "examples": {
                         "time_translation": "→ energy conservation",
                         "space_translation": "→ momentum conservation",
                         "rotation": "→ angular momentum conservation",
                         "gauge_U1": "→ charge conservation",
                     },
                     "proved": "Emmy Noether 1918",
                     "significance": "deepest connection between symmetry and physics"},
             weight=1.0, direction="bidirectional"),
    ]

    return Grammar(
        name="algebra",
        domain="mathematical",
        rules=rules,
        productions=productions,
    )


# ---------------------------------------------------------------------------
# Calculus grammar
# ---------------------------------------------------------------------------

def build_calculus_grammar() -> Grammar:
    """Build a grammar for calculus — differentiation, integration, limits.

    Calculus is the grammar of change: derivatives describe instantaneous
    rates, integrals accumulate quantities, and limits define the boundary
    behaviour of sequences.

    The strange loop: the Fundamental Theorem of Calculus connects
    differentiation and integration as inverse operations — each
    undoes the other, like forward and backward derivation.
    """

    productions = [
        # --- Derivative notation ---
        Production("Derivative", ["D", "LBRACKET", "MathExpr", "RBRACKET"], "calculus"),
        Production("Derivative", ["D", "MathExpr", "SLASH", "D", "Variable"], "calculus"),
        Production("PartialDeriv", ["PARTIAL", "MathExpr", "SLASH", "PARTIAL", "Variable"], "calculus"),

        # --- Integral notation ---
        Production("Integral", ["INT", "MathExpr", "D", "Variable"], "calculus"),
        Production("DefIntegral", ["INT", "Number", "CARET", "Number", "MathExpr", "D", "Variable"], "calculus"),
        Production("MultiIntegral", ["IINT", "MathExpr", "D", "Variable", "D", "Variable"], "calculus"),

        # --- Limits ---
        Production("Limit", ["LIM", "Variable", "ARROW", "Value", "MathExpr"], "calculus"),

        # --- Series ---
        Production("Series", ["SIGMA", "N", "EQUALS", "Number", "CARET", "INF", "Term"], "calculus"),
        Production("TaylorSeries", ["SIGMA", "N", "EQUALS", "Number", "CARET", "INF",
                                     "Coefficient", "TIMES", "Variable", "CARET", "N"], "calculus"),

        # --- Differential equations ---
        Production("ODE", ["Derivative", "EQUALS", "MathExpr"], "calculus"),
        Production("PDE", ["PartialDeriv", "EQUALS", "MathExpr"], "calculus"),
    ]

    rules = [
        # --- Differentiation rules ---
        Rule(
            name="power_rule",
            pattern={"function": "xⁿ"},
            result={"derivative": "n × xⁿ⁻¹",
                    "isomorphism": "like sound change: systematic transformation of form"},
            weight=1.0,
            direction="forward",
        ),
        Rule(
            name="chain_rule",
            pattern={"function": "f(g(x))"},
            result={"derivative": "f'(g(x)) × g'(x)",
                    "description": "derivative of composition = outer' × inner'",
                    "isomorphism": "like morphological decomposition: derive each layer"},
            weight=1.0,
            direction="forward",
        ),
        Rule(
            name="product_rule",
            pattern={"function": "f(x) × g(x)"},
            result={"derivative": "f'(x) × g(x) + f(x) × g'(x)",
                    "description": "Leibniz rule"},
            weight=1.0,
            direction="forward",
        ),
        Rule(
            name="quotient_rule",
            pattern={"function": "f(x) / g(x)"},
            result={"derivative": "(f'g − fg') / g²",
                    "condition": "g(x) ≠ 0"},
            weight=0.95,
            direction="forward",
        ),

        # --- Integration rules ---
        Rule(
            name="power_integration",
            pattern={"integrand": "xⁿ"},
            result={"integral": "xⁿ⁺¹ / (n+1) + C",
                    "condition": "n ≠ −1"},
            weight=1.0,
            direction="forward",
        ),
        Rule(
            name="integration_by_parts",
            pattern={"integrand": "u × dv"},
            result={"integral": "u × v − ∫v × du",
                    "description": "inverse of product rule",
                    "isomorphism": "like etymological reconstruction through layered derivation"},
            weight=0.9,
            direction="forward",
        ),

        # --- Fundamental Theorem ---
        Rule(
            name="fundamental_theorem_calculus",
            pattern={"theorem": "∫ₐᵇ f'(x) dx"},
            result={"equals": "f(b) − f(a)",
                    "significance": "differentiation and integration are inverse operations",
                    "strange_loop": "forward (derive) and backward (integrate) cancel"},
            weight=1.0,
            direction="bidirectional",
            self_referential=True,
        ),

        # --- Limits ---
        Rule(
            name="limit_definition_derivative",
            pattern={"definition": "f'(x)"},
            result={"limit": "lim(h→0) [f(x+h) − f(x)] / h",
                    "description": "derivative as limit of difference quotient"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="lhopital_rule",
            pattern={"indeterminate": "0/0 or ∞/∞"},
            result={"resolution": "lim f/g = lim f'/g'",
                    "condition": "both limits give indeterminate form"},
            weight=0.9,
            direction="forward",
        ),

        # --- Mean value theorem ---
        Rule(name="mean_value_theorem",
             pattern={"function": "f continuous on [a,b], differentiable on (a,b)"},
             result={"theorem": "∃c ∈ (a,b): f'(c) = (f(b)-f(a))/(b-a)",
                     "meaning": "instantaneous rate equals average rate somewhere"},
             weight=1.0, direction="forward"),
        # --- Intermediate value theorem ---
        Rule(name="intermediate_value_theorem",
             pattern={"function": "f continuous on [a,b]"},
             result={"theorem": "for any y between f(a) and f(b), ∃c ∈ [a,b]: f(c) = y",
                     "consequence": "continuous functions take all intermediate values",
                     "application": "proves existence of roots"},
             weight=1.0, direction="forward"),
        # --- Taylor series ---
        Rule(name="taylor_series",
             pattern={"function": "f infinitely differentiable at a"},
             result={"expansion": "f(x) = Σ f⁽ⁿ⁾(a)/n! × (x-a)ⁿ",
                     "maclaurin": "Taylor at a=0",
                     "examples": {"eˣ": "Σ xⁿ/n!", "sin x": "Σ (-1)ⁿx²ⁿ⁺¹/(2n+1)!",
                                  "cos x": "Σ (-1)ⁿx²ⁿ/(2n)!"}},
             weight=1.0, direction="bidirectional"),
        # --- Stokes' theorem (generalised) ---
        Rule(name="stokes_theorem_general",
             pattern={"manifold": "oriented manifold M with boundary ∂M"},
             result={"theorem": "∫_M dω = ∫_∂M ω",
                     "special_cases": {
                         "1D": "fundamental theorem of calculus",
                         "2D_line": "Green's theorem",
                         "2D_surface": "classical Stokes' theorem",
                         "3D": "divergence theorem (Gauss)",
                     },
                     "significance": "unifies all of vector calculus in one statement"},
             weight=1.0, direction="bidirectional"),
        # --- Integration by parts ---
        Rule(name="integration_by_parts",
             pattern={"integral": "∫ u dv"},
             result={"formula": "∫ u dv = uv - ∫ v du",
                     "derivation": "from product rule d(uv) = u dv + v du"},
             weight=1.0, direction="bidirectional"),
        # --- Picard-Lindelöf (ODE existence) ---
        Rule(name="picard_lindelof",
             pattern={"ode": "y' = f(t,y), f Lipschitz in y"},
             result={"theorem": "unique solution exists in neighbourhood of initial condition",
                     "significance": "determinism of differential equations"},
             weight=1.0, direction="forward"),
    ]

    strange_loop = StrangeLoop(
        entry_rule="fundamental_theorem_calculus",
        cycle=[
            "differentiation_produces_rate_of_change",
            "integration_accumulates_rates",
            "accumulation_recovers_original_function",
        ],
        level_shift="abstraction_level_shift_between_function_and_derivative",
    )

    return Grammar(
        name="calculus",
        domain="mathematical",
        rules=rules,
        productions=productions,
        sub_grammars=[strange_loop],
    )


# ---------------------------------------------------------------------------
# Logic grammar
# ---------------------------------------------------------------------------

def build_logic_grammar() -> Grammar:
    """Build a grammar for formal logic — propositional, predicate, modal.

    Logic is the grammar of reasoning itself: premises derive conclusions,
    proofs are derivation trees, and the rules of inference are the
    productions of the grammar.

    The strange loop: Gödel's incompleteness theorem — a logical system
    that constructs a statement about its own provability.
    """

    productions = [
        # --- Propositional logic ---
        Production("Proposition", ["AtomicProp"], "logic"),
        Production("Proposition", ["NOT", "Proposition"], "logic"),
        Production("Proposition", ["Proposition", "AND", "Proposition"], "logic"),
        Production("Proposition", ["Proposition", "OR", "Proposition"], "logic"),
        Production("Proposition", ["Proposition", "IMPLIES", "Proposition"], "logic"),
        Production("Proposition", ["Proposition", "IFF", "Proposition"], "logic"),
        Production("Proposition", ["LPAREN", "Proposition", "RPAREN"], "logic"),

        # --- Predicate logic ---
        Production("Predicate", ["PredName", "LPAREN", "TermList", "RPAREN"], "logic"),
        Production("TermList", ["LogicTerm"], "logic"),
        Production("TermList", ["LogicTerm", "COMMA", "TermList"], "logic"),
        Production("LogicTerm", ["Constant"], "logic"),
        Production("LogicTerm", ["LogicVariable"], "logic"),
        Production("LogicTerm", ["Function", "LPAREN", "TermList", "RPAREN"], "logic"),
        Production("Quantified", ["FORALL", "LogicVariable", "Proposition"], "logic"),
        Production("Quantified", ["EXISTS", "LogicVariable", "Proposition"], "logic"),

        # --- Proof structure ---
        Production("Proof", ["Premise", "THEREFORE", "Conclusion"], "logic"),
        Production("Proof", ["Premise", "Premise", "THEREFORE", "Conclusion"], "logic"),
    ]

    rules = [
        # --- Inference rules ---
        Rule(
            name="modus_ponens",
            pattern={"premises": ["P", "P → Q"]},
            result={"conclusion": "Q",
                    "description": "if P and P implies Q, then Q",
                    "isomorphism": "like rule application in any grammar"},
            weight=1.0,
            direction="forward",
        ),
        Rule(
            name="modus_tollens",
            pattern={"premises": ["¬Q", "P → Q"]},
            result={"conclusion": "¬P",
                    "description": "if not Q and P implies Q, then not P"},
            weight=1.0,
            direction="forward",
        ),
        Rule(
            name="de_morgan_and",
            pattern={"expression": "¬(P ∧ Q)"},
            result={"equivalent": "¬P ∨ ¬Q",
                    "description": "De Morgan's law for conjunction"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="de_morgan_or",
            pattern={"expression": "¬(P ∨ Q)"},
            result={"equivalent": "¬P ∧ ¬Q",
                    "description": "De Morgan's law for disjunction"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="contrapositive",
            pattern={"implication": "P → Q"},
            result={"equivalent": "¬Q → ¬P",
                    "description": "contrapositive is logically equivalent"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="double_negation",
            pattern={"expression": "¬¬P"},
            result={"equivalent": "P",
                    "description": "double negation elimination (classical logic)"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Quantifier rules ---
        Rule(
            name="universal_instantiation",
            pattern={"quantified": "∀x. P(x)"},
            result={"instance": "P(a) for any a in the domain",
                    "description": "what holds for all holds for each"},
            weight=1.0,
            direction="forward",
        ),
        Rule(
            name="existential_generalisation",
            pattern={"instance": "P(a) for some specific a"},
            result={"quantified": "∃x. P(x)",
                    "description": "if it holds for one, it holds for some"},
            weight=0.9,
            direction="forward",
        ),

        # --- Gödel's incompleteness ---
        Rule(
            name="godel_incompleteness",
            pattern={"system": "sufficiently powerful formal system S"},
            result={"theorem": [
                "There exists a sentence G in S such that:",
                "G says 'G is not provable in S'",
                "If S is consistent, then G is true but unprovable",
                "S cannot prove its own consistency",
            ],
                    "strange_loop": "the sentence refers to its own provability",
                    "isomorphism": "like a quine in code, or the liar paradox in language"},
            conditions={"system_strength": "contains arithmetic"},
            weight=0.85,
            direction="bidirectional",
            self_referential=True,
        ),

        # --- Gödel's completeness theorem ---
        Rule(name="godel_completeness",
             pattern={"system": "first-order predicate logic"},
             result={"theorem": "every logically valid formula is provable",
                     "equivalently": "if Γ ⊨ φ then Γ ⊢ φ",
                     "contrast": "incompleteness is about arithmetic, completeness about pure logic",
                     "proved": "Gödel 1929 (PhD thesis)"},
             weight=1.0, direction="bidirectional"),
        # --- Compactness theorem ---
        Rule(name="compactness_theorem",
             pattern={"theory": "set of first-order sentences Γ"},
             result={"theorem": "Γ has a model iff every finite subset has a model",
                     "consequence": "can transfer properties from finite to infinite",
                     "proved": "follows from completeness (Gödel 1930)"},
             weight=1.0, direction="bidirectional"),
        # --- Church-Turing thesis ---
        Rule(name="church_turing_thesis",
             pattern={"concept": "effective computability"},
             result={"thesis": "every effectively computable function is Turing-computable",
                     "equivalences": ["Turing machine", "lambda calculus", "recursive functions",
                                      "Post system", "Markov algorithm"],
                     "status": "thesis, not theorem — cannot be formally proved",
                     "isomorphism": "like universal grammar hypothesis — all formalisms converge"},
             weight=1.0, direction="bidirectional"),
        # --- Halting problem ---
        Rule(name="halting_problem",
             pattern={"question": "does program P halt on input x?"},
             result={"theorem": "no algorithm can decide this for all P and x",
                     "proof": "diagonalisation (Turing 1936)",
                     "consequence": "limits of computation are absolute, not technological",
                     "isomorphism": "like Gödel — self-reference creates undecidability"},
             weight=1.0, direction="forward"),
        # --- Curry-Howard correspondence ---
        Rule(name="curry_howard",
             pattern={"observation": "structural similarity between proofs and programs"},
             result={"correspondence": {
                         "propositions": "↔ types",
                         "proofs": "↔ programs",
                         "proof_simplification": "↔ program evaluation",
                         "implication A→B": "↔ function type A → B",
                         "conjunction A∧B": "↔ product type (A, B)",
                         "disjunction A∨B": "↔ sum type A | B",
                     },
                     "significance": "logic and computation are the same structure",
                     "isomorphism": "bridges mathematics, computation, and linguistics"},
             weight=1.0, direction="bidirectional"),
        # --- Tarski's undefinability ---
        Rule(name="tarskis_undefinability",
             pattern={"language": "sufficiently expressive formal language L"},
             result={"theorem": "truth in L cannot be defined within L",
                     "consequence": "metalanguage needed to talk about truth",
                     "isomorphism": "like use/mention distinction in linguistics"},
             weight=1.0, direction="forward"),
    ]

    strange_loop = StrangeLoop(
        entry_rule="godel_incompleteness",
        cycle=[
            "system_encodes_statement_about_itself",
            "statement_asserts_own_unprovability",
            "truth_and_provability_diverge",
        ],
        level_shift="godelian_self_reference",
    )

    return Grammar(
        name="formal_logic",
        domain="mathematical",
        rules=rules,
        productions=productions,
        sub_grammars=[strange_loop],
    )


# ---------------------------------------------------------------------------
# Number theory grammar
# ---------------------------------------------------------------------------

def build_number_theory_grammar() -> Grammar:
    """Build a grammar for number theory — primes, divisibility, congruences.

    Number theory is the grammar of the integers: the rules governing
    divisibility, primality, and congruence are production rules that
    generate the infinite structure of ℤ from simple axioms.
    """

    rules = [
        # --- Divisibility ---
        Rule(
            name="division_algorithm",
            pattern={"input": "integers a, b with b > 0"},
            result={"theorem": "∃! q,r: a = bq + r, 0 ≤ r < b",
                    "description": "every integer can be divided with remainder"},
            weight=1.0,
            direction="forward",
        ),
        Rule(
            name="euclidean_algorithm",
            pattern={"task": "compute gcd(a, b)"},
            result={"algorithm": [
                "gcd(a, 0) = a",
                "gcd(a, b) = gcd(b, a mod b)",
            ],
                    "isomorphism": "like recursive etymological reduction to root form"},
            weight=1.0,
            direction="forward",
        ),

        # --- Primality ---
        Rule(
            name="fundamental_theorem_arithmetic",
            pattern={"input": "integer n > 1"},
            result={"theorem": "n = p₁^e₁ × p₂^e₂ × ... × pₖ^eₖ (unique factorisation)",
                    "significance": "primes are the atoms of multiplication",
                    "isomorphism": "like morphemes as atoms of word formation"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="prime_number_theorem",
            pattern={"question": "how many primes ≤ x?"},
            result={"asymptotic": "π(x) ~ x / ln(x)",
                    "description": "primes thin out logarithmically"},
            weight=0.85,
            direction="forward",
        ),

        # --- Modular arithmetic ---
        Rule(
            name="modular_congruence",
            pattern={"relation": "a ≡ b (mod n)"},
            result={"meaning": "n divides (a − b)",
                    "properties": [
                        "reflexive: a ≡ a",
                        "symmetric: a ≡ b ⟹ b ≡ a",
                        "transitive: a ≡ b, b ≡ c ⟹ a ≡ c",
                    ]},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="fermats_little_theorem",
            pattern={"conditions": "p prime, gcd(a, p) = 1"},
            result={"theorem": "aᵖ⁻¹ ≡ 1 (mod p)",
                    "corollary": "aᵖ ≡ a (mod p) for all a"},
            weight=0.9,
            direction="forward",
        ),
        Rule(
            name="chinese_remainder_theorem",
            pattern={"system": "x ≡ aᵢ (mod nᵢ), pairwise coprime nᵢ"},
            result={"theorem": "unique solution mod N = n₁ × n₂ × ... × nₖ",
                    "description": "decompose modular arithmetic into independent components"},
            weight=0.85,
            direction="forward",
        ),

        # --- Fibonacci / recurrences ---
        Rule(
            name="fibonacci_recurrence",
            pattern={"sequence": "F(n)"},
            result={"definition": "F(n) = F(n-1) + F(n-2), F(0)=0, F(1)=1",
                    "closed_form": "Binet's formula: F(n) = (φⁿ − ψⁿ) / √5",
                    "golden_ratio": "φ = (1 + √5) / 2",
                    "isomorphism": "like biological growth patterns (phyllotaxis)"},
            weight=0.9,
            direction="bidirectional",
        ),

        # --- Euclid's theorem ---
        Rule(name="euclids_theorem_primes",
             pattern={"question": "are there finitely many primes?"},
             result={"theorem": "there are infinitely many primes",
                     "proof": "assume finitely many p₁...pₖ; p₁×...×pₖ + 1 has prime factor not in list",
                     "proved": "Euclid, ~300 BCE; oldest known mathematical proof"},
             weight=1.0, direction="forward"),
        # --- Euler's theorem ---
        Rule(name="eulers_theorem",
             pattern={"conditions": "gcd(a, n) = 1"},
             result={"theorem": "a^φ(n) ≡ 1 (mod n)",
                     "euler_totient": "φ(n) = count of integers 1..n coprime to n",
                     "generalises": "Fermat's little theorem (when n is prime, φ(p) = p-1)"},
             weight=1.0, direction="forward"),
        # --- Quadratic reciprocity ---
        Rule(name="quadratic_reciprocity",
             pattern={"primes": "distinct odd primes p, q"},
             result={"theorem": "(p/q)(q/p) = (-1)^((p-1)/2 · (q-1)/2)",
                     "meaning": "solvability of x² ≡ p (mod q) related to x² ≡ q (mod p)",
                     "proved": "Gauss 1801 (published 8 proofs)",
                     "significance": "called 'the gem of arithmetic' by Gauss"},
             weight=1.0, direction="bidirectional"),
        # --- Bezout's identity ---
        Rule(name="bezouts_identity",
             pattern={"integers": "a, b with gcd(a,b) = d"},
             result={"theorem": "∃x,y ∈ ℤ: ax + by = d",
                     "found_by": "extended Euclidean algorithm",
                     "consequence": "gcd is the smallest positive linear combination"},
             weight=1.0, direction="bidirectional"),
        # --- Dirichlet's theorem ---
        Rule(name="dirichlets_theorem",
             pattern={"arithmetic_progression": "a, a+d, a+2d, ... with gcd(a,d)=1"},
             result={"theorem": "contains infinitely many primes",
                     "proved": "Dirichlet 1837 (using L-functions)",
                     "example": "3, 7, 11, 19, 23, ... (≡ 3 mod 4) has infinitely many primes"},
             weight=1.0, direction="forward"),
        # --- Perfect numbers ---
        Rule(name="euclid_euler_perfect",
             pattern={"number": "n is an even perfect number"},
             result={"theorem": "n = 2^(p-1)(2^p - 1) where 2^p - 1 is a Mersenne prime",
                     "direction": "Euclid (sufficient), Euler (necessary for even)",
                     "open_question": "do odd perfect numbers exist? (none found, none disproved)"},
             weight=1.0, direction="bidirectional"),
        # --- Euler's identity ---
        Rule(name="eulers_identity",
             pattern={"equation": "e^(iπ) + 1 = 0"},
             result={"significance": "connects five fundamental constants: e, i, π, 1, 0",
                     "generalised": "e^(ix) = cos(x) + i·sin(x) (Euler's formula)",
                     "isomorphism": "unifies analysis, algebra, geometry, and trigonometry"},
             weight=1.0, direction="bidirectional"),
    ]

    productions = [
        Production("NumberTheoryExpr", ["Integer"], "number_theory"),
        Production("NumberTheoryExpr", ["Prime"], "number_theory"),
        Production("NumberTheoryExpr", ["Composite"], "number_theory"),
        Production("Composite", ["Prime", "TIMES", "NumberTheoryExpr"], "number_theory"),
        Production("Congruence", ["Integer", "EQUIV", "Integer", "MOD", "Integer"], "number_theory"),
        Production("GCD", ["GCD_FUNC", "LPAREN", "Integer", "COMMA", "Integer", "RPAREN"], "number_theory"),
    ]

    return Grammar(
        name="number_theory",
        domain="mathematical",
        rules=rules,
        productions=productions,
    )
