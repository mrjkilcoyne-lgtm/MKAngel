"""Programming language grammars — syntax, types, and design patterns.

Code is the most explicit grammar humans write.  Programming language
syntax is a formal grammar (context-free, occasionally context-sensitive).
Type systems are grammars over values — they classify, constrain, and
compose.  Design patterns are higher-order grammatical productions: they
are recurring derivation templates that generate families of programs.

The strange loop: programs that produce their own source code (quines),
compilers that compile themselves (bootstrapping), and type systems
expressive enough to encode their own consistency proofs.
"""

from glm.core.grammar import Rule, Production, Grammar, StrangeLoop


# ---------------------------------------------------------------------------
# Syntax grammar
# ---------------------------------------------------------------------------

def build_syntax_grammar() -> Grammar:
    """Build a context-free grammar for a universal programming language subset.

    Covers the syntactic core shared by most imperative and functional
    languages: expressions, statements, functions, types, and modules.

    This is a real grammar in the Chomsky sense — each production is a
    derivation rule that can generate valid program fragments (forward)
    or parse them (backward).
    """

    productions = [
        # --- Program structure ---
        Production("Program", ["StatementList"], "syntax"),
        Production("StatementList", ["Statement"], "syntax"),
        Production("StatementList", ["Statement", "StatementList"], "syntax"),

        # --- Statements ---
        Production("Statement", ["ExprStatement"], "syntax"),
        Production("Statement", ["Declaration"], "syntax"),
        Production("Statement", ["Assignment"], "syntax"),
        Production("Statement", ["IfStatement"], "syntax"),
        Production("Statement", ["WhileStatement"], "syntax"),
        Production("Statement", ["ForStatement"], "syntax"),
        Production("Statement", ["ReturnStatement"], "syntax"),
        Production("Statement", ["FunctionDef"], "syntax"),
        Production("Statement", ["ClassDef"], "syntax"),
        Production("Statement", ["ImportStatement"], "syntax"),
        Production("Statement", ["Block"], "syntax"),

        # --- Expressions ---
        Production("ExprStatement", ["Expression", "SEMICOLON"], "syntax"),
        Production("Expression", ["Literal"], "syntax"),
        Production("Expression", ["Identifier"], "syntax"),
        Production("Expression", ["BinaryExpr"], "syntax"),
        Production("Expression", ["UnaryExpr"], "syntax"),
        Production("Expression", ["CallExpr"], "syntax"),
        Production("Expression", ["IndexExpr"], "syntax"),
        Production("Expression", ["MemberExpr"], "syntax"),
        Production("Expression", ["LambdaExpr"], "syntax"),
        Production("Expression", ["TernaryExpr"], "syntax"),
        Production("Expression", ["LPAREN", "Expression", "RPAREN"], "syntax"),

        # --- Binary / unary expressions ---
        Production("BinaryExpr", ["Expression", "BinOp", "Expression"], "syntax"),
        Production("UnaryExpr", ["UnOp", "Expression"], "syntax"),
        Production("BinOp", ["PLUS"], "syntax"),
        Production("BinOp", ["MINUS"], "syntax"),
        Production("BinOp", ["STAR"], "syntax"),
        Production("BinOp", ["SLASH"], "syntax"),
        Production("BinOp", ["MODULO"], "syntax"),
        Production("BinOp", ["EQ"], "syntax"),
        Production("BinOp", ["NEQ"], "syntax"),
        Production("BinOp", ["LT"], "syntax"),
        Production("BinOp", ["GT"], "syntax"),
        Production("BinOp", ["LTE"], "syntax"),
        Production("BinOp", ["GTE"], "syntax"),
        Production("BinOp", ["AND"], "syntax"),
        Production("BinOp", ["OR"], "syntax"),
        Production("UnOp", ["NOT"], "syntax"),
        Production("UnOp", ["MINUS"], "syntax"),

        # --- Literals ---
        Production("Literal", ["IntLiteral"], "syntax"),
        Production("Literal", ["FloatLiteral"], "syntax"),
        Production("Literal", ["StringLiteral"], "syntax"),
        Production("Literal", ["BoolLiteral"], "syntax"),
        Production("Literal", ["NoneLiteral"], "syntax"),
        Production("Literal", ["ListLiteral"], "syntax"),
        Production("Literal", ["DictLiteral"], "syntax"),

        # --- Control flow ---
        Production("IfStatement", ["IF", "Expression", "Block"], "syntax"),
        Production("IfStatement", ["IF", "Expression", "Block", "ELSE", "Block"], "syntax"),
        Production("IfStatement", ["IF", "Expression", "Block", "ELSE", "IfStatement"], "syntax"),
        Production("WhileStatement", ["WHILE", "Expression", "Block"], "syntax"),
        Production("ForStatement", ["FOR", "Identifier", "IN", "Expression", "Block"], "syntax"),

        # --- Functions ---
        Production("FunctionDef", ["FN", "Identifier", "ParamList", "Block"], "syntax"),
        Production("FunctionDef", ["FN", "Identifier", "ParamList", "ARROW", "TypeExpr", "Block"], "syntax"),
        Production("ParamList", ["LPAREN", "RPAREN"], "syntax"),
        Production("ParamList", ["LPAREN", "Params", "RPAREN"], "syntax"),
        Production("Params", ["Param"], "syntax"),
        Production("Params", ["Param", "COMMA", "Params"], "syntax"),
        Production("Param", ["Identifier"], "syntax"),
        Production("Param", ["Identifier", "COLON", "TypeExpr"], "syntax"),
        Production("LambdaExpr", ["LAMBDA", "ParamList", "ARROW", "Expression"], "syntax"),
        Production("CallExpr", ["Expression", "ArgList"], "syntax"),
        Production("ArgList", ["LPAREN", "RPAREN"], "syntax"),
        Production("ArgList", ["LPAREN", "Args", "RPAREN"], "syntax"),
        Production("Args", ["Expression"], "syntax"),
        Production("Args", ["Expression", "COMMA", "Args"], "syntax"),
        Production("ReturnStatement", ["RETURN", "Expression", "SEMICOLON"], "syntax"),

        # --- Declarations ---
        Production("Declaration", ["LET", "Identifier", "ASSIGN", "Expression", "SEMICOLON"], "syntax"),
        Production("Declaration", ["LET", "Identifier", "COLON", "TypeExpr", "ASSIGN", "Expression", "SEMICOLON"], "syntax"),
        Production("Assignment", ["Identifier", "ASSIGN", "Expression", "SEMICOLON"], "syntax"),

        # --- Classes ---
        Production("ClassDef", ["CLASS", "Identifier", "ClassBody"], "syntax"),
        Production("ClassDef", ["CLASS", "Identifier", "EXTENDS", "Identifier", "ClassBody"], "syntax"),
        Production("ClassBody", ["LBRACE", "ClassMembers", "RBRACE"], "syntax"),
        Production("ClassMembers", ["ClassMember"], "syntax"),
        Production("ClassMembers", ["ClassMember", "ClassMembers"], "syntax"),
        Production("ClassMember", ["FunctionDef"], "syntax"),
        Production("ClassMember", ["Declaration"], "syntax"),

        # --- Blocks ---
        Production("Block", ["LBRACE", "StatementList", "RBRACE"], "syntax"),
        Production("Block", ["LBRACE", "RBRACE"], "syntax"),

        # --- Access ---
        Production("IndexExpr", ["Expression", "LBRACKET", "Expression", "RBRACKET"], "syntax"),
        Production("MemberExpr", ["Expression", "DOT", "Identifier"], "syntax"),
        Production("TernaryExpr", ["Expression", "QUESTION", "Expression", "COLON", "Expression"], "syntax"),
    ]

    rules = [
        # --- Operator precedence (disambiguation) ---
        Rule(
            name="operator_precedence",
            pattern={"ambiguous": "E op1 E op2 E"},
            result={"resolution": "higher-precedence operator binds tighter",
                    "precedence_levels": [
                        "1 (lowest): OR",
                        "2: AND",
                        "3: EQ, NEQ",
                        "4: LT, GT, LTE, GTE",
                        "5: PLUS, MINUS",
                        "6: STAR, SLASH, MODULO",
                        "7 (highest): UnaryOp, CallExpr, IndexExpr, MemberExpr",
                    ]},
            conditions={"associativity": "left-to-right (except assignment: right)"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Scope rules ---
        Rule(
            name="lexical_scoping",
            pattern={"reference": "identifier used in expression"},
            result={"resolution": "search inner scope → outer scope → global → error",
                    "principle": "static/lexical scoping (closure captures enclosing scope)",
                    "shadowing": "inner declaration hides outer same-name binding"},
            conditions={"scope_model": "lexical"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Expression vs statement ---
        Rule(
            name="expression_statement_duality",
            pattern={"construct": "if/match/block"},
            result={"principle": "in expression-oriented languages, everything returns a value",
                    "examples": [
                        "let x = if cond { a } else { b };",
                        "let y = match v { ... };",
                        "let z = { stmt; stmt; expr };",
                    ]},
            conditions={"language_style": "expression_oriented"},
            weight=0.8,
            direction="bidirectional",
        ),

        # --- Pattern matching ---
        Rule(
            name="pattern_matching_exhaustiveness",
            pattern={"match_expression": "match value { patterns }"},
            result={"constraint": "patterns must be exhaustive (cover all cases)",
                    "pattern_types": [
                        "literal pattern: 42, 'hello'",
                        "variable pattern: x (binds to value)",
                        "constructor pattern: Some(x), Cons(h, t)",
                        "wildcard: _ (matches anything)",
                        "guard: pattern if condition",
                    ]},
            conditions={"compiler_checks": "exhaustiveness and reachability"},
            weight=0.9,
            direction="bidirectional",
        ),
    ]

    return Grammar(
        name="programming_language_syntax",
        domain="computation",
        rules=rules,
        productions=productions,
    )


# ---------------------------------------------------------------------------
# Type grammar
# ---------------------------------------------------------------------------

def build_type_grammar() -> Grammar:
    """Build a grammar for type systems.

    Types are a grammar over values: they classify values into sets,
    define allowed operations, and ensure compositional correctness.
    Type inference discovers the implicit grammar of a program.

    Includes:
    - Primitive and composite types
    - Subtyping rules (Liskov substitution as a grammatical constraint)
    - Parametric polymorphism (generics)
    - Type inference rules (Hindley-Milner)
    - Algebraic data types (sum and product types)
    """

    productions = [
        # --- Type expressions ---
        Production("TypeExpr", ["PrimitiveType"], "type"),
        Production("TypeExpr", ["NamedType"], "type"),
        Production("TypeExpr", ["FunctionType"], "type"),
        Production("TypeExpr", ["GenericType"], "type"),
        Production("TypeExpr", ["ArrayType"], "type"),
        Production("TypeExpr", ["TupleType"], "type"),
        Production("TypeExpr", ["UnionType"], "type"),
        Production("TypeExpr", ["IntersectionType"], "type"),
        Production("TypeExpr", ["OptionalType"], "type"),

        # --- Primitives ---
        Production("PrimitiveType", ["INT"], "type"),
        Production("PrimitiveType", ["FLOAT"], "type"),
        Production("PrimitiveType", ["BOOL"], "type"),
        Production("PrimitiveType", ["STRING"], "type"),
        Production("PrimitiveType", ["UNIT"], "type"),
        Production("PrimitiveType", ["NEVER"], "type"),

        # --- Composite types ---
        Production("FunctionType", ["TypeExpr", "ARROW", "TypeExpr"], "type"),
        Production("GenericType", ["NamedType", "LANGLE", "TypeArgs", "RANGLE"], "type"),
        Production("TypeArgs", ["TypeExpr"], "type"),
        Production("TypeArgs", ["TypeExpr", "COMMA", "TypeArgs"], "type"),
        Production("ArrayType", ["LBRACKET", "TypeExpr", "RBRACKET"], "type"),
        Production("TupleType", ["LPAREN", "TypeList", "RPAREN"], "type"),
        Production("TypeList", ["TypeExpr", "COMMA", "TypeExpr"], "type"),
        Production("TypeList", ["TypeExpr", "COMMA", "TypeList"], "type"),
        Production("UnionType", ["TypeExpr", "PIPE", "TypeExpr"], "type"),
        Production("IntersectionType", ["TypeExpr", "AMPERSAND", "TypeExpr"], "type"),
        Production("OptionalType", ["TypeExpr", "QUESTION"], "type"),

        # --- Algebraic data types ---
        Production("ADT", ["SumType"], "type"),
        Production("ADT", ["ProductType"], "type"),
        Production("SumType", ["Variant", "PIPE", "SumType"], "type"),
        Production("SumType", ["Variant"], "type"),
        Production("Variant", ["ConstructorName"], "type"),
        Production("Variant", ["ConstructorName", "LPAREN", "TypeList", "RPAREN"], "type"),
        Production("ProductType", ["LBRACE", "FieldList", "RBRACE"], "type"),
        Production("FieldList", ["Field"], "type"),
        Production("FieldList", ["Field", "COMMA", "FieldList"], "type"),
        Production("Field", ["Identifier", "COLON", "TypeExpr"], "type"),
    ]

    rules = [
        # --- Type inference (Hindley-Milner core) ---
        Rule(
            name="var_rule",
            pattern={"judgement": "Γ ⊢ x : τ"},
            result={"rule": "if x:σ ∈ Γ, then τ = instantiate(σ)",
                    "description": "variable has the type declared in the environment"},
            conditions={"x_in_context": True},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="app_rule",
            pattern={"judgement": "Γ ⊢ f(e) : τ₂"},
            result={"rule": "if Γ ⊢ f : τ₁ → τ₂ and Γ ⊢ e : τ₁",
                    "description": "application: function return type is result type",
                    "unification": "τ₁ from function must unify with type of argument"},
            conditions={"function_type": "τ₁ → τ₂"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="abs_rule",
            pattern={"judgement": "Γ ⊢ λx.e : τ₁ → τ₂"},
            result={"rule": "if Γ,x:τ₁ ⊢ e : τ₂",
                    "description": "lambda abstraction: parameter type → body type"},
            conditions={"introduces_binding": True},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="let_rule",
            pattern={"judgement": "Γ ⊢ let x = e₁ in e₂ : τ₂"},
            result={"rule": "if Γ ⊢ e₁ : τ₁ and Γ,x:Gen(τ₁) ⊢ e₂ : τ₂",
                    "description": "let-binding generalises the type of e₁ (let-polymorphism)",
                    "key_insight": "this is where polymorphism enters HM"},
            conditions={"generalisation": "quantify free type variables not in Γ"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="unification",
            pattern={"task": "make two type expressions equal"},
            result={"algorithm": [
                "1. If both are type variables, bind one to the other",
                "2. If one is a variable, bind it (occurs check first)",
                "3. If both are constructors, unify component-wise",
                "4. Otherwise, fail (type error)",
            ],
                    "occurs_check": "prevents infinite types (τ = τ → τ)"},
            conditions={"algorithm": "Robinson_unification"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Subtyping ---
        Rule(
            name="subtype_reflexivity",
            pattern={"rule": "τ <: τ"},
            result={"description": "every type is a subtype of itself"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="subtype_transitivity",
            pattern={"rule": "if τ₁ <: τ₂ and τ₂ <: τ₃ then τ₁ <: τ₃"},
            result={"description": "subtyping is transitive"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="function_subtyping",
            pattern={"rule": "τ₁' → τ₂' <: τ₁ → τ₂"},
            result={"conditions": "if τ₁ <: τ₁' (contravariant) and τ₂' <: τ₂ (covariant)",
                    "description": ("functions are contravariant in parameter types, "
                                    "covariant in return types"),
                    "intuition": "a function that accepts more and returns less is a subtype"},
            conditions={"variance": {"input": "contravariant", "output": "covariant"}},
            weight=0.95,
            direction="bidirectional",
        ),
        Rule(
            name="liskov_substitution",
            pattern={"principle": "if S <: T, then S can replace T anywhere"},
            result={"constraints": [
                "preconditions of S methods ≤ preconditions of T methods",
                "postconditions of S methods ≥ postconditions of T methods",
                "invariants of T preserved by S",
                "S may not throw new exception types",
            ],
                    "description": "behavioural subtyping — the L in SOLID"},
            conditions={"design_principle": True},
            weight=0.9,
            direction="bidirectional",
        ),

        # --- Parametric polymorphism ---
        Rule(
            name="parametric_polymorphism",
            pattern={"generic": "∀α. τ(α)"},
            result={"meaning": "type works uniformly for all instantiations of α",
                    "examples": [
                        "id : ∀α. α → α",
                        "map : ∀α β. (α → β) → [α] → [β]",
                        "fold : ∀α β. (β → α → β) → β → [α] → β",
                    ],
                    "parametricity": "free theorems from type alone"},
            conditions={"kind": "universal_quantification"},
            weight=0.9,
            direction="bidirectional",
        ),

        # --- Algebraic data type semantics ---
        Rule(
            name="sum_type_semantics",
            pattern={"type": "A | B | C"},
            result={"cardinality": "|A| + |B| + |C|",
                    "usage": "pattern matching / tagged union / variant",
                    "examples": [
                        "Option<T> = Some(T) | None",
                        "Result<T,E> = Ok(T) | Err(E)",
                        "List<T> = Cons(T, List<T>) | Nil",
                    ]},
            conditions={"algebraic_interpretation": "coproduct / disjoint union"},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="product_type_semantics",
            pattern={"type": "A × B × C"},
            result={"cardinality": "|A| × |B| × |C|",
                    "usage": "struct / record / tuple",
                    "examples": [
                        "Point = { x: Float, y: Float }",
                        "(String, Int) — a tuple",
                    ]},
            conditions={"algebraic_interpretation": "cartesian product"},
            weight=0.9,
            direction="bidirectional",
        ),
    ]

    return Grammar(
        name="type_system",
        domain="computation",
        rules=rules,
        productions=productions,
    )


# ---------------------------------------------------------------------------
# Design pattern grammar
# ---------------------------------------------------------------------------

def build_pattern_grammar() -> Grammar:
    """Build a grammar for design patterns as derivation rules.

    Design patterns are higher-order grammatical productions: they are
    templates that, when applied, generate families of structurally
    similar programs.  Each pattern is a production rule at the
    architectural level.

    Includes:
    - Creational patterns (Factory, Builder, Singleton)
    - Structural patterns (Adapter, Decorator, Composite)
    - Behavioural patterns (Observer, Strategy, Visitor)
    - The strange loop: quines and self-hosting compilers
    """

    rules = [
        # ===================================================================
        # CREATIONAL PATTERNS
        # ===================================================================

        Rule(
            name="factory_method",
            pattern={"problem": "object creation depends on runtime conditions",
                     "structure": "Creator → ConcreteCreator, Product → ConcreteProduct"},
            result={"solution": [
                "1. Define Product interface",
                "2. Creator declares factory_method() → Product",
                "3. ConcreteCreator overrides factory_method() → ConcreteProduct",
                "4. Client code uses Creator.factory_method(), not constructors",
            ],
                    "benefit": "decouples creation from usage",
                    "isomorphism": "like a grammar rule: LHS (interface) derives RHS (implementation)"},
            conditions={"when": "subclasses should decide which class to instantiate"},
            weight=0.9,
            direction="forward",
        ),
        Rule(
            name="builder_pattern",
            pattern={"problem": "complex object with many optional components",
                     "structure": "Builder → ConcreteBuilder, Director orchestrates"},
            result={"solution": [
                "1. Builder interface with step methods (build_part_A, build_part_B...)",
                "2. ConcreteBuilder implements steps, accumulates result",
                "3. Director calls steps in specific order",
                "4. Client retrieves product from builder",
            ],
                    "benefit": "separates construction algorithm from representation",
                    "isomorphism": "like morphological derivation: affix by affix builds a word"},
            conditions={"when": "construction requires multiple steps or configurations"},
            weight=0.85,
            direction="forward",
        ),
        Rule(
            name="singleton_pattern",
            pattern={"problem": "exactly one instance of a class needed globally"},
            result={"solution": [
                "1. Private constructor",
                "2. Static instance field",
                "3. Public static get_instance() method",
                "4. Lazy initialisation (create on first access)",
            ],
                    "caution": "global mutable state — often an anti-pattern",
                    "alternatives": ["dependency injection", "module-level instance"]},
            conditions={"when": "shared resource (logger, config, connection pool)"},
            weight=0.7,
            direction="forward",
        ),

        # ===================================================================
        # STRUCTURAL PATTERNS
        # ===================================================================

        Rule(
            name="adapter_pattern",
            pattern={"problem": "incompatible interfaces need to work together",
                     "structure": "Client → Target, Adapter wraps Adaptee"},
            result={"solution": [
                "1. Target defines the interface the client expects",
                "2. Adaptee has useful behaviour but wrong interface",
                "3. Adapter implements Target, delegates to Adaptee",
                "4. Client uses Adapter through Target interface",
            ],
                    "types": ["class adapter (inheritance)", "object adapter (composition)"],
                    "isomorphism": "like phonological adaptation of loanwords"},
            conditions={"when": "integrating legacy or third-party code"},
            weight=0.85,
            direction="forward",
        ),
        Rule(
            name="decorator_pattern",
            pattern={"problem": "add responsibilities to object dynamically",
                     "structure": "Component → ConcreteComponent, Decorator wraps Component"},
            result={"solution": [
                "1. Component interface (operation())",
                "2. ConcreteComponent implements base operation",
                "3. Decorator implements Component, wraps another Component",
                "4. ConcreteDecorator adds behaviour before/after delegating",
                "5. Decorators can be stacked arbitrarily",
            ],
                    "benefit": "open/closed principle — extend without modifying",
                    "isomorphism": "like derivational morphology: each affix wraps the stem"},
            conditions={"when": "subclassing would create exponential class hierarchy"},
            weight=0.9,
            direction="forward",
        ),
        Rule(
            name="composite_pattern",
            pattern={"problem": "tree structure where leaves and branches treated uniformly",
                     "structure": "Component → Leaf | Composite(Component*)"},
            result={"solution": [
                "1. Component interface with operation()",
                "2. Leaf implements operation() directly",
                "3. Composite holds children (list of Component)",
                "4. Composite.operation() delegates to each child",
            ],
                    "benefit": "clients don't distinguish leaves from composites",
                    "isomorphism": "like phrase structure: NP can be a single N or N + PP + CP"},
            conditions={"when": "part-whole hierarchies"},
            weight=0.9,
            direction="forward",
        ),

        # ===================================================================
        # BEHAVIOURAL PATTERNS
        # ===================================================================

        Rule(
            name="observer_pattern",
            pattern={"problem": "one-to-many dependency: when subject changes, notify dependants",
                     "structure": "Subject holds list of Observers, notifies on change"},
            result={"solution": [
                "1. Subject interface: attach(Observer), detach(Observer), notify()",
                "2. Observer interface: update(Subject)",
                "3. ConcreteSubject stores state, calls notify() on change",
                "4. ConcreteObserver implements update() to sync with subject",
            ],
                    "variants": ["push model (subject sends data)", "pull model (observer queries)"],
                    "isomorphism": "like regulatory networks in biology: gene expression triggers cascades"},
            conditions={"when": "decoupled event-driven communication"},
            weight=0.9,
            direction="forward",
        ),
        Rule(
            name="strategy_pattern",
            pattern={"problem": "algorithm should be interchangeable at runtime",
                     "structure": "Context uses Strategy interface, ConcreteStrategies implement it"},
            result={"solution": [
                "1. Strategy interface with execute() method",
                "2. ConcreteStrategy A, B, C implement different algorithms",
                "3. Context holds a Strategy reference",
                "4. Client sets the strategy; context delegates to it",
            ],
                    "benefit": "eliminates conditional branching over algorithm selection",
                    "isomorphism": "like allomorphs: same morpheme, different surface realisations"},
            conditions={"when": "family of interchangeable algorithms"},
            weight=0.9,
            direction="forward",
        ),
        Rule(
            name="visitor_pattern",
            pattern={"problem": "add operations to object structure without modifying classes",
                     "structure": "Visitor with visit(Element) methods, Elements accept(Visitor)"},
            result={"solution": [
                "1. Visitor interface with visit_X() for each Element type",
                "2. Element interface with accept(Visitor) method",
                "3. ConcreteElement.accept(v) calls v.visit_ConcreteElement(self)",
                "4. ConcreteVisitor implements visit_X() for each element",
                "5. Double dispatch: runtime types of both visitor and element matter",
            ],
                    "benefit": "new operations without touching element classes",
                    "isomorphism": ("like tree traversal in syntax: the visitor walks the "
                                    "parse tree applying transformations at each node")},
            conditions={"when": "stable element hierarchy, frequently changing operations"},
            weight=0.85,
            direction="forward",
        ),

        # ===================================================================
        # FUNCTIONAL PATTERNS
        # ===================================================================

        Rule(
            name="monad_pattern",
            pattern={"problem": "sequencing computations with effects (Maybe, IO, List, etc.)",
                     "structure": "M<A> with unit/return and bind/flatMap"},
            result={"laws": {
                "left_identity": "return(a).bind(f) == f(a)",
                "right_identity": "m.bind(return) == m",
                "associativity": "m.bind(f).bind(g) == m.bind(x => f(x).bind(g))",
            },
                    "common_monads": {
                        "Maybe/Option": "computation that might fail",
                        "Either/Result": "computation with typed errors",
                        "List": "nondeterministic computation",
                        "IO": "computation with side effects",
                        "State": "computation with mutable state",
                    },
                    "isomorphism": "like sequential derivation steps in a grammar"},
            conditions={"paradigm": "functional"},
            weight=0.85,
            direction="bidirectional",
        ),

        # ===================================================================
        # STRANGE LOOPS
        # ===================================================================

        Rule(
            name="quine",
            pattern={"problem": "program that outputs its own source code"},
            result={"structure": [
                "1. Program has two parts: DATA and CODE",
                "2. DATA encodes the source as a string/data structure",
                "3. CODE uses DATA to reconstruct the full source",
                "4. The output equals the source — a fixed point",
            ],
                    "fixed_point": "quine(source) == source",
                    "isomorphism": "Godel sentence: statement that asserts its own provability"},
            conditions={"constraint": "no reading own source file"},
            weight=0.8,
            direction="bidirectional",
            self_referential=True,
        ),
        Rule(
            name="self_hosting_compiler",
            pattern={"problem": "compiler written in the language it compiles"},
            result={"bootstrap_chain": [
                "1. Write compiler v0 in existing language X",
                "2. v0 compiles source of compiler v1 (written in target language)",
                "3. v1 compiles itself → v2 (should be identical to v1)",
                "4. v2 == v1 confirms correctness (fixed point)",
            ],
                    "strange_loop": "the compiler is both subject and object of compilation",
                    "examples": ["GCC (C)", "rustc (Rust)", "Go compiler (Go)",
                                 "PyPy (Python)"]},
            conditions={"requires": "initial bootstrap from another language"},
            weight=0.8,
            direction="bidirectional",
            self_referential=True,
        ),
        Rule(
            name="metacircular_evaluator",
            pattern={"problem": "interpreter for language L written in language L"},
            result={"structure": [
                "1. eval(expr, env) dispatches on expression type",
                "2. apply(proc, args) handles function application",
                "3. env maps names to values (lexical scoping)",
                "4. The evaluator defines L's semantics in terms of L itself",
            ],
                    "strange_loop": "meaning is defined in terms of itself",
                    "reference": "SICP chapter 4 — the metacircular evaluator"},
            conditions={"language": "homoiconic or reflective"},
            weight=0.85,
            direction="bidirectional",
            self_referential=True,
        ),
    ]

    productions = [
        # Pattern instantiation as derivation
        Production("Architecture", ["Pattern", "Architecture"], "pattern"),
        Production("Architecture", ["Pattern"], "pattern"),
        Production("Pattern", ["CreationalPattern"], "pattern"),
        Production("Pattern", ["StructuralPattern"], "pattern"),
        Production("Pattern", ["BehaviouralPattern"], "pattern"),
        Production("Pattern", ["FunctionalPattern"], "pattern"),
        Production("CreationalPattern", ["Factory"], "pattern"),
        Production("CreationalPattern", ["Builder"], "pattern"),
        Production("CreationalPattern", ["Singleton"], "pattern"),
        Production("StructuralPattern", ["Adapter"], "pattern"),
        Production("StructuralPattern", ["Decorator"], "pattern"),
        Production("StructuralPattern", ["Composite"], "pattern"),
        Production("BehaviouralPattern", ["Observer"], "pattern"),
        Production("BehaviouralPattern", ["Strategy"], "pattern"),
        Production("BehaviouralPattern", ["Visitor"], "pattern"),
        Production("FunctionalPattern", ["Monad"], "pattern"),
        Production("FunctionalPattern", ["Functor"], "pattern"),
    ]

    # Strange loop: programs that write programs that write programs
    strange_loop = StrangeLoop(
        entry_rule="self_reference",
        cycle=[
            "program_outputs_source_code",
            "source_code_is_the_program",
            "the_program_is_its_own_output",
        ],
        level_shift="godelian_fixed_point",
    )

    return Grammar(
        name="design_patterns",
        domain="computation",
        rules=rules,
        productions=productions,
        sub_grammars=[strange_loop],
    )
