"""
Symbolic Substrate — the grammar of formal and programming languages.

Code is grammar made visible.  Where natural language hides its grammar
beneath surface irregularity, programming languages make every rule
explicit: syntax is defined by formal grammars, scope is managed by
stack discipline, types enforce agreement, and every well-formed program
is a derivation tree.

This module treats source code as a substrate — tokens are the atoms,
syntax trees are the molecular structure, and type systems are the
feature systems.

Strange loops: a compiler written in its own language (self-hosting),
a quine (a program that prints its own source), a type system that types
itself (System U), eval/apply in Lisp.  Code is the most explicitly
self-referential substrate.

Fugues: concurrent processes, coroutines, and pipelines — multiple
streams of tokens executing in coordinated parallel.

Isomorphisms:
- Token ↔ phoneme (both are categorised by features: keyword/stop,
  operator/fricative, identifier/vowel).
- Syntax tree ↔ molecular structure ↔ syllable tree.
- Scope rules ↔ vowel harmony (both are long-distance dependencies).
- Type checking ↔ valence checking ↔ morphological agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..core.substrate import Sequence, Substrate, Symbol, TransformationRule


# ---------------------------------------------------------------------------
# Token categories
# ---------------------------------------------------------------------------

class TokenCategory(Enum):
    """Broad syntactic categories for tokens — the "parts of speech" of code."""
    KEYWORD = auto()
    IDENTIFIER = auto()
    OPERATOR = auto()
    LITERAL = auto()
    SEPARATOR = auto()
    COMMENT = auto()
    WHITESPACE = auto()
    STRING = auto()
    EOF = auto()


# ---------------------------------------------------------------------------
# Token — Symbol subclass for code tokens
# ---------------------------------------------------------------------------

@dataclass
class Token(Symbol):
    """A token — the atomic unit of source code.

    Like phonemes, tokens are categorised by features:
    - category: KEYWORD, IDENTIFIER, OPERATOR, LITERAL, …
    - subcategory: finer distinctions (e.g. 'comparison' for operators)
    - data_type: for literals, the inferred type ('int', 'float', 'str', …)
    - scope_effect: how this token affects scope ('open', 'close', None)

    The ``valence`` of a token describes its syntactic appetite:
    - A binary operator has valence 2 (needs left and right operands).
    - A unary operator has valence 1.
    - A keyword like 'if' has valence 2 (condition + body) or 3 (+else).
    - An identifier has valence 0 (it is a leaf).
    """

    category: TokenCategory = TokenCategory.IDENTIFIER
    subcategory: str = ""
    data_type: str = ""
    scope_effect: Optional[str] = None  # "open", "close", None

    def __post_init__(self) -> None:
        if self.domain == "generic":
            self.domain = "symbolic"
        # Populate features dict from token fields
        self.features.setdefault("category", self.category.name.lower())
        if self.subcategory:
            self.features["subcategory"] = self.subcategory
        if self.data_type:
            self.features["data_type"] = self.data_type
        if self.scope_effect:
            self.features["scope_effect"] = self.scope_effect

    def __hash__(self) -> int:
        return hash((self.form, self.domain, self.category.name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Token):
            return Symbol.__eq__(self, other)
        return (self.form == other.form and self.domain == other.domain
                and self.category == other.category)

    @property
    def is_keyword(self) -> bool:
        return self.category == TokenCategory.KEYWORD

    @property
    def is_identifier(self) -> bool:
        return self.category == TokenCategory.IDENTIFIER

    @property
    def is_operator(self) -> bool:
        return self.category == TokenCategory.OPERATOR

    @property
    def is_literal(self) -> bool:
        return self.category == TokenCategory.LITERAL

    @property
    def opens_scope(self) -> bool:
        return self.scope_effect == "open"

    @property
    def closes_scope(self) -> bool:
        return self.scope_effect == "close"


# ---------------------------------------------------------------------------
# AST Node (lightweight)
# ---------------------------------------------------------------------------

@dataclass
class ASTNode:
    """A lightweight abstract syntax tree node.

    The AST is to code what molecular structure is to chemistry —
    it reveals the deep constituency hidden in the linear token stream.
    """
    node_type: str
    token: Optional[Token] = None
    children: List["ASTNode"] = field(default_factory=list)
    parent: Optional["ASTNode"] = None

    def add_child(self, child: "ASTNode") -> None:
        child.parent = self
        self.children.append(child)

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def leaf_count(self) -> int:
        if not self.children:
            return 1
        return sum(c.leaf_count() for c in self.children)

    def __repr__(self) -> str:
        label = self.token.form if self.token else self.node_type
        if self.children:
            return f"({label} {' '.join(repr(c) for c in self.children)})"
        return label


# ---------------------------------------------------------------------------
# Built-in keyword / operator data
# ---------------------------------------------------------------------------

# Python-like keywords
KEYWORDS: Dict[str, Dict[str, Any]] = {
    "if":       {"valence": 2, "subcategory": "conditional"},
    "elif":     {"valence": 2, "subcategory": "conditional"},
    "else":     {"valence": 1, "subcategory": "conditional"},
    "while":    {"valence": 2, "subcategory": "loop"},
    "for":      {"valence": 3, "subcategory": "loop"},
    "def":      {"valence": 3, "subcategory": "definition", "scope_effect": "open"},
    "class":    {"valence": 2, "subcategory": "definition", "scope_effect": "open"},
    "return":   {"valence": 1, "subcategory": "control"},
    "import":   {"valence": 1, "subcategory": "import"},
    "from":     {"valence": 2, "subcategory": "import"},
    "try":      {"valence": 1, "subcategory": "exception", "scope_effect": "open"},
    "except":   {"valence": 1, "subcategory": "exception"},
    "raise":    {"valence": 1, "subcategory": "exception"},
    "with":     {"valence": 2, "subcategory": "context", "scope_effect": "open"},
    "as":       {"valence": 1, "subcategory": "alias"},
    "pass":     {"valence": 0, "subcategory": "no-op"},
    "break":    {"valence": 0, "subcategory": "control"},
    "continue": {"valence": 0, "subcategory": "control"},
    "and":      {"valence": 2, "subcategory": "logical"},
    "or":       {"valence": 2, "subcategory": "logical"},
    "not":      {"valence": 1, "subcategory": "logical"},
    "in":       {"valence": 2, "subcategory": "membership"},
    "is":       {"valence": 2, "subcategory": "identity"},
    "None":     {"valence": 0, "subcategory": "literal"},
    "True":     {"valence": 0, "subcategory": "literal"},
    "False":    {"valence": 0, "subcategory": "literal"},
    "lambda":   {"valence": 2, "subcategory": "definition"},
    "yield":    {"valence": 1, "subcategory": "control"},
}

OPERATORS: Dict[str, Dict[str, Any]] = {
    "+":  {"valence": 2, "subcategory": "arithmetic"},
    "-":  {"valence": 2, "subcategory": "arithmetic"},
    "*":  {"valence": 2, "subcategory": "arithmetic"},
    "/":  {"valence": 2, "subcategory": "arithmetic"},
    "//": {"valence": 2, "subcategory": "arithmetic"},
    "%":  {"valence": 2, "subcategory": "arithmetic"},
    "**": {"valence": 2, "subcategory": "arithmetic"},
    "=":  {"valence": 2, "subcategory": "assignment"},
    "+=": {"valence": 2, "subcategory": "assignment"},
    "-=": {"valence": 2, "subcategory": "assignment"},
    "*=": {"valence": 2, "subcategory": "assignment"},
    "/=": {"valence": 2, "subcategory": "assignment"},
    "==": {"valence": 2, "subcategory": "comparison"},
    "!=": {"valence": 2, "subcategory": "comparison"},
    "<":  {"valence": 2, "subcategory": "comparison"},
    ">":  {"valence": 2, "subcategory": "comparison"},
    "<=": {"valence": 2, "subcategory": "comparison"},
    ">=": {"valence": 2, "subcategory": "comparison"},
    ".":  {"valence": 2, "subcategory": "access"},
}

SCOPE_OPENERS = {"(": ")", "[": "]", "{": "}"}
SCOPE_CLOSERS = {")", "]", "}"}


# ---------------------------------------------------------------------------
# SymbolicSubstrate
# ---------------------------------------------------------------------------

class SymbolicSubstrate(Substrate):
    """Substrate for formal / programming languages.

    Treats source code as a symbolic system whose grammar is made
    completely explicit.  Provides tokenisation, scope tracking,
    bracket matching, AST construction, and type-level analysis.
    """

    def __init__(self, name: str = "symbolic") -> None:
        super().__init__(name, domain="symbolic")
        self._build_inventory()

    def _build_inventory(self) -> None:
        """Register keywords and operators as symbols."""
        for kw, data in KEYWORDS.items():
            tok = Token(
                form=kw,
                features={"category": "keyword", "subcategory": data.get("subcategory", "")},
                domain="symbolic",
                valence=data.get("valence", 0),
                category=TokenCategory.KEYWORD,
                subcategory=data.get("subcategory", ""),
                scope_effect=data.get("scope_effect"),
            )
            self.add_symbol(tok)

        for op, data in OPERATORS.items():
            tok = Token(
                form=op,
                features={"category": "operator", "subcategory": data.get("subcategory", "")},
                domain="symbolic",
                valence=data.get("valence", 2),
                category=TokenCategory.OPERATOR,
                subcategory=data.get("subcategory", ""),
            )
            self.add_symbol(tok)

        for opener, closer in SCOPE_OPENERS.items():
            self.add_symbol(Token(
                form=opener, domain="symbolic", valence=0,
                category=TokenCategory.SEPARATOR,
                scope_effect="open",
            ))
            self.add_symbol(Token(
                form=closer, domain="symbolic", valence=0,
                category=TokenCategory.SEPARATOR,
                scope_effect="close",
            ))

        for sep in (",", ";", ":", "->"):
            self.add_symbol(Token(
                form=sep, domain="symbolic", valence=0,
                category=TokenCategory.SEPARATOR,
            ))

    # -- encode (tokenise) --------------------------------------------------

    def encode(self, raw_input: str) -> Sequence:
        """Tokenise source code into a Sequence of Tokens.

        This is a simplified lexer that handles:
        - Keywords (Python-like)
        - Multi-character operators (==, !=, <=, >=, //, **, +=, etc.)
        - Numeric literals (int and float)
        - String literals (single and double quoted)
        - Identifiers
        - Separators / brackets
        - Comments (# to end of line)

        Whitespace is skipped (not tokenised).
        """
        tokens: List[Symbol] = []
        i = 0
        src = raw_input

        while i < len(src):
            # Skip whitespace
            if src[i] in (" ", "\t", "\n", "\r"):
                i += 1
                continue

            # Comments: # to end of line
            if src[i] == "#":
                end = src.find("\n", i)
                if end < 0:
                    end = len(src)
                comment_text = src[i + 1 : end].strip()
                tokens.append(Token(
                    form=comment_text, domain="symbolic", valence=0,
                    category=TokenCategory.COMMENT,
                ))
                i = end
                continue

            # String literals
            if src[i] in ('"', "'"):
                quote = src[i]
                j = i + 1
                while j < len(src) and src[j] != quote:
                    if src[j] == "\\":
                        j += 1  # skip escaped char
                    j += 1
                j += 1  # past closing quote
                string_val = src[i:j]
                tokens.append(Token(
                    form=string_val, domain="symbolic", valence=0,
                    category=TokenCategory.STRING,
                    data_type="str",
                ))
                i = j
                continue

            # Multi-character operators (try longest match first)
            matched_op = False
            for length in (3, 2):
                candidate = src[i : i + length]
                if candidate in OPERATORS:
                    data = OPERATORS[candidate]
                    tokens.append(Token(
                        form=candidate, domain="symbolic",
                        valence=data.get("valence", 2),
                        category=TokenCategory.OPERATOR,
                        subcategory=data.get("subcategory", ""),
                    ))
                    i += length
                    matched_op = True
                    break
            if matched_op:
                continue

            # Single-character operators
            if src[i] in OPERATORS:
                data = OPERATORS[src[i]]
                tokens.append(Token(
                    form=src[i], domain="symbolic",
                    valence=data.get("valence", 2),
                    category=TokenCategory.OPERATOR,
                    subcategory=data.get("subcategory", ""),
                ))
                i += 1
                continue

            # Separators / brackets
            if src[i] in SCOPE_OPENERS:
                tokens.append(Token(
                    form=src[i], domain="symbolic", valence=0,
                    category=TokenCategory.SEPARATOR,
                    scope_effect="open",
                ))
                i += 1
                continue
            if src[i] in SCOPE_CLOSERS:
                tokens.append(Token(
                    form=src[i], domain="symbolic", valence=0,
                    category=TokenCategory.SEPARATOR,
                    scope_effect="close",
                ))
                i += 1
                continue
            if src[i] in (",", ";", ":"):
                tokens.append(Token(
                    form=src[i], domain="symbolic", valence=0,
                    category=TokenCategory.SEPARATOR,
                ))
                i += 1
                continue

            # Numeric literals
            if src[i].isdigit() or (src[i] == "." and i + 1 < len(src) and src[i + 1].isdigit()):
                j = i
                has_dot = False
                while j < len(src) and (src[j].isdigit() or (src[j] == "." and not has_dot)):
                    if src[j] == ".":
                        has_dot = True
                    j += 1
                num_str = src[i:j]
                tokens.append(Token(
                    form=num_str, domain="symbolic", valence=0,
                    category=TokenCategory.LITERAL,
                    data_type="float" if has_dot else "int",
                ))
                i = j
                continue

            # Identifiers / keywords
            if src[i].isalpha() or src[i] == "_":
                j = i
                while j < len(src) and (src[j].isalnum() or src[j] == "_"):
                    j += 1
                word = src[i:j]
                if word in KEYWORDS:
                    data = KEYWORDS[word]
                    tokens.append(Token(
                        form=word, domain="symbolic",
                        valence=data.get("valence", 0),
                        category=TokenCategory.KEYWORD,
                        subcategory=data.get("subcategory", ""),
                        scope_effect=data.get("scope_effect"),
                    ))
                else:
                    tokens.append(Token(
                        form=word, domain="symbolic", valence=0,
                        category=TokenCategory.IDENTIFIER,
                    ))
                i = j
                continue

            # Unknown character — skip
            i += 1

        return Sequence(tokens)

    def decode(self, sequence: Sequence) -> str:
        """Reconstruct source code from a token sequence.

        Inserts spaces between tokens, with context-aware spacing:
        no space after openers or before closers / commas.
        """
        parts: List[str] = []
        for idx, sym in enumerate(sequence):
            if not isinstance(sym, Token):
                parts.append(sym.form)
                continue

            # Decide whether to prepend a space
            if idx > 0:
                prev = sequence[idx - 1]
                # No space after opening bracket (but not keywords like def/class)
                if (isinstance(prev, Token) and prev.opens_scope
                        and prev.category == TokenCategory.SEPARATOR):
                    pass
                elif (sym.closes_scope
                      and sym.category == TokenCategory.SEPARATOR) or sym.form in (",", ";", ":"):
                    pass
                elif sym.form == "." or (isinstance(prev, Token) and prev.form == "."):
                    pass
                else:
                    parts.append(" ")
            parts.append(sym.form)
        return "".join(parts)

    # -- scope / bracket analysis -------------------------------------------

    def check_brackets(self, sequence: Sequence) -> List[str]:
        """Validate bracket matching.

        Returns a list of error descriptions (empty = valid).
        Bracket matching is the scope analog of valence checking.
        """
        errors: List[str] = []
        stack: List[Tuple[str, int]] = []

        for idx, sym in enumerate(sequence):
            if not isinstance(sym, Token):
                continue
            if sym.form in SCOPE_OPENERS:
                stack.append((sym.form, idx))
            elif sym.form in SCOPE_CLOSERS:
                if not stack:
                    errors.append(f"Unmatched '{sym.form}' at position {idx}")
                else:
                    opener, open_idx = stack.pop()
                    expected_closer = SCOPE_OPENERS[opener]
                    if sym.form != expected_closer:
                        errors.append(
                            f"Mismatched brackets: '{opener}' at {open_idx} "
                            f"closed by '{sym.form}' at {idx}"
                        )
        for opener, open_idx in stack:
            errors.append(f"Unclosed '{opener}' at position {open_idx}")
        return errors

    def scope_depth(self, sequence: Sequence) -> List[int]:
        """Compute the scope depth at each token position.

        Scope depth is the code analog of embedding depth in syntax,
        or electron shell level in chemistry.
        """
        depths: List[int] = []
        current = 0
        for sym in sequence:
            if isinstance(sym, Token) and sym.opens_scope:
                current += 1
            depths.append(current)
            if isinstance(sym, Token) and sym.closes_scope:
                current = max(0, current - 1)
        return depths

    # -- AST construction (simplified) --------------------------------------

    def build_ast(self, sequence: Sequence) -> ASTNode:
        """Build a simplified AST from a token sequence.

        This is a recursive-descent-style parse that handles:
        - Statements: keyword-led structures (if, while, def, etc.)
        - Expressions: binary operators, function calls, literals,
          identifiers
        - Blocks: bracket-delimited groups

        The result is a tree whose structure mirrors constituency in
        natural language and molecular bonding in chemistry.
        """
        root = ASTNode(node_type="program")
        tokens = [s for s in sequence if isinstance(s, Token)]
        self._parse_into(tokens, 0, root)
        return root

    def _parse_into(
        self,
        tokens: List[Token],
        pos: int,
        parent: ASTNode,
    ) -> int:
        """Parse tokens starting at *pos*, adding nodes to *parent*.
        Returns the position after parsing."""
        while pos < len(tokens):
            tok = tokens[pos]

            # Skip comments
            if tok.category == TokenCategory.COMMENT:
                pos += 1
                continue

            # Closing bracket — return to caller
            if tok.closes_scope:
                return pos

            # Keyword-led statement
            if tok.is_keyword and tok.subcategory in (
                "conditional", "loop", "definition", "control",
                "exception", "context", "import",
            ):
                stmt = ASTNode(node_type=f"stmt_{tok.form}", token=tok)
                parent.add_child(stmt)
                pos += 1
                # Consume tokens until end of "line" or scope
                while pos < len(tokens):
                    next_tok = tokens[pos]
                    if next_tok.form == ":":
                        pos += 1
                        break
                    if next_tok.closes_scope:
                        break
                    if next_tok.is_keyword and next_tok.subcategory in (
                        "conditional", "loop", "definition", "exception",
                    ):
                        break
                    child = ASTNode(
                        node_type=f"arg_{next_tok.category.name.lower()}",
                        token=next_tok,
                    )
                    stmt.add_child(child)
                    pos += 1
                continue

            # Opening bracket — parse sub-group
            if tok.opens_scope:
                group = ASTNode(node_type=f"group_{tok.form}")
                parent.add_child(group)
                pos = self._parse_into(tokens, pos + 1, group)
                if pos < len(tokens) and tokens[pos].closes_scope:
                    pos += 1
                continue

            # Expression: operator
            if tok.is_operator:
                expr = ASTNode(node_type="expr_op", token=tok)
                parent.add_child(expr)
                pos += 1
                continue

            # Literal, identifier, separator, string
            node = ASTNode(
                node_type=tok.category.name.lower(),
                token=tok,
            )
            parent.add_child(node)
            pos += 1

        return pos

    # -- pattern analysis ---------------------------------------------------

    def find_idioms(
        self, sequence: Sequence
    ) -> List[Tuple[str, int, int]]:
        """Detect common code idioms (structural patterns).

        Idioms in code are like collocations in language or functional
        groups in chemistry — recurring multi-token patterns with
        conventional meaning.
        """
        results: List[Tuple[str, int, int]] = []
        tokens = [s for s in sequence if isinstance(s, Token)]
        n = len(tokens)

        # Pattern: for x in range(...)
        for i in range(n - 3):
            if (tokens[i].form == "for"
                    and i + 2 < n and tokens[i + 2].form == "in"
                    and i + 3 < n and tokens[i + 3].form == "range"):
                results.append(("for-in-range loop", i, i + 4))

        # Pattern: if x is None / if x is not None
        for i in range(n - 2):
            if (tokens[i].form == "if"
                    and i + 2 < n and tokens[i + 2].form == "is"):
                end = i + 3
                if end < n and tokens[end].form == "not":
                    end += 1
                if end < n and tokens[end].form == "None":
                    results.append(("None check", i, end + 1))

        # Pattern: try/except
        for i in range(n - 1):
            if tokens[i].form == "try":
                for j in range(i + 1, min(i + 20, n)):
                    if tokens[j].form == "except":
                        results.append(("try-except block", i, j + 1))
                        break

        # Pattern: list comprehension [... for ... in ...]
        for i in range(n - 4):
            if tokens[i].form == "[":
                for j in range(i + 1, min(i + 30, n)):
                    if tokens[j].form == "]":
                        inner = tokens[i + 1 : j]
                        if any(t.form == "for" for t in inner) and any(t.form == "in" for t in inner):
                            results.append(("list comprehension", i, j + 1))
                        break

        return results

    # -- type inference (simplified) ----------------------------------------

    def infer_types(
        self, sequence: Sequence
    ) -> Dict[str, str]:
        """Simple type inference for variable assignments.

        Scans for `identifier = literal` patterns and records the
        literal's type.  This is a toy version of Hindley-Milner —
        just enough to demonstrate the concept.
        """
        types: Dict[str, str] = {}
        tokens = [s for s in sequence if isinstance(s, Token)]
        for i in range(len(tokens) - 2):
            if (tokens[i].is_identifier
                    and tokens[i + 1].form == "="
                    and tokens[i + 2].is_literal):
                types[tokens[i].form] = tokens[i + 2].data_type or "unknown"
            elif (tokens[i].is_identifier
                    and tokens[i + 1].form == "="
                    and tokens[i + 2].category == TokenCategory.STRING):
                types[tokens[i].form] = "str"
            elif (tokens[i].is_identifier
                    and tokens[i + 1].form == "="
                    and tokens[i + 2].form in ("True", "False")):
                types[tokens[i].form] = "bool"
            elif (tokens[i].is_identifier
                    and tokens[i + 1].form == "="
                    and tokens[i + 2].form == "None"):
                types[tokens[i].form] = "NoneType"
        return types

    # -- strange loop detection ---------------------------------------------

    def detect_self_reference(
        self, sequence: Sequence
    ) -> List[Tuple[int, int, str]]:
        """Detect self-referential structures in code.

        Self-reference in code includes:
        1. Recursive function calls (a function that calls itself).
        2. Quine-like patterns (code that constructs its own representation).
        3. Meta-programming (eval, exec, compile).
        4. Self-assignment patterns.
        """
        loops: List[Tuple[int, int, str]] = []
        tokens = [s for s in sequence if isinstance(s, Token)]
        n = len(tokens)

        # Find function definitions
        func_names: Dict[str, int] = {}
        for i in range(n - 1):
            if tokens[i].form == "def" and tokens[i + 1].is_identifier:
                func_names[tokens[i + 1].form] = i

        # Find recursive calls (function name appears inside its own body)
        for fname, def_pos in func_names.items():
            # Find the body span (from def to next def or end)
            body_start = def_pos + 2
            body_end = n
            for j in range(body_start, n):
                if tokens[j].form == "def" and j != def_pos:
                    body_end = j
                    break
            # Check if function name appears in body
            for j in range(body_start, body_end):
                if tokens[j].form == fname and tokens[j].is_identifier:
                    loops.append((
                        j, j + 1,
                        f"Recursive call: '{fname}' calls itself "
                        f"(strange loop in function definition)"
                    ))

        # Detect meta-programming
        meta_keywords = {"eval", "exec", "compile", "__import__"}
        for i in range(n):
            if tokens[i].form in meta_keywords:
                loops.append((
                    i, i + 1,
                    f"Meta-programming: '{tokens[i].form}' — code that "
                    f"manipulates code (self-referential substrate)"
                ))

        # Detect self-assignment (x = x)
        for i in range(n - 2):
            if (tokens[i].is_identifier
                    and tokens[i + 1].form == "="
                    and tokens[i + 2].is_identifier
                    and tokens[i].form == tokens[i + 2].form):
                loops.append((
                    i, i + 3,
                    f"Self-assignment: '{tokens[i].form} = {tokens[i].form}'"
                ))

        # Repeating structural patterns
        patterns = self.find_patterns(sequence, min_length=3)
        for pat, positions in patterns.items():
            if len(positions) >= 3:
                loops.append((
                    positions[0],
                    positions[0] + len(pat.split()),
                    f"Repeated code pattern [{pat}] occurs {len(positions)} "
                    f"times (structural self-similarity)",
                ))

        return loops

    # -- complexity metrics -------------------------------------------------

    def cyclomatic_complexity(self, sequence: Sequence) -> int:
        """Compute cyclomatic complexity (simplified).

        Counts decision points: if, elif, while, for, except, and, or.
        Complexity = decisions + 1.
        """
        decision_keywords = {"if", "elif", "while", "for", "except", "and", "or"}
        decisions = sum(
            1 for s in sequence
            if isinstance(s, Token) and s.form in decision_keywords
        )
        return decisions + 1

    def token_statistics(self, sequence: Sequence) -> Dict[str, int]:
        """Compute token-category statistics."""
        stats: Dict[str, int] = {}
        for s in sequence:
            if isinstance(s, Token):
                key = s.category.name.lower()
                stats[key] = stats.get(key, 0) + 1
        return stats
