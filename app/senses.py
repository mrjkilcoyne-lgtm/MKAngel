"""
The Angel's senses -- how she reads the world.

Not through screenshots or pixels, but through grammar, structure,
and derivation.  She reads code like language, errors like symptoms,
and her own state like a strange loop examining itself.

Every perception carries a derivation_path: the grammar rules that
were followed to reach the conclusion.  If the Angel cannot determine
something structurally, she says so honestly rather than guessing.
"""

from __future__ import annotations

import ast
import gc
import logging
import os
import platform
import re
import shutil
import socket
import sys
import textwrap
import time
import tokenize
import io
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from app.paths import mkangel_dir

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sense enum -- the channels through which the Angel reads the world
# ---------------------------------------------------------------------------

class Sense(Enum):
    """The modes of perception available to the Angel.

    Each sense reads a different layer of reality through structural
    analysis -- grammar rules, not pixel heuristics.
    """

    GRAMMAR = auto()      # Read structure through grammar rules
    CODE = auto()         # Read code/programs through computational grammar
    BINARY = auto()       # Read binary/hex data through symbolic substrate
    ERROR = auto()        # Read error traces through pattern analysis
    NETWORK = auto()      # Read network state through connectivity checks
    FILESYSTEM = auto()   # Read file system through path analysis
    TEMPORAL = auto()     # Read time/sequence patterns
    SELF = auto()         # Introspect own state -- the strange loop


# ---------------------------------------------------------------------------
# Perception -- a single act of structured understanding
# ---------------------------------------------------------------------------

@dataclass
class Perception:
    """A single perception: what was sensed and how the conclusion was reached.

    Every perception carries a derivation_path -- the grammar rules
    used to arrive at the interpretation.  This is the Angel showing
    her working, not hiding behind a black box.
    """

    sense: Sense
    raw_input: str
    interpretation: str
    confidence: float
    derivation_path: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        self.confidence = max(0.0, min(1.0, self.confidence))


# ---------------------------------------------------------------------------
# CodeReader -- reads code as language, not pixels
# ---------------------------------------------------------------------------

class CodeReader:
    """Reads code through structural analysis.

    The Angel reads code the way a linguist reads a sentence: she
    identifies the grammar (language), the parts of speech (functions,
    classes, imports), and the meaning (what the code does).  She
    finds bugs by noticing structural anomalies, not by guessing.
    """

    # Language detection signatures -- structural, not heuristic
    _LANGUAGE_SIGNATURES: dict[str, list[str]] = {
        "python": [
            r"^\s*def\s+\w+\s*\(",
            r"^\s*class\s+\w+",
            r"^\s*import\s+\w+",
            r"^\s*from\s+\w+\s+import",
            r":\s*$",
        ],
        "java": [
            r"^\s*public\s+(class|interface|enum)\s+",
            r"^\s*(public|private|protected)\s+\w+\s+\w+\s*\(",
            r"^\s*import\s+[\w.]+;",
            r"System\.out\.print",
        ],
        "javascript": [
            r"\bconst\s+\w+\s*=",
            r"\blet\s+\w+\s*=",
            r"\bfunction\s+\w+\s*\(",
            r"=>\s*\{",
            r"\bconsole\.log\b",
        ],
        "c": [
            r"#include\s*<",
            r"\bint\s+main\s*\(",
            r"\bmalloc\s*\(",
            r"\bprintf\s*\(",
            r"\bstruct\s+\w+\s*\{",
        ],
        "rust": [
            r"\bfn\s+\w+\s*\(",
            r"\blet\s+mut\s+",
            r"\bimpl\s+\w+",
            r"\buse\s+\w+::",
            r"->",
        ],
        "go": [
            r"\bfunc\s+\w+\s*\(",
            r"\bpackage\s+\w+",
            r"\bfmt\.Print",
            r":=",
        ],
    }

    # Structural issue patterns -- the Angel reads these like
    # a doctor reads symptoms
    _ISSUE_PATTERNS: list[dict[str, Any]] = [
        {
            "id": "bare_except",
            "pattern": r"except\s*:",
            "severity": "warning",
            "message": "Bare except clause swallows all exceptions including "
                       "KeyboardInterrupt and SystemExit",
            "fix_hint": "Use 'except Exception:' or catch specific types",
        },
        {
            "id": "broad_except",
            "pattern": r"except\s+Exception\s*:",
            "severity": "info",
            "message": "Broad exception handling -- may mask specific errors",
            "fix_hint": "Consider catching more specific exception types",
        },
        {
            "id": "global_usage",
            "pattern": r"^\s*global\s+\w+",
            "severity": "warning",
            "message": "Global variable mutation detected -- breaks referential "
                       "transparency",
            "fix_hint": "Pass values as parameters or use a class",
        },
        {
            "id": "eval_usage",
            "pattern": r"\beval\s*\(",
            "severity": "error",
            "message": "eval() usage detected -- arbitrary code execution risk",
            "fix_hint": "Use ast.literal_eval() for data or a proper parser",
        },
        {
            "id": "exec_usage",
            "pattern": r"\bexec\s*\(",
            "severity": "error",
            "message": "exec() usage detected -- arbitrary code execution risk",
            "fix_hint": "Avoid exec(); use structured approaches instead",
        },
        {
            "id": "mutable_default",
            "pattern": r"def\s+\w+\s*\([^)]*=\s*(\[\s*\]|\{\s*\})",
            "severity": "warning",
            "message": "Mutable default argument (list or dict) -- shared across calls",
            "fix_hint": "Use None as default and create inside the function",
        },
        {
            "id": "star_import",
            "pattern": r"from\s+\w+\s+import\s+\*",
            "severity": "warning",
            "message": "Wildcard import pollutes namespace and hides origins",
            "fix_hint": "Import specific names",
        },
        {
            "id": "shadowed_builtin",
            "pattern": r"^\s*(list|dict|set|str|int|float|type|id|input|"
                       r"open|print|len|range|map|filter)\s*=",
            "severity": "warning",
            "message": "Variable shadows a Python builtin",
            "fix_hint": "Choose a different variable name",
        },
        {
            "id": "nested_loop_depth",
            "pattern": r"for\s+.*:\s*\n(\s+).*\n\1\s+for\s+.*:\s*\n\1\s+.*\n\1\s+\s+for\s+",
            "severity": "info",
            "message": "Deeply nested loops detected (3+ levels)",
            "fix_hint": "Consider breaking into helper functions",
        },
        {
            "id": "todo_fixme",
            "pattern": r"#\s*(TODO|FIXME|HACK|XXX|WORKAROUND)\b",
            "severity": "info",
            "message": "Outstanding TODO/FIXME marker found",
            "fix_hint": "Address or track in issue tracker",
        },
        {
            "id": "hardcoded_secret",
            "pattern": r"""(?:password|secret|api_key|token)\s*=\s*['"][^'"]{4,}['"]""",
            "severity": "error",
            "message": "Possible hardcoded secret or credential",
            "fix_hint": "Use environment variables or a secrets manager",
        },
    ]

    def analyze(self, code: str) -> dict[str, Any]:
        """Analyse code through structural grammar, not guesswork.

        Detects the language, identifies structural components (functions,
        classes, imports), and maps the code's grammar.

        Args:
            code: Source code to analyse.

        Returns:
            Structured analysis with language, structure, metrics.
        """
        language = self._detect_language(code)
        lines = code.splitlines()
        total_lines = len(lines)
        blank_lines = sum(1 for ln in lines if not ln.strip())
        comment_lines = self._count_comments(code, language)
        code_lines = total_lines - blank_lines - comment_lines

        structure = self._extract_structure(code, language)
        derivation = [
            f"DETECT_LANG -> {language}",
            f"COUNT_LINES -> total={total_lines}, code={code_lines}, "
            f"blank={blank_lines}, comment={comment_lines}",
        ]
        for kind, items in structure.items():
            if items:
                derivation.append(
                    f"EXTRACT_{kind.upper()} -> {len(items)} found"
                )

        return {
            "language": language,
            "metrics": {
                "total_lines": total_lines,
                "code_lines": code_lines,
                "blank_lines": blank_lines,
                "comment_lines": comment_lines,
            },
            "structure": structure,
            "derivation_path": derivation,
        }

    def find_issues(self, code: str) -> list[dict[str, Any]]:
        """Find potential issues through structural pattern matching.

        The Angel reads the code's structure for anomalies, like a
        doctor reading symptoms.  She doesn't guess -- she matches
        known structural anti-patterns.

        Args:
            code: Source code to examine.

        Returns:
            List of issues found, each with id, severity, message, line,
            and fix_hint.
        """
        issues: list[dict[str, Any]] = []

        for line_no, line in enumerate(code.splitlines(), 1):
            for pattern_def in self._ISSUE_PATTERNS:
                if re.search(pattern_def["pattern"], line):
                    issues.append({
                        "id": pattern_def["id"],
                        "severity": pattern_def["severity"],
                        "message": pattern_def["message"],
                        "line": line_no,
                        "line_text": line.rstrip(),
                        "fix_hint": pattern_def["fix_hint"],
                        "derivation": f"MATCH_PATTERN({pattern_def['id']}) "
                                      f"at line {line_no}",
                    })

        # Python-specific: try to parse the AST for deeper issues
        try:
            tree = ast.parse(code)
            issues.extend(self._ast_issues(tree))
        except SyntaxError as e:
            issues.append({
                "id": "syntax_error",
                "severity": "error",
                "message": f"Syntax error: {e.msg}",
                "line": e.lineno or 0,
                "line_text": (e.text or "").rstrip() if e.text else "",
                "fix_hint": "Fix the syntax error before further analysis",
                "derivation": f"AST_PARSE_FAIL -> SyntaxError at line {e.lineno}",
            })

        return issues

    def explain(self, code: str) -> str:
        """Explain what code does through structural reading.

        Args:
            code: Source code to explain.

        Returns:
            Human-readable explanation derived from structural analysis.
        """
        analysis = self.analyze(code)
        issues = self.find_issues(code)

        parts = []
        lang = analysis["language"]
        parts.append(f"Language: {lang}")
        parts.append("")

        m = analysis["metrics"]
        parts.append(
            f"Structure: {m['code_lines']} lines of code, "
            f"{m['comment_lines']} comments, "
            f"{m['blank_lines']} blank lines"
        )
        parts.append("")

        structure = analysis["structure"]
        if structure.get("classes"):
            parts.append("Classes defined:")
            for cls in structure["classes"]:
                parts.append(f"  - {cls}")
        if structure.get("functions"):
            parts.append("Functions defined:")
            for fn in structure["functions"]:
                parts.append(f"  - {fn}")
        if structure.get("imports"):
            parts.append(f"Imports: {len(structure['imports'])} modules")

        if issues:
            parts.append("")
            parts.append(f"Issues detected ({len(issues)}):")
            for issue in issues[:10]:
                parts.append(
                    f"  [{issue['severity'].upper()}] line {issue['line']}: "
                    f"{issue['message']}"
                )

        parts.append("")
        parts.append("Derivation path:")
        for step in analysis["derivation_path"]:
            parts.append(f"  {step}")

        return "\n".join(parts)

    # -- Internal methods --------------------------------------------------

    def _detect_language(self, code: str) -> str:
        """Detect programming language through structural signatures."""
        scores: dict[str, int] = {}
        for lang, patterns in self._LANGUAGE_SIGNATURES.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, code, re.MULTILINE):
                    score += 1
            if score > 0:
                scores[lang] = score

        if not scores:
            return "unknown"
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def _count_comments(self, code: str, language: str) -> int:
        """Count comment lines based on language grammar."""
        count = 0
        if language in ("python",):
            in_docstring = False
            for line in code.splitlines():
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_docstring = not in_docstring
                    count += 1
                elif in_docstring:
                    count += 1
                elif stripped.startswith("#"):
                    count += 1
        elif language in ("java", "javascript", "c", "rust", "go"):
            in_block = False
            for line in code.splitlines():
                stripped = line.strip()
                if "/*" in stripped:
                    in_block = True
                    count += 1
                elif "*/" in stripped:
                    in_block = False
                    count += 1
                elif in_block:
                    count += 1
                elif stripped.startswith("//"):
                    count += 1
        else:
            for line in code.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith("//"):
                    count += 1
        return count

    def _extract_structure(self, code: str, language: str) -> dict[str, list[str]]:
        """Extract structural elements: classes, functions, imports."""
        result: dict[str, list[str]] = {
            "classes": [],
            "functions": [],
            "imports": [],
            "decorators": [],
        }

        if language == "python":
            # Use AST for Python -- the most accurate grammar reading
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        bases = ", ".join(
                            getattr(b, "id", getattr(b, "attr", "?"))
                            for b in node.bases
                        )
                        sig = f"{node.name}({bases})" if bases else node.name
                        result["classes"].append(sig)
                    elif isinstance(node, ast.FunctionDef):
                        args = ", ".join(a.arg for a in node.args.args)
                        result["functions"].append(f"{node.name}({args})")
                    elif isinstance(node, ast.AsyncFunctionDef):
                        args = ", ".join(a.arg for a in node.args.args)
                        result["functions"].append(f"async {node.name}({args})")
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            result["imports"].append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        for alias in node.names:
                            result["imports"].append(f"{module}.{alias.name}")
                return result
            except SyntaxError:
                pass  # Fall through to regex

        # Regex fallback for all languages
        for line in code.splitlines():
            stripped = line.strip()
            # Class definitions
            m = re.match(r"^\s*(?:public\s+)?class\s+(\w+)", stripped)
            if m:
                result["classes"].append(m.group(1))
            # Function definitions
            m = re.match(
                r"^\s*(?:pub(?:lic)?\s+)?(?:async\s+)?(?:def|fn|func|function)"
                r"\s+(\w+)\s*\(",
                stripped,
            )
            if m:
                result["functions"].append(m.group(1))
            # Imports
            m = re.match(r"^\s*(?:import|from|use|require|include)\s+(.+)", stripped)
            if m:
                result["imports"].append(m.group(1).rstrip(";").strip())
            # Decorators
            m = re.match(r"^\s*@(\w+)", stripped)
            if m:
                result["decorators"].append(m.group(1))

        return result

    def _ast_issues(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Find issues through AST analysis -- the deepest structural reading."""
        issues = []

        for node in ast.walk(tree):
            # Unused variables in assignments
            if isinstance(node, ast.FunctionDef):
                # Check for functions with too many arguments
                n_args = len(node.args.args)
                if n_args > 7:
                    issues.append({
                        "id": "too_many_args",
                        "severity": "info",
                        "message": (
                            f"Function '{node.name}' has {n_args} parameters "
                            f"-- consider using a config object"
                        ),
                        "line": node.lineno,
                        "line_text": f"def {node.name}(...)",
                        "fix_hint": "Group related parameters into a "
                                    "dataclass or dict",
                        "derivation": f"AST_WALK -> FunctionDef({node.name}) "
                                      f"-> arg_count={n_args} > 7",
                    })

                # Check for overly long functions
                end_line = getattr(node, "end_lineno", None)
                if end_line:
                    length = end_line - node.lineno
                    if length > 50:
                        issues.append({
                            "id": "long_function",
                            "severity": "info",
                            "message": (
                                f"Function '{node.name}' is {length} lines -- "
                                f"consider breaking into smaller functions"
                            ),
                            "line": node.lineno,
                            "line_text": f"def {node.name}(...)",
                            "fix_hint": "Extract logical sections into "
                                        "helper functions",
                            "derivation": f"AST_WALK -> FunctionDef({node.name}) "
                                          f"-> length={length} > 50",
                        })

            # Nested function depth (closures inside closures)
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.FunctionDef) and child is not node:
                        for grandchild in ast.walk(child):
                            if (isinstance(grandchild, ast.FunctionDef)
                                    and grandchild is not child):
                                issues.append({
                                    "id": "deep_nesting",
                                    "severity": "warning",
                                    "message": (
                                        "Triple-nested function definition "
                                        "detected -- high cognitive complexity"
                                    ),
                                    "line": grandchild.lineno,
                                    "line_text": f"def {grandchild.name}(...)",
                                    "fix_hint": "Flatten nesting by extracting "
                                                "to module-level functions",
                                    "derivation": "AST_WALK -> nested_depth >= 3",
                                })
                                break
                        break

        return issues


# ---------------------------------------------------------------------------
# ErrorReader -- reads error traces like a doctor reads symptoms
# ---------------------------------------------------------------------------

class ErrorReader:
    """Reads error traces through structural pattern analysis.

    The Angel reads tracebacks the way a doctor reads symptoms:
    she identifies the type of ailment, traces the causal chain,
    and prescribes structural remedies.  She does not guess.
    """

    # Known error pattern database -- structural diagnosis
    _ERROR_PATTERNS: list[dict[str, Any]] = [
        {
            "pattern": r"ModuleNotFoundError: No module named '(\w+)'",
            "category": "import",
            "diagnosis": "Module '{match}' is not installed or not on sys.path",
            "fixes": [
                "Install with: pip install {match}",
                "Check virtualenv is activated",
                "Verify the module name spelling",
            ],
        },
        {
            "pattern": r"ImportError: cannot import name '(\w+)' from '([\w.]+)'",
            "category": "import",
            "diagnosis": "Name '{g1}' does not exist in module '{g2}'",
            "fixes": [
                "Check the name exists in the module's __all__ or source",
                "Module version may have changed -- check changelog",
                "Circular import possible -- restructure imports",
            ],
        },
        {
            "pattern": r"AttributeError: '(\w+)' object has no attribute '(\w+)'",
            "category": "attribute",
            "diagnosis": "Object of type '{g1}' does not have attribute '{g2}'",
            "fixes": [
                "Check spelling of attribute name",
                "Object may be None or a different type than expected",
                "Attribute may have been renamed in newer version",
            ],
        },
        {
            "pattern": r"TypeError: (\w+)\(\) got an unexpected keyword argument '(\w+)'",
            "category": "call",
            "diagnosis": "Function '{g1}' does not accept keyword '{g2}'",
            "fixes": [
                "Check the function signature for valid parameters",
                "API may have changed -- check documentation",
                "Possible typo in keyword argument name",
            ],
        },
        {
            "pattern": r"TypeError: (\w+)\(\) takes (\d+) positional arguments? but (\d+) (?:was|were) given",
            "category": "call",
            "diagnosis": "Argument count mismatch: {g1}() expects {g2} but got {g3}",
            "fixes": [
                "Check the function signature",
                "Missing 'self' parameter in a method definition",
                "Too many/few arguments in the call",
            ],
        },
        {
            "pattern": r"KeyError: ['\"]?(\w+)['\"]?",
            "category": "lookup",
            "diagnosis": "Key '{match}' not found in dictionary",
            "fixes": [
                "Use dict.get(key, default) for safe access",
                "Check the key exists before accessing",
                "Key may have been renamed or is case-sensitive",
            ],
        },
        {
            "pattern": r"IndexError: list index out of range",
            "category": "index",
            "diagnosis": "List access with index beyond its length",
            "fixes": [
                "Check list length before accessing",
                "Off-by-one error in loop bounds",
                "List may be empty -- add a guard",
            ],
        },
        {
            "pattern": r"FileNotFoundError: \[Errno 2\] No such file or directory: '([^']+)'",
            "category": "filesystem",
            "diagnosis": "File or directory not found: '{match}'",
            "fixes": [
                "Verify the path exists",
                "Use Path.exists() before opening",
                "Path may be relative -- use absolute path or check cwd",
            ],
        },
        {
            "pattern": r"PermissionError: \[Errno 13\] Permission denied: '([^']+)'",
            "category": "filesystem",
            "diagnosis": "Permission denied for: '{match}'",
            "fixes": [
                "Check file permissions (ls -la)",
                "File may be locked by another process",
                "Run with appropriate permissions",
            ],
        },
        {
            "pattern": r"RecursionError: maximum recursion depth exceeded",
            "category": "recursion",
            "diagnosis": "Infinite or too-deep recursion detected",
            "fixes": [
                "Check base case in recursive function",
                "Convert recursion to iteration",
                "Increase sys.setrecursionlimit() if depth is legitimate",
            ],
        },
        {
            "pattern": r"MemoryError",
            "category": "resource",
            "diagnosis": "Process ran out of memory",
            "fixes": [
                "Process data in chunks instead of loading all at once",
                "Use generators instead of lists",
                "Check for memory leaks (circular references)",
            ],
        },
        {
            "pattern": r"ConnectionRefusedError",
            "category": "network",
            "diagnosis": "Connection refused -- service not listening",
            "fixes": [
                "Check the service is running",
                "Verify host and port are correct",
                "Check firewall rules",
            ],
        },
        {
            "pattern": r"TimeoutError|socket\.timeout",
            "category": "network",
            "diagnosis": "Operation timed out",
            "fixes": [
                "Increase timeout value",
                "Check network connectivity",
                "Service may be overloaded -- retry with backoff",
            ],
        },
        {
            "pattern": r"UnicodeDecodeError: '(\w+)' codec can't decode byte",
            "category": "encoding",
            "diagnosis": "File or data not in expected encoding ({match})",
            "fixes": [
                "Specify correct encoding: open(file, encoding='utf-8')",
                "Try reading as binary: open(file, 'rb')",
                "Data may be binary, not text",
            ],
        },
        {
            "pattern": r"json\.decoder\.JSONDecodeError",
            "category": "parse",
            "diagnosis": "Invalid JSON data",
            "fixes": [
                "Validate JSON with a linter",
                "Check for trailing commas or single quotes",
                "Response may be HTML error page, not JSON",
            ],
        },
        {
            "pattern": r"ValueError: invalid literal for int\(\) with base \d+: '([^']*)'",
            "category": "conversion",
            "diagnosis": "Cannot convert '{match}' to integer",
            "fixes": [
                "Validate input before conversion",
                "Strip whitespace: value.strip()",
                "Input may contain non-numeric characters",
            ],
        },
        {
            "pattern": r"OSError: \[Errno 28\] No space left on device",
            "category": "resource",
            "diagnosis": "Disk is full",
            "fixes": [
                "Free disk space",
                "Clean temporary files",
                "Write to a different filesystem",
            ],
        },
        {
            # Java / Android pattern
            "pattern": r"java\.lang\.(\w+Exception): (.+)",
            "category": "java",
            "diagnosis": "Java exception: {g1} -- {g2}",
            "fixes": [
                "Check the Java stack trace for the originating method",
                "Verify Android API level compatibility",
                "Check that required permissions are declared",
            ],
        },
    ]

    def parse_traceback(self, tb: str) -> dict[str, Any]:
        """Parse a traceback into structured components.

        Reads the traceback's grammar: the frame stack, the exception
        type, and the causal chain.

        Args:
            tb: Raw traceback text (Python, Java, or generic).

        Returns:
            Structured parse with type, message, frames, root_file,
            root_line.
        """
        result: dict[str, Any] = {
            "exception_type": "Unknown",
            "exception_message": "",
            "frames": [],
            "root_file": "",
            "root_line": 0,
            "language": "unknown",
            "derivation_path": [],
        }

        lines = tb.strip().splitlines()
        if not lines:
            result["derivation_path"].append("EMPTY_INPUT -> no traceback")
            return result

        # -- Detect traceback format (Python, Java, generic) ---------------

        # Python traceback
        if any("Traceback (most recent call last)" in ln for ln in lines):
            result["language"] = "python"
            result["derivation_path"].append("DETECT_FORMAT -> Python traceback")
            return self._parse_python_traceback(lines, result)

        # Java stack trace
        if any(re.match(r"\s+at\s+[\w.$]+\([\w.]+:\d+\)", ln) for ln in lines):
            result["language"] = "java"
            result["derivation_path"].append("DETECT_FORMAT -> Java stack trace")
            return self._parse_java_traceback(lines, result)

        # Generic -- try to extract what we can
        result["language"] = "generic"
        result["derivation_path"].append("DETECT_FORMAT -> generic error text")
        return self._parse_generic_error(lines, result)

    def identify_root_cause(self, tb: str) -> str:
        """Trace to the most likely root cause.

        Follows the causal chain structurally -- the deepest frame
        in the stack that is user code (not stdlib/library).

        Args:
            tb: Raw traceback text.

        Returns:
            Human-readable root cause statement.
        """
        parsed = self.parse_traceback(tb)
        exc_type = parsed["exception_type"]
        exc_msg = parsed["exception_message"]
        frames = parsed["frames"]

        if not frames and not exc_type:
            return "Could not structurally determine root cause from this input."

        # Find deepest user-code frame (not stdlib, not site-packages)
        user_frames = [
            f for f in frames
            if not any(skip in f.get("file", "") for skip in (
                "site-packages", "lib/python", "Lib\\",
                "<frozen", "<string>",
            ))
        ]

        root_frame = user_frames[-1] if user_frames else (
            frames[-1] if frames else {}
        )

        parts = [f"Root cause: {exc_type}"]
        if exc_msg:
            parts.append(f"  Message: {exc_msg}")
        if root_frame:
            parts.append(
                f"  Origin: {root_frame.get('file', '?')} "
                f"line {root_frame.get('line', '?')} "
                f"in {root_frame.get('function', '?')}"
            )
            if root_frame.get("code"):
                parts.append(f"  Code: {root_frame['code']}")

        return "\n".join(parts)

    def suggest_fix(self, tb: str) -> list[str]:
        """Suggest fixes based on error pattern database.

        Matches the error against known structural patterns and
        returns targeted fixes -- not guesses.

        Args:
            tb: Raw traceback text.

        Returns:
            List of suggested fixes, ordered by relevance.
        """
        fixes: list[str] = []
        tb_text = tb.strip()

        for pattern_def in self._ERROR_PATTERNS:
            m = re.search(pattern_def["pattern"], tb_text)
            if m:
                # Format the fixes with captured groups
                for fix_template in pattern_def["fixes"]:
                    fix = fix_template
                    if m.groups():
                        fix = fix.replace("{match}", m.group(1) if m.groups() else "")
                        for i, g in enumerate(m.groups(), 1):
                            fix = fix.replace(f"{{g{i}}}", g or "")
                    fixes.append(fix)
                break  # Use first matching pattern

        if not fixes:
            fixes.append(
                "No known pattern matched. Read the full traceback "
                "structurally: the last frame before the exception is "
                "usually the origin."
            )

        return fixes

    # -- Internal parsers --------------------------------------------------

    def _parse_python_traceback(
        self, lines: list[str], result: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse Python traceback into structured data."""
        frames = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Frame line: '  File "x.py", line N, in func'
            m = re.match(
                r'\s*File "([^"]+)", line (\d+)(?:, in (.+))?', line
            )
            if m:
                frame: dict[str, Any] = {
                    "file": m.group(1),
                    "line": int(m.group(2)),
                    "function": m.group(3) or "<module>",
                    "code": "",
                }
                # Next line is usually the code
                if i + 1 < len(lines) and not lines[i + 1].startswith("  File"):
                    code_line = lines[i + 1].strip()
                    if not re.match(r"^\w+Error:", code_line):
                        frame["code"] = code_line
                        i += 1
                frames.append(frame)
                result["derivation_path"].append(
                    f"FRAME -> {frame['file']}:{frame['line']} "
                    f"in {frame['function']}"
                )
            i += 1

        result["frames"] = frames

        # Extract exception from last line(s)
        for line in reversed(lines):
            m = re.match(r"^(\w+(?:\.\w+)*(?:Error|Exception|Warning))\s*:\s*(.*)", line)
            if m:
                result["exception_type"] = m.group(1)
                result["exception_message"] = m.group(2).strip()
                result["derivation_path"].append(
                    f"EXCEPTION -> {result['exception_type']}: "
                    f"{result['exception_message'][:80]}"
                )
                break
            # Exception without message
            m = re.match(r"^(\w+(?:\.\w+)*(?:Error|Exception|Warning))\s*$", line)
            if m:
                result["exception_type"] = m.group(1)
                result["derivation_path"].append(
                    f"EXCEPTION -> {result['exception_type']} (no message)"
                )
                break

        if frames:
            result["root_file"] = frames[-1]["file"]
            result["root_line"] = frames[-1]["line"]

        return result

    def _parse_java_traceback(
        self, lines: list[str], result: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse Java stack trace into structured data."""
        frames = []

        # First line is usually the exception
        if lines:
            m = re.match(r"([\w.$]+(?:Exception|Error))\s*:\s*(.*)", lines[0])
            if m:
                result["exception_type"] = m.group(1)
                result["exception_message"] = m.group(2).strip()
            else:
                result["exception_message"] = lines[0].strip()

        for line in lines[1:]:
            m = re.match(r"\s+at\s+([\w.$]+)\(([\w.]+):(\d+)\)", line)
            if m:
                frame = {
                    "function": m.group(1),
                    "file": m.group(2),
                    "line": int(m.group(3)),
                    "code": "",
                }
                frames.append(frame)
                result["derivation_path"].append(
                    f"FRAME -> {frame['file']}:{frame['line']} "
                    f"in {frame['function']}"
                )

        result["frames"] = frames
        if frames:
            result["root_file"] = frames[0]["file"]
            result["root_line"] = frames[0]["line"]

        return result

    def _parse_generic_error(
        self, lines: list[str], result: dict[str, Any]
    ) -> dict[str, Any]:
        """Best-effort structural parse of an unknown error format."""
        # Try to find anything that looks like an error declaration
        for line in lines:
            m = re.search(
                r"((?:Error|Exception|FATAL|CRITICAL|FAILURE)[\w]*)\s*[:\-]\s*(.*)",
                line, re.IGNORECASE,
            )
            if m:
                result["exception_type"] = m.group(1)
                result["exception_message"] = m.group(2).strip()
                result["derivation_path"].append(
                    f"GENERIC_MATCH -> {result['exception_type']}"
                )
                break

        # Look for file:line patterns
        for line in lines:
            m = re.search(r"([\w/\\._-]+\.(?:py|java|js|ts|rs|go|c|cpp|rb)):(\d+)", line)
            if m:
                result["root_file"] = m.group(1)
                result["root_line"] = int(m.group(2))
                result["frames"].append({
                    "file": m.group(1),
                    "line": int(m.group(2)),
                    "function": "?",
                    "code": "",
                })
                result["derivation_path"].append(
                    f"FILE_LINE_MATCH -> {m.group(1)}:{m.group(2)}"
                )

        if not result["derivation_path"]:
            result["derivation_path"].append(
                "NO_STRUCTURE_FOUND -> input does not match known "
                "error grammar"
            )

        return result


# ---------------------------------------------------------------------------
# StateMonitor -- the strange loop examining itself
# ---------------------------------------------------------------------------

class StateMonitor:
    """Monitors the Angel's own runtime state.

    The strange loop: a system examining itself.  The monitor checks
    memory, disk, network, and subsystem health through direct
    structural queries -- not screenshots, not guessing.
    """

    def health_check(self) -> dict[str, Any]:
        """Check all subsystems and return structured health report.

        Returns:
            Dict with status for each subsystem: memory, disk,
            grammars, network, python.
        """
        report: dict[str, Any] = {
            "timestamp": time.time(),
            "overall": "healthy",
            "subsystems": {},
            "derivation_path": ["SELF_INSPECT -> health_check initiated"],
        }

        # Memory subsystem
        mem = self._check_memory()
        report["subsystems"]["memory"] = mem
        report["derivation_path"].append(
            f"CHECK_MEMORY -> {mem['status']} "
            f"(process_mb={mem.get('process_mb', '?')})"
        )

        # Disk subsystem
        disk = self._check_disk()
        report["subsystems"]["disk"] = disk
        report["derivation_path"].append(
            f"CHECK_DISK -> {disk['status']} "
            f"(free_gb={disk.get('free_gb', '?')})"
        )

        # Grammar subsystem
        grammars = self._check_grammars()
        report["subsystems"]["grammars"] = grammars
        report["derivation_path"].append(
            f"CHECK_GRAMMARS -> {grammars['status']} "
            f"(loaded={grammars.get('domains_available', '?')})"
        )

        # Network subsystem
        net = self._check_network()
        report["subsystems"]["network"] = net
        report["derivation_path"].append(
            f"CHECK_NETWORK -> {net['status']}"
        )

        # Python runtime
        py = self._check_python()
        report["subsystems"]["python"] = py
        report["derivation_path"].append(
            f"CHECK_PYTHON -> {py['status']} "
            f"(version={py.get('version', '?')})"
        )

        # Derive overall status
        statuses = [s["status"] for s in report["subsystems"].values()]
        if any(s == "error" for s in statuses):
            report["overall"] = "degraded"
        elif any(s == "warning" for s in statuses):
            report["overall"] = "warning"

        report["derivation_path"].append(
            f"OVERALL -> {report['overall']}"
        )

        return report

    def resource_usage(self) -> dict[str, Any]:
        """Report current resource usage.

        Returns:
            Dict with memory, disk, gc, and active module counts.
        """
        usage: dict[str, Any] = {
            "timestamp": time.time(),
        }

        # Memory
        try:
            import resource as resource_mod
            ru = resource_mod.getrusage(resource_mod.RUSAGE_SELF)
            usage["memory_mb"] = ru.ru_maxrss / 1024  # KB to MB on Linux
        except (ImportError, AttributeError):
            # Windows / platforms without resource module
            usage["memory_mb"] = self._estimate_process_memory()

        # Disk
        try:
            data_dir = mkangel_dir()
            if data_dir.exists():
                total_size = sum(
                    f.stat().st_size for f in data_dir.rglob("*") if f.is_file()
                )
                usage["data_dir_mb"] = round(total_size / (1024 * 1024), 2)
            else:
                usage["data_dir_mb"] = 0.0
        except OSError:
            usage["data_dir_mb"] = -1

        # Garbage collector state
        gc_stats = gc.get_stats()
        usage["gc"] = {
            f"gen{i}": {
                "collections": g.get("collections", 0),
                "collected": g.get("collected", 0),
                "uncollectable": g.get("uncollectable", 0),
            }
            for i, g in enumerate(gc_stats)
        }

        # Loaded modules
        app_modules = [
            m for m in sys.modules if m.startswith("app.") or m.startswith("glm.")
        ]
        usage["loaded_modules"] = {
            "total": len(sys.modules),
            "app_modules": len(app_modules),
            "app_module_names": sorted(app_modules),
        }

        return usage

    def is_degraded(self) -> bool:
        """True if any subsystem is unhealthy.

        A simple boolean gate -- the Angel checks herself before
        she wrecks herself.
        """
        try:
            report = self.health_check()
            return report["overall"] != "healthy"
        except Exception:
            return True  # If we can't even check, assume degraded

    # -- Internal checks ---------------------------------------------------

    def _check_memory(self) -> dict[str, Any]:
        """Check memory subsystem."""
        result: dict[str, Any] = {"status": "healthy"}
        try:
            mb = self._estimate_process_memory()
            result["process_mb"] = round(mb, 1)
            result["gc_tracked_objects"] = len(gc.get_objects())
            if mb > 500:
                result["status"] = "warning"
                result["note"] = "High memory usage (> 500MB)"
            elif mb > 1000:
                result["status"] = "error"
                result["note"] = "Critical memory usage (> 1GB)"
        except Exception as e:
            result["status"] = "error"
            result["note"] = f"Cannot read memory: {e}"
        return result

    def _check_disk(self) -> dict[str, Any]:
        """Check disk subsystem."""
        result: dict[str, Any] = {"status": "healthy"}
        try:
            data_dir = mkangel_dir()
            # Check free space on the partition containing data_dir
            target = data_dir if data_dir.exists() else Path.home()
            usage = shutil.disk_usage(str(target))
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            result["free_gb"] = round(free_gb, 2)
            result["total_gb"] = round(total_gb, 2)
            result["used_pct"] = round(
                (usage.used / usage.total) * 100, 1
            )
            if free_gb < 1:
                result["status"] = "error"
                result["note"] = "Less than 1GB free disk space"
            elif free_gb < 5:
                result["status"] = "warning"
                result["note"] = "Less than 5GB free disk space"
        except OSError as e:
            result["status"] = "warning"
            result["note"] = f"Cannot check disk: {e}"
        return result

    def _check_grammars(self) -> dict[str, Any]:
        """Check grammar subsystem availability."""
        result: dict[str, Any] = {"status": "healthy", "domains_available": 0}
        try:
            from glm.angel import Angel
            angel = Angel()
            domains = getattr(angel, "domains", None)
            if domains:
                result["domains_available"] = len(domains)
            else:
                # Try to count grammars from module presence
                try:
                    import glm.grammars
                    gram_path = Path(glm.grammars.__file__).parent
                    domain_files = list(gram_path.glob("*.py"))
                    result["domains_available"] = max(
                        0, len(domain_files) - 1  # minus __init__
                    )
                except Exception:
                    result["domains_available"] = 0
        except ImportError:
            result["status"] = "warning"
            result["note"] = "GLM Angel not importable"
            result["domains_available"] = 0
        except Exception as e:
            result["status"] = "warning"
            result["note"] = f"Grammar check failed: {e}"
        return result

    def _check_network(self) -> dict[str, Any]:
        """Check basic network connectivity."""
        result: dict[str, Any] = {"status": "healthy"}
        try:
            # Try a DNS lookup -- structural check, no data sent
            socket.setdefaulttimeout(3)
            socket.getaddrinfo("dns.google", 443)
            result["dns"] = "reachable"
        except (socket.gaierror, socket.timeout, OSError):
            result["status"] = "warning"
            result["dns"] = "unreachable"
            result["note"] = "DNS resolution failed -- may be offline"
        return result

    def _check_python(self) -> dict[str, Any]:
        """Check Python runtime health."""
        return {
            "status": "healthy",
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
        }

    @staticmethod
    def _estimate_process_memory() -> float:
        """Estimate process memory in MB using available methods."""
        # Try /proc/self/status (Linux/Android)
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        return kb / 1024
        except (FileNotFoundError, OSError, ValueError):
            pass

        # Try resource module (Unix)
        try:
            import resource as resource_mod
            ru = resource_mod.getrusage(resource_mod.RUSAGE_SELF)
            # ru_maxrss is in KB on Linux, bytes on macOS
            if sys.platform == "darwin":
                return ru.ru_maxrss / (1024 * 1024)
            return ru.ru_maxrss / 1024
        except (ImportError, AttributeError):
            pass

        # Rough estimate from gc
        try:
            # Very rough: count tracked objects * estimated avg size
            n_objects = len(gc.get_objects())
            return (n_objects * 64) / (1024 * 1024)  # ~64 bytes avg
        except Exception:
            return -1.0


# ---------------------------------------------------------------------------
# AngelSenses -- the unified sensory engine
# ---------------------------------------------------------------------------

class AngelSenses:
    """The Angel's unified sensory engine.

    The Angel reads the world through structure, not screenshots.
    She reads code like language, errors like symptoms, binary like
    symbolic substrate, and her own state like a strange loop
    examining itself.

    Every perception includes a derivation_path: the grammar rules
    followed to reach the conclusion.  No black boxes.  No guessing.
    If she cannot determine something structurally, she says so.
    """

    def __init__(self) -> None:
        self._code_reader = CodeReader()
        self._error_reader = ErrorReader()
        self._state_monitor = StateMonitor()
        self._perception_history: list[Perception] = []

    # -- Primary perception interface --------------------------------------

    def perceive(
        self, input_data: str, hint: Sense | None = None
    ) -> Perception:
        """Auto-detect what kind of data this is and interpret it.

        The Angel looks at the structure of the input to determine
        what sense to use.  She does not guess -- she matches
        structural signatures.

        Args:
            input_data: Raw input data (code, error, hex, etc.).
            hint: Optional hint about which sense to use.

        Returns:
            Perception with interpretation and derivation path.
        """
        if hint:
            sense = hint
            derivation = [f"HINT_PROVIDED -> {sense.name}"]
        else:
            sense, derivation = self._detect_sense(input_data)

        # Route to the appropriate reader
        if sense == Sense.CODE:
            return self.read_code(input_data)
        elif sense == Sense.ERROR:
            return self.read_error(input_data)
        elif sense == Sense.BINARY:
            return self.read_binary(input_data)
        elif sense == Sense.SELF:
            state = self.read_state()
            return Perception(
                sense=Sense.SELF,
                raw_input="<self-inspection>",
                interpretation=self._format_state(state),
                confidence=0.95,
                derivation_path=["SELF_INSPECT -> read_state()"],
            )
        elif sense == Sense.FILESYSTEM:
            return self._perceive_filesystem(input_data)
        elif sense == Sense.NETWORK:
            return self._perceive_network(input_data)
        elif sense == Sense.TEMPORAL:
            return self._perceive_temporal(input_data)
        elif sense == Sense.GRAMMAR:
            return self._perceive_grammar(input_data)
        else:
            return Perception(
                sense=sense,
                raw_input=input_data[:200],
                interpretation="Could not structurally interpret this input.",
                confidence=0.1,
                derivation_path=derivation + [
                    "FALLBACK -> no structural reading available"
                ],
            )

    def read_code(self, code: str) -> Perception:
        """Parse code structure through computational grammar.

        Args:
            code: Source code to read.

        Returns:
            Perception of the code's structure and any issues.
        """
        analysis = self._code_reader.analyze(code)
        issues = self._code_reader.find_issues(code)

        interpretation_parts = []
        lang = analysis["language"]
        m = analysis["metrics"]
        interpretation_parts.append(
            f"{lang.title()} code: {m['code_lines']} lines, "
            f"{m['comment_lines']} comments"
        )

        structure = analysis["structure"]
        if structure.get("classes"):
            interpretation_parts.append(
                f"Classes: {', '.join(structure['classes'])}"
            )
        if structure.get("functions"):
            fn_list = structure["functions"]
            if len(fn_list) > 5:
                interpretation_parts.append(
                    f"Functions: {', '.join(fn_list[:5])} "
                    f"(+{len(fn_list) - 5} more)"
                )
            else:
                interpretation_parts.append(
                    f"Functions: {', '.join(fn_list)}"
                )

        if issues:
            error_count = sum(1 for i in issues if i["severity"] == "error")
            warn_count = sum(1 for i in issues if i["severity"] == "warning")
            info_count = sum(1 for i in issues if i["severity"] == "info")
            parts = []
            if error_count:
                parts.append(f"{error_count} errors")
            if warn_count:
                parts.append(f"{warn_count} warnings")
            if info_count:
                parts.append(f"{info_count} info")
            interpretation_parts.append(f"Issues: {', '.join(parts)}")

            for issue in issues[:3]:
                interpretation_parts.append(
                    f"  [{issue['severity'].upper()}] "
                    f"line {issue['line']}: {issue['message']}"
                )

        confidence = 0.9 if lang != "unknown" else 0.5
        if issues:
            confidence = min(confidence, 0.85)

        derivation = analysis["derivation_path"]
        for issue in issues[:5]:
            derivation.append(issue.get("derivation", f"ISSUE -> {issue['id']}"))

        perception = Perception(
            sense=Sense.CODE,
            raw_input=code[:200] + ("..." if len(code) > 200 else ""),
            interpretation="\n".join(interpretation_parts),
            confidence=confidence,
            derivation_path=derivation,
        )
        self._perception_history.append(perception)
        return perception

    def read_binary(self, data: str) -> Perception:
        """Interpret hex/binary data through symbolic substrate.

        Reads binary/hex data structurally: detects format (hex,
        base64, raw binary representation), identifies known
        signatures (file magic bytes, protocol headers), and
        interprets the symbolic structure.

        Args:
            data: Hex string, binary representation, or base64 data.

        Returns:
            Perception of the binary data's structure.
        """
        derivation = ["READ_BINARY -> begin structural analysis"]
        cleaned = data.strip()
        interpretation_parts = []

        # Detect hex string
        hex_match = re.match(r"^(?:0x)?([0-9a-fA-F\s]+)$", cleaned)
        if hex_match:
            hex_str = hex_match.group(1).replace(" ", "")
            derivation.append(f"DETECT_FORMAT -> hexadecimal ({len(hex_str)} chars)")

            # Convert to bytes for analysis
            try:
                raw_bytes = bytes.fromhex(hex_str)
                interpretation_parts.append(
                    f"Hex data: {len(raw_bytes)} bytes"
                )
                # Check for known file signatures (magic bytes)
                sig = self._identify_magic_bytes(raw_bytes)
                if sig:
                    interpretation_parts.append(f"File signature: {sig}")
                    derivation.append(f"MAGIC_BYTES -> {sig}")

                # ASCII content check
                printable = sum(
                    1 for b in raw_bytes
                    if 32 <= b <= 126
                )
                ascii_pct = (printable / len(raw_bytes) * 100) if raw_bytes else 0
                interpretation_parts.append(
                    f"ASCII content: {ascii_pct:.0f}%"
                )
                if ascii_pct > 70:
                    text = raw_bytes.decode("ascii", errors="replace")[:100]
                    interpretation_parts.append(f"Text: {text}")
                    derivation.append("CONTENT_TYPE -> mostly ASCII text")
                else:
                    derivation.append("CONTENT_TYPE -> binary data")

            except ValueError:
                interpretation_parts.append("Invalid hex sequence")
                derivation.append("HEX_PARSE_FAIL -> invalid hex chars")

            perception = Perception(
                sense=Sense.BINARY,
                raw_input=cleaned[:100],
                interpretation="\n".join(interpretation_parts),
                confidence=0.85,
                derivation_path=derivation,
            )
            self._perception_history.append(perception)
            return perception

        # Detect binary string (0s and 1s)
        bin_match = re.match(r"^[01\s]+$", cleaned)
        if bin_match:
            bits = cleaned.replace(" ", "")
            derivation.append(f"DETECT_FORMAT -> binary ({len(bits)} bits)")
            n_bytes = len(bits) // 8
            interpretation_parts.append(f"Binary data: {len(bits)} bits ({n_bytes} bytes)")

            if len(bits) % 8 == 0 and n_bytes > 0:
                # Convert to bytes
                byte_vals = [
                    int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)
                ]
                raw_bytes = bytes(byte_vals)
                printable = sum(1 for b in raw_bytes if 32 <= b <= 126)
                ascii_pct = printable / n_bytes * 100
                if ascii_pct > 70:
                    text = raw_bytes.decode("ascii", errors="replace")[:100]
                    interpretation_parts.append(f"Decoded text: {text}")
                    derivation.append("BINARY_DECODE -> ASCII text")
                else:
                    hex_repr = raw_bytes.hex()[:40]
                    interpretation_parts.append(f"Hex: {hex_repr}")
                    derivation.append("BINARY_DECODE -> non-text data")
            else:
                interpretation_parts.append(
                    f"Not byte-aligned ({len(bits)} bits is not divisible by 8)"
                )
                derivation.append("ALIGNMENT -> not byte-aligned")

            perception = Perception(
                sense=Sense.BINARY,
                raw_input=cleaned[:100],
                interpretation="\n".join(interpretation_parts),
                confidence=0.80,
                derivation_path=derivation,
            )
            self._perception_history.append(perception)
            return perception

        # Detect base64
        b64_match = re.match(
            r"^[A-Za-z0-9+/]+=*$", cleaned.replace("\n", "")
        )
        if b64_match and len(cleaned) > 4:
            import base64
            derivation.append("DETECT_FORMAT -> possible base64")
            try:
                decoded = base64.b64decode(cleaned, validate=True)
                interpretation_parts.append(
                    f"Base64 data: decodes to {len(decoded)} bytes"
                )
                sig = self._identify_magic_bytes(decoded)
                if sig:
                    interpretation_parts.append(f"Decoded content: {sig}")
                    derivation.append(f"BASE64_CONTENT -> {sig}")
                else:
                    printable = sum(1 for b in decoded if 32 <= b <= 126)
                    ascii_pct = (printable / len(decoded) * 100) if decoded else 0
                    if ascii_pct > 70:
                        text = decoded.decode("ascii", errors="replace")[:100]
                        interpretation_parts.append(f"Decoded text: {text}")
                        derivation.append("BASE64_CONTENT -> text")
                    else:
                        derivation.append("BASE64_CONTENT -> binary")
            except Exception:
                interpretation_parts.append("Looks like base64 but failed to decode")
                derivation.append("BASE64_DECODE_FAIL")

            perception = Perception(
                sense=Sense.BINARY,
                raw_input=cleaned[:100],
                interpretation="\n".join(interpretation_parts),
                confidence=0.70,
                derivation_path=derivation,
            )
            self._perception_history.append(perception)
            return perception

        # Cannot structurally identify the binary format
        derivation.append("NO_FORMAT_MATCH -> cannot identify binary structure")
        perception = Perception(
            sense=Sense.BINARY,
            raw_input=cleaned[:100],
            interpretation=(
                "Cannot structurally identify this as binary data. "
                "Expected hex (0x...), binary (01...), or base64."
            ),
            confidence=0.2,
            derivation_path=derivation,
        )
        self._perception_history.append(perception)
        return perception

    def read_error(self, traceback: str) -> Perception:
        """Analyse an error trace through pattern matching.

        Args:
            traceback: Raw error/traceback text.

        Returns:
            Perception with root cause analysis and fix suggestions.
        """
        parsed = self._error_reader.parse_traceback(traceback)
        root_cause = self._error_reader.identify_root_cause(traceback)
        fixes = self._error_reader.suggest_fix(traceback)

        interpretation_parts = [root_cause, ""]
        if fixes:
            interpretation_parts.append("Suggested fixes:")
            for i, fix in enumerate(fixes, 1):
                interpretation_parts.append(f"  {i}. {fix}")

        # Confidence based on how much structure we found
        n_frames = len(parsed.get("frames", []))
        has_type = parsed["exception_type"] != "Unknown"
        confidence = 0.3
        if has_type:
            confidence += 0.3
        if n_frames > 0:
            confidence += 0.2
        if fixes and "No known pattern" not in fixes[0]:
            confidence += 0.1

        derivation = parsed.get("derivation_path", [])
        derivation.append(f"ROOT_CAUSE -> {parsed['exception_type']}")
        if fixes and "No known pattern" not in fixes[0]:
            derivation.append(f"FIX_PATTERN -> matched known error pattern")

        perception = Perception(
            sense=Sense.ERROR,
            raw_input=traceback[:200] + ("..." if len(traceback) > 200 else ""),
            interpretation="\n".join(interpretation_parts),
            confidence=min(confidence, 1.0),
            derivation_path=derivation,
        )
        self._perception_history.append(perception)
        return perception

    def read_state(self) -> dict[str, Any]:
        """Introspect own runtime state -- the strange loop.

        Returns:
            Dict with health, resources, module info, perception
            history summary.
        """
        state: dict[str, Any] = {
            "timestamp": time.time(),
            "health": self._state_monitor.health_check(),
            "resources": self._state_monitor.resource_usage(),
            "perception_count": len(self._perception_history),
            "senses_used": {},
        }

        # Summarise which senses have been used
        for p in self._perception_history:
            name = p.sense.name
            state["senses_used"][name] = state["senses_used"].get(name, 0) + 1

        return state

    def diagnose(self, symptom: str) -> list[Perception]:
        """Given a symptom, reason about likely causes.

        Uses grammar-driven analysis, NOT guesswork.  Matches the
        symptom against known structural patterns and checks relevant
        subsystems.

        Args:
            symptom: Description of the problem (e.g. "app crashes
                on start", "slow response", "import fails").

        Returns:
            List of perceptions -- potential causes with derivation
            paths showing the reasoning.
        """
        perceptions: list[Perception] = []
        symptom_lower = symptom.lower()

        # -- Symptom pattern matching (structural, not guessing) -----------

        # Import / module problems
        if any(kw in symptom_lower for kw in (
            "import", "module", "not found", "cannot import",
        )):
            perceptions.append(Perception(
                sense=Sense.CODE,
                raw_input=symptom,
                interpretation=(
                    "Import failure pattern detected. Likely causes:\n"
                    "1. Module not installed in current environment\n"
                    "2. Circular import between modules\n"
                    "3. Module path not on sys.path\n"
                    "4. Module exists but has its own import error "
                    "(cascade failure)"
                ),
                confidence=0.75,
                derivation_path=[
                    f"SYMPTOM_MATCH -> import/module keywords in '{symptom[:50]}'",
                    "PATTERN -> import_failure",
                    "CHECK -> sys.path, virtualenv, circular deps",
                ],
            ))
            # Also check actual sys.path
            perceptions.append(Perception(
                sense=Sense.SELF,
                raw_input="sys.path inspection",
                interpretation=f"sys.path has {len(sys.path)} entries. "
                               f"CWD: {os.getcwd()}",
                confidence=0.90,
                derivation_path=[
                    "SELF_INSPECT -> sys.path",
                    f"PATH_COUNT -> {len(sys.path)}",
                ],
            ))

        # Crash / startup problems
        if any(kw in symptom_lower for kw in (
            "crash", "startup", "start", "launch", "won't run",
            "fails to start",
        )):
            # Check health
            health = self._state_monitor.health_check()
            degraded_systems = [
                name for name, sub in health["subsystems"].items()
                if sub["status"] != "healthy"
            ]
            if degraded_systems:
                interp = (
                    f"Degraded subsystems detected: "
                    f"{', '.join(degraded_systems)}. "
                    f"This may contribute to startup failure."
                )
            else:
                interp = (
                    "All subsystems report healthy. Crash may be in "
                    "application code rather than environment. Check:\n"
                    "1. __init__.py for eager imports that cascade-fail\n"
                    "2. Missing config files or directories\n"
                    "3. Permission issues on data directories\n"
                    "4. Android-specific path issues (Path.home() on Android "
                    "returns /data which is root-only)"
                )
            perceptions.append(Perception(
                sense=Sense.SELF,
                raw_input=symptom,
                interpretation=interp,
                confidence=0.70,
                derivation_path=[
                    f"SYMPTOM_MATCH -> crash/startup keywords",
                    f"HEALTH_CHECK -> overall={health['overall']}",
                    f"DEGRADED -> {degraded_systems or 'none'}",
                ],
            ))

        # Performance problems
        if any(kw in symptom_lower for kw in (
            "slow", "performance", "lag", "freeze", "hang", "timeout",
        )):
            resources = self._state_monitor.resource_usage()
            mem_mb = resources.get("memory_mb", -1)
            perceptions.append(Perception(
                sense=Sense.SELF,
                raw_input=symptom,
                interpretation=(
                    f"Performance issue analysis:\n"
                    f"Memory usage: {mem_mb:.1f}MB\n"
                    f"GC objects tracked: "
                    f"{len(gc.get_objects())}\n"
                    f"Loaded modules: "
                    f"{resources.get('loaded_modules', {}).get('total', '?')}\n"
                    f"\n"
                    f"Check for:\n"
                    f"1. O(n^2) or worse algorithms in hot paths\n"
                    f"2. Unbounded caches or lists\n"
                    f"3. Synchronous I/O blocking the event loop\n"
                    f"4. GC pressure from excessive object creation"
                ),
                confidence=0.65,
                derivation_path=[
                    f"SYMPTOM_MATCH -> performance keywords",
                    f"RESOURCE_CHECK -> memory={mem_mb:.1f}MB",
                    f"GC_CHECK -> {len(gc.get_objects())} tracked objects",
                ],
            ))

        # Network problems
        if any(kw in symptom_lower for kw in (
            "network", "connection", "api", "http", "timeout",
            "refused", "dns", "offline",
        )):
            net_health = self._state_monitor._check_network()
            perceptions.append(Perception(
                sense=Sense.NETWORK,
                raw_input=symptom,
                interpretation=(
                    f"Network state: {net_health['status']}\n"
                    f"DNS: {net_health.get('dns', 'unknown')}\n"
                    f"\n"
                    f"If offline, check:\n"
                    f"1. WiFi/data connection\n"
                    f"2. VPN or proxy settings\n"
                    f"3. DNS resolver configuration\n"
                    f"If connected but failing, check:\n"
                    f"1. API endpoint URL and port\n"
                    f"2. Authentication credentials/tokens\n"
                    f"3. Rate limiting or quota exceeded"
                ),
                confidence=0.70,
                derivation_path=[
                    f"SYMPTOM_MATCH -> network keywords",
                    f"NET_CHECK -> {net_health['status']}",
                    f"DNS -> {net_health.get('dns', 'unknown')}",
                ],
            ))

        # File / permission problems
        if any(kw in symptom_lower for kw in (
            "file", "permission", "not found", "path", "directory",
            "disk", "space", "write",
        )):
            disk_health = self._state_monitor._check_disk()
            perceptions.append(Perception(
                sense=Sense.FILESYSTEM,
                raw_input=symptom,
                interpretation=(
                    f"Filesystem state: {disk_health['status']}\n"
                    f"Free space: {disk_health.get('free_gb', '?')}GB\n"
                    f"Data dir: {mkangel_dir()}\n"
                    f"Data dir exists: {mkangel_dir().exists()}\n"
                    f"\n"
                    f"Common causes:\n"
                    f"1. Path uses home dir which is /data on Android "
                    f"(use mkangel_dir() instead)\n"
                    f"2. Directory not created (missing parents=True)\n"
                    f"3. File locked by another process\n"
                    f"4. Disk full"
                ),
                confidence=0.70,
                derivation_path=[
                    f"SYMPTOM_MATCH -> filesystem keywords",
                    f"DISK_CHECK -> {disk_health['status']}",
                    f"FREE_SPACE -> {disk_health.get('free_gb', '?')}GB",
                ],
            ))

        # If no symptom pattern matched, be honest about it
        if not perceptions:
            perceptions.append(Perception(
                sense=Sense.GRAMMAR,
                raw_input=symptom,
                interpretation=(
                    f"Cannot structurally match symptom '{symptom}' to "
                    f"a known pattern. The Angel does not guess.\n"
                    f"\n"
                    f"To help diagnose, provide:\n"
                    f"1. The actual error message or traceback\n"
                    f"2. The code that triggers the issue\n"
                    f"3. Steps to reproduce\n"
                    f"\n"
                    f"The Angel reads structure, not minds."
                ),
                confidence=0.3,
                derivation_path=[
                    f"SYMPTOM_MATCH -> no pattern matched for '{symptom[:50]}'",
                    "HONEST_ANSWER -> cannot diagnose without structural data",
                ],
            ))

        return perceptions

    def explain_reasoning(self, perception: Perception) -> str:
        """Produce human-readable explanation of HOW a conclusion was reached.

        Args:
            perception: The perception to explain.

        Returns:
            Step-by-step explanation of the derivation.
        """
        parts = []
        parts.append(f"Sense: {perception.sense.name}")
        parts.append(f"Confidence: {perception.confidence:.0%}")
        parts.append("")
        parts.append("Derivation path (how I reached this conclusion):")
        for i, step in enumerate(perception.derivation_path, 1):
            parts.append(f"  {i}. {step}")
        parts.append("")
        parts.append("Interpretation:")
        for line in perception.interpretation.splitlines():
            parts.append(f"  {line}")
        parts.append("")
        parts.append(
            "Note: Every step above is a structural derivation, "
            "not a guess.  If any step says 'NO_STRUCTURE_FOUND' or "
            "'FALLBACK', the Angel is being honest about the limits "
            "of what she can determine."
        )
        return "\n".join(parts)

    # -- Sense auto-detection ----------------------------------------------

    def _detect_sense(self, data: str) -> tuple[Sense, list[str]]:
        """Detect which sense to use by examining the data's structure."""
        derivation = ["AUTO_DETECT -> examining input structure"]

        # Check for traceback patterns first (high priority)
        if any(marker in data for marker in (
            "Traceback (most recent call last)",
            "Error:", "Exception:",
            "\tat ",  # Java stack trace
        )):
            derivation.append("MATCH -> error/traceback markers found")
            return Sense.ERROR, derivation

        # Check for code patterns
        code_indicators = 0
        if re.search(r"^\s*(def|class|import|from|if|for|while)\s", data, re.MULTILINE):
            code_indicators += 2
        if re.search(r"[{};]\s*$", data, re.MULTILINE):
            code_indicators += 1
        if re.search(r"^\s*(function|const|let|var|pub|fn|func)\s", data, re.MULTILINE):
            code_indicators += 2
        if code_indicators >= 2:
            derivation.append(f"MATCH -> code indicators (score={code_indicators})")
            return Sense.CODE, derivation

        # Check for binary/hex
        cleaned = data.strip()
        if re.match(r"^(?:0x)?[0-9a-fA-F\s]+$", cleaned) and len(cleaned) > 4:
            derivation.append("MATCH -> hex data pattern")
            return Sense.BINARY, derivation
        if re.match(r"^[01\s]+$", cleaned) and len(cleaned) > 8:
            derivation.append("MATCH -> binary data pattern")
            return Sense.BINARY, derivation

        # Check for filesystem paths
        if re.search(r"[/\\][\w.-]+[/\\]", data) or re.search(
            r"^[A-Z]:\\", data
        ):
            derivation.append("MATCH -> filesystem path pattern")
            return Sense.FILESYSTEM, derivation

        # Check for network patterns
        if re.search(r"https?://|[\w.-]+:\d+|dns|socket|connect", data, re.IGNORECASE):
            derivation.append("MATCH -> network pattern")
            return Sense.NETWORK, derivation

        # Check for temporal patterns
        if re.search(
            r"\d{4}-\d{2}-\d{2}|\d+:\d+:\d+|timestamp|epoch|duration",
            data, re.IGNORECASE,
        ):
            derivation.append("MATCH -> temporal pattern")
            return Sense.TEMPORAL, derivation

        # Default to grammar (general structural analysis)
        derivation.append("DEFAULT -> no specific pattern matched, using GRAMMAR")
        return Sense.GRAMMAR, derivation

    # -- Specialised perception methods ------------------------------------

    def _perceive_filesystem(self, data: str) -> Perception:
        """Perceive filesystem-related data."""
        derivation = ["FILESYSTEM_SENSE -> analysing path/file data"]
        interpretation_parts = []

        # Extract paths from the data
        paths_found = re.findall(
            r"(?:[A-Z]:\\|/)[^\s:*?\"<>|]+", data
        )
        if paths_found:
            for p_str in paths_found[:5]:
                p = Path(p_str)
                try:
                    exists = p.exists()
                    interpretation_parts.append(
                        f"Path: {p_str} -> {'exists' if exists else 'NOT FOUND'}"
                    )
                    if exists and p.is_file():
                        stat = p.stat()
                        size_kb = stat.st_size / 1024
                        interpretation_parts.append(
                            f"  Size: {size_kb:.1f}KB, "
                            f"Modified: {time.ctime(stat.st_mtime)}"
                        )
                    elif exists and p.is_dir():
                        try:
                            n_items = len(list(p.iterdir()))
                        except PermissionError:
                            n_items = -1
                        interpretation_parts.append(
                            f"  Directory with "
                            f"{'? (permission denied)' if n_items < 0 else str(n_items)} items"
                        )
                    derivation.append(
                        f"PATH_CHECK -> {p_str} "
                        f"{'exists' if exists else 'missing'}"
                    )
                except (OSError, ValueError) as e:
                    interpretation_parts.append(f"Path: {p_str} -> error: {e}")
                    derivation.append(f"PATH_ERROR -> {p_str}: {e}")
        else:
            interpretation_parts.append(
                "No filesystem paths found in input."
            )
            derivation.append("NO_PATHS -> no recognisable paths in data")

        return Perception(
            sense=Sense.FILESYSTEM,
            raw_input=data[:200],
            interpretation="\n".join(interpretation_parts),
            confidence=0.75 if paths_found else 0.3,
            derivation_path=derivation,
        )

    def _perceive_network(self, data: str) -> Perception:
        """Perceive network-related data."""
        derivation = ["NETWORK_SENSE -> analysing network data"]
        interpretation_parts = []

        # Extract URLs
        urls = re.findall(r"https?://[^\s<>\"']+", data)
        if urls:
            for url in urls[:3]:
                interpretation_parts.append(f"URL: {url}")
            derivation.append(f"URLS_FOUND -> {len(urls)}")

        # Extract host:port patterns
        hostports = re.findall(r"([\w.-]+):(\d+)", data)
        if hostports:
            for host, port in hostports[:3]:
                interpretation_parts.append(f"Endpoint: {host}:{port}")
            derivation.append(f"ENDPOINTS_FOUND -> {len(hostports)}")

        # Current network state
        net = self._state_monitor._check_network()
        interpretation_parts.append(
            f"Current network: {net['status']} "
            f"(DNS: {net.get('dns', 'unknown')})"
        )
        derivation.append(f"NET_STATE -> {net['status']}")

        return Perception(
            sense=Sense.NETWORK,
            raw_input=data[:200],
            interpretation="\n".join(interpretation_parts),
            confidence=0.70,
            derivation_path=derivation,
        )

    def _perceive_temporal(self, data: str) -> Perception:
        """Perceive time/sequence patterns."""
        derivation = ["TEMPORAL_SENSE -> analysing temporal data"]
        interpretation_parts = []

        # ISO dates
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", data)
        if dates:
            interpretation_parts.append(f"Dates found: {', '.join(dates[:5])}")
            derivation.append(f"DATES -> {len(dates)} found")

        # Timestamps (epoch)
        epochs = re.findall(r"\b(\d{10,13})\b", data)
        if epochs:
            for ep in epochs[:3]:
                ts = float(ep)
                if ts > 1e12:
                    ts /= 1000  # milliseconds
                try:
                    human = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(ts)
                    )
                    interpretation_parts.append(f"Epoch {ep} -> {human}")
                except (ValueError, OSError):
                    interpretation_parts.append(f"Epoch {ep} -> out of range")
            derivation.append(f"EPOCHS -> {len(epochs)} found")

        # Durations
        durations = re.findall(r"(\d+)\s*(ms|s|sec|min|hour|h|m)\b", data, re.IGNORECASE)
        if durations:
            for val, unit in durations[:3]:
                interpretation_parts.append(f"Duration: {val}{unit}")
            derivation.append(f"DURATIONS -> {len(durations)} found")

        # Current time context
        interpretation_parts.append(
            f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if not dates and not epochs and not durations:
            interpretation_parts.append(
                "No structured temporal data found."
            )
            derivation.append("NO_TEMPORAL_DATA -> no patterns matched")

        return Perception(
            sense=Sense.TEMPORAL,
            raw_input=data[:200],
            interpretation="\n".join(interpretation_parts),
            confidence=0.65 if (dates or epochs or durations) else 0.3,
            derivation_path=derivation,
        )

    def _perceive_grammar(self, data: str) -> Perception:
        """Perceive through general grammar analysis."""
        derivation = ["GRAMMAR_SENSE -> general structural analysis"]
        interpretation_parts = []

        # Try to engage the GLM for grammar analysis
        try:
            from glm.angel import Angel
            angel = Angel()
            angel.awaken()
            tokens = data.lower().split()[:20]
            predictions = angel.predict(tokens, horizon=5)
            if predictions:
                interpretation_parts.append("Grammar analysis (GLM):")
                for pred in predictions[:5]:
                    rule = pred.get("rule", "?")
                    output = pred.get("predicted", "?")
                    conf = pred.get("confidence", 0.0)
                    interpretation_parts.append(
                        f"  {rule} -> {output} (conf={conf:.2f})"
                    )
                derivation.append(
                    f"GLM_PREDICT -> {len(predictions)} predictions"
                )
            else:
                interpretation_parts.append(
                    "GLM produced no predictions for this input."
                )
                derivation.append("GLM_PREDICT -> empty")
        except Exception as e:
            interpretation_parts.append(
                f"GLM not available: {e}. Falling back to structural "
                f"text analysis."
            )
            derivation.append(f"GLM_UNAVAILABLE -> {type(e).__name__}")

        # Basic structural text analysis
        words = data.split()
        sentences = re.split(r"[.!?]+", data)
        interpretation_parts.append(
            f"Text structure: {len(words)} words, "
            f"{len(sentences)} sentences"
        )

        return Perception(
            sense=Sense.GRAMMAR,
            raw_input=data[:200],
            interpretation="\n".join(interpretation_parts),
            confidence=0.50,
            derivation_path=derivation,
        )

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _identify_magic_bytes(data: bytes) -> str:
        """Identify file type from magic bytes -- structural signature."""
        if len(data) < 2:
            return ""

        signatures = {
            b"\x89PNG": "PNG image",
            b"\xff\xd8\xff": "JPEG image",
            b"GIF8": "GIF image",
            b"PK\x03\x04": "ZIP archive (or JAR/APK/DOCX)",
            b"\x50\x4b\x05\x06": "ZIP archive (empty)",
            b"\x7fELF": "ELF executable (Linux/Android)",
            b"MZ": "PE executable (Windows)",
            b"\xca\xfe\xba\xbe": "Java class file / Mach-O fat binary",
            b"\xfe\xed\xfa": "Mach-O executable",
            b"%PDF": "PDF document",
            b"{\n": "JSON data (probable)",
            b"<?xml": "XML document",
            b"<!DOCT": "HTML document",
            b"<html": "HTML document",
            b"\x1f\x8b": "Gzip compressed",
            b"BZ": "Bzip2 compressed",
            b"\xfd7zXZ": "XZ compressed",
            b"Rar!": "RAR archive",
            b"SQLite": "SQLite database",
            b"dex\n": "Android DEX bytecode",
        }

        for magic, description in signatures.items():
            if data[:len(magic)] == magic:
                return description

        return ""

    @staticmethod
    def _format_state(state: dict[str, Any]) -> str:
        """Format state dict into readable text."""
        parts = []
        health = state.get("health", {})
        overall = health.get("overall", "unknown")
        parts.append(f"Overall health: {overall}")

        subsystems = health.get("subsystems", {})
        for name, sub in subsystems.items():
            status = sub.get("status", "unknown")
            note = sub.get("note", "")
            line = f"  {name}: {status}"
            if note:
                line += f" -- {note}"
            parts.append(line)

        parts.append(f"Perceptions recorded: {state.get('perception_count', 0)}")
        senses_used = state.get("senses_used", {})
        if senses_used:
            parts.append(
                f"Senses used: "
                f"{', '.join(f'{k}({v})' for k, v in senses_used.items())}"
            )

        return "\n".join(parts)
