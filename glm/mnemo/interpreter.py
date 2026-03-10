"""
interpreter.py — MNEMO interpreter.

The ``MnemoInterpreter`` executes MNEMO programs.  A MNEMO sequence is a
program that drives the grammar engine: each token is an instruction that
selects a domain, invokes an operation, and applies a modifier.

MNEMO programs are executed left-to-right.  Each token updates the
interpreter's state — the current domain, the active grammar rules,
and the results stack.  Tokens can be chained:

    ``"Lp+ Xd~ Bt-"``

means: linguistic predict forward, then computational derive
bidirectional, then biological translate backward.  The output of each
step feeds into the next, creating a pipeline of grammar transformations.

The interpreter maintains a full execution trace so that programs can be
debugged, replayed, and introspected (especially via M-domain meta-ops).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import rules as _rules
from .language import MnemoGrammar, MnemoToken, expand as _expand


# ---------------------------------------------------------------------------
# Execution state
# ---------------------------------------------------------------------------

class ExecutionStatus(Enum):
    """Status of the interpreter after executing a program or step."""
    OK = "ok"
    ERROR = "error"
    HALTED = "halted"
    YIELDED = "yielded"


@dataclass
class ExecutionState:
    """The interpreter's mutable state during MNEMO program execution.

    Attributes:
        current_domain:  The active domain (set by the most recent token).
        active_grammars: Names of grammar rule sets currently loaded.
        results_stack:   Stack of results from executed operations.
                         Each entry is a dict with the operation result
                         and metadata.
        variables:       Named values accessible during execution.
        step_count:      Number of steps executed so far.
        status:          Current execution status.
        error:           Error message if status is ERROR.
        trace:           Full execution trace — one entry per step.
    """
    current_domain: str = "universal"
    active_grammars: List[str] = field(default_factory=list)
    results_stack: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    status: ExecutionStatus = ExecutionStatus.OK
    error: str = ""
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def push(self, result: Dict[str, Any]) -> None:
        """Push a result onto the results stack."""
        self.results_stack.append(result)

    def pop(self) -> Optional[Dict[str, Any]]:
        """Pop the top result from the stack, or None if empty."""
        if self.results_stack:
            return self.results_stack.pop()
        return None

    def peek(self) -> Optional[Dict[str, Any]]:
        """Peek at the top result without removing it."""
        if self.results_stack:
            return self.results_stack[-1]
        return None

    @property
    def last_result(self) -> Any:
        """The value of the most recent result, or None."""
        top = self.peek()
        return top.get("value") if top else None

    def clone(self) -> "ExecutionState":
        """Create a deep-ish copy of this state (for branching execution)."""
        return ExecutionState(
            current_domain=self.current_domain,
            active_grammars=list(self.active_grammars),
            results_stack=[dict(r) for r in self.results_stack],
            variables=dict(self.variables),
            step_count=self.step_count,
            status=self.status,
            error=self.error,
            trace=[dict(t) for t in self.trace],
        )


# ---------------------------------------------------------------------------
# Step result — what a single instruction produces
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """The result of executing a single MNEMO instruction.

    Attributes:
        token:     The token that was executed.
        value:     The computed value (may be None for side-effect ops).
        state:     The state after execution.
        duration:  Wall-clock time for this step (seconds).
        actions:   Grammar actions that were invoked.
    """
    token: MnemoToken
    value: Any
    state: ExecutionState
    duration: float = 0.0
    actions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MnemoInterpreter
# ---------------------------------------------------------------------------

class MnemoInterpreter:
    """Executes MNEMO programs against the grammar engine.

    The interpreter is a stack machine:
    - Each MNEMO token is an instruction.
    - Instructions consume input from the context and/or the results stack.
    - Instructions push their output onto the results stack.
    - The interpreter tracks domain transitions, enabling cross-domain
      pipelines.

    Usage::

        interp = MnemoInterpreter()
        result = interp.execute("Lp+ Xd~ Bt-", context={"input": "hello"})
        print(result["results"])

    Attributes:
        grammar:   The MNEMO grammar for parsing tokens.
        handlers:  Registry of operation handlers.  Each handler is a
                   callable ``(token, state, context) -> value``.
        history:   Execution history across all ``execute()`` calls.
    """

    def __init__(self, grammar: Optional[MnemoGrammar] = None) -> None:
        self.grammar = grammar or MnemoGrammar()
        self.handlers: Dict[str, Callable[..., Any]] = {}
        self.history: List[Dict[str, Any]] = []
        self._register_default_handlers()

    # -- public API ---------------------------------------------------------

    def execute(
        self,
        mnemo_program: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        state: Optional[ExecutionState] = None,
        max_steps: int = 1000,
    ) -> Dict[str, Any]:
        """Parse and execute a MNEMO sequence.

        Parameters:
            mnemo_program: A space-separated MNEMO string.
            context:       External data available to the program.
                           Common keys: ``input``, ``grammar``, ``corpus``.
            state:         Optional initial state (for resuming execution).
            max_steps:     Safety limit on execution steps.

        Returns:
            A dict with:
                status:       Final execution status.
                results:      The results stack (bottom to top).
                final_value:  The top of the results stack (the "return value").
                state:        The final execution state.
                trace:        Full step-by-step execution trace.
                token_count:  Number of tokens executed.
        """
        context = context or {}
        state = state or ExecutionState()

        # Validate the program.
        valid, errors = _rules.validate(mnemo_program)
        if not valid:
            state.status = ExecutionStatus.ERROR
            state.error = f"Invalid MNEMO program: {'; '.join(errors)}"
            return self._make_result(state, 0)

        # Parse into tokens.
        tokens = self.grammar.tokenize(mnemo_program)

        # Execute each token.
        for i, token in enumerate(tokens):
            if state.step_count >= max_steps:
                state.status = ExecutionStatus.HALTED
                state.error = f"Exceeded max_steps ({max_steps})"
                break

            if state.status == ExecutionStatus.ERROR:
                break

            step_result = self.step(token, state, context)
            state = step_result.state

        # Build the final result.
        result = self._make_result(state, len(tokens))

        # Record in history.
        self.history.append({
            "program": mnemo_program,
            "token_count": len(tokens),
            "status": state.status.value,
            "final_value": state.last_result,
            "timestamp": time.time(),
        })

        return result

    def step(
        self,
        mnemo_token: MnemoToken,
        state: ExecutionState,
        context: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        """Execute a single MNEMO instruction.

        Parameters:
            mnemo_token: The token to execute.
            state:       The current execution state (will be mutated).
            context:     External data available to the instruction.

        Returns:
            A ``StepResult`` with the computed value and updated state.
        """
        context = context or {}
        start_time = time.time()

        # Update domain.
        domain_code = _rules.DOMAIN_CODES.get(mnemo_token.domain_char)
        if domain_code:
            state.current_domain = domain_code.name
            if domain_code.name not in state.active_grammars:
                state.active_grammars.append(domain_code.name)

        # Expand the token to get grammar actions.
        token_expansion = self.grammar.expand_token(mnemo_token)
        actions = token_expansion.get("grammar_actions", [])

        # Execute the operation via handler dispatch.
        value = self._dispatch(mnemo_token, state, context)

        # Push result onto stack.
        result_entry = {
            "token": mnemo_token.raw,
            "value": value,
            "domain": state.current_domain,
            "operation": mnemo_token.operation,
            "modifier": mnemo_token.modifier,
            "step": state.step_count,
        }
        state.push(result_entry)

        # Record in trace.
        duration = time.time() - start_time
        state.trace.append({
            "step": state.step_count,
            "token": mnemo_token.raw,
            "domain": state.current_domain,
            "value": _safe_repr(value),
            "stack_depth": len(state.results_stack),
            "duration": duration,
        })

        state.step_count += 1

        return StepResult(
            token=mnemo_token,
            value=value,
            state=state,
            duration=duration,
            actions=actions,
        )

    # -- handler registration -----------------------------------------------

    def register_handler(
        self,
        operation: str,
        handler: Callable[..., Any],
    ) -> None:
        """Register a custom handler for an operation.

        Parameters:
            operation: The operation name (e.g. ``"predict"``, ``"analyze"``).
            handler:   A callable ``(token, state, context) -> value``.
        """
        self.handlers[operation] = handler

    # -- internal: dispatch -------------------------------------------------

    def _dispatch(
        self,
        token: MnemoToken,
        state: ExecutionState,
        context: Dict[str, Any],
    ) -> Any:
        """Dispatch a token to the appropriate handler."""
        # Check for a custom handler first.
        if token.operation in self.handlers:
            try:
                return self.handlers[token.operation](token, state, context)
            except Exception as exc:
                state.status = ExecutionStatus.ERROR
                state.error = f"Handler error for {token.raw}: {exc}"
                return None

        # Fall back to built-in handlers.
        builtin = self._builtins.get(token.operation)
        if builtin:
            try:
                return builtin(self, token, state, context)
            except Exception as exc:
                state.status = ExecutionStatus.ERROR
                state.error = f"Builtin error for {token.raw}: {exc}"
                return None

        # Unknown operation — return a descriptor.
        return {
            "instruction": token.raw,
            "description": token.description,
            "note": "no handler registered; returning descriptor",
        }

    # -- internal: built-in handlers ----------------------------------------

    def _handle_predict(
        self, token: MnemoToken, state: ExecutionState, context: Dict[str, Any]
    ) -> Any:
        """Handle predict (p) operations.

        Uses the results stack or context input as the source form,
        then applies forward derivation in the current domain.
        """
        source = self._resolve_input(state, context)
        direction = "forward" if token.modifier != "backward" else "backward"

        return {
            "operation": "predict",
            "domain": state.current_domain,
            "direction": direction,
            "source": _safe_repr(source),
            "prediction": self._simulate_derivation(source, state.current_domain, direction),
        }

    def _handle_reconstruct(
        self, token: MnemoToken, state: ExecutionState, context: Dict[str, Any]
    ) -> Any:
        """Handle reconstruct (r) operations."""
        source = self._resolve_input(state, context)

        return {
            "operation": "reconstruct",
            "domain": state.current_domain,
            "source": _safe_repr(source),
            "reconstruction": self._simulate_derivation(
                source, state.current_domain, "backward"
            ),
        }

    def _handle_forecast(
        self, token: MnemoToken, state: ExecutionState, context: Dict[str, Any]
    ) -> Any:
        """Handle forecast (f) operations."""
        source = self._resolve_input(state, context)
        horizon = context.get("horizon", 5)

        steps: List[Dict[str, Any]] = []
        current = source
        for i in range(1, horizon + 1):
            derived = self._simulate_derivation(current, state.current_domain, "forward")
            steps.append({"step": i, "form": derived, "confidence": max(0.1, 1.0 - i * 0.15)})
            current = derived

        return {
            "operation": "forecast",
            "domain": state.current_domain,
            "source": _safe_repr(source),
            "horizon": horizon,
            "steps": steps,
        }

    def _handle_translate(
        self, token: MnemoToken, state: ExecutionState, context: Dict[str, Any]
    ) -> Any:
        """Handle translate (t) operations."""
        source = self._resolve_input(state, context)
        target_domain = context.get("target_domain", state.current_domain)

        return {
            "operation": "translate",
            "source_domain": state.current_domain,
            "target_domain": target_domain,
            "source": _safe_repr(source),
            "translation": {
                "from": state.current_domain,
                "to": target_domain,
                "form": source,
                "note": "cross-domain grammar isomorphism mapping",
            },
        }

    def _handle_introspect(
        self, token: MnemoToken, state: ExecutionState, context: Dict[str, Any]
    ) -> Any:
        """Handle introspect (i) operations.

        When the domain is meta (M), this introspects on MNEMO itself.
        """
        if token.is_meta:
            return {
                "operation": "meta_introspect",
                "vocabulary_size": len(_rules.MNEMO_VOCABULARY),
                "domains": list(_rules.DOMAIN_CODES.keys()),
                "operations": list(_rules.OPERATION_CODES.keys()),
                "modifiers": list(_rules.MODIFIER_CODES.keys()),
                "compound_count": len(_rules.COMPOUND_OPERATIONS),
                "meta_op_count": len(_rules.META_OPERATIONS),
                "state_summary": {
                    "current_domain": state.current_domain,
                    "stack_depth": len(state.results_stack),
                    "step_count": state.step_count,
                    "active_grammars": state.active_grammars,
                },
            }

        return {
            "operation": "introspect",
            "domain": state.current_domain,
            "active_grammars": list(state.active_grammars),
            "stack_depth": len(state.results_stack),
            "variables": list(state.variables.keys()),
        }

    def _handle_derive(
        self, token: MnemoToken, state: ExecutionState, context: Dict[str, Any]
    ) -> Any:
        """Handle derive (d) operations."""
        source = self._resolve_input(state, context)
        direction = _modifier_to_direction(token.modifier)

        results: List[Any] = []
        if direction in ("forward", "bidirectional"):
            results.append(self._simulate_derivation(source, state.current_domain, "forward"))
        if direction in ("backward", "bidirectional"):
            results.append(self._simulate_derivation(source, state.current_domain, "backward"))

        return {
            "operation": "derive",
            "domain": state.current_domain,
            "direction": direction,
            "source": _safe_repr(source),
            "derivations": results,
        }

    def _handle_compose(
        self, token: MnemoToken, state: ExecutionState, context: Dict[str, Any]
    ) -> Any:
        """Handle compose (c) operations — fugue composition."""
        grammars = list(state.active_grammars)

        return {
            "operation": "compose",
            "grammars": grammars,
            "mode": "parallel" if token.modifier == "parallel" else "sequential",
            "note": "fugue composition of active grammars",
            "voice_count": len(grammars),
        }

    def _handle_search(
        self, token: MnemoToken, state: ExecutionState, context: Dict[str, Any]
    ) -> Any:
        """Handle search (s) operations."""
        query = self._resolve_input(state, context)
        is_query_mode = token.modifier == "query"
        is_count_mode = token.modifier == "count"

        if is_query_mode:
            # Return matching tokens from the vocabulary.
            domain_filter = None if token.is_universal else token.domain_char
            matches = []
            for tok_str, entry in _rules.MNEMO_VOCABULARY.items():
                if domain_filter and not tok_str.startswith(domain_filter):
                    continue
                matches.append({"token": tok_str, "description": entry["description"]})
            return {
                "operation": "search_query",
                "domain": state.current_domain,
                "matches": matches[:20],  # Cap output.
                "total": len(matches),
            }

        if is_count_mode:
            domain_filter = None if token.is_universal else token.domain_char
            count = sum(
                1 for t in _rules.MNEMO_VOCABULARY
                if not domain_filter or t.startswith(domain_filter)
            )
            return {
                "operation": "search_count",
                "domain": state.current_domain,
                "count": count,
            }

        return {
            "operation": "search",
            "domain": state.current_domain,
            "query": _safe_repr(query),
            "note": "grammar space pattern search",
        }

    def _handle_analyze(
        self, token: MnemoToken, state: ExecutionState, context: Dict[str, Any]
    ) -> Any:
        """Handle analyze (a) operations."""
        source = self._resolve_input(state, context)

        analysis: Dict[str, Any] = {
            "operation": "analyze",
            "domain": state.current_domain,
            "source": _safe_repr(source),
        }

        # Structural analysis of the source.
        if isinstance(source, str):
            analysis["structure"] = {
                "type": "text",
                "length": len(source),
                "words": len(source.split()),
            }
        elif isinstance(source, dict):
            analysis["structure"] = {
                "type": "mapping",
                "keys": list(source.keys())[:10],
                "depth": _dict_depth(source),
            }
        elif isinstance(source, (list, tuple)):
            analysis["structure"] = {
                "type": "sequence",
                "length": len(source),
            }
        else:
            analysis["structure"] = {
                "type": type(source).__name__,
            }

        return analysis

    # -- internal: handler registry -----------------------------------------

    # Map operation names to bound methods.  Built lazily on first use.
    _builtins: Dict[str, Callable[..., Any]] = {}

    def _register_default_handlers(self) -> None:
        """Register the built-in operation handlers."""
        # Use instance-level builtins dict to avoid class-level mutation issues.
        self._builtins = {
            "predict": MnemoInterpreter._handle_predict,
            "reconstruct": MnemoInterpreter._handle_reconstruct,
            "forecast": MnemoInterpreter._handle_forecast,
            "translate": MnemoInterpreter._handle_translate,
            "introspect": MnemoInterpreter._handle_introspect,
            "derive": MnemoInterpreter._handle_derive,
            "compose": MnemoInterpreter._handle_compose,
            "search": MnemoInterpreter._handle_search,
            "analyze": MnemoInterpreter._handle_analyze,
        }

    # -- internal: input resolution -----------------------------------------

    @staticmethod
    def _resolve_input(state: ExecutionState, context: Dict[str, Any]) -> Any:
        """Resolve the input for an operation.

        Priority:
        1. Top of the results stack (previous operation's output).
        2. ``context["input"]``.
        3. None.
        """
        top = state.peek()
        if top is not None:
            return top.get("value", top)
        return context.get("input")

    # -- internal: simulated derivation -------------------------------------

    @staticmethod
    def _simulate_derivation(source: Any, domain: str, direction: str) -> Any:
        """Simulate a grammar derivation step.

        In a full system this would invoke the ``DerivationEngine``.
        Here we produce a structured descriptor that the engine can
        later resolve against actual grammar rules.
        """
        return {
            "derived_from": _safe_repr(source),
            "domain": domain,
            "direction": direction,
            "rule": f"{domain}.{direction}",
            "status": "pending_engine_resolution",
        }

    # -- internal: result builder -------------------------------------------

    @staticmethod
    def _make_result(state: ExecutionState, token_count: int) -> Dict[str, Any]:
        """Build the final execution result dict."""
        return {
            "status": state.status.value,
            "results": state.results_stack,
            "final_value": state.last_result,
            "state": state,
            "trace": state.trace,
            "token_count": token_count,
            "step_count": state.step_count,
            "error": state.error if state.status == ExecutionStatus.ERROR else None,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _safe_repr(value: Any, max_len: int = 200) -> Any:
    """Return a safe, bounded representation of a value for tracing."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value[:max_len] + ("..." if len(value) > max_len else "")
    if isinstance(value, dict):
        return value  # Keep dicts as-is for structured results.
    try:
        s = repr(value)
        return s[:max_len] + ("..." if len(s) > max_len else "")
    except Exception:
        return "<unrepresentable>"


def _modifier_to_direction(modifier: str) -> str:
    """Convert a modifier name to a derivation direction."""
    return {
        "forward": "forward",
        "backward": "backward",
        "bidirectional": "bidirectional",
        "": "forward",  # default
    }.get(modifier, "forward")


def _dict_depth(d: Any, depth: int = 0) -> int:
    """Compute nesting depth of a dict."""
    if not isinstance(d, dict) or not d:
        return depth
    return max(_dict_depth(v, depth + 1) for v in d.values())
