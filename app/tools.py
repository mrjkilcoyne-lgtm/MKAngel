"""
Tool System — the Angel's hands.

Tools are the actions the Angel can take in the world.
Chat is her voice; tools are her hands.

Each tool is a simple callable with a standardized interface:
    tool(input: str, context: dict) -> ToolResult

Tools are discovered, registered, and invoked by the Router.
They can be chained — the output of one tool feeds the next.

Categories:
  - Code: execute, analyze, refactor, explain code
  - File: read, write, list, search files
  - Web: search, fetch, summarize web content
  - System: app settings, notifications, device control
  - Document: create, edit, export documents
  - Math: calculate, solve, graph equations
  - Language: translate, define, etymology
"""

from __future__ import annotations

import os
import json
import time
import subprocess
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable
from pathlib import Path


class ToolCategory(Enum):
    CODE = auto()
    FILE = auto()
    WEB = auto()
    SYSTEM = auto()
    DOCUMENT = auto()
    MATH = auto()
    LANGUAGE = auto()
    GRAMMAR = auto()  # GLM-specific tools


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: str
    data: dict[str, Any] = field(default_factory=dict)
    tool_name: str = ""
    duration: float = 0.0
    error: str = ""

    def __str__(self):
        if self.success:
            return self.output
        return f"[Tool Error: {self.tool_name}] {self.error}"


@dataclass
class ToolSpec:
    """Specification for a registered tool."""
    name: str
    description: str
    category: ToolCategory
    handler: Callable[[str, dict], ToolResult]
    keywords: list[str] = field(default_factory=list)
    requires_network: bool = False


class ToolRegistry:
    """Registry of available tools.

    Tools register themselves here. The Router queries
    the registry to find tools matching a user's intent.
    """

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def find_by_category(self, category: ToolCategory) -> list[ToolSpec]:
        return [t for t in self._tools.values() if t.category == category]

    def find_by_keyword(self, keyword: str) -> list[ToolSpec]:
        kw = keyword.lower()
        return [
            t for t in self._tools.values()
            if any(kw in k for k in t.keywords) or kw in t.name.lower()
        ]

    def all_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def describe_all(self) -> str:
        lines = ["Available tools:"]
        for cat in ToolCategory:
            tools = self.find_by_category(cat)
            if tools:
                lines.append(f"\n  [{cat.name}]")
                for t in tools:
                    lines.append(f"    {t.name}: {t.description}")
        return "\n".join(lines)


class ToolChain:
    """Chain multiple tools together — output of one feeds the next."""

    def __init__(self, tools: list[tuple[str, dict]]):
        self._steps = tools  # [(tool_name, extra_context), ...]

    def execute(self, initial_input: str, registry: ToolRegistry) -> ToolResult:
        current_input = initial_input
        results = []

        for tool_name, extra_ctx in self._steps:
            spec = registry.get(tool_name)
            if not spec:
                return ToolResult(
                    success=False, output="",
                    error=f"Tool '{tool_name}' not found",
                    tool_name=tool_name,
                )
            ctx = {"chain_results": results, **extra_ctx}
            result = spec.handler(current_input, ctx)
            results.append(result)
            if not result.success:
                return result
            current_input = result.output

        if results:
            final = results[-1]
            final.data["chain_length"] = len(results)
            return final
        return ToolResult(success=True, output=initial_input, tool_name="chain")


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

def _tool_code_explain(input_text: str, context: dict) -> ToolResult:
    """Explain code structure using GLM grammar analysis."""
    start = time.time()
    try:
        from glm.angel import Angel
        angel = Angel()
        angel.awaken()
        tokens = input_text.split()[:20]
        preds = angel.predict(tokens, domain="computational", horizon=5)
        explanation = []
        for p in preds[:3]:
            explanation.append(
                f"Grammar '{p.get('grammar', '?')}' identifies: "
                f"{p.get('predicted', '?')} "
                f"(confidence: {p.get('confidence', 0):.2f})"
            )
        output = "\n".join(explanation) if explanation else (
            "No computational grammar derivations found for this input."
        )
        return ToolResult(
            success=True, output=output,
            tool_name="code_explain",
            duration=time.time() - start,
            data={"predictions": preds[:3]},
        )
    except Exception as exc:
        return ToolResult(
            success=False, output="", error=str(exc),
            tool_name="code_explain", duration=time.time() - start,
        )


def _tool_file_read(input_text: str, context: dict) -> ToolResult:
    """Read a file from the filesystem."""
    start = time.time()
    path = input_text.strip()
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return ToolResult(
                success=False, output="",
                error=f"File not found: {path}",
                tool_name="file_read", duration=time.time() - start,
            )
        content = p.read_text(encoding="utf-8", errors="replace")
        # Truncate if very large
        if len(content) > 10000:
            content = content[:10000] + f"\n... [truncated, {len(content)} chars total]"
        return ToolResult(
            success=True, output=content,
            tool_name="file_read", duration=time.time() - start,
            data={"path": str(p), "size": p.stat().st_size},
        )
    except Exception as exc:
        return ToolResult(
            success=False, output="", error=str(exc),
            tool_name="file_read", duration=time.time() - start,
        )


def _tool_file_list(input_text: str, context: dict) -> ToolResult:
    """List files in a directory."""
    start = time.time()
    path = input_text.strip() or "."
    try:
        p = Path(path).expanduser()
        if not p.is_dir():
            return ToolResult(
                success=False, output="",
                error=f"Not a directory: {path}",
                tool_name="file_list", duration=time.time() - start,
            )
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        lines = []
        for e in entries[:100]:
            prefix = "[dir] " if e.is_dir() else "      "
            lines.append(f"{prefix}{e.name}")
        output = "\n".join(lines) if lines else "(empty directory)"
        return ToolResult(
            success=True, output=output,
            tool_name="file_list", duration=time.time() - start,
            data={"path": str(p), "count": len(entries)},
        )
    except Exception as exc:
        return ToolResult(
            success=False, output="", error=str(exc),
            tool_name="file_list", duration=time.time() - start,
        )


def _tool_web_search(input_text: str, context: dict) -> ToolResult:
    """Search the web using a simple HTTP request.

    Uses DuckDuckGo HTML search (no API key required).
    """
    start = time.time()
    query = input_text.strip()
    if not query:
        return ToolResult(
            success=False, output="", error="Empty search query",
            tool_name="web_search", duration=time.time() - start,
        )
    try:
        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        req = urllib.request.Request(url, headers={
            "User-Agent": "MKAngel/1.0 (Grammar Language Model)",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Simple extraction of result snippets
        results = []
        # Look for result snippets in the HTML
        import re
        snippets = re.findall(r'class="result__snippet">(.*?)</a>', html, re.DOTALL)
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)

        for i, (title, snippet) in enumerate(zip(titles[:5], snippets[:5])):
            # Strip HTML tags
            clean_title = re.sub(r'<[^>]+>', '', title).strip()
            clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            results.append(f"{i+1}. {clean_title}\n   {clean_snippet}")

        output = (
            f"Search results for '{query}':\n\n" + "\n\n".join(results)
            if results
            else f"No results found for '{query}'"
        )
        return ToolResult(
            success=True, output=output,
            tool_name="web_search", duration=time.time() - start,
            data={"query": query, "num_results": len(results)},
        )
    except Exception as exc:
        return ToolResult(
            success=False, output="",
            error=f"Search failed: {exc}",
            tool_name="web_search", duration=time.time() - start,
        )


def _tool_web_fetch(input_text: str, context: dict) -> ToolResult:
    """Fetch a URL and return its text content."""
    start = time.time()
    url = input_text.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "MKAngel/1.0 (Grammar Language Model)",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8", errors="replace")

        # Strip HTML tags for readability
        import re
        # Remove script and style blocks
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
        # Strip remaining tags
        text = re.sub(r'<[^>]+>', ' ', content)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) > 5000:
            text = text[:5000] + f"\n... [truncated, {len(text)} chars total]"

        return ToolResult(
            success=True, output=text,
            tool_name="web_fetch", duration=time.time() - start,
            data={"url": url, "length": len(text)},
        )
    except Exception as exc:
        return ToolResult(
            success=False, output="",
            error=f"Fetch failed: {exc}",
            tool_name="web_fetch", duration=time.time() - start,
        )


def _tool_calculate(input_text: str, context: dict) -> ToolResult:
    """Safe mathematical expression evaluator."""
    start = time.time()
    expr = input_text.strip()
    # Whitelist safe math operations
    import math
    safe_dict = {
        "__builtins__": {},
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "len": len, "int": int, "float": float,
        "pow": pow, "divmod": divmod,
        "pi": math.pi, "e": math.e, "tau": math.tau,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10,
        "log2": math.log2, "exp": math.exp, "floor": math.floor,
        "ceil": math.ceil, "factorial": math.factorial,
        "gcd": math.gcd,
    }
    try:
        result = eval(expr, safe_dict)
        return ToolResult(
            success=True, output=str(result),
            tool_name="calculate", duration=time.time() - start,
            data={"expression": expr, "result": result},
        )
    except Exception as exc:
        return ToolResult(
            success=False, output="",
            error=f"Calculation error: {exc}",
            tool_name="calculate", duration=time.time() - start,
        )


def _tool_grammar_predict(input_text: str, context: dict) -> ToolResult:
    """Use GLM to predict continuations."""
    start = time.time()
    domain = context.get("domain", "linguistic")
    try:
        from glm.angel import Angel
        angel = Angel()
        angel.awaken()
        tokens = input_text.lower().split()
        preds = angel.predict(tokens, domain=domain, horizon=8)
        lines = [f"Predictions (domain: {domain}):"]
        for p in preds[:5]:
            lines.append(
                f"  {p.get('predicted', '?')} "
                f"[{p.get('grammar', '?')}] "
                f"confidence: {p.get('confidence', 0):.2f}"
            )
        return ToolResult(
            success=True, output="\n".join(lines),
            tool_name="grammar_predict", duration=time.time() - start,
            data={"predictions": preds[:5]},
        )
    except Exception as exc:
        return ToolResult(
            success=False, output="", error=str(exc),
            tool_name="grammar_predict", duration=time.time() - start,
        )


def _tool_grammar_translate(input_text: str, context: dict) -> ToolResult:
    """Translate patterns between grammar domains."""
    start = time.time()
    src = context.get("source_domain", "linguistic")
    dst = context.get("target_domain", "computational")
    try:
        from glm.angel import Angel
        angel = Angel()
        angel.awaken()
        tokens = input_text.lower().split()
        translations = angel.translate(tokens, src, dst)
        lines = [f"Domain translation: {src} -> {dst}"]
        for t in translations[:5]:
            lines.append(
                f"  {t.get('source_grammar', '?')} -> "
                f"{t.get('target_grammar', '?')}: "
                f"{t.get('mapping', {}).get('type', 'unknown')}"
            )
        return ToolResult(
            success=True, output="\n".join(lines),
            tool_name="grammar_translate", duration=time.time() - start,
            data={"translations": translations[:5]},
        )
    except Exception as exc:
        return ToolResult(
            success=False, output="", error=str(exc),
            tool_name="grammar_translate", duration=time.time() - start,
        )


def _tool_grammar_introspect(input_text: str, context: dict) -> ToolResult:
    """Angel introspection — examine internal state."""
    start = time.time()
    try:
        from glm.angel import Angel
        angel = Angel()
        angel.awaken()
        info = angel.introspect()
        lines = [
            "Angel Introspection:",
            f"  Domains:       {', '.join(info['domains_loaded'])}",
            f"  Grammars:      {info['total_grammars']}",
            f"  Rules:         {info['total_rules']}",
            f"  Productions:   {info['total_productions']}",
            f"  Strange loops: {info['strange_loops_detected']}",
            f"  Substrates:    {', '.join(info['substrates_loaded'])}",
            f"  Model params:  {info['model_params']}",
        ]
        return ToolResult(
            success=True, output="\n".join(lines),
            tool_name="grammar_introspect", duration=time.time() - start,
            data=info,
        )
    except Exception as exc:
        return ToolResult(
            success=False, output="", error=str(exc),
            tool_name="grammar_introspect", duration=time.time() - start,
        )


def _tool_system_status(input_text: str, context: dict) -> ToolResult:
    """Report system status."""
    start = time.time()
    import platform
    import sys
    lines = [
        "MKAngel System Status:",
        f"  Platform:    {platform.system()} {platform.release()}",
        f"  Python:      {sys.version.split()[0]}",
        f"  Machine:     {platform.machine()}",
    ]
    # Check network
    try:
        urllib.request.urlopen("https://httpbin.org/get", timeout=3)
        lines.append("  Network:     connected")
    except Exception:
        lines.append("  Network:     offline")

    return ToolResult(
        success=True, output="\n".join(lines),
        tool_name="system_status", duration=time.time() - start,
    )


# ---------------------------------------------------------------------------
# Default registry factory
# ---------------------------------------------------------------------------

def create_default_registry() -> ToolRegistry:
    """Create and populate the default tool registry."""
    reg = ToolRegistry()

    reg.register(ToolSpec(
        name="code_explain", description="Explain code structure using GLM",
        category=ToolCategory.CODE, handler=_tool_code_explain,
        keywords=["code", "explain", "analyze", "function", "class"],
    ))
    reg.register(ToolSpec(
        name="file_read", description="Read a file",
        category=ToolCategory.FILE, handler=_tool_file_read,
        keywords=["read", "file", "open", "cat", "show"],
    ))
    reg.register(ToolSpec(
        name="file_list", description="List directory contents",
        category=ToolCategory.FILE, handler=_tool_file_list,
        keywords=["list", "ls", "dir", "files", "directory"],
    ))
    reg.register(ToolSpec(
        name="web_search", description="Search the web",
        category=ToolCategory.WEB, handler=_tool_web_search,
        keywords=["search", "google", "find", "lookup", "web"],
        requires_network=True,
    ))
    reg.register(ToolSpec(
        name="web_fetch", description="Fetch a web page",
        category=ToolCategory.WEB, handler=_tool_web_fetch,
        keywords=["fetch", "get", "url", "page", "website"],
        requires_network=True,
    ))
    reg.register(ToolSpec(
        name="calculate", description="Evaluate math expressions",
        category=ToolCategory.MATH, handler=_tool_calculate,
        keywords=["calc", "math", "compute", "evaluate", "solve"],
    ))
    reg.register(ToolSpec(
        name="grammar_predict", description="GLM grammar predictions",
        category=ToolCategory.GRAMMAR, handler=_tool_grammar_predict,
        keywords=["predict", "forecast", "grammar", "derive"],
    ))
    reg.register(ToolSpec(
        name="grammar_translate", description="Translate between grammar domains",
        category=ToolCategory.GRAMMAR, handler=_tool_grammar_translate,
        keywords=["translate", "domain", "isomorphism", "cross-domain"],
    ))
    reg.register(ToolSpec(
        name="grammar_introspect", description="Angel self-inspection",
        category=ToolCategory.GRAMMAR, handler=_tool_grammar_introspect,
        keywords=["introspect", "status", "angel", "self"],
    ))
    reg.register(ToolSpec(
        name="system_status", description="System status report",
        category=ToolCategory.SYSTEM, handler=_tool_system_status,
        keywords=["status", "system", "info", "health"],
    ))

    return reg
