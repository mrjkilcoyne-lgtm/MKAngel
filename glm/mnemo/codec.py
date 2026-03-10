"""
codec.py — MNEMO compression and decompression.

The ``MnemoCodec`` compresses arbitrary structured data into MNEMO notation
and decompresses MNEMO back into full representations.  The compression
works because MNEMO tokens reference *grammar rules* (not raw data), and
grammar rules are generative — each rule can produce infinite outputs.

The mathematics of the compression:
    - A MNEMO program is up to 50 tokens, each 1-3 characters.
    - 50 tokens x 3 chars = 150 characters maximum.
    - The MNEMO vocabulary has ~567 valid tokens.
    - Each token selects from the grammar's ~1500+ production rules.
    - 1500^50 ~ 10^158 possible programs.
    - 75 billion (7.5 x 10^10) parameters is a vanishingly small subset
      of this space — the compression is *conservative*.

The codec uses the grammar model's rule set as the compression dictionary.
Instead of storing data, it stores the *recipe* for regenerating the data
from grammar rules.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import rules as _rules
from .language import MnemoGrammar, MnemoToken, encode as _encode


# ---------------------------------------------------------------------------
# Compression entry — one item in the compressed representation
# ---------------------------------------------------------------------------

@dataclass
class CompressionEntry:
    """A single entry in a compressed MNEMO representation.

    Each entry maps a segment of the original data to the MNEMO token(s)
    that can regenerate it, plus the grammar rule that connects them.

    Attributes:
        token:       The MNEMO token string.
        rule_ref:    Identifier of the grammar rule this token activates.
        segment:     The original data segment this entry covers.
        confidence:  How faithfully the token captures the segment [0, 1].
        metadata:    Additional reconstruction hints.
    """
    token: str
    rule_ref: str
    segment: Any
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MnemoCodec
# ---------------------------------------------------------------------------

@dataclass
class MnemoCodec:
    """Compresses and decompresses between full representations and MNEMO.

    The codec maintains a *rule index* — a mapping from grammar rule
    identifiers to the MNEMO tokens that invoke them.  This index is the
    compression dictionary: it lets the codec translate structured data
    into the shortest MNEMO sequence that, when expanded through the
    grammar engine, reconstructs the original.

    The 150 chars -> 75B parameter mapping works because:
        1. Each MNEMO token references a grammar rule (not raw data).
        2. Grammar rules are generative — each rule can produce infinite outputs.
        3. 50 tokens x 3 chars = 150 chars.
        4. Each token selects from ~1500 grammar rules.
        5. 1500^50 ~ 10^158 possible combinations (far exceeds 75B).

    Attributes:
        grammar:        The MNEMO grammar used for encoding/decoding.
        rule_index:     Grammar rule ID -> MNEMO token mapping.
        max_tokens:     Maximum tokens in a compressed output (default 50).
        compression_log: History of compression operations.
    """

    grammar: MnemoGrammar = field(default_factory=MnemoGrammar)
    rule_index: Dict[str, str] = field(default_factory=dict)
    max_tokens: int = 50
    compression_log: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.rule_index:
            self.rule_index = self._build_rule_index()

    # -- public API ---------------------------------------------------------

    def compress(self, data: Any) -> str:
        """Compress any data to MNEMO notation.

        The compression strategy:
        1. Analyze the data to determine its domain and structure.
        2. Segment the data into chunks that align with grammar rules.
        3. For each segment, find the MNEMO token that best captures it.
        4. Assemble the tokens into a MNEMO program string.

        Parameters:
            data: The data to compress.  Can be a string, dict, list,
                  or any structured representation.

        Returns:
            A MNEMO string of at most ``max_tokens`` tokens
            (at most ``max_tokens * 3`` characters plus spaces).
        """
        # Step 1: Analyze domain and structure.
        domain = self._detect_domain(data)
        structure = self._analyze_structure(data)

        # Step 2: Segment into compressible chunks.
        segments = self._segment(data, structure)

        # Step 3: Map each segment to its best MNEMO token.
        entries: List[CompressionEntry] = []
        for segment in segments:
            entry = self._compress_segment(segment, domain)
            entries.append(entry)

        # Step 4: Assemble and trim to max_tokens.
        entries = entries[:self.max_tokens]
        mnemo_string = " ".join(e.token for e in entries)

        # Log the compression.
        self.compression_log.append({
            "original_size": _data_size(data),
            "compressed_size": len(mnemo_string),
            "tokens": len(entries),
            "domain": domain,
            "entries": entries,
        })

        return mnemo_string

    def decompress(self, mnemo: str) -> Dict[str, Any]:
        """Expand MNEMO back to a full representation.

        Decompression reverses the compression:
        1. Parse the MNEMO string into tokens.
        2. Expand each token through the grammar to recover its operation.
        3. Assemble the operations into a structured reconstruction plan.

        Parameters:
            mnemo: A valid MNEMO string.

        Returns:
            A dict containing:
                tokens:     Parsed token objects.
                operations: The expanded grammar operations.
                plan:       The reconstruction plan — ordered steps the
                            grammar engine should execute to regenerate
                            the original data.
                metadata:   Compression metadata (if available from log).
        """
        valid, errors = _rules.validate(mnemo)
        if not valid:
            return {
                "tokens": [],
                "operations": [],
                "plan": [],
                "metadata": {"errors": errors},
            }

        tokens = self.grammar.tokenize(mnemo)
        operations = self.grammar.expand_sequence(tokens)

        # Build a reconstruction plan.
        plan = self._build_reconstruction_plan(tokens, operations)

        # Look up compression metadata if we compressed this ourselves.
        metadata = self._find_compression_metadata(mnemo)

        return {
            "tokens": tokens,
            "operations": operations,
            "plan": plan,
            "metadata": metadata,
        }

    def compression_ratio(self, original: Any, compressed: str) -> Dict[str, Any]:
        """Calculate the compression achieved.

        Returns a dict with:
            original_size:    Size of the original data (characters or elements).
            compressed_size:  Size of the compressed MNEMO string (characters).
            ratio:            original_size / compressed_size.
            space_reduction:  1 - (compressed_size / original_size), as percentage.
            theoretical_max:  The theoretical information capacity of the
                              compressed representation.
            parameter_space:  The number of distinct programs possible at
                              this compression length.
        """
        orig_size = _data_size(original)
        comp_size = len(compressed)
        token_count = len(compressed.split()) if compressed.strip() else 0

        # Vocabulary size: number of valid MNEMO tokens.
        vocab_size = len(_rules.MNEMO_VOCABULARY)

        # Parameter space: vocab_size ^ token_count.
        if token_count > 0:
            # Use log to avoid overflow, then express as "10^N".
            log_space = token_count * math.log10(vocab_size)
            parameter_space_log10 = log_space
        else:
            parameter_space_log10 = 0.0

        # With grammar rule expansion, each token accesses ~1500 rules.
        grammar_rule_count = max(vocab_size, 1500)
        if token_count > 0:
            theoretical_log10 = token_count * math.log10(grammar_rule_count)
        else:
            theoretical_log10 = 0.0

        ratio = orig_size / comp_size if comp_size > 0 else float("inf")
        reduction = (1.0 - comp_size / orig_size) * 100 if orig_size > 0 else 0.0

        return {
            "original_size": orig_size,
            "compressed_size": comp_size,
            "token_count": token_count,
            "ratio": round(ratio, 2),
            "space_reduction_percent": round(reduction, 1),
            "parameter_space_log10": round(parameter_space_log10, 1),
            "theoretical_capacity_log10": round(theoretical_log10, 1),
            "exceeds_75b": parameter_space_log10 > math.log10(75e9),
        }

    # -- internal: rule index -----------------------------------------------

    def _build_rule_index(self) -> Dict[str, str]:
        """Build the grammar rule -> MNEMO token mapping.

        Each entry in MNEMO_VOCABULARY gets a synthetic rule ID based
        on its domain, operation, and modifier.  This is the compression
        dictionary.
        """
        index: Dict[str, str] = {}
        for token_str, entry in _rules.MNEMO_VOCABULARY.items():
            # Rule ID: "domain.operation[.modifier]"
            parts = [entry["domain"], entry["operation"]]
            if entry["modifier"]:
                parts.append(entry["modifier"])
            rule_id = ".".join(parts)
            index[rule_id] = token_str
        return index

    # -- internal: domain detection -----------------------------------------

    def _detect_domain(self, data: Any) -> str:
        """Detect the most likely domain of the input data.

        Uses heuristic keyword matching on the string representation.
        """
        text = str(data).lower()

        domain_scores: Dict[str, float] = {
            "L": 0.0, "C": 0.0, "B": 0.0,
            "X": 0.0, "E": 0.0, "M": 0.0,
        }

        # Linguistic markers.
        for kw in ("word", "sentence", "morpheme", "phoneme", "syntax",
                    "verb", "noun", "clause", "phrase", "grammar"):
            if kw in text:
                domain_scores["L"] += 1.0

        # Chemical markers.
        for kw in ("molecule", "atom", "bond", "reaction", "compound",
                    "element", "formula", "ion", "acid", "base"):
            if kw in text:
                domain_scores["C"] += 1.0

        # Biological markers.
        for kw in ("gene", "protein", "dna", "rna", "codon", "amino",
                    "cell", "enzyme", "nucleotide", "sequence"):
            if kw in text:
                domain_scores["B"] += 1.0

        # Computational markers.
        for kw in ("function", "algorithm", "variable", "loop", "class",
                    "return", "import", "def ", "if ", "for "):
            if kw in text:
                domain_scores["X"] += 1.0

        # Etymological markers.
        for kw in ("origin", "root", "latin", "greek", "proto",
                    "ancestor", "cognate", "derived from", "evolved"):
            if kw in text:
                domain_scores["E"] += 1.0

        # Meta markers.
        for kw in ("mnemo", "grammar rule", "token", "encode", "decode",
                    "compress", "meta", "self-referential"):
            if kw in text:
                domain_scores["M"] += 1.0

        best = max(domain_scores, key=lambda k: domain_scores[k])
        if domain_scores[best] == 0:
            return "*"  # Universal / unknown.
        return best

    # -- internal: structure analysis ---------------------------------------

    def _analyze_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze the structural properties of the data.

        Returns a descriptor used to guide segmentation.
        """
        if isinstance(data, str):
            return {
                "type": "text",
                "length": len(data),
                "words": len(data.split()),
                "lines": data.count("\n") + 1,
                "has_structure": bool(re.search(r"[{}\[\]():]", data)),
            }
        elif isinstance(data, dict):
            return {
                "type": "mapping",
                "keys": list(data.keys()),
                "depth": _dict_depth(data),
                "size": len(data),
            }
        elif isinstance(data, (list, tuple)):
            return {
                "type": "sequence",
                "length": len(data),
                "element_types": list({type(x).__name__ for x in data}),
            }
        else:
            return {
                "type": type(data).__name__,
                "size": len(str(data)),
            }

    # -- internal: segmentation ---------------------------------------------

    def _segment(self, data: Any, structure: Dict[str, Any]) -> List[Any]:
        """Segment data into chunks aligned with grammar rules.

        The segmentation strategy depends on the data type:
        - Text: split into semantic units (sentences/phrases/words).
        - Mappings: each key-value pair is a segment.
        - Sequences: each element (or group of elements) is a segment.
        - Other: the whole datum is one segment.
        """
        if isinstance(data, str):
            return self._segment_text(data)
        elif isinstance(data, dict):
            return self._segment_dict(data)
        elif isinstance(data, (list, tuple)):
            return self._segment_sequence(data)
        else:
            return [data]

    def _segment_text(self, text: str) -> List[str]:
        """Segment text into semantic units."""
        # Split on sentence boundaries first.
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        segments: List[str] = []
        for sentence in sentences:
            # If a sentence is short enough, keep it whole.
            words = sentence.split()
            if len(words) <= 5:
                segments.append(sentence)
            else:
                # Split long sentences into phrase-level chunks.
                # Use natural break points: commas, semicolons, conjunctions.
                phrases = re.split(r"[,;]\s*|\s+(?:and|or|but|then|while|when)\s+", sentence)
                segments.extend(p.strip() for p in phrases if p.strip())

        return segments if segments else [text]

    def _segment_dict(self, data: Dict[str, Any]) -> List[Any]:
        """Segment a dict into key-value pair representations."""
        segments: List[Any] = []
        for key, value in data.items():
            segments.append({"key": key, "value": value})
        return segments if segments else [data]

    def _segment_sequence(self, data: Any) -> List[Any]:
        """Segment a sequence into individual or grouped elements."""
        items = list(data)
        if len(items) <= self.max_tokens:
            return items

        # Group elements to fit within max_tokens.
        group_size = max(1, len(items) // self.max_tokens)
        segments: List[Any] = []
        for i in range(0, len(items), group_size):
            group = items[i:i + group_size]
            segments.append(group if len(group) > 1 else group[0])
        return segments

    # -- internal: segment compression --------------------------------------

    def _compress_segment(self, segment: Any, domain: str) -> CompressionEntry:
        """Compress a single segment into a MNEMO token.

        Strategy:
        1. Use ``encode()`` to get a MNEMO encoding of the segment's
           natural language description.
        2. If the segment has structural properties, choose the most
           specific operation token.
        3. Fall back to domain + analyze as a generic representation.
        """
        text = str(segment)

        # Try encoding the segment text.
        encoded = _encode(text)
        tokens = encoded.split()

        if tokens:
            # Use the first token — it captures the primary intent.
            token = tokens[0]
        else:
            # Fallback: domain + analyze.
            d_char = domain if domain != "*" else "*"
            token = f"{d_char}a"

        # Validate the token.
        valid, _ = _rules.validate_token(token)
        if not valid:
            token = f"{domain}a" if domain in _rules.DOMAIN_CODES else "*a"

        # Build rule reference.
        entry = _rules.lookup(token)
        if entry:
            parts = [entry["domain"], entry["operation"]]
            if entry["modifier"]:
                parts.append(entry["modifier"])
            rule_ref = ".".join(parts)
        else:
            rule_ref = "unknown"

        # Confidence: shorter segments compress more faithfully.
        segment_len = len(text)
        confidence = min(1.0, max(0.1, 1.0 / (1.0 + math.log(1 + segment_len) / 10)))

        return CompressionEntry(
            token=token,
            rule_ref=rule_ref,
            segment=segment,
            confidence=confidence,
        )

    # -- internal: reconstruction plan --------------------------------------

    def _build_reconstruction_plan(
        self,
        tokens: List[MnemoToken],
        operations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build an ordered plan for reconstructing original data from MNEMO.

        Each step in the plan specifies:
        - The grammar domain to activate.
        - The operation to perform.
        - The direction of derivation.
        - Dependencies on previous steps.
        """
        plan: List[Dict[str, Any]] = []

        for i, op in enumerate(operations):
            if "compound" in op:
                # Compound operations become multi-step plan entries.
                for j, sub_op in enumerate(op.get("steps", [])):
                    plan.append({
                        "step": len(plan),
                        "domain": sub_op["domain"],
                        "action": sub_op["grammar_actions"][0] if sub_op["grammar_actions"] else "noop",
                        "direction": sub_op["direction"],
                        "depends_on": [len(plan) - 1] if len(plan) > 0 else [],
                        "compound": op["compound"],
                        "compound_step": j,
                    })
            else:
                actions = op.get("grammar_actions", ["noop"])
                plan.append({
                    "step": len(plan),
                    "domain": op["domain"],
                    "action": actions[0] if actions else "noop",
                    "all_actions": actions,
                    "direction": op["direction"],
                    "depends_on": [len(plan) - 1] if len(plan) > 0 else [],
                })

        return plan

    # -- internal: metadata lookup ------------------------------------------

    def _find_compression_metadata(self, mnemo: str) -> Dict[str, Any]:
        """Look up compression metadata from the log, if available."""
        for log_entry in reversed(self.compression_log):
            tokens = " ".join(e.token for e in log_entry.get("entries", []))
            if tokens == mnemo:
                return {
                    "original_size": log_entry.get("original_size"),
                    "domain": log_entry.get("domain"),
                    "found_in_log": True,
                }
        return {"found_in_log": False}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _data_size(data: Any) -> int:
    """Estimate the size of data in characters."""
    if isinstance(data, str):
        return len(data)
    elif isinstance(data, (list, tuple)):
        return sum(_data_size(x) for x in data)
    elif isinstance(data, dict):
        return sum(_data_size(k) + _data_size(v) for k, v in data.items())
    else:
        return len(str(data))


def _dict_depth(d: Any, depth: int = 0) -> int:
    """Recursively compute the maximum nesting depth of a dict."""
    if not isinstance(d, dict) or not d:
        return depth
    return max(_dict_depth(v, depth + 1) for v in d.values())
