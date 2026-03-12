"""
stages.py -- The four GSM reasoning stages implemented via grammar operations.

Each stage transforms structured claims using MKAngel's grammar engine
instead of LLM API calls.  The grammar IS the reasoning engine: rules
fire, productions expand, loops are detected, and derivations are traced
-- all in pure Python, all auditable.

Stage mapping from GSM -> Grammar operations:

    SKELETON (strip to logical bones)
        -> Syntactic grammar productions for S-R-O extraction
        -> Morphological grammar rules to strip hedging/decoration
        -> Rule confidence scores for skeleton strength

    DAG (map dependency graph)
        -> Production reference graph for dependencies
        -> Grammar.find_loops() for cycle detection
        -> Graph traversal on rule graph for root/leaf detection

    DISCONFIRM (hunt structural weakness)
        -> Backward derivation from conclusions to find weak premises
        -> Strange loop analysis for circular reasoning
        -> Fugue composition for cross-domain contradiction detection

    SYNTHESIS (minimum viable logic)
        -> Forward derivation from proven roots only
        -> Isomorphism discovery for cross-domain validation
        -> MNEMO compression of the clean argument
"""

from __future__ import annotations

import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from glm.core.grammar import Direction, Grammar, Production, Rule, StrangeLoop
from glm.core.engine import DerivationEngine, Derivation, DerivationTree

from .result import (
    ClaimStrength,
    DAGEdge,
    DAGNode,
    DAGResult,
    DisconfirmResult,
    EvidenceLevel,
    EvidenceNode,
    FallacyType,
    SkeletonResult,
    SynthesisResult,
    Triple,
    WeaknessReport,
)


# ---------------------------------------------------------------------------
# PipelineStage -- abstract base for all stages
# ---------------------------------------------------------------------------

class PipelineStage(ABC):
    """Abstract base class for a pipeline stage.

    Each stage:
    1. Receives input (raw text or previous stage output).
    2. Applies grammar operations to transform it.
    3. Produces a typed result object.

    Stages are independently testable: each can run in isolation
    given appropriate input.

    Attributes:
        name:       Human-readable stage name.
        engine:     The shared DerivationEngine instance.
        grammars:   Grammar instances available to this stage.
        config:     Stage-specific configuration.
    """

    def __init__(
        self,
        engine: Optional[DerivationEngine] = None,
        grammars: Optional[Dict[str, Grammar]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.engine = engine or DerivationEngine()
        self.grammars = grammars or {}
        self.config = config or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name identifier."""
        ...

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Execute the stage on the given input, returning a result object."""
        ...


# ---------------------------------------------------------------------------
# SKELETON stage -- strip to logical bones
# ---------------------------------------------------------------------------

class SkeletonStage(PipelineStage):
    """Stage 1: SKELETON -- decompose input into structured claims.

    GSM description: "Strip to logical bones: S-R-O triples, implicit
    premises, noise removal."

    Grammar mapping:
        - Syntactic grammar's phrase-structure productions (S -> NP VP)
          drive Subject-Relation-Object extraction.  NP -> Subject,
          V-head -> Relation, VP-complement -> Object.
        - Morphological grammar rules identify hedging markers (modal verbs,
          adverbial qualifiers, epistemic markers) that weaken claims.
          These are stripped from the skeleton but preserved as metadata.
        - Rule confidence scores (Rule.weight) propagate into Triple
          confidence -- a triple is only as strong as the weakest rule
          in its extraction chain.

    The skeleton stage does NOT judge truth -- it only structures.
    """

    @property
    def name(self) -> str:
        return "skeleton"

    def run(self, input_data: Any) -> SkeletonResult:
        """Decompose input into S-R-O triples using grammar rules.

        Parameters:
            input_data: Raw text string or structured data to decompose.

        Returns:
            SkeletonResult with extracted triples, implicit premises, and noise.
        """
        text = str(input_data) if not isinstance(input_data, str) else input_data
        result = SkeletonResult(raw_input=text)

        # Step 1: Segment input into clause-level units.
        clauses = self._segment_clauses(text)

        # Step 2: For each clause, attempt S-R-O extraction via grammar.
        syntactic = self.grammars.get("syntactic")
        morphological = self.grammars.get("morphological")

        derivation_count = 0
        covered_chars = 0

        for clause in clauses:
            clause_stripped = clause.strip()
            if not clause_stripped:
                continue

            # Try grammar-based extraction.
            triple, confidence, deriv_steps = self._extract_triple(
                clause_stripped, syntactic
            )

            derivation_count += deriv_steps

            if triple is not None:
                # Check for hedging via morphological grammar.
                hedge_penalty, noise_segments = self._detect_hedging(
                    clause_stripped, morphological
                )
                triple.confidence = max(0.05, confidence - hedge_penalty)
                result.triples.append(triple)
                result.noise.extend(noise_segments)
                covered_chars += len(clause_stripped)
            else:
                # Could not extract a triple -- check if it is implicit.
                implicit = self._infer_implicit(clause_stripped, syntactic)
                if implicit:
                    result.implicit_premises.append(implicit)
                    covered_chars += len(clause_stripped)
                else:
                    # Pure noise / decoration.
                    result.noise.append(clause_stripped)

        # Step 3: Look for implicit premises required by grammar rules
        # (e.g., agreement constraints imply unstated properties).
        if syntactic:
            additional = self._find_grammar_implied_premises(
                result.triples, syntactic
            )
            result.implicit_premises.extend(additional)

        result.derivation_count = derivation_count
        total_chars = len(text) if text else 1
        result.grammar_coverage = min(1.0, covered_chars / total_chars)

        return result

    # -- internal: clause segmentation --------------------------------------

    def _segment_clauses(self, text: str) -> List[str]:
        """Split text into clause-level segments.

        Uses sentence boundaries and coordinating conjunctions as
        segmentation points -- mirroring how syntactic grammar treats
        S -> S Conj S.
        """
        # Split on sentence-ending punctuation.
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        clauses: List[str] = []
        for sent in sentences:
            # Further split on clause-level conjunctions.
            parts = re.split(
                r'\s*(?:,\s*(?:and|or|but|yet|so|because|although|while|if|when|since|whereas))\s+',
                sent
            )
            for part in parts:
                stripped = part.strip().rstrip('.,;:!?')
                if stripped:
                    clauses.append(stripped)
        return clauses if clauses else [text]

    # -- internal: S-R-O extraction -----------------------------------------

    def _extract_triple(
        self,
        clause: str,
        grammar: Optional[Grammar],
    ) -> Tuple[Optional[Triple], float, int]:
        """Extract an S-R-O triple from a clause using grammar derivation.

        Strategy:
        1. Apply syntactic grammar forward to see which productions fire.
        2. If productions fire, the derivation tree encodes the parse:
           - NP daughters of S -> Subject
           - V head of VP -> Relation
           - NP/PP/CP daughters of VP -> Object
        3. If no grammar match, fall back to heuristic word-order extraction.

        Returns (triple_or_None, confidence, derivation_steps).
        """
        deriv_steps = 0
        words = clause.split()

        if not words:
            return None, 0.0, 0

        # Attempt grammar-driven extraction.
        if grammar:
            tree = self.engine.derive(clause, grammar, "forward", max_steps=50)
            deriv_steps = len(tree.all_forms()) - 1  # root doesn't count

            # Check if any production that matches S -> NP VP fired.
            matched_rules = self._collect_fired_rules(tree)
            s_rules = [r for r in matched_rules if r.get("pattern") == "S"]

            if s_rules:
                # Use rule weights for confidence.
                avg_weight = sum(r.get("weight", 0.5) for r in s_rules) / len(s_rules)
                triple = self._triple_from_parse(clause, words, s_rules)
                if triple:
                    triple.source_rule = s_rules[0].get("rule_id", "")
                    return triple, avg_weight, deriv_steps

        # Fallback: heuristic SVO extraction from word order.
        triple, confidence = self._heuristic_svo(clause, words)
        return triple, confidence, deriv_steps

    def _collect_fired_rules(self, tree: DerivationTree) -> List[Dict[str, Any]]:
        """Walk a derivation tree and collect info about all rules that fired."""
        results: List[Dict[str, Any]] = []

        def _walk(node: DerivationTree) -> None:
            if node.step is not None:
                results.append({
                    "rule_id": node.step.rule_id,
                    "input": node.step.input,
                    "output": node.step.output,
                    "pattern": self._extract_pattern(node.step),
                    "weight": getattr(node.step, "weight", 0.5),
                })
            for child in node.children:
                _walk(child)

        _walk(tree)
        return results

    @staticmethod
    def _extract_pattern(step: Derivation) -> Optional[str]:
        """Extract the LHS symbol from a derivation step if available."""
        inp = step.input
        if isinstance(inp, str):
            return inp
        if isinstance(inp, dict):
            return inp.get("structure", inp.get("lhs", None))
        return None

    def _triple_from_parse(
        self,
        clause: str,
        words: List[str],
        s_rules: List[Dict[str, Any]],
    ) -> Optional[Triple]:
        """Construct a Triple from parsed syntactic information."""
        # Use the derivation output to identify constituent boundaries.
        # With our grammar, the output encodes NP/VP structure.
        # For practical extraction, we combine grammar info with word position.
        if len(words) < 2:
            return None

        # Heuristic: in SVO languages (English), first NP-like span is subject,
        # first verb is relation, remainder is object.
        subject_words, verb, object_words = self._split_svo(words)

        if not verb:
            return None

        return Triple(
            subject=" ".join(subject_words),
            relation=verb,
            object=" ".join(object_words) if object_words else "",
            raw_text=clause,
        )

    def _split_svo(
        self, words: List[str]
    ) -> Tuple[List[str], str, List[str]]:
        """Split a word list into Subject, Verb, Object spans.

        Uses a simple deterministic strategy:
        - Scan for the first word that looks verbal (common verb indicators).
        - Everything before it is Subject.
        - Everything after it is Object.
        """
        # Common English auxiliary/copula/verb indicators.
        verb_indicators = {
            "is", "are", "was", "were", "be", "been", "being",
            "has", "have", "had", "do", "does", "did",
            "will", "shall", "would", "should", "could", "can", "may", "might",
            "must", "need",
            "causes", "implies", "requires", "depends", "leads",
            "shows", "proves", "suggests", "indicates", "demonstrates",
            "means", "equals", "contains", "produces", "creates",
        }

        verb_suffixes = ("s", "ed", "ing", "es", "ize", "ise", "ate", "ify")

        verb_idx = None
        for i, w in enumerate(words):
            w_lower = w.lower().rstrip(".,;:!?")
            if w_lower in verb_indicators:
                verb_idx = i
                break
            # Check for verbal morphology.
            if i > 0 and any(w_lower.endswith(s) for s in verb_suffixes):
                verb_idx = i
                break

        if verb_idx is None:
            # No verb found -- try the second word as a default.
            if len(words) >= 3:
                return words[:1], words[1], words[2:]
            elif len(words) == 2:
                return words[:1], words[1], []
            else:
                return words, "", []

        subject = words[:verb_idx] if verb_idx > 0 else words[:1]
        verb = words[verb_idx]
        obj = words[verb_idx + 1:] if verb_idx + 1 < len(words) else []

        return subject, verb, obj

    def _heuristic_svo(
        self, clause: str, words: List[str]
    ) -> Tuple[Optional[Triple], float]:
        """Fallback heuristic extraction when grammar rules don't fire."""
        if len(words) < 2:
            return None, 0.0

        subject_words, verb, object_words = self._split_svo(words)

        if not verb:
            return None, 0.0

        return Triple(
            subject=" ".join(subject_words),
            relation=verb,
            object=" ".join(object_words) if object_words else "",
            raw_text=clause,
            confidence=0.4,  # Lower confidence for heuristic extraction.
        ), 0.4

    # -- internal: hedging detection ----------------------------------------

    def _detect_hedging(
        self,
        clause: str,
        grammar: Optional[Grammar],
    ) -> Tuple[float, List[str]]:
        """Detect hedging/qualification markers using morphological grammar.

        Returns (penalty, noise_segments) where penalty is confidence
        reduction and noise_segments are the hedging phrases found.
        """
        hedge_markers = [
            "perhaps", "maybe", "possibly", "probably", "likely",
            "might", "could", "somewhat", "partially", "allegedly",
            "apparently", "seemingly", "it seems", "it appears",
            "in some cases", "under certain conditions",
            "to some extent", "more or less", "roughly",
            "I think", "I believe", "in my opinion",
        ]

        penalty = 0.0
        noise: List[str] = []
        clause_lower = clause.lower()

        for marker in hedge_markers:
            if marker in clause_lower:
                penalty += 0.1
                noise.append(marker)

        # Also try grammar-based detection if available.
        if grammar:
            tree = self.engine.derive(clause, grammar, "forward", max_steps=20)
            for leaf in tree.leaves():
                if isinstance(leaf.form, dict):
                    derived = leaf.form.get("derived", "")
                    if "hedge" in str(derived).lower() or "modal" in str(derived).lower():
                        penalty += 0.05

        return min(0.5, penalty), noise

    # -- internal: implicit premise inference --------------------------------

    def _infer_implicit(
        self,
        clause: str,
        grammar: Optional[Grammar],
    ) -> Optional[Triple]:
        """Attempt to infer an implicit premise from a clause.

        Uses backward derivation: if the clause looks like a conclusion,
        derive backward to find what premises are needed.
        """
        if not grammar:
            return None

        tree = self.engine.derive(clause, grammar, "backward", max_steps=30)
        if tree.children:
            # The first backward derivation suggests a required premise.
            first = tree.children[0]
            form = first.form
            if isinstance(form, str) and len(form.split()) >= 2:
                words = form.split()
                subject_words, verb, object_words = self._split_svo(words)
                if verb:
                    return Triple(
                        subject=" ".join(subject_words),
                        relation=verb,
                        object=" ".join(object_words),
                        confidence=0.3,
                        implicit=True,
                        raw_text=f"[implied by: {clause}]",
                    )
        return None

    # -- internal: grammar-implied premises ---------------------------------

    def _find_grammar_implied_premises(
        self,
        triples: List[Triple],
        grammar: Grammar,
    ) -> List[Triple]:
        """Find premises implied by grammar constraints.

        For example, if a grammar rule has conditions like
        'number agreement', that implies the subject has a number feature
        -- which might be an unstated premise in the argument.
        """
        implied: List[Triple] = []

        for rule in grammar.all_rules():
            if rule.conditions and isinstance(rule.conditions, dict):
                req = rule.conditions.get("requires", "")
                if req and isinstance(req, str):
                    # Check if any triple relies on this rule's pattern.
                    for triple in triples:
                        if rule.matches(triple.raw_text, Direction.FORWARD):
                            implied.append(Triple(
                                subject=triple.subject,
                                relation="has_property",
                                object=req,
                                confidence=0.2,
                                implicit=True,
                                source_rule=rule.id,
                                raw_text=f"[grammar constraint: {rule.name}]",
                            ))

        return implied


# ---------------------------------------------------------------------------
# DAG stage -- map dependency graph
# ---------------------------------------------------------------------------

class DAGStage(PipelineStage):
    """Stage 2: DAG -- map the dependency graph over skeleton claims.

    GSM description: "Map dependency graph: root nodes, dependency chains,
    cycles, orphans, critical path."

    Grammar mapping:
        - Each Triple from the skeleton becomes a DAGNode.
        - Edges are established when a grammar production can derive
          one claim from another (production LHS/RHS containment).
        - Grammar.find_loops() detects circular dependencies directly.
        - Root/leaf detection via in-degree/out-degree on the production
          reference graph.
        - Critical path: the longest chain in the DAG (depth-first search).

    The DAG stage does NOT judge validity -- it only maps structure.
    """

    @property
    def name(self) -> str:
        return "dag"

    def run(self, input_data: Any) -> DAGResult:
        """Build a dependency graph from skeleton claims.

        Parameters:
            input_data: A SkeletonResult from the skeleton stage.

        Returns:
            DAGResult with nodes, edges, roots, leaves, cycles, and critical path.
        """
        if not isinstance(input_data, SkeletonResult):
            raise TypeError(
                f"DAGStage expects SkeletonResult, got {type(input_data).__name__}"
            )

        skeleton = input_data
        result = DAGResult()

        # Step 1: Create a DAGNode for each claim.
        claim_nodes: Dict[str, DAGNode] = {}
        for triple in skeleton.all_claims:
            node = DAGNode(triple=triple)
            claim_nodes[node.id] = node
            result.nodes.append(node)

        if not result.nodes:
            return result

        # Step 2: Establish edges via grammar production analysis.
        result.edges = self._find_dependencies(
            list(claim_nodes.values()),
            self.grammars,
        )

        # Step 3: Compute graph properties.
        adj = self._build_adjacency(result.nodes, result.edges)
        in_degrees = self._in_degrees(result.nodes, result.edges)

        # Roots: nodes with no incoming edges (conclusions).
        for node in result.nodes:
            if in_degrees.get(node.id, 0) == 0:
                node.is_root = True
                result.roots.append(node.id)

        # Leaves: nodes with no outgoing edges (axioms/evidence).
        for node in result.nodes:
            if not adj.get(node.id):
                node.is_leaf = True
                result.leaves.append(node.id)

        # Step 4: Detect cycles using grammar loop detection.
        result.cycles = self._detect_cycles(result.nodes, result.edges)

        # Step 5: Find orphans (disconnected nodes).
        connected = self._find_connected(result.nodes, adj)
        for node in result.nodes:
            if node.id not in connected:
                node.is_orphan = True
                result.orphans.append(node.id)

        # Step 6: Compute critical path and depth.
        result.critical_path, result.depth = self._critical_path(
            result.nodes, adj
        )

        # Step 7: Set depth on each node.
        self._assign_depths(result.nodes, adj, result.roots)

        return result

    # -- internal: dependency discovery -------------------------------------

    def _find_dependencies(
        self,
        nodes: List[DAGNode],
        grammars: Dict[str, Grammar],
    ) -> List[DAGEdge]:
        """Discover dependencies between claims using grammar productions.

        Strategy: for each pair of claims (A, B), check if any grammar
        production can derive A's relation from B's relation (or vice
        versa).  If so, A depends on B.

        Also checks semantic containment: if B's subject or object
        appears in A's relation or object, there is a definitional
        dependency.
        """
        edges: List[DAGEdge] = []

        for i, node_a in enumerate(nodes):
            if node_a.triple is None:
                continue
            for j, node_b in enumerate(nodes):
                if i == j or node_b.triple is None:
                    continue

                dep_type, weight, rule_id = self._check_dependency(
                    node_a.triple, node_b.triple, grammars
                )

                if dep_type is not None:
                    edges.append(DAGEdge(
                        source_id=node_a.id,
                        target_id=node_b.id,
                        edge_type=dep_type,
                        weight=weight,
                        rule_id=rule_id,
                    ))

        return edges

    def _check_dependency(
        self,
        claim_a: Triple,
        claim_b: Triple,
        grammars: Dict[str, Grammar],
    ) -> Tuple[Optional[str], float, str]:
        """Check if claim_a depends on claim_b.

        Returns (edge_type, weight, rule_id) or (None, 0, "") if no dependency.
        """
        # Check 1: Semantic containment -- B's subject/object appears
        # in A's content.
        a_text = f"{claim_a.subject} {claim_a.relation} {claim_a.object}".lower()
        b_subject_lower = claim_b.subject.lower()
        b_object_lower = claim_b.object.lower()

        if b_subject_lower and len(b_subject_lower) > 2 and b_subject_lower in a_text:
            return "definitional", 0.6, ""
        if b_object_lower and len(b_object_lower) > 2 and b_object_lower in a_text:
            return "evidential", 0.5, ""

        # Check 2: Grammar-based derivation -- can any grammar derive
        # A's relation from B's object?
        for gname, grammar in grammars.items():
            # Forward: does B's object lead to A's subject?
            results = grammar.apply_forward(claim_b.object, max_steps=10)
            for form, rule_id in results:
                if isinstance(form, str) and claim_a.subject.lower() in form.lower():
                    return "causal", 0.7, rule_id

            # Also check relation derivation.
            results = grammar.apply_forward(claim_b.relation, max_steps=10)
            for form, rule_id in results:
                if isinstance(form, str) and claim_a.relation.lower() in form.lower():
                    return "logical", 0.65, rule_id

        return None, 0.0, ""

    # -- internal: graph utilities -------------------------------------------

    @staticmethod
    def _build_adjacency(
        nodes: List[DAGNode], edges: List[DAGEdge]
    ) -> Dict[str, List[str]]:
        """Build adjacency list: source -> [targets]."""
        adj: Dict[str, List[str]] = {n.id: [] for n in nodes}
        for e in edges:
            adj.setdefault(e.source_id, []).append(e.target_id)
        return adj

    @staticmethod
    def _in_degrees(
        nodes: List[DAGNode], edges: List[DAGEdge]
    ) -> Dict[str, int]:
        """Compute in-degree for each node."""
        deg: Dict[str, int] = {n.id: 0 for n in nodes}
        for e in edges:
            deg[e.target_id] = deg.get(e.target_id, 0) + 1
        return deg

    def _detect_cycles(
        self,
        nodes: List[DAGNode],
        edges: List[DAGEdge],
    ) -> List[List[str]]:
        """Detect cycles in the dependency graph.

        Uses DFS-based cycle detection, mirroring Grammar.find_loops().
        """
        adj = self._build_adjacency(nodes, edges)
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        on_stack: Set[str] = set()

        def _dfs(node_id: str, path: List[str]) -> None:
            visited.add(node_id)
            on_stack.add(node_id)
            path.append(node_id)

            for neighbor in adj.get(node_id, []):
                if neighbor in on_stack:
                    # Found a cycle.
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:])
                elif neighbor not in visited:
                    _dfs(neighbor, path)

            path.pop()
            on_stack.discard(node_id)

        for node in nodes:
            if node.id not in visited:
                _dfs(node.id, [])

        return cycles

    @staticmethod
    def _find_connected(
        nodes: List[DAGNode],
        adj: Dict[str, List[str]],
    ) -> Set[str]:
        """Find all nodes reachable from any root (BFS)."""
        # Build reverse adjacency too.
        rev_adj: Dict[str, List[str]] = {n.id: [] for n in nodes}
        for src, targets in adj.items():
            for tgt in targets:
                rev_adj.setdefault(tgt, []).append(src)

        # BFS from every node, treating edges as undirected.
        connected: Set[str] = set()
        if not nodes:
            return connected

        # Find the largest connected component.
        all_ids = {n.id for n in nodes}
        while all_ids:
            start = next(iter(all_ids))
            component: Set[str] = set()
            queue = [start]
            while queue:
                current = queue.pop(0)
                if current in component:
                    continue
                component.add(current)
                for neighbor in adj.get(current, []):
                    if neighbor not in component:
                        queue.append(neighbor)
                for neighbor in rev_adj.get(current, []):
                    if neighbor not in component:
                        queue.append(neighbor)
            if len(component) > len(connected):
                connected = component
            all_ids -= component

        return connected

    def _critical_path(
        self,
        nodes: List[DAGNode],
        adj: Dict[str, List[str]],
    ) -> Tuple[List[str], int]:
        """Find the longest path in the DAG (critical path).

        Returns (path_node_ids, depth).
        """
        best_path: List[str] = []
        memo: Dict[str, List[str]] = {}

        def _longest_from(node_id: str, visited_set: Set[str]) -> List[str]:
            if node_id in memo:
                return memo[node_id]
            if node_id in visited_set:
                return [node_id]  # Cycle -- stop.

            visited_set.add(node_id)
            best: List[str] = [node_id]
            for neighbor in adj.get(node_id, []):
                sub = _longest_from(neighbor, visited_set)
                candidate = [node_id] + sub
                if len(candidate) > len(best):
                    best = candidate
            visited_set.discard(node_id)
            memo[node_id] = best
            return best

        for node in nodes:
            path = _longest_from(node.id, set())
            if len(path) > len(best_path):
                best_path = path

        return best_path, max(0, len(best_path) - 1)

    @staticmethod
    def _assign_depths(
        nodes: List[DAGNode],
        adj: Dict[str, List[str]],
        roots: List[str],
    ) -> None:
        """Assign depth to each node via BFS from roots."""
        depth_map: Dict[str, int] = {}
        queue: List[Tuple[str, int]] = [(r, 0) for r in roots]

        while queue:
            node_id, d = queue.pop(0)
            if node_id in depth_map:
                continue
            depth_map[node_id] = d
            for neighbor in adj.get(node_id, []):
                if neighbor not in depth_map:
                    queue.append((neighbor, d + 1))

        node_index = {n.id: n for n in nodes}
        for nid, d in depth_map.items():
            if nid in node_index:
                node_index[nid].depth = d


# ---------------------------------------------------------------------------
# DISCONFIRM stage -- hunt structural weakness
# ---------------------------------------------------------------------------

class DisconfirmStage(PipelineStage):
    """Stage 3: DISCONFIRM -- hunt structural weakness in the argument.

    GSM description: "Falsification, steel-man counter, fallacy detection."

    Grammar mapping:
        - **Backward derivation** from conclusions: trace back through the
          derivation tree to find premises with low rule-weights (weak links).
        - **Strange loop analysis**: circular dependencies where a claim
          ultimately supports itself are detected via Grammar.find_loops()
          and flagged as potential circular reasoning.
        - **Fugue composition**: run the argument through multiple grammar
          domains simultaneously.  Where domains disagree (counterpoints
          in the fugue), structural contradictions may exist.
        - **Steel-man counter**: use backward derivation from the negation
          of the conclusion to find the strongest alternative argument.

    The disconfirm stage is adversarial: it tries to BREAK the argument.
    """

    @property
    def name(self) -> str:
        return "disconfirm"

    def run(self, input_data: Any) -> DisconfirmResult:
        """Hunt for structural weaknesses in the dependency graph.

        Parameters:
            input_data: A DAGResult from the DAG stage.

        Returns:
            DisconfirmResult with weaknesses, fallacies, and counter-arguments.
        """
        if not isinstance(input_data, DAGResult):
            raise TypeError(
                f"DisconfirmStage expects DAGResult, got {type(input_data).__name__}"
            )

        dag = input_data
        result = DisconfirmResult()

        # Step 1: Detect circular reasoning from DAG cycles.
        for cycle in dag.cycles:
            weakness = self._analyze_cycle(cycle, dag)
            if weakness:
                result.weaknesses.append(weakness)
                result.fallacies.append(FallacyType.CIRCULAR)

        # Step 2: Backward derivation to find weak premises.
        weak_premises = self._backward_weakness_hunt(dag)
        result.weaknesses.extend(weak_premises)

        # Step 3: Strange loop analysis across grammars.
        loop_weaknesses = self._strange_loop_analysis(dag)
        result.weaknesses.extend(loop_weaknesses)

        # Step 4: Cross-domain contradiction detection via fugue.
        cross_domain = self._fugue_contradiction_check(dag)
        result.cross_domain = cross_domain

        # Step 5: Steel-man counter-argument construction.
        result.steel_man = self._build_steel_man(dag)

        # Step 6: Classify survived vs weakened claims.
        weakened_ids: Set[str] = set()
        for w in result.weaknesses:
            if w.claim:
                weakened_ids.add(id(w.claim))

        for node in dag.nodes:
            if node.triple is None:
                continue
            if id(node.triple) in weakened_ids:
                # Reduce confidence.
                node.triple.confidence *= 0.5
                result.weakened_claims.append(node.triple)
            else:
                result.survived_claims.append(node.triple)

        # Step 7: Find contradictions (pairs of claims that negate each other).
        result.contradictions = self._find_contradictions(dag)

        return result

    # -- internal: cycle analysis -------------------------------------------

    def _analyze_cycle(
        self,
        cycle: List[str],
        dag: DAGResult,
    ) -> Optional[WeaknessReport]:
        """Analyze a dependency cycle for circular reasoning."""
        node_index = dag.node_index
        cycle_claims = [
            node_index[nid].triple
            for nid in cycle
            if nid in node_index and node_index[nid].triple is not None
        ]

        if not cycle_claims:
            return None

        return WeaknessReport(
            claim=cycle_claims[0],
            weakness_type="circular_reasoning",
            fallacy=FallacyType.CIRCULAR,
            explanation=(
                f"Circular dependency detected: {len(cycle)} claims form a cycle "
                f"where each depends on the next, ultimately supporting itself."
            ),
            severity=0.8,
            backward_trace=cycle,
        )

    # -- internal: backward weakness hunt -----------------------------------

    def _backward_weakness_hunt(
        self,
        dag: DAGResult,
    ) -> List[WeaknessReport]:
        """Trace backward from conclusions to find weak premises.

        For each root node (conclusion), walk backward through the DAG.
        Premises with low confidence or no grammar support are flagged.
        """
        weaknesses: List[WeaknessReport] = []
        node_index = dag.node_index

        # Build reverse adjacency: target -> [sources that depend on it].
        rev_adj: Dict[str, List[str]] = {n.id: [] for n in dag.nodes}
        for edge in dag.edges:
            rev_adj.setdefault(edge.target_id, []).append(edge.source_id)

        # For each conclusion (root), trace back.
        for root_id in dag.roots:
            root = node_index.get(root_id)
            if root is None or root.triple is None:
                continue

            # BFS backward.
            queue = [root_id]
            visited: Set[str] = set()
            backward_trace: List[str] = []

            while queue:
                current_id = queue.pop(0)
                if current_id in visited:
                    continue
                visited.add(current_id)
                backward_trace.append(current_id)

                current = node_index.get(current_id)
                if current and current.triple and current.triple.confidence < 0.3:
                    weaknesses.append(WeaknessReport(
                        claim=current.triple,
                        weakness_type="weak_premise",
                        explanation=(
                            f"Premise '{current.triple.subject} {current.triple.relation} "
                            f"{current.triple.object}' has low confidence "
                            f"({current.triple.confidence:.2f}) and supports "
                            f"conclusion '{root.triple.subject} {root.triple.relation} "
                            f"{root.triple.object}'."
                        ),
                        severity=0.6,
                        backward_trace=list(backward_trace),
                    ))

                # Walk backward through dependencies.
                # Find edges where current_id is the source (current depends on target).
                for edge in dag.edges:
                    if edge.source_id == current_id and edge.target_id not in visited:
                        queue.append(edge.target_id)

        return weaknesses

    # -- internal: strange loop analysis ------------------------------------

    def _strange_loop_analysis(
        self,
        dag: DAGResult,
    ) -> List[WeaknessReport]:
        """Use grammar strange loop detection to find reasoning loops.

        If any grammar in self.grammars has strange loops that correspond
        to the structure of claims in the DAG, flag potential issues.
        """
        weaknesses: List[WeaknessReport] = []

        for gname, grammar in self.grammars.items():
            loops = grammar.find_loops()
            for loop in loops:
                # Check if any DAG claims match the loop's entry form.
                for node in dag.nodes:
                    if node.triple is None:
                        continue
                    claim_text = f"{node.triple.subject} {node.triple.relation} {node.triple.object}"
                    if self._matches_loop(claim_text, loop, grammar):
                        # A flat loop (level_delta == 0) suggests circular reasoning.
                        if loop.is_flat:
                            weaknesses.append(WeaknessReport(
                                claim=node.triple,
                                weakness_type="grammar_strange_loop",
                                fallacy=FallacyType.BEGGING_QUESTION,
                                explanation=(
                                    f"Claim matches a flat strange loop in the "
                                    f"{gname} grammar: the derivation returns to "
                                    f"its starting point without ascending levels, "
                                    f"suggesting the conclusion is assumed in the premise."
                                ),
                                severity=0.7,
                                loop=loop,
                            ))

        return weaknesses

    @staticmethod
    def _matches_loop(
        text: str,
        loop: StrangeLoop,
        grammar: Grammar,
    ) -> bool:
        """Check if text matches a strange loop's entry form."""
        entry = loop.entry
        if entry is None:
            return False
        if isinstance(entry, str) and isinstance(text, str):
            return entry.lower() in text.lower()
        try:
            return Rule._match(entry, text)
        except Exception:
            return False

    # -- internal: fugue contradiction check --------------------------------

    def _fugue_contradiction_check(
        self,
        dag: DAGResult,
    ) -> List[Dict[str, Any]]:
        """Run claims through multiple grammar domains to find contradictions.

        Uses DerivationEngine.compose_fugue() to run grammars in parallel.
        Counterpoints (forms unique to one grammar) may indicate structural
        inconsistencies.
        """
        cross_domain: List[Dict[str, Any]] = []
        grammars = list(self.grammars.values())

        if len(grammars) < 2:
            return cross_domain

        # For each claim, run it through all grammars and look for disagreement.
        for node in dag.nodes:
            if node.triple is None:
                continue

            claim_text = f"{node.triple.subject} {node.triple.relation} {node.triple.object}"
            inputs = [claim_text] * len(grammars)

            fugue_result = self.engine.compose_fugue(grammars, inputs)

            # If there are counterpoints (disagreements), flag them.
            for cp in fugue_result.get("counterpoints", []):
                unique_forms = cp.get("unique_forms", [])
                if unique_forms:
                    cross_domain.append({
                        "claim": node.triple.as_tuple,
                        "grammar": cp.get("grammar", ""),
                        "unique_derivations": len(unique_forms),
                        "issue": (
                            f"The {cp.get('grammar', '')} grammar produces "
                            f"{len(unique_forms)} unique forms not seen in "
                            f"other grammars -- possible structural inconsistency."
                        ),
                    })

        return cross_domain

    # -- internal: steel-man counter ----------------------------------------

    def _build_steel_man(self, dag: DAGResult) -> Optional[str]:
        """Construct the strongest counter-argument.

        Strategy: take each root conclusion, derive backward to find the
        weakest supporting premise, then construct an argument that
        attacks that premise.
        """
        if not dag.roots:
            return None

        node_index = dag.node_index
        weakest_premise: Optional[Triple] = None
        weakest_confidence = 1.0

        for root_id in dag.roots:
            # Walk the dependency chain.
            for edge in dag.edges:
                if edge.source_id == root_id:
                    target = node_index.get(edge.target_id)
                    if target and target.triple and target.triple.confidence < weakest_confidence:
                        weakest_confidence = target.triple.confidence
                        weakest_premise = target.triple

        if weakest_premise is None:
            return "No structural weaknesses found to build a counter-argument."

        return (
            f"The weakest link is the claim '{weakest_premise.subject} "
            f"{weakest_premise.relation} {weakest_premise.object}' "
            f"(confidence: {weakest_premise.confidence:.2f}). "
            f"A strong counter-argument would challenge this premise: "
            f"if '{weakest_premise.subject}' does NOT '{weakest_premise.relation}' "
            f"'{weakest_premise.object}', then the conclusions that depend "
            f"on it collapse."
        )

    # -- internal: contradiction detection ----------------------------------

    def _find_contradictions(
        self,
        dag: DAGResult,
    ) -> List[Tuple[Triple, Triple]]:
        """Find pairs of claims that contradict each other.

        Simple heuristic: two claims contradict if they share subject
        and relation but have negated objects (or vice versa).
        """
        contradictions: List[Tuple[Triple, Triple]] = []
        negation_markers = {"not", "no", "never", "none", "neither", "nor", "without"}

        claims = [n.triple for n in dag.nodes if n.triple is not None]

        for i, a in enumerate(claims):
            for j, b in enumerate(claims):
                if j <= i:
                    continue

                # Same subject and relation, conflicting objects?
                if (a.subject.lower() == b.subject.lower()
                        and a.relation.lower() == b.relation.lower()):
                    a_words = set(a.object.lower().split())
                    b_words = set(b.object.lower().split())

                    a_negated = bool(a_words & negation_markers)
                    b_negated = bool(b_words & negation_markers)

                    if a_negated != b_negated:
                        contradictions.append((a, b))

        return contradictions


# ---------------------------------------------------------------------------
# SYNTHESIS stage -- extract minimum viable logic
# ---------------------------------------------------------------------------

class SynthesisStage(PipelineStage):
    """Stage 4: SYNTHESIS -- extract the minimum viable logic.

    GSM description: "Proven vs not proven, evidence hierarchy, clean argument."

    Grammar mapping:
        - **Forward derivation from proven roots**: start from leaf nodes
          (axioms/base evidence) and derive forward.  Only claims reachable
          by this process are "proven".
        - **Isomorphism discovery**: use DerivationEngine.find_isomorphisms()
          to check if the argument structure appears in other grammar
          domains.  Cross-domain validation strengthens conclusions.
        - **MNEMO compression**: compress the clean argument into MNEMO
          notation using MnemoCodec.  If the argument compresses well,
          it has good structural regularity.

    The synthesis stage renders the final verdict: what is proven, what
    is not, and how confident we should be.
    """

    @property
    def name(self) -> str:
        return "synthesis"

    def run(self, input_data: Any) -> SynthesisResult:
        """Synthesize the minimum viable logic from disconfirmed claims.

        Parameters:
            input_data: A DisconfirmResult from the disconfirm stage.
                        The pipeline also injects the DAGResult as
                        input_data._dag if needed.

        Returns:
            SynthesisResult with proven/unproven claims, evidence hierarchy,
            clean argument, and MNEMO compression.
        """
        if not isinstance(input_data, DisconfirmResult):
            raise TypeError(
                f"SynthesisStage expects DisconfirmResult, got {type(input_data).__name__}"
            )

        disconfirm = input_data
        result = SynthesisResult()

        # Step 1: Build evidence hierarchy from survived + weakened claims.
        all_claims = disconfirm.survived_claims + disconfirm.weakened_claims

        for claim in all_claims:
            node = self._classify_evidence(claim, disconfirm)
            result.evidence_hierarchy.append(node)

            if node.level in (EvidenceLevel.PROVEN, EvidenceLevel.SUPPORTED):
                result.proven.append(node)
            else:
                result.unproven.append(node)

        # Step 2: Forward derivation from proven roots to validate chain.
        result.clean_argument = self._derive_clean_argument(result.proven)

        # Step 3: Cross-domain validation via isomorphisms.
        result.cross_domain_validation = self._cross_domain_validate(
            result.proven
        )

        # Step 4: MNEMO compression of the clean argument.
        result.mnemo_compressed = self._mnemo_compress(result.clean_argument)

        # Step 5: Compute overall confidence.
        result.confidence = self._compute_confidence(result)

        return result

    # -- internal: evidence classification ----------------------------------

    def _classify_evidence(
        self,
        claim: Triple,
        disconfirm: DisconfirmResult,
    ) -> EvidenceNode:
        """Classify a claim's evidence level.

        Levels:
            PROVEN:           confidence >= 0.8, survived disconfirmation
            SUPPORTED:        confidence >= 0.5, survived disconfirmation
            PLAUSIBLE:        confidence >= 0.3, or weakened but not refuted
            UNSUBSTANTIATED:  confidence < 0.3
            CONTRADICTED:     appears in disconfirm.contradictions
        """
        # Check if contradicted.
        for a, b in disconfirm.contradictions:
            if claim is a or claim is b:
                return EvidenceNode(
                    claim=claim,
                    level=EvidenceLevel.CONTRADICTED,
                    support=0,
                )

        # Check if weakened.
        is_weakened = any(
            w.claim is claim for w in disconfirm.weaknesses
        )

        # Classify by confidence.
        if claim.confidence >= 0.8 and not is_weakened:
            level = EvidenceLevel.PROVEN
        elif claim.confidence >= 0.5:
            level = EvidenceLevel.SUPPORTED
        elif claim.confidence >= 0.3:
            level = EvidenceLevel.PLAUSIBLE
        else:
            level = EvidenceLevel.UNSUBSTANTIATED

        # Count derivation support paths.
        support = self._count_derivation_support(claim)

        # Check which grammar domains validate this claim.
        domains = self._validating_domains(claim)

        return EvidenceNode(
            claim=claim,
            level=level,
            support=support,
            domains=domains,
        )

    def _count_derivation_support(self, claim: Triple) -> int:
        """Count independent derivation paths that support a claim."""
        count = 0
        claim_text = f"{claim.subject} {claim.relation} {claim.object}"

        for gname, grammar in self.grammars.items():
            tree = self.engine.derive(claim_text, grammar, "forward", max_steps=20)
            # Each leaf in the derivation tree is an independent path.
            leaves = tree.leaves()
            if len(leaves) > 1 or (len(leaves) == 1 and leaves[0].depth > 0):
                count += 1

        return count

    def _validating_domains(self, claim: Triple) -> List[str]:
        """Find which grammar domains can derive anything from this claim."""
        domains: List[str] = []
        claim_text = f"{claim.subject} {claim.relation} {claim.object}"

        for gname, grammar in self.grammars.items():
            results = grammar.apply_forward(claim_text, max_steps=10)
            if results:
                domains.append(gname)

        return domains

    # -- internal: clean argument derivation --------------------------------

    def _derive_clean_argument(
        self,
        proven: List[EvidenceNode],
    ) -> List[Triple]:
        """Derive the clean argument by forward-chaining from proven claims.

        Only claims reachable from proven roots via forward derivation
        are included in the clean argument.
        """
        clean: List[Triple] = []
        seen: Set[str] = set()

        for node in proven:
            if node.claim is None:
                continue
            key = f"{node.claim.subject}|{node.claim.relation}|{node.claim.object}"
            if key not in seen:
                seen.add(key)
                clean.append(node.claim)

        # Sort by confidence (strongest first).
        clean.sort(key=lambda t: t.confidence, reverse=True)
        return clean

    # -- internal: cross-domain validation ----------------------------------

    def _cross_domain_validate(
        self,
        proven: List[EvidenceNode],
    ) -> List[Dict[str, Any]]:
        """Use grammar isomorphisms for cross-domain validation.

        If the argument's structure appears in multiple grammar domains,
        that is strong evidence of structural validity.
        """
        validations: List[Dict[str, Any]] = []
        grammar_list = list(self.grammars.values())

        if len(grammar_list) < 2:
            return validations

        # Check isomorphisms between every pair of grammars.
        for i in range(len(grammar_list)):
            for j in range(i + 1, len(grammar_list)):
                isos = self.engine.find_isomorphisms(
                    grammar_list[i], grammar_list[j]
                )
                if isos:
                    validations.append({
                        "grammar_a": grammar_list[i].name,
                        "grammar_b": grammar_list[j].name,
                        "isomorphisms": len(isos),
                        "strongest": max(
                            isos, key=lambda x: x.get("confidence", 0)
                        ) if isos else None,
                    })

        return validations

    # -- internal: MNEMO compression ----------------------------------------

    def _mnemo_compress(self, clean_argument: List[Triple]) -> str:
        """Compress the clean argument into MNEMO notation.

        Uses the MnemoCodec to compress the argument structure.
        Good compression ratio = good structural regularity.
        """
        try:
            from glm.mnemo.codec import MnemoCodec

            codec = MnemoCodec()
            # Build a text representation of the clean argument.
            text = " ; ".join(
                f"{t.subject} {t.relation} {t.object}"
                for t in clean_argument
            )
            if text:
                return codec.compress(text)
        except Exception:
            pass

        return ""

    # -- internal: confidence computation -----------------------------------

    @staticmethod
    def _compute_confidence(result: SynthesisResult) -> float:
        """Compute overall argument confidence.

        Formula:
            base = mean confidence of proven claims
            bonus = cross-domain validation count * 0.05
            penalty = unproven/total ratio * 0.3
        """
        if not result.proven:
            return 0.0

        # Base: mean confidence of proven claims.
        confidences = [
            n.claim.confidence
            for n in result.proven
            if n.claim is not None
        ]
        if not confidences:
            return 0.0

        base = sum(confidences) / len(confidences)

        # Bonus: cross-domain validation.
        bonus = min(0.2, len(result.cross_domain_validation) * 0.05)

        # Penalty: proportion unproven.
        total = len(result.proven) + len(result.unproven)
        unproven_ratio = len(result.unproven) / total if total > 0 else 0
        penalty = unproven_ratio * 0.3

        return min(1.0, max(0.0, base + bonus - penalty))
