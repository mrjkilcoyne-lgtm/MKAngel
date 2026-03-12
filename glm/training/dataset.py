"""Training data generation from grammars.

The GLM does not learn from raw text corpora.  Instead it learns from
*grammatical derivations* -- sequences of rule applications that
demonstrate how grammars work.  This module generates training examples
by running the derivation engine over the GLM's grammar library.

Four kinds of training examples are generated:

1. **Forward derivation** -- input sequence with target = next symbol
   in the derivation chain.  Teaches the model to predict forward.

2. **Backward derivation** -- output sequence with target = the
   preceding symbol.  Teaches reconstruction / parsing.

3. **Cross-domain isomorphism pairs** -- two derivation sequences from
   different substrates that share the same structural pattern (same
   production rule shape, different surface symbols).  Teaches the
   model to detect structural parallels.

4. **Self-referential examples** -- sequences that contain strange
   loops (palindromes, quines, fixed-point patterns).  Teaches the
   model when to fire its loop-detection mechanism.

All generated data uses integer symbol IDs in the range
[0, vocab_size), suitable for direct consumption by
GrammarLanguageModel.forward().
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# =======================================================================
# Training example data structures
# =======================================================================

@dataclass
class TrainingExample:
    """A single training example for the GLM.

    Attributes
    ----------
    input_ids : list[int]
        Input symbol sequence.
    target_ids : list[int]
        Target symbols for forward prediction (input shifted right by 1).
    prev_ids : list[int]
        Target symbols for backward reconstruction (input shifted left by 1).
    substrate_id : int
        Which substrate domain this example comes from.
    derivation_depths : list[int]
        Per-position derivation depth in the grammar tree.
    has_strange_loop : bool
        Whether this example contains a self-referential pattern.
    rule_lhs : int or None
        LHS of the grammar rule that generated this example.
    rule_body : list[int] or None
        Body of the generating rule.
    """

    input_ids: List[int] = field(default_factory=list)
    target_ids: List[int] = field(default_factory=list)
    prev_ids: List[int] = field(default_factory=list)
    substrate_id: int = 0
    derivation_depths: List[int] = field(default_factory=list)
    has_strange_loop: bool = False
    rule_lhs: Optional[int] = None
    rule_body: Optional[List[int]] = None


@dataclass
class IsomorphismPair:
    """A pair of training examples from different substrates that
    share the same grammatical structure.

    Used for contrastive isomorphism training.
    """

    anchor: TrainingExample
    positive: TrainingExample  # same structure, different substrate
    negative: Optional[TrainingExample] = None  # different structure


# =======================================================================
# GrammarDataset
# =======================================================================

class GrammarDataset:
    """Generates training data from grammar rules.

    The dataset is generated procedurally -- no files on disk.  Each
    call to ``generate()`` produces a fresh batch of examples by
    running the grammar rules.

    Parameters
    ----------
    vocab_size : int
        Total vocabulary size (symbol IDs will be in [0, vocab_size)).
    num_substrates : int
        Number of substrate domains.
    max_seq_len : int
        Maximum sequence length for generated examples.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        num_substrates: int = 8,
        max_seq_len: int = 64,
        seed: Optional[int] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.num_substrates = num_substrates
        self.max_seq_len = max_seq_len
        self._rng = random.Random(seed)

    # -------------------------------------------------------------------
    # Main generation entry point
    # -------------------------------------------------------------------

    def generate(
        self,
        num_examples: int = 500,
        rules: Optional[List[Tuple[int, List[int]]]] = None,
        forward_ratio: float = 0.35,
        backward_ratio: float = 0.25,
        isomorphism_ratio: float = 0.20,
        loop_ratio: float = 0.20,
    ) -> Dict[str, Any]:
        """Generate a full training dataset.

        Parameters
        ----------
        num_examples : int
            Total number of examples to generate.
        rules : list of (lhs, body) tuples, optional
            Grammar rules to use.  If *None*, generates synthetic rules.
        forward_ratio : float
            Fraction of examples for forward derivation.
        backward_ratio : float
            Fraction for backward derivation.
        isomorphism_ratio : float
            Fraction for isomorphism pairs.
        loop_ratio : float
            Fraction for strange-loop examples.

        Returns
        -------
        dict with keys:
            ``forward``: list[TrainingExample]
            ``backward``: list[TrainingExample]
            ``isomorphism``: list[IsomorphismPair]
            ``loop``: list[TrainingExample]
            ``all``: list[TrainingExample] (all examples flattened)
            ``rules``: the grammar rules used
        """
        if rules is None:
            rules = self._generate_synthetic_rules()

        n_fwd = int(num_examples * forward_ratio)
        n_bwd = int(num_examples * backward_ratio)
        n_iso = int(num_examples * isomorphism_ratio)
        n_loop = num_examples - n_fwd - n_bwd - n_iso

        fwd = self._generate_forward(rules, n_fwd)
        bwd = self._generate_backward(rules, n_bwd)
        iso = self._generate_isomorphism_pairs(rules, n_iso)
        loops = self._generate_loop_examples(rules, n_loop)

        # Flatten all examples for the trainer
        all_examples: List[TrainingExample] = []
        all_examples.extend(fwd)
        all_examples.extend(bwd)
        for pair in iso:
            all_examples.append(pair.anchor)
            all_examples.append(pair.positive)
            if pair.negative:
                all_examples.append(pair.negative)
        all_examples.extend(loops)

        self._rng.shuffle(all_examples)

        return {
            "forward": fwd,
            "backward": bwd,
            "isomorphism": iso,
            "loop": loops,
            "all": all_examples,
            "rules": rules,
        }

    # -------------------------------------------------------------------
    # Forward derivation examples
    # -------------------------------------------------------------------

    def _generate_forward(
        self,
        rules: List[Tuple[int, List[int]]],
        count: int,
    ) -> List[TrainingExample]:
        """Generate forward-derivation training examples.

        Each example is a sequence derived by applying grammar rules
        forward.  The target is the next symbol in the derivation.
        """
        examples: List[TrainingExample] = []

        for _ in range(count):
            seq, depths, lhs, body = self._derive_sequence(rules)
            if len(seq) < 2:
                continue

            # Target: next symbol (teacher forcing)
            target_ids = seq[1:] + [seq[0]]  # wrap around

            substrate = self._rng.randint(0, self.num_substrates - 1)

            examples.append(TrainingExample(
                input_ids=seq,
                target_ids=target_ids,
                prev_ids=[],
                substrate_id=substrate,
                derivation_depths=depths,
                has_strange_loop=False,
                rule_lhs=lhs,
                rule_body=body,
            ))

        return examples

    # -------------------------------------------------------------------
    # Backward derivation examples
    # -------------------------------------------------------------------

    def _generate_backward(
        self,
        rules: List[Tuple[int, List[int]]],
        count: int,
    ) -> List[TrainingExample]:
        """Generate backward-reconstruction training examples.

        The input is a derived sequence; the target is the *preceding*
        symbol at each position.
        """
        examples: List[TrainingExample] = []

        for _ in range(count):
            seq, depths, lhs, body = self._derive_sequence(rules)
            if len(seq) < 2:
                continue

            # Previous IDs: shift left
            prev_ids = [seq[-1]] + seq[:-1]
            substrate = self._rng.randint(0, self.num_substrates - 1)

            examples.append(TrainingExample(
                input_ids=seq,
                target_ids=[],
                prev_ids=prev_ids,
                substrate_id=substrate,
                derivation_depths=depths,
                has_strange_loop=False,
                rule_lhs=lhs,
                rule_body=body,
            ))

        return examples

    # -------------------------------------------------------------------
    # Isomorphism pair examples
    # -------------------------------------------------------------------

    def _generate_isomorphism_pairs(
        self,
        rules: List[Tuple[int, List[int]]],
        count: int,
    ) -> List[IsomorphismPair]:
        """Generate cross-domain isomorphism training pairs.

        For each pair:
        - Anchor and positive share the same derivation structure
          (same rule sequence) but use symbols from different substrates
          (shifted by an offset in symbol space).
        - Negative has a different structure entirely.
        """
        pairs: List[IsomorphismPair] = []

        for _ in range(count):
            # Generate anchor
            seq_a, depths_a, lhs_a, body_a = self._derive_sequence(rules)
            if len(seq_a) < 2:
                continue

            sub_a = self._rng.randint(0, self.num_substrates - 1)
            sub_b = (sub_a + self._rng.randint(1, self.num_substrates - 1)) % self.num_substrates

            # Positive: same structure, different substrate
            # Apply a systematic offset to symbol IDs to simulate
            # the same grammar operating on a different substrate
            offset = self._rng.randint(20, self.vocab_size // 2)
            seq_b = [(s + offset) % self.vocab_size for s in seq_a]
            depths_b = list(depths_a)

            anchor = TrainingExample(
                input_ids=seq_a,
                target_ids=seq_a[1:] + [seq_a[0]],
                substrate_id=sub_a,
                derivation_depths=depths_a,
                rule_lhs=lhs_a,
                rule_body=body_a,
            )
            positive = TrainingExample(
                input_ids=seq_b,
                target_ids=seq_b[1:] + [seq_b[0]],
                substrate_id=sub_b,
                derivation_depths=depths_b,
                rule_lhs=(lhs_a + offset) % self.vocab_size if lhs_a is not None else None,
                rule_body=[(b + offset) % self.vocab_size for b in body_a] if body_a else None,
            )

            # Negative: different structure
            neg_seq, neg_depths, neg_lhs, neg_body = self._derive_sequence(rules)
            # Shuffle to break any accidental structural similarity
            neg_seq_shuffled = list(neg_seq)
            self._rng.shuffle(neg_seq_shuffled)
            sub_c = self._rng.randint(0, self.num_substrates - 1)

            negative = TrainingExample(
                input_ids=neg_seq_shuffled,
                target_ids=neg_seq_shuffled[1:] + [neg_seq_shuffled[0]] if len(neg_seq_shuffled) > 1 else neg_seq_shuffled,
                substrate_id=sub_c,
                derivation_depths=neg_depths[:len(neg_seq_shuffled)],
                has_strange_loop=False,
            )

            pairs.append(IsomorphismPair(
                anchor=anchor,
                positive=positive,
                negative=negative,
            ))

        return pairs

    # -------------------------------------------------------------------
    # Strange-loop examples
    # -------------------------------------------------------------------

    def _generate_loop_examples(
        self,
        rules: List[Tuple[int, List[int]]],
        count: int,
    ) -> List[TrainingExample]:
        """Generate examples for strange-loop detection.

        Half are genuine loops (palindromes, repeated motifs, quine-like
        patterns); half are non-loops (random sequences) for balanced
        training.
        """
        examples: List[TrainingExample] = []
        n_positive = count // 2
        n_negative = count - n_positive

        # Positive: genuine loops
        for _ in range(n_positive):
            pattern_type = self._rng.choice(["palindrome", "repeat", "cycle", "fixedpoint"])
            seq = self._make_loop_sequence(pattern_type)
            depths = [0] * len(seq)

            examples.append(TrainingExample(
                input_ids=seq,
                target_ids=seq[1:] + [seq[0]] if len(seq) > 1 else seq,
                prev_ids=[seq[-1]] + seq[:-1] if len(seq) > 1 else seq,
                substrate_id=self._rng.randint(0, self.num_substrates - 1),
                derivation_depths=depths,
                has_strange_loop=True,
            ))

        # Negative: non-loops (random sequences)
        for _ in range(n_negative):
            length = self._rng.randint(4, min(16, self.max_seq_len))
            seq = [self._rng.randint(0, self.vocab_size - 1) for _ in range(length)]
            # Make sure it is NOT accidentally a palindrome or repeat
            if seq == seq[::-1] or self._is_repeated(seq):
                seq[-1] = (seq[-1] + 1) % self.vocab_size

            depths = [0] * len(seq)

            examples.append(TrainingExample(
                input_ids=seq,
                target_ids=seq[1:] + [seq[0]] if len(seq) > 1 else seq,
                prev_ids=[seq[-1]] + seq[:-1] if len(seq) > 1 else seq,
                substrate_id=self._rng.randint(0, self.num_substrates - 1),
                derivation_depths=depths,
                has_strange_loop=False,
            ))

        return examples

    # -------------------------------------------------------------------
    # Derivation helpers
    # -------------------------------------------------------------------

    def _derive_sequence(
        self,
        rules: List[Tuple[int, List[int]]],
    ) -> Tuple[List[int], List[int], Optional[int], Optional[List[int]]]:
        """Derive a single training sequence from grammar rules.

        Picks a random rule, expands its body by applying additional
        rules, and tracks derivation depths.

        Returns (sequence, depths, lhs, body).
        """
        if not rules:
            length = self._rng.randint(3, 8)
            seq = [self._rng.randint(0, self.vocab_size - 1) for _ in range(length)]
            return seq, [0] * length, None, None

        lhs, body = self._rng.choice(rules)
        seq = list(body)
        depths = [1] * len(seq)

        target_len = self._rng.randint(4, min(20, self.max_seq_len))
        attempts = 0

        while len(seq) < target_len and attempts < 50:
            # Find expandable symbols
            expandable = [
                (i, s)
                for i, s in enumerate(seq)
                if any(r[0] == s for r in rules)
            ]
            if not expandable:
                break

            idx, sym = self._rng.choice(expandable)
            matching = [r for r in rules if r[0] == sym]
            _, expansion = self._rng.choice(matching)
            current_depth = depths[idx]

            seq = seq[:idx] + list(expansion) + seq[idx + 1:]
            new_depths = [current_depth + 1] * len(expansion)
            depths = depths[:idx] + new_depths + depths[idx + 1:]
            attempts += 1

        # Truncate
        seq = seq[:self.max_seq_len]
        depths = depths[:self.max_seq_len]

        # Clamp to vocab range
        seq = [s % self.vocab_size for s in seq]

        if len(seq) < 2:
            seq = list(body[:self.max_seq_len])
            if len(seq) < 2:
                seq = [lhs % self.vocab_size, body[0] % self.vocab_size if body else 0]
            depths = [1] * len(seq)

        return seq, depths, lhs, list(body)

    def _make_loop_sequence(self, pattern_type: str) -> List[int]:
        """Create a sequence that contains a self-referential pattern."""
        if pattern_type == "palindrome":
            half_len = self._rng.randint(2, 6)
            half = [self._rng.randint(0, self.vocab_size - 1) for _ in range(half_len)]
            return half + half[::-1]

        elif pattern_type == "repeat":
            motif_len = self._rng.randint(2, 4)
            motif = [self._rng.randint(0, self.vocab_size - 1) for _ in range(motif_len)]
            repeats = self._rng.randint(2, 4)
            return (motif * repeats)[:self.max_seq_len]

        elif pattern_type == "cycle":
            # A B C A B C -- cyclic repetition
            cycle_len = self._rng.randint(3, 5)
            cycle = [self._rng.randint(0, self.vocab_size - 1) for _ in range(cycle_len)]
            reps = self._rng.randint(2, 3)
            return (cycle * reps)[:self.max_seq_len]

        elif pattern_type == "fixedpoint":
            # A sequence where applying a simple transformation returns
            # the same sequence (e.g., all same symbol, or alternating)
            sym = self._rng.randint(0, self.vocab_size - 1)
            length = self._rng.randint(4, 8)
            return [sym] * length

        # Fallback
        length = self._rng.randint(4, 8)
        half = [self._rng.randint(0, self.vocab_size - 1) for _ in range(length // 2)]
        return half + half[::-1]

    def _is_repeated(self, seq: List[int]) -> bool:
        """Check if a sequence is a repetition of a shorter motif."""
        n = len(seq)
        for period in range(1, n // 2 + 1):
            if n % period == 0:
                motif = seq[:period]
                if all(seq[i] == motif[i % period] for i in range(n)):
                    return True
        return False

    def _generate_synthetic_rules(self) -> List[Tuple[int, List[int]]]:
        """Generate a set of synthetic grammar rules.

        Creates rules that mimic real grammar patterns:
        - Simple expansions (S -> A B)
        - Recursive rules (S -> A S)
        - Chain rules (A -> B -> C)
        - Branching rules (S -> A B | C D)
        """
        rules: List[Tuple[int, List[int]]] = []

        # Reserve low IDs for non-terminals, high for terminals
        nt_range = range(0, 30)
        t_range = range(30, min(100, self.vocab_size))

        # Simple expansion rules
        for _ in range(10):
            lhs = self._rng.choice(nt_range)
            body_len = self._rng.randint(2, 4)
            body = [self._rng.choice(list(nt_range) + list(t_range)) for _ in range(body_len)]
            rules.append((lhs, body))

        # Recursive rules (self-referential non-terminals)
        for _ in range(5):
            lhs = self._rng.choice(nt_range)
            pre = [self._rng.choice(t_range) for _ in range(self._rng.randint(1, 2))]
            post = [self._rng.choice(t_range) for _ in range(self._rng.randint(0, 1))]
            rules.append((lhs, pre + [lhs] + post))

        # Chain rules
        for _ in range(5):
            a = self._rng.choice(nt_range)
            b = self._rng.choice(nt_range)
            c = self._rng.choice(t_range)
            rules.append((a, [b]))
            rules.append((b, [c]))

        # Terminal-only rules
        for _ in range(5):
            lhs = self._rng.choice(nt_range)
            terminals = [self._rng.choice(t_range) for _ in range(self._rng.randint(1, 3))]
            rules.append((lhs, terminals))

        return rules

    # -------------------------------------------------------------------
    # Batch generation
    # -------------------------------------------------------------------

    def generate_batch(
        self,
        examples: List[TrainingExample],
        batch_size: int,
    ) -> List[List[TrainingExample]]:
        """Split examples into batches."""
        indices = list(range(len(examples)))
        self._rng.shuffle(indices)

        batches: List[List[TrainingExample]] = []
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batches.append([examples[i] for i in batch_idx])

        return batches

    def train_val_split(
        self,
        examples: List[TrainingExample],
        val_ratio: float = 0.1,
    ) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """Split examples into training and validation sets."""
        n = len(examples)
        n_val = max(1, int(n * val_ratio))
        indices = list(range(n))
        self._rng.shuffle(indices)
        val_indices = set(indices[:n_val])

        train = [examples[i] for i in range(n) if i not in val_indices]
        val = [examples[i] for i in range(n) if i in val_indices]

        return train, val
