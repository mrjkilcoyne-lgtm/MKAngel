"""
Molecular Substrate — the grammar of chemistry.

Chemistry has its own grammar: atoms bond according to valence rules,
molecules fold into shapes dictated by electron geometry, reactions follow
conservation laws.  A SMILES string is a sentence; a reaction is a
transformation rule; the periodic table is the feature system.

The parallels to natural language are not metaphorical — they are
structural isomorphisms:

- Atomic valence ↔ syntactic valence (verb argument structure).
- Molecular formula ↔ morphological template.
- Functional group ↔ morpheme (both modify the properties of a core).
- Chemical reaction ↔ phonological rule (both rewrite structures).
- VSEPR geometry ↔ phrase structure (both are constituency trees).

Strange loops: catalysts that catalyse their own synthesis; autocatalytic
sets; the ribosome — a molecular machine whose own blueprint is encoded
in the molecules it assembles.

Fugues: metabolic pathways where multiple reactions run in parallel,
feeding into each other in counterpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.substrate import Sequence, Substrate, Symbol, TransformationRule


# ---------------------------------------------------------------------------
# Periodic table data (subset)
# ---------------------------------------------------------------------------

ELEMENT_DATA: Dict[str, Dict[str, Any]] = {
    "H":  {"number": 1,  "name": "hydrogen",  "group": 1,  "period": 1, "electronegativity": 2.20, "valence": 1},
    "He": {"number": 2,  "name": "helium",    "group": 18, "period": 1, "electronegativity": 0.0,  "valence": 0},
    "Li": {"number": 3,  "name": "lithium",   "group": 1,  "period": 2, "electronegativity": 0.98, "valence": 1},
    "Be": {"number": 4,  "name": "beryllium", "group": 2,  "period": 2, "electronegativity": 1.57, "valence": 2},
    "B":  {"number": 5,  "name": "boron",     "group": 13, "period": 2, "electronegativity": 2.04, "valence": 3},
    "C":  {"number": 6,  "name": "carbon",    "group": 14, "period": 2, "electronegativity": 2.55, "valence": 4},
    "N":  {"number": 7,  "name": "nitrogen",  "group": 15, "period": 2, "electronegativity": 3.04, "valence": 3},
    "O":  {"number": 8,  "name": "oxygen",    "group": 16, "period": 2, "electronegativity": 3.44, "valence": 2},
    "F":  {"number": 9,  "name": "fluorine",  "group": 17, "period": 2, "electronegativity": 3.98, "valence": 1},
    "Ne": {"number": 10, "name": "neon",      "group": 18, "period": 2, "electronegativity": 0.0,  "valence": 0},
    "Na": {"number": 11, "name": "sodium",    "group": 1,  "period": 3, "electronegativity": 0.93, "valence": 1},
    "Mg": {"number": 12, "name": "magnesium", "group": 2,  "period": 3, "electronegativity": 1.31, "valence": 2},
    "Al": {"number": 13, "name": "aluminium", "group": 13, "period": 3, "electronegativity": 1.61, "valence": 3},
    "Si": {"number": 14, "name": "silicon",   "group": 14, "period": 3, "electronegativity": 1.90, "valence": 4},
    "P":  {"number": 15, "name": "phosphorus","group": 15, "period": 3, "electronegativity": 2.19, "valence": 3},
    "S":  {"number": 16, "name": "sulfur",    "group": 16, "period": 3, "electronegativity": 2.58, "valence": 2},
    "Cl": {"number": 17, "name": "chlorine",  "group": 17, "period": 3, "electronegativity": 3.16, "valence": 1},
    "Ar": {"number": 18, "name": "argon",     "group": 18, "period": 3, "electronegativity": 0.0,  "valence": 0},
    "K":  {"number": 19, "name": "potassium", "group": 1,  "period": 4, "electronegativity": 0.82, "valence": 1},
    "Ca": {"number": 20, "name": "calcium",   "group": 2,  "period": 4, "electronegativity": 1.00, "valence": 2},
    "Fe": {"number": 26, "name": "iron",      "group": 8,  "period": 4, "electronegativity": 1.83, "valence": 3},
    "Cu": {"number": 29, "name": "copper",    "group": 11, "period": 4, "electronegativity": 1.90, "valence": 2},
    "Zn": {"number": 30, "name": "zinc",      "group": 12, "period": 4, "electronegativity": 1.65, "valence": 2},
    "Br": {"number": 35, "name": "bromine",   "group": 17, "period": 4, "electronegativity": 2.96, "valence": 1},
    "I":  {"number": 53, "name": "iodine",    "group": 17, "period": 5, "electronegativity": 2.66, "valence": 1},
}

# Bond types and their orders
BOND_ORDERS = {"-": 1, "=": 2, "#": 3, ":": 1.5}  # single, double, triple, aromatic

# Common functional group SMILES patterns
FUNCTIONAL_GROUPS = {
    "hydroxyl": "O",
    "carbonyl": "C(=O)",
    "carboxyl": "C(=O)O",
    "amine": "N",
    "amide": "C(=O)N",
    "methyl": "C",
    "ethyl": "CC",
    "phenyl": "c1ccccc1",
}


# ---------------------------------------------------------------------------
# Atom — Symbol subclass
# ---------------------------------------------------------------------------

@dataclass
class Atom(Symbol):
    """An atom — the irreducible symbol of chemistry.

    Features encode: element, atomic number, charge, group, period,
    electronegativity, and available bonding capacity (valence).

    The valence of an Atom mirrors both chemical valence (number of bonds
    an element can form) and syntactic valence (number of arguments a verb
    takes).  Carbon is a transitive verb with four argument slots.
    """

    charge: int = 0
    bond_count: int = 0  # how many bonds currently formed

    def __post_init__(self) -> None:
        if self.domain == "generic":
            self.domain = "molecular"
        # Auto-populate features from element data
        elem = ELEMENT_DATA.get(self.form)
        if elem and not self.features:
            self.features = dict(elem)
            self.valence = elem.get("valence", 1)

    @property
    def element(self) -> str:
        return self.form

    @property
    def atomic_number(self) -> int:
        return self.features.get("number", 0)

    @property
    def electronegativity(self) -> float:
        return self.features.get("electronegativity", 0.0)

    @property
    def group(self) -> int:
        return self.features.get("group", 0)

    @property
    def period(self) -> int:
        return self.features.get("period", 0)

    @property
    def remaining_valence(self) -> int:
        """How many more bonds this atom can form."""
        return max(0, self.valence - self.bond_count + abs(self.charge))

    def is_noble_gas(self) -> bool:
        return self.group == 18

    def is_halogen(self) -> bool:
        return self.group == 17

    def is_metal(self) -> bool:
        return self.group in (1, 2) or (3 <= self.group <= 12)

    def can_bond_with(self, other: "Atom") -> bool:
        """Check whether two atoms can form a bond based on valence."""
        return self.remaining_valence > 0 and other.remaining_valence > 0


# ---------------------------------------------------------------------------
# Bond representation
# ---------------------------------------------------------------------------

@dataclass
class Bond:
    """A bond between two atom indices in a molecular sequence."""
    atom_a: int
    atom_b: int
    order: float = 1.0  # 1=single, 2=double, 3=triple, 1.5=aromatic

    def __repr__(self) -> str:
        labels = {1.0: "-", 2.0: "=", 3.0: "#", 1.5: ":"}
        return f"Bond({self.atom_a}{labels.get(self.order, '?')}{self.atom_b})"


# ---------------------------------------------------------------------------
# MolecularSubstrate
# ---------------------------------------------------------------------------

class MolecularSubstrate(Substrate):
    """Substrate for chemical structures — atoms, bonds, molecular grammar.

    Encodes SMILES-like notation into atom Sequences, understands bonding
    rules, detects functional groups (structural motifs), and can validate
    molecular formulae against valence constraints.

    The grammar of chemistry:
    - Alphabet = elements (atoms)
    - Words = molecules (bonded atom sequences)
    - Syntax = valence rules, bonding geometry
    - Morphology = functional groups (modify a molecular "root")
    - Phonology = spectroscopic signatures (the "sound" of a molecule)
    """

    def __init__(self, name: str = "molecular") -> None:
        super().__init__(name, domain="molecular")
        self._bonds: List[Bond] = []
        self._build_inventory()

        # Combination rule: respect valence
        self.add_combination_rule(self._valence_rule)

    def _build_inventory(self) -> None:
        for symbol, data in ELEMENT_DATA.items():
            atom = Atom(form=symbol, features=dict(data), domain="molecular",
                        valence=data.get("valence", 1))
            self.add_symbol(atom)

    @staticmethod
    def _valence_rule(a: Symbol, b: Symbol) -> bool:
        """Two atoms can combine only if both have remaining valence."""
        if isinstance(a, Atom) and isinstance(b, Atom):
            return a.remaining_valence > 0 and b.remaining_valence > 0
        return a.can_bond(b)

    # -- encode / decode (SMILES-like) --------------------------------------

    def encode(self, raw_input: str) -> Sequence:
        """Parse a simplified SMILES-like string into a Sequence of Atoms.

        Supported SMILES features:
        - Single-character elements: C, N, O, S, etc.
        - Bracketed elements: [Na], [Fe], [Cl]
        - Bond symbols: - (single), = (double), # (triple)
        - Branches: parentheses ()
        - Implicit hydrogens are NOT added automatically.

        This is intentionally simplified — a full SMILES parser would need
        ring-closure digits, chirality, etc.  The goal is to demonstrate
        the grammatical structure.
        """
        atoms: List[Symbol] = []
        bonds: List[Bond] = []
        stack: List[int] = []  # branch stack of atom indices
        i = 0
        prev_idx: Optional[int] = None
        pending_bond_order: float = 1.0

        while i < len(raw_input):
            ch = raw_input[i]

            # Skip whitespace
            if ch in (" ", "\t"):
                i += 1
                continue

            # Bond symbol
            if ch in BOND_ORDERS:
                pending_bond_order = BOND_ORDERS[ch]
                i += 1
                continue

            # Branch open
            if ch == "(":
                if prev_idx is not None:
                    stack.append(prev_idx)
                i += 1
                continue

            # Branch close
            if ch == ")":
                if stack:
                    prev_idx = stack.pop()
                i += 1
                continue

            # Bracketed element: [XX]
            if ch == "[":
                end = raw_input.find("]", i)
                if end < 0:
                    end = len(raw_input)
                elem_str = raw_input[i + 1 : end]
                # Parse charge if present
                charge = 0
                clean_elem = elem_str
                if "+" in elem_str:
                    parts = elem_str.split("+")
                    clean_elem = parts[0]
                    charge = int(parts[1]) if len(parts) > 1 and parts[1] else 1
                elif "-" in elem_str and elem_str[0] != "-":
                    parts = elem_str.split("-")
                    clean_elem = parts[0]
                    charge = -(int(parts[1]) if len(parts) > 1 and parts[1] else 1)

                atom = self._make_atom(clean_elem, charge)
                idx = len(atoms)
                atoms.append(atom)
                if prev_idx is not None:
                    bonds.append(Bond(prev_idx, idx, pending_bond_order))
                    pending_bond_order = 1.0
                prev_idx = idx
                i = end + 1
                continue

            # Two-character element (uppercase + lowercase)
            if ch.isupper() and i + 1 < len(raw_input) and raw_input[i + 1].islower():
                elem_str = raw_input[i : i + 2]
                atom = self._make_atom(elem_str)
                idx = len(atoms)
                atoms.append(atom)
                if prev_idx is not None:
                    bonds.append(Bond(prev_idx, idx, pending_bond_order))
                    pending_bond_order = 1.0
                prev_idx = idx
                i += 2
                continue

            # Single-character element
            if ch.isupper():
                atom = self._make_atom(ch)
                idx = len(atoms)
                atoms.append(atom)
                if prev_idx is not None:
                    bonds.append(Bond(prev_idx, idx, pending_bond_order))
                    pending_bond_order = 1.0
                prev_idx = idx
                i += 1
                continue

            # Skip digits (ring closures — not fully supported)
            i += 1

        self._bonds = bonds
        return Sequence(atoms)

    def decode(self, sequence: Sequence) -> str:
        """Convert a Sequence of Atoms back to a SMILES-like string.

        Produces a linear sequence of element symbols.  Does not reconstruct
        branch notation or ring closures.
        """
        return "".join(s.form for s in sequence)

    def _make_atom(self, element: str, charge: int = 0) -> Atom:
        """Create an Atom from an element symbol."""
        data = ELEMENT_DATA.get(element, {})
        features = dict(data) if data else {"name": element.lower()}
        return Atom(
            form=element,
            features=features,
            domain="molecular",
            valence=data.get("valence", 1),
            charge=charge,
        )

    # -- bonding and valence ------------------------------------------------

    @property
    def bonds(self) -> List[Bond]:
        return list(self._bonds)

    def add_bond(self, atom_a_idx: int, atom_b_idx: int,
                 order: float = 1.0) -> Bond:
        """Manually add a bond between two atoms."""
        bond = Bond(atom_a_idx, atom_b_idx, order)
        self._bonds.append(bond)
        return bond

    def validate_valence(self, sequence: Sequence) -> List[str]:
        """Check that no atom exceeds its valence capacity.

        Returns a list of violation descriptions (empty = valid).
        This is the chemical analog of syntactic well-formedness checking.
        """
        violations: List[str] = []
        bond_counts: Dict[int, float] = {}
        for bond in self._bonds:
            bond_counts[bond.atom_a] = bond_counts.get(bond.atom_a, 0) + bond.order
            bond_counts[bond.atom_b] = bond_counts.get(bond.atom_b, 0) + bond.order

        for idx, sym in enumerate(sequence):
            if isinstance(sym, Atom):
                used = bond_counts.get(idx, 0)
                max_val = sym.valence + abs(sym.charge)
                if used > max_val:
                    violations.append(
                        f"Atom {idx} ({sym.form}): {used} bonds exceed "
                        f"valence {max_val}"
                    )
        return violations

    def molecular_formula(self, sequence: Sequence) -> str:
        """Compute the molecular formula (e.g., 'C6H12O6').

        Elements are sorted in Hill order: C first, H second, then
        alphabetical.
        """
        counts: Dict[str, int] = {}
        for sym in sequence:
            counts[sym.form] = counts.get(sym.form, 0) + 1

        # Hill system: C first, H second, rest alphabetical
        parts: List[str] = []
        ordered = []
        if "C" in counts:
            ordered.append(("C", counts.pop("C")))
        if "H" in counts:
            ordered.append(("H", counts.pop("H")))
        for elem in sorted(counts):
            ordered.append((elem, counts[elem]))

        for elem, count in ordered:
            parts.append(elem + (str(count) if count > 1 else ""))
        return "".join(parts)

    # -- functional group detection -----------------------------------------

    def detect_functional_groups(
        self, sequence: Sequence
    ) -> List[Tuple[str, int, int]]:
        """Find known functional groups (structural motifs) in a molecule.

        Returns (group_name, start_index, end_index) triples.
        This is the chemical analog of morpheme detection.
        """
        results: List[Tuple[str, int, int]] = []
        forms = [s.form for s in sequence]
        n = len(forms)

        # Pattern-based detection on the linear sequence
        patterns: Dict[str, List[str]] = {
            "hydroxyl": ["O", "H"],
            "carbonyl": ["C", "O"],  # simplified
            "amine": ["N", "H"],
            "carboxyl": ["C", "O", "O"],
            "methyl": ["C", "H", "H", "H"],
        }

        for group_name, pattern in patterns.items():
            plen = len(pattern)
            for i in range(n - plen + 1):
                if forms[i : i + plen] == pattern:
                    # For carbonyl, verify the C=O bond exists
                    if group_name == "carbonyl":
                        has_double = any(
                            b for b in self._bonds
                            if {b.atom_a, b.atom_b} == {i, i + 1}
                            and b.order == 2.0
                        )
                        if not has_double:
                            continue
                    results.append((group_name, i, i + plen))
        return results

    # -- molecular similarity -----------------------------------------------

    def tanimoto_similarity(
        self, seq_a: Sequence, seq_b: Sequence
    ) -> float:
        """Compute Tanimoto coefficient based on element composition.

        A simple fingerprint-based similarity — counts shared elements.
        """
        def _fingerprint(seq: Sequence) -> Dict[str, int]:
            fp: Dict[str, int] = {}
            for s in seq:
                fp[s.form] = fp.get(s.form, 0) + 1
            return fp

        fp_a = _fingerprint(seq_a)
        fp_b = _fingerprint(seq_b)
        all_keys = set(fp_a) | set(fp_b)
        if not all_keys:
            return 1.0

        intersection = sum(
            min(fp_a.get(k, 0), fp_b.get(k, 0)) for k in all_keys
        )
        union = sum(
            max(fp_a.get(k, 0), fp_b.get(k, 0)) for k in all_keys
        )
        return intersection / union if union > 0 else 0.0

    # -- molecular weight ---------------------------------------------------

    # Approximate atomic weights for common elements
    ATOMIC_WEIGHTS: Dict[str, float] = {
        "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
        "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
        "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086,
        "P": 30.974, "S": 32.065, "Cl": 35.453, "Ar": 39.948,
        "K": 39.098, "Ca": 40.078, "Fe": 55.845, "Cu": 63.546,
        "Zn": 65.38, "Br": 79.904, "I": 126.904,
    }

    def molecular_weight(self, sequence: Sequence) -> float:
        """Compute the molecular weight of a sequence."""
        return sum(
            self.ATOMIC_WEIGHTS.get(s.form, 0.0) for s in sequence
        )

    # -- electronegativity & bond polarity ----------------------------------

    def bond_polarity(self, bond: Bond, sequence: Sequence) -> float:
        """Compute the electronegativity difference across a bond.

        Large difference = ionic character; small = covalent.
        """
        a = sequence[bond.atom_a]
        b = sequence[bond.atom_b]
        en_a = a.features.get("electronegativity", 0.0)
        en_b = b.features.get("electronegativity", 0.0)
        return abs(en_a - en_b)

    # -- alignment override -------------------------------------------------

    def align(
        self,
        seq_a: Sequence,
        seq_b: Sequence,
        **kwargs: Any,
    ) -> Tuple[List[Optional[Symbol]], List[Optional[Symbol]], float]:
        """Molecular alignment using element-aware scoring.

        Same element = strong match; same group = partial match.
        """
        kwargs.setdefault("match_score", 3.0)
        kwargs.setdefault("mismatch_penalty", -1.5)
        kwargs.setdefault("gap_penalty", -1.0)
        kwargs.setdefault("feature_weight", 1.0)
        return Sequence.align(seq_a, seq_b, **kwargs)

    # -- strange loop: autocatalysis ----------------------------------------

    def detect_self_reference(
        self, sequence: Sequence
    ) -> List[Tuple[int, int, str]]:
        """Detect autocatalytic / self-referential structures.

        In chemistry, self-reference manifests as:
        1. Repeating structural motifs (polymers, crystals).
        2. Functional groups that catalyse the formation of the
           same functional group (autocatalysis).
        3. Template-directed synthesis (DNA replication).
        """
        loops: List[Tuple[int, int, str]] = []

        # Detect repeating units (polymer-like self-similarity)
        patterns = self.find_patterns(sequence, min_length=2)
        for pat, positions in patterns.items():
            if len(positions) >= 2:
                pat_len = len(pat.split())
                loops.append((
                    positions[0], positions[0] + pat_len,
                    f"Repeating unit [{pat}] occurs {len(positions)} times "
                    f"(polymer/crystal self-similarity)"
                ))

        # Detect palindromic sequences (like DNA palindromes)
        forms = [s.form for s in sequence]
        n = len(forms)
        for length in range(2, n // 2 + 1):
            for start in range(n - 2 * length + 1):
                forward = forms[start : start + length]
                reverse = forms[start + length : start + 2 * length][::-1]
                if forward == reverse:
                    loops.append((
                        start, start + 2 * length,
                        f"Palindromic structure: {forward} mirrors {reverse}"
                    ))

        return loops
