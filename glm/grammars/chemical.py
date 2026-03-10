"""Chemical grammars — bonding, reactions, and molecular structure.

Chemistry has a grammar as precise as any natural language.  Atoms combine
according to valence rules (the "syntax" of chemistry), reactions transform
molecules via grammatical derivations (reactants → products), and molecular
structure follows phrase-structure-like rules where functional groups are
the "phrases" and bonds are the "relations".

The strange loop: autocatalytic molecules that catalyse their own formation,
closing the loop between product and process — a chemical quine.
"""

from glm.core.grammar import Rule, Production, Grammar, StrangeLoop


# ---------------------------------------------------------------------------
# Bonding grammar
# ---------------------------------------------------------------------------

def build_bonding_grammar() -> Grammar:
    """Build a grammar for chemical bonding — the syntax of chemistry.

    Encodes:
    - Valence rules (how many bonds an atom can form)
    - Electronegativity-driven bond type (ionic, covalent, metallic)
    - Octet rule and its exceptions
    - Orbital hybridisation (sp, sp2, sp3, etc.)
    - Bond polarity and dipole moments

    These are the "agreement rules" of chemistry: just as a verb must
    agree with its subject, atoms must satisfy their valence.
    """

    rules = [
        # --- Valence rules ---
        Rule(
            name="valence_hydrogen",
            pattern={"element": "H", "group": 1},
            result={"valence": 1, "max_bonds": 1,
                    "electron_config": "1s¹"},
            conditions={"period": 1},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="valence_carbon",
            pattern={"element": "C", "group": 14},
            result={"valence": 4, "max_bonds": 4,
                    "hybridisations": ["sp3 (tetrahedral)",
                                       "sp2 (trigonal planar)",
                                       "sp (linear)"],
                    "electron_config": "[He] 2s² 2p²"},
            conditions={"period": 2},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="valence_nitrogen",
            pattern={"element": "N", "group": 15},
            result={"valence": 3, "max_bonds": 4,
                    "lone_pairs": 1,
                    "note": "can form 4 bonds via dative/coordinate bonding"},
            conditions={"period": 2},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="valence_oxygen",
            pattern={"element": "O", "group": 16},
            result={"valence": 2, "max_bonds": 2,
                    "lone_pairs": 2,
                    "electron_config": "[He] 2s² 2p⁴"},
            conditions={"period": 2},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="valence_halogens",
            pattern={"group": 17, "elements": ["F", "Cl", "Br", "I"]},
            result={"valence": 1, "max_bonds": 1,
                    "lone_pairs": 3,
                    "note": "Cl, Br, I can expand octet (d orbitals)"},
            conditions={"electronegativity": "high"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="valence_transition_metals",
            pattern={"block": "d", "elements": "transition metals"},
            result={"variable_valence": True,
                    "common_states": {
                        "Fe": [2, 3], "Cu": [1, 2], "Mn": [2, 4, 7],
                        "Cr": [2, 3, 6], "Co": [2, 3], "Ni": [2],
                    },
                    "coordination_numbers": [4, 6]},
            conditions={"d_orbital_involvement": True},
            weight=0.9,
            direction="bidirectional",
        ),

        # --- Bond type determination ---
        Rule(
            name="ionic_bond_formation",
            pattern={"delta_EN": "> 1.7",
                     "atoms": "metal + non-metal"},
            result={"bond_type": "ionic",
                    "mechanism": "electron transfer",
                    "properties": ["high melting point", "conducts when dissolved",
                                   "crystalline lattice"]},
            conditions={"electronegativity_difference": "large"},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="covalent_bond_formation",
            pattern={"delta_EN": "< 1.7",
                     "atoms": "non-metal + non-metal"},
            result={"bond_type": "covalent",
                    "subtypes": ["nonpolar (ΔEN ≈ 0)", "polar (0 < ΔEN < 1.7)"],
                    "mechanism": "electron sharing"},
            conditions={"electronegativity_difference": "small_to_moderate"},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="metallic_bond_formation",
            pattern={"atoms": "metal + metal"},
            result={"bond_type": "metallic",
                    "mechanism": "delocalised electron sea",
                    "properties": ["conductivity", "malleability", "lustre"]},
            conditions={"both_metallic": True},
            weight=0.85,
            direction="bidirectional",
        ),

        # --- Octet rule ---
        Rule(
            name="octet_rule",
            pattern={"element": "period 2-3 main group"},
            result={"stable_config": "8 electrons in valence shell",
                    "mechanism": "share/gain/lose electrons to reach noble gas config"},
            conditions={"exceptions": ["H (duet)", "B (sextet ok)",
                                       "P, S (expanded octet possible)"]},
            weight=0.95,
            direction="bidirectional",
        ),

        # --- Hybridisation ---
        Rule(
            name="sp3_hybridisation",
            pattern={"steric_number": 4},
            result={"hybridisation": "sp3",
                    "geometry": "tetrahedral",
                    "bond_angle": "109.5°",
                    "examples": ["CH₄", "NH₃ (with lone pair → pyramidal)",
                                 "H₂O (with 2 lone pairs → bent)"]},
            conditions={"electron_domains": 4},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="sp2_hybridisation",
            pattern={"steric_number": 3},
            result={"hybridisation": "sp2",
                    "geometry": "trigonal planar",
                    "bond_angle": "120°",
                    "examples": ["BF₃", "C₂H₄ (ethylene)", "CO₃²⁻"]},
            conditions={"electron_domains": 3},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="sp_hybridisation",
            pattern={"steric_number": 2},
            result={"hybridisation": "sp",
                    "geometry": "linear",
                    "bond_angle": "180°",
                    "examples": ["C₂H₂ (acetylene)", "CO₂", "HCN"]},
            conditions={"electron_domains": 2},
            weight=1.0,
            direction="bidirectional",
        ),
    ]

    productions = [
        Production("Molecule", ["Atom", "Bond", "Atom"], "bonding"),
        Production("Bond", ["SingleBond"], "bonding"),
        Production("Bond", ["DoubleBond"], "bonding"),
        Production("Bond", ["TripleBond"], "bonding"),
        Production("Bond", ["IonicBond"], "bonding"),
        Production("Bond", ["CoordinateBond"], "bonding"),
        Production("Atom", ["Element", "FormalCharge"], "bonding"),
        Production("Atom", ["Element", "LonePairs", "BondingSites"], "bonding"),
    ]

    return Grammar(
        name="chemical_bonding",
        domain="chemistry",
        rules=rules,
        productions=productions,
    )


# ---------------------------------------------------------------------------
# Reaction grammar
# ---------------------------------------------------------------------------

def build_reaction_grammar() -> Grammar:
    """Build a grammar for chemical reactions as transformations.

    Reactions are grammatical derivations: reactants are transformed into
    products according to rules.  Catalysts are the "pragmatic context"
    that enables derivations without being consumed.

    Includes:
    - Reaction type classification (synthesis, decomposition, redox, etc.)
    - Stoichiometric balancing as a grammatical constraint
    - Thermodynamic feasibility rules
    - Kinetic considerations (activation energy, catalysis)
    - Equilibrium as bidirectional derivation
    """

    rules = [
        # --- Reaction types ---
        Rule(
            name="synthesis_reaction",
            pattern={"form": "A + B → AB"},
            result={"type": "combination/synthesis",
                    "examples": [
                        "2H₂ + O₂ → 2H₂O",
                        "N₂ + 3H₂ → 2NH₃ (Haber process)",
                        "Na + Cl → NaCl (with appropriate conditions)",
                    ]},
            conditions={"reactants": "simpler species",
                        "products": "more complex species"},
            weight=0.9,
            direction="forward",
        ),
        Rule(
            name="decomposition_reaction",
            pattern={"form": "AB → A + B"},
            result={"type": "decomposition",
                    "examples": [
                        "2H₂O₂ → 2H₂O + O₂",
                        "CaCO₃ → CaO + CO₂ (thermal decomposition)",
                        "2KClO₃ → 2KCl + 3O₂",
                    ]},
            conditions={"energy_input": "typically required (heat, light, electricity)"},
            weight=0.85,
            direction="forward",
        ),
        Rule(
            name="single_displacement",
            pattern={"form": "A + BC → AC + B"},
            result={"type": "single displacement/substitution",
                    "examples": [
                        "Zn + CuSO₄ → ZnSO₄ + Cu",
                        "Fe + CuCl₂ → FeCl₂ + Cu",
                    ],
                    "condition_rule": "A must be more reactive than B (activity series)"},
            conditions={"reactivity": "A > B in activity series"},
            weight=0.85,
            direction="forward",
        ),
        Rule(
            name="double_displacement",
            pattern={"form": "AB + CD → AD + CB"},
            result={"type": "double displacement/metathesis",
                    "examples": [
                        "AgNO₃ + NaCl → AgCl↓ + NaNO₃",
                        "HCl + NaOH → NaCl + H₂O",
                    ],
                    "driving_forces": ["precipitate formation", "water formation",
                                       "gas evolution"]},
            conditions={"driving_force": "at least one product removed from solution"},
            weight=0.9,
            direction="forward",
        ),
        Rule(
            name="redox_reaction",
            pattern={"form": "oxidation + reduction coupled"},
            result={"type": "redox",
                    "principles": [
                        "oxidation = loss of electrons (OIL)",
                        "reduction = gain of electrons (RIG)",
                        "electrons lost = electrons gained",
                    ],
                    "examples": [
                        "2Fe₂O₃ + 3C → 4Fe + 3CO₂",
                        "Zn + Cu²⁺ → Zn²⁺ + Cu",
                    ]},
            conditions={"electron_transfer": True},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="acid_base_reaction",
            pattern={"form": "HA + BOH → BA + H₂O"},
            result={"type": "neutralisation",
                    "models": {
                        "Arrhenius": "H⁺ + OH⁻ → H₂O",
                        "Bronsted_Lowry": "proton transfer from acid to base",
                        "Lewis": "electron pair donation from base to acid",
                    }},
            conditions={"proton_transfer": True},
            weight=0.95,
            direction="bidirectional",
        ),

        # --- Stoichiometric balancing (grammatical constraint) ---
        Rule(
            name="conservation_of_mass",
            pattern={"constraint": "atoms_in == atoms_out"},
            result={"rule": "coefficients must balance all elements",
                    "method": "systematic: balance metals, then nonmetals, then H, then O"},
            conditions={"law": "Lavoisier"},
            weight=1.0,
            direction="bidirectional",
        ),
        Rule(
            name="charge_balance",
            pattern={"constraint": "total_charge_in == total_charge_out"},
            result={"rule": "net ionic equations must balance charge",
                    "method": "half-reaction method for redox"},
            conditions={"ionic_equation": True},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Thermodynamics ---
        Rule(
            name="gibbs_free_energy",
            pattern={"criterion": "ΔG = ΔH − TΔS"},
            result={"spontaneous_if": "ΔG < 0",
                    "cases": [
                        "ΔH<0, ΔS>0: always spontaneous",
                        "ΔH>0, ΔS<0: never spontaneous",
                        "ΔH<0, ΔS<0: spontaneous at low T",
                        "ΔH>0, ΔS>0: spontaneous at high T",
                    ]},
            conditions={"applies_to": "all_reactions"},
            weight=0.95,
            direction="bidirectional",
        ),

        # --- Catalysis ---
        Rule(
            name="catalysis",
            pattern={"catalyst": "substance that lowers activation energy"},
            result={"effect": "increases rate without being consumed",
                    "types": {
                        "homogeneous": "catalyst in same phase as reactants",
                        "heterogeneous": "catalyst in different phase",
                        "enzymatic": "biological catalyst (protein)",
                    },
                    "mechanism": "provides alternative reaction pathway"},
            conditions={"thermodynamics_unchanged": True,
                        "kinetics_changed": True},
            weight=0.9,
            direction="forward",
        ),

        # --- Equilibrium (bidirectional derivation) ---
        Rule(
            name="chemical_equilibrium",
            pattern={"form": "A + B ⇌ C + D"},
            result={"principle": "rates of forward and reverse reactions equal",
                    "le_chatelier": [
                        "increase [reactant] → shifts right",
                        "increase [product] → shifts left",
                        "increase T → shifts toward endothermic direction",
                        "increase P → shifts toward fewer gas moles",
                    ],
                    "K_expression": "K = [C][D] / [A][B]"},
            conditions={"dynamic_equilibrium": True},
            weight=0.95,
            direction="bidirectional",
        ),

        # --- Organic reaction mechanisms ---
        Rule(
            name="nucleophilic_substitution_sn2",
            pattern={"mechanism": "backside attack, concerted"},
            result={"features": ["bimolecular kinetics (rate = k[Nu][RX])",
                                  "inversion of configuration",
                                  "favoured for primary substrates",
                                  "strong nucleophile required"],
                    "example": "HO⁻ + CH₃Br → CH₃OH + Br⁻"},
            conditions={"substrate": "primary_or_methyl", "nucleophile": "strong"},
            weight=0.9,
            direction="forward",
        ),
        Rule(
            name="electrophilic_addition",
            pattern={"mechanism": "addition across double bond"},
            result={"features": ["Markovnikov regiochemistry",
                                  "carbocation intermediate",
                                  "anti-Markovnikov with peroxides (radical)"],
                    "example": "CH₂=CH₂ + HBr → CH₃CH₂Br"},
            conditions={"substrate": "alkene", "reagent": "electrophile"},
            weight=0.85,
            direction="forward",
        ),
    ]

    productions = [
        Production("Reaction", ["Reactants", "Arrow", "Products"], "reaction"),
        Production("Reactants", ["Species"], "reaction"),
        Production("Reactants", ["Species", "Plus", "Reactants"], "reaction"),
        Production("Products", ["Species"], "reaction"),
        Production("Products", ["Species", "Plus", "Products"], "reaction"),
        Production("Species", ["Coefficient", "Formula"], "reaction"),
        Production("Formula", ["Element", "Subscript"], "reaction"),
        Production("Formula", ["Group", "Subscript"], "reaction"),
        Production("Arrow", ["RightArrow"], "reaction"),
        Production("Arrow", ["EquilibriumArrow"], "reaction"),
    ]

    return Grammar(
        name="chemical_reactions",
        domain="chemistry",
        rules=rules,
        productions=productions,
    )


# ---------------------------------------------------------------------------
# Molecular grammar
# ---------------------------------------------------------------------------

def build_molecular_grammar() -> Grammar:
    """Build a grammar for molecular structure.

    Functional groups are the "phrases" of molecular grammar.  Just as
    NP + VP = S, functional groups combine according to structural rules
    to form molecules.

    Includes:
    - Functional group identification (the "lexical categories" of chemistry)
    - VSEPR geometry (molecular shape from electron domains)
    - Chirality and stereochemistry
    - Aromaticity rules
    - Intermolecular forces

    Strange loop: autocatalytic molecules that catalyse their own formation.
    """

    rules = [
        # --- Functional groups as lexical categories ---
        Rule(
            name="hydroxyl_group",
            pattern={"group": "-OH", "SMARTS": "[OX2H]"},
            result={"category": "alcohol (on carbon) / hydroxide",
                    "properties": ["hydrogen bonding", "polar",
                                   "acidic proton (weakly)"],
                    "reactions": ["oxidation → aldehyde/ketone → carboxylic acid",
                                  "dehydration → alkene",
                                  "esterification with carboxylic acid"]},
            conditions={"attached_to": "carbon or metal"},
            weight=0.95,
            direction="bidirectional",
        ),
        Rule(
            name="carbonyl_group",
            pattern={"group": "C=O", "SMARTS": "[CX3]=[OX1]"},
            result={"subtypes": {
                "aldehyde": "R-CHO (terminal C=O)",
                "ketone": "R-CO-R' (internal C=O)",
            },
                    "properties": ["polar", "nucleophilic addition target",
                                   "IR stretch ~1700 cm⁻¹"],
                    "reactions": ["nucleophilic addition", "reduction → alcohol",
                                  "oxidation (aldehyde → acid)"]},
            conditions={"carbon_bonded_to": "oxygen via double bond"},
            weight=0.95,
            direction="bidirectional",
        ),
        Rule(
            name="carboxyl_group",
            pattern={"group": "-COOH", "SMARTS": "[CX3](=O)[OX2H1]"},
            result={"category": "carboxylic acid",
                    "properties": ["acidic (pKa ~4-5)", "hydrogen bonding (dimer)",
                                   "polar"],
                    "reactions": ["deprotonation → carboxylate",
                                  "esterification", "amide formation",
                                  "decarboxylation"]},
            weight=0.95,
            direction="bidirectional",
        ),
        Rule(
            name="amino_group",
            pattern={"group": "-NH₂", "SMARTS": "[NX3;H2]"},
            result={"category": "primary amine",
                    "properties": ["basic (lone pair on N)", "hydrogen bonding",
                                   "nucleophilic"],
                    "reactions": ["protonation → ammonium",
                                  "acylation → amide",
                                  "alkylation → secondary amine"]},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="ester_group",
            pattern={"group": "-COO-", "SMARTS": "[CX3](=O)[OX2][CX4]"},
            result={"category": "ester",
                    "properties": ["fruity odour", "moderate polarity",
                                   "lower boiling point than acids"],
                    "reactions": ["hydrolysis → acid + alcohol",
                                  "transesterification",
                                  "reduction → alcohol"]},
            weight=0.9,
            direction="bidirectional",
        ),

        # --- VSEPR geometry ---
        Rule(
            name="vsepr_prediction",
            pattern={"input": "central atom + number of bonding and lone pairs"},
            result={"geometries": {
                "(2,0)": "linear (180°)",
                "(3,0)": "trigonal planar (120°)",
                "(3,1)": "trigonal pyramidal (<109.5°)",
                "(4,0)": "tetrahedral (109.5°)",
                "(4,1)": "seesaw",
                "(4,2)": "square planar",
                "(2,2)": "bent (~104.5°)",
                "(5,0)": "trigonal bipyramidal",
                "(6,0)": "octahedral",
            }},
            conditions={"principle": "electron pairs repel to maximise separation"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Chirality ---
        Rule(
            name="chirality_detection",
            pattern={"structure": "carbon with 4 different substituents"},
            result={"chiral": True,
                    "consequences": [
                        "two enantiomers (R and S)",
                        "optical activity (rotate plane-polarised light)",
                        "different biological activity possible",
                    ],
                    "assignment": "Cahn-Ingold-Prelog priority rules"},
            conditions={"sp3_carbon": True, "distinct_groups": 4},
            weight=0.95,
            direction="bidirectional",
        ),
        Rule(
            name="cip_priority_rules",
            pattern={"task": "assign R/S configuration"},
            result={"rules": [
                "1. Higher atomic number = higher priority",
                "2. If tied, compare next atoms outward",
                "3. Double bond = two single bonds to same atom (phantom atoms)",
                "4. Orient lowest priority away; CW = R, CCW = S",
            ]},
            conditions={"chiral_centre": True},
            weight=0.9,
            direction="forward",
        ),

        # --- Aromaticity ---
        Rule(
            name="huckel_aromaticity",
            pattern={"ring": "planar, conjugated, cyclic"},
            result={"aromatic_if": "4n+2 π electrons (Hückel's rule)",
                    "examples": {
                        "benzene": "6 π e⁻ (n=1) → aromatic",
                        "cyclobutadiene": "4 π e⁻ (n=1 → 4n) → antiaromatic",
                        "cyclopentadienyl_anion": "6 π e⁻ → aromatic",
                        "pyridine": "6 π e⁻ → aromatic (N lone pair not in π)",
                    }},
            conditions={"planar": True, "fully_conjugated": True, "cyclic": True},
            weight=0.95,
            direction="bidirectional",
        ),

        # --- Intermolecular forces ---
        Rule(
            name="intermolecular_forces_hierarchy",
            pattern={"molecule": "any"},
            result={"forces_weakest_to_strongest": [
                "London dispersion (all molecules, ∝ surface area)",
                "dipole-dipole (polar molecules)",
                "hydrogen bonding (N-H, O-H, F-H with lone pair acceptor)",
                "ion-dipole (ions + polar molecules)",
            ],
                    "determines": ["boiling point", "solubility", "viscosity"]},
            conditions={"always_present": "London dispersion at minimum"},
            weight=0.9,
            direction="bidirectional",
        ),
    ]

    productions = [
        # Molecular structure as phrase structure
        Production("Molecule", ["Backbone", "FunctionalGroups"], "molecular"),
        Production("Backbone", ["CarbonChain"], "molecular"),
        Production("Backbone", ["Ring"], "molecular"),
        Production("Backbone", ["CarbonChain", "Ring"], "molecular"),
        Production("CarbonChain", ["C", "CarbonChain"], "molecular"),
        Production("CarbonChain", ["C"], "molecular"),
        Production("Ring", ["AromaticRing"], "molecular"),
        Production("Ring", ["AliphaticRing"], "molecular"),
        Production("FunctionalGroups", ["FunctionalGroup"], "molecular"),
        Production("FunctionalGroups", ["FunctionalGroup", "FunctionalGroups"], "molecular"),
        Production("FunctionalGroup", ["Hydroxyl"], "molecular"),
        Production("FunctionalGroup", ["Carbonyl"], "molecular"),
        Production("FunctionalGroup", ["Carboxyl"], "molecular"),
        Production("FunctionalGroup", ["Amino"], "molecular"),
        Production("FunctionalGroup", ["Ester"], "molecular"),
        Production("FunctionalGroup", ["Halide"], "molecular"),
    ]

    # Strange loop: autocatalysis — product catalyses its own formation
    strange_loop = StrangeLoop(
        entry_rule="autocatalytic_cycle",
        cycle=[
            "molecule_A_catalyses_formation_of_B",
            "molecule_B_catalyses_formation_of_C",
            "molecule_C_catalyses_formation_of_A",
        ],
        level_shift="self_sustaining_cycle",
    )

    return Grammar(
        name="molecular_structure",
        domain="chemistry",
        rules=rules,
        productions=productions,
        sub_grammars=[strange_loop],
    )
