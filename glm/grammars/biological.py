"""Biological encoding grammars — genetics, proteins, and evolution.

Life is a grammar.  The genetic code is a formal language with codons as
morphemes, reading frames as syntactic structure, and regulatory sequences
as pragmatic markers.  Protein folding follows derivation rules from
primary sequence to tertiary structure.  Evolution is the ultimate
temporal grammar — predicting forward and reconstructing backward.

The strange loop at the heart of all biology:
    DNA → RNA → Protein → (regulates) → DNA
The central dogma, except it is not a line but a circle.
"""

from glm.core.grammar import Rule, Production, Grammar, StrangeLoop


# ---------------------------------------------------------------------------
# Genetic grammar
# ---------------------------------------------------------------------------

def build_genetic_grammar() -> Grammar:
    """Build a formal grammar for the genetic code.

    The genetic code maps triplet codons to amino acids — a formal
    grammar with 64 productions and 20 terminal symbols (plus stop).

    Includes:
    - Codon → amino acid mapping (the translation table)
    - Reading frame rules
    - Start and stop codon semantics
    - Splicing rules (intron/exon boundaries)
    - Degeneracy patterns (wobble position)
    - Regulatory grammar (promoters, enhancers, silencers)
    """

    rules = [
        # --- The genetic code (standard) ---
        Rule(
            name="codon_table_standard",
            pattern={"codon_triplet": "NNN (any 3 nucleotides from {A,U,G,C})"},
            result={"mapping": {
                # Phenylalanine
                "UUU": "Phe", "UUC": "Phe",
                # Leucine
                "UUA": "Leu", "UUG": "Leu",
                "CUU": "Leu", "CUC": "Leu", "CUA": "Leu", "CUG": "Leu",
                # Isoleucine
                "AUU": "Ile", "AUC": "Ile", "AUA": "Ile",
                # Methionine (start)
                "AUG": "Met (START)",
                # Valine
                "GUU": "Val", "GUC": "Val", "GUA": "Val", "GUG": "Val",
                # Serine
                "UCU": "Ser", "UCC": "Ser", "UCA": "Ser", "UCG": "Ser",
                "AGU": "Ser", "AGC": "Ser",
                # Proline
                "CCU": "Pro", "CCC": "Pro", "CCA": "Pro", "CCG": "Pro",
                # Threonine
                "ACU": "Thr", "ACC": "Thr", "ACA": "Thr", "ACG": "Thr",
                # Alanine
                "GCU": "Ala", "GCC": "Ala", "GCA": "Ala", "GCG": "Ala",
                # Tyrosine
                "UAU": "Tyr", "UAC": "Tyr",
                # Stop codons
                "UAA": "STOP (ochre)", "UAG": "STOP (amber)",
                "UGA": "STOP (opal)",
                # Histidine
                "CAU": "His", "CAC": "His",
                # Glutamine
                "CAA": "Gln", "CAG": "Gln",
                # Asparagine
                "AAU": "Asn", "AAC": "Asn",
                # Lysine
                "AAA": "Lys", "AAG": "Lys",
                # Aspartic acid
                "GAU": "Asp", "GAC": "Asp",
                # Glutamic acid
                "GAA": "Glu", "GAG": "Glu",
                # Cysteine
                "UGU": "Cys", "UGC": "Cys",
                # Tryptophan
                "UGG": "Trp",
                # Arginine
                "CGU": "Arg", "CGC": "Arg", "CGA": "Arg", "CGG": "Arg",
                "AGA": "Arg", "AGG": "Arg",
                # Glycine
                "GGU": "Gly", "GGC": "Gly", "GGA": "Gly", "GGG": "Gly",
            }},
            conditions={"code": "standard_genetic_code"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Wobble position degeneracy ---
        Rule(
            name="wobble_position",
            pattern={"position": "third nucleotide of codon"},
            result={"degeneracy_pattern": {
                "4-fold": "any base at 3rd position → same amino acid (e.g., Val, Ala, Gly, Pro)",
                "2-fold_pyrimidine": "U/C at 3rd position (e.g., Phe, Tyr, His, Asn, Asp, Cys)",
                "2-fold_purine": "A/G at 3rd position (e.g., Lys, Glu, Gln)",
                "unique": "only one codon (Met=AUG, Trp=UGG)",
            },
                    "explanation": ("3rd position pairing is relaxed — "
                                    "wobble base pairing allows G:U pairs")},
            conditions={"molecular_basis": "wobble_hypothesis_Crick_1966"},
            weight=0.95,
            direction="bidirectional",
        ),

        # --- Reading frame ---
        Rule(
            name="reading_frame_determination",
            pattern={"sequence": "continuous nucleotide sequence"},
            result={"frames": {
                "+1": "start at position 1, read triplets",
                "+2": "start at position 2, read triplets",
                "+3": "start at position 3, read triplets",
                "-1": "reverse complement, start at position 1",
                "-2": "reverse complement, start at position 2",
                "-3": "reverse complement, start at position 3",
            },
                    "correct_frame": "determined by AUG start codon context"},
            conditions={"open_reading_frame": "AUG to stop codon, no internal stops"},
            weight=1.0,
            direction="forward",
        ),

        # --- Splicing rules ---
        Rule(
            name="intron_exon_splicing",
            pattern={"pre_mRNA": "Exon1-Intron-Exon2"},
            result={"mature_mRNA": "Exon1-Exon2",
                    "splice_signals": {
                        "5_prime_donor": "GU (almost invariant)",
                        "3_prime_acceptor": "AG (almost invariant)",
                        "branch_point": "A residue 18-40nt upstream of 3' splice site",
                    },
                    "mechanism": "spliceosome (U1, U2, U4, U5, U6 snRNPs)"},
            conditions={"eukaryotic": True},
            weight=0.95,
            direction="forward",
        ),
        Rule(
            name="alternative_splicing",
            pattern={"pre_mRNA": "Exon1-Intron1-Exon2-Intron2-Exon3"},
            result={"isoforms": [
                "Exon1-Exon2-Exon3 (constitutive)",
                "Exon1-Exon3 (exon skipping)",
                "Exon1-Exon2a-Exon3 (alternative 5' splice site)",
                "Exon1-Exon2-Exon2b-Exon3 (intron retention)",
            ],
                    "significance": "one gene → multiple proteins"},
            conditions={"regulatory_signals": "tissue-specific splicing factors"},
            weight=0.85,
            direction="forward",
        ),

        # --- Regulatory grammar ---
        Rule(
            name="promoter_recognition",
            pattern={"region": "upstream of transcription start site"},
            result={"elements": {
                "TATA_box": "TATAAA at -25 to -30 (eukaryotic)",
                "pribnow_box": "TATAAT at -10 (prokaryotic)",
                "minus_35": "TTGACA at -35 (prokaryotic)",
                "CpG_island": "GC-rich region (eukaryotic, often unmethylated)",
                "CAAT_box": "GGCCAATCT at -75 (eukaryotic)",
            }},
            conditions={"function": "RNA polymerase binding and initiation"},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="kozak_sequence",
            pattern={"context": "sequence around AUG start codon"},
            result={"consensus": "gcc(A/G)ccAUGG",
                    "critical_positions": {
                        "-3": "A or G (purine) — most important",
                        "+4": "G — second most important",
                    },
                    "function": "efficient translation initiation in eukaryotes"},
            conditions={"organism": "eukaryotic"},
            weight=0.85,
            direction="bidirectional",
        ),
    ]

    productions = [
        # Gene structure
        Production("Gene", ["Promoter", "CodingRegion", "Terminator"], "genetic"),
        Production("CodingRegion", ["StartCodon", "CodingSequence", "StopCodon"], "genetic"),
        Production("CodingSequence", ["Codon", "CodingSequence"], "genetic"),
        Production("CodingSequence", ["Codon"], "genetic"),
        Production("Codon", ["Nucleotide", "Nucleotide", "Nucleotide"], "genetic"),
        Production("Nucleotide", ["A"], "genetic"),
        Production("Nucleotide", ["U"], "genetic"),
        Production("Nucleotide", ["G"], "genetic"),
        Production("Nucleotide", ["C"], "genetic"),

        # Pre-mRNA structure
        Production("PreMRNA", ["Exon", "Intron", "PreMRNA"], "genetic"),
        Production("PreMRNA", ["Exon"], "genetic"),
        Production("MatureMRNA", ["Exon", "MatureMRNA"], "genetic"),
        Production("MatureMRNA", ["Exon"], "genetic"),
    ]

    return Grammar(
        name="genetic_code",
        domain="biology",
        rules=rules,
        productions=productions,
    )


# ---------------------------------------------------------------------------
# Protein grammar
# ---------------------------------------------------------------------------

def build_protein_grammar() -> Grammar:
    """Build a grammar for protein structure and folding.

    Protein folding is a derivation: from the primary sequence (linear
    string of amino acids), the protein derives its secondary structure
    (helices, sheets), then tertiary structure (3D fold), then quaternary
    structure (multi-subunit assembly).

    Includes:
    - Secondary structure prediction rules (helix/sheet propensity)
    - Domain grammar (independently folding units)
    - Active site patterns (catalytic triads, binding motifs)
    - Post-translational modifications
    """

    rules = [
        # --- Secondary structure propensities ---
        Rule(
            name="alpha_helix_propensity",
            pattern={"sequence_motif": "stretch of helix-favouring residues"},
            result={"structure": "alpha helix",
                    "helix_formers": {
                        "strong": ["Ala", "Glu", "Leu", "Met"],
                        "moderate": ["Phe", "Gln", "Trp", "Val", "Ile"],
                    },
                    "helix_breakers": ["Pro", "Gly"],
                    "geometry": {
                        "residues_per_turn": 3.6,
                        "rise_per_residue": "1.5 Å",
                        "hydrogen_bond": "i → i+4 backbone NH...O=C",
                    }},
            conditions={"minimum_length": "4-5 residues"},
            weight=0.85,
            direction="forward",
        ),
        Rule(
            name="beta_sheet_propensity",
            pattern={"sequence_motif": "stretch of sheet-favouring residues"},
            result={"structure": "beta sheet",
                    "sheet_formers": {
                        "strong": ["Val", "Ile", "Tyr", "Trp", "Phe", "Thr"],
                    },
                    "arrangement": ["parallel", "antiparallel"],
                    "geometry": {
                        "hydrogen_bonds": "between strands (inter-strand)",
                        "rise_per_residue": "3.4 Å",
                        "side_chains": "alternate above and below sheet plane",
                    }},
            conditions={"minimum_strands": 2},
            weight=0.8,
            direction="forward",
        ),
        Rule(
            name="turn_prediction",
            pattern={"sequence_motif": "short connecting loop"},
            result={"structure": "beta turn / reverse turn",
                    "types": {
                        "type_I": "φ₂=-60, ψ₂=-30, φ₃=-90, ψ₃=0",
                        "type_II": "φ₂=-60, ψ₂=120, φ₃=80, ψ₃=0",
                    },
                    "favoured_residues": {
                        "position_i+1": ["Pro", "Ser", "Asp"],
                        "position_i+2": ["Gly (type II)", "Asn"],
                    }},
            conditions={"length": "4 residues", "reverses_chain": True},
            weight=0.75,
            direction="forward",
        ),

        # --- Domain grammar ---
        Rule(
            name="protein_domain",
            pattern={"definition": "independently folding structural/functional unit"},
            result={"common_domains": {
                "SH2": "binds phosphotyrosine, ~100 residues",
                "SH3": "binds proline-rich motifs, ~60 residues",
                "kinase": "transfers phosphate from ATP, ~250 residues",
                "immunoglobulin": "beta-sandwich fold, ~100 residues",
                "helix_turn_helix": "DNA-binding, ~20 residues",
                "zinc_finger": "DNA/RNA-binding, coordinated by Zn²⁺, ~30 residues",
                "leucine_zipper": "dimerisation via coiled-coil, ~30 residues",
            },
                    "principle": "domains are the 'words' of protein grammar"},
            conditions={"evolutionary": "domains are shuffled and recombined"},
            weight=0.85,
            direction="bidirectional",
        ),

        # --- Active site patterns ---
        Rule(
            name="catalytic_triad",
            pattern={"residues": ["Ser", "His", "Asp"],
                     "spatial": "within hydrogen bonding distance"},
            result={"function": "nucleophilic catalysis (serine proteases)",
                    "mechanism": [
                        "1. Asp stabilises His via H-bond",
                        "2. His deprotonates Ser hydroxyl",
                        "3. Ser-O⁻ attacks peptide bond (nucleophilic)",
                        "4. Tetrahedral intermediate collapses",
                        "5. Products released, triad regenerated",
                    ],
                    "enzymes": ["chymotrypsin", "trypsin", "elastase", "subtilisin"]},
            conditions={"convergent_evolution": "found in unrelated fold families"},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="p_loop_ntpase",
            pattern={"motif": "GxxxxGK[ST] (Walker A motif)"},
            result={"function": "nucleotide (ATP/GTP) binding",
                    "mechanism": "Lys coordinates β and γ phosphates",
                    "examples": ["kinases", "GTPases (Ras)", "ABC transporters",
                                 "helicases"]},
            conditions={"followed_by": "Walker B motif (hhhhD, h=hydrophobic)"},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="ef_hand_calcium_binding",
            pattern={"motif": "helix-loop-helix, loop coordinates Ca²⁺"},
            result={"function": "calcium sensing and signalling",
                    "coordination": "loop residues at positions 1,3,5,7,9,12 coordinate Ca²⁺",
                    "examples": ["calmodulin", "troponin C", "S100 proteins"]},
            conditions={"ion": "Ca²⁺", "geometry": "pentagonal bipyramidal"},
            weight=0.85,
            direction="bidirectional",
        ),

        # --- Post-translational modifications ---
        Rule(
            name="phosphorylation",
            pattern={"target": "Ser/Thr/Tyr hydroxyl"},
            result={"modification": "addition of PO₄³⁻ group",
                    "effect": ["conformational change", "creates binding site",
                               "activates or inactivates enzyme"],
                    "enzymes": {"add": "kinase", "remove": "phosphatase"},
                    "significance": "major regulatory switch in signalling"},
            conditions={"consensus_motifs": {
                "PKA": "RRxS/T", "CK2": "S/TxxE/D", "CDK": "S/TP"}},
            weight=0.9,
            direction="forward",
        ),
        Rule(
            name="glycosylation",
            pattern={"target": "N-linked: NxS/T sequon; O-linked: Ser/Thr"},
            result={"modification": "addition of sugar chains",
                    "functions": ["protein folding assistance", "cell recognition",
                                  "protection from proteolysis", "half-life extension"],
                    "types": {
                        "N_linked": "attached to Asn in NxS/T (x ≠ Pro)",
                        "O_linked": "attached to Ser/Thr (no consensus sequence)",
                    }},
            conditions={"location": "ER and Golgi apparatus"},
            weight=0.85,
            direction="forward",
        ),
    ]

    productions = [
        # Protein structural hierarchy
        Production("Protein", ["Domain", "Linker", "Protein"], "protein"),
        Production("Protein", ["Domain"], "protein"),
        Production("Domain", ["SecondaryElement", "Domain"], "protein"),
        Production("Domain", ["SecondaryElement"], "protein"),
        Production("SecondaryElement", ["AlphaHelix"], "protein"),
        Production("SecondaryElement", ["BetaStrand"], "protein"),
        Production("SecondaryElement", ["Loop"], "protein"),
        Production("SecondaryElement", ["Turn"], "protein"),
        Production("AlphaHelix", ["HelixResidue", "AlphaHelix"], "protein"),
        Production("AlphaHelix", ["HelixResidue"], "protein"),
        Production("BetaStrand", ["StrandResidue", "BetaStrand"], "protein"),
        Production("BetaStrand", ["StrandResidue"], "protein"),
    ]

    return Grammar(
        name="protein_structure",
        domain="biology",
        rules=rules,
        productions=productions,
    )


# ---------------------------------------------------------------------------
# Evolutionary grammar
# ---------------------------------------------------------------------------

def build_evolutionary_grammar() -> Grammar:
    """Build a grammar for sequence evolution over time.

    Evolution is a temporal grammar that works in both directions:
    forward (predict how a sequence will change under selection/drift)
    and backward (reconstruct ancestral sequences from extant ones).

    Includes:
    - Mutation patterns (transition/transversion bias, CpG deamination)
    - Selection models (purifying, positive, neutral)
    - Molecular clock calibration
    - Ancestral sequence reconstruction
    - Convergent evolution detection
    """

    rules = [
        # --- Mutation patterns ---
        Rule(
            name="transition_transversion_bias",
            pattern={"mutation_type": "point substitution"},
            result={"bias": {
                "transitions": {
                    "A↔G": "purine ↔ purine (more frequent)",
                    "C↔U": "pyrimidine ↔ pyrimidine (more frequent)",
                },
                "transversions": {
                    "A↔C": "purine ↔ pyrimidine (less frequent)",
                    "A↔U": "purine ↔ pyrimidine",
                    "G↔C": "purine ↔ pyrimidine",
                    "G↔U": "purine ↔ pyrimidine",
                },
            },
                    "ratio": "Ti/Tv ≈ 2:1 in most genomes (higher in mtDNA)"},
            conditions={"model": "Kimura_two_parameter"},
            weight=0.95,
            direction="bidirectional",
        ),
        Rule(
            name="cpg_deamination",
            pattern={"context": "CpG dinucleotide"},
            result={"mutation": "methylated C → T (by spontaneous deamination)",
                    "rate": "10-50x higher than other transitions",
                    "consequences": [
                        "CpG depletion in vertebrate genomes",
                        "CpG islands preserved near promoters (unmethylated)",
                        "major source of de novo mutations in humans",
                    ]},
            conditions={"methylation": True, "organism": "vertebrate"},
            weight=0.9,
            direction="forward",
        ),
        Rule(
            name="codon_position_rate_variation",
            pattern={"position_in_codon": "1st, 2nd, or 3rd"},
            result={"relative_rates": {
                "1st_position": "moderate (some synonymous changes)",
                "2nd_position": "slowest (all changes nonsynonymous)",
                "3rd_position": "fastest (most changes synonymous / wobble)",
            },
                    "ratio": "3rd >> 1st > 2nd"},
            conditions={"coding_sequence": True},
            weight=0.9,
            direction="bidirectional",
        ),

        # --- Selection models ---
        Rule(
            name="purifying_selection",
            pattern={"signal": "dN/dS < 1 (Ka/Ks < 1)"},
            result={"interpretation": "nonsynonymous mutations removed by selection",
                    "indicates": "functional constraint / conservation",
                    "typical_for": ["essential enzymes", "structural proteins",
                                     "ribosomal RNA", "tRNA"]},
            conditions={"dN_dS_ratio": "significantly less than 1"},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="positive_selection",
            pattern={"signal": "dN/dS > 1 (Ka/Ks > 1)"},
            result={"interpretation": "nonsynonymous mutations favoured by selection",
                    "indicates": "adaptive evolution / arms race",
                    "typical_for": ["immune genes (MHC)", "reproductive proteins",
                                     "pathogen resistance genes",
                                     "venom proteins"]},
            conditions={"dN_dS_ratio": "significantly greater than 1"},
            weight=0.85,
            direction="bidirectional",
        ),
        Rule(
            name="neutral_evolution",
            pattern={"signal": "dN/dS ≈ 1"},
            result={"interpretation": "mutations neither favoured nor removed",
                    "indicates": "neutral drift (Kimura)",
                    "typical_for": ["pseudogenes", "intergenic regions",
                                     "synonymous sites"]},
            conditions={"dN_dS_ratio": "approximately 1"},
            weight=0.8,
            direction="bidirectional",
        ),

        # --- Molecular clock ---
        Rule(
            name="molecular_clock_calibration",
            pattern={"input": "sequence divergence between species"},
            result={"method": [
                "1. Align orthologous sequences",
                "2. Count substitutions (corrected for multiple hits)",
                "3. Calibrate with fossil record / geological events",
                "4. Estimate divergence time = divergence / (2 × rate)",
            ],
                    "models": {
                        "JC69": "equal rates, single parameter",
                        "K80": "transition/transversion bias, two parameters",
                        "HKY85": "Ti/Tv bias + unequal base frequencies",
                        "GTR": "general time-reversible, most parameters",
                    }},
            conditions={"assumption": "roughly constant substitution rate"},
            weight=0.8,
            direction="bidirectional",
        ),

        # --- Ancestral reconstruction ---
        Rule(
            name="ancestral_sequence_reconstruction",
            pattern={"input": "alignment of extant sequences + phylogeny"},
            result={"output": "inferred ancestral sequences at internal nodes",
                    "methods": {
                        "maximum_parsimony": "minimise total changes on tree",
                        "maximum_likelihood": "most probable ancestral states given model",
                        "bayesian": "posterior distribution of ancestral states",
                    },
                    "applications": ["ancestral protein resurrection",
                                      "tracing functional evolution",
                                      "dating gene duplications"]},
            conditions={"requires": "reliable phylogeny and alignment"},
            weight=0.85,
            direction="backward",
        ),

        # --- Convergent evolution ---
        Rule(
            name="convergent_evolution_detection",
            pattern={"observation": "same trait/sequence in unrelated lineages"},
            result={"tests": [
                "1. Phylogenetic incongruence (gene tree ≠ species tree)",
                "2. Same amino acid at same position in distant taxa",
                "3. Similar selective pressure in different environments",
                "4. Structural/functional constraint channels evolution",
            ],
                    "examples": [
                        "echolocation: bats and dolphins (Prestin gene)",
                        "C4 photosynthesis: evolved >60 times independently",
                        "antifreeze proteins: Arctic and Antarctic fish",
                    ]},
            conditions={"distinguishes_from": "shared ancestry / homology"},
            weight=0.8,
            direction="bidirectional",
        ),

        # --- Temporal prediction (superforecasting) ---
        Rule(
            name="predict_future_evolution",
            pattern={"input": "current sequence + selective environment"},
            result={"prediction_factors": [
                "mutation spectrum (which changes are chemically likely)",
                "selection landscape (which changes are functionally viable)",
                "population size (drift vs selection effectiveness)",
                "generation time (molecular clock rate)",
                "recombination rate (linkage effects)",
            ],
                    "applications": ["influenza strain prediction",
                                      "antibiotic resistance forecasting",
                                      "cancer evolution modelling"]},
            conditions={"method": "evolutionary_extrapolation"},
            weight=0.7,
            direction="forward",
        ),
    ]

    productions = [
        # Evolutionary event hierarchy
        Production("Evolution", ["Mutation", "Selection", "Drift"], "evolutionary"),
        Production("Mutation", ["PointMutation"], "evolutionary"),
        Production("Mutation", ["Insertion"], "evolutionary"),
        Production("Mutation", ["Deletion"], "evolutionary"),
        Production("Mutation", ["Duplication"], "evolutionary"),
        Production("Mutation", ["Inversion"], "evolutionary"),
        Production("PointMutation", ["Transition"], "evolutionary"),
        Production("PointMutation", ["Transversion"], "evolutionary"),
        Production("Selection", ["Purifying"], "evolutionary"),
        Production("Selection", ["Positive"], "evolutionary"),
        Production("Selection", ["Balancing"], "evolutionary"),
        Production("Drift", ["BottleneckDrift"], "evolutionary"),
        Production("Drift", ["FounderDrift"], "evolutionary"),
        Production("Drift", ["NeutralDrift"], "evolutionary"),
    ]

    # The ultimate strange loop: DNA → RNA → Protein → (regulates) → DNA
    strange_loop = StrangeLoop(
        entry_rule="central_dogma",
        cycle=[
            "DNA_transcribed_to_RNA",
            "RNA_translated_to_protein",
            "protein_regulates_DNA_transcription",
            "regulated_DNA_changes_which_RNA_is_made",
        ],
        level_shift="autopoietic_closure",
    )

    return Grammar(
        name="evolutionary_processes",
        domain="biology",
        rules=rules,
        productions=productions,
        sub_grammars=[strange_loop],
    )
