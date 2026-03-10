"""Etymology and historical derivation grammars.

These grammars encode how words travel through time and across languages.
Sound correspondences (Grimm's Law, Verner's Law, the Great Vowel Shift)
are formal rules that operate bidirectionally: given a modern word we can
reconstruct its proto-form, and given a proto-form we can predict its
modern reflexes in daughter languages.

Substrate transfer rules capture how grammatical patterns migrate between
languages in contact — the deep currents beneath surface borrowing.
Cognate detection rules identify shared ancestry across languages,
functioning as cross-domain pattern matchers.
"""

from glm.core.grammar import Rule, Production, Grammar, StrangeLoop


# ---------------------------------------------------------------------------
# Etymology grammar
# ---------------------------------------------------------------------------

def build_etymology_grammar() -> Grammar:
    """Build a grammar for etymological derivation.

    Includes:
    - Sound correspondences between language families (Grimm's Law,
      Verner's Law, Grassmann's Law, the Ruki rule)
    - Semantic drift patterns (metaphor, metonymy, narrowing, broadening,
      amelioration, pejoration)
    - Temporal rules that work in both directions: reconstruction and
      prediction of reflexes

    Returns a Grammar encoding the systematic transformations that connect
    proto-languages to their modern descendants.
    """

    rules = [
        # ===================================================================
        # SOUND CORRESPONDENCES — Indo-European
        # ===================================================================

        # --- Grimm's Law (Proto-IE → Proto-Germanic) ----------------------
        Rule(
            name="grimms_law_voiceless_stops",
            pattern={"PIE": "*p, *t, *k, *kʷ"},
            result={"PGmc": "*f, *θ, *x, *xʷ",
                    "description": "voiceless stops → voiceless fricatives",
                    "examples": [
                        "*pōds → *fōts (foot)",
                        "*tréyes → *θrīz (three)",
                        "*ḱm̥tóm → *xundam (hundred)",
                    ]},
            conditions={"branch": "Germanic", "not_after": "s"},
            weight=0.95,
            direction="bidirectional",
        ),
        Rule(
            name="grimms_law_voiced_stops",
            pattern={"PIE": "*b, *d, *g, *gʷ"},
            result={"PGmc": "*p, *t, *k, *kʷ",
                    "description": "voiced stops → voiceless stops",
                    "examples": [
                        "*dékm̥t → *texun (ten)",
                        "*genu → *knewa- (knee)",
                        "*gʷīwós → *kwikwaz (alive/quick)",
                    ]},
            conditions={"branch": "Germanic"},
            weight=0.95,
            direction="bidirectional",
        ),
        Rule(
            name="grimms_law_voiced_aspirates",
            pattern={"PIE": "*bʰ, *dʰ, *gʰ, *gʷʰ"},
            result={"PGmc": "*b, *d, *g, *gʷ",
                    "description": "voiced aspirates → voiced stops",
                    "examples": [
                        "*bʰréh₂tēr → *brōþēr (brother)",
                        "*dʰeh₁- → *dōn (do)",
                        "*gʰóstis → *gastiz (guest)",
                    ]},
            conditions={"branch": "Germanic"},
            weight=0.95,
            direction="bidirectional",
        ),

        # --- Verner's Law (exception to Grimm's Law) ---------------------
        Rule(
            name="verners_law",
            pattern={"PGmc_fricative": "*f, *θ, *x, *s"},
            result={"PGmc_voiced": "*b, *d, *g, *z",
                    "description": ("voiceless fricatives voice when "
                                    "preceding syllable was unstressed in PIE"),
                    "examples": [
                        "*ph₂tḗr → *faþēr but cf. *bróh₂tēr → *brōþēr",
                        "was/were alternation in English",
                    ]},
            conditions={"accent": "not_on_preceding_syllable",
                        "applies_after": "grimms_law"},
            weight=0.85,
            direction="bidirectional",
        ),

        # --- Ruki rule (PIE → Indo-Iranian, Baltic, Slavic) ---------------
        Rule(
            name="ruki_rule",
            pattern={"PIE": "*s"},
            result={"reflex": "*š → ṣ (retroflex)",
                    "description": "s → š after r, u, k, i",
                    "examples": [
                        "PIE *h₂ŕ̥tkos → Skt ṛkṣa- (bear)",
                        "PIE *mus- → Skt mūṣ- (mouse/steal)",
                    ]},
            conditions={"environment": "after_r_u_k_i",
                        "branches": ["Indo-Iranian", "Baltic", "Slavic"]},
            weight=0.9,
            direction="bidirectional",
        ),

        # --- Latin → Romance (Ibero-Romance examples) --------------------
        Rule(
            name="latin_intervocalic_voicing",
            pattern={"Latin": "voiceless stop between vowels"},
            result={"Romance": "voiced stop / fricative / deletion",
                    "chain": "p→b, t→d, k→g (then further lenition)",
                    "examples": [
                        "lupum → lobo (Spanish)",
                        "vitam → vida (Spanish)",
                        "focum → fuego (Spanish)",
                    ]},
            conditions={"environment": "intervocalic", "branch": "Romance"},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="latin_initial_f_to_h",
            pattern={"Latin": "f- (word-initial before vowel)"},
            result={"Castilian": "h- (silent in modern Spanish)",
                    "examples": [
                        "filium → hijo",
                        "facere → hacer",
                        "fabulare → hablar",
                    ]},
            conditions={"branch": "Castilian", "not": "Portuguese/Catalan"},
            weight=0.85,
            direction="bidirectional",
        ),

        # --- Proto-Austronesian → daughter languages ---------------------
        Rule(
            name="austronesian_consonant_split",
            pattern={"PAN": "*C (pre-Austronesian voiced stops)"},
            result={"reflexes": {
                "Tagalog": "retained as stops",
                "Hawaiian": "merged/deleted (fewer consonants)",
                "Malay": "retained with shifts",
            }},
            conditions={"branch": "Austronesian"},
            weight=0.8,
            direction="bidirectional",
        ),

        # ===================================================================
        # SEMANTIC DRIFT PATTERNS
        # ===================================================================

        Rule(
            name="semantic_narrowing",
            pattern={"original": "general meaning"},
            result={"derived": "more specific meaning",
                    "examples": [
                        "deer: any animal → specifically Cervidae",
                        "hound: any dog → specific breed type",
                        "meat: any food → specifically flesh",
                        "starve: to die (any cause) → to die of hunger",
                    ]},
            conditions={"direction": "general_to_specific"},
            weight=0.8,
            direction="forward",
        ),
        Rule(
            name="semantic_broadening",
            pattern={"original": "specific meaning"},
            result={"derived": "more general meaning",
                    "examples": [
                        "bird: young bird → any bird",
                        "dog: specific breed → any canine",
                        "thing: assembly → any entity",
                        "salary: salt-money → any wages",
                    ]},
            conditions={"direction": "specific_to_general"},
            weight=0.75,
            direction="forward",
        ),
        Rule(
            name="semantic_metaphor",
            pattern={"source_domain": "concrete/physical"},
            result={"target_domain": "abstract/mental",
                    "examples": [
                        "grasp: physically seize → understand",
                        "see: perceive visually → understand",
                        "bitter: taste → emotional quality",
                        "weigh: measure mass → consider carefully",
                    ]},
            conditions={"mapping": "concrete_to_abstract"},
            weight=0.85,
            direction="forward",
        ),
        Rule(
            name="semantic_metonymy",
            pattern={"source": "associated entity"},
            result={"target": "thing itself",
                    "examples": [
                        "crown: headgear → monarchy",
                        "tongue: body part → language",
                        "hand: body part → worker/help",
                        "bureau: desk → office/agency",
                    ]},
            conditions={"mapping": "contiguity"},
            weight=0.8,
            direction="forward",
        ),
        Rule(
            name="amelioration",
            pattern={"original": "neutral/negative connotation"},
            result={"derived": "positive connotation",
                    "examples": [
                        "nice: ignorant (Latin nescius) → pleasant",
                        "pretty: tricky/cunning → attractive",
                        "knight: boy/servant → noble warrior",
                    ]},
            conditions={"drift_direction": "positive"},
            weight=0.6,
            direction="forward",
        ),
        Rule(
            name="pejoration",
            pattern={"original": "neutral/positive connotation"},
            result={"derived": "negative connotation",
                    "examples": [
                        "villain: farm worker → evil person",
                        "silly: blessed/happy → foolish",
                        "knave: boy → dishonest person",
                    ]},
            conditions={"drift_direction": "negative"},
            weight=0.6,
            direction="forward",
        ),

        # ===================================================================
        # TEMPORAL BIDIRECTIONAL RULES
        # ===================================================================

        Rule(
            name="reconstruct_proto_form",
            pattern={"modern_reflexes": "set of cognates across daughter languages"},
            result={"proto_form": "reconstructed ancestral form (*-marked)",
                    "method": "comparative_method",
                    "steps": [
                        "1. Align cognate sets",
                        "2. Identify systematic correspondences",
                        "3. Apply directionality principles",
                        "4. Reconstruct proto-phoneme",
                    ]},
            conditions={"requires": "multiple_daughter_languages"},
            weight=0.9,
            direction="backward",
        ),
        Rule(
            name="predict_modern_reflex",
            pattern={"proto_form": "*reconstructed form"},
            result={"modern_reflexes": "predicted forms in daughter languages",
                    "method": "apply_sound_laws_chronologically",
                    "steps": [
                        "1. Apply earliest sound changes first",
                        "2. Chain-apply subsequent changes",
                        "3. Apply analogy / levelling",
                        "4. Predict modern pronunciation",
                    ]},
            conditions={"requires": "known_sound_correspondences"},
            weight=0.85,
            direction="forward",
        ),
    ]

    productions = [
        # Etymological derivation chain
        Production("ModernWord", ["SoundChange", "IntermediateForm"], "etymological"),
        Production("IntermediateForm", ["SoundChange", "ProtoForm"], "etymological"),
        Production("ProtoForm", ["Root", "ProtoMorphology"], "etymological"),
        Production("Root", ["PIE_Root"], "etymological"),
        Production("Root", ["PAN_Root"], "etymological"),
        Production("Root", ["PSem_Root"], "etymological"),

        # Borrowing path
        Production("ModernWord", ["Borrowing", "SourceWord"], "etymological"),
        Production("SourceWord", ["SoundChange", "SourceProtoForm"], "etymological"),

        # Calque / loan-translation
        Production("ModernWord", ["Calque", "SourceCompound"], "etymological"),
        Production("SourceCompound", ["Morpheme", "Morpheme"], "etymological"),
    ]

    return Grammar(
        name="etymological_derivation",
        domain="etymology",
        rules=rules,
        productions=productions,
    )


# ---------------------------------------------------------------------------
# Substrate transfer grammar
# ---------------------------------------------------------------------------

def build_substrate_transfer_grammar() -> Grammar:
    """Build a grammar for substrate/contact-induced language transfer.

    Models how grammatical patterns migrate between languages in contact:
    - Phonological transfer (new phonemes, prosodic patterns)
    - Morphosyntactic calquing (word-order shifts, case-system loss)
    - Semantic calquing (extending meaning to match contact language)
    - Pragmatic transfer (discourse patterns, politeness strategies)

    These rules capture the deep currents beneath surface-level borrowing.
    """

    rules = [
        # --- Phonological transfer ---
        Rule(
            name="phoneme_adoption",
            pattern={"source_language": "phoneme /X/ not in target inventory"},
            result={"outcomes": [
                "adoption: target adds /X/ to inventory",
                "substitution: target maps /X/ to nearest native phoneme",
                "conditioned adoption: /X/ enters only in loanwords",
            ],
                    "examples": [
                        "English /ʒ/ adopted via French loans (genre, beige)",
                        "Japanese maps English /l,r/ to /ɾ/",
                    ]},
            conditions={"contact_intensity": "moderate_to_high"},
            weight=0.7,
            direction="forward",
        ),
        Rule(
            name="prosodic_transfer",
            pattern={"source": "stress/tone pattern from dominant language"},
            result={"target_shift": "subordinate language adopts prosodic features",
                    "examples": [
                        "South Asian English: syllable-timed rhythm from Hindi",
                        "Irish English: pitch patterns from Irish Gaelic",
                    ]},
            conditions={"contact_intensity": "high", "bilingualism": "widespread"},
            weight=0.65,
            direction="forward",
        ),

        # --- Morphosyntactic transfer ---
        Rule(
            name="word_order_shift",
            pattern={"substrate": "SOV order in substrate language"},
            result={"superstrate_affected": "SOV tendencies in local variety",
                    "examples": [
                        "South Asian English: verb-final tendencies",
                        "Hiberno-English: VSO residue from Irish",
                        "Yiddish influence on English: OV in some constructions",
                    ]},
            conditions={"mechanism": "imperfect_L2_acquisition"},
            weight=0.6,
            direction="forward",
        ),
        Rule(
            name="case_system_loss_through_contact",
            pattern={"situation": "inflectional language in contact with analytic language"},
            result={"change": "gradual loss of case distinctions",
                    "examples": [
                        "Old English → Middle English: loss of case after Norse contact",
                        "Baltic German simplified under Latvian/Estonian contact",
                    ]},
            conditions={"contact_type": "prolonged", "social": "intermarriage"},
            weight=0.55,
            direction="forward",
        ),
        Rule(
            name="evidentiality_transfer",
            pattern={"substrate": "grammaticalised evidentiality markers"},
            result={"superstrate_affected": "development of evidential particles/verbs",
                    "examples": [
                        "Andean Spanish: use of dice que (says that) for hearsay",
                        "Balkan Sprachbund: shared evidential past tenses",
                    ]},
            conditions={"areal_feature": True},
            weight=0.7,
            direction="forward",
        ),

        # --- Semantic calquing ---
        Rule(
            name="semantic_extension_by_contact",
            pattern={"source_word": "polysemous word in language A"},
            result={"target_word": "cognate/translation in B extends to match A's senses",
                    "examples": [
                        "Spanish realizar gaining English 'realize' sense (understand)",
                        "German 'realisieren' gaining 'understand' sense from English",
                        "French 'réaliser' → 'to understand' (from English influence)",
                    ]},
            conditions={"mechanism": "bilingual_polysemy_copying"},
            weight=0.75,
            direction="forward",
        ),

        # --- Areal features (Sprachbund) ---
        Rule(
            name="sprachbund_convergence",
            pattern={"area": "geographically bounded multilingual region"},
            result={"shared_features": [
                "definite article postposed (Balkan)",
                "loss of infinitive (Balkan)",
                "evidential mood (Balkan)",
                "retroflex consonants (South Asian)",
                "SOV + postpositions (South Asian)",
                "tonal systems (Mainland Southeast Asian)",
            ]},
            conditions={"mechanism": "prolonged_multilateral_contact"},
            weight=0.8,
            direction="bidirectional",
        ),
    ]

    productions = [
        Production("ContactOutcome", ["PhonologicalTransfer"], "substrate"),
        Production("ContactOutcome", ["MorphosyntacticTransfer"], "substrate"),
        Production("ContactOutcome", ["SemanticTransfer"], "substrate"),
        Production("ContactOutcome", ["PragmaticTransfer"], "substrate"),
        Production("MorphosyntacticTransfer", ["WordOrderShift"], "substrate"),
        Production("MorphosyntacticTransfer", ["CaseSystemChange"], "substrate"),
        Production("MorphosyntacticTransfer", ["EvidentialityAdoption"], "substrate"),
    ]

    return Grammar(
        name="substrate_transfer",
        domain="etymology",
        rules=rules,
        productions=productions,
    )


# ---------------------------------------------------------------------------
# Cognate detection grammar
# ---------------------------------------------------------------------------

def build_cognate_detection_grammar() -> Grammar:
    """Build a grammar for identifying cognates across languages.

    Cognates are words in different languages descended from the same
    ancestral form.  This grammar provides rules for:
    - Systematic sound correspondence matching
    - Distinguishing true cognates from false friends and borrowings
    - Scoring cognate likelihood based on phonological, semantic, and
      morphological alignment
    - Cross-domain pattern matching (the same method works for gene
      homology and chemical series)

    This is inherently bidirectional: given two words, determine if they
    share a root; given a root, predict its cognate set.
    """

    rules = [
        # --- Correspondence matching ---
        Rule(
            name="regular_sound_correspondence",
            pattern={"word_A": "phoneme X in language A",
                     "word_B": "phoneme Y in language B"},
            result={"cognate_signal": "positive if X↔Y is a known regular correspondence",
                    "examples": [
                        "English f ↔ Latin p (father/pater)",
                        "English th ↔ Latin t (three/tres)",
                        "English h ↔ Latin c (hundred/centum)",
                        "English t ↔ Latin d (ten/decem)",
                    ]},
            conditions={"correspondence_is_regular": True,
                        "position_matches": True},
            weight=0.95,
            direction="bidirectional",
        ),
        Rule(
            name="semantic_compatibility_check",
            pattern={"meaning_A": "meaning in language A",
                     "meaning_B": "meaning in language B"},
            result={"cognate_signal": ("positive if meanings overlap or are "
                                        "connected by known drift pattern"),
                    "drift_tolerance": [
                        "identical meaning (strong signal)",
                        "related by narrowing/broadening (moderate signal)",
                        "related by metaphor (weak but valid signal)",
                        "unrelated meanings (negative signal unless explained)",
                    ]},
            conditions={"requires": "semantic_drift_model"},
            weight=0.8,
            direction="bidirectional",
        ),

        # --- False friends ---
        Rule(
            name="false_friend_detection",
            pattern={"surface_similarity": "high",
                     "sound_correspondence": "irregular"},
            result={"verdict": "likely false friend or borrowing, not cognate",
                    "examples": [
                        "English 'much' vs Spanish 'mucho': true cognates (< Latin)",
                        "English 'actual' vs French 'actuel': false friend (meaning shifted)",
                        "English 'gift' vs German 'Gift' (poison): meaning divergence",
                    ]},
            conditions={"correspondence_is_regular": False},
            weight=0.85,
            direction="bidirectional",
        ),
        Rule(
            name="borrowing_vs_inheritance",
            pattern={"candidate_pair": "similar words in related languages"},
            result={"tests": [
                "1. Does it show expected sound correspondences? (inherited)",
                "2. Does it violate regular correspondences? (borrowed)",
                "3. Is it limited to a semantic field of cultural contact? (borrowed)",
                "4. Does the morphology integrate natively? (inherited)",
                "5. Is it present in all branches? (inherited = old, one branch = borrowed)",
            ]},
            conditions={"disambiguation_needed": True},
            weight=0.9,
            direction="bidirectional",
        ),

        # --- Cognate scoring ---
        Rule(
            name="cognate_confidence_scoring",
            pattern={"candidate_pair": "word_A, word_B"},
            result={"score_components": {
                "phonological_alignment": "0.0-1.0 (regular correspondences)",
                "semantic_alignment": "0.0-1.0 (meaning compatibility)",
                "morphological_alignment": "0.0-1.0 (structural parallel)",
                "geographical_plausibility": "0.0-1.0 (contact history)",
                "temporal_consistency": "0.0-1.0 (dating of changes)",
            },
                    "threshold": "combined score > 0.7 → probable cognate"},
            conditions={"method": "weighted_multi-factor"},
            weight=1.0,
            direction="bidirectional",
        ),

        # --- Cross-domain isomorphism ---
        Rule(
            name="cognate_homology_isomorphism",
            pattern={"domain_A": "linguistic cognate detection",
                     "domain_B": "biological homology detection"},
            result={"shared_method": [
                "systematic correspondence = conserved positions",
                "regular sound change = consistent mutation pattern",
                "cognate set = gene family",
                "proto-form reconstruction = ancestral sequence inference",
                "borrowing vs inheritance = horizontal vs vertical transfer",
            ]},
            conditions={"isomorphism": "fugue_across_domains"},
            weight=0.7,
            direction="bidirectional",
        ),

        # --- Reconstruction from cognate sets ---
        Rule(
            name="majority_rule_reconstruction",
            pattern={"cognate_set": "reflexes from multiple daughter languages"},
            result={"proto_phoneme": "most common reflex, adjusted for directionality",
                    "principle": ("unconditional mergers are irreversible — "
                                  "if daughters disagree, the distinction was "
                                  "present in the proto-language")},
            conditions={"method": "comparative_reconstruction"},
            weight=0.9,
            direction="backward",
        ),
    ]

    productions = [
        Production("CognateJudgement", ["PhonMatch", "SemMatch", "MorphMatch"], "cognate"),
        Production("PhonMatch", ["Correspondence", "Position"], "cognate"),
        Production("SemMatch", ["MeaningOverlap", "DriftPath"], "cognate"),
        Production("MorphMatch", ["RootShape", "AffixPattern"], "cognate"),
    ]

    # Strange loop: cognate detection relies on sound laws, sound laws are
    # discovered by examining cognate sets — each presupposes the other.
    strange_loop = StrangeLoop(
        entry_rule="cognate_identification",
        cycle=[
            "identify_cognates_using_sound_laws",
            "discover_sound_laws_from_cognate_sets",
            "refine_cognate_sets_with_new_sound_laws",
        ],
        level_shift="bootstrapping_spiral",
    )

    return Grammar(
        name="cognate_detection",
        domain="etymology",
        rules=rules,
        productions=productions,
        sub_grammars=[strange_loop],
    )
