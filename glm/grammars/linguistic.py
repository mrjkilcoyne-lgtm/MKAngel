"""Natural language grammars — syntax, phonology, and morphology.

These grammars encode the deep structural rules of human language.  They
work bidirectionally: the same rule set that generates a sentence can
parse one, the same phonological rule that predicts a modern reflex can
reconstruct a proto-form.

The strange loop here is the pragmatics cycle:
    syntax → semantics → pragmatics → syntax
Syntactic structure determines meaning (semantics), meaning determines
communicative intent (pragmatics), and intent reshapes syntactic choices —
an inescapable tangled hierarchy at the heart of every utterance.
"""

from glm.core.grammar import Rule, Production, Grammar, StrangeLoop


# ---------------------------------------------------------------------------
# Syntactic grammar
# ---------------------------------------------------------------------------

def build_syntactic_grammar() -> Grammar:
    """Build a phrase-structure grammar for natural language.

    Includes:
    - Context-free production rules for NP, VP, PP, CP, IP
    - Agreement rules (number, person, gender)
    - Movement / transformation rules (wh-movement, passive, topicalisation)
    - Bidirectional rules for generation and parsing

    Returns a Grammar whose productions mirror the X-bar schema and whose
    rules encode the constraints that make sentences well-formed.
    """

    # -- Phrase-structure productions (X-bar theory) -----------------------

    productions = [
        # Sentence level
        Production("S", ["NP", "VP"], "syntactic"),
        Production("S", ["CP"], "syntactic"),
        Production("S", ["Adv", "S"], "syntactic"),

        # Complementiser phrase (for embedded clauses / questions)
        Production("CP", ["C", "IP"], "syntactic"),
        Production("CP", ["Wh", "C", "IP"], "syntactic"),

        # Inflectional phrase
        Production("IP", ["NP", "I'"], "syntactic"),
        Production("I'", ["I", "VP"], "syntactic"),

        # Noun phrases
        Production("NP", ["Det", "N'"], "syntactic"),
        Production("NP", ["N'"], "syntactic"),
        Production("NP", ["NP", "PP"], "syntactic"),
        Production("NP", ["NP", "CP"], "syntactic"),       # relative clause
        Production("N'", ["Adj", "N'"], "syntactic"),
        Production("N'", ["N"], "syntactic"),
        Production("N'", ["N", "PP"], "syntactic"),

        # Verb phrases
        Production("VP", ["V"], "syntactic"),
        Production("VP", ["V", "NP"], "syntactic"),
        Production("VP", ["V", "NP", "NP"], "syntactic"),  # ditransitive
        Production("VP", ["V", "NP", "PP"], "syntactic"),
        Production("VP", ["V", "CP"], "syntactic"),         # sentential complement
        Production("VP", ["V", "VP"], "syntactic"),         # control / raising
        Production("VP", ["Adv", "VP"], "syntactic"),
        Production("VP", ["VP", "PP"], "syntactic"),

        # Prepositional phrases
        Production("PP", ["P", "NP"], "syntactic"),
        Production("PP", ["P", "CP"], "syntactic"),

        # Adjective phrases
        Production("AdjP", ["Deg", "Adj"], "syntactic"),
        Production("AdjP", ["Adj", "PP"], "syntactic"),
        Production("AdjP", ["Adj"], "syntactic"),

        # Adverb phrases
        Production("AdvP", ["Deg", "Adv"], "syntactic"),
        Production("AdvP", ["Adv"], "syntactic"),
    ]

    # -- Constraint / agreement rules -------------------------------------

    rules = [
        # Subject-verb agreement: number
        Rule(
            name="subject_verb_number_agreement",
            pattern={"structure": "NP[num=?n] VP[num=?n]"},
            result={"constraint": "NP.number == VP.number"},
            conditions={"requires": "number_feature"},
            weight=1.0,
            direction="bidirectional",
        ),

        # Subject-verb agreement: person
        Rule(
            name="subject_verb_person_agreement",
            pattern={"structure": "NP[per=?p] VP[per=?p]"},
            result={"constraint": "NP.person == VP.person"},
            conditions={"requires": "person_feature"},
            weight=1.0,
            direction="bidirectional",
        ),

        # Determiner-noun agreement: gender (for gendered languages)
        Rule(
            name="det_noun_gender_agreement",
            pattern={"structure": "Det[gen=?g] N[gen=?g]"},
            result={"constraint": "Det.gender == N.gender"},
            conditions={"requires": "gender_feature"},
            weight=0.9,
            direction="bidirectional",
        ),

        # Case assignment: nominative for subjects
        Rule(
            name="nominative_case_assignment",
            pattern={"structure": "[Spec, IP] NP"},
            result={"case": "nominative"},
            conditions={"position": "specifier_of_IP"},
            weight=1.0,
            direction="forward",
        ),

        # Case assignment: accusative for objects
        Rule(
            name="accusative_case_assignment",
            pattern={"structure": "[Comp, V] NP"},
            result={"case": "accusative"},
            conditions={"position": "complement_of_V"},
            weight=1.0,
            direction="forward",
        ),

        # Wh-movement: move wh-phrase to Spec-CP
        Rule(
            name="wh_movement",
            pattern={"structure": "CP[+Q] → Wh_i ... t_i"},
            result={"movement": "wh_to_spec_CP", "trace": "t_i"},
            conditions={"clause_type": "interrogative", "island_constraints": True},
            weight=0.95,
            direction="bidirectional",
        ),

        # Passive transformation
        Rule(
            name="passive_transformation",
            pattern={"active": "NP_subj V NP_obj"},
            result={"passive": "NP_obj be V-en (by NP_subj)"},
            conditions={"verb_type": "transitive"},
            weight=0.85,
            direction="bidirectional",
        ),

        # Topicalisation: fronting a constituent for emphasis
        Rule(
            name="topicalisation",
            pattern={"base": "S → NP VP[...XP...]"},
            result={"topicalised": "XP_i S → NP VP[...t_i...]"},
            conditions={"information_structure": "topic_prominent"},
            weight=0.7,
            direction="bidirectional",
        ),

        # Head-directionality parameter (head-initial vs head-final)
        Rule(
            name="head_directionality",
            pattern={"head_position": "?pos"},
            result={"order": "head before complement if head-initial"},
            conditions={"language_type": "configurable"},
            weight=1.0,
            direction="bidirectional",
        ),

        # Pro-drop: null subject licensing
        Rule(
            name="pro_drop",
            pattern={"structure": "IP → pro I' "},
            result={"null_subject": True, "rich_agreement": True},
            conditions={"agreement_paradigm": "rich"},
            weight=0.8,
            direction="forward",
        ),

        # Binding principle A: anaphors must be bound in their governing category
        Rule(
            name="binding_principle_A",
            pattern={"anaphor": "reflexive/reciprocal"},
            result={"must_be_bound_in": "local_domain"},
            conditions={"binding_domain": "governing_category"},
            weight=1.0,
            direction="bidirectional",
        ),
    ]

    # -- Strange loop: syntax → semantics → pragmatics → syntax ------------

    strange_loop = StrangeLoop(
        entry_rule="syntactic_structure",
        cycle=[
            "syntax_determines_semantics",
            "semantics_determines_pragmatics",
            "pragmatics_reshapes_syntax",
        ],
        level_shift="upward_then_fold",
    )

    return Grammar(
        name="natural_language_syntax",
        domain="linguistics",
        rules=rules,
        productions=productions,
        sub_grammars=[strange_loop],
    )


# ---------------------------------------------------------------------------
# Phonological grammar
# ---------------------------------------------------------------------------

def build_phonological_grammar() -> Grammar:
    """Build a phonological grammar capturing sound-change rules.

    Includes rules for:
    - Assimilation (progressive and regressive)
    - Dissimilation
    - Metathesis
    - Vowel harmony (palatal and labial)
    - Lenition and fortition
    - Epenthesis and syncope

    These rules are temporal: they predict how sounds change over time
    (forward) and reconstruct earlier forms (backward).
    """

    rules = [
        # --- Assimilation ---
        Rule(
            name="regressive_nasal_assimilation",
            pattern={"context": "N → [+nasal, αplace] / _ C[αplace]"},
            result={"change": "nasal adopts place of following consonant",
                    "examples": ["/n+p/ → [mp]", "/n+k/ → [ŋk]"]},
            conditions={"environment": "before_obstruent"},
            weight=0.95,
            direction="bidirectional",
        ),
        Rule(
            name="progressive_voicing_assimilation",
            pattern={"context": "C₂ → [αvoice] / C₁[αvoice] _"},
            result={"change": "second consonant matches voicing of first",
                    "examples": ["/s+d/ → [zd]", "cats /kætz/ → [kæts]"]},
            conditions={"environment": "consonant_cluster"},
            weight=0.85,
            direction="forward",
        ),
        Rule(
            name="vowel_nasalisation",
            pattern={"context": "V → [+nasal] / _ N"},
            result={"change": "vowel nasalised before nasal consonant",
                    "examples": ["French: /bon/ → [bɔ̃]"]},
            conditions={"environment": "before_nasal"},
            weight=0.8,
            direction="forward",
        ),

        # --- Dissimilation ---
        Rule(
            name="grassmanns_law",
            pattern={"context": "C[+asp] ... C[+asp] → C[-asp] ... C[+asp]"},
            result={"change": "first aspirate loses aspiration when second exists",
                    "examples": ["Greek: *thi-thē-mi → ti-thē-mi"]},
            conditions={"domain": "Indo-European", "scope": "within_word"},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="lateral_dissimilation",
            pattern={"context": "l ... l → r ... l"},
            result={"change": "first lateral becomes rhotic to avoid repetition",
                    "examples": ["Latin: militāris → *militālis (blocked)"]},
            conditions={"scope": "within_word"},
            weight=0.7,
            direction="forward",
        ),

        # --- Metathesis ---
        Rule(
            name="metathesis_CV",
            pattern={"context": "CV₁C₂ → C₂V₁C"},
            result={"change": "adjacent segments swap positions",
                    "examples": ["Old English: brid → bird",
                                 "Old English: hros → horse"]},
            conditions={"trigger": "ease_of_articulation"},
            weight=0.6,
            direction="bidirectional",
        ),
        Rule(
            name="long_distance_metathesis",
            pattern={"context": "...X...Y... → ...Y...X..."},
            result={"change": "non-adjacent segments swap",
                    "examples": ["Spanish: parabola → palabra"]},
            conditions={"trigger": "perceptual_similarity"},
            weight=0.4,
            direction="bidirectional",
        ),

        # --- Vowel harmony ---
        Rule(
            name="palatal_vowel_harmony",
            pattern={"context": "V[αback] ... V → V[αback] ... V[αback]"},
            result={"change": "suffix vowels match backness of root vowel",
                    "examples": ["Turkish: ev-ler (houses) vs at-lar (horses)"]},
            conditions={"harmony_type": "palatal", "domain": "word"},
            weight=0.95,
            direction="bidirectional",
        ),
        Rule(
            name="labial_vowel_harmony",
            pattern={"context": "V[αround] ... V → V[αround] ... V[αround]"},
            result={"change": "suffix vowels match rounding of root vowel",
                    "examples": ["Turkish: göz-ler → göz-ler",
                                 "göz-ün (of the eye)"]},
            conditions={"harmony_type": "labial", "domain": "word"},
            weight=0.9,
            direction="bidirectional",
        ),

        # --- Lenition / fortition ---
        Rule(
            name="intervocalic_lenition",
            pattern={"context": "C[-son] → C[+cont] / V _ V"},
            result={"change": "stops weaken between vowels",
                    "examples": ["Latin vita → Spanish vida",
                                 "Latin lupum → Spanish lobo"]},
            conditions={"environment": "intervocalic"},
            weight=0.85,
            direction="forward",
        ),
        Rule(
            name="word_initial_fortition",
            pattern={"context": "C[+cont] → C[-cont] / # _"},
            result={"change": "fricatives strengthen to stops word-initially",
                    "examples": ["Proto-Celtic *windos → Welsh gwynt"]},
            conditions={"environment": "word_initial"},
            weight=0.6,
            direction="forward",
        ),

        # --- Epenthesis / syncope ---
        Rule(
            name="epenthesis",
            pattern={"context": "∅ → V / C₁ _ C₂ (complex cluster)"},
            result={"change": "vowel inserted to break up consonant cluster",
                    "examples": ["film → [fɪləm] (Irish English)",
                                 "bnei → benei (Hebrew)"]},
            conditions={"trigger": "sonority_sequencing_violation"},
            weight=0.7,
            direction="forward",
        ),
        Rule(
            name="syncope",
            pattern={"context": "V[−stress] → ∅ / C _ C"},
            result={"change": "unstressed vowel deleted between consonants",
                    "examples": ["Latin: operam → French: oeuvre",
                                 "camera → English camra"]},
            conditions={"trigger": "unstressed_position"},
            weight=0.75,
            direction="forward",
        ),

        # --- Great Vowel Shift (example of chained sound change) ---
        Rule(
            name="great_vowel_shift_raising",
            pattern={"context": "V[+long, -high] → V[+long, +1height]"},
            result={"change": "long vowels raise one step",
                    "chain": ["/aː/→/ɛː/", "/ɛː/→/eː/",
                              "/eː/→/iː/", "/ɔː/→/oː/", "/oː/→/uː/"]},
            conditions={"period": "1400-1700", "language": "English"},
            weight=0.9,
            direction="forward",
        ),
        Rule(
            name="great_vowel_shift_diphthongisation",
            pattern={"context": "V[+long, +high] → VV (diphthong)"},
            result={"change": "high long vowels diphthongise",
                    "chain": ["/iː/ → /aɪ/", "/uː/ → /aʊ/"]},
            conditions={"period": "1400-1700", "language": "English"},
            weight=0.9,
            direction="forward",
        ),
    ]

    return Grammar(
        name="phonological_processes",
        domain="linguistics",
        rules=rules,
    )


# ---------------------------------------------------------------------------
# Morphological grammar
# ---------------------------------------------------------------------------

def build_morphological_grammar() -> Grammar:
    """Build a morphological grammar for word formation.

    Includes:
    - Derivation rules (affixation changing category or meaning)
    - Inflection patterns (tense, aspect, mood, number, case, person)
    - Compounding rules
    - Morphophonological alternations

    Productions model the internal structure of words analogously to
    phrase structure in syntax.
    """

    productions = [
        # Word internal structure
        Production("Word", ["Stem", "InflSuffix"], "morphological"),
        Production("Word", ["Prefix", "Stem"], "morphological"),
        Production("Word", ["Prefix", "Stem", "InflSuffix"], "morphological"),
        Production("Stem", ["Root"], "morphological"),
        Production("Stem", ["DerivPrefix", "Root"], "morphological"),
        Production("Stem", ["Root", "DerivSuffix"], "morphological"),
        Production("Stem", ["DerivPrefix", "Root", "DerivSuffix"], "morphological"),

        # Compounding
        Production("Stem", ["Stem", "Stem"], "morphological"),
        Production("Stem", ["Stem", "LinkMorph", "Stem"], "morphological"),

        # Reduplication (partial)
        Production("Stem", ["RedupPrefix", "Root"], "morphological"),
    ]

    rules = [
        # --- Derivation ---
        Rule(
            name="nominalisation_verb_to_noun",
            pattern={"base": "V", "affix": "-tion/-ment/-ance/-al"},
            result={"derived": "N", "meaning": "event/result of V-ing"},
            conditions={"base_category": "verb"},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="agentive_nominalisation",
            pattern={"base": "V", "affix": "-er/-or/-ist"},
            result={"derived": "N", "meaning": "one who V-s"},
            conditions={"base_category": "verb"},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="adjectivalisation",
            pattern={"base": "N/V", "affix": "-ful/-less/-able/-ous/-ive"},
            result={"derived": "Adj", "meaning": "having quality of base"},
            conditions={"base_category": ["noun", "verb"]},
            weight=0.85,
            direction="bidirectional",
        ),
        Rule(
            name="verbalisation",
            pattern={"base": "N/Adj", "affix": "-ise/-ify/-en"},
            result={"derived": "V", "meaning": "to make/become base"},
            conditions={"base_category": ["noun", "adjective"]},
            weight=0.8,
            direction="bidirectional",
        ),
        Rule(
            name="negative_prefix",
            pattern={"base": "Adj/N/V", "prefix": "un-/in-/dis-/non-"},
            result={"derived": "same_category", "meaning": "negation of base"},
            conditions={"allomorphy": {"in": ["im (before bilabial)",
                                               "il (before l)",
                                               "ir (before r)"]}},
            weight=0.9,
            direction="bidirectional",
        ),

        # --- Inflection ---
        Rule(
            name="english_past_tense_regular",
            pattern={"stem": "V", "suffix": "-ed"},
            result={"features": {"tense": "past", "aspect": "simple"},
                    "allomorphs": ["-t (after voiceless)", "-d (after voiced)",
                                   "-ɪd (after alveolar stop)"]},
            conditions={"verb_class": "regular"},
            weight=0.95,
            direction="bidirectional",
        ),
        Rule(
            name="english_past_tense_ablaut",
            pattern={"stem": "V", "vowel_change": "ablaut"},
            result={"features": {"tense": "past"},
                    "patterns": ["i→a→u (sing/sang/sung)",
                                 "i→o→i (begin/began/begun)",
                                 "iː→ɛ→iː (read/read/read)"]},
            conditions={"verb_class": "strong"},
            weight=0.8,
            direction="bidirectional",
        ),
        Rule(
            name="noun_plural_formation",
            pattern={"stem": "N", "suffix": "-s/-es"},
            result={"features": {"number": "plural"},
                    "allomorphs": ["-s (default)", "-əz (after sibilant)",
                                   "-z (after voiced)"]},
            conditions={"noun_class": "regular"},
            weight=0.95,
            direction="bidirectional",
        ),
        Rule(
            name="case_marking",
            pattern={"stem": "NP", "suffix": "case_morpheme"},
            result={"features": {"case": "?c"},
                    "paradigm": {"nom": "-∅ / -s", "acc": "-m / -n",
                                 "gen": "-s / -is", "dat": "-i / -e"}},
            conditions={"language_type": "fusional_or_agglutinative"},
            weight=0.85,
            direction="bidirectional",
        ),

        # --- Compounding ---
        Rule(
            name="compound_headedness",
            pattern={"structure": "N₁ + N₂"},
            result={"head": "N₂ (right-headed in English)",
                    "meaning": "N₂ that is related to N₁",
                    "examples": ["doghouse = house for dog",
                                 "bookshelf = shelf for books"]},
            conditions={"language": "head_final_compounds"},
            weight=0.9,
            direction="bidirectional",
        ),
        Rule(
            name="compound_stress",
            pattern={"structure": "N₁N₂"},
            result={"stress": "primary stress on N₁ in compounds",
                    "contrast": "phrasal stress on N₂ in NPs",
                    "examples": ["BLACKbird (compound) vs black BIRD (phrase)"]},
            conditions={"phonological_rule": "compound_stress_rule"},
            weight=0.85,
            direction="forward",
        ),

        # --- Morphophonological alternation ---
        Rule(
            name="umlaut",
            pattern={"trigger": "suffix with front vowel",
                     "target": "root back vowel"},
            result={"change": "root vowel fronts",
                    "examples": ["German: Maus → Mäuse",
                                 "English: foot → feet (historical i-umlaut)"]},
            conditions={"process": "vowel_fronting_by_suffix"},
            weight=0.8,
            direction="bidirectional",
        ),
        Rule(
            name="consonant_mutation",
            pattern={"trigger": "morphological environment",
                     "target": "initial consonant"},
            result={"change": "initial consonant undergoes lenition/nasalisation",
                    "examples": ["Welsh: pen (head) → fy mhen (my head)",
                                 "Irish: bó (cow) → mo bhó (my cow)"]},
            conditions={"process": "initial_mutation"},
            weight=0.75,
            direction="bidirectional",
        ),
    ]

    # Strange loop: morphology feeds syntax, syntax requires morphology
    strange_loop = StrangeLoop(
        entry_rule="word_formation",
        cycle=[
            "morphology_builds_words",
            "words_fill_syntactic_slots",
            "syntactic_requirements_trigger_inflection",
            "inflection_is_morphology",
        ],
        level_shift="cyclic",
    )

    return Grammar(
        name="morphological_processes",
        domain="linguistics",
        rules=rules,
        productions=productions,
        sub_grammars=[strange_loop],
    )
