"""
Welsh (Cymraeg) grammar for the Grammar Language Model.

Welsh is a VSO (Verb-Subject-Object) Brythonic Celtic language with
initial consonant mutation, inflected prepositions, and a rich
system of periphrastic verb constructions.

Two registers: formal (literary) and colloquial (spoken).
"""

from __future__ import annotations
from glm.core.grammar import Grammar, Rule, Production, Direction


# ---------------------------------------------------------------------------
# Syntactic grammar — Welsh word order and phrase structure
# ---------------------------------------------------------------------------

def build_welsh_syntactic_grammar() -> Grammar:
    """Welsh syntactic grammar: VSO order with mutation triggers."""
    g = Grammar(name="welsh_syntax", domain="welsh")

    # Welsh is VSO: Verb Subject Object
    # But colloquial Welsh often uses periphrastic: S + bod + yn + VN
    prods = [
        # Formal VSO
        Production("S", ["VP", "NP_subj", "NP_obj"], "welsh_syntax"),
        Production("S", ["VP", "NP_subj"], "welsh_syntax"),
        Production("S", ["VP", "NP_subj", "PP"], "welsh_syntax"),
        # Periphrastic (colloquial): Mae'r gath yn eistedd
        Production("S", ["Aux_bod", "NP_subj", "Particle_yn", "VN"], "welsh_syntax"),
        Production("S", ["Aux_bod", "NP_subj", "Particle_yn", "VN", "NP_obj"], "welsh_syntax"),
        Production("S", ["Aux_bod", "NP_subj", "Particle_yn", "Adj"], "welsh_syntax"),
        # Emphatic/identification: NP sydd yn VN
        Production("S", ["NP_subj", "Rel_sydd", "Particle_yn", "VN"], "welsh_syntax"),
        # Noun phrases
        Production("NP", ["Art", "N"], "welsh_syntax"),
        Production("NP", ["N"], "welsh_syntax"),
        Production("NP", ["Art", "N", "Adj"], "welsh_syntax"),
        Production("NP", ["N", "Adj"], "welsh_syntax"),
        Production("NP", ["Pron"], "welsh_syntax"),
        Production("NP", ["NP", "PP"], "welsh_syntax"),
        # Genitive: N + N (head-first: car y dyn = car the man)
        Production("NP", ["N", "Art", "N"], "welsh_syntax"),
        # Verb phrases
        Production("VP", ["V"], "welsh_syntax"),
        Production("VP", ["V", "NP"], "welsh_syntax"),
        Production("VP", ["V", "NP", "PP"], "welsh_syntax"),
        # Prepositional phrases
        Production("PP", ["Prep", "NP"], "welsh_syntax"),
        Production("PP", ["PrepInfl"], "welsh_syntax"),
    ]

    rules = [
        # Soft mutation after article (feminine singular)
        Rule(
            name="soft_mutation_fem_art",
            pattern={"trigger": "Art_fem + N"},
            result={"mutation": "soft", "applies_to": "N_initial"},
            weight=0.95,
            direction="bidirectional",
        ),
        # Soft mutation after yn (predicative)
        Rule(
            name="soft_mutation_yn",
            pattern={"trigger": "yn + Adj"},
            result={"mutation": "soft", "applies_to": "Adj_initial"},
            weight=0.9,
            direction="forward",
        ),
        # Nasal mutation after fy (my)
        Rule(
            name="nasal_mutation_fy",
            pattern={"trigger": "fy + N"},
            result={"mutation": "nasal", "applies_to": "N_initial"},
            weight=0.95,
            direction="bidirectional",
        ),
        # Aspirate mutation after ei (her)
        Rule(
            name="aspirate_mutation_ei",
            pattern={"trigger": "ei + N"},
            result={"mutation": "aspirate", "applies_to": "N_initial"},
            weight=0.9,
            direction="bidirectional",
        ),
        # Soft mutation after prepositions
        Rule(
            name="soft_mutation_prep",
            pattern={"trigger": "Prep + N"},
            result={"mutation": "soft", "applies_to": "N_initial"},
            weight=0.85,
            direction="forward",
        ),
        # Bod conjugation agreement
        Rule(
            name="bod_present_agreement",
            pattern={"trigger": "Aux_bod[tense=present]"},
            result={"forms": {
                "1sg": "rydw i", "2sg": "rwyt ti", "3sg_m": "mae e",
                "3sg_f": "mae hi", "1pl": "rydyn ni", "2pl": "rydych chi",
                "3pl": "maen nhw",
            }},
            weight=1.0,
            direction="bidirectional",
        ),
    ]

    for p in prods:
        g.add_production(p)
    for r in rules:
        g.add_rule(r)
    return g


# ---------------------------------------------------------------------------
# Morphological grammar — mutations and inflections
# ---------------------------------------------------------------------------

def build_welsh_morphological_grammar() -> Grammar:
    """Welsh morphological grammar: three mutation systems."""
    g = Grammar(name="welsh_morphology", domain="welsh")

    rules = [
        # === Soft Mutation (Treiglad Meddal) ===
        Rule(name="soft_p", pattern="p", result="b",
             conditions={"mutation": "soft"}, weight=1.0, direction="bidirectional"),
        Rule(name="soft_t", pattern="t", result="d",
             conditions={"mutation": "soft"}, weight=1.0, direction="bidirectional"),
        Rule(name="soft_c", pattern="c", result="g",
             conditions={"mutation": "soft"}, weight=1.0, direction="bidirectional"),
        Rule(name="soft_b", pattern="b", result="f",
             conditions={"mutation": "soft"}, weight=1.0, direction="bidirectional"),
        Rule(name="soft_d", pattern="d", result="dd",
             conditions={"mutation": "soft"}, weight=1.0, direction="bidirectional"),
        Rule(name="soft_g", pattern="g", result="",
             conditions={"mutation": "soft"}, weight=1.0, direction="bidirectional"),
        Rule(name="soft_m", pattern="m", result="f",
             conditions={"mutation": "soft"}, weight=1.0, direction="bidirectional"),
        Rule(name="soft_ll", pattern="ll", result="l",
             conditions={"mutation": "soft"}, weight=1.0, direction="bidirectional"),
        Rule(name="soft_rh", pattern="rh", result="r",
             conditions={"mutation": "soft"}, weight=1.0, direction="bidirectional"),

        # === Nasal Mutation (Treiglad Trwynol) ===
        Rule(name="nasal_p", pattern="p", result="mh",
             conditions={"mutation": "nasal"}, weight=1.0, direction="bidirectional"),
        Rule(name="nasal_t", pattern="t", result="nh",
             conditions={"mutation": "nasal"}, weight=1.0, direction="bidirectional"),
        Rule(name="nasal_c", pattern="c", result="ngh",
             conditions={"mutation": "nasal"}, weight=1.0, direction="bidirectional"),
        Rule(name="nasal_b", pattern="b", result="m",
             conditions={"mutation": "nasal"}, weight=1.0, direction="bidirectional"),
        Rule(name="nasal_d", pattern="d", result="n",
             conditions={"mutation": "nasal"}, weight=1.0, direction="bidirectional"),
        Rule(name="nasal_g", pattern="g", result="ng",
             conditions={"mutation": "nasal"}, weight=1.0, direction="bidirectional"),

        # === Aspirate Mutation (Treiglad Llaes) ===
        Rule(name="aspirate_p", pattern="p", result="ph",
             conditions={"mutation": "aspirate"}, weight=1.0, direction="bidirectional"),
        Rule(name="aspirate_t", pattern="t", result="th",
             conditions={"mutation": "aspirate"}, weight=1.0, direction="bidirectional"),
        Rule(name="aspirate_c", pattern="c", result="ch",
             conditions={"mutation": "aspirate"}, weight=1.0, direction="bidirectional"),

        # === Plural formation ===
        Rule(name="plural_au", pattern={"suffix": ""},
             result={"suffix": "au"}, conditions={"plural": "type_au"},
             weight=0.7, direction="forward"),
        Rule(name="plural_iau", pattern={"suffix": ""},
             result={"suffix": "iau"}, conditions={"plural": "type_iau"},
             weight=0.5, direction="forward"),
        Rule(name="plural_oedd", pattern={"suffix": ""},
             result={"suffix": "oedd"}, conditions={"plural": "type_oedd"},
             weight=0.4, direction="forward"),

        # === Gender (masculine/feminine) affects mutation triggers ===
        Rule(name="fem_noun_triggers_soft",
             pattern={"gender": "feminine", "after": "article"},
             result={"mutation": "soft"},
             weight=0.95, direction="forward"),
    ]

    for r in rules:
        g.add_rule(r)
    return g


# ---------------------------------------------------------------------------
# Phonological grammar
# ---------------------------------------------------------------------------

def build_welsh_phonological_grammar() -> Grammar:
    """Welsh phonological rules: vowel system, stress, ll/ch/dd."""
    g = Grammar(name="welsh_phonology", domain="welsh")

    rules = [
        # Welsh stress: penultimate syllable (almost always)
        Rule(name="penultimate_stress",
             pattern={"context": "polysyllabic word"},
             result={"stress": "penultimate"},
             weight=0.95, direction="forward"),
        # Welsh ll = voiceless lateral fricative
        Rule(name="ll_realisation",
             pattern="ll", result="/ɬ/",
             weight=1.0, direction="forward"),
        # Welsh ch = voiceless velar fricative
        Rule(name="ch_realisation",
             pattern="ch", result="/x/",
             weight=1.0, direction="forward"),
        # Welsh dd = voiced dental fricative
        Rule(name="dd_realisation",
             pattern="dd", result="/ð/",
             weight=1.0, direction="forward"),
        # Welsh ff = /f/, f = /v/
        Rule(name="ff_realisation",
             pattern="ff", result="/f/",
             weight=1.0, direction="forward"),
        Rule(name="f_realisation",
             pattern="f", result="/v/",
             weight=1.0, direction="forward"),
        # Welsh vowels: a e i o u w y
        Rule(name="w_vowel",
             pattern={"context": "w as vowel"},
             result="/ʉ/",
             weight=0.8, direction="forward"),
        Rule(name="y_clear",
             pattern={"context": "y in final syllable"},
             result="/ɨ/",
             weight=0.8, direction="forward"),
        Rule(name="y_obscure",
             pattern={"context": "y in non-final syllable"},
             result="/ə/",
             weight=0.8, direction="forward"),
    ]

    for r in rules:
        g.add_rule(r)
    return g


# ---------------------------------------------------------------------------
# Welsh lexicon entries
# ---------------------------------------------------------------------------

def welsh_lexicon_seeds():
    """Core Welsh vocabulary with categories and proto-roots."""
    return [
        # Verbs (berfau)
        ("bod",      "V",    ["welsh"], "*bhew-"),
        ("mynd",     "V",    ["welsh"], "*men-"),
        ("dod",      "V",    ["welsh"], "*deh-"),
        ("gwneud",   "V",    ["welsh"], "*gwen-"),
        ("cael",     "V",    ["welsh"], ""),
        ("gweld",    "V",    ["welsh"], "*wel-"),
        ("dweud",    "V",    ["welsh"], ""),
        ("gwybod",   "V",    ["welsh"], "*weid-"),
        ("gallu",    "V",    ["welsh"], ""),
        ("rhoi",     "V",    ["welsh"], ""),
        ("cymryd",   "V",    ["welsh"], ""),
        ("eistedd",  "V",    ["welsh"], ""),
        ("siarad",   "V",    ["welsh"], ""),
        ("bwyta",    "V",    ["welsh"], ""),
        ("yfed",     "V",    ["welsh"], ""),
        ("cerdded",  "V",    ["welsh"], ""),
        ("rhedeg",   "V",    ["welsh"], ""),
        ("cysgu",    "V",    ["welsh"], ""),
        ("gweithio", "V",    ["welsh"], "*werg-"),
        ("dysgu",    "V",    ["welsh"], ""),
        ("darllain", "V",    ["welsh"], ""),
        ("ysgrifennu","V",   ["welsh"], ""),
        ("canu",     "V",    ["welsh"], "*kan-"),
        ("hoffi",    "V",    ["welsh"], ""),
        ("caru",     "V",    ["welsh"], "*kar-"),
        # Nouns (enwau)
        ("dyn",      "N",    ["welsh"], "*dhuH-"),
        ("dynes",    "N",    ["welsh"], "*dhuH-"),
        ("plentyn",  "N",    ["welsh"], ""),
        ("plant",    "N",    ["welsh"], ""),
        ("bachgen",  "N",    ["welsh"], ""),
        ("merch",    "N",    ["welsh"], ""),
        ("tad",      "N",    ["welsh"], "*ph₂ter-"),
        ("mam",      "N",    ["welsh"], "*meh₂ter-"),
        ("brawd",    "N",    ["welsh"], "*bhreh₂ter-"),
        ("chwaer",   "N",    ["welsh"], "*swesor-"),
        ("ty",       "N",    ["welsh"], ""),
        ("ysgol",    "N",    ["welsh"], ""),
        ("dwr",      "N",    ["welsh", "chemical"], "*wed-"),
        ("tan",      "N",    ["welsh", "chemical"], "*tep-"),
        ("gwynt",    "N",    ["welsh", "physics"], "*weh₁-"),
        ("haul",     "N",    ["welsh", "physics"], "*seh₂wl-"),
        ("lleuad",   "N",    ["welsh", "physics"], "*lewk-"),
        ("seren",    "N",    ["welsh", "physics"], "*h₂ster-"),
        ("coeden",   "N",    ["welsh", "biological"], "*deru-"),
        ("cath",     "N",    ["welsh", "biological"], "*kattus"),
        ("ci",       "N",    ["welsh", "biological"], "*kwon-"),
        ("aderyn",   "N",    ["welsh", "biological"], ""),
        ("pysgodyn", "N",    ["welsh", "biological"], "*peisk-"),
        ("bara",     "N",    ["welsh"], ""),
        ("llaeth",   "N",    ["welsh"], "*glakt-"),
        ("caws",     "N",    ["welsh"], "*kawseo-"),
        ("bwyd",     "N",    ["welsh"], ""),
        ("ffordd",   "N",    ["welsh"], ""),
        ("afon",     "N",    ["welsh"], "*h₂ep-"),
        ("mor",      "N",    ["welsh"], "*mori-"),
        ("mynydd",   "N",    ["welsh"], "*monti-"),
        ("calon",    "N",    ["welsh", "biological"], "*kerd-"),
        ("pen",      "N",    ["welsh", "biological"], "*kwennom"),
        ("llaw",     "N",    ["welsh", "biological"], ""),
        ("llygad",   "N",    ["welsh", "biological"], ""),
        ("enw",      "N",    ["welsh"], "*h₁nomn-"),
        ("iaith",    "N",    ["welsh", "linguistic"], ""),
        ("gair",     "N",    ["welsh", "linguistic"], ""),
        ("llyfr",    "N",    ["welsh"], "*libr-"),
        ("amser",    "N",    ["welsh"], ""),
        ("dydd",     "N",    ["welsh"], "*diw-"),
        ("nos",      "N",    ["welsh"], "*nokwt-"),
        ("blwyddyn", "N",    ["welsh"], ""),
        # Adjectives (ansoddeiriau)
        ("mawr",     "Adj",  ["welsh"], "*meh₂-"),
        ("bach",     "Adj",  ["welsh"], ""),
        ("da",       "Adj",  ["welsh"], ""),
        ("drwg",     "Adj",  ["welsh"], ""),
        ("hen",      "Adj",  ["welsh"], "*sen-"),
        ("ifanc",    "Adj",  ["welsh"], ""),
        ("newydd",   "Adj",  ["welsh"], "*newyo-"),
        ("hardd",    "Adj",  ["welsh"], ""),
        ("tywyll",   "Adj",  ["welsh"], ""),
        ("golau",    "Adj",  ["welsh"], ""),
        ("oer",      "Adj",  ["welsh"], ""),
        ("poeth",    "Adj",  ["welsh"], ""),
        ("cyflym",   "Adj",  ["welsh"], ""),
        ("araf",     "Adj",  ["welsh"], ""),
        ("cryf",     "Adj",  ["welsh"], ""),
        ("gwan",     "Adj",  ["welsh"], ""),
        ("hapus",    "Adj",  ["welsh"], ""),
        ("trist",    "Adj",  ["welsh"], ""),
        ("prydferth","Adj",  ["welsh"], ""),
        ("hir",      "Adj",  ["welsh"], ""),
        ("byr",      "Adj",  ["welsh"], ""),
        # Determiners / particles
        ("y",        "Art",  ["welsh"], ""),
        ("yr",       "Art",  ["welsh"], ""),
        ("r",        "Art",  ["welsh"], ""),
        ("un",       "Det",  ["welsh"], ""),
        ("yn",       "Particle", ["welsh"], ""),
        ("wedi",     "Particle", ["welsh"], ""),
        ("mae",      "Aux",  ["welsh"], "*bhew-"),
        ("roedd",    "Aux",  ["welsh"], "*bhew-"),
        ("bydd",     "Aux",  ["welsh"], "*bhew-"),
        # Pronouns
        ("fi",       "Pron", ["welsh"], ""),
        ("ti",       "Pron", ["welsh"], ""),
        ("fe",       "Pron", ["welsh"], ""),
        ("hi",       "Pron", ["welsh"], ""),
        ("ni",       "Pron", ["welsh"], ""),
        ("chi",      "Pron", ["welsh"], ""),
        ("nhw",      "Pron", ["welsh"], ""),
        # Prepositions (arddodiaid)
        ("yn",       "Prep", ["welsh"], ""),
        ("ar",       "Prep", ["welsh"], ""),
        ("i",        "Prep", ["welsh"], ""),
        ("o",        "Prep", ["welsh"], ""),
        ("am",       "Prep", ["welsh"], ""),
        ("gan",      "Prep", ["welsh"], ""),
        ("gyda",     "Prep", ["welsh"], ""),
        ("heb",      "Prep", ["welsh"], ""),
        ("dan",      "Prep", ["welsh"], ""),
        ("dros",     "Prep", ["welsh"], ""),
        ("trwy",     "Prep", ["welsh"], ""),
        ("rhwng",    "Prep", ["welsh"], ""),
        # Conjunctions
        ("a",        "Conj", ["welsh"], ""),
        ("ac",       "Conj", ["welsh"], ""),
        ("ond",      "Conj", ["welsh"], ""),
        ("neu",      "Conj", ["welsh"], ""),
        ("achos",    "Conj", ["welsh"], ""),
        ("os",       "Conj", ["welsh"], ""),
        # Numbers (rhifau)
        ("un",       "Num",  ["welsh"], ""),
        ("dau",      "Num",  ["welsh"], "*dwo-"),
        ("tri",      "Num",  ["welsh"], "*treyes-"),
        ("pedwar",   "Num",  ["welsh"], "*kwetwor-"),
        ("pump",     "Num",  ["welsh"], "*penkwe-"),
        ("chwech",   "Num",  ["welsh"], "*sweks-"),
        ("saith",    "Num",  ["welsh"], "*septm-"),
        ("wyth",     "Num",  ["welsh"], "*okto-"),
        ("naw",      "Num",  ["welsh"], "*h₁newn-"),
        ("deg",      "Num",  ["welsh"], "*dekm-"),
        ("cant",     "Num",  ["welsh"], "*kmtom-"),
    ]
