"""
Hindi (हिन्दी) grammar for the Grammar Language Model.

Hindi is an SOV Indo-Aryan language with postpositions, split
ergativity, verb agreement, and a Devanagari script. Shares
Proto-Indo-European roots with Celtic and Germanic languages
through the Indo-Aryan branch.
"""

from __future__ import annotations
from glm.core.grammar import Grammar, Rule, Production, Direction


def build_hindi_syntactic_grammar() -> Grammar:
    """Hindi syntactic grammar: SOV word order."""
    g = Grammar(name="hindi_syntax", domain="hindi")
    prods = [
        # SOV order
        Production("S", ["NP_subj", "NP_obj", "VP"], "hindi_syntax"),
        Production("S", ["NP_subj", "VP"], "hindi_syntax"),
        Production("S", ["NP_subj", "PP", "VP"], "hindi_syntax"),
        Production("S", ["NP_subj", "NP_obj", "PP", "VP"], "hindi_syntax"),
        # NP
        Production("NP", ["N"], "hindi_syntax"),
        Production("NP", ["Adj", "N"], "hindi_syntax"),
        Production("NP", ["Det", "N"], "hindi_syntax"),
        Production("NP", ["Det", "Adj", "N"], "hindi_syntax"),
        Production("NP", ["Pron"], "hindi_syntax"),
        Production("NP", ["NP", "Postp"], "hindi_syntax"),
        # VP: verb-final, auxiliaries follow main verb
        Production("VP", ["V"], "hindi_syntax"),
        Production("VP", ["V", "Aux"], "hindi_syntax"),
        Production("VP", ["Adv", "V"], "hindi_syntax"),
        Production("VP", ["V_comp", "V_light"], "hindi_syntax"),
        # PP: postpositional (NP + postposition)
        Production("PP", ["NP", "Postp"], "hindi_syntax"),
    ]
    rules = [
        Rule(name="hindi_ergative",
             pattern={"trigger": "perfective + transitive"},
             result={"case": "ergative", "marker": "ne"},
             weight=0.9, direction="forward"),
        Rule(name="hindi_verb_agreement",
             pattern={"trigger": "V[tense=?t, gender=?g, number=?n]"},
             result={"constraint": "V agrees with unmarked NP"},
             weight=0.95, direction="bidirectional"),
        Rule(name="hindi_compound_verb",
             pattern={"trigger": "V_stem + V_light"},
             result={"aspect": "completive/intensive"},
             weight=0.8, direction="forward"),
    ]
    for p in prods:
        g.add_production(p)
    for r in rules:
        g.add_rule(r)
    return g


def build_hindi_morphological_grammar() -> Grammar:
    g = Grammar(name="hindi_morphology", domain="hindi")
    rules = [
        # Gender: masculine -a, feminine -i
        Rule(name="hindi_masc_direct", pattern={"suffix": "aa"},
             result={"gender": "masculine", "case": "direct"},
             weight=0.8, direction="bidirectional"),
        Rule(name="hindi_masc_oblique", pattern={"suffix": "e"},
             result={"gender": "masculine", "case": "oblique"},
             weight=0.8, direction="bidirectional"),
        Rule(name="hindi_fem_marker", pattern={"suffix": "ii"},
             result={"gender": "feminine"},
             weight=0.7, direction="bidirectional"),
        # Verb conjugation: stem + tense/aspect/mood
        Rule(name="hindi_present_hab", pattern={"suffix": "taa/tii/te"},
             result={"tense": "present", "aspect": "habitual"},
             weight=0.9, direction="forward"),
        Rule(name="hindi_past_perf", pattern={"suffix": "aa/ii/e"},
             result={"tense": "past", "aspect": "perfective"},
             weight=0.9, direction="forward"),
        Rule(name="hindi_progressive", pattern={"suffix": "rahaa/rahii"},
             result={"aspect": "progressive"},
             weight=0.85, direction="forward"),
        # Plural: masculine -e, feminine -iyaan
        Rule(name="hindi_masc_plural", pattern={"suffix_change": "aa->e"},
             result={"number": "plural", "gender": "masculine"},
             weight=0.8, direction="forward"),
        Rule(name="hindi_fem_plural", pattern={"suffix": "iyaan"},
             result={"number": "plural", "gender": "feminine"},
             weight=0.7, direction="forward"),
    ]
    for r in rules:
        g.add_rule(r)
    return g


def hindi_lexicon_seeds():
    return [
        # Verbs (romanised)
        ("karna",    "V", ["hindi"], ""),
        ("hona",     "V", ["hindi"], "*bhew-"),
        ("jana",     "V", ["hindi"], ""),
        ("aana",     "V", ["hindi"], ""),
        ("dekhna",   "V", ["hindi"], ""),
        ("sunna",    "V", ["hindi"], ""),
        ("bolna",    "V", ["hindi"], ""),
        ("kehna",    "V", ["hindi"], ""),
        ("khana",    "V", ["hindi"], "*h₁ed-"),
        ("peena",    "V", ["hindi"], "*peh₃-"),
        ("likhna",   "V", ["hindi"], ""),
        ("padhna",   "V", ["hindi"], ""),
        ("dena",     "V", ["hindi"], "*deh₃-"),
        ("lena",     "V", ["hindi"], ""),
        ("samajhna", "V", ["hindi"], ""),
        ("jaanna",   "V", ["hindi"], "*gneh₃-"),
        ("sochna",   "V", ["hindi"], ""),
        ("chahna",   "V", ["hindi"], ""),
        ("rakhna",   "V", ["hindi"], ""),
        ("milna",    "V", ["hindi"], ""),
        # Nouns
        ("aadmi",    "N", ["hindi"], ""),
        ("aurat",    "N", ["hindi"], ""),
        ("ladka",    "N", ["hindi"], ""),
        ("ladki",    "N", ["hindi"], ""),
        ("bachcha",  "N", ["hindi"], ""),
        ("pitaji",   "N", ["hindi"], "*ph₂ter-"),
        ("mataji",   "N", ["hindi"], "*meh₂ter-"),
        ("bhai",     "N", ["hindi"], "*bhreh₂ter-"),
        ("behen",    "N", ["hindi"], "*swesor-"),
        ("beti",     "N", ["hindi"], "*dhugh₂ter-"),
        ("beta",     "N", ["hindi"], ""),
        ("ghar",     "N", ["hindi"], ""),
        ("paani",    "N", ["hindi", "chemical"], "*wed-"),
        ("aag",      "N", ["hindi", "chemical"], "*h₁ngwnis"),
        ("hawa",     "N", ["hindi", "physics"], "*weh₁-"),
        ("suraj",    "N", ["hindi", "physics"], "*seh₂wl-"),
        ("chaand",   "N", ["hindi", "physics"], ""),
        ("taara",    "N", ["hindi", "physics"], "*h₂ster-"),
        ("ped",      "N", ["hindi", "biological"], ""),
        ("billi",    "N", ["hindi", "biological"], ""),
        ("kutta",    "N", ["hindi", "biological"], ""),
        ("pakshi",   "N", ["hindi", "biological"], ""),
        ("machli",   "N", ["hindi", "biological"], ""),
        ("khaana",   "N", ["hindi"], ""),
        ("roti",     "N", ["hindi"], ""),
        ("doodh",    "N", ["hindi"], ""),
        ("raasta",   "N", ["hindi"], ""),
        ("nadi",     "N", ["hindi"], ""),
        ("samundar", "N", ["hindi"], ""),
        ("pahad",    "N", ["hindi"], ""),
        ("dil",      "N", ["hindi", "biological"], "*kerd-"),
        ("sir",      "N", ["hindi", "biological"], ""),
        ("haath",    "N", ["hindi", "biological"], ""),
        ("aankh",    "N", ["hindi", "biological"], "*h₃ekw-"),
        ("shabd",    "N", ["hindi", "linguistic"], ""),
        ("kitab",    "N", ["hindi"], ""),
        ("din",      "N", ["hindi"], "*diw-"),
        ("raat",     "N", ["hindi"], "*nokwt-"),
        ("naam",     "N", ["hindi"], "*h₁nomn-"),
        ("bhasha",   "N", ["hindi", "linguistic"], ""),
        # Adjectives
        ("bada",     "Adj", ["hindi"], ""),
        ("chhota",   "Adj", ["hindi"], ""),
        ("achha",    "Adj", ["hindi"], ""),
        ("bura",     "Adj", ["hindi"], ""),
        ("purana",   "Adj", ["hindi"], ""),
        ("naya",     "Adj", ["hindi"], "*newyo-"),
        ("sundar",   "Adj", ["hindi"], ""),
        ("andhera",  "Adj", ["hindi"], ""),
        ("ujla",     "Adj", ["hindi"], ""),
        ("thanda",   "Adj", ["hindi"], ""),
        ("garam",    "Adj", ["hindi"], ""),
        ("mazboot",  "Adj", ["hindi"], ""),
        ("kamzor",   "Adj", ["hindi"], ""),
        ("khush",    "Adj", ["hindi"], ""),
        ("udas",     "Adj", ["hindi"], ""),
        # Postpositions
        ("mein",     "Postp", ["hindi"], ""),
        ("par",      "Postp", ["hindi"], ""),
        ("se",       "Postp", ["hindi"], ""),
        ("ko",       "Postp", ["hindi"], ""),
        ("ke_liye",  "Postp", ["hindi"], ""),
        ("tak",      "Postp", ["hindi"], ""),
        ("ke_saath", "Postp", ["hindi"], ""),
        ("ke_baare_mein", "Postp", ["hindi"], ""),
        ("ne",       "Postp", ["hindi"], ""),
        ("ka",       "Postp", ["hindi"], ""),
        # Pronouns
        ("main",     "Pron", ["hindi"], ""),
        ("tum",      "Pron", ["hindi"], ""),
        ("aap",      "Pron", ["hindi"], ""),
        ("voh",      "Pron", ["hindi"], ""),
        ("yeh",      "Pron", ["hindi"], ""),
        ("hum",      "Pron", ["hindi"], ""),
        ("ve",       "Pron", ["hindi"], ""),
        # Auxiliaries
        ("hai",      "Aux", ["hindi"], "*bhew-"),
        ("hain",     "Aux", ["hindi"], "*bhew-"),
        ("tha",      "Aux", ["hindi"], ""),
        ("thi",      "Aux", ["hindi"], ""),
        ("hoga",     "Aux", ["hindi"], ""),
        # Conjunctions
        ("aur",      "Conj", ["hindi"], ""),
        ("ya",       "Conj", ["hindi"], ""),
        ("lekin",    "Conj", ["hindi"], ""),
        ("kyunki",   "Conj", ["hindi"], ""),
        ("agar",     "Conj", ["hindi"], ""),
        # Numbers
        ("ek",       "Num", ["hindi"], ""),
        ("do",       "Num", ["hindi"], "*dwo-"),
        ("teen",     "Num", ["hindi"], "*treyes-"),
        ("chaar",    "Num", ["hindi"], "*kwetwor-"),
        ("paanch",   "Num", ["hindi"], "*penkwe-"),
        ("cheh",     "Num", ["hindi"], "*sweks-"),
        ("saat",     "Num", ["hindi"], "*septm-"),
        ("aath",     "Num", ["hindi"], "*okto-"),
        ("nau",      "Num", ["hindi"], "*h₁newn-"),
        ("das",      "Num", ["hindi"], "*dekm-"),
        ("sau",      "Num", ["hindi"], "*kmtom-"),
    ]
