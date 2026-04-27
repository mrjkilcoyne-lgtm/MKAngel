"""
Polish (Polski) grammar for the Grammar Language Model.

Polish is an SVO Slavic language with free word order (pragmatically
driven), seven grammatical cases, three genders, aspect pairs for
verbs, and a rich consonant cluster system.
"""

from __future__ import annotations
from glm.core.grammar import Grammar, Rule, Production, Direction


def build_polish_syntactic_grammar() -> Grammar:
    g = Grammar(name="polish_syntax", domain="polish")
    prods = [
        Production("S", ["NP_subj", "VP"], "polish_syntax"),
        Production("S", ["NP_subj", "VP", "NP_obj"], "polish_syntax"),
        Production("S", ["VP", "NP_subj"], "polish_syntax"),
        Production("S", ["NP_subj", "VP", "PP"], "polish_syntax"),
        Production("NP", ["N"], "polish_syntax"),
        Production("NP", ["Adj", "N"], "polish_syntax"),
        Production("NP", ["Det", "N"], "polish_syntax"),
        Production("NP", ["Det", "Adj", "N"], "polish_syntax"),
        Production("NP", ["Pron"], "polish_syntax"),
        Production("NP", ["NP", "NP_gen"], "polish_syntax"),
        Production("VP", ["V"], "polish_syntax"),
        Production("VP", ["V", "NP"], "polish_syntax"),
        Production("VP", ["Aux", "V_inf"], "polish_syntax"),
        Production("VP", ["Adv", "V"], "polish_syntax"),
        Production("PP", ["Prep", "NP"], "polish_syntax"),
    ]
    rules = [
        Rule(name="polish_case_assignment",
             pattern={"trigger": "Prep[governs=?case]"},
             result={"case": "governed_by_preposition"},
             weight=0.9, direction="forward"),
        Rule(name="polish_aspect_pair",
             pattern={"trigger": "V[aspect=imperfective]"},
             result={"pair": "perfective_counterpart"},
             weight=0.85, direction="bidirectional"),
        Rule(name="polish_negation",
             pattern={"trigger": "nie + V"},
             result={"polarity": "negative", "case_shift": "gen"},
             weight=0.95, direction="forward"),
    ]
    for p in prods:
        g.add_production(p)
    for r in rules:
        g.add_rule(r)
    return g


def build_polish_morphological_grammar() -> Grammar:
    g = Grammar(name="polish_morphology", domain="polish")
    rules = [
        # 7 cases
        Rule(name="pl_nominative", pattern={"case": "nom"},
             result={"function": "subject"}, weight=1.0, direction="bidirectional"),
        Rule(name="pl_genitive", pattern={"case": "gen"},
             result={"function": "possession/negation/partitive"},
             weight=0.9, direction="bidirectional"),
        Rule(name="pl_dative", pattern={"case": "dat"},
             result={"function": "indirect_object"}, weight=0.9, direction="bidirectional"),
        Rule(name="pl_accusative", pattern={"case": "acc"},
             result={"function": "direct_object"}, weight=0.95, direction="bidirectional"),
        Rule(name="pl_instrumental", pattern={"case": "ins"},
             result={"function": "instrument/means"}, weight=0.9, direction="bidirectional"),
        Rule(name="pl_locative", pattern={"case": "loc"},
             result={"function": "location"}, weight=0.9, direction="bidirectional"),
        Rule(name="pl_vocative", pattern={"case": "voc"},
             result={"function": "address"}, weight=0.8, direction="bidirectional"),
        # Aspect
        Rule(name="pl_perfective_prefix",
             pattern={"prefix": "na/za/po/prze/wy/z"},
             result={"aspect": "perfective"},
             weight=0.85, direction="forward"),
    ]
    for r in rules:
        g.add_rule(r)
    return g


def polish_lexicon_seeds():
    return [
        ("byc",      "V", ["polish"], "*bhew-"),
        ("miec",     "V", ["polish"], ""),
        ("robic",    "V", ["polish"], ""),
        ("isc",      "V", ["polish"], ""),
        ("mowic",    "V", ["polish"], ""),
        ("widziec",  "V", ["polish"], "*weid-"),
        ("slyszec",  "V", ["polish"], ""),
        ("jesc",     "V", ["polish"], "*h₁ed-"),
        ("pic",      "V", ["polish"], "*peh₃-"),
        ("pisac",    "V", ["polish"], ""),
        ("czytac",   "V", ["polish"], ""),
        ("dawac",    "V", ["polish"], "*deh₃-"),
        ("brac",     "V", ["polish"], ""),
        ("pracowac", "V", ["polish"], "*werg-"),
        ("spac",     "V", ["polish"], ""),
        ("znac",     "V", ["polish"], "*gneh₃-"),
        ("kochac",   "V", ["polish"], ""),
        ("myslec",   "V", ["polish"], ""),
        ("chciec",   "V", ["polish"], ""),
        ("czlowiek", "N", ["polish"], ""),
        ("kobieta",  "N", ["polish"], ""),
        ("mezczyzna","N", ["polish"], ""),
        ("dziecko",  "N", ["polish"], ""),
        ("chlopiec", "N", ["polish"], ""),
        ("dziewczyna","N", ["polish"], ""),
        ("ojciec",   "N", ["polish"], "*ph₂ter-"),
        ("matka",    "N", ["polish"], "*meh₂ter-"),
        ("brat",     "N", ["polish"], "*bhreh₂ter-"),
        ("siostra",  "N", ["polish"], "*swesor-"),
        ("syn",      "N", ["polish"], "*suHnus-"),
        ("corka",    "N", ["polish"], "*dhugh₂ter-"),
        ("dom",      "N", ["polish"], ""),
        ("woda",     "N", ["polish", "chemical"], "*wed-"),
        ("ogien",    "N", ["polish", "chemical"], "*h₁ngwnis"),
        ("wiatr",    "N", ["polish", "physics"], "*weh₁-"),
        ("slonce",   "N", ["polish", "physics"], "*seh₂wl-"),
        ("ksiezyc",  "N", ["polish", "physics"], ""),
        ("gwiazda",  "N", ["polish", "physics"], "*h₂ster-"),
        ("drzewo",   "N", ["polish", "biological"], "*deru-"),
        ("kot",      "N", ["polish", "biological"], "*kattus"),
        ("pies",     "N", ["polish", "biological"], ""),
        ("ptak",     "N", ["polish", "biological"], ""),
        ("ryba",     "N", ["polish", "biological"], ""),
        ("chleb",    "N", ["polish"], ""),
        ("mleko",    "N", ["polish"], "*h₂melg-"),
        ("morze",    "N", ["polish"], "*mori-"),
        ("gora",     "N", ["polish"], ""),
        ("serce",    "N", ["polish", "biological"], "*kerd-"),
        ("glowa",    "N", ["polish", "biological"], ""),
        ("reka",     "N", ["polish", "biological"], ""),
        ("oko",      "N", ["polish", "biological"], "*h₃ekw-"),
        ("slowo",    "N", ["polish", "linguistic"], ""),
        ("ksiazka",  "N", ["polish"], ""),
        ("dzien",    "N", ["polish"], "*diw-"),
        ("noc",      "N", ["polish"], "*nokwt-"),
        ("imie",     "N", ["polish"], "*h₁nomn-"),
        ("jezyk",    "N", ["polish", "linguistic"], ""),
        ("duzy",     "Adj", ["polish"], ""),
        ("maly",     "Adj", ["polish"], ""),
        ("dobry",    "Adj", ["polish"], ""),
        ("zly",      "Adj", ["polish"], ""),
        ("stary",    "Adj", ["polish"], "*sen-"),
        ("mlody",    "Adj", ["polish"], ""),
        ("nowy",     "Adj", ["polish"], "*newyo-"),
        ("piekny",   "Adj", ["polish"], ""),
        ("ciemny",   "Adj", ["polish"], ""),
        ("jasny",    "Adj", ["polish"], ""),
        ("zimny",    "Adj", ["polish"], ""),
        ("goracy",   "Adj", ["polish"], ""),
        ("silny",    "Adj", ["polish"], ""),
        ("slaby",    "Adj", ["polish"], ""),
        ("ten",      "Det", ["polish"], ""),
        ("ta",       "Det", ["polish"], ""),
        ("to",       "Det", ["polish"], ""),
        ("ja",       "Pron", ["polish"], ""),
        ("ty",       "Pron", ["polish"], ""),
        ("on",       "Pron", ["polish"], ""),
        ("ona",      "Pron", ["polish"], ""),
        ("ono",      "Pron", ["polish"], ""),
        ("my",       "Pron", ["polish"], ""),
        ("wy",       "Pron", ["polish"], ""),
        ("oni",      "Pron", ["polish"], ""),
        ("w",        "Prep", ["polish"], ""),
        ("na",       "Prep", ["polish"], ""),
        ("z",        "Prep", ["polish"], ""),
        ("do",       "Prep", ["polish"], ""),
        ("od",       "Prep", ["polish"], ""),
        ("dla",      "Prep", ["polish"], ""),
        ("przez",    "Prep", ["polish"], ""),
        ("po",       "Prep", ["polish"], ""),
        ("i",        "Conj", ["polish"], ""),
        ("lub",      "Conj", ["polish"], ""),
        ("ale",      "Conj", ["polish"], ""),
        ("bo",       "Conj", ["polish"], ""),
        ("jeden",    "Num", ["polish"], ""),
        ("dwa",      "Num", ["polish"], "*dwo-"),
        ("trzy",     "Num", ["polish"], "*treyes-"),
        ("cztery",   "Num", ["polish"], "*kwetwor-"),
        ("piec",     "Num", ["polish"], "*penkwe-"),
        ("szesc",    "Num", ["polish"], "*sweks-"),
        ("siedem",   "Num", ["polish"], "*septm-"),
        ("osiem",    "Num", ["polish"], "*okto-"),
        ("dziewiec", "Num", ["polish"], "*h₁newn-"),
        ("dziesiec", "Num", ["polish"], "*dekm-"),
        ("sto",      "Num", ["polish"], "*kmtom-"),
    ]
