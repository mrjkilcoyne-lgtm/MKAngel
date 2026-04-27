"""
Irish Gaelic (Gaeilge) grammar for the Grammar Language Model.

Irish is a VSO Goidelic Celtic language with initial consonant
mutation (lenition and eclipsis), a copula/substantive verb
distinction, and prepositional pronouns.
"""

from __future__ import annotations
from glm.core.grammar import Grammar, Rule, Production, Direction


def build_irish_syntactic_grammar() -> Grammar:
    """Irish syntactic grammar: VSO order with copula distinction."""
    g = Grammar(name="irish_syntax", domain="irish")

    prods = [
        # VSO order
        Production("S", ["VP", "NP_subj", "NP_obj"], "irish_syntax"),
        Production("S", ["VP", "NP_subj"], "irish_syntax"),
        Production("S", ["VP", "NP_subj", "PP"], "irish_syntax"),
        # Copula sentences: Is + Pred + Subj
        Production("S_cop", ["Cop", "NP_pred", "NP_subj"], "irish_syntax"),
        Production("S_cop", ["Cop", "Adj", "NP_subj"], "irish_syntax"),
        # Periphrastic: Tá + NP + ag + VN
        Production("S", ["Aux_ta", "NP_subj", "Particle_ag", "VN"], "irish_syntax"),
        Production("S", ["Aux_ta", "NP_subj", "Particle_ag", "VN", "NP_obj"], "irish_syntax"),
        # Noun phrases
        Production("NP", ["Art", "N"], "irish_syntax"),
        Production("NP", ["N"], "irish_syntax"),
        Production("NP", ["Art", "N", "Adj"], "irish_syntax"),
        Production("NP", ["Pron"], "irish_syntax"),
        # Genitive: N + N (an teach an fhir = the house of the man)
        Production("NP", ["Art", "N", "Art_gen", "N_gen"], "irish_syntax"),
        # VP
        Production("VP", ["V"], "irish_syntax"),
        Production("VP", ["V", "NP"], "irish_syntax"),
        # PP
        Production("PP", ["Prep", "NP"], "irish_syntax"),
        Production("PP", ["PrepPron"], "irish_syntax"),
    ]

    rules = [
        # Lenition after article (feminine nominative singular)
        Rule(name="lenition_fem_art",
             pattern={"trigger": "an + N_fem"},
             result={"mutation": "lenition"},
             weight=0.95, direction="bidirectional"),
        # Eclipsis after preposition + article
        Rule(name="eclipsis_prep_art",
             pattern={"trigger": "Prep + an + N"},
             result={"mutation": "eclipsis"},
             weight=0.9, direction="forward"),
        # Lenition after particles (ní, an, go)
        Rule(name="lenition_particle",
             pattern={"trigger": "particle + V"},
             result={"mutation": "lenition"},
             weight=0.9, direction="forward"),
        # Eclipsis after verbal particles (an, nach, go)
        Rule(name="eclipsis_verbal",
             pattern={"trigger": "an_interr + V"},
             result={"mutation": "eclipsis"},
             weight=0.9, direction="forward"),
    ]

    for p in prods:
        g.add_production(p)
    for r in rules:
        g.add_rule(r)
    return g


def build_irish_morphological_grammar() -> Grammar:
    """Irish morphological grammar: lenition and eclipsis."""
    g = Grammar(name="irish_morphology", domain="irish")

    rules = [
        # === Lenition (Séimhiú) ===
        Rule(name="len_b", pattern="b", result="bh",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="len_c", pattern="c", result="ch",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="len_d", pattern="d", result="dh",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="len_f", pattern="f", result="fh",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="len_g", pattern="g", result="gh",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="len_m", pattern="m", result="mh",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="len_p", pattern="p", result="ph",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="len_s", pattern="s", result="sh",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="len_t", pattern="t", result="th",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),

        # === Eclipsis (Urú) ===
        Rule(name="ecl_b", pattern="b", result="mb",
             conditions={"mutation": "eclipsis"}, weight=1.0, direction="bidirectional"),
        Rule(name="ecl_c", pattern="c", result="gc",
             conditions={"mutation": "eclipsis"}, weight=1.0, direction="bidirectional"),
        Rule(name="ecl_d", pattern="d", result="nd",
             conditions={"mutation": "eclipsis"}, weight=1.0, direction="bidirectional"),
        Rule(name="ecl_f", pattern="f", result="bhf",
             conditions={"mutation": "eclipsis"}, weight=1.0, direction="bidirectional"),
        Rule(name="ecl_g", pattern="g", result="ng",
             conditions={"mutation": "eclipsis"}, weight=1.0, direction="bidirectional"),
        Rule(name="ecl_p", pattern="p", result="bp",
             conditions={"mutation": "eclipsis"}, weight=1.0, direction="bidirectional"),
        Rule(name="ecl_t", pattern="t", result="dt",
             conditions={"mutation": "eclipsis"}, weight=1.0, direction="bidirectional"),
    ]

    for r in rules:
        g.add_rule(r)
    return g


def irish_lexicon_seeds():
    """Core Irish vocabulary."""
    return [
        # Verbs
        ("bi",       "V",    ["irish"], "*bhew-"),
        ("abair",    "V",    ["irish"], ""),
        ("beir",     "V",    ["irish"], "*bher-"),
        ("clois",    "V",    ["irish"], ""),
        ("dean",     "V",    ["irish"], ""),
        ("faigh",    "V",    ["irish"], ""),
        ("feic",     "V",    ["irish"], ""),
        ("ith",      "V",    ["irish"], "*h₁ed-"),
        ("ol",       "V",    ["irish"], ""),
        ("tar",      "V",    ["irish"], ""),
        ("teigh",    "V",    ["irish"], ""),
        ("tabhair",  "V",    ["irish"], ""),
        ("ceannaigh","V",    ["irish"], ""),
        ("siuil",    "V",    ["irish"], ""),
        ("rith",     "V",    ["irish"], ""),
        ("leig",     "V",    ["irish"], ""),
        ("scrios",   "V",    ["irish"], ""),
        ("imir",     "V",    ["irish"], ""),
        ("can",      "V",    ["irish"], "*kan-"),
        ("obair",    "V",    ["irish"], "*werg-"),
        # Nouns
        ("fear",     "N",    ["irish"], "*wiros"),
        ("bean",     "N",    ["irish"], "*gwen-"),
        ("paiste",   "N",    ["irish"], ""),
        ("leanbh",   "N",    ["irish"], ""),
        ("mac",      "N",    ["irish"], "*makwos"),
        ("inion",    "N",    ["irish"], ""),
        ("athair",   "N",    ["irish"], "*ph₂ter-"),
        ("mathair",  "N",    ["irish"], "*meh₂ter-"),
        ("dearthair","N",    ["irish"], "*bhreh₂ter-"),
        ("deirfiur", "N",    ["irish"], "*swesor-"),
        ("teach",    "N",    ["irish"], ""),
        ("uisce",    "N",    ["irish", "chemical"], "*wed-"),
        ("tine",     "N",    ["irish", "chemical"], "*tep-"),
        ("gaoth",    "N",    ["irish", "physics"], "*weh₁-"),
        ("grian",    "N",    ["irish", "physics"], ""),
        ("gealach",  "N",    ["irish", "physics"], ""),
        ("realt",    "N",    ["irish", "physics"], "*h₂ster-"),
        ("crann",    "N",    ["irish", "biological"], "*kwresnom"),
        ("cat",      "N",    ["irish", "biological"], "*kattus"),
        ("cu",       "N",    ["irish", "biological"], "*kwon-"),
        ("ein",      "N",    ["irish", "biological"], ""),
        ("iasc",     "N",    ["irish", "biological"], "*peisk-"),
        ("bia",      "N",    ["irish"], ""),
        ("aran",     "N",    ["irish"], ""),
        ("bainne",   "N",    ["irish"], "*glakt-"),
        ("bothar",   "N",    ["irish"], ""),
        ("abhainn",  "N",    ["irish"], "*h₂ep-"),
        ("muir",     "N",    ["irish"], "*mori-"),
        ("sliabh",   "N",    ["irish"], ""),
        ("croi",     "N",    ["irish", "biological"], "*kerd-"),
        ("ceann",    "N",    ["irish", "biological"], "*kwennom"),
        ("lamh",     "N",    ["irish", "biological"], ""),
        ("suil",     "N",    ["irish", "biological"], ""),
        ("focal",    "N",    ["irish", "linguistic"], ""),
        ("leabhar",  "N",    ["irish"], "*libr-"),
        ("la",       "N",    ["irish"], ""),
        ("oiche",    "N",    ["irish"], "*nokwt-"),
        ("ainm",     "N",    ["irish"], "*h₁nomn-"),
        ("teanga",   "N",    ["irish", "linguistic"], ""),
        # Adjectives
        ("mor",      "Adj",  ["irish"], "*meh₂-"),
        ("beag",     "Adj",  ["irish"], ""),
        ("maith",    "Adj",  ["irish"], ""),
        ("dona",     "Adj",  ["irish"], ""),
        ("sean",     "Adj",  ["irish"], "*sen-"),
        ("og",       "Adj",  ["irish"], ""),
        ("nua",      "Adj",  ["irish"], "*newyo-"),
        ("alainn",   "Adj",  ["irish"], ""),
        ("dorcha",   "Adj",  ["irish"], ""),
        ("geal",     "Adj",  ["irish"], ""),
        ("fuar",     "Adj",  ["irish"], ""),
        ("te",       "Adj",  ["irish"], ""),
        ("laidir",   "Adj",  ["irish"], ""),
        ("lag",      "Adj",  ["irish"], ""),
        ("ard",      "Adj",  ["irish"], ""),
        # Determiners / particles
        ("an",       "Art",  ["irish"], ""),
        ("na",       "Art",  ["irish"], ""),
        ("is",       "Cop",  ["irish"], ""),
        ("ta",       "Aux",  ["irish"], "*bhew-"),
        ("ag",       "Particle", ["irish"], ""),
        ("ni",       "Particle", ["irish"], ""),
        ("go",       "Particle", ["irish"], ""),
        # Pronouns
        ("me",       "Pron", ["irish"], ""),
        ("tu",       "Pron", ["irish"], ""),
        ("se",       "Pron", ["irish"], ""),
        ("si",       "Pron", ["irish"], ""),
        ("muid",     "Pron", ["irish"], ""),
        ("sibh",     "Pron", ["irish"], ""),
        ("siad",     "Pron", ["irish"], ""),
        # Prepositions
        ("i",        "Prep", ["irish"], ""),
        ("ar",       "Prep", ["irish"], ""),
        ("ag",       "Prep", ["irish"], ""),
        ("le",       "Prep", ["irish"], ""),
        ("do",       "Prep", ["irish"], ""),
        ("as",       "Prep", ["irish"], ""),
        ("de",       "Prep", ["irish"], ""),
        ("faoi",     "Prep", ["irish"], ""),
        ("o",        "Prep", ["irish"], ""),
        ("trid",     "Prep", ["irish"], ""),
        # Numbers
        ("aon",      "Num",  ["irish"], ""),
        ("do",       "Num",  ["irish"], "*dwo-"),
        ("tri",      "Num",  ["irish"], "*treyes-"),
        ("ceathair", "Num",  ["irish"], "*kwetwor-"),
        ("cuig",     "Num",  ["irish"], "*penkwe-"),
        ("se",       "Num",  ["irish"], "*sweks-"),
        ("seacht",   "Num",  ["irish"], "*septm-"),
        ("ocht",     "Num",  ["irish"], "*okto-"),
        ("naoi",     "Num",  ["irish"], "*h₁newn-"),
        ("deich",    "Num",  ["irish"], "*dekm-"),
    ]
