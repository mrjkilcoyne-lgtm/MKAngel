"""
Scots Gaelic (Gaidhlig) grammar for the Grammar Language Model.

Scots Gaelic is a VSO Goidelic Celtic language closely related to
Irish, with lenition, a bi/is copula distinction, and prepositional
pronouns. Shares Proto-Celtic roots with Welsh and Irish.
"""

from __future__ import annotations
from glm.core.grammar import Grammar, Rule, Production, Direction


def build_scots_gaelic_syntactic_grammar() -> Grammar:
    g = Grammar(name="scots_gaelic_syntax", domain="scots_gaelic")
    prods = [
        Production("S", ["VP", "NP_subj", "NP_obj"], "sg_syntax"),
        Production("S", ["VP", "NP_subj"], "sg_syntax"),
        Production("S", ["VP", "NP_subj", "PP"], "sg_syntax"),
        Production("S", ["Aux_bi", "NP_subj", "Particle_ag", "VN"], "sg_syntax"),
        Production("S_cop", ["Cop_is", "NP_pred", "NP_subj"], "sg_syntax"),
        Production("NP", ["Art", "N"], "sg_syntax"),
        Production("NP", ["Art", "N", "Adj"], "sg_syntax"),
        Production("NP", ["N"], "sg_syntax"),
        Production("NP", ["Pron"], "sg_syntax"),
        Production("VP", ["V"], "sg_syntax"),
        Production("VP", ["V", "NP"], "sg_syntax"),
        Production("PP", ["Prep", "NP"], "sg_syntax"),
    ]
    rules = [
        Rule(name="sg_lenition_fem_art",
             pattern={"trigger": "an + N_fem"},
             result={"mutation": "lenition"},
             weight=0.95, direction="bidirectional"),
        Rule(name="sg_lenition_past",
             pattern={"trigger": "past_tense + V"},
             result={"mutation": "lenition"},
             weight=0.95, direction="forward"),
    ]
    for p in prods:
        g.add_production(p)
    for r in rules:
        g.add_rule(r)
    return g


def build_scots_gaelic_morphological_grammar() -> Grammar:
    g = Grammar(name="scots_gaelic_morphology", domain="scots_gaelic")
    rules = [
        Rule(name="sg_len_b", pattern="b", result="bh",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="sg_len_c", pattern="c", result="ch",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="sg_len_d", pattern="d", result="dh",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="sg_len_f", pattern="f", result="fh",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="sg_len_g", pattern="g", result="gh",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="sg_len_m", pattern="m", result="mh",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="sg_len_p", pattern="p", result="ph",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="sg_len_s", pattern="s", result="sh",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
        Rule(name="sg_len_t", pattern="t", result="th",
             conditions={"mutation": "lenition"}, weight=1.0, direction="bidirectional"),
    ]
    for r in rules:
        g.add_rule(r)
    return g


def scots_gaelic_lexicon_seeds():
    return [
        ("bi",       "V",    ["scots_gaelic"], "*bhew-"),
        ("rach",     "V",    ["scots_gaelic"], ""),
        ("thig",     "V",    ["scots_gaelic"], ""),
        ("dean",     "V",    ["scots_gaelic"], ""),
        ("abair",    "V",    ["scots_gaelic"], ""),
        ("faic",     "V",    ["scots_gaelic"], ""),
        ("cluinn",   "V",    ["scots_gaelic"], ""),
        ("ith",      "V",    ["scots_gaelic"], "*h₁ed-"),
        ("ol",       "V",    ["scots_gaelic"], ""),
        ("obraich",  "V",    ["scots_gaelic"], "*werg-"),
        ("duine",    "N",    ["scots_gaelic"], "*dhuH-"),
        ("boireannach","N",  ["scots_gaelic"], ""),
        ("balach",   "N",    ["scots_gaelic"], ""),
        ("caileag",  "N",    ["scots_gaelic"], ""),
        ("athair",   "N",    ["scots_gaelic"], "*ph₂ter-"),
        ("mathair",  "N",    ["scots_gaelic"], "*meh₂ter-"),
        ("brathair", "N",    ["scots_gaelic"], "*bhreh₂ter-"),
        ("piuthar",  "N",    ["scots_gaelic"], "*swesor-"),
        ("taigh",    "N",    ["scots_gaelic"], ""),
        ("uisge",    "N",    ["scots_gaelic", "chemical"], "*wed-"),
        ("teine",    "N",    ["scots_gaelic", "chemical"], "*tep-"),
        ("gaoth",    "N",    ["scots_gaelic", "physics"], "*weh₁-"),
        ("grian",    "N",    ["scots_gaelic", "physics"], ""),
        ("gealach",  "N",    ["scots_gaelic", "physics"], ""),
        ("reul",     "N",    ["scots_gaelic", "physics"], "*h₂ster-"),
        ("craobh",   "N",    ["scots_gaelic", "biological"], ""),
        ("cat",      "N",    ["scots_gaelic", "biological"], "*kattus"),
        ("cu",       "N",    ["scots_gaelic", "biological"], "*kwon-"),
        ("muir",     "N",    ["scots_gaelic"], "*mori-"),
        ("beinn",    "N",    ["scots_gaelic"], ""),
        ("mor",      "Adj",  ["scots_gaelic"], "*meh₂-"),
        ("beag",     "Adj",  ["scots_gaelic"], ""),
        ("math",     "Adj",  ["scots_gaelic"], ""),
        ("dona",     "Adj",  ["scots_gaelic"], ""),
        ("sean",     "Adj",  ["scots_gaelic"], "*sen-"),
        ("og",       "Adj",  ["scots_gaelic"], ""),
        ("ur",       "Adj",  ["scots_gaelic"], "*newyo-"),
        ("an",       "Art",  ["scots_gaelic"], ""),
        ("na",       "Art",  ["scots_gaelic"], ""),
        ("tha",      "Aux",  ["scots_gaelic"], "*bhew-"),
        ("is",       "Cop",  ["scots_gaelic"], ""),
        ("mi",       "Pron", ["scots_gaelic"], ""),
        ("thu",      "Pron", ["scots_gaelic"], ""),
        ("e",        "Pron", ["scots_gaelic"], ""),
        ("i",        "Pron", ["scots_gaelic"], ""),
        ("sinn",     "Pron", ["scots_gaelic"], ""),
        ("sibh",     "Pron", ["scots_gaelic"], ""),
        ("iad",      "Pron", ["scots_gaelic"], ""),
        ("ann",      "Prep", ["scots_gaelic"], ""),
        ("air",      "Prep", ["scots_gaelic"], ""),
        ("aig",      "Prep", ["scots_gaelic"], ""),
        ("le",       "Prep", ["scots_gaelic"], ""),
        ("do",       "Prep", ["scots_gaelic"], ""),
        ("bho",      "Prep", ["scots_gaelic"], ""),
        ("gu",       "Prep", ["scots_gaelic"], ""),
        ("ri",       "Prep", ["scots_gaelic"], ""),
    ]
