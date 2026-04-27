"""
Punjabi (ਪੰਜਾਬੀ / پنجابی) grammar for the Grammar Language Model.

Punjabi is an SOV Indo-Aryan language with tonal distinctions
(unique among Indo-Aryan), two genders, postpositions, and
Gurmukhi (India) / Shahmukhi (Pakistan) scripts.
"""

from __future__ import annotations
from glm.core.grammar import Grammar, Rule, Production, Direction


def build_punjabi_syntactic_grammar() -> Grammar:
    g = Grammar(name="punjabi_syntax", domain="punjabi")
    prods = [
        Production("S", ["NP_subj", "NP_obj", "VP"], "punjabi_syntax"),
        Production("S", ["NP_subj", "VP"], "punjabi_syntax"),
        Production("NP", ["N"], "punjabi_syntax"),
        Production("NP", ["Adj", "N"], "punjabi_syntax"),
        Production("NP", ["Pron"], "punjabi_syntax"),
        Production("NP", ["NP", "Postp"], "punjabi_syntax"),
        Production("VP", ["V"], "punjabi_syntax"),
        Production("VP", ["V", "Aux"], "punjabi_syntax"),
        Production("PP", ["NP", "Postp"], "punjabi_syntax"),
    ]
    rules = [
        Rule(name="pbi_tone_from_mutation",
             pattern={"trigger": "historical_voiced_aspirate"},
             result={"tone": "low_rising/high_falling"},
             weight=0.85, direction="forward"),
        Rule(name="pbi_ergative",
             pattern={"trigger": "perfective + transitive"},
             result={"case": "ergative", "marker": "ne"},
             weight=0.9, direction="forward"),
    ]
    for p in prods:
        g.add_production(p)
    for r in rules:
        g.add_rule(r)
    return g


def punjabi_lexicon_seeds():
    return [
        ("hona",     "V", ["punjabi"], "*bhew-"),
        ("karna",    "V", ["punjabi"], ""),
        ("jana",     "V", ["punjabi"], ""),
        ("auna",     "V", ["punjabi"], ""),
        ("dekhna",   "V", ["punjabi"], ""),
        ("sunna",    "V", ["punjabi"], ""),
        ("bolna",    "V", ["punjabi"], ""),
        ("likhna",   "V", ["punjabi"], ""),
        ("padhna",   "V", ["punjabi"], ""),
        ("khana",    "V", ["punjabi"], "*h₁ed-"),
        ("peena",    "V", ["punjabi"], "*peh₃-"),
        ("jaanana",  "V", ["punjabi"], "*gneh₃-"),
        ("banda",    "N", ["punjabi"], ""),
        ("tivi",     "N", ["punjabi"], ""),
        ("munda",    "N", ["punjabi"], ""),
        ("kudi",     "N", ["punjabi"], ""),
        ("bachcha",  "N", ["punjabi"], ""),
        ("peo",      "N", ["punjabi"], "*ph₂ter-"),
        ("maa",      "N", ["punjabi"], "*meh₂ter-"),
        ("bhra",     "N", ["punjabi"], "*bhreh₂ter-"),
        ("bhain",    "N", ["punjabi"], "*swesor-"),
        ("dhee",     "N", ["punjabi"], "*dhugh₂ter-"),
        ("putt",     "N", ["punjabi"], ""),
        ("ghar",     "N", ["punjabi"], ""),
        ("paani",    "N", ["punjabi", "chemical"], "*wed-"),
        ("agg",      "N", ["punjabi", "chemical"], "*h₁ngwnis"),
        ("hava",     "N", ["punjabi", "physics"], "*weh₁-"),
        ("suraj",    "N", ["punjabi", "physics"], "*seh₂wl-"),
        ("chand",    "N", ["punjabi", "physics"], ""),
        ("tara",     "N", ["punjabi", "physics"], "*h₂ster-"),
        ("darakht",  "N", ["punjabi", "biological"], ""),
        ("billi",    "N", ["punjabi", "biological"], ""),
        ("kutta",    "N", ["punjabi", "biological"], ""),
        ("dil",      "N", ["punjabi", "biological"], "*kerd-"),
        ("akh",      "N", ["punjabi", "biological"], "*h₃ekw-"),
        ("lafz",     "N", ["punjabi", "linguistic"], ""),
        ("kitab",    "N", ["punjabi"], ""),
        ("din",      "N", ["punjabi"], "*diw-"),
        ("raat",     "N", ["punjabi"], "*nokwt-"),
        ("naa",      "N", ["punjabi"], "*h₁nomn-"),
        ("boli",     "N", ["punjabi", "linguistic"], ""),
        ("vadda",    "Adj", ["punjabi"], ""),
        ("chhota",   "Adj", ["punjabi"], ""),
        ("changga",  "Adj", ["punjabi"], ""),
        ("manda",    "Adj", ["punjabi"], ""),
        ("nava",     "Adj", ["punjabi"], "*newyo-"),
        ("purana",   "Adj", ["punjabi"], ""),
        ("sohna",    "Adj", ["punjabi"], ""),
        ("main",     "Pron", ["punjabi"], ""),
        ("tusi",     "Pron", ["punjabi"], ""),
        ("oh",       "Pron", ["punjabi"], ""),
        ("assi",     "Pron", ["punjabi"], ""),
        ("hai",      "Aux", ["punjabi"], "*bhew-"),
        ("si",       "Aux", ["punjabi"], ""),
        ("vich",     "Postp", ["punjabi"], ""),
        ("te",       "Postp", ["punjabi"], ""),
        ("to",       "Postp", ["punjabi"], ""),
        ("nu",       "Postp", ["punjabi"], ""),
        ("ne",       "Postp", ["punjabi"], ""),
        ("da",       "Postp", ["punjabi"], ""),
        ("te",       "Conj", ["punjabi"], ""),
        ("ja",       "Conj", ["punjabi"], ""),
        ("par",      "Conj", ["punjabi"], ""),
        ("ikk",      "Num", ["punjabi"], ""),
        ("do",       "Num", ["punjabi"], "*dwo-"),
        ("tinn",     "Num", ["punjabi"], "*treyes-"),
        ("chaar",    "Num", ["punjabi"], "*kwetwor-"),
        ("panj",     "Num", ["punjabi"], "*penkwe-"),
        ("das",      "Num", ["punjabi"], "*dekm-"),
        ("sau",      "Num", ["punjabi"], "*kmtom-"),
    ]
