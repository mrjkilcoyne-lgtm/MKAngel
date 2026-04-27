"""
Bengali (বাংলা) grammar for the Grammar Language Model.

Bengali is an SOV Indo-Aryan language spoken in Bangladesh and
West Bengal. Has a classifier system, no grammatical gender for
nouns, and a rich verb morphology.
"""

from __future__ import annotations
from glm.core.grammar import Grammar, Rule, Production, Direction


def build_bengali_syntactic_grammar() -> Grammar:
    g = Grammar(name="bengali_syntax", domain="bengali")
    prods = [
        Production("S", ["NP_subj", "NP_obj", "VP"], "bengali_syntax"),
        Production("S", ["NP_subj", "VP"], "bengali_syntax"),
        Production("NP", ["N"], "bengali_syntax"),
        Production("NP", ["Adj", "N"], "bengali_syntax"),
        Production("NP", ["N", "Classifier"], "bengali_syntax"),
        Production("NP", ["Num", "Classifier", "N"], "bengali_syntax"),
        Production("NP", ["Pron"], "bengali_syntax"),
        Production("NP", ["NP", "Postp"], "bengali_syntax"),
        Production("VP", ["V"], "bengali_syntax"),
        Production("VP", ["V", "Aux"], "bengali_syntax"),
        Production("PP", ["NP", "Postp"], "bengali_syntax"),
    ]
    rules = [
        Rule(name="bn_classifier_required",
             pattern={"trigger": "Num + N"},
             result={"requires": "classifier"},
             weight=0.9, direction="forward"),
        Rule(name="bn_verb_honorific",
             pattern={"trigger": "V[person=2]"},
             result={"register": "tui/tumi/apni"},
             weight=0.85, direction="forward"),
    ]
    for p in prods:
        g.add_production(p)
    for r in rules:
        g.add_rule(r)
    return g


def bengali_lexicon_seeds():
    return [
        ("howa",     "V", ["bengali"], "*bhew-"),
        ("kora",     "V", ["bengali"], ""),
        ("jaowa",    "V", ["bengali"], ""),
        ("asha",     "V", ["bengali"], ""),
        ("dekha",    "V", ["bengali"], ""),
        ("shona",    "V", ["bengali"], ""),
        ("bola",     "V", ["bengali"], ""),
        ("lekha",    "V", ["bengali"], ""),
        ("pora",     "V", ["bengali"], ""),
        ("khawa",    "V", ["bengali"], "*h₁ed-"),
        ("pan_kora", "V", ["bengali"], "*peh₃-"),
        ("bujha",    "V", ["bengali"], ""),
        ("janla",    "V", ["bengali"], "*gneh₃-"),
        ("manush",   "N", ["bengali"], ""),
        ("mohila",   "N", ["bengali"], ""),
        ("purush",   "N", ["bengali"], ""),
        ("shishu",   "N", ["bengali"], ""),
        ("baba",     "N", ["bengali"], "*ph₂ter-"),
        ("ma",       "N", ["bengali"], "*meh₂ter-"),
        ("bhai",     "N", ["bengali"], "*bhreh₂ter-"),
        ("bon",      "N", ["bengali"], "*swesor-"),
        ("meye",     "N", ["bengali"], "*dhugh₂ter-"),
        ("chhele",   "N", ["bengali"], ""),
        ("bari",     "N", ["bengali"], ""),
        ("jol",      "N", ["bengali", "chemical"], "*wed-"),
        ("agun",     "N", ["bengali", "chemical"], "*h₁ngwnis"),
        ("batash",   "N", ["bengali", "physics"], "*weh₁-"),
        ("surjo",    "N", ["bengali", "physics"], "*seh₂wl-"),
        ("chand",    "N", ["bengali", "physics"], ""),
        ("tara",     "N", ["bengali", "physics"], "*h₂ster-"),
        ("gachh",    "N", ["bengali", "biological"], ""),
        ("biral",    "N", ["bengali", "biological"], ""),
        ("kukur",    "N", ["bengali", "biological"], ""),
        ("hridoy",   "N", ["bengali", "biological"], "*kerd-"),
        ("chokh",    "N", ["bengali", "biological"], "*h₃ekw-"),
        ("shobdo",   "N", ["bengali", "linguistic"], ""),
        ("boi",      "N", ["bengali"], ""),
        ("din",      "N", ["bengali"], "*diw-"),
        ("raat",     "N", ["bengali"], "*nokwt-"),
        ("naam",     "N", ["bengali"], "*h₁nomn-"),
        ("bhasha",   "N", ["bengali", "linguistic"], ""),
        ("boro",     "Adj", ["bengali"], ""),
        ("chhoto",   "Adj", ["bengali"], ""),
        ("bhalo",    "Adj", ["bengali"], ""),
        ("kharap",   "Adj", ["bengali"], ""),
        ("notun",    "Adj", ["bengali"], "*newyo-"),
        ("purano",   "Adj", ["bengali"], ""),
        ("shundor",  "Adj", ["bengali"], ""),
        ("ami",      "Pron", ["bengali"], ""),
        ("tumi",     "Pron", ["bengali"], ""),
        ("apni",     "Pron", ["bengali"], ""),
        ("she",      "Pron", ["bengali"], ""),
        ("amra",     "Pron", ["bengali"], ""),
        ("tara",     "Pron", ["bengali"], ""),
        ("e",        "Prep", ["bengali"], ""),
        ("theke",    "Postp", ["bengali"], ""),
        ("jonno",    "Postp", ["bengali"], ""),
        ("diye",     "Postp", ["bengali"], ""),
        ("ebong",    "Conj", ["bengali"], ""),
        ("ba",       "Conj", ["bengali"], ""),
        ("kintu",    "Conj", ["bengali"], ""),
        ("ek",       "Num", ["bengali"], ""),
        ("dui",      "Num", ["bengali"], "*dwo-"),
        ("tin",      "Num", ["bengali"], "*treyes-"),
        ("char",     "Num", ["bengali"], "*kwetwor-"),
        ("panch",    "Num", ["bengali"], "*penkwe-"),
        ("dosh",     "Num", ["bengali"], "*dekm-"),
        ("sho",      "Num", ["bengali"], "*kmtom-"),
        ("ta",       "Classifier", ["bengali"], ""),
        ("jon",      "Classifier", ["bengali"], ""),
        ("khana",    "Classifier", ["bengali"], ""),
    ]
