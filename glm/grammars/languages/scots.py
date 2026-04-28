"""
Scots (Braid Scots) grammar for the Grammar Language Model.

Scots is a West Germanic language (not a dialect of English) with
its own vocabulary, phonology, and grammar. Distinct from Scots
Gaelic (which is Celtic). SVO word order with distinctive modal
system and vowel inventory.
"""

from __future__ import annotations
from glm.core.grammar import Grammar, Rule, Production, Direction


def build_scots_syntactic_grammar() -> Grammar:
    g = Grammar(name="scots_syntax", domain="scots")
    prods = [
        Production("S", ["NP_subj", "VP"], "scots_syntax"),
        Production("S", ["NP_subj", "VP", "NP_obj"], "scots_syntax"),
        Production("S", ["NP_subj", "VP", "PP"], "scots_syntax"),
        Production("NP", ["Art", "N"], "scots_syntax"),
        Production("NP", ["Art", "Adj", "N"], "scots_syntax"),
        Production("NP", ["Pron"], "scots_syntax"),
        Production("NP", ["N"], "scots_syntax"),
        Production("VP", ["V"], "scots_syntax"),
        Production("VP", ["V", "NP"], "scots_syntax"),
        Production("VP", ["Aux", "V"], "scots_syntax"),
        Production("VP", ["V", "Adv"], "scots_syntax"),
        Production("PP", ["Prep", "NP"], "scots_syntax"),
    ]
    rules = [
        Rule(name="scots_negation_nae",
             pattern={"trigger": "V + nae"},
             result={"polarity": "negative", "form": "nae/naw"},
             weight=0.9, direction="forward"),
        Rule(name="scots_plural_s",
             pattern={"trigger": "N[plural]"},
             result={"suffix": "-s (but een/shoon irregular)"},
             weight=0.8, direction="forward"),
    ]
    for p in prods:
        g.add_production(p)
    for r in rules:
        g.add_rule(r)
    return g


def scots_lexicon_seeds():
    return [
        ("be",       "V", ["scots"], "*bhew-"),
        ("hae",      "V", ["scots"], ""),
        ("gang",     "V", ["scots"], ""),
        ("ken",      "V", ["scots"], "*gneh₃-"),
        ("speir",    "V", ["scots"], ""),
        ("greet",    "V", ["scots"], ""),
        ("bide",     "V", ["scots"], ""),
        ("dinna",    "V", ["scots"], ""),
        ("cannae",   "V", ["scots"], ""),
        ("winna",    "V", ["scots"], ""),
        ("tak",      "V", ["scots"], ""),
        ("gie",      "V", ["scots"], "*deh₃-"),
        ("ettle",    "V", ["scots"], ""),
        ("fash",     "V", ["scots"], ""),
        ("bairn",    "N", ["scots"], ""),
        ("lassie",   "N", ["scots"], ""),
        ("laddie",   "N", ["scots"], ""),
        ("wean",     "N", ["scots"], ""),
        ("mither",   "N", ["scots"], "*meh₂ter-"),
        ("faither",  "N", ["scots"], "*ph₂ter-"),
        ("brither",  "N", ["scots"], "*bhreh₂ter-"),
        ("dochter",  "N", ["scots"], "*dhugh₂ter-"),
        ("hoose",    "N", ["scots"], ""),
        ("kirk",     "N", ["scots"], ""),
        ("brae",     "N", ["scots"], ""),
        ("burn",     "N", ["scots"], ""),
        ("loch",     "N", ["scots"], ""),
        ("ben",      "N", ["scots"], ""),
        ("watter",   "N", ["scots", "chemical"], "*wed-"),
        ("stane",    "N", ["scots"], ""),
        ("hert",     "N", ["scots", "biological"], "*kerd-"),
        ("een",      "N", ["scots", "biological"], "*h₃ekw-"),
        ("heid",     "N", ["scots", "biological"], ""),
        ("haund",    "N", ["scots", "biological"], ""),
        ("nicht",    "N", ["scots"], "*nokwt-"),
        ("morn",     "N", ["scots"], ""),
        ("auld",     "Adj", ["scots"], ""),
        ("bonnie",   "Adj", ["scots"], ""),
        ("braw",     "Adj", ["scots"], ""),
        ("wee",      "Adj", ["scots"], ""),
        ("muckle",   "Adj", ["scots"], ""),
        ("canny",    "Adj", ["scots"], ""),
        ("dreich",   "Adj", ["scots"], ""),
        ("gallus",   "Adj", ["scots"], ""),
        ("glaikit",  "Adj", ["scots"], ""),
        ("scunner",  "Adj", ["scots"], ""),
        ("blether",  "N", ["scots"], ""),
        ("the",      "Art", ["scots"], ""),
        ("a",        "Art", ["scots"], ""),
        ("ah",       "Pron", ["scots"], ""),
        ("ye",       "Pron", ["scots"], ""),
        ("he",       "Pron", ["scots"], ""),
        ("she",      "Pron", ["scots"], ""),
        ("we",       "Pron", ["scots"], ""),
        ("they",     "Pron", ["scots"], ""),
        ("aye",      "Adv", ["scots"], ""),
        ("nae",      "Adv", ["scots"], ""),
        ("noo",      "Adv", ["scots"], ""),
        ("awfy",     "Adv", ["scots"], ""),
        ("tae",      "Prep", ["scots"], ""),
        ("frae",     "Prep", ["scots"], ""),
        ("wi",       "Prep", ["scots"], ""),
        ("an",       "Conj", ["scots"], ""),
        ("but",      "Conj", ["scots"], ""),
        ("ane",      "Num", ["scots"], ""),
        ("twa",      "Num", ["scots"], "*dwo-"),
        ("three",    "Num", ["scots"], "*treyes-"),
    ]
