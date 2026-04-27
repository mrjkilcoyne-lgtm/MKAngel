"""
Urdu (اردو) grammar for the Grammar Language Model.

Urdu is an SOV Indo-Aryan language, mutually intelligible with Hindi
at the colloquial level but with Perso-Arabic vocabulary and Nastaliq
script. Shares the same grammatical structure as Hindi.
"""

from __future__ import annotations
from glm.core.grammar import Grammar, Rule, Production, Direction


def build_urdu_syntactic_grammar() -> Grammar:
    """Urdu shares Hindi's SOV structure."""
    g = Grammar(name="urdu_syntax", domain="urdu")
    prods = [
        Production("S", ["NP_subj", "NP_obj", "VP"], "urdu_syntax"),
        Production("S", ["NP_subj", "VP"], "urdu_syntax"),
        Production("S", ["NP_subj", "PP", "VP"], "urdu_syntax"),
        Production("NP", ["N"], "urdu_syntax"),
        Production("NP", ["Adj", "N"], "urdu_syntax"),
        Production("NP", ["Pron"], "urdu_syntax"),
        Production("NP", ["NP", "Postp"], "urdu_syntax"),
        Production("VP", ["V"], "urdu_syntax"),
        Production("VP", ["V", "Aux"], "urdu_syntax"),
        Production("VP", ["V_comp", "V_light"], "urdu_syntax"),
        Production("PP", ["NP", "Postp"], "urdu_syntax"),
    ]
    rules = [
        Rule(name="urdu_ergative", pattern={"trigger": "perfective + transitive"},
             result={"case": "ergative", "marker": "ne"}, weight=0.9, direction="forward"),
        Rule(name="urdu_izafat", pattern={"trigger": "N + -e + Adj/N"},
             result={"construction": "izafat", "origin": "persian"},
             weight=0.8, direction="forward"),
    ]
    for p in prods:
        g.add_production(p)
    for r in rules:
        g.add_rule(r)
    return g


def urdu_lexicon_seeds():
    """Urdu lexicon — Perso-Arabic vocabulary layer over shared Hindi grammar."""
    return [
        ("hona",     "V", ["urdu"], "*bhew-"),
        ("karna",    "V", ["urdu"], ""),
        ("jana",     "V", ["urdu"], ""),
        ("aana",     "V", ["urdu"], ""),
        ("dekhna",   "V", ["urdu"], ""),
        ("sunna",    "V", ["urdu"], ""),
        ("kehna",    "V", ["urdu"], ""),
        ("likhna",   "V", ["urdu"], ""),
        ("padhna",   "V", ["urdu"], ""),
        ("khana",    "V", ["urdu"], "*h₁ed-"),
        ("peena",    "V", ["urdu"], "*peh₃-"),
        ("samajhna", "V", ["urdu"], ""),
        ("sochna",   "V", ["urdu"], ""),
        ("shakhs",   "N", ["urdu"], ""),
        ("aurat",    "N", ["urdu"], ""),
        ("mard",     "N", ["urdu"], ""),
        ("bachcha",  "N", ["urdu"], ""),
        ("walid",    "N", ["urdu"], ""),
        ("walida",   "N", ["urdu"], ""),
        ("bhai",     "N", ["urdu"], "*bhreh₂ter-"),
        ("behen",    "N", ["urdu"], "*swesor-"),
        ("beti",     "N", ["urdu"], "*dhugh₂ter-"),
        ("beta",     "N", ["urdu"], ""),
        ("ghar",     "N", ["urdu"], ""),
        ("paani",    "N", ["urdu", "chemical"], "*wed-"),
        ("aag",      "N", ["urdu", "chemical"], "*h₁ngwnis"),
        ("hawa",     "N", ["urdu", "physics"], "*weh₁-"),
        ("suraj",    "N", ["urdu", "physics"], "*seh₂wl-"),
        ("chaand",   "N", ["urdu", "physics"], ""),
        ("sitara",   "N", ["urdu", "physics"], "*h₂ster-"),
        ("darakht",  "N", ["urdu", "biological"], ""),
        ("billi",    "N", ["urdu", "biological"], ""),
        ("kutta",    "N", ["urdu", "biological"], ""),
        ("dil",      "N", ["urdu", "biological"], "*kerd-"),
        ("aankh",    "N", ["urdu", "biological"], "*h₃ekw-"),
        ("lafz",     "N", ["urdu", "linguistic"], ""),
        ("kitab",    "N", ["urdu"], ""),
        ("din",      "N", ["urdu"], "*diw-"),
        ("raat",     "N", ["urdu"], "*nokwt-"),
        ("naam",     "N", ["urdu"], "*h₁nomn-"),
        ("zubaan",   "N", ["urdu", "linguistic"], ""),
        ("duniya",   "N", ["urdu"], ""),
        ("waqt",     "N", ["urdu"], ""),
        ("mohabbat", "N", ["urdu"], ""),
        ("khwab",    "N", ["urdu"], ""),
        ("bada",     "Adj", ["urdu"], ""),
        ("chhota",   "Adj", ["urdu"], ""),
        ("achha",    "Adj", ["urdu"], ""),
        ("bura",     "Adj", ["urdu"], ""),
        ("purana",   "Adj", ["urdu"], ""),
        ("naya",     "Adj", ["urdu"], "*newyo-"),
        ("khubsurat","Adj", ["urdu"], ""),
        ("main",     "Pron", ["urdu"], ""),
        ("tum",      "Pron", ["urdu"], ""),
        ("aap",      "Pron", ["urdu"], ""),
        ("voh",      "Pron", ["urdu"], ""),
        ("hum",      "Pron", ["urdu"], ""),
        ("hai",      "Aux", ["urdu"], "*bhew-"),
        ("hain",     "Aux", ["urdu"], "*bhew-"),
        ("tha",      "Aux", ["urdu"], ""),
        ("mein",     "Postp", ["urdu"], ""),
        ("par",      "Postp", ["urdu"], ""),
        ("se",       "Postp", ["urdu"], ""),
        ("ko",       "Postp", ["urdu"], ""),
        ("ne",       "Postp", ["urdu"], ""),
        ("ka",       "Postp", ["urdu"], ""),
        ("aur",      "Conj", ["urdu"], ""),
        ("ya",       "Conj", ["urdu"], ""),
        ("lekin",    "Conj", ["urdu"], ""),
        ("magar",    "Conj", ["urdu"], ""),
        ("ek",       "Num", ["urdu"], ""),
        ("do",       "Num", ["urdu"], "*dwo-"),
        ("teen",     "Num", ["urdu"], "*treyes-"),
        ("chaar",    "Num", ["urdu"], "*kwetwor-"),
        ("paanch",   "Num", ["urdu"], "*penkwe-"),
        ("das",      "Num", ["urdu"], "*dekm-"),
        ("sau",      "Num", ["urdu"], "*kmtom-"),
    ]
