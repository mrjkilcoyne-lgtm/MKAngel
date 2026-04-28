"""
British English dialect grammars for the Grammar Language Model.

Phonological and lexical transformation rules for major British
dialects: RP, Estuary, Scouse, Manc, Home Counties, Geordie.

Each dialect is modelled as a set of phonological rules that
transform RP-standard forms, plus dialect-specific vocabulary.
"""

from __future__ import annotations
from glm.core.grammar import Grammar, Rule, Production, Direction


# ---------------------------------------------------------------------------
# RP (Received Pronunciation) — the reference standard
# ---------------------------------------------------------------------------

def build_rp_grammar() -> Grammar:
    g = Grammar(name="rp_phonology", domain="rp")
    rules = [
        Rule(name="rp_non_rhotic", pattern="/r/ word-final",
             result="deleted", weight=1.0, direction="forward"),
        Rule(name="rp_trap_bath_split", pattern="/a/ in BATH words",
             result="/ɑː/", weight=0.95, direction="forward"),
        Rule(name="rp_long_a", pattern="grass/bath/path/last",
             result="/ɡrɑːs/ /bɑːθ/ /pɑːθ/ /lɑːst/",
             weight=0.95, direction="forward"),
        Rule(name="rp_t_release", pattern="/t/ word-final",
             result="fully released [tʰ]", weight=0.9, direction="forward"),
        Rule(name="rp_linking_r", pattern="vowel + vowel across words",
             result="intrusive/linking /r/", weight=0.8, direction="forward"),
        Rule(name="rp_diphthongs", pattern="FACE/GOAT",
             result="/eɪ/ /əʊ/", weight=0.95, direction="forward"),
    ]
    for r in rules:
        g.add_rule(r)
    return g


# ---------------------------------------------------------------------------
# Estuary English — Thames Estuary, spreading standard
# ---------------------------------------------------------------------------

def build_estuary_grammar() -> Grammar:
    g = Grammar(name="estuary_phonology", domain="estuary")
    rules = [
        Rule(name="est_t_glottal", pattern="/t/ intervocalic or final",
             result="glottal stop [ʔ]", weight=0.85, direction="forward"),
        Rule(name="est_l_vocalise", pattern="/l/ syllable-final",
             result="vocalised [w] or [ʊ]", weight=0.8, direction="forward"),
        Rule(name="est_th_fronting", pattern="/θ/ /ð/",
             result="/f/ /v/ (partial)", weight=0.6, direction="forward"),
        Rule(name="est_yod_coalescence", pattern="/tj/ /dj/",
             result="/tʃ/ /dʒ/ (tune->chune)", weight=0.7, direction="forward"),
        Rule(name="est_non_rhotic", pattern="/r/ word-final",
             result="deleted", weight=0.95, direction="forward"),
    ]
    for r in rules:
        g.add_rule(r)
    return g


# ---------------------------------------------------------------------------
# Scouse — Liverpool
# ---------------------------------------------------------------------------

def build_scouse_grammar() -> Grammar:
    g = Grammar(name="scouse_phonology", domain="scouse")
    rules = [
        Rule(name="sco_t_affricate", pattern="/t/ word-final",
             result="affricate [ts]", weight=0.85, direction="forward"),
        Rule(name="sco_k_affricate", pattern="/k/ word-final",
             result="affricate [kx]", weight=0.8, direction="forward"),
        Rule(name="sco_th_stop", pattern="/θ/ /ð/",
             result="/t/ /d/ (thick->tick)", weight=0.7, direction="forward"),
        Rule(name="sco_fair_diphthong", pattern="SQUARE vowel",
             result="[ɛː] monophthong", weight=0.8, direction="forward"),
        Rule(name="sco_nurse_split", pattern="NURSE vowel",
             result="[ɜː] distinct from SQUARE", weight=0.75, direction="forward"),
        Rule(name="sco_intonation", pattern="declarative",
             result="high rising terminal (uptalk)", weight=0.6, direction="forward"),
    ]
    for r in rules:
        g.add_rule(r)
    return g


def scouse_lexicon_seeds():
    return [
        ("boss",     "Adj", ["scouse"], ""),
        ("sound",    "Adj", ["scouse"], ""),
        ("made_up",  "Adj", ["scouse"], ""),
        ("bevvy",    "N", ["scouse"], ""),
        ("la",       "N", ["scouse"], ""),
        ("lid",      "N", ["scouse"], ""),
        ("lecky",    "N", ["scouse"], ""),
        ("ozzy",     "N", ["scouse"], ""),
        ("jarg",     "Adj", ["scouse"], ""),
        ("devoed",   "Adj", ["scouse"], ""),
        ("scran",    "N", ["scouse"], ""),
        ("dead",     "Adv", ["scouse"], ""),
        ("proper",   "Adv", ["scouse"], ""),
    ]


# ---------------------------------------------------------------------------
# Manc — Manchester
# ---------------------------------------------------------------------------

def build_manc_grammar() -> Grammar:
    g = Grammar(name="manc_phonology", domain="manc")
    rules = [
        Rule(name="manc_short_a", pattern="/ɑː/ in BATH words",
             result="/a/ (flat, northern)", weight=0.9, direction="forward"),
        Rule(name="manc_t_glottal", pattern="/t/ intervocalic",
             result="glottal stop [ʔ]", weight=0.75, direction="forward"),
        Rule(name="manc_ng_coalescence", pattern="/ŋ/ word-final",
             result="/ŋɡ/ (sing-ging)", weight=0.7, direction="forward"),
        Rule(name="manc_strut_foot_nosplit", pattern="STRUT/FOOT vowel",
             result="/ʊ/ (no split: bus=bʊs)", weight=0.9, direction="forward"),
        Rule(name="manc_h_dropping", pattern="/h/ word-initial (some)",
             result="deleted (variable)", weight=0.5, direction="forward"),
    ]
    for r in rules:
        g.add_rule(r)
    return g


def manc_lexicon_seeds():
    return [
        ("mint",     "Adj", ["manc"], ""),
        ("buzzin",   "Adj", ["manc"], ""),
        ("mither",   "V", ["manc"], ""),
        ("ginnel",   "N", ["manc"], ""),
        ("mardy",    "Adj", ["manc"], ""),
        ("brew",     "N", ["manc"], ""),
        ("our_kid",  "N", ["manc"], ""),
        ("nowt",     "Pron", ["manc"], ""),
        ("summat",   "Pron", ["manc"], ""),
        ("dead",     "Adv", ["manc"], ""),
        ("well",     "Adv", ["manc"], ""),
        ("proper",   "Adv", ["manc"], ""),
    ]


# ---------------------------------------------------------------------------
# Home Counties — south-east England (outside London)
# ---------------------------------------------------------------------------

def build_home_counties_grammar() -> Grammar:
    g = Grammar(name="home_counties_phonology", domain="home_counties")
    rules = [
        Rule(name="hc_near_rp", pattern="general",
             result="close to RP with Estuary influence",
             weight=0.9, direction="forward"),
        Rule(name="hc_t_glottal_mild", pattern="/t/ intervocalic",
             result="mild glottal reinforcement", weight=0.6, direction="forward"),
        Rule(name="hc_trap_bath_split", pattern="/a/ in BATH words",
             result="/ɑː/ (same as RP)", weight=0.9, direction="forward"),
        Rule(name="hc_l_vocalise_mild", pattern="/l/ syllable-final",
             result="mild vocalisation", weight=0.5, direction="forward"),
    ]
    for r in rules:
        g.add_rule(r)
    return g


# ---------------------------------------------------------------------------
# Geordie — Newcastle / Tyneside
# ---------------------------------------------------------------------------

def build_geordie_grammar() -> Grammar:
    g = Grammar(name="geordie_phonology", domain="geordie")
    rules = [
        Rule(name="geo_glottal_p_t_k", pattern="/p/ /t/ /k/ intervocalic",
             result="glottalised [ʔp] [ʔt] [ʔk]", weight=0.85, direction="forward"),
        Rule(name="geo_face_monophthong", pattern="FACE vowel",
             result="/eː/ monophthong (not diphthong)", weight=0.9, direction="forward"),
        Rule(name="geo_goat_monophthong", pattern="GOAT vowel",
             result="/oː/ monophthong", weight=0.9, direction="forward"),
        Rule(name="geo_nurse_vowel", pattern="NURSE vowel",
             result="/ɜː/ → [øː]", weight=0.8, direction="forward"),
        Rule(name="geo_h_retention", pattern="/h/ word-initial",
             result="retained (unlike southern)", weight=0.9, direction="forward"),
        Rule(name="geo_ng_split", pattern="/ŋ/ word-final",
             result="/ŋ/ (no /ɡ/)", weight=0.85, direction="forward"),
        Rule(name="geo_r_uvular", pattern="/r/",
             result="Northumbrian burr [ʁ] (historically)",
             weight=0.3, direction="forward"),
    ]
    for r in rules:
        g.add_rule(r)
    return g


def geordie_lexicon_seeds():
    return [
        ("howay",    "V", ["geordie"], ""),
        ("gan",      "V", ["geordie"], ""),
        ("divvent",  "V", ["geordie"], ""),
        ("canny",    "Adj", ["geordie"], ""),
        ("belta",    "Adj", ["geordie"], ""),
        ("lush",     "Adj", ["geordie"], ""),
        ("bairn",    "N", ["geordie"], ""),
        ("gadgie",   "N", ["geordie"], ""),
        ("netty",    "N", ["geordie"], ""),
        ("tab",      "N", ["geordie"], ""),
        ("pet",      "N", ["geordie"], ""),
        ("hinny",    "N", ["geordie"], ""),
        ("wey_aye",  "Adv", ["geordie"], ""),
        ("nee",      "Det", ["geordie"], ""),
        ("us",       "Pron", ["geordie"], ""),
        ("radgie",   "Adj", ["geordie"], ""),
        ("stottie",  "N", ["geordie"], ""),
        ("toon",     "N", ["geordie"], ""),
    ]


# ---------------------------------------------------------------------------
# Convenience: get all dialect grammars
# ---------------------------------------------------------------------------

def all_dialect_grammars() -> dict[str, Grammar]:
    """Return all dialect grammars keyed by name."""
    return {
        "rp": build_rp_grammar(),
        "estuary": build_estuary_grammar(),
        "scouse": build_scouse_grammar(),
        "manc": build_manc_grammar(),
        "home_counties": build_home_counties_grammar(),
        "geordie": build_geordie_grammar(),
    }


def all_dialect_lexicon_seeds() -> list[tuple]:
    """Return all dialect-specific vocabulary."""
    seeds: list[tuple] = []
    seeds.extend(scouse_lexicon_seeds())
    seeds.extend(manc_lexicon_seeds())
    seeds.extend(geordie_lexicon_seeds())
    return seeds
