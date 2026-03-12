"""Welsh (Cymraeg) surface templates — evidential hedging via periphrastic + particle strategies.

Welsh uses a VSO (Verb-Subject-Object) word order and has:
  - 'mae' (is/be) for present facts
  - 'roedd' (was) for past observation
  - 'efallai' (perhaps) for possibility
  - 'yn ôl pob sôn' (reportedly) for hearsay
  - 'mae'n debyg' (it's likely) for inference
  - Soft mutation after certain particles (grammaticalised evidentiality trace)

Welsh treats knowledge claims through periphrastic constructions rather
than dedicated morphological evidentials, but the particle system and
mutation patterns encode speaker stance grammatically.
"""

from __future__ import annotations
from . import SurfaceTemplate, TemplateRegistry

WELSH_TEMPLATES = [
    # ═══════════════════════════════════════════════════════════════════
    # MATHEMATICAL — 5 templates
    # ═══════════════════════════════════════════════════════════════════
    SurfaceTemplate("Mae {operation} {expression} yn rhoi {result}.", domain="mathematical", weight=1.2),
    SurfaceTemplate("{expression} = {result}.", domain="mathematical"),
    SurfaceTemplate("Y canlyniad yw {result}.", domain="mathematical"),
    SurfaceTemplate("Wrth gyfrifo {operation} o {expression}, cawn {result}.", domain="mathematical"),
    SurfaceTemplate("{operation}({expression}) = {result}.", domain="mathematical"),

    # ═══════════════════════════════════════════════════════════════════
    # LINGUISTIC — 5 templates
    # ═══════════════════════════════════════════════════════════════════
    SurfaceTemplate("Mae '{word}' yn {part_of_speech}, sy'n golygu: {definition}.", domain="linguistic", weight=1.2),
    SurfaceTemplate("'{word}' ({phonetic}): {definition}.", domain="linguistic"),
    SurfaceTemplate("Defnyddir '{word}' fel {role}.", domain="linguistic"),
    SurfaceTemplate("Ystyr '{word}': {definition}.", domain="linguistic"),
    SurfaceTemplate("Mae'r gair '{word}' yn golygu {definition}.", domain="linguistic"),

    # ═══════════════════════════════════════════════════════════════════
    # BIOLOGICAL — 3 templates
    # ═══════════════════════════════════════════════════════════════════
    SurfaceTemplate("Mae'r genyn {gene} yn codio {protein} yn {organism}.", domain="biological", weight=1.2),
    SurfaceTemplate("Mae {gene} yn cynhyrchu'r protein {protein} yn {organism}.", domain="biological"),
    SurfaceTemplate("Yn {organism}, mae {gene} yn gyfrifol am {protein}.", domain="biological"),

    # ═══════════════════════════════════════════════════════════════════
    # CHEMICAL — 3 templates
    # ═══════════════════════════════════════════════════════════════════
    SurfaceTemplate("Fformiwla {compound} ({iupac_name}) yw {formula}, MC={molecular_weight}.", domain="chemical", weight=1.2),
    SurfaceTemplate("Mae fformiwla foleciwlaidd {compound} yn {formula}.", domain="chemical"),
    SurfaceTemplate("{compound}: {formula}, màs moleciwlaidd {molecular_weight}.", domain="chemical"),

    # ═══════════════════════════════════════════════════════════════════
    # PHYSICAL — 3 templates
    # ═══════════════════════════════════════════════════════════════════
    SurfaceTemplate("Mae'r cysonyn {symbol} ({concept}) yn {value} {unit}.", domain="physical", weight=1.2),
    SurfaceTemplate("{concept} ({symbol}) = {value} {unit}.", domain="physical"),
    SurfaceTemplate("Gwerth {symbol} yw {value} {unit}.", domain="physical"),

    # ═══════════════════════════════════════════════════════════════════
    # COMPUTATIONAL — 2 templates
    # ═══════════════════════════════════════════════════════════════════
    SurfaceTemplate("Mae cymhlethdod {concept} yn {complexity}.", domain="computational", weight=1.1),
    SurfaceTemplate("Mae algorithm {concept} yn rhedeg mewn {complexity}.", domain="computational"),

    # ═══════════════════════════════════════════════════════════════════
    # ETYMOLOGICAL — 2 templates
    # ═══════════════════════════════════════════════════════════════════
    SurfaceTemplate("Daw '{word}' o'r {language} '{root}', sy'n golygu '{meaning}'.", domain="etymological", weight=1.2),
    SurfaceTemplate("Tarddiad '{word}': o'r {origin}, '{meaning}'.", domain="etymological"),

    # ═══════════════════════════════════════════════════════════════════
    # EVIDENTIAL HEDGING — Welsh periphrastic + particle strategy
    # ═══════════════════════════════════════════════════════════════════

    # mae (present fact, observed, certain)
    SurfaceTemplate("{content}.", domain="general",
                    evidential_source="obs", evidential_confidence="cert"),
    # Observed + probable
    SurfaceTemplate("Yn ôl yr arsylwadau, {content}.", domain="general",
                    evidential_source="obs", evidential_confidence="prob"),

    # mae'n debyg (inference, probable) — "it's likely"
    SurfaceTemplate("Mae'n debyg bod {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="prob"),
    # mae'n ymddangos (inference, possible) — "it appears"
    SurfaceTemplate("Mae'n ymddangos bod {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="poss"),

    # Computed / calculated truth
    SurfaceTemplate("Mae'r cyfrifiad yn cadarnhau bod {content}.", domain="general",
                    evidential_source="comp", evidential_confidence="cert"),

    # yn ôl pob sôn (reported, hearsay) — "reportedly"
    SurfaceTemplate("Yn ôl pob sôn, {content}.", domain="general",
                    evidential_source="rep", evidential_confidence="prob"),

    # efallai (possibility) — "perhaps"
    SurfaceTemplate("Efallai bod {content}.", domain="general",
                    evidential_source="spec", evidential_confidence="poss"),

    # Unlikely
    SurfaceTemplate("Mae'n annhebygol bod {content}.", domain="general",
                    evidential_source="spec", evidential_confidence="unl"),
]


def register_welsh(registry: TemplateRegistry) -> None:
    for t in WELSH_TEMPLATES:
        registry.add(t, language="cy")
