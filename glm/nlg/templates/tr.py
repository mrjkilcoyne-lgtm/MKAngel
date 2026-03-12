"""Turkish surface templates — MORPHOLOGICAL evidentials (not lexical).

Turkish has grammaticalized evidentiality:
  -DI  = direct evidence (observed, witnessed)
  -mIş = indirect evidence (reported, inferred, hearsay)
  -DIr = general/computed truth
  -(y)EbIlIr = possibility
  -(y)AcAk = future/prediction

This is the showcase language for why MKAngel treats evidentiality as GRAMMAR.
"""

from __future__ import annotations
from . import SurfaceTemplate, TemplateRegistry

TURKISH_TEMPLATES = [
    # Mathematical — -DI (witnessed computation) and -DIr (general truth)
    SurfaceTemplate("{expression} {operation} sonucu {result} bulundu.", domain="mathematical", weight=1.2),  # -DI: we found
    SurfaceTemplate("{expression} = {result}.", domain="mathematical"),
    SurfaceTemplate("Sonuç {result}'dir.", domain="mathematical"),  # -DIr: it IS (general truth)
    SurfaceTemplate("{operation}({expression}) = {result} olarak hesaplandı.", domain="mathematical"),  # was calculated

    # Linguistic
    SurfaceTemplate("'{word}' bir {part_of_speech}'dır, anlamı: {definition}.", domain="linguistic", weight=1.2),  # -DIr
    SurfaceTemplate("'{word}' ({phonetic}): {definition}.", domain="linguistic"),
    SurfaceTemplate("'{word}' kelimesi {role} olarak kullanılır.", domain="linguistic"),
    SurfaceTemplate("'{word}' sözcüğünün tanımı: {definition}.", domain="linguistic"),

    # Biological
    SurfaceTemplate("{organism}'da {gene} geni {protein} proteinini kodlar.", domain="biological", weight=1.2),
    SurfaceTemplate("{gene}, {organism}'da {protein} üretir.", domain="biological"),

    # Chemical
    SurfaceTemplate("{compound}'nin formülü {formula}'dır, MA={molecular_weight}.", domain="chemical", weight=1.2),  # -DIr
    SurfaceTemplate("{compound} ({iupac_name}): {formula}.", domain="chemical"),

    # Physical
    SurfaceTemplate("{symbol} sabiti ({concept}) {value} {unit}'dir.", domain="physical", weight=1.2),  # -DIr
    SurfaceTemplate("{concept} ({symbol}) = {value} {unit}.", domain="physical"),

    # Computational
    SurfaceTemplate("{concept} algoritmasının karmaşıklığı {complexity}'dır.", domain="computational", weight=1.1),

    # Etymological
    SurfaceTemplate("'{word}' sözcüğü {language} '{root}' kökünden gelir, anlamı '{meaning}'.", domain="etymological", weight=1.2),

    # ═══════════════════════════════════════════════════════════════════
    # EVIDENTIAL HEDGING — Turkish MORPHOLOGICAL strategy
    # This is the money: evidentiality baked into verb suffixes
    # ═══════════════════════════════════════════════════════════════════

    # -DI: direct observation, witnessed, certain
    SurfaceTemplate("{content}.", domain="general",
                    evidential_source="obs", evidential_confidence="cert"),
    # -DI + gözlem: observed
    SurfaceTemplate("Gözlemlere göre, {content}.", domain="general",
                    evidential_source="obs", evidential_confidence="prob"),

    # -mIş: indirect evidence, reported, hearsay (THE morphological evidential)
    SurfaceTemplate("Anlaşılan {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="prob"),  # "apparently"
    SurfaceTemplate("{content} gibi görünüyor.", domain="general",
                    evidential_source="inf", evidential_confidence="poss"),  # "it looks like"

    # -DIr: computed/established truth
    SurfaceTemplate("Hesaplama sonucunda {content}.", domain="general",
                    evidential_source="comp", evidential_confidence="cert"),

    # -mIş (hearsay): reported speech
    SurfaceTemplate("Bildirildiğine göre, {content}.", domain="general",
                    evidential_source="rep", evidential_confidence="prob"),  # "reportedly"

    # -(y)EbIlIr: possibility
    SurfaceTemplate("{content} olabilir.", domain="general",
                    evidential_source="spec", evidential_confidence="poss"),  # "might be"

    # Unlikely
    SurfaceTemplate("{content} pek olası değil.", domain="general",
                    evidential_source="spec", evidential_confidence="unl"),
]


def register_turkish(registry: TemplateRegistry) -> None:
    for t in TURKISH_TEMPLATES:
        registry.add(t, language="tr")
