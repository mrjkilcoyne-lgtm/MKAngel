"""German surface templates — evidential hedging via Konjunktiv I (reported speech)."""

from __future__ import annotations
from . import SurfaceTemplate, TemplateRegistry

GERMAN_TEMPLATES = [
    # Mathematical
    SurfaceTemplate("Das {operation} von {expression} ergibt {result}.", domain="mathematical", weight=1.2),
    SurfaceTemplate("{expression} ist gleich {result}.", domain="mathematical"),
    SurfaceTemplate("Das Ergebnis ist {result}.", domain="mathematical"),
    SurfaceTemplate("Berechnung: {operation}({expression}) = {result}.", domain="mathematical"),

    # Linguistic
    SurfaceTemplate("'{word}' ist ein {part_of_speech}, definiert als: {definition}.", domain="linguistic", weight=1.2),
    SurfaceTemplate("'{word}' ({phonetic}) bedeutet: {definition}.", domain="linguistic"),
    SurfaceTemplate("Das Wort '{word}' fungiert als {role}.", domain="linguistic"),
    SurfaceTemplate("Definition von '{word}': {definition}.", domain="linguistic"),

    # Biological
    SurfaceTemplate("Das Gen {gene} kodiert {protein} in {organism}.", domain="biological", weight=1.2),
    SurfaceTemplate("{gene} produziert das Protein {protein} in {organism}.", domain="biological"),

    # Chemical
    SurfaceTemplate("{compound} ({iupac_name}) hat die Formel {formula}, MG={molecular_weight}.", domain="chemical", weight=1.2),
    SurfaceTemplate("Die Molekularformel von {compound} ist {formula}.", domain="chemical"),

    # Physical
    SurfaceTemplate("Die Konstante {symbol} ({concept}) beträgt {value} {unit}.", domain="physical", weight=1.2),
    SurfaceTemplate("{concept} ({symbol}) = {value} {unit}.", domain="physical"),

    # Computational
    SurfaceTemplate("{concept} hat eine Komplexität von {complexity}.", domain="computational", weight=1.1),

    # Etymological
    SurfaceTemplate("'{word}' stammt aus dem {language} '{root}', was '{meaning}' bedeutet.", domain="etymological", weight=1.2),

    # Evidential hedging (German: Konjunktiv I for reported, modal particles)
    SurfaceTemplate("{content}", domain="general",
                    evidential_source="obs", evidential_confidence="cert"),
    SurfaceTemplate("Den Beobachtungen zufolge {content}.", domain="general",
                    evidential_source="obs", evidential_confidence="prob"),
    SurfaceTemplate("Die Daten deuten darauf hin, dass {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="prob"),
    SurfaceTemplate("Es scheint, dass {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="poss"),
    SurfaceTemplate("Die Berechnung bestätigt, dass {content}.", domain="general",
                    evidential_source="comp", evidential_confidence="cert"),
    SurfaceTemplate("Berichten zufolge {content}.", domain="general",
                    evidential_source="rep", evidential_confidence="prob"),
    SurfaceTemplate("Es sei unwahrscheinlich, dass {content}.", domain="general",
                    evidential_source="spec", evidential_confidence="unl"),
]


def register_german(registry: TemplateRegistry) -> None:
    for t in GERMAN_TEMPLATES:
        registry.add(t, language="de")
