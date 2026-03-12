"""Spanish surface templates — evidential hedging via subjunctive + modal."""

from __future__ import annotations
from . import SurfaceTemplate, TemplateRegistry

SPANISH_TEMPLATES = [
    # Mathematical
    SurfaceTemplate("El {operation} de {expression} da {result}.", domain="mathematical", weight=1.2),
    SurfaceTemplate("{expression} es igual a {result}.", domain="mathematical"),
    SurfaceTemplate("El resultado es {result}.", domain="mathematical"),
    SurfaceTemplate("Calculando {operation} de {expression}: {result}.", domain="mathematical"),

    # Linguistic
    SurfaceTemplate("'{word}' es un {part_of_speech}, definido como: {definition}.", domain="linguistic", weight=1.2),
    SurfaceTemplate("'{word}' ({phonetic}) significa: {definition}.", domain="linguistic"),
    SurfaceTemplate("La palabra '{word}' funciona como {role}.", domain="linguistic"),
    SurfaceTemplate("Definición de '{word}': {definition}.", domain="linguistic"),

    # Biological
    SurfaceTemplate("El gen {gene} codifica {protein} en {organism}.", domain="biological", weight=1.2),
    SurfaceTemplate("{gene} produce la proteína {protein} en {organism}.", domain="biological"),

    # Chemical
    SurfaceTemplate("{compound} ({iupac_name}) tiene fórmula {formula}, PM={molecular_weight}.", domain="chemical", weight=1.2),
    SurfaceTemplate("La fórmula molecular de {compound} es {formula}.", domain="chemical"),

    # Physical
    SurfaceTemplate("La constante {symbol} ({concept}) es {value} {unit}.", domain="physical", weight=1.2),
    SurfaceTemplate("{concept} ({symbol}) = {value} {unit}.", domain="physical"),

    # Computational
    SurfaceTemplate("{concept} tiene complejidad {complexity}.", domain="computational", weight=1.1),

    # Etymological
    SurfaceTemplate("'{word}' proviene del {language} '{root}', que significa '{meaning}'.", domain="etymological", weight=1.2),

    # Evidential hedging (Spanish: subjunctive + modal)
    SurfaceTemplate("{content}", domain="general",
                    evidential_source="obs", evidential_confidence="cert"),
    SurfaceTemplate("Según lo observado, {content}.", domain="general",
                    evidential_source="obs", evidential_confidence="prob"),
    SurfaceTemplate("La evidencia sugiere que {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="prob"),
    SurfaceTemplate("Parece que {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="poss"),
    SurfaceTemplate("El cálculo confirma que {content}.", domain="general",
                    evidential_source="comp", evidential_confidence="cert"),
    SurfaceTemplate("Según los informes, {content}.", domain="general",
                    evidential_source="rep", evidential_confidence="prob"),
    SurfaceTemplate("Es poco probable que {content}.", domain="general",
                    evidential_source="spec", evidential_confidence="unl"),
]


def register_spanish(registry: TemplateRegistry) -> None:
    for t in SPANISH_TEMPLATES:
        registry.add(t, language="es")
