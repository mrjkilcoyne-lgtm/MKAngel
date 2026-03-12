"""French surface templates — evidential hedging via lexical + modal strategies."""

from __future__ import annotations
from . import SurfaceTemplate, TemplateRegistry

FRENCH_TEMPLATES = [
    # Mathematical
    SurfaceTemplate("Le {operation} de {expression} donne {result}.", domain="mathematical", weight=1.2),
    SurfaceTemplate("{expression} vaut {result}.", domain="mathematical"),
    SurfaceTemplate("Le résultat est {result}.", domain="mathematical"),
    SurfaceTemplate("En calculant {operation} de {expression}, on obtient {result}.", domain="mathematical"),
    SurfaceTemplate("{operation}({expression}) = {result}.", domain="mathematical"),

    # Linguistic
    SurfaceTemplate("'{word}' est un {part_of_speech}, défini comme : {definition}.", domain="linguistic", weight=1.2),
    SurfaceTemplate("'{word}' ({phonetic}) signifie : {definition}.", domain="linguistic"),
    SurfaceTemplate("Le mot '{word}' s'emploie comme {role}.", domain="linguistic"),
    SurfaceTemplate("'{word}' — {part_of_speech} : {definition}.", domain="linguistic"),
    SurfaceTemplate("Définition de '{word}' : {definition}.", domain="linguistic"),

    # Biological
    SurfaceTemplate("Le gène {gene} code pour {protein} chez {organism}.", domain="biological", weight=1.2),
    SurfaceTemplate("{gene} produit la protéine {protein} chez {organism}.", domain="biological"),
    SurfaceTemplate("Chez {organism}, {gene} est responsable de {protein}.", domain="biological"),

    # Chemical
    SurfaceTemplate("{compound} ({iupac_name}) a pour formule {formula}, MM={molecular_weight}.", domain="chemical", weight=1.2),
    SurfaceTemplate("La formule moléculaire de {compound} est {formula}.", domain="chemical"),
    SurfaceTemplate("{compound} : {formula}, masse moléculaire {molecular_weight}.", domain="chemical"),

    # Physical
    SurfaceTemplate("La constante {symbol} ({concept}) vaut {value} {unit}.", domain="physical", weight=1.2),
    SurfaceTemplate("{concept} ({symbol}) = {value} {unit}.", domain="physical"),
    SurfaceTemplate("La valeur de {symbol} est {value} {unit}.", domain="physical"),

    # Computational
    SurfaceTemplate("{concept} a une complexité de {complexity}.", domain="computational", weight=1.1),
    SurfaceTemplate("L'algorithme {concept} fonctionne en {complexity}.", domain="computational"),

    # Etymological
    SurfaceTemplate("'{word}' vient du {language} '{root}', signifiant '{meaning}'.", domain="etymological", weight=1.2),
    SurfaceTemplate("L'étymologie de '{word}' : du {origin}, '{meaning}'.", domain="etymological"),

    # Evidential hedging (French: modal + lexical)
    SurfaceTemplate("{content}", domain="general",
                    evidential_source="obs", evidential_confidence="cert"),
    SurfaceTemplate("D'après les observations, {content}.", domain="general",
                    evidential_source="obs", evidential_confidence="prob"),
    SurfaceTemplate("Les données suggèrent que {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="prob"),
    SurfaceTemplate("Il semblerait que {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="poss"),
    SurfaceTemplate("Le calcul confirme que {content}.", domain="general",
                    evidential_source="comp", evidential_confidence="cert"),
    SurfaceTemplate("Selon les rapports, {content}.", domain="general",
                    evidential_source="rep", evidential_confidence="prob"),
    SurfaceTemplate("Il est peu probable que {content}.", domain="general",
                    evidential_source="spec", evidential_confidence="unl"),
]


def register_french(registry: TemplateRegistry) -> None:
    for t in FRENCH_TEMPLATES:
        registry.add(t, language="fr")
