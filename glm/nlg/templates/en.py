"""English surface templates for all 7 domain grammars + evidential hedging.

The true timeline is grammatically the case — evidential markers map to
language-appropriate realisations. English uses lexical hedging; other
languages will use morphological strategies (Turkish -mIs, Quechua -mi/-si).
"""

from __future__ import annotations

from . import SurfaceTemplate, TemplateRegistry


ENGLISH_TEMPLATES = [
    # --- Mathematical ---
    SurfaceTemplate("{result}", domain="mathematical", slots={"result"}),
    SurfaceTemplate("The result is {result}.", domain="mathematical"),
    SurfaceTemplate("Solving this gives {result}.", domain="mathematical"),
    SurfaceTemplate("By {method}, we get {result}.", domain="mathematical"),
    SurfaceTemplate("The {operation} of {operand} yields {result}.", domain="mathematical"),
    # API-friendly: Newton API returns expression + result
    SurfaceTemplate("The {operation} of {expression} gives {result}.", domain="mathematical", weight=1.2),
    SurfaceTemplate("{expression} evaluates to {result}.", domain="mathematical"),

    # --- Linguistic ---
    SurfaceTemplate("The word '{word}' {analysis}.", domain="linguistic"),
    SurfaceTemplate("In this context, '{word}' functions as {role}.", domain="linguistic"),
    SurfaceTemplate("The phrase structure is {structure}.", domain="linguistic"),
    SurfaceTemplate("Morphologically, '{word}' breaks down as {breakdown}.", domain="linguistic"),
    SurfaceTemplate("The grammatical pattern here is {pattern}.", domain="linguistic"),
    # API-friendly: Dictionary API returns definition + part_of_speech
    SurfaceTemplate("'{word}' is a {part_of_speech}, defined as: {definition}.", domain="linguistic", weight=1.2),
    SurfaceTemplate("'{word}' ({phonetic}) means: {definition}.", domain="linguistic", weight=1.1),

    # --- Biological ---
    SurfaceTemplate("The sequence {sequence} codes for {protein}.", domain="biological"),
    SurfaceTemplate("This biological process involves {process}.", domain="biological"),
    SurfaceTemplate("The {organism} exhibits {trait}.", domain="biological"),
    SurfaceTemplate("At the cellular level, {mechanism}.", domain="biological"),
    # API-friendly: UniProt returns gene + protein + organism
    SurfaceTemplate("The gene {gene} encodes {protein} in {organism}.", domain="biological", weight=1.2),

    # --- Chemical ---
    SurfaceTemplate("The compound {compound} has the formula {formula}.", domain="chemical"),
    SurfaceTemplate("This reaction produces {product}.", domain="chemical"),
    SurfaceTemplate("The bond between {atom1} and {atom2} is {bond_type}.", domain="chemical"),
    SurfaceTemplate("{reactant} reacts with {reagent} to form {product}.", domain="chemical"),
    # API-friendly: PubChem returns formula + molecular_weight + iupac_name
    SurfaceTemplate("{compound} ({iupac_name}) has formula {formula}, MW={molecular_weight}.", domain="chemical", weight=1.2),

    # --- Physical ---
    SurfaceTemplate("The {quantity} equals {value} {unit}.", domain="physical"),
    SurfaceTemplate("By {law}, {consequence}.", domain="physical"),
    SurfaceTemplate("The system exhibits {behaviour}.", domain="physical"),
    SurfaceTemplate("At this scale, {phenomenon} dominates.", domain="physical"),
    # Built-in: constants table returns symbol + value + unit + concept
    SurfaceTemplate("The constant {symbol} ({concept}) is {value} {unit}.", domain="physical", weight=1.2),

    # --- Computational ---
    SurfaceTemplate("The algorithm {description}.", domain="computational"),
    SurfaceTemplate("This has {complexity} complexity.", domain="computational"),
    SurfaceTemplate("The function returns {result}.", domain="computational"),
    SurfaceTemplate("Recursively, {description}.", domain="computational"),
    # API-friendly: algo KB returns concept + complexity
    SurfaceTemplate("{concept} involves {complexity} complexity.", domain="computational", weight=1.1),

    # --- Etymological ---
    SurfaceTemplate("'{word}' derives from {origin} '{root}'.", domain="etymological"),
    SurfaceTemplate("The root '{root}' means '{meaning}' in {language}.", domain="etymological"),
    SurfaceTemplate("Historically, '{word}' evolved from {ancestor}.", domain="etymological"),
    SurfaceTemplate("This is a {loan_type} from {source_language}.", domain="etymological"),
    # API-friendly: Datamuse returns definition; built-in returns root_language
    SurfaceTemplate("'{word}' comes from {origin}, meaning '{meaning}'.", domain="etymological", weight=1.2),

    # --- Evidential hedging (English: lexical strategy) ---
    SurfaceTemplate("{content}", domain="general",
                    evidential_source="obs", evidential_confidence="cert", weight=1.0),
    SurfaceTemplate("Based on observation, {content}.", domain="general",
                    evidential_source="obs", evidential_confidence="prob"),
    SurfaceTemplate("The evidence suggests that {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="prob"),
    SurfaceTemplate("It appears that {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="poss"),
    SurfaceTemplate("Computation confirms that {content}.", domain="general",
                    evidential_source="comp", evidential_confidence="cert"),
    SurfaceTemplate("According to reports, {content}.", domain="general",
                    evidential_source="rep", evidential_confidence="prob"),
    SurfaceTemplate("Speculatively, {content}.", domain="general",
                    evidential_source="spec", evidential_confidence="poss"),
    SurfaceTemplate("It is unlikely, but {content}.", domain="general",
                    evidential_source="spec", evidential_confidence="unl"),
    SurfaceTemplate("If that were the case, {content}.", domain="general",
                    evidential_source="ctr", evidential_confidence="poss"),
]


def register_english(registry: TemplateRegistry) -> None:
    """Register all English templates into the given registry."""
    for t in ENGLISH_TEMPLATES:
        registry.add(t)
