"""English surface templates — 120+ patterns across 7 domains + evidential hedging.

The true timeline is grammatically the case — evidential markers map to
language-appropriate realisations. English uses lexical hedging; other
languages will use morphological strategies (Turkish -mIs, Quechua -mi/-si).
"""

from __future__ import annotations

from . import SurfaceTemplate, TemplateRegistry


ENGLISH_TEMPLATES = [
    # ═══════════════════════════════════════════════════════════════════
    # MATHEMATICAL — 15 templates
    # ═══════════════════════════════════════════════════════════════════
    SurfaceTemplate("{result}", domain="mathematical", slots={"result"}),
    SurfaceTemplate("The result is {result}.", domain="mathematical"),
    SurfaceTemplate("Solving this gives {result}.", domain="mathematical"),
    SurfaceTemplate("By {method}, we get {result}.", domain="mathematical"),
    SurfaceTemplate("The {operation} of {operand} yields {result}.", domain="mathematical"),
    SurfaceTemplate("The {operation} of {expression} gives {result}.", domain="mathematical", weight=1.2),
    SurfaceTemplate("{expression} evaluates to {result}.", domain="mathematical"),
    SurfaceTemplate("Applying {operation} to {expression}: {result}.", domain="mathematical", weight=1.1),
    SurfaceTemplate("We find that {expression} = {result}.", domain="mathematical", weight=1.1),
    SurfaceTemplate("The answer is {result}.", domain="mathematical"),
    SurfaceTemplate("{operation}({expression}) = {result}.", domain="mathematical", weight=1.15),
    SurfaceTemplate("Computing {operation} of {expression} yields {result}.", domain="mathematical"),
    SurfaceTemplate("If we {operation} {expression}, the result is {result}.", domain="mathematical"),
    SurfaceTemplate("The {method} approach gives {result} for {expression}.", domain="mathematical"),
    SurfaceTemplate("{expression} simplifies to {result}.", domain="mathematical"),

    # ═══════════════════════════════════════════════════════════════════
    # LINGUISTIC — 18 templates
    # ═══════════════════════════════════════════════════════════════════
    SurfaceTemplate("The word '{word}' {analysis}.", domain="linguistic"),
    SurfaceTemplate("In this context, '{word}' functions as {role}.", domain="linguistic"),
    SurfaceTemplate("The phrase structure is {structure}.", domain="linguistic"),
    SurfaceTemplate("Morphologically, '{word}' breaks down as {breakdown}.", domain="linguistic"),
    SurfaceTemplate("The grammatical pattern here is {pattern}.", domain="linguistic"),
    SurfaceTemplate("'{word}' is a {part_of_speech}, defined as: {definition}.", domain="linguistic", weight=1.2),
    SurfaceTemplate("'{word}' ({phonetic}) means: {definition}.", domain="linguistic", weight=1.1),
    SurfaceTemplate("'{word}' — {part_of_speech}: {definition}.", domain="linguistic", weight=1.15),
    SurfaceTemplate("Definition of '{word}': {definition}.", domain="linguistic", weight=1.1),
    SurfaceTemplate("'{word}' is used as a {part_of_speech}. It means: {definition}.", domain="linguistic"),
    SurfaceTemplate("As a {part_of_speech}, '{word}' refers to: {definition}.", domain="linguistic"),
    SurfaceTemplate("The term '{word}' ({part_of_speech}) means {definition}.", domain="linguistic"),
    SurfaceTemplate("'{word}' [{phonetic}] — ({part_of_speech}) {definition}.", domain="linguistic", weight=1.2),
    SurfaceTemplate("Pronounced {phonetic}, '{word}' is a {part_of_speech} meaning: {definition}.", domain="linguistic"),
    SurfaceTemplate("'{word}' carries the meaning: {definition}.", domain="linguistic"),
    SurfaceTemplate("In English, '{word}' means {definition}.", domain="linguistic"),
    SurfaceTemplate("A {part_of_speech}, '{word}' denotes {definition}.", domain="linguistic"),
    SurfaceTemplate("'{word}' — {definition}.", domain="linguistic"),

    # ═══════════════════════════════════════════════════════════════════
    # BIOLOGICAL — 15 templates
    # ═══════════════════════════════════════════════════════════════════
    SurfaceTemplate("The sequence {sequence} codes for {protein}.", domain="biological"),
    SurfaceTemplate("This biological process involves {process}.", domain="biological"),
    SurfaceTemplate("The {organism} exhibits {trait}.", domain="biological"),
    SurfaceTemplate("At the cellular level, {mechanism}.", domain="biological"),
    SurfaceTemplate("The gene {gene} encodes {protein} in {organism}.", domain="biological", weight=1.2),
    SurfaceTemplate("{gene} produces the protein {protein}, found in {organism}.", domain="biological", weight=1.1),
    SurfaceTemplate("In {organism}, the {gene} gene is responsible for {protein}.", domain="biological", weight=1.1),
    SurfaceTemplate("{protein} is encoded by {gene} in {organism}.", domain="biological"),
    SurfaceTemplate("The protein {protein} ({gene}) plays a role in {process}.", domain="biological"),
    SurfaceTemplate("{organism} expresses {gene}, producing {protein}.", domain="biological"),
    SurfaceTemplate("Gene {gene} → protein {protein} in {organism}.", domain="biological", weight=1.15),
    SurfaceTemplate("The {gene} locus in {organism} encodes for {protein}.", domain="biological"),
    SurfaceTemplate("{protein} is a key protein in {organism}, encoded by {gene}.", domain="biological"),
    SurfaceTemplate("Biologically, {gene} maps to {protein} within {organism}.", domain="biological"),
    SurfaceTemplate("The {organism} genome contains {gene}, which produces {protein}.", domain="biological"),

    # ═══════════════════════════════════════════════════════════════════
    # CHEMICAL — 15 templates
    # ═══════════════════════════════════════════════════════════════════
    SurfaceTemplate("The compound {compound} has the formula {formula}.", domain="chemical"),
    SurfaceTemplate("This reaction produces {product}.", domain="chemical"),
    SurfaceTemplate("The bond between {atom1} and {atom2} is {bond_type}.", domain="chemical"),
    SurfaceTemplate("{reactant} reacts with {reagent} to form {product}.", domain="chemical"),
    SurfaceTemplate("{compound} ({iupac_name}) has formula {formula}, MW={molecular_weight}.", domain="chemical", weight=1.2),
    SurfaceTemplate("{compound}: {formula}, molecular weight {molecular_weight}.", domain="chemical", weight=1.15),
    SurfaceTemplate("The molecular formula of {compound} is {formula} (MW: {molecular_weight}).", domain="chemical", weight=1.1),
    SurfaceTemplate("{compound} is known as {iupac_name}, with formula {formula}.", domain="chemical"),
    SurfaceTemplate("Chemical formula: {formula}. Compound: {compound}. MW: {molecular_weight}.", domain="chemical"),
    SurfaceTemplate("{compound} has a molecular weight of {molecular_weight} g/mol.", domain="chemical"),
    SurfaceTemplate("IUPAC name: {iupac_name}. Formula: {formula}.", domain="chemical"),
    SurfaceTemplate("The compound {compound} ({formula}) weighs {molecular_weight} Da.", domain="chemical"),
    SurfaceTemplate("{formula} represents {compound}, MW = {molecular_weight}.", domain="chemical"),
    SurfaceTemplate("{compound} — formula {formula}, weight {molecular_weight} g/mol.", domain="chemical"),
    SurfaceTemplate("Structurally, {compound} is {formula} with molecular weight {molecular_weight}.", domain="chemical"),

    # ═══════════════════════════════════════════════════════════════════
    # PHYSICAL — 15 templates
    # ═══════════════════════════════════════════════════════════════════
    SurfaceTemplate("The {quantity} equals {value} {unit}.", domain="physical"),
    SurfaceTemplate("By {law}, {consequence}.", domain="physical"),
    SurfaceTemplate("The system exhibits {behaviour}.", domain="physical"),
    SurfaceTemplate("At this scale, {phenomenon} dominates.", domain="physical"),
    SurfaceTemplate("The constant {symbol} ({concept}) is {value} {unit}.", domain="physical", weight=1.2),
    SurfaceTemplate("{concept} ({symbol}) = {value} {unit}.", domain="physical", weight=1.15),
    SurfaceTemplate("The value of {symbol} is {value} {unit}.", domain="physical", weight=1.1),
    SurfaceTemplate("{concept}: {value} {unit} (symbol: {symbol}).", domain="physical"),
    SurfaceTemplate("In physics, {symbol} represents {concept}, valued at {value} {unit}.", domain="physical"),
    SurfaceTemplate("The fundamental constant {concept} has the value {value} {unit}.", domain="physical"),
    SurfaceTemplate("{symbol} = {value} {unit} — the {concept}.", domain="physical"),
    SurfaceTemplate("Measured value of {concept}: {value} {unit}.", domain="physical"),
    SurfaceTemplate("The {concept} is precisely {value} {unit}.", domain="physical"),
    SurfaceTemplate("Physics constant {symbol}: {value} {unit}.", domain="physical"),
    SurfaceTemplate("{concept} equals exactly {value} {unit} (denoted {symbol}).", domain="physical"),

    # ═══════════════════════════════════════════════════════════════════
    # COMPUTATIONAL — 15 templates
    # ═══════════════════════════════════════════════════════════════════
    SurfaceTemplate("The algorithm {description}.", domain="computational"),
    SurfaceTemplate("This has {complexity} complexity.", domain="computational"),
    SurfaceTemplate("The function returns {result}.", domain="computational"),
    SurfaceTemplate("Recursively, {description}.", domain="computational"),
    SurfaceTemplate("{concept} involves {complexity} complexity.", domain="computational", weight=1.1),
    SurfaceTemplate("{concept} runs in {complexity} time.", domain="computational", weight=1.15),
    SurfaceTemplate("The {concept} algorithm has {complexity} complexity.", domain="computational", weight=1.1),
    SurfaceTemplate("{concept}: time complexity is {complexity}.", domain="computational"),
    SurfaceTemplate("In computer science, {concept} operates at {complexity}.", domain="computational"),
    SurfaceTemplate("{concept} — {description}.", domain="computational"),
    SurfaceTemplate("The complexity of {concept} is {complexity}.", domain="computational"),
    SurfaceTemplate("{number} is {fact}.", domain="computational"),
    SurfaceTemplate("The number {number}: {fact}.", domain="computational"),
    SurfaceTemplate("Computationally, {concept} requires {complexity} operations.", domain="computational"),
    SurfaceTemplate("{concept} achieves {complexity} performance.", domain="computational"),

    # ═══════════════════════════════════════════════════════════════════
    # ETYMOLOGICAL — 15 templates
    # ═══════════════════════════════════════════════════════════════════
    SurfaceTemplate("'{word}' derives from {origin} '{root}'.", domain="etymological"),
    SurfaceTemplate("The root '{root}' means '{meaning}' in {language}.", domain="etymological"),
    SurfaceTemplate("Historically, '{word}' evolved from {ancestor}.", domain="etymological"),
    SurfaceTemplate("This is a {loan_type} from {source_language}.", domain="etymological"),
    SurfaceTemplate("'{word}' comes from {origin}, meaning '{meaning}'.", domain="etymological", weight=1.2),
    SurfaceTemplate("'{word}' traces back to {origin} ({language}).", domain="etymological", weight=1.1),
    SurfaceTemplate("The etymology of '{word}': from {origin}, '{meaning}'.", domain="etymological"),
    SurfaceTemplate("'{word}' has {language} roots, from '{root}' meaning '{meaning}'.", domain="etymological"),
    SurfaceTemplate("Origin: {origin}. '{word}' meant '{meaning}' in {language}.", domain="etymological"),
    SurfaceTemplate("From {language} '{root}', we get '{word}' — {meaning}.", domain="etymological"),
    SurfaceTemplate("'{word}' entered English from {source_language}.", domain="etymological"),
    SurfaceTemplate("The word '{word}' originates from {language} '{root}'.", domain="etymological"),
    SurfaceTemplate("Etymologically, '{word}' descends from {origin}.", domain="etymological"),
    SurfaceTemplate("'{word}': {language} origin, root '{root}', meaning '{meaning}'.", domain="etymological"),
    SurfaceTemplate("Borrowed from {source_language}, '{word}' originally meant '{meaning}'.", domain="etymological"),

    # ═══════════════════════════════════════════════════════════════════
    # EVIDENTIAL HEDGING — 15 templates (English: lexical strategy)
    # ═══════════════════════════════════════════════════════════════════
    # Observed + certain
    SurfaceTemplate("{content}", domain="general",
                    evidential_source="obs", evidential_confidence="cert", weight=1.0),
    SurfaceTemplate("I can confirm that {content}.", domain="general",
                    evidential_source="obs", evidential_confidence="cert"),
    # Observed + probable
    SurfaceTemplate("Based on observation, {content}.", domain="general",
                    evidential_source="obs", evidential_confidence="prob"),
    SurfaceTemplate("From what we can see, {content}.", domain="general",
                    evidential_source="obs", evidential_confidence="prob"),
    # Inferred + probable
    SurfaceTemplate("The evidence suggests that {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="prob"),
    SurfaceTemplate("It can be inferred that {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="prob"),
    # Inferred + possible
    SurfaceTemplate("It appears that {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="poss"),
    SurfaceTemplate("It seems likely that {content}.", domain="general",
                    evidential_source="inf", evidential_confidence="poss"),
    # Computed + certain
    SurfaceTemplate("Computation confirms that {content}.", domain="general",
                    evidential_source="comp", evidential_confidence="cert"),
    SurfaceTemplate("Calculation shows: {content}.", domain="general",
                    evidential_source="comp", evidential_confidence="cert"),
    # Reported
    SurfaceTemplate("According to reports, {content}.", domain="general",
                    evidential_source="rep", evidential_confidence="prob"),
    SurfaceTemplate("It is reported that {content}.", domain="general",
                    evidential_source="rep", evidential_confidence="prob"),
    # Speculative
    SurfaceTemplate("Speculatively, {content}.", domain="general",
                    evidential_source="spec", evidential_confidence="poss"),
    # Unlikely
    SurfaceTemplate("It is unlikely, but {content}.", domain="general",
                    evidential_source="spec", evidential_confidence="unl"),
    # Counterfactual
    SurfaceTemplate("If that were the case, {content}.", domain="general",
                    evidential_source="ctr", evidential_confidence="poss"),
]


def register_english(registry: TemplateRegistry) -> None:
    """Register all English templates into the given registry."""
    for t in ENGLISH_TEMPLATES:
        registry.add(t)
