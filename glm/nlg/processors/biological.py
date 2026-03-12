"""Biological domain processor — UniProt REST API integration.

API: https://rest.uniprot.org/uniprotkb/search?query={term}&fields=...&format=json
Free, no auth required.

Returns gene names, protein descriptions, and organism data.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Dict

from glm.core.mnemo_substrate import MnemoSequence
from . import DomainProcessor
from ._http import fetch_json

_UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"
_FIELDS = "gene_names,protein_name,organism_name"

_BIO_KEYWORDS = frozenset({
    "gene", "protein", "enzyme", "cell", "dna", "rna", "mutation", "genome",
    "chromosome", "mitosis", "meiosis", "ribosome", "amino", "nucleotide",
    "transcription", "translation", "replication", "organism", "insulin",
    "hemoglobin", "collagen", "keratin", "actin", "myosin",
})

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "with", "of", "and", "in", "to", "encodes",
    "for", "by", "from", "this", "that",
})


def _extract_bio_term(text: str) -> str:
    words = re.findall(r'[a-zA-Z]+', text.lower())
    content = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    for w in content:
        if w in _BIO_KEYWORDS:
            return w
    return content[0] if content else "gene"


class BiologicalProcessor(DomainProcessor):
    domain = "biological"

    def process(self, text: str, mnemo_seq: MnemoSequence) -> Dict[str, str]:
        term = _extract_bio_term(text)
        query = urllib.parse.quote(term)
        url = f"{_UNIPROT_BASE}?query={query}&fields={_FIELDS}&size=1&format=json"

        data = fetch_json(url)
        if data and data.get("results"):
            entry = data["results"][0]
            gene_list = entry.get("genes", [])
            gene = (
                gene_list[0].get("geneName", {}).get("value", "")
                if gene_list else ""
            )
            protein_desc = entry.get("proteinDescription", {})
            rec_name = protein_desc.get("recommendedName", {})
            protein = rec_name.get("fullName", {}).get("value", "")
            organism = entry.get("organism", {}).get("scientificName", "")

            return {
                "gene": gene or term,
                "sequence": gene or term,
                "protein": protein or f"{term}-related protein",
                "organism": organism,
                "process": f"{gene or term} expression",
                "trait": protein or f"{term} activity",
                "mechanism": f"{gene or term} pathway",
                "result": protein or gene or term,
                "description": (
                    f"{gene}: {protein} ({organism})"
                    if protein else f"biological data for {term}"
                ),
            }

        return {
            "gene": term,
            "sequence": term,
            "protein": f"{term}-related protein",
            "process": f"{term} expression",
            "description": f"biological data for {term}",
        }
