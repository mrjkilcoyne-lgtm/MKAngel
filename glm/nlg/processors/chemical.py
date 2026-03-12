"""Chemical domain processor — PubChem PUG REST API integration.

API: https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/{props}/JSON
Free, no auth required.

Returns molecular formula, molecular weight, and IUPAC name.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Dict

from glm.core.mnemo_substrate import MnemoSequence
from . import DomainProcessor
from ._http import fetch_json

_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
_PROPS = "MolecularFormula,MolecularWeight,IUPACName"

_COMPOUND_KEYWORDS = frozenset({
    "water", "oxygen", "hydrogen", "carbon", "nitrogen", "methane", "ethanol",
    "glucose", "benzene", "ammonia", "sulfur", "chlorine", "sodium", "potassium",
    "calcium", "iron", "gold", "silver", "copper", "helium", "neon", "argon",
    "acetone", "propane", "butane", "methanol", "glycerol", "aspirin",
    "caffeine", "ethylene", "acetylene", "phosphorus", "silicon", "lithium",
})

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "with", "of", "and", "in", "to", "by", "from",
    "this", "that", "reacts", "bonds", "molecule", "compound", "atoms",
})


def _extract_compound(text: str) -> str:
    lower = text.lower()
    words = re.findall(r'[a-zA-Z]+', lower)
    for w in words:
        if w in _COMPOUND_KEYWORDS:
            return w
    content = [w for w in words if w not in _STOPWORDS and len(w) > 3]
    return content[0] if content else "water"


class ChemicalProcessor(DomainProcessor):
    domain = "chemical"

    def process(self, text: str, mnemo_seq: MnemoSequence) -> Dict[str, str]:
        compound = _extract_compound(text)
        encoded = urllib.parse.quote(compound)
        url = f"{_PUBCHEM_BASE}/{encoded}/property/{_PROPS}/JSON"

        data = fetch_json(url)
        if data:
            props_list = data.get("PropertyTable", {}).get("Properties", [])
            if props_list:
                p = props_list[0]
                formula = p.get("MolecularFormula", "")
                weight = str(p.get("MolecularWeight", ""))
                iupac = p.get("IUPACName", "")
                return {
                    "compound": compound,
                    "formula": formula,
                    "molecular_weight": weight,
                    "iupac_name": iupac,
                    "product": formula,
                    "result": formula or compound,
                    "description": (
                        f"{compound}: {formula}, MW={weight}"
                        if formula else f"chemical analysis of {compound}"
                    ),
                }

        return {
            "compound": compound,
            "description": f"chemical analysis of {compound}",
        }
