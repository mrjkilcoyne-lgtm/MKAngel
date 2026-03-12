"""Data crawler — pulls structured knowledge from free public APIs.

This module feeds the GLM template engine with real data from open sources.
Each crawler function returns slot-fill dicts ready for template rendering.
All calls are cached via _http.fetch_json (5-min TTL) and fail gracefully.

Sources:
  - Wikipedia REST API (summaries, extracts)
  - Wiktionary API (definitions)
  - Open Trivia DB (general knowledge)
  - Numbers API (number facts)
  - PubChem (chemical data)
  - UniProt (protein data)
  - arXiv (paper metadata)
  - REST Countries (geography)

None of these require API keys — all are open and free.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any, Dict, List, Optional

from glm.nlg.processors._http import fetch_json


# ═══════════════════════════════════════════════════════════════════
# WIKIPEDIA — General knowledge summaries
# ═══════════════════════════════════════════════════════════════════

def crawl_wikipedia_summary(topic: str, lang: str = "en") -> Dict[str, str]:
    """Fetch a short summary from Wikipedia REST API.

    Returns: {"topic": ..., "summary": ..., "url": ...} or {}
    """
    slug = urllib.parse.quote(topic.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{slug}"
    data = fetch_json(url)
    if not data or "extract" not in data:
        return {}
    return {
        "topic": data.get("title", topic),
        "summary": data["extract"][:500],
        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        "content": data["extract"][:200],  # Short form for evidential templates
    }


def crawl_wikipedia_search(query: str, lang: str = "en", limit: int = 5) -> List[Dict[str, str]]:
    """Search Wikipedia and return matching article summaries."""
    q = urllib.parse.quote(query)
    url = (
        f"https://{lang}.wikipedia.org/w/api.php?"
        f"action=query&list=search&srsearch={q}&srlimit={limit}"
        f"&format=json&utf8=1"
    )
    data = fetch_json(url)
    if not data or "query" not in data:
        return []
    results = []
    for item in data["query"].get("search", []):
        # Strip HTML tags from snippet
        snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
        results.append({
            "topic": item.get("title", ""),
            "summary": snippet,
            "content": snippet[:200],
        })
    return results


# ═══════════════════════════════════════════════════════════════════
# WIKTIONARY — Word definitions and etymology
# ═══════════════════════════════════════════════════════════════════

def crawl_wiktionary(word: str, lang: str = "en") -> Dict[str, str]:
    """Fetch word definition from Free Dictionary API (backed by Wiktionary).

    Returns: {"word": ..., "definition": ..., "part_of_speech": ..., "phonetic": ...}
    """
    w = urllib.parse.quote(word.lower().strip())
    url = f"https://api.dictionaryapi.dev/api/v2/entries/{lang}/{w}"
    data = fetch_json(url)
    if not data or not isinstance(data, list) or len(data) == 0:
        return {}
    entry = data[0]
    result: Dict[str, str] = {"word": entry.get("word", word)}
    if entry.get("phonetic"):
        result["phonetic"] = entry["phonetic"]
    meanings = entry.get("meanings", [])
    if meanings:
        m = meanings[0]
        result["part_of_speech"] = m.get("partOfSpeech", "")
        defs = m.get("definitions", [])
        if defs:
            result["definition"] = defs[0].get("definition", "")
    return result


# ═══════════════════════════════════════════════════════════════════
# NUMBERS API — Fun number facts
# ═══════════════════════════════════════════════════════════════════

def crawl_number_fact(number: int, fact_type: str = "trivia") -> Dict[str, str]:
    """Fetch a fact about a number.

    fact_type: 'trivia', 'math', 'date', 'year'
    Returns: {"number": ..., "fact": ...}
    """
    url = f"http://numbersapi.com/{number}/{fact_type}?json"
    data = fetch_json(url)
    if not data or "text" not in data:
        return {}
    return {
        "number": str(number),
        "fact": data["text"],
        "content": data["text"],
    }


# ═══════════════════════════════════════════════════════════════════
# PUBCHEM — Chemical compound data
# ═══════════════════════════════════════════════════════════════════

def crawl_pubchem(compound: str) -> Dict[str, str]:
    """Fetch compound data from PubChem REST API.

    Returns: {"compound": ..., "formula": ..., "molecular_weight": ..., "iupac_name": ...}
    """
    c = urllib.parse.quote(compound)
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{c}"
        f"/property/MolecularFormula,MolecularWeight,IUPACName/JSON"
    )
    data = fetch_json(url)
    if not data:
        return {}
    props_list = data.get("PropertyTable", {}).get("Properties", [])
    if not props_list:
        return {}
    props = props_list[0]
    return {
        "compound": compound,
        "formula": props.get("MolecularFormula", ""),
        "molecular_weight": str(props.get("MolecularWeight", "")),
        "iupac_name": props.get("IUPACName", ""),
    }


# ═══════════════════════════════════════════════════════════════════
# UNIPROT — Protein/gene data
# ═══════════════════════════════════════════════════════════════════

def crawl_uniprot(query: str, limit: int = 1) -> List[Dict[str, str]]:
    """Search UniProt for protein/gene info.

    Returns list of: {"gene": ..., "protein": ..., "organism": ...}
    """
    q = urllib.parse.quote(query)
    url = (
        f"https://rest.uniprot.org/uniprotkb/search?"
        f"query={q}&format=json&size={limit}&fields=gene_names,protein_name,organism_name"
    )
    data = fetch_json(url, timeout=8.0)
    if not data or "results" not in data:
        return []
    results = []
    for entry in data["results"]:
        gene = ""
        genes = entry.get("genes", [])
        if genes:
            gene = genes[0].get("geneName", {}).get("value", "")
        protein = entry.get("proteinDescription", {}).get(
            "recommendedName", {}
        ).get("fullName", {}).get("value", "")
        organism = entry.get("organism", {}).get("scientificName", "")
        if gene and protein:
            results.append({
                "gene": gene,
                "protein": protein,
                "organism": organism,
            })
    return results


# ═══════════════════════════════════════════════════════════════════
# REST COUNTRIES — Geography data
# ═══════════════════════════════════════════════════════════════════

def crawl_country(name: str) -> Dict[str, str]:
    """Fetch country data from REST Countries API.

    Returns: {"country": ..., "capital": ..., "population": ..., "region": ..., "languages": ...}
    """
    n = urllib.parse.quote(name)
    url = f"https://restcountries.com/v3.1/name/{n}?fields=name,capital,population,region,languages"
    data = fetch_json(url)
    if not data or not isinstance(data, list) or len(data) == 0:
        return {}
    c = data[0]
    langs = c.get("languages", {})
    return {
        "country": c.get("name", {}).get("common", name),
        "capital": ", ".join(c.get("capital", [])),
        "population": f"{c.get('population', 0):,}",
        "region": c.get("region", ""),
        "languages": ", ".join(langs.values()) if langs else "",
        "content": f"{c.get('name', {}).get('common', name)}, capital {', '.join(c.get('capital', []))}",
    }


# ═══════════════════════════════════════════════════════════════════
# ARXIV — Academic paper metadata
# ═══════════════════════════════════════════════════════════════════

def crawl_arxiv(query: str, limit: int = 3) -> List[Dict[str, str]]:
    """Search arXiv for paper metadata (title, authors, summary).

    Returns list of: {"title": ..., "authors": ..., "summary": ..., "url": ...}
    """
    q = urllib.parse.quote(query)
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=all:{q}&start=0&max_results={limit}"
    )
    # arXiv returns Atom XML, not JSON — parse minimally
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MKAngel-GLM/1.0"})
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            xml = resp.read().decode("utf-8")
    except Exception:
        return []

    results = []
    # Minimal XML extraction (no lxml dependency)
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
    for entry in entries:
        title = _xml_text(entry, "title").replace("\n", " ").strip()
        summary = _xml_text(entry, "summary").replace("\n", " ").strip()[:300]
        authors = re.findall(r"<name>(.*?)</name>", entry)
        link_match = re.search(r'<id>(.*?)</id>', entry)
        url_str = link_match.group(1) if link_match else ""
        if title:
            results.append({
                "title": title,
                "authors": ", ".join(authors[:3]),
                "summary": summary,
                "url": url_str,
                "content": summary[:200],
            })
    return results


def _xml_text(xml: str, tag: str) -> str:
    """Extract text from a simple XML tag."""
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL)
    return m.group(1) if m else ""


# ═══════════════════════════════════════════════════════════════════
# OPEN TRIVIA — General knowledge Q&A
# ═══════════════════════════════════════════════════════════════════

def crawl_trivia(amount: int = 5, category: Optional[int] = None) -> List[Dict[str, str]]:
    """Fetch trivia questions from Open Trivia DB.

    Returns list of: {"question": ..., "answer": ..., "category": ...}
    """
    url = f"https://opentdb.com/api.php?amount={amount}&type=multiple"
    if category is not None:
        url += f"&category={category}"
    data = fetch_json(url)
    if not data or data.get("response_code") != 0:
        return []
    results = []
    for q in data.get("results", []):
        question = _unescape_html(q.get("question", ""))
        answer = _unescape_html(q.get("correct_answer", ""))
        results.append({
            "question": question,
            "answer": answer,
            "category": q.get("category", ""),
            "content": f"{question} — {answer}",
        })
    return results


def _unescape_html(text: str) -> str:
    """Minimal HTML entity unescaping."""
    import html
    return html.unescape(text)


# ═══════════════════════════════════════════════════════════════════
# UNIFIED INTERFACE — Route by domain
# ═══════════════════════════════════════════════════════════════════

def crawl(query: str, domain: str = "general", lang: str = "en") -> Dict[str, str]:
    """Unified crawl interface — routes to the best source for the domain.

    Always returns a dict of slot fills (possibly empty).
    """
    if domain == "chemical":
        return crawl_pubchem(query)
    elif domain == "biological":
        results = crawl_uniprot(query, limit=1)
        return results[0] if results else {}
    elif domain == "linguistic":
        return crawl_wiktionary(query, lang=lang)
    elif domain == "etymological":
        # Wiktionary has etymological data in the definitions too
        return crawl_wiktionary(query, lang=lang)
    else:
        # Default: try Wikipedia
        return crawl_wikipedia_summary(query, lang=lang)
