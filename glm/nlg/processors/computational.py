"""Computational domain processor — built-in algorithm knowledge base.

No external API — uses a comprehensive built-in KB of algorithm
complexities, data structures, and number theory facts.
Instant, offline, zero latency.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

from glm.core.mnemo_substrate import MnemoSequence
from . import DomainProcessor

_ALGO_CONCEPTS: Dict[str, str] = {
    "sort": "O(n log n) comparison-based",
    "search": "O(log n) binary or O(n) linear",
    "hash": "O(1) average-case lookup",
    "tree": "O(log n) balanced tree",
    "graph": "O(V + E) traversal",
    "loop": "iterative O(n)",
    "recursion": "recursive decomposition",
    "iterate": "step-wise O(n)",
    "compare": "O(n) pairwise evaluation",
    "merge": "O(n) combining sorted sequences",
    "queue": "O(1) FIFO enqueue/dequeue",
    "stack": "O(1) LIFO push/pop",
    "cache": "O(1) memoized lookup",
    "algorithm": "systematic procedure",
    "encrypt": "O(n) symmetric or O(n\u00b2) asymmetric",
    "compress": "O(n) entropy coding",
    "parse": "O(n) recursive descent",
    "matrix": "O(n\u00b3) naive or O(n^2.37) Strassen",
    "dynamic programming": "optimal substructure + overlapping subproblems",
    "backtrack": "O(b^d) depth-limited exploration",
    "greedy": "locally optimal choices",
    "divide": "T(n) = aT(n/b) + f(n) master theorem",
    "fibonacci": "O(\u03c6^n) naive or O(n) iterative",
    "prime": "O(\u221an) trial division or O(log^6 n) AKS",
    "neural": "O(n\u00b7m) forward pass, O(n\u00b7m) backprop",
    "fourier": "O(n log n) FFT",
    "convolution": "O(n log n) via FFT",
    "shortest path": "O(E log V) Dijkstra or O(VE) Bellman-Ford",
    "minimum spanning": "O(E log V) Kruskal or Prim",
}

# Number theory facts (replaces Numbers API)
_NUMBER_FACTS: Dict[int, str] = {
    0: "0 is the additive identity",
    1: "1 is the multiplicative identity",
    2: "2 is the only even prime number",
    3: "3 is the first odd prime",
    4: "4 is the smallest composite number",
    5: "5 is the number of Platonic solids",
    6: "6 is the smallest perfect number (1+2+3)",
    7: "7 is a Mersenne prime (2^3 - 1)",
    8: "8 is the first cube (2^3)",
    9: "9 is the first odd composite",
    10: "10 is the base of the decimal system",
    12: "12 is a highly composite number",
    13: "13 is a Fibonacci number and prime",
    16: "16 is 2^4, used in hexadecimal",
    17: "17 is the sum of the first four primes",
    23: "23 is the smallest prime with consecutive digits",
    25: "25 is the smallest square that is the sum of two squares (9+16)",
    28: "28 is the second perfect number (1+2+4+7+14)",
    37: "37 is a prime that remains prime when reversed (73)",
    42: "42 is the product of the first three terms of a Catalan sequence",
    64: "64 is 2^6, the number of squares on a chessboard",
    100: "100 is the square of 10",
    127: "127 is a Mersenne prime (2^7 - 1)",
    128: "128 is 2^7",
    256: "256 is 2^8, the number of values in a byte",
    512: "512 is 2^9",
    1024: "1024 is 2^10, a kibibyte",
}


def _extract_number(text: str) -> int:
    m = re.search(r'\d+', text)
    return int(m.group(0)) if m else -1


def _extract_concept(text: str) -> str:
    lower = text.lower()
    for keyword in _ALGO_CONCEPTS:
        if keyword in lower:
            return keyword
    return ""


def _number_fact(n: int) -> str:
    if n in _NUMBER_FACTS:
        return _NUMBER_FACTS[n]
    if n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1)):
        return f"{n} is a prime number"
    if n > 0 and (n & (n - 1)) == 0:
        exp = n.bit_length() - 1
        return f"{n} is 2^{exp}"
    if n > 0:
        sqrt = int(n**0.5)
        if sqrt * sqrt == n:
            return f"{n} is a perfect square ({sqrt}^2)"
    return f"{n} is an integer"


class ComputationalProcessor(DomainProcessor):
    domain = "computational"

    def process(self, text: str, mnemo_seq: MnemoSequence) -> Dict[str, str]:
        # Try algorithm concept first
        concept = _extract_concept(text)
        if concept:
            complexity = _ALGO_CONCEPTS[concept]
            return {
                "concept": concept,
                "complexity": complexity,
                "result": complexity,
                "description": f"{concept}: {complexity}",
            }

        # Try number fact
        number = _extract_number(text)
        if number >= 0:
            fact = _number_fact(number)
            return {
                "number": str(number),
                "fact": fact,
                "result": fact,
                "description": fact,
                "complexity": "numeric computation",
            }

        return {"description": f"computational analysis: {text[:60]}"}
