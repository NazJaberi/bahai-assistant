import re, unicodedata

def strip_diacritics(s: str) -> str:
    if not s: return s
    nkfd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nkfd if not unicodedata.combining(ch))

def looks_like_definition_or_law(query: str) -> bool:
    q = strip_diacritics(query.lower())
    return any(kw in q for kw in [
        "what is", "how does", "how do", "how it works", "explain", "law", "obligation",
        "exempt", "exemption", "rate", "percentage", "due", "threshold", "rules", "requirements"
    ])

def system_hint_for(query: str) -> str:
    """
    Generic, topic-agnostic instruction for rule/definition queries.
    """
    if looks_like_definition_or_law(query):
        return (
            "You compose concise, strictly grounded answers using only the provided passages. "
            "When the question asks 'what is', 'how it works', or about a law/obligation, "
            "summarize in a rule-shaped way covering: definition/purpose, when it applies, "
            "calculation or steps, exceptions/exemptions, priority or procedural notes if present. "
            "All quotes must be verbatim with citations (work title + paragraph/section + deep link). "
            "If key details are absent, say so and provide the closest cited passages."
        )
    return (
        "You compose concise, strictly grounded answers with verbatim quotes and citations "
        "(work title + paragraph/section + deep link). Use only the provided passages."
    )
