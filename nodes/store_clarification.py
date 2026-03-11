import re
from typing import Dict, Any

from state import GraphState
from nodes.ask_clarification import TERMS_SEP

# Question-starter patterns (Polish + English) that indicate a NEW question
# rather than a clarification answer.
_PL = (
    r"jakie|jaki|jaka|które|który|która|"
    r"ile|kto|co|gdzie|kiedy|jak|czy|"
    r"pokaż|znajdź|wyświetl|podaj|sprawdź|wymień|lista|opisz"
)
_EN = (
    r"what|which|who|where|when|how|"
    r"show|find|display|list|give|check|describe|get|fetch"
)
_NEW_QUESTION_PATTERN = re.compile(
    rf"^\s*({_PL}|{_EN})\b",
    re.IGNORECASE,
)


def _is_new_question(text: str) -> bool:
    """Return True if text looks like a new question, not a reply."""
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    return bool(_NEW_QUESTION_PATTERN.match(stripped))


def store_clarification(state: GraphState) -> Dict[str, Any]:
    """
    Store the user's clarification reply and restore the original query.

    pending_clarification_for may hold multiple term names joined by
    TERMS_SEP.  The user's single reply is saved as the definition for
    every pending term, giving the SQL planner the full context.

    If the incoming message looks like a new independent question, the
    pending context is discarded and the graph re-classifies from scratch.

    Reads:  user_query, pending_clarification_for, last_query, known_terms
    Writes: known_terms, pending_clarification_for (cleared), user_query,
            ambiguous_terms (cleared), question_type (reset for new q)
    """
    pending = state.pending_clarification_for or ""
    terms = [t for t in pending.split(TERMS_SEP) if t]
    incoming = (state.user_query or "").strip()
    original_query = state.last_query or ""

    if _is_new_question(incoming):
        return {
            "known_terms": dict(state.known_terms or {}),
            "pending_clarification_for": None,
            "last_query": None,
            "user_query": incoming,
            "ambiguous_terms": [],
            "clarification_question": None,
            "question_type": None,
        }

    # Save the user's answer as the definition for every pending term.
    known_terms = dict(state.known_terms or {})
    for term in terms:
        if term and incoming:
            known_terms[term] = incoming

    return {
        "known_terms": known_terms,
        "pending_clarification_for": None,
        "user_query": original_query,
        "ambiguous_terms": [],
        "clarification_question": None,
    }
