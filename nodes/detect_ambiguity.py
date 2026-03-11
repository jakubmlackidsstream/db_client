from pydantic import BaseModel, Field
from typing import Dict, Any, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from state import GraphState

# Only ask for clarification when the model is highly confident the term
# is genuinely ambiguous.  Lower this value to ask more questions.
AMBIGUITY_CONFIDENCE_THRESHOLD = 0.85


class AmbiguousTerm(BaseModel):
    term: str
    confidence: float = Field(
        description=(
            "0.0-1.0. How confident are you that this term is genuinely "
            "ambiguous and would require different SQL depending on the "
            "answer? Use high values (>0.85) only for real business-domain "
            "terms. Use low values (<0.5) for words that are clear in context."
        )
    )


class AmbiguityCheck(BaseModel):
    candidates: List[AmbiguousTerm]


def make_detect_ambiguity_node(llm: BaseChatModel):
    """
    Factory that creates a detect_ambiguity node.

    Scans the user query for vague or domain-specific terms that are
    not yet defined in known_terms.  Returns an empty list when the
    query is clear enough to proceed to SQL planning.

    Each candidate term is scored with a confidence value; only terms
    above AMBIGUITY_CONFIDENCE_THRESHOLD are propagated to the next node.

    Reads:  user_query, known_terms
    Writes: ambiguous_terms
    """
    structured_llm = llm.with_structured_output(AmbiguityCheck)

    def detect_ambiguity(state: GraphState) -> Dict[str, Any]:
        user_query = (state.user_query or "").strip()
        known_terms = state.known_terms or {}

        known_text = (
            "\n".join(f"- {k}: {v}" for k, v in known_terms.items())
            if known_terms
            else "None defined yet."
        )

        system_prompt = (
            "You are a database assistant pre-processing user queries.\n"
            "Identify domain-specific or vague terms in the query that "
            "would need clarification to write a precise SQL query.\n"
            "For each candidate term, assign a confidence score (0.0-1.0) "
            "indicating how certain you are that clarification is needed.\n\n"
            "Rules:\n"
            "- Do NOT flag standard SQL/DB concepts "
            "(table, row, column, count, sum, join, average, structure, "
            "schema, etc.).\n"
            "- Do NOT flag terms already defined in the known-terms list.\n"
            "- Do NOT flag clearly numeric thresholds "
            "(e.g. 'more than 100 orders').\n"
            "- Do NOT flag common verbs or UI words "
            "(show, list, find, display, get, give, fetch, check, describe, "
            "pokaż, znajdź, wyświetl, podaj, sprawdź, wymień, etc.).\n"
            "- Do NOT flag pronouns or articles "
            "(me, all, any, some, every, każdy, wszystkie, etc.).\n"
            "- Do NOT flag time-related words in meta-discovery queries "
            "(period, years, months, quarters, lata, miesiące, kwartały) "
            "when the query asks what values exist in the data "
            "(e.g. 'what years do we have', "
            "'from which period do we have orders').\n"
            "- Do NOT flag 'different', 'distinct', 'unique', 'various', "
            "'różne', 'unikalne' before a noun — they mean SQL DISTINCT "
            "(e.g. 'different countries', 'różne kraje').\n"
            "- Do NOT flag superlatives or ranking words "
            "(greatest, most, highest, lowest, best, worst, top, bottom, "
            "largest, smallest, najwięcej, najmniej, najwyższy, najniższy) "
            "— they translate to ORDER BY … [DESC|ASC] LIMIT.\n"
            "- Do NOT flag standard statistical terms "
            "(median, average, mean, mode, percentile, std dev, variance, "
            "mediana, średnia, odchylenie) "
            "— they have precise mathematical definitions.\n"
            "- Do NOT ask about time frames if the question contains no "
            "date, year, month, quarter, or period — assume full dataset.\n"
            "- Do NOT flag table or column names even if unfamiliar.\n"
            "- ONLY flag genuine domain-specific business terms whose "
            "interpretation depends on business rules and would produce "
            "different SQL depending on the answer.\n"
            "- Return an empty list when the query is already precise enough "
            "to write SQL without additional business context.\n\n"
            "Confidence guide:\n"
            "  0.9+ : clearly ambiguous business term with no obvious default "
            "         (e.g. 'active customer', 'VIP', 'high-value order')\n"
            "  0.7-0.9: probably ambiguous but context gives some hint\n"
            "  <0.7 : likely clear enough; prefer NOT to ask\n\n"
            "Examples of truly ambiguous terms (confidence >= 0.85): "
            "'overloaded', 'busy', 'recent', 'top', 'large', 'active', "
            "'slow', 'healthy', 'VIP customer', 'high-value order'."
        )

        user_prompt = (
            f"User query: {user_query}\n\n"
            f"Already known terms:\n{known_text}"
        )

        result: AmbiguityCheck = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        confirmed = [
            t.term
            for t in result.candidates
            if t.confidence >= AMBIGUITY_CONFIDENCE_THRESHOLD
        ]

        return {"ambiguous_terms": confirmed}

    return detect_ambiguity
