from typing import Dict, Any

from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from state import GraphState
from nodes.ask_clarification import TERMS_SEP


class _Intent(BaseModel):
    is_new_question: bool


def make_store_clarification_node(llm: BaseChatModel):
    """
    Factory that creates a store_clarification node.

    Uses an LLM to decide whether the incoming message is a clarification
    answer (to the pending question) or an entirely new question.

    If it is a new question, the pending context is discarded and the graph
    re-classifies from scratch.  Otherwise the user's reply is stored as the
    definition for every pending term.

    Reads:  user_query, pending_clarification_for, last_query,
            clarification_question, known_terms
    Writes: known_terms, pending_clarification_for (cleared), user_query,
            ambiguous_terms (cleared), question_type (reset for new q)
    """
    structured_llm = llm.with_structured_output(_Intent)

    def store_clarification(state: GraphState) -> Dict[str, Any]:
        pending = state.pending_clarification_for or ""
        terms = [t for t in pending.split(TERMS_SEP) if t]
        incoming = (state.user_query or "").strip()
        original_query = state.last_query or ""
        clarification_q = state.clarification_question or ""

        system_prompt = (
            "You are classifying a user message in a database assistant chat.\n"
            "The assistant previously asked the user a clarification question. "
            "Decide whether the user's reply is:\n"
            "  - a CLARIFICATION ANSWER (responds to the pending question) → is_new_question=false\n"
            "  - a NEW INDEPENDENT QUESTION (unrelated to the pending clarification) → is_new_question=true\n\n"
            "Guidelines:\n"
            "- Short factual replies, yes/no, numbered lists (1) … 2) …), or "
            "domain terms are almost always clarification answers.\n"
            "- A full analytic question about data is a new question.\n"
            "- When in doubt, treat it as a clarification answer."
        )

        user_prompt = (
            f"Pending clarification question: {clarification_q}\n"
            f"User's reply: {incoming}"
        )

        result: _Intent = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        if result.is_new_question:
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

    return store_clarification
