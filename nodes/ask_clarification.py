from pydantic import BaseModel
from typing import Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from state import GraphState

# Separator used to encode multiple pending terms in a single string field.
TERMS_SEP = "||"


class ClarificationQuestion(BaseModel):
    question: str


def make_ask_clarification_node(llm: BaseChatModel):
    """
    Factory that creates an ask_clarification node.

    Asks about ALL ambiguous terms in a single message so the user
    only needs to reply once.  All term names are stored together in
    pending_clarification_for (joined by TERMS_SEP) and the original
    query is preserved in last_query.

    Reads:  user_query, ambiguous_terms
    Writes: clarification_question, pending_clarification_for,
            last_query, final_answer
    """
    structured_llm = llm.with_structured_output(ClarificationQuestion)

    def ask_clarification(state: GraphState) -> Dict[str, Any]:
        terms = state.ambiguous_terms
        user_query = state.user_query or ""

        system_prompt = (
            "You are a database assistant. A user asked a question that "
            "contains one or more ambiguous terms.\n"
            "Write a single, concise message that asks for clarification "
            "of ALL listed terms at once — do not split into separate "
            "messages.\n"
            "For each term, suggest 2-3 concrete, measurable examples "
            "(e.g. thresholds, time ranges, status values).\n"
            "Number each sub-question clearly (1., 2., …)."
        )

        terms_text = "\n".join(f"- '{t}'" for t in terms)
        user_prompt = (
            f"User query: {user_query}\n"
            f"Ambiguous terms:\n{terms_text}\n\n"
            "Generate one combined clarifying message."
        )

        result: ClarificationQuestion = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        pending = TERMS_SEP.join(terms)

        return {
            "clarification_question": result.question,
            "pending_clarification_for": pending,
            "last_query": user_query,
            "final_answer": result.question,
        }

    return ask_clarification
