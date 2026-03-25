from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from typing import Literal, Dict, Any
from pydantic import BaseModel

from state import GraphState  # adjust to your actual path


class QuestionClassification(BaseModel):
    question_type: Literal["db_read", "db_schema_question", "chitchat", "unsupported", "manage_terms"]
    intent_description: str


def make_question_classifier_node(llm: BaseChatModel):
    """
    Factory that creates a question_classifier node bound to a specific LLM.
    Uses structured output (Pydantic) instead of manual JSON parsing.
    """

    structured_llm = llm.with_structured_output(QuestionClassification)

    def question_classifier(state: GraphState) -> Dict[str, Any]:
        """
        Classify the current user_query into a high-level type:

        - db_read            : needs to read/analyze data from the DB
        - db_schema_question : asks about tables/columns/structure
        - chitchat           : general questions / small talk
        - unsupported        : write/delete/modify or out-of-scope

        Returns a partial state update.
        """
        user_query = state.user_query.strip()

        # Fallback: if user_query is empty, try to get it from the last HumanMessage
        if not user_query and state.messages:
            for msg in reversed(state.messages):
                if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
                    user_query = msg.content.strip()
                    break

        # Safety fallback: still empty → mark as unsupported
        if not user_query:
            return {
                "question_type": "unsupported",
                "intent_description": "Empty or missing user query.",
            }

        # Optional: compact conversation history for context
        history_snippets = state.conversation_history[-5:] if state.conversation_history else []
        history_text = "\n".join(history_snippets) if history_snippets else "No prior conversation."

        system_prompt = (
            "You are a router for a database question-answering agent. "
            "Classify the user's request into one of:\n"
            "- db_read: wants to read or analyze data from the database.\n"
            "- db_schema_question: asks about tables, columns, or database structure.\n"
            "- chitchat: general questions or explanations not requiring the database.\n"
            "- manage_terms: wants to add, update, delete, or list stored term definitions "
            "(e.g. 'define X as Y', 'change the definition of X', 'forget X', 'what terms do you know').\n"
            "- unsupported: wants to write/modify/delete data, or otherwise out of scope.\n\n"
            "Use the provided response schema."
        )

        user_prompt = (
            f"User query:\n{user_query}\n\n"
            f"Recent conversation history:\n{history_text}"
        )

        messages: list[AnyMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # This returns an instance of QuestionClassification
        classification: QuestionClassification = structured_llm.invoke(messages)

        return {
            "question_type": classification.question_type,
            "intent_description": classification.intent_description,
        }

    return question_classifier