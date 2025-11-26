# nodes/llm_direct_answer.py

from typing import Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from state import GraphState


def make_llm_direct_answer_node(llm: BaseChatModel):
    """
    Node for answering general non-database questions or chitchat.
    """

    def llm_direct_answer(state: GraphState) -> Dict[str, Any]:
        user_query = (state.user_query or "").strip()

        if not user_query:
            return {"final_answer": "I didn't receive a question to answer."}

        system_prompt = (
            "You are a helpful assistant. "
            "Respond directly and clearly to the user's question. "
            "Do NOT reference the database or any schema. "
            "This mode is for general knowledge, explanations, or chitchat."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ]

        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        return {"final_answer": answer}

    return llm_direct_answer
