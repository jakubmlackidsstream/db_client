import json
from pathlib import Path
from typing import Dict, Any, Optional, Literal

from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from state import GraphState


class _TermsAction(BaseModel):
    action: Literal["add", "update", "delete", "list"]
    term: Optional[str] = None
    definition: Optional[str] = None


def make_manage_terms_node(llm: BaseChatModel, knowledge_path: str = "knowledge.json"):
    """
    Node that lets users add, update, delete, or list entries in known_terms
    through natural language (e.g. "define X as Y", "forget X", "what terms do you know?").

    Reads:  user_query, known_terms
    Writes: known_terms, final_answer, messages
    """
    structured_llm = llm.with_structured_output(_TermsAction)
    path = Path(knowledge_path)

    def manage_terms(state: GraphState) -> Dict[str, Any]:
        user_query = (state.user_query or "").strip()
        known_terms = dict(state.known_terms or {})

        known_text = (
            "\n".join(f"- {k}: {v}" for k, v in known_terms.items())
            if known_terms
            else "None defined yet."
        )

        system_prompt = (
            "You are managing a vocabulary store for a database assistant.\n"
            "Parse the user's request and extract:\n"
            "  action: 'add'/'update' — store or overwrite a term definition\n"
            "          'delete'        — remove a term\n"
            "          'list'          — show all stored terms\n"
            "  term:       the exact term (omit for 'list')\n"
            "  definition: the meaning (omit for 'delete' and 'list')\n\n"
            "Current stored terms:\n"
            f"{known_text}"
        )

        result: _TermsAction = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ])

        if result.action in ("add", "update") and result.term and result.definition:
            known_terms[result.term] = result.definition
            answer = f"Defined **{result.term}** as: *{result.definition}*"
        elif result.action == "delete" and result.term:
            if result.term in known_terms:
                del known_terms[result.term]
                answer = f"Removed definition for **{result.term}**."
            else:
                answer = f"No definition found for **{result.term}**."
        elif result.action == "list":
            if known_terms:
                lines = [f"- **{k}**: {v}" for k, v in known_terms.items()]
                answer = "Stored definitions:\n" + "\n".join(lines)
            else:
                answer = "No definitions stored yet."
        else:
            answer = "I couldn't understand that vocabulary request. Try: 'define X as Y', 'forget X', or 'list definitions'."

        # Persist to file immediately (same logic as save_knowledge_store)
        existing: Dict[str, Any] = {}
        if path.exists():
            with open(path) as f:
                existing = json.load(f)
        existing["known_terms"] = known_terms
        with open(path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

        return {
            "known_terms": known_terms,
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
        }

    return manage_terms
