"""Agent LangGraph minimal, sans LLM (aucune clé d'API nécessaire).

Le graphe a deux nœuds pour bien voir plusieurs chunks en stream_mode="updates" :
    START -> analyser -> repondre -> END
"""

from typing import Annotated

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]
    nb_mots: int


def analyser(state: State) -> dict:
    """Compte les mots du dernier message utilisateur."""
    question = state["messages"][-1].content
    return {"nb_mots": len(question.split())}


def repondre(state: State) -> dict:
    """Produit une réponse (à remplacer par un appel LLM)."""
    question = state["messages"][-1].content
    reponse = (
        f"Tu m'as demandé : « {question} » ({state['nb_mots']} mots). "
        "Je suis un agent LangGraph local servi par `langgraph dev`."
    )
    return {"messages": [AIMessage(content=reponse)]}


builder = StateGraph(State)
builder.add_node("analyser", analyser)
builder.add_node("repondre", repondre)
builder.add_edge(START, "analyser")
builder.add_edge("analyser", "repondre")
builder.add_edge("repondre", END)

# Variable référencée dans langgraph.json
graph = builder.compile()
