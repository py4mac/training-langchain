from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
import uuid 
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, ToolRuntime
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass

from langgraph.types import Command
from langchain.agents import create_agent, AgentState
from langchain.messages import ToolMessage
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from typing import TypedDict
from dataclasses import dataclass
from typing import Literal
from langchain.tools import tool
from langchain.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END

load_dotenv()

print("Enabling langsmith")
os.environ["LANGSMITH_API_KEY"]=os.environ["LANGSMITH_API_KEY"]
# Si vous choisissez la zone Europe en créant votre compte :
os.environ["LANGSMITH_ENDPOINT"]="https://eu.api.smith.langchain.com"
# Si vous choisissez la zone US (par défaut) :
# os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
if (os.environ["LANGSMITH_API_KEY"]):
  os.environ["LANGSMITH_TRACING"]="true"
  os.environ["LANGSMITH_PROJECT"]="formation-langchain-jour-2"
  from langsmith import Client
  client = Client()
  # Test pour vérifier la bonne connexion
  print([*client.list_projects()])
else:
  print("LangSmith n'est pas activé (pas de clé d'API)")


# https://openrouter.ai/docs/guides/community/langchain
model = init_chat_model(
    model="google/gemini-3.8-flash",
    # model="meta-llama/llama-3.3-70b-instruct:free" # modèle gratuit
    # OpenRouter utilise le même format d'API qu'OpenAI
    # qui est un standard "de fait"
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    )

@dataclass
class Context:
    user_id: str
class State(TypedDict):
    input: str
    output: str

# Ex basic
# def call_model(state: State, runtime: Runtime[Context]):
#   prompt=f"""User {runtime.context["user_id"]}: {state["input"]}"""
#   return {
#       "output": model.invoke(prompt).content
#   }

# one_node_builder = StateGraph(
#     state_schema=State
# )
# one_node_builder.add_node(call_model)
# one_node_builder.add_edge(START, call_model.__name__)


# one_node_graph = one_node_builder.compile()
# res=one_node_graph.invoke(
#     {"input": ["from langchain.agents import AgentState"]},
#     context={"user_id": "42"})
# print(res["output"])

# Guardrail
# def is_about_langchain(state: State, runtime: Runtime[Context])->Literal["call_model", "END"]:
#   if state["input"].find("langchain") > -1:
#     return "call_model"
#   else:
#     return END

# def call_model(state: State, runtime: Runtime[Context]):
#   prompt=f"""User {runtime.context["user_id"]}: {state["input"]}"""
#   return {
#       "output": model.invoke(prompt).content
#   }

# router_builder = StateGraph(
#     state_schema=State
# )
# router_builder.add_node(call_model)
# router_builder.add_conditional_edges(
#                                     # start node
#                                       START,
#                                      # condition
#                                      is_about_langchain,
#                                      # Possible end nodes
#                                      [call_model.__name__, END])
# # Alternative syntax : add type to the edge function: ->Literal["call_model", "__end__"]:

# router_graph = router_builder.compile()

# print("\nRELEVANT--------")
# res_relevant=router_graph.invoke(
#     {"input": "Question about langchain : what is from langchain.agents import AgentState"},
#     context={"user_id": "42"})
# print(res_relevant["output"])

# print("\nIRRELEVANT--------")
# res_irrelevant=router_graph.invoke(
#     {"input": "What is the weather in Scotland"},
#     context={"user_id": "42"})
# print("output" in res_irrelevant)

# Rewrite Create Agent using Langgraph
def my_create_agent(model, tools, system_prompt: str = ""):
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    # Node 1: the LLM decides whether to answer or to call tools
    def call_model(state: MessagesState):
        messages = state["messages"]
        if system_prompt:
            messages = [SystemMessage(system_prompt)] + messages
        return {"messages": [model_with_tools.invoke(messages)]}

    # Node 2: run every tool call from the last AI message
    def call_tools(state: MessagesState):
        last = state["messages"][-1]
        results = []
        for tc in last.tool_calls:
            output = tools_by_name[tc["name"]].invoke(tc["args"])
            results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
        return {"messages": results}

    # Edge: loop back to the tools node, or stop
    def should_continue(state: MessagesState):
        return "call_tools" if state["messages"][-1].tool_calls else END

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(call_tools)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["call_tools", END])
    builder.add_edge("call_tools", "call_model")
    return builder.compile()


# --- Usage ---
@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"It's always sunny in {city}!"

agent = my_create_agent(model, [get_weather], system_prompt="You are a helpful assistant.")
res = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Scotland?"}]})
for m in res["messages"]:
    m.pretty_print()