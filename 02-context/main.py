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

def pretty_chunk(chunk):
  """
  Small util for text chunk display
  """
  if "model" in chunk:
    chunk["model"]["messages"][-1].pretty_print()
  else:
    print(chunk)

@dataclass
class Context:
  username: str

thread_id=uuid.uuid4()

# # Gestion context

# @tool
# def read_runtime_context(runtime: ToolRuntime[Context]):
#   """Read the LangChain runtime context"""
#   print(runtime)
#   return runtime

# context_reader_agent=create_agent(
#     system_prompt="Tell user about the runtime variable",
#     model=model,
#     tools=[read_runtime_context],
#     context_schema=Context,
# )

# context={"username": "Eric"}
# for res in context_reader_agent.stream(
#     {    "messages":["What's in the context"]},
#     context={"username": "Eric"}):
#     pretty_chunk(res)



# State = mémoire courte terme (plus structurée que juste la conversation)
class RecipePreferences(TypedDict):
  is_cake:bool
  is_vegan:bool

# State has messages in addition to other field
# (TODO: find a way to avoid redundancy, multiple inheritance is not possible with typed dict)
class RecipeAgentState(AgentState):
    is_cake:bool
    is_vegan:bool
    # AgentState will add "messages: list[]"


@tool
def update_state(runtime:ToolRuntime[Context, RecipeAgentState],
                # state without the "messages" part
                 new_state:RecipePreferences):
  """
  Update state with user preferences.
  Only include modified keys in new_state."""
  return Command(update={
      **new_state,
      "messages": [
            ToolMessage(
                "Successfully updated user information",
                tool_call_id=runtime.tool_call_id
            )
        ]})
# Note : runtime est un nom d'argument spécial, réservé par LangChain
# https://docs.langchain.com/oss/python/langchain/tools#reserved-argument-names
@tool
def read_state(runtime: ToolRuntime[Context, RecipeAgentState]):
  """Read current state with current user choices"""
  # LangChain handles "messages" internally,
  # the LLM model already has it
  # so we should not output it
  del runtime.state["messages"]
  return runtime.state

agent_with_state=create_agent(
    system_prompt=""""
    Help user to design a recipe
    Remember user ingredient choices in your state
    """,
    model=model,
    state_schema=RecipeAgentState,
    checkpointer=InMemorySaver(),
    tools=[read_state, update_state]
    )

for res in agent_with_state.stream(
    {"messages": "I want to bake a caramel cake, remember this choice in your state"},
    config={"configurable": {"thread_id": thread_id}},
    ):
  pretty_chunk(res)

for res in agent_with_state.stream(
    {"messages": "What do I want in my recipe?"},
    config={"configurable": {"thread_id": thread_id}},
    ):
  pretty_chunk(res)