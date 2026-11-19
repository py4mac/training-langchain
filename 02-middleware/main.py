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
import time
from langchain.agents.middleware import wrap_tool_call
from langchain.agents.middleware import ToolErrorMiddleware, ToolCallRequest

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


# @wrap_tool_call
# def log_tool_latency(request, handler):
#   # We could also interrupt the tool using a signal
#   # https://stackoverflow.com/questions/492519/timeout-on-a-function-call
#   start_time = time.time()
#   # handler = the tool ; request = the params
#   result = handler(request)
#   end_time = time.time()
#   # Calculate the execution time
#   execution_time = end_time - start_time
#   print(f"Function result: {result}")
#   print(f"Function execution time: {execution_time:.2}")
#   return result

# @tool
# def long_running_tool():
#   """A tool for a demo"""
#   time.sleep(0.3)

# middleware_agent=create_agent(
#     model=model,
#     tools=[long_running_tool],
#     middleware=[log_tool_latency]
# )

# res=middleware_agent.invoke({"messages": "Call the tool"})
# print(res)

# def on_error(exc: Exception, request: ToolCallRequest) -> str | None:
#     # L'agent va détecter le problème, mais ne va pas s'arrêter
#     if isinstance(exc, ZeroDivisionError):
#         return f"`{request.tool_call['name']}` failed with {type(exc).__name__}."
#     # Toute exception non rattrapée va faire planter l'agent

def division_tool(a: int, b: int):
  """Divides a with b
  a and b must be integer
  """
  return a/b

# zero_division_agent=create_agent(
#     model=model,
#     system_prompt="You are a mathematician.",
#     tools=[division_tool],
#     # comment this line to see the actual exception
#     middleware=[ToolErrorMiddleware(on_error=on_error)]
# )


@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        return handler(request)
    except ZeroDivisionError as exc:
        # L'erreur est renvoyée au modèle sous forme de message : l'agent continue
        return ToolMessage(
            content=f"`{request.tool_call['name']}` failed with {type(exc).__name__}: {exc}",
            tool_call_id=request.tool_call["id"],  # obligatoire pour relier la réponse à l'appel
            status="error",
        )
    # Toute autre exception remonte et fait planter l'agent

zero_division_agent = create_agent(
    model=model,
    system_prompt="You are a mathematician.",
    tools=[division_tool],
    middleware=[handle_tool_errors],
)


division_res=zero_division_agent.invoke({"messages": "For the sake of a demo, try to use the division tool with '0', what happens?"})
for msg in division_res["messages"]:
  msg.pretty_print()