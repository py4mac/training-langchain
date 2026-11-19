from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
import uuid 
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, ToolRuntime

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

# LangChain will use the type specific to this provider eg ChatMistralAI
#print(type(model))
#model.invoke("test")

def pretty_chunk(chunk):
  """
  Small util for text chunk display
  """
  if "model" in chunk:
    chunk["model"]["messages"][-1].pretty_print()
  else:
    print(chunk)

no_memory_agent=create_agent(
    model=model,
    system_prompt="Talk about code with user",
    #checkpointer=InMemorySaver(),
)

talking_agent=create_agent(
    model=model,
    system_prompt="Talk about code with user",
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=InMemorySaver(),
)

# # Premier appel (sans mémoire)
# no_memory_agent.invoke({"messages": "My name is Nick"})
# # Second appel sans mémoire : le premier message est oublié (pas de base de données)
# for msg in no_memory_agent.invoke({"messages": "What's my name again?"})["messages"]:
#   msg.pretty_print()

# Memoire court termes
# thread_id=uuid.uuid4()
# print(thread_id)
# config = {
#     "configurable": {"thread_id": thread_id,},
#     #For langsmith: you need to repeat the thread_id
#     # https://docs.langchain.com/langsmith/observability-concepts#threads
#      "metadata": {
#         "thread_id": thread_id  # Explicitly pass to LangSmith
#     }}

# # Premier appel avec checkpointer + thread (= id de la conversation)
# talking_agent.invoke({"messages": "My name is Nick"}, config)

# # Second appel : la conversation est mémorisée
# for msg in talking_agent.invoke({"messages": "What's my name again?"}, config)["messages"]:
#   msg.pretty_print()


# Memoire long termes
store = InMemoryStore()

@tool
def updateMemory(runtime: ToolRuntime[any], preference_key:str, user_preference:str):
  """
  Store one user preference
  """
  namespace=(runtime.context["user_id"], "users")
  runtime.store.put(namespace, "preferences", {preference_key:user_preference})

@tool
def readMemory(runtime: ToolRuntime[any]):
  """
  Get user preferences
  """
  namespace=(runtime.context["user_id"], "users")
  return runtime.store.get(namespace, "preferences")

long_memory_agent=create_agent(
    model=model,
    system_prompt="Store user preferences in order your memory to personnalize your answers",
    tools=[updateMemory, readMemory],
    store=store,
)

user_id="bernard"
res1=long_memory_agent.invoke(
    {"messages":["I do like chocolate"]},
    context={"user_id": user_id}
    )
res2=long_memory_agent.invoke(
    {"messages":["Do I like chocolate?"]},
        context={"user_id": user_id}
    )
print(res1["messages"][-1].content)
print(res2["messages"][-1].content)

# Direct access
print(long_memory_agent.store.get((user_id, "users"), "preferences"))