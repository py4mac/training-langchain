from dotenv import load_dotenv
import os, re
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
from langchain_docling.loader import ExportType, DoclingLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

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

splitter = MarkdownHeaderTextSplitter([("#", "title 1"), ("##", "title 2"), ("###", "title 3")])

def ingest_doc():
  url_to_parse="https://docs.langchain.com/oss/python/langchain/overview"
  loader = DoclingLoader(url_to_parse, export_type=ExportType.MARKDOWN)
  docs=loader.load()
  doc=docs[0]
  chunks=splitter.split_text(doc.page_content)
  return [chunk.page_content for chunk in chunks]

chunks=ingest_doc()


def rag(user_query):
  relevant_chunks=[]
  if re.search("langchain", user_query, re.IGNORECASE):
    relevant_chunks=[
        chunk
        for chunk in chunks
        if re.search("langchain", chunk, re.IGNORECASE)
        ]
  if not len(relevant_chunks):
    return "No relevant doc for this questions"
  prompt=f"""
    User question:
    {user_query}
    Relevant docs:
    {"\n---\n".join(relevant_chunks)}
  """
  return model.invoke(prompt).content


print(rag("How to create an agent with LangChain v1"))