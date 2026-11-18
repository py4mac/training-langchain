import requests
import os
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="qwen/qwen3.8-flash",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    )

@tool
def fetch_url(url: str) -> str:
    """Fetch text content from a URL"""
    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    # add length limit to avoid issues
    return response.text[0:24000]

system_prompt = """
Use fetch_url when you need to fetch information from a web-page;
quote relevant snippets.
Let's think step by step.
"""

agent = create_agent(
    model=model,#"google_genai:gemini-2.5-flash",
    tools=[fetch_url], # A tool for retrieval
    system_prompt=system_prompt,
)

res=agent.invoke({
    "messages":[
        #"What are the core benefits in LangChain agents? From this doc https://docs.langchain.com/oss/python/langchain/overview"
        "What are the core benefits in LangGraph ? From this doc https://docs.langchain.com/oss/python/langchain/overview"
    ]
})
for msg in res["messages"]:
  msg.pretty_print()