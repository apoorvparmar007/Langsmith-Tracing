from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent


tool = DuckDuckGoSearchRun()
# response = tool.invoke("latest news")



# llm = ChatOllama(model='qwen3:8b')
agent = create_react_agent(model='ollama:qwen3:8b', tools=[tool])

result = agent.invoke({"messages": [{"role": "user", "content": "Latest updates in Agentic AI"}]})
print(result['messages'][-1])


