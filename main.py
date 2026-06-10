from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
import os

load_dotenv()


def main():
    llm = ChatOpenAI(model="gpt-5")
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools)
    result = agent.invoke(
        {"messages": [HumanMessage(content="Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details!")]}
    )
    print(result)


if __name__ == "__main__":
    main()
