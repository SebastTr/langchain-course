from typing import List
from pydantic import BaseModel, Field
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


class Source(BaseModel):
    """
    Schema for the source used by the agent
    """
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """
    Schema for agent response with answer and sources
    """
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


def main():
    llm = ChatOpenAI(model="gpt-5")
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details!"
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
