from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os

load_dotenv()


def main():
    print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
