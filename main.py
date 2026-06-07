from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os

load_dotenv()


def main():
    print("Hello from langchain-course!")

    os.environ["MEINE_VARIABLE"] = "mein_wert"
    print(os.environ.get("MEINE_VARIABLE"))

    information = """
        Explain the concept of machine learning in one sentence.
    """

    summary_template = """
        Explain the following: 
        {information}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm_gpt = ChatOpenAI(temperature=0, model="gpt-5.4-mini")
    llm_gemma4 = ChatOllama(temperature=0, model="gemma4:e2b")

    response = {}

    chain_llm_gpt = summary_prompt_template | llm_gpt
    response_llm_gpt = chain_llm_gpt.invoke(input={"information": information})

    chain_llm_gemma4 = summary_prompt_template | llm_gemma4
    response_llm_gemma4 = chain_llm_gemma4.invoke(input={"information": information})

    response = {
        "response_llm_gpt":response_llm_gpt.content,
        "response_llm_gemma4":response_llm_gemma4.content,
    }

    print(response)


if __name__ == "__main__":
    main()
