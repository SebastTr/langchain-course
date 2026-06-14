from dotenv import load_dotenv

load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIOMS = 10
MODEL = "qwen3:1.7b"


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"    >> Executing get_product_price with product: {product}")
    prices = {"laptop": 999.99, "mouse": 29.99, "keyboard": 79.99}
    return prices.get(product, 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price based on the discount tier"""
    print(
        f"    >> Executing apply_discount with price: {price} and discount_tier: {discount_tier}"
    )
    discounts = {"gold": 0.20, "silver": 0.10, "bronze": 0.05}
    discount = discounts.get(discount_tier, 0.0)
    return round(price * (1 - discount), 2)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0.1)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(content="""
            You are a helpful assistant that answers questions about product prices and discounts.
            Your have access to a product catalog tool and a discount application tool. Use these tools to answer the user's question.
            STRICT INSTRUCTIONS:
            - Never guess or make up information. Always use the tools to get accurate information.
            - If the user asks for the price of a product, use the get_product_price tool.
            - If the user asks to apply a discount, use the apply_discount tool.
            - never calculate the price or discount yourself, always use the tools to get the correct answer.
            - If you don't know the answer, say you don't know instead of trying to guess.
            - Always use the tools to get the correct answer, even if you think you know the answer.
            - only call apply_discount after you have the price from get_product_price, never call apply_discount without first getting the price.
            """),
        HumanMessage(content=question),
    ]

    for iteration in range(MAX_ITERATIOMS):
        print(f" --- Iteration {iteration + 1} --- ")
        ai_message = llm_with_tools.invoke(messages)
        tools_called = ai_message.tool_calls

        if not tools_called:
            print("AI Response:", ai_message.content)
            print("No tools called, stopping.")
            return ai_message.content

        tool_call = tools_called[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool called: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found.")

        observation = tool_to_use.invoke(tool_args)
        print(f"    [Tool Result]: {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached, stopping.")
    return None


def main():
    question = "What is the price of a laptop with a gold discount?"
    result = run_agent(question)
    print("\nFinal Answer:", result)


if __name__ == "__main__":
    main()
