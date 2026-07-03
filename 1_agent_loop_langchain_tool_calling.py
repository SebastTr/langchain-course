from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

# Umgebungsvariablen aus der .env-Datei laden (z. B. API-Keys)
load_dotenv()

# Maximale Anzahl an Agent-Iterationen, um Endlosschleifen zu verhindern
MAX_ITERATIOMS = 10

# Verfügbare Modellnamen für Ollama
qwen3 = "qwen3:1.7b"
qwen35 = "qwen3.5:4b"
gemma = "gemma4:e2b"
ministral = "ministral-3:3b"
# Aktiv verwendetes Modell
MODEL = gemma


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"    >> Executing get_product_price with product: {product}")
    # Simulierter Produktkatalog mit festen Preisen
    prices = {"laptop": 999.99, "mouse": 29.99, "keyboard": 79.99}
    # Gibt 0.0 zurück, wenn das Produkt nicht im Katalog gefunden wurde
    return prices.get(product, 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price based on the discount tier"""
    print(
        f"    >> Executing apply_discount with price: {price} and discount_tier: {discount_tier}"
    )
    # Rabattstufen mit ihren prozentualen Abzügen
    discounts = {"gold": 0.20, "silver": 0.10, "bronze": 0.05}
    # Kein Rabatt (0.0), wenn die Stufe unbekannt ist
    discount = discounts.get(discount_tier, 0.0)
    # Rabatt vom Preis abziehen und auf zwei Dezimalstellen runden
    return round(price * (1 - discount), 2)


# @traceable aktiviert LangSmith-Tracing für diese Funktion
@traceable(name="LangChain Agent Loop")
def run_agent(question: str, system_message: str | None = None) -> str | None:
    # Verfügbare Tools registrieren und als Dictionary für schnellen Zugriff ablegen
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    # LLM initialisieren und mit den Tools verknüpfen
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0.1)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    # Hinweis an das Modell anhängen, damit es die SystemMessage berücksichtigt
    question = question + " See SystemMessage for instructions."
    print("=" * 60)

    # Gesprächsverlauf mit System- und erster Benutzernachricht initialisieren
    messages = [
        SystemMessage(content="""
            You are a helpful assistant that answers questions about product prices and discounts.
            Your have access to a product catalog tool and a discount application tool. Use these tools to answer the user's question.
            STRICT INSTRUCTIONS:
            - Never guess or make up information. Always use the tools to get accurate information.
            - If the user asks for the price of a product, use the get_product_price tool. Use a single word product name as the search argument. Iterate until you find the product or determine it is not in the catalog.
            - After you have found a product: If the user asks to apply a discount, use the apply_discount tool. Use a single word discount name as the search argument. Iterate until you find the discount or determine it is not in the catalog.
            - never calculate the price or discount yourself, always use the tools to get the correct answer.
            - Always use the tools to get the correct answer, even if you think you know the answer.
            - only call apply_discount after you have the price from get_product_price, never call apply_discount without first getting the price.
            """),
        HumanMessage(content=question),
    ]

    # Agent-Schleife: das Modell wird wiederholt aufgerufen, bis es eine Antwort
    # ohne Tool-Aufruf liefert oder die maximale Iterationsanzahl erreicht ist
    for iteration in range(MAX_ITERATIOMS):
        print(f" --- Iteration {iteration + 1} --- ")
        ai_message = llm_with_tools.invoke(messages)
        tools_called = ai_message.tool_calls

        # Wenn das Modell kein Tool aufruft, ist die Antwort fertig
        if not tools_called:
            print("AI Response:", ai_message.content)
            print("No tools called, stopping.")
            return str(ai_message.content)

        # Nur den ersten Tool-Aufruf verarbeiten (ein Schritt pro Iteration)
        tool_call = tools_called[0]
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call["id"]

        print(f"Tool called: {tool_name} with args: {tool_args}")

        # Tool anhand des Namens nachschlagen
        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found.")

        # Tool ausführen und Ergebnis speichern
        observation = tool_to_use.invoke(tool_args)
        print(f"    [Tool Result]: {observation}")

        # KI-Nachricht und Tool-Ergebnis zum Gesprächsverlauf hinzufügen,
        # damit das Modell im nächsten Schritt den Kontext kennt
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached, stopping.")
    return None


def main():
    question = "What does a laptop cost after applying a gold discount?"
    result = run_agent(question)
    print("\nFinal Answer:", result)


if __name__ == "__main__":
    main()
