from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIOMS = 10
MODEL = "qwen3.5:397b-cloud"

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"    >> Executing get_product_price with product: {product}")
    
    prices = {
        "laptop": 999.99,
        "mouse": 29.99,
        "keyboard": 79.99
    }
    
    return prices.get(product, 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price based on the discount tier"""
    print(f"    >> Executing apply_discount with price: {price} and discount_tier: {discount_tier}")
    
    discounts = {
        "gold": 0.20,
        "silver": 0.10,
        "bronze": 0.05
    }
    
    discount = discounts.get(discount_tier, 0.0)
    return round(price * (1 - discount), 2) 

def run_agent(question: str):
    pass