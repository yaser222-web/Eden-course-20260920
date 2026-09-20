import os
import time

os.environ["HTTP_PROXY"] =  "http://127.0.0.1:8000"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8000"
# این خط را اضافه کن تا مسیر سرور لنگ‌اسمیت از داخل پروکسی گم نشود
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
MAX_ITERATIONS = 10
MODEL = "queen-7b"
print("API Key loaded:", os.environ.get("LANGCHAIN_API_KEY") is not None)

#--------------------------define the langchain tools------------------------

@tool
def get_the_product_price(product : str) -> float :
    """Look up the price of the product in catalog"""
    print(f"   >>Executing get the product price function on product : {product}")
    prices = {"laptop":1299.99, "headphones":149.95, "keyboard": 89.6}
    return prices.get("product", 0)

@tool
def apply_discount(price : float, discount_tier : str) -> float:
    #discount_tier is the type of discount which is bronze, gold, silver
    """Apply a discount tier to a price and return the final price.
    Available tiers: silver,bronze, gold."""
    print(f"    >>Executing for price :{price}, with discount :{discount_tier}")
    discount_percentage = {"bromze" : 5, "silver" : 12, "gold" : 23}
    discount = discount_percentage.get(discount_tier, 0)
    return round(price*(1-discount/100), 2)

#-------------------Agent loop--------------------
@traceable(name="Lnagchain agent loop")
def run_agent(question : str):
    pass

if __name__ == "__main__":
    print("Hello lnagchain Agent (.bind_tools)!")
    print()
    result = run_agent("Wha is the price of laptop after applying a gold discount")

