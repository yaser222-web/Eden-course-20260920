import os
import time
import json
import difflib

os.environ["HTTP_PROXY"] =  "http://127.0.0.1:8010"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8010"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
# این خط را اضافه کن تا مسیر سرور لنگ‌اسمیت از داخل پروکسی گم نشود
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"
#print("API Key loaded:", os.environ.get("LANGCHAIN_API_KEY") is not None)

#--------------------------define the langchain tools------------------------

@tool
def get_product_price(product : str) -> float | str :
    """Look up the price of the product in catalog. The input 'product' must be a simple string 
    of the exact product name, like 'laptop' or 'keyboard'"""
    print(f"   >>Executing get the product price function on product : {product}")
    prices = {"laptop":1299.99, "headphones":149.95, "keyboard": 89.6}
    valid_products = list(prices.keys())
    matches = difflib.get_close_matches(product.lower(), valid_products, n=1, cutoff=0.6)
    if matches :
        matched_product = matches[0] #because the first element of the list is the most similar word to the product list
        print(f" >> smart match : changed '{product}' to '{matched_product}'")
        return prices.get(matched_product)
    else:
        return "Error: product not found in catalog"

@tool
def apply_discount(price : float, discount_tier : str) -> float:
    #discount_tier is the type of discount which is bronze, gold, silver
    """Apply a discount tier to a price and return the final price.
    Available tiers: silver,bronze, gold."""
    print(f"    >>Executing for price :{price}, with discount :{discount_tier}")
    discount_percentage = {"bronze" : 5, "silver" : 12, "gold" : 23}
    discount = discount_percentage.get(discount_tier, 0)
    return round(price*(1-discount/100), 2)

#-------------------Agent loop--------------------
@traceable(name="Lnagchain agent loop")
def run_agent(question : str):
    tools = [get_product_price, apply_discount]
    tools_dic = {t.name : t for t in tools}
    llm = init_chat_model(f"openai:gpt-6-luna", temperature = 0)
    llm_with_tools = llm.bind_tools(tools) # معرفی ابزارها به llm بعد از تعریف llm
    #print(f"question : {question}")
    messages = [
        SystemMessage(content=
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
                "5. If a tool indicates that a product is not found "
                "(for example, returns an 'Error' message), you must NOT " \
                "attempt to guess similar words, fix typos, or call the tool again. " \
                "Immediately stop using tools and directly inform the user that the " \
                "requested product is not available."),
                               HumanMessage(content=question)]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n ________ITERATION________:  {iteration}")
        ai_message = llm_with_tools.invoke(messages) #calling LLM with tools and active it
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\nThe final response is :{ai_message.content}")
            return ai_message.content
        #else:
            #print(" >> The system i using a tool \n")

           # for tools in ai_message.tool_calls:
               # print(f"the tool name is : {tools['name']} ang the args is : {tools['args']}")

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_Id = tool_call.get("id")
        print(f"The tool name is : {tool_name} and tool args :{tool_args}")
        tool_to_use = tools_dic.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"The '{tool_name}' can not be found!\n")

        observation = tool_to_use.invoke(tool_args)
        print(f" [tool result] : {observation}")
        messages.append(ai_message)
        #messages.append(ToolMessage(content=json.dumps(observation)), tool_call_id = tool_call_Id)
        messages.append(ToolMessage(content=str(observation), tool_call_id = tool_call_Id))
    print("ERROR: max irretation reached without final answer")
    return None







if __name__ == "__main__":
    print("Hello langchain agent with bind the tools\n")
    print()
    while(True):
        human_message = input("Please enter your request : (Goodbye to exit)\n")
        if(human_message.lower()=="goodbye"):
            break

        result = run_agent(human_message)
        print(f"The final answer is '{result}'/n")

       

