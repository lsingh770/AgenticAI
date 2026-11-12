#---------------------------
# STEP 1: Picking AI Brain
#---------------------------
#from langchain.chat_models import init_chat_model
#model  = init_chat_model("openai:gpt-4o-mini")
from langchain_ollama import ChatOllama
model = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
)

#---------------------------
# STEP 2: Defining Your Tools
#---------------------------
#from langchain_core.tools import tool
from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
import math

@tool
def add(a:float, b:float) -> float:
    """Add two numbers together. Use for addition operations"""
    return a + b

@tool
def multiply(a: float, b:float) -> float:
    """Multiply two numbers together. Use for multiplication operaions"""
    return a * b

@tool
def divide(a: float, b:float) -> float:
    """Divide the first number by the second number. Return error when divided by zero"""
    if b==0:
        return "Error: Cannot divide by zero"
    return a/b

@tool
def square_root(n:float) -> float:
    """Calculate the square root of provided number."""
    if n < 0:
        return "Error: Cannot take square root of a negative number"
    return math.sqrt(n)

tools = [add, multiply, divide, square_root]

#-------------------------
# STEP 3: Create the Agent
#-------------------------
# create_agent builds a full ReAct loop:
# Reason -> Act (call tool) -> Observe -> Repeat

from langchain.agents import create_agent

agent  = create_agent(
    model = model,
    tools = tools,
)

#-------------------------
# STEP 4: Run the Agent
#-------------------------


def run_agent(question: str):
    """Run the agent and print the execution trace."""
    print(f" User: {question}")
    print("-" * 60)

    result = agent.invoke({
    "messages": [
        ("system", "Respond in plain text only. Do not use LaTeX, "
                   "Markdown formatting, or mathematical symbols. "
                   "Use + for addition, * for multiplication and / for division."
                   "Also give complete reasoning for your answer, the tool used, and show all the steps of your calculations."
                   "When tool is not present then tell the user that the tool is not available and will not be able to answer the question."),
        ("user", question)
        ]
    })

    # Get the final AI response
    final_message = result["messages"][-1]

    print(f"Agent: {final_message.content}\n")

#Simple: Single tool call
#run_agent("What is 42+100 ?")

#Medium: multiple tools call in a sequence
#run_agent("What is 15 multiplied by 8 and then divide by 3")

#Negative: the agent must handle errors gracefully
#run_agent("What is 15 multiplied by 8 and then minus the result by 3")

#run_agent("square root -4 ?")

#Complex: the agent must plan a multistep approach
#run_agent(
#    "I have a rectangle with width 12 and height 7.\n"
#    "What is its area and what is the square root of that area?"
#)

