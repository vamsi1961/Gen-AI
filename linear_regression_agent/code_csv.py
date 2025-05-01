import os
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import create_react_agent as create_graph_react_agent
from langchain.agents import create_react_agent as create_chain_react_agent
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Union
from langchain.tools import Tool
from langgraph.graph import END
from langgraph.graph import StateGraph, START
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor
from langchain import hub
import subprocess
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain.schema import AgentAction, AgentFinish


# Load environment variables
load_dotenv()

# Configure Azure OpenAI
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
gpt4_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

# Disable LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    deployment_name=gpt4_deployment_name,
    api_version="2024-08-01-preview",
    temperature=0
)

csv_agent = create_csv_agent(
    llm=llm,
    path="episode_info.csv",
    verbose=True,
    allow_dangerous_code=True
)

# Create a tool from the CSV agent
csv_tool = Tool(
    name="CSVAgent",
    func=lambda input_text: csv_agent.invoke({"input": input_text})["output"],
    description="Write and execute CSV code based on requirements."
)

# Define tools for the react agent
tools = [csv_tool, TavilySearchResults(max_results=3)]  # Include both Python REPL and CSV tools

# Create a React agent with the tools
prompt = "You are a helpful assistant specialized in data analysis. You have access to a CSV file and Python tools to help answer questions about the data."
agent_executor = create_graph_react_agent(llm, tools, prompt=prompt)

# Define state types for the workflow
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

class Response(BaseModel):
    """Response to user."""
    response: str

class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

# Define prompts for planning and replanning
planner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """For the given objective, create a step-by-step plan.
        Your steps will be executed by a tool that can write and run code.
        Write concise, clear steps that explain what needs to be done.
        Focus only on the necessary steps to complete the objective.
        """,
    ),
    ("placeholder", "{messages}"),
])

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, update your step-by-step plan based on progress so far.

Your objective was this: {input}

Your original plan was: {plan}

You have currently completed these steps: {past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, 
provide a final response. Otherwise, list only the remaining steps that need to be done.
Do not include previously completed steps in your updated plan."""
)

# Create the planner and replanner
planner = planner_prompt | llm.with_structured_output(Plan)
replanner = replanner_prompt | llm.with_structured_output(Act)
def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
        for tool in tools:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"Tool with name {tool_name} not found")
        
# Define workflow functions
def execute_step(state: PlanExecute,max_iterations: int = 10):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan: {plan_str}\n\n You are tasked with executing step {1}, {task}."""
    
    agent_response = agent_executor.invoke(
        {"messages": [("user", task_formatted)]}
    )

    # print(f"agent_response : {agent_response}")
    print(f"type of agent response is {type(agent_response)}")

    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


def plan_step(state: PlanExecute):
    plan = planner.invoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}

def replan_step(state: PlanExecute):
    output = replanner.invoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        print(" I am replanning")
        return {"plan": output.action.steps}

def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"

# Build the workflow
workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)

# Connect the nodes
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges(
    "replan",
    should_end,
    ["agent", END],
)

# Compile the workflow
app = workflow.compile()

# Define the input for the workflow
config = {"recursion_limit": 50}  
inputs = {
        "input": "First, check what columns are available. Then find which column might contain writer names. Finally, tell me which writer has the most episodes."
    }
# Run the workflow
def main():
    print("Starting workflow to analyze episode_info.csv...")
    for event in app.stream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print("\n--- STEP UPDATE ---")
                print(v)
    print("\nWorkflow completed.")

# Execute the workflow
if __name__ == "__main__":
    main()