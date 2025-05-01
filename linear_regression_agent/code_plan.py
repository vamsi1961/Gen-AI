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

def write_execute_py(code: str) -> str:
    try:
        # Strip surrounding triple quotes if present
        if code.startswith('"""') and code.endswith('"""'):
            code = code[3:-3]
        elif code.startswith("'''") and code.endswith("'''"):
            code = code[3:-3]
        elif code.startswith("```") and code.endswith("```"):
            code = code[3:-3]
        # Properly handle markdown code blocks

        if code.startswith("```python") and code.endswith("```"):
            code = code[len("```python"):-3].strip()

        elif code.startswith("```") and code.endswith("```"):
            code = code[3:-3].strip()
        
        # Ensure no language identifier at start of file
        code_lines = code.strip().split('\n')
        if code_lines and code_lines[0].strip() == 'python':
            code = '\n'.join(code_lines[1:])

        # Optionally strip a trailing quote if one got appended wrongly
        code = code.rstrip('"\'')

        with open("machine.py", "w") as f:
            f.write(code.strip())
        print("Running machine.py ...")
        result = subprocess.run(["python", "machine.py"], capture_output=True, text=True, timeout=10)
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        if result.returncode == 0:
            return f"APPROVED - machine.py ran successfully.\nOutput:\n{result.stdout.strip()}"
        else:
            return f"NEEDS REVISION - machine.py failed.\nErrors:\n{result.stderr.strip()}"

    except Exception as e:
        return f"Failed to update machine.py: {str(e)}"
    except subprocess.TimeoutExpired:
        return "NEEDS REVISION - machine.py timed out during execution."

FileWriteTool = Tool(
    name="WriteToFile",
    func=write_execute_py,
    description="Writes the provided Python code to machine.py file and executes it if it says it require revision then you have to call PythonREPLTool to re-write the code. Input should be valid Python code as a string."
)

    
tools = [PythonREPLTool(), FileWriteTool]
# Code Writer Agent
def read_existing_code() -> str:
    """Read the content of machine.py if it exists."""
    try:
        if os.path.exists("machine.py"):
            with open("machine.py", "r") as f:
                return f.read()
        return "No existing code found."
    except Exception as e:
        return f"Error reading machine.py: {str(e)}"

# Get the existing code
existing_code = read_existing_code()
print(f"Existing code found: {'Yes' if existing_code != 'No existing code found.' else 'No'}")

instructions = f"""
    You are an agent designed to write Python code based on user requirements.
    
    Here is the EXISTING CODE from machine.py that you should use as your starting point:
    
    ```python
    {existing_code}
    ```
    Only Test the code using PythonRePL tool dont test the final code. See what you can add to meet the requirements if you have to remove it to meet requirements then remove it
    Modify the existing code to meet the new requirements rather than writing from scratch.
    Only make necessary changes to fulfill the requirements.
    Once you have the code working, use the WriteToFile tool to save it.
    
    Do NOT include ```python or ``` markers in your final answer - write only valid raw Python code.
    """
    
# Code Writer Agent with embedded existing code
writer_prompt = hub.pull("langchain-ai/react-agent-template").partial(
        instructions=instructions
    )


# print(f"writer_prompt is {writer_prompt}")

writer_agent = create_chain_react_agent(llm=llm, tools=tools, prompt=writer_prompt)
writer_executor = AgentExecutor(agent=writer_agent, tools=tools, verbose=True, handle_parsing_errors=True)

WriterTool = Tool(
        name="WriterAgent",
        func=lambda input_text: writer_executor.invoke({"input": input_text})["output"],
        description="Writes or updates machine.py with Python code based on the input prompt."
    )

tools = [WriterTool]

# Define tools for the react agent
# tools = [WriterTool , TavilySearchResults(max_results=3)]  
# Include both Python REPL and CSV tools

# Create a React agent with the tools

template = """
    Answer the following questions as best you can. You have access to the following tools:
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: follow the instruction and write the code
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    {agent_scratchpad}
    """
def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
        for tool in tools:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"Tool with name {tool_name} not found")

prompt = PromptTemplate.from_template(template=template).partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)


code_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    }
    | prompt
    | llm
    | ReActSingleInputOutputParser()
    )

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

    intermediate_steps = []
    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1} ---")
        agent_step = code_agent.invoke({
            "input": task,
            "agent_scratchpad": intermediate_steps,
        })

        print(f"Agent output: {agent_step}")

        # Handle dict-based output from ReAct pipeline
        if agent_step.get("type") == "agent_finish":
            print("### Agent Finished ###")
            print(f"Final Answer: {agent_step['return_values']['output']}")
            return {"past_steps": [(task, agent_step)]}

        elif agent_step.get("type") == "agent_action":
            tool_name = agent_step["tool"]
            print(f"Selected tool: {tool_name}")
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step["tool_input"]
            observation = tool_to_use.func(str(tool_input))
            print(f"Observation: {observation}")
            intermediate_steps.append((agent_step, str(observation)))

        else:
            print("Unknown agent output format")

            
    print("Reached maximum iterations without finishing")
    return None

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
# inputs = {"input": "what is the hometown of the current Australia open winner?"}
inputs={"input": "First, write a code to add 2 numbers by taking input Then do logarithm of the result do step by step. Do it in only 2 steps"}    

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