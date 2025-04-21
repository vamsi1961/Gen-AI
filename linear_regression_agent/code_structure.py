import os
import subprocess
from dotenv import load_dotenv

from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_react_agent
from langchain.schema import AgentAction, AgentFinish
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.runnables import RunnableMap

# Load environment
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

# Azure OpenAI config
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
gpt4_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

# LLM config
azure_llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    deployment_name=gpt4_deployment_name,
    temperature=0
)

# Tool to run machine.py
def execute_machine_py(_):
    try:
        result = subprocess.run(["python", "machine.py"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return f"APPROVED - machine.py ran successfully.\nOutput:\n{result.stdout.strip()}"
        else:
            return f"NEEDS REVISION - machine.py failed.\nErrors:\n{result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "NEEDS REVISION - machine.py timed out during execution."

ExecutionTool = Tool(
    name="RunMachinePy",
    func=execute_machine_py,
    description="Runs machine.py using subprocess and returns output or errors."
)

# Shared tools
tools = [PythonREPLTool(), ExecutionTool]

# Writer agent (ReAct style)
writer_prompt = hub.pull("langchain-ai/react-agent-template").partial(
    instructions="""You are an agent designed to write Python code based on user requirements.
    You have access to a Python REPL.
    Write clean code without any comments and save it as machine.py. If machine.py already exists, update it.
    """
)
writer_agent = create_react_agent(llm=azure_llm, tools=tools, prompt=writer_prompt)

# Evaluator agent (ReAct style)
evaluator_prompt = hub.pull("langchain-ai/react-agent-template").partial(
    instructions="""You are an agent that checks if the code in machine.py works.
    Use the 'RunMachinePy' tool to run it and evaluate its correctness.
    If the code runs without errors and meets the user_request, return 'APPROVED'.
    If not, return 'NEEDS REVISION' and describe the problem.
    """
)
evaluator_agent = create_react_agent(llm=azure_llm, tools=tools, prompt=evaluator_prompt)

# ReAct interpreter
def run_agent_with_steps(agent, tools, input_text: str, max_iterations: int = 10):
    intermediate_steps = []

    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1} ---")
        agent_step = agent.invoke({
            "input": input_text,
            "intermediate_steps": intermediate_steps,
        })

        if isinstance(agent_step, AgentFinish):
            print(f"Final Answer: {agent_step.return_values['output']}")
            return agent_step.return_values["output"]

        elif isinstance(agent_step, AgentAction):
            tool = next(t for t in tools if t.name == agent_step.tool)
            observation = tool.run(agent_step.tool_input)
            intermediate_steps.append((agent_step, observation))
            print(f"Tool: {agent_step.tool} | Input: {agent_step.tool_input} | Output: {observation}")
        else:
            print("Unexpected agent output:", agent_step)

    return "Max iterations reached."

# Scratchpad formatter
def format_log_to_str(intermediate_steps):
    scratchpad = ""
    for action, observation in intermediate_steps:
        scratchpad += f"Thought: {action.log}\n"
        scratchpad += f"Action: {action.tool}\n"
        scratchpad += f"Action Input: {action.tool_input}\n"
        scratchpad += f"Observation: {observation}\n"
    return scratchpad

# Task Creator Agent setup
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
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template).partial(
    tools="WriterAgent: writes code\nEvaluatorAgent: checks correctness",
    tool_names="WriterAgent, EvaluatorAgent"
)

task_creator_llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    deployment_name=gpt4_deployment_name,
    temperature=0,
    stop=["\nObservation", "Observation"]
)

task_creator_agent = (
    RunnableMap({
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x.get("intermediate_steps", [])),
    })
    | prompt
    | task_creator_llm
    | ReActSingleInputOutputParser()
)

# Wrap writer and evaluator agents as tools
WriterTool = Tool(
    name="WriterAgent",
    func=lambda input: run_agent_with_steps(writer_agent, tools, input),
    description="Writes Python code to machine.py based on the input"
)

EvaluatorTool = Tool(
    name="EvaluatorAgent",
    func=lambda input: run_agent_with_steps(evaluator_agent, tools, input),
    description="Checks machine.py for correctness and functionality"
)

# Final executor
def run_task_creator(input_text, max_iterations=10):
    tools_map = {
        "WriterAgent": WriterTool,
        "EvaluatorAgent": EvaluatorTool,
    }
    intermediate_steps = []

    for i in range(max_iterations):
        print(f"\n--- Task Creator Iteration {i+1} ---")
        output = task_creator_agent.invoke({
            "input": input_text,
            "intermediate_steps": intermediate_steps
        })

        if isinstance(output, AgentFinish):
            print("Final Answer:", output.return_values["output"])
            return output.return_values["output"]

        elif isinstance(output, AgentAction):
            tool = tools_map[output.tool]
            result = tool.run(output.tool_input)
            intermediate_steps.append((output, result))
            print(f"Tool: {output.tool} | Input: {output.tool_input} | Output: {result}")
        else:
            print("Unexpected output from task creator:", output)

    print("Max iterations reached.")
    return "Task creator could not resolve the request."

# Main
if __name__ == "__main__":
    user_request = "Create a random dataframe of shape (3,4), then reduce it to half the size, and print its final shape. Do it step-by-step."
    result = run_task_creator(user_request)
    print("\nFinal Result:\n", result)
