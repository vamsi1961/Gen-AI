from dotenv import load_dotenv
import os
import subprocess
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
gpt4_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    deployment_name=gpt4_deployment_name,
    temperature=0
)

# ---------------------- TOOLS ----------------------

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

# ---------------------- WRITER AGENT ----------------------
writer_prompt = hub.pull("langchain-ai/react-agent-template").partial(
    instructions="""You are an agent designed to write Python code based on a specific subtask.
    You can access a Python REPL.
    Write clean code and update machine.py for the current step."""
)

writer_tools = [PythonREPLTool()]
writer_agent = create_react_agent(llm=llm, tools=writer_tools, prompt=writer_prompt)
writer_executor = AgentExecutor(agent=writer_agent, tools=writer_tools, verbose=True)

WriterTool = Tool(
    name="WriterAgent",
    func=lambda input_text: writer_executor.invoke({"input": input_text})["output"],
    description="Writes or updates machine.py with Python code for a specific step."
)

# ---------------------- EVALUATOR AGENT ----------------------
evaluator_prompt = hub.pull("langchain-ai/react-agent-template").partial(
    instructions="""You are an evaluator. Use the RunMachinePy tool to check if the current code step is implemented correctly.
    Return 'APPROVED' or 'NEEDS REVISION'."""
)

evaluator_tools = [ExecutionTool]
evaluator_agent = create_react_agent(llm=llm, tools=evaluator_tools, prompt=evaluator_prompt)
evaluator_executor = AgentExecutor(agent=evaluator_agent, tools=evaluator_tools, verbose=True)

EvaluationTool = Tool(
    name="EvaluationAgent",
    func=lambda input_text: evaluator_executor.invoke({"input": input_text})["output"],
    description="Evaluates machine.py to see if the current step is complete."
)

# ---------------------- PLANNER AGENT ----------------------
planner_prompt = PromptTemplate.from_template("""
You are a planning agent. Break down the following coding task into exactly 10 sequential development steps.

Task: {input}

Return steps as:
Step 1: ...
Step 2: ...
...
Step 10: ...
""")

planner_agent = (
    {"input": lambda x: x["input"]} | planner_prompt | llm
)

# ---------------------- EXECUTION LOOP ----------------------
def run_code_builder_pipeline(user_goal: str):
    # Generate plan
    print("\n📋 Generating task plan...")
    tasks_text = planner_agent.invoke({"input": user_goal}).content
    task_list = [line for line in tasks_text.split("\n") if line.strip().startswith("Step")]

    for idx, task in enumerate(task_list):
        print(f"\n🚀 Executing {task}")
        success = False

        while not success:
            writer_output = WriterTool.func(task)
            print(f"✍️ WriterAgent Output:\n{writer_output}")

            eval_output = EvaluationTool.func(f"Check if step {idx+1} is complete: {task}")
            print(f"🔍 EvaluationAgent Output:\n{eval_output}")

            if "APPROVED" in eval_output:
                print(f"✅ Step {idx+1} approved.")
                success = True
            else:
                print(f"🔁 Step {idx+1} needs revision, retrying...")

    print("\n🎉 All tasks completed successfully!")


if __name__ == "__main__":
    run_code_builder_pipeline("Write a Python program that generates Fibonacci series up to 10 and prints 'Hello World'. Do it in 10 clean development steps.")
