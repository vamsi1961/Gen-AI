from dotenv import load_dotenv
import os
import subprocess
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_react_agent
from langchain.schema import AgentAction, AgentFinish
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import Tool


os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""


# Load .env
load_dotenv()

# Run ReAct-style agent with intermediate steps
def run_agent_with_steps(agent, tools, input_text: str, max_iterations: int = 10):
    intermediate_steps = []

    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1} ---")
        agent_step = agent.invoke({
                "input": input_text,
                "intermediate_steps": intermediate_steps,
            })

        print(f"Agent output: {agent_step}")

        if isinstance(agent_step, AgentFinish):
            print("### Agent Finished ###")
            print(f"Final Answer: {agent_step.return_values['output']}")
            return agent_step.return_values["output"]

        elif isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_input = agent_step.tool_input
            tool = next(t for t in tools if t.name == tool_name)
            observation = tool.run(tool_input)
            intermediate_steps.append((agent_step, observation))
            print(f"Tool: {tool_name} | Input: {tool_input} | Output: {observation}")
        else:
            print("Unexpected output:", agent_step)

    print("Max iterations reached.")
    return None

def main():


    print("Starting AI Code Assistant...")

    # Configure Azure OpenAI
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    gpt4_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

    azure_llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        deployment_name=gpt4_deployment_name,
        temperature=0
    )

    # Custom Tool: Executes machine.py
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

    # Tools
    tools = [PythonREPLTool(), ExecutionTool]

    # Code Writer Agent
    writer_prompt = hub.pull("langchain-ai/react-agent-template").partial(
        instructions="""You are an agent designed to write Python code based on user requirements.
        You have access to a Python REPL.
        Write clean code without any comments and save it as machine.py. If machine.py already exists, update it.
        """
    )

    writer_agent = create_react_agent(llm=azure_llm, tools=tools, prompt=writer_prompt)

    # Code Evaluator Agent
    evaluator_prompt = hub.pull("langchain-ai/react-agent-template").partial(
        instructions="""You are an agent that checks if the code in machine.py works.
        Use the 'RunMachinePy' tool to run it and evaluate its correctness.
        If the code runs without errors and meets the user_request, return 'APPROVED'.
        If not, return 'NEEDS REVISION' and describe the problem.
        """
    )
    evaluator_agent = create_react_agent(llm=azure_llm, tools=tools, prompt=evaluator_prompt)

    # Code generation + evaluation loop
    def process_code_request(user_request):
        max_iterations = 10
        for iteration in range(1, max_iterations + 1):
            print(f"\n===== ITERATION {iteration} =====")

            print("\n[WRITER AGENT]: Generating code...\n")
            writer_output = run_agent_with_steps(writer_agent, tools, f"Write Python code to: {user_request}")
            if writer_output:
                written_code = writer_output
            else:
                written_code = "# Failed to generate code"

            print("\n[EVALUATOR AGENT]: Running and evaluating machine.py...\n")
            eval_output = run_agent_with_steps(evaluator_agent, tools, f"Check if the code in machine.py works correctly for: {user_request}")
            evaluation = eval_output or "NEEDS REVISION - Evaluation failed."

            print("\n[EVALUATION RESULT]:\n", evaluation)

            if "APPROVED" in evaluation:
                print("\nCODE APPROVED!")
                return {"status": "success", "code": written_code, "evaluation": evaluation}

            print("\nCode needs revision. Rewriting...")

        print("\n Maximum iterations reached. Returning best attempt.")
        return {"status": "partial", "code": written_code, "evaluation": evaluation}

    # Start interaction
    user_request = "create a random dataframe of size (3,4) and half the sie and then give the shape of it. do it step by step "
    result = process_code_request(user_request)

    if result["status"] in ["success", "partial"]:
        with open("generated_application.py", "w") as f:
            f.write(result["code"])
        print("\n Code saved to generated_application.py")

if __name__ == "__main__":
    main()
