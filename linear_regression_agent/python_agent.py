from dotenv import load_dotenv
import os
import subprocess
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import Tool

# Load .env
load_dotenv()


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


    analyser_prompt = hub.pull("langchain-ai/react-agent-template").partial(
        instructions="""You are an Machine learning Scientist agent, You have to write Machine Learning code for the given input and output in csv and question asked.
        You have to finally write a code including preprocesing, ML code and testing for that you have to give a plan of action like for example pre-processing you have give steps like test 
        1. any null values in csv
        2. correlation etc. 
        then writer_executor runs the code and evaluator_executor checks the code if 1st part is good then you have to go next code and code has to update.
        """
    )

    # Code Writer Agent
    writer_prompt = hub.pull("langchain-ai/react-agent-template").partial(
        instructions="""You are an agent designed to write Python code based on user requirements.
        You have access to a Python REPL.
        Write clean dont write any comments and save it as machine.py. If machine.py already exists, update it.
        """
    )
    writer_agent = create_react_agent(llm=azure_llm, tools=tools, prompt=writer_prompt)
    writer_executor = AgentExecutor(agent=writer_agent, tools=tools, verbose=True, handle_parsing_errors=True)

    

    # Code Evaluator Agent
    evaluator_prompt = hub.pull("langchain-ai/react-agent-template").partial(
        instructions="""You are an agent that checks if the code in machine.py works.
        Use the 'RunMachinePy' tool to run it and evaluate its correctness.
        If the code runs without errors, return 'APPROVED'.
        If not, return 'NEEDS REVISION' and describe the problem.
        """
    )
    evaluator_agent = create_react_agent(llm=azure_llm, tools=tools, prompt=evaluator_prompt)
    evaluator_executor = AgentExecutor(agent=evaluator_agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # Code generation + evaluation loop
    def process_code_request(user_request):
        max_iterations = 100
        for iteration in range(1, max_iterations + 1):
            print(f"\n===== ITERATION {iteration} =====")

            print("\n[WRITER AGENT]: Generating code...\n")
            writer_result = writer_executor.invoke(
                {"input": f"Write Python code to: {user_request}"}
            )
            written_code = writer_result["output"]

            print("\n[EVALUATOR AGENT]: Running and evaluating machine.py...\n")
            eval_prompt = f"Check if the code in machine.py works correctly for: {user_request}"
            eval_result = evaluator_executor.invoke({"input": eval_prompt})
            evaluation = eval_result["output"]

            print("\n[EVALUATION RESULT]:\n", evaluation)

            if "APPROVED" in evaluation:
                print("\n✅ CODE APPROVED!")
                return {"status": "success", "code": written_code, "evaluation": evaluation}

            print("\n🔁 Code needs revision. Rewriting...")

        print("\n⚠️ Maximum iterations reached. Returning best attempt.")
        return {"status": "partial", "code": written_code, "evaluation": evaluation}

    # Start interaction
    user_request = input("\nWhat application would you like me to create? ")
    result = process_code_request(user_request)

    if result["status"] in ["success", "partial"]:
        with open("generated_application.py", "w") as f:
            f.write(result["code"])
        print("\n💾 Code saved to generated_application.py")

if __name__ == "__main__":
    main()
