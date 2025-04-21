from dotenv import load_dotenv
import os
import subprocess
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

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

    analyser_prompt = PromptTemplate.from_template("""
        You are a data scientist. You have to break down the tasks into pre-processing, code and post processing.
        
        In pre-processing you have to write code and check 
            1. if any null value is there
            2. correlation what can be done
            
        Make sure:
        - Steps are very fine-grained (1 action per step).
        - Do not write code.
        - Start from checking if file exists.
        - Proceed to data loading only if it exists.
        - Then proceed to the required computation like printing length.
        - Give instructions and check so agents write code and tests 

        {input}
        """)
    analyser_agent = create_react_agent(llm=azure_llm, tools=[], prompt=analyser_prompt)
    analyser_executor = AgentExecutor(agent=analyser_agent, tools=[], verbose=True,handle_parsing_errors=True)



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
        print("\n[ANALYSER AGENT]: Generating fine-grained steps...\n")
        analysis_result = analyser_executor.invoke({"input": user_request})
        steps_text = analysis_result["output"]

        print("\n[PLAN]:\n", steps_text)
        steps = [line.strip() for line in steps_text.split("\n") if line.lower().startswith("step")]

        written_code = ""
        for i, step in enumerate(steps, 1):
            print(f"\n===== STEP {i}: {step} =====")

            # Writer writes this part
            print("\n[WRITER AGENT]: Writing code for step...\n")
            writer_result = writer_executor.invoke({"input": f"Update machine.py to implement: {step}"})
            written_code = writer_result["output"]

            # Evaluator runs it
            print("\n[EVALUATOR AGENT]: Evaluating code...\n")
            eval_result = evaluator_executor.invoke({"input": f"Check if machine.py correctly implements: {step}"})
            evaluation = eval_result["output"]

            print("\n[EVALUATION RESULT]:\n", evaluation)

            if "APPROVED" in evaluation:
                print("✅ Step approved. Proceeding.\n")
            else:
                print("❌ Step failed. Halting pipeline.\n")
                return {"status": "partial", "step": i, "code": written_code, "evaluation": evaluation}

        return {"status": "success", "code": written_code, "evaluation": "All steps approved."}

    # Start interaction
    user_request = input("\nWhat application would you like me to create? ")
    result = process_code_request(user_request)

    if result["status"] in ["success", "partial"]:
        with open("generated_application.py", "w") as f:
            f.write(result["code"])
        print("\n💾 Code saved to generated_application.py")

if __name__ == "__main__":
    main()
