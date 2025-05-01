from dotenv import load_dotenv
import os
import subprocess
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import Tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.prompts import PromptTemplate
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.schema import AgentAction, AgentFinish
from typing import Union, List, Dict, Any


# Load .env
load_dotenv()

def main():
    print("Starting AI Code Assistant...")

    # Configure Azure OpenAI
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    gpt4_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print(" I am good")
    azure_llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        deployment_name=gpt4_deployment_name,
        temperature=0
    )
    print("I am perfect")
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

    # tools = [PythonREPLTool()]
    print(f"writer_prompt is {writer_prompt}")

    writer_agent = create_react_agent(llm=azure_llm, tools=tools, prompt=writer_prompt)
    writer_executor = AgentExecutor(agent=writer_agent, tools=tools, verbose=True, handle_parsing_errors=True)

    WriterTool = Tool(
            name="WriterAgent",
            func=lambda input_text: writer_executor.invoke({"input": input_text})["output"],
            description="Writes or updates machine.py with Python code based on the input prompt."
        )

    # Code Evaluator Agent
    evaluator_prompt = hub.pull("langchain-ai/react-agent-template").partial(
        instructions="""You are an agent that checks if the code in machine.py works.
        Use the 'RunMachinePy' tool to run it and evaluate its correctness.
        If the code runs without errors, return 'APPROVED'.
        If not, return 'NEEDS REVISION' and describe the problem.
        """
    )
    # tools = [ExecutionTool]
    evaluator_agent = create_react_agent(llm=azure_llm, tools=tools, prompt=evaluator_prompt)
    evaluator_executor = AgentExecutor(agent=evaluator_agent, tools=tools, verbose=True, handle_parsing_errors=True)


    EvaluationTool = Tool(
    name="EvaluationAgent",
    func=lambda input_text: evaluator_executor.invoke({"input": input_text}),
    description="execute the funtion and pass the message"
)
    

    tools = [WriterTool, EvaluationTool]

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
    Thought: If the EvaluationTool says code is APPROVED then only you have to confirm that code is good.After wrtiting the code using WriterTool you have to use EvaluationTool to check if it correct or not
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    {agent_scratchpad}
    """
    
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        deployment_name=gpt4_deployment_name,
        temperature=0,
        stop=["\nObservation", "Observation"]
    )
    
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )
    def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
        for tool in tools:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"Tool with name {tool_name} not found")


    # Code generation + evaluation loop
    def run_agent_with_steps(agent, tools, input_text: str, max_iterations: int = 10):
        """Run the agent for multiple iterations until it reaches a final answer or max iterations"""
        intermediate_steps = []
        
        for i in range(max_iterations):
            print(f"\n--- Iteration {i+1} ---")
            agent_step = agent.invoke(
                {
                    "input": input_text,
                    "agent_scratchpad": intermediate_steps,
                }
            )
            
            print(f"Agent output: {agent_step}")
            
            if isinstance(agent_step, AgentFinish):
                print("### Agent Finished ###")
                print(f"Final Answer: {agent_step.return_values['output']}")
                return agent_step
                
            if isinstance(agent_step, AgentAction):
                tool_name = agent_step.tool
                print(f"Selected tool: {tool_name}")
                tool_to_use = find_tool_by_name(tools, tool_name)
                tool_input = agent_step.tool_input
                observation = tool_to_use.func(str(tool_input))
                print(f"Observation: {observation}")
                intermediate_steps.append((agent_step, str(observation)))
                print(f"intermediate_steps is {intermediate_steps}")
            
        print("Reached maximum iterations without finishing")
        return None

    # Start interaction
    user_request = "write a python code for fibonacci series upto 10 and also write print hello world program do step by step update the code "
    result = run_agent_with_steps(agent, tools, user_request)
    print(result)

    print("\n Code is good")

if __name__ == "__main__":
    
    main()

"https://chatgpt.com/share/68111baf-edc8-800a-b2e6-824df2e9157d"