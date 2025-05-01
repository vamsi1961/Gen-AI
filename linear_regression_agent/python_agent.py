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
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        deployment_name=gpt4_deployment_name,
        temperature=0,
        stop=["\nObservation", "Observation"]
    )

    # Tools

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



    print(f"writer_prompt is {writer_prompt}")

    writer_agent = create_react_agent(llm=llm, tools=tools, prompt=writer_prompt)
    writer_executor = AgentExecutor(agent=writer_agent, tools=tools, verbose=True, handle_parsing_errors=True)

    WriterTool = Tool(
            name="WriterAgent",
            func=lambda input_text: writer_executor.invoke({"input": input_text})["output"],
            description="Writes or updates machine.py with Python code based on the input prompt."
        )
    
    tools = [WriterTool]

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
    
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
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
    user_request = "write a code to add 2 numbers by taking input and then do logarithm of the result do step by step "
    result = run_agent_with_steps(agent, tools, user_request)
    print(result)

    print("\n💾 Code is good")

if __name__ == "__main__":
    
    main()
