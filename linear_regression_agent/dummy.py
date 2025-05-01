from dotenv import load_dotenv
import os
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
from typing import List


# Load .env
load_dotenv()


def main():
    print("Starting AI Code Assistant...")

    # Configure Azure OpenAI
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    gpt4_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    azure_llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        deployment_name=gpt4_deployment_name,
        temperature=0
    )

    # Tools

    def write_to_machine_py(code: str) -> str:
        try:
            if code.startswith('"""') and code.endswith('"""'):
                code = code[3:-3]
            elif code.startswith("'''") and code.endswith("'''"):
                code = code[3:-3]
            code = code.rstrip('"\'')

            with open("machine.py", "w") as f:
                f.write(code.strip())
            return "machine.py updated successfully."
        except Exception as e:
            return f"Failed to update machine.py: {str(e)}"

    FileWriteTool = Tool(
        name="WriteToFile",
        func=write_to_machine_py,
        description="Writes the provided Python code to machine.py file. Input should be valid Python code as a string."
    )

    def read_machine_py(_: str = "") -> str:
        try:
            with open("machine.py", "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    FileReadTool = Tool(
        name="ReadFile",
        func=read_machine_py,
        description="Reads the contents of machine.py and returns it as a string. Input can be empty."
    )

    tools = [PythonREPLTool(), FileWriteTool, FileReadTool]

    # Code Writer Agent
    writer_prompt = hub.pull("langchain-ai/react-agent-template").partial(
        instructions="""
        You are an agent designed to write Python code based on user requirements.
        Use the ReadFile tool to see current contents of machine.py **before** updating it.
        Write clean Python code using Python REPL to test it (if needed).
        Do NOT include ```python or ``` markers — write only valid raw Python code.
        Do not explain, do not comment, do not return markdown.
        Once finalized, use the WriteToFile tool to overwrite machine.py with the updated version.
        Always include **all** code, not just what changed.
        Avoid using comments.
        """
    )

    print(f"writer_prompt is {writer_prompt}")

    writer_agent = create_react_agent(llm=azure_llm, tools=tools, prompt=writer_prompt)
    writer_executor = AgentExecutor(agent=writer_agent, tools=tools, verbose=True, handle_parsing_errors=True)

    WriterTool = Tool(
        name="WriterAgent",
        func=lambda input_text: writer_executor.invoke({"input": input_text})["output"],
        description="Writes or updates machine.py with Python code based on the input prompt."
    )
    
    tools = [PythonREPLTool(), FileWriteTool, FileReadTool,WriterTool]

    # ReAct agent prompt setup
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

        print("Reached maximum iterations without finishing")
        return None

    # Start interaction
    user_request = (
        "write a python code for fibonacci series upto 10 and also write print hello world "
        "program do step by step update the code"
    )
    result = run_agent_with_steps(agent, tools, user_request)
    print(result)

    print("\n💾 Code is good")


if __name__ == "__main__":
    main()
