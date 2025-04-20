from typing import Any

from dotenv import load_dotenv
import os
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import AzureChatOpenAI
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent


load_dotenv()

def main():
    print("Start...")
    
    # Azure OpenAI configuration
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    # Make sure to set these deployment names in your Azure OpenAI service
    gpt4_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]
    
    # Initialize Azure OpenAI client instead of OpenAI
    azure_llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        deployment_name=gpt4_deployment_name,
        temperature=0
    )
    
    python_agent = create_react_agent(
        prompt=prompt,
        llm=azure_llm,
        tools=tools,
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    csv_agent_executor = create_csv_agent(
    llm=azure_llm,
    path="episode_info.csv",
    verbose=True,
    allow_dangerous_code=True,
    return_intermediate_steps=False,
    return_direct=True  # <--- IMPORTANT!
)


    ################################ Router Grand Agent ########################################################


    # this does the invoking
    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})
    
    def csv_agent_executor_wrapper(prompt: str) -> dict[str, Any]:
        return csv_agent_executor.invoke({"input": prompt})

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""useful when you need to transform natural language to python and execute the python code,
                          returning the results of the code execution
                          DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor_wrapper,
            description="""useful when you need to answer question over episode_info.csv file,
                        takes the input question and returns the answer after running pandas calculations"""
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=azure_llm,
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    print(
        grand_agent_executor.invoke(
            {
                "input": "which season has the most episodes?",
            }
        )
    )

if __name__ == "__main__":
    main()