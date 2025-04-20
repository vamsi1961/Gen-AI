from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import AzureChatOpenAI

import os



load_dotenv()


def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    # Make sure to set these deployment names in your Azure OpenAI service
    gpt4_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")


    prompt = base_prompt.partial(instructions=instructions)

    azure_llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        deployment_name=gpt4_deployment_name,
        temperature=0
    )

    tools = [PythonREPLTool()]
    agent = create_react_agent(
        prompt=prompt,
        llm=azure_llm,
        tools=tools,
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executor.invoke(
        input={
            "input": """ Create a Python file named 'hello_world.py' with content that prints 'Hello World' 
    and then execute that file to show the output."""
        }
    )


if __name__ == "__main__":
    main()
