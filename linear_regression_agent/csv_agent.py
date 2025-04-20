from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import AzureChatOpenAI
import os


load_dotenv()


def main():
    print("Start...")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    # Make sure to set these deployment names in your Azure OpenAI service
    gpt4_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")


    azure_llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        deployment_name=gpt4_deployment_name,
        temperature=0
    )
    tools = [PythonREPLTool()]
    csv_agent = create_csv_agent(
        llm=azure_llm,
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
        tools = tools
    )

    csv_agent.invoke(
        input={"input": "how many columns are there in file episode_info.csv"}
    )

if __name__ == "__main__":
    main()
