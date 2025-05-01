from dotenv import load_dotenv
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
    # Disable LangChain tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


    azure_llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        deployment_name=gpt4_deployment_name,
        temperature=0
    )
    csv_agent = create_csv_agent(
        llm=azure_llm,
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
        handle_parsing_errors=True
    )

    csv_agent.invoke(
        input={"input": "First, check what columns are available. Then find which column might contain writer names. Finally, tell me which writer has the most episodes."}    
        )

if __name__ == "__main__":
    main()
