from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent

load_dotenv()


def main():
    print("Start...")


    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4o"),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    csv_agent.invoke(
        input={"input": "how many columns are there in file episode_info.csv"}
    )
    csv_agent.invoke(
        input={
            "input": "which season has most episodes?"
        }
    )

    csv_agent.invoke(
        input={
            "input": "write a simple python hello world code"
        }
    )
    
if __name__ == "__main__":
    main()
