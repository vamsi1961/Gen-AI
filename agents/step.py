from typing import Union, List, Dict, Any
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.tools.render import render_text_description
from langchain_openai import AzureChatOpenAI
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length called with {text=}")
    text = text.strip("'\n").strip('"')  # stripping away non-alphabetic characters just in case
    return len(text)

@tool
def count_vowels(text: str) -> int:
    """Count the number of vowels in the given text"""
    print(f"count_vowels called with {text=}")
    text = text.strip("'\n").strip('"')
    vowels = "aeiouAEIOU"
    return sum(1 for char in text if char in vowels)

@tool
def count_words(text: str) -> int:
    """Count the number of words in the given text"""
    print(f"count_words called with {text=}")
    text = text.strip("'\n").strip('"')
    return len(text.split())

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
            print(f"intermediate_steps is {intermediate_steps}")
        
    print("Reached maximum iterations without finishing")
    return None

if __name__ == "__main__":
    print("Hello ReAct LangChain with Multiple Steps!")
    
    # Define multiple tools to make the task more complex
    tools = [get_text_length, count_vowels, count_words]
    
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
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    {agent_scratchpad}
    """
    
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    gpt4_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    
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
    
    # Use a more complex question that requires multiple steps
    complex_question = "Tell me about the sentence 'The quick brown fox jumps over the lazy dog', including its length, vowel count, and word count."
    
    final_result = run_agent_with_steps(agent, tools, complex_question)