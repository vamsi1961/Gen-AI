from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools import TavilySearchResults
from typing import List


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



# Define your JSON data here
raw_json_string = """
{
  "name": "John Doe",
  "age": 30,
  "occupation": "Software Engineer",
  "skills": ["Python", "JavaScript", "Machine Learning"],
    "email": "john.doe@example.com",
    "phone": "123-456-7890"
  }
}
"""

# Define the prompt template
prompt = PromptTemplate.from_template("""
You are a smart JSON data extractor.

JSON:
{json_data}

Question:
{query}

Answer in concise JSON or text.
""")

# Initialize the LLM
# llm = OpenAI(temperature=0)  # Make sure to set your OPENAI_API_KEY


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

# Define the LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Define the function that uses the chain
def query_json_tool(query, json_data):
    return chain.run({"json_data": json_data, "query": query})

# Create a simpler function that just returns the raw JSON
def get_raw_json():
    return raw_json_string

# Create the tools
json_query_tool = Tool(
    name="JSONQueryTool",
    func=lambda q: query_json_tool(raw_json_string,q),
    description="Use this to extract information from a JSON blob using natural language queries."
)

raw_json_tool = Tool(
    name="GetRawJSON",
    func=get_raw_json,
    description="Returns the complete JSON data that's available for querying."
)

# Initialize TavilySearchResults
tavily_tool = TavilySearchResults()

# Define a simple tool for string length
def string_length(input_string: str) -> str:
    return f"Length is {len(input_string)}"

get_length = Tool(
    name="StringLengthCalculator",
    func=string_length,
    description="Calculates the number of characters in a string."
)

# Add your tools to a list - put the raw JSON tool first for visibility
tools = [raw_json_tool, json_query_tool, tavily_tool, get_length]

def render_text_description(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

def format_log_to_str(intermediate_steps):
    log_str = ""
    for action, observation in intermediate_steps:
        log_str += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\n"
    return log_str

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

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    To answer questions about JSON data, you should use the GetRawJSON tool first to see what JSON data is available.
    
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
    Thought: {agent_scratchpad}
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

# Print the tools to verify they're correctly defined
print(tools)

# Now let's test both questions
print("\n=== Testing 'what is the json input' ===")
complex_question = "what is the json input"
final_result = run_agent_with_steps(agent, tools, complex_question)

print("\n=== Testing 'what is John's age' ===")
specific_question = "what is John's age"
final_result2 = run_agent_with_steps(agent, tools, specific_question)