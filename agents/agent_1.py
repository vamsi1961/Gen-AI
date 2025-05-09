from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.tools import Tool
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools import TavilySearchResults
import re
from typing import List
from dotenv import load_dotenv
import os

# Set environment variables
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
"""

# Define the prompt template for JSON extraction
prompt = PromptTemplate.from_template("""
You are a smart JSON data extractor.

JSON:
{json_data}

Question:
{query}

Answer in concise JSON or text.
""")

# Initialize the Azure OpenAI LLM
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

# Create a function that queries the JSON based on natural language
def query_json_tool(query):
    return llm.invoke(prompt.format(json_data=raw_json_string, query=query))

# Create a function that returns the raw JSON (FIX: now accepts an unused parameter)
def get_raw_json(unused_input=""):
    """Returns the raw JSON data regardless of input."""
    return raw_json_string

# Create the tools
json_query_tool = Tool(
    name="JSONQueryTool",
    func=query_json_tool,
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

# Add your tools to a list
tools = [raw_json_tool, json_query_tool, tavily_tool, get_length]

def render_text_description(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

def format_log_to_str(intermediate_steps):
    log_str = ""
    for action, observation in intermediate_steps:
        log_str += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\n"
    return log_str

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
        try:
            agent_step = agent.invoke(
                {
                    "input": input_text,
                    "agent_scratchpad": format_log_to_str(intermediate_steps),
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
                
        except Exception as e:
            print(f"Error in iteration {i+1}: {str(e)}")
            # Add debugging information
            print("Trying to recover and continue...")
            
            if len(intermediate_steps) > 0:
                last_observation = intermediate_steps[-1][1]
                # Create a manual thought to help guide the LLM back on track
                manual_thought = (
                    AgentAction(
                        tool="Thought",
                        tool_input="Based on the JSON data I retrieved, I should now provide a final answer.",
                        log="Thought: Based on the JSON data I retrieved, I should now provide a final answer."
                    ),
                    "Based on the observation, I can now answer the question."
                )
                intermediate_steps.append(manual_thought)
                print("Added recovery step to get back on track")
                continue
            else:
                print("Cannot recover, exiting...")
                break
        
    print("Reached maximum iterations without finishing")
    return None

template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    To answer questions about JSON data, you should use the GetRawJSON tool first to see what JSON data is available.
    
    Use the following format EXACTLY:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    After each observation, you MUST respond with a Thought followed by either an Action or Final Answer.
    NEVER respond with just a summary or explanation after an observation.
    ALWAYS follow the format above, with Thought/Action/Action Input or Thought/Final Answer.
    
    Begin!
    
    Question: {input}
    {agent_scratchpad}
    """

prompt = PromptTemplate.from_template(template).partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# Custom output parser with more robust error handling
class CustomReActOutputParser(ReActSingleInputOutputParser):
    def parse(self, text):
        try:
            return super().parse(text)
        except Exception as e:
            print(f"Parser error: {str(e)}")
            print(f"Raw text causing the error: {text}")
            
            # Check if this looks like a final answer attempt
            if "The JSON input" in text or "JSON data" in text:
                print("Detected attempt at final answer, converting format...")
                # Convert to proper format
                reformatted = f"Thought: I now know the final answer\nFinal Answer: {text}"
                return super().parse(reformatted)
            
            # If no specific pattern recognized, raise the original error
            raise e

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: x["agent_scratchpad"],
    }
    | prompt
    | llm
    | CustomReActOutputParser()
)

# Main execution
if __name__ == "__main__":
    # Print the tools to verify they're correctly defined
    print("Available tools:")
    print(render_text_description(tools))
    
    # Test question 1
    print("\n=== Testing 'what is the json input' ===")
    complex_question = "what is the json input"
    final_result = run_agent_with_steps(agent, tools, complex_question)
    
    # Test question 2
    print("\n=== Testing 'what is John's age' ===")
    specific_question = "what is John's age"
    final_result2 = run_agent_with_steps(agent, tools, specific_question)