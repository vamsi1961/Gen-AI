import os


from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Union
from typing import Literal
from langgraph.graph import END
from langgraph.graph import StateGraph, START
from IPython.display import Image, display
import asyncio
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
gpt4_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

os.environ["LANGCHAIN_TRACING_V2"] = "false"    # Your code that triggers the error, e.g., a LangChain run
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"



load_dotenv()
llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        deployment_name=gpt4_deployment_name,
        api_version="2024-08-01-preview",
        temperature=0
    )

tools = [TavilySearchResults(max_results=3)]

# Choose the LLM that will drive the agent
prompt = "You are a helpful assistant."
agent_executor = create_react_agent(llm, tools, prompt=prompt)


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan.For the given objective, You have to give plan to write a code to wroter agent so give simple steps that agent should understnad what to do 
      come up with a simple step by step plan. 
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this: {input}

Your original plan was this: {plan}

You have currently done the follow steps: {past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


replanner = replanner_prompt | llm.with_structured_output(Act)

async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan: {plan_str}\n\n You are tasked with executing step {1}, {task}."""
    
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )

    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }

async def plan_step(state: PlanExecute):

    plan = await planner.ainvoke({"messages": [("user", state["input"])]})

    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"

workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("agent", execute_step)

# Add a replan node
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")

# From plan we go to agent
workflow.add_edge("planner", "agent")

# From agent, we replan

workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["agent", END],
)

app = workflow.compile()

display(Image(app.get_graph(xray=True).draw_mermaid_png()))

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For given ibjective you have to plan step by step. make sure you give clear steps, 
            you steps are taken by writer tool it writes the code and then evaluationtool evaluates it if is evaluated properly then you have to go to next step
            write concise steps dont write unnecessary matter just what to do
            """,
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | llm.with_structured_output(Plan)
planner.invoke(
    {
        "messages": [ ("user", "what is the hometown of the current Australia open winner?") ]
    }
)


config = {"recursion_limit": 50}
inputs = {"input": "what is the hometown of the current Australia open winner?"}

async def main():
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print("........")
                print(v)

asyncio.run(main())


