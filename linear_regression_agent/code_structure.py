from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain.agents.agent import AgentExecutor
from langchain.agents import Tool, ZeroShotAgent, LLMSingleActionAgent, AgentOutputParser
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

from langchain.prompts import StringPromptTemplate
from langchain import OpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from langchain.tools import BaseTool

import pandas as pd
from typing import TYPE_CHECKING
import tiktoken
import math

from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks,
)

from langchain.schema import (
    BaseLLMOutputParser,
    BasePromptTemplate,
    LLMResult,
    PromptValue,
)

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts.base import StringPromptValue

from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

from app.model.agent_df_processed_temp import *

model = 'gpt-4'        
llm = ChatOpenAI(temperature=0, model=model)


# Run code in python and pass local files as arguments    
tools = [PythonAstREPLTool(locals={"df": df)]
tool_names = [tool.name for tool in tools]


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format(self, **kwargs) -> str:
        # enc = tiktoken.get_encoding("cl100k_base")
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


partial_prompt = prompt.partial(df=str(df.head())))


# Custom LLM Chain
class CustomLLMChain(LLMChain):  
    summary_model: BaseLLM = None
    
    def __init__(self, summary_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary_model = summary_model
    
    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        if len(prompts)==0:
            string_tmpr = prompts[0].to_string()
            tokens = ENCODING.encode(string_tmpr)
            if len(tokens)>LIMIT_TOKENS:
                prompt = self.summarize(string_tmpr)
                prompt += TEMPLATE + prompt
                #print("Summarized prompt", prompt)
                prompts = [StringPromptValue(text=prompt)]
                
        
        return self.llm.generate_prompt(
            prompts,
            stop,
            callbacks=run_manager.get_child() if run_manager else None,
            **self.llm_kwargs,
        )
    
    def summarize(self, input: str) -> str:
        SUMMARY_SYS_MSG = """
        You are SummaryGPT, a model designed to ingest content and summarize it concisely and accurately
        You will receive an input string, and your response will be a summary of this information for futher next steps. Lets think step by step
        """

        """Generate LLM result summary from inputs too meet token limits."""
        system_message = SystemMessagePromptTemplate.from_template(
            template=SUMMARY_SYS_MSG
        )
        human_message = HumanMessagePromptTemplate.from_template(
            template="Input: {input}"
        )

        chunks = chunk(chunk_str=input)

        summary = ""

        for i in chunks:
            prompt = ChatPromptTemplate(
                input_variables=["input"],
                messages=[system_message, human_message],
            )

            _input = prompt.format_prompt(input=i)
            output = self.summary_model(_input.to_messages())
            summary += f"\n{output.content}"

        sum_tokens = token_len(input=summary)

        if sum_tokens > LIMIT_TOKENS:
            return summarize(input=summary)

        return summary



llm_chain = CustomLLMChain(
        llm=llm,
        prompt=partial_prompt,
        callback_manager = callback_manager,
        summary_model = llm,
    )