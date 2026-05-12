from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (JsonOutputToolsParser,
                                                        PydanticToolsParser)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", base_url="https://openrouter.ai/api/v1")
# An output parser that returns the tools called by the llm in form of dictionary
tool_parser = JsonOutputToolsParser(return_id=True)
# A parser that parses the llm output to follow the AnswerQuestion format
pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are an expert researcher.
    Current time: {time}

    1. {first_instruction}
    2. Reflect and critique your answer. Be severe to maximize improvements.
    3. Recommend search queries to research information and improve your answer.
    """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.now().isoformat())


revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

# first_responder_prompt_template = actor_prompt_template.partial(
#     first_instruction="Provide a detailed ~250 word answer."
# )

first_responder = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
) | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


# if __name__ == "__main__":
#     chain = (
#         first_responder_prompt_template
#         | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
#         | pydantic_parser
#     )

#     res = chain.invoke(input={"messages": [HumanMessage(content="Write about Ai SOC/autonomous soc problem domain")]})
#     print(res)
