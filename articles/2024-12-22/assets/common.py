from langchain_community.chat_models import ChatLiteLLM
from langchain.output_parsers import YamlOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_debug
from pydantic import BaseModel, Field
from pprint import pprint
from typing import List
from datetime import datetime

def llm_unstructured_query(
    query_text, model="ollama/llama3.1:8b", temperature=0, max_tokens=5000, debug=False
):
    
    with open("log.txt","a") as f:
        f.write(f"{datetime.now()}-----{query_text}----{model}----{temperature}\n")
    
    chat = ChatLiteLLM(
        model=model, temperature=temperature, max_tokens=max_tokens, cache=False
    )

    messages = [HumanMessage(content=query_text)]
    
    content = chat.invoke(messages).content
    
    with open("log.txt","a") as f:
        f.write(f"{content}\nEND")
    
    return content


# Define the function
def llm_structured_query(
    query_text,
    response_class,
    model="ollama/llama3.1:8b",
    temperature=0.3,
    max_tokens=5000,
    debug=False,
):

    if debug:
        set_debug(True)

    model = ChatLiteLLM(
        model=model, temperature=temperature, max_tokens=max_tokens, cache=False
    )

    parser = JsonOutputParser(pydantic_object=response_class)

    prompt_template = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt_template | model | parser

    return chain.invoke({"query": query_text})


if __name__ == "__main__":

    class QuestionAnswer(BaseModel):
        question: str = Field(description="A question")
        answer: str = Field(description="The answer to the question")
        explanation: str = Field(description="Explanation of why this is the answer")

    class QuestionAnswerList(BaseModel):
        question_answer_list: List[QuestionAnswer] = Field(
            description="List of question,answer,explanation"
        )

    result = llm_structured_query(
        "10 questions and answers with explanations.", QuestionAnswerList
    )
    pprint(result)
    result = llm_unstructured_query("10 questions and answers with explanations.")
    pprint(result)
