from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv


# class CityLocation(BaseModel):
#     city: str
#     country: str

# load_dotenv()
# ollama_model = OpenAIModel(
#     model_name='llama3.2', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
# )
# agent = Agent(ollama_model, output_type=CityLocation)

# result = agent.run_sync('Where were the olympics held in 2012?')
# print(result.output)
# #> city='London' country='United Kingdom'
# print(result.usage())
# """
# Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
# """


# Retrieval Grader
local_llm = "gemma3:1b"

ollama_model = OpenAIModel(
    model_name=local_llm, provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from ingest import retriever

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)


prompt = PromptTemplate(
    template="""
You are a grader checking if a retrieved document is relevant to a user's question.

A document is relevant if it contains keywords or information clearly related to the question.
This is not a strict test—your goal is to filter out obviously unrelated results.

Return a binary relevance score:
- Use "yes" if the document is relevant.
- Use "no" if the document is not relevant.

Output the result as a JSON object with a single key "score". Do not include any explanation or additional text.

Document:
{document}

Question:
{question}

Your answer (JSON only):
""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()
question = "what is an agent?"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))