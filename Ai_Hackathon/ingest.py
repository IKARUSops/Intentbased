import os
import sys
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200,add_start_index=True, 
)
doc_splits = text_splitter.split_documents(docs_list)
documents=doc_splits

embeddings = OllamaEmbeddings(model="all-minilm")


db = FAISS.from_documents(documents, embeddings)

retriever = db.as_retriever()

def main():
    urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200,add_start_index=True, 
    )
    doc_splits = text_splitter.split_documents(docs_list)
    documents=doc_splits

    embeddings = OllamaEmbeddings(model="all-minilm")


    db = FAISS.from_documents(documents, embeddings)

    retriever = db.as_retriever()

if __name__ == "__main__":
    main()