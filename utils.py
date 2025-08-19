import os
import tempfile
import pandas as pd
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)

def get_vector_db_retriever(csv_path="data/microbiology.csv", top_k=4):
    """
    Returns a LangChain retriever for a Q&A CSV dataset.
    Each row in CSV should have 'Instruction' and 'Output' columns.
    """
    persist_path = os.path.join(tempfile.gettempdir(), "qa_vectorstore.parquet")
    embeddings = OpenAIEmbeddings()

    # Load existing vector store if available
    if os.path.exists(persist_path):
        vectorstore = SKLearnVectorStore(
            embedding=embeddings,
            persist_path=persist_path,
            serializer="parquet"
        )
        return vectorstore.as_retriever(search_kwargs={"k": top_k})

    # Load CSV
    df = pd.read_csv(csv_path)
    df = df.head(10)
    docs = [
        Document(
            page_content=f"Question: {row['Instruction']}\nAnswer: {row['Output']}",
            metadata={"source": "csv"}
        )
        for _, row in df.iterrows()
    ]

    # Split each QA into a single chunk (preserve QA together)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    # doc_splits = text_splitter.split_documents(docs)
    doc_splits = docs

    # Create vector store
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
        persist_path=persist_path,
        serializer="parquet"
    )
    vectorstore.persist()

    return vectorstore.as_retriever(search_kwargs={"k": top_k})

if __name__ == "__main__":
    retriever = get_vector_db_retriever(csv_path="data/microbiology.csv", top_k=4)

    query = "အဏုဇီဝဗေဒ ဆိုတာ ဘာလဲ ရှင်းပြပါ"
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        print("--- Retrieved Document ---")
        print(doc.page_content)
