import os
import tempfile
import pandas as pd
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)

from sentence_transformers import SentenceTransformer, util

# Load embeddings
embeddings = OpenAIEmbeddings()

# Load multilingual reranker model (supports Burmese + English)
reranker_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def get_vector_db_retriever(csv_path="data/microbiology.csv", top_k=10):
    """
    Returns a LangChain retriever for a Q&A CSV dataset.
    Each row in CSV should have 'Instruction' and 'Output' columns.
    """
    persist_path = os.path.join(tempfile.gettempdir(), "qa_vectorstore.parquet")

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


def rerank(query: str, documents: list[Document], top_n=4):
    """Re-rank retrieved docs with multilingual model (Burmese + English)."""
    if not documents:
        return []
    
    query_emb = reranker_model.encode(query, convert_to_tensor=True)
    doc_embs = reranker_model.encode([d.page_content for d in documents], convert_to_tensor=True)
    scores = util.cos_sim(query_emb, doc_embs)[0]

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]


if __name__ == "__main__":
    retriever = get_vector_db_retriever(csv_path="data/microbiology.csv", top_k=4)

    query = "အဏုဇီဝဗေဒ ဆိုတာ ဘာလဲ ရှင်းပြပါ"
    docs = retriever.get_relevant_documents(query)
    rerank_docs = rerank(query, docs, top_n=4)
    for doc in rerank_docs:
        print("--- Retrieved Document ---")
        print(doc.page_content)
