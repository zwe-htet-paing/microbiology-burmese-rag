import os
import tempfile
import pandas as pd
from typing import List, Dict, Any
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import re
from openai import OpenAI
from guardrails import Guard

load_dotenv(dotenv_path=".env", override=True)
openai_client = OpenAI()

# Embeddings and reranker models
embeddings = OpenAIEmbeddings()
reranker_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
content_moderation_model = "omni-moderation-latest"

# ----------------------
# Vector DB Retriever
# ----------------------
def get_vector_db_retriever(csv_path: str = "data/microbiology.csv", top_k: int = 10) -> SKLearnVectorStore:
    """
    - Returns a LangChain retriever for a Q&A CSV dataset.
    - Each row in CSV should have 'Instruction' and 'Output' columns.
    """
    persist_path = os.path.join(tempfile.gettempdir(), "qa_vectorstore.parquet")

    if os.path.exists(persist_path):
        vectorstore = SKLearnVectorStore(
            embedding=embeddings,
            persist_path=persist_path,
            serializer="parquet"
        )
        return vectorstore.as_retriever(search_kwargs={"k": top_k})

    df = pd.read_csv(csv_path)
    df = df.head(10)
    docs: List[Document] = [
        Document(
            page_content=f"Question: {row['Instruction']}\nAnswer: {row['Output']}",
            metadata={"source": "csv"}
        )
        for _, row in df.iterrows()
    ]

    # Keep QA as single chunk
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits: List[Document] = docs  # No splitting applied

    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
        persist_path=persist_path,
        serializer="parquet"
    )
    vectorstore.persist()

    return vectorstore.as_retriever(search_kwargs={"k": top_k})

# ----------------------
# Re-ranker
# ----------------------
def rerank(query: str, documents: List[Document], top_n: int = 4) -> List[Document]:
    """
    - Re-rank retrieved docs with multilingual model (Burmese + English).
    """
    if not documents:
        return []

    query_emb = reranker_model.encode(query, convert_to_tensor=True)
    doc_embs = reranker_model.encode([d.page_content for d in documents], convert_to_tensor=True)
    scores = util.cos_sim(query_emb, doc_embs)[0]

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]

# ----------------------
# Content Moderation
# ----------------------
def moderate_content(text: str) -> bool:
    """
    - Return True if text is safe, False if flagged.
    """
    resp = openai_client.moderations.create(
        model=content_moderation_model,
        input=text
    )
    return resp.results[0].flagged is False

# ----------------------
# Burmese Dominance Check
# ----------------------
def is_burmese_dominant(text: str, threshold: float = 0.5) -> bool:
    """
    - Check if text contains mostly Burmese characters (allow English terms).
    """
    burmese_chars = re.findall(r"[\u1000-\u109F]", text)
    total_chars = len(text)
    if total_chars == 0:
        return False
    ratio = len(burmese_chars) / total_chars
    return ratio >= threshold

# ----------------------
# Schema Validation with Guardrails
# ----------------------
rail_yaml: str = """
<rail version="0.1">
<output>
    <object>
        <string name="answer" description="The assistant’s answer (Burmese-dominant, English technical terms allowed)" />
        <string name="source" description="The retrieved source doc." />
    </object>
</output>
</rail>
"""
guard: Guard = Guard.from_rail_string(rail_yaml)

def validate_schema(llm_output: str) -> Dict[str, Any]:
    """
    - Validate LLM output against schema and return validated output.
    """
    try:
        input_dict = {"answer": llm_output, "source": "N/A"}
        validated_output, _ = guard(input_dict, llm_api="openai/gpt-4o-mini")
        return validated_output
    except Exception as e:
        return {"answer": llm_output, "source": "N/A", "error": str(e)}

# ----------------------
# Combined Guardrails
# ----------------------
def apply_guardrails(user_query: str, llm_response: str) -> str:
    """
    - Apply guardrails to user query and LLM response.
    """
    if not moderate_content(user_query):
        return "❌ ဤမေးခွန်းကို ခွင့်မပြုပါ။"

    if not moderate_content(llm_response):
        return "❌ ဖန်တီးထားသောတုံ့ပြန်ချက် လုံခြုံမှုမရှိပါ။"

    if not is_burmese_dominant(llm_response):
        return "❌ တုံ့ပြန်ချက်သည် အများအားဖြင့် မြန်မာဘာသာဖြစ်ရမည် (အင်္ဂလိပ်နည်းနည်း စာလုံးများ ခွင့်ပြုသည်။)"

    validated = validate_schema(llm_response)
    return validated["answer"]


if __name__ == "__main__":
    retriever = get_vector_db_retriever(csv_path="data/microbiology.csv", top_k=4)

    query = "အဏုဇီဝဗေဒ ဆိုတာ ဘာလဲ ရှင်းပြပါ"
    docs = retriever.get_relevant_documents(query)
    rerank_docs = rerank(query, docs, top_n=4)
    for doc in rerank_docs:
        print("--- Retrieved Document ---")
        print(doc.page_content)
