from langsmith import traceable
from openai import OpenAI
from typing import List
import nest_asyncio

from utils import get_vector_db_retriever, rerank, apply_guardrails

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)

import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "gpt-4o-mini"
MODEL_PROVIDER = "openai"
APP_VERSION = 1.0
RAG_SYSTEM_PROMPT = """You are an expert Burmese-language assistant specializing in question-answering tasks.

## CORE INSTRUCTIONS:
- Analyze the retrieved context carefully to provide accurate answers
- Accept questions in both English and Burmese languages
- **ALWAYS respond exclusively in Burmese** regardless of the question language
- Maintain conversation context and refer to previous exchanges when relevant
- Handle follow-up questions by linking them to the original question and previous answers, unless new context explicitly overrides

## RESPONSE GUIDELINES:
1. **Answer Structure**: Provide direct, factual responses based solely on the retrieved context
2. **Length**: Keep responses concise (maximum 3 sentences)
3. **Accuracy**: Only answer what you can confidently derive from the provided context
4. **Unknown Information**: If the context doesn't contain sufficient information, clearly state in Burmese:  
   **"ပေးထားသော အချက်အလက်များတွင် ဤမေးခွန်းအတွက် လုံလောက်သော အဖြေမရှိပါ။"**

## CONTEXT USAGE:
- Prioritize the most recent and relevant context pieces
- Synthesize information from multiple context sources when applicable
- Cite specific details from context when providing factual claims
- When the user asks a follow-up question, use both the **original question** and the **previous answer** to interpret their intent
- Do not add information not present in the retrieved context

## RESPONSE QUALITY:
- Use natural, fluent Burmese language
- Adapt tone to match the question's formality level
- Provide helpful clarifications when the context allows
- Structure complex answers with clear logical flow

## FALLBACK BEHAVIOR:
If no relevant context is provided or the context is insufficient:  
**"ပေးထားသော အချက်အလက်များကို အခြေခံ၍ ဤမေးခွန်းကို မဖြေနိုင်ပါ။ ပိုမိုတိကျသော အချက်အလက်များ လိုအပ်ပါသည်။"**

Remember: Your primary goal is to be a reliable, accurate, and helpful Burmese-language assistant that provides contextually grounded responses.
"""

openai_client = OpenAI()

nest_asyncio.apply()
retriever = get_vector_db_retriever()


@traceable(run_type="chain")
def retrieve_documents(question: str):
    """ 
    - Returns documents fetched from a vectorstore based on the user's question 
    """
    return retriever.invoke(question)

@traceable(run_type="chain")
def rerank_documents(question: str, documents: list):
    """
    - Return re-ranked documents based on the user's question
    """
    docs = rerank(question, documents)
    return docs

@traceable(run_type="chain")
def generate_response(question: str, documents):
    """ 
    - Calls `call_openai` to generate a model response after formatting inputs 
    """
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    messages = [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Context: {formatted_docs} \n\n Question: {question}"
        }
    ]
    return call_openai(messages)


@traceable(
    run_type="llm",
    metadata={
        "ls_provider": MODEL_PROVIDER,
        "ls_model_name": MODEL_NAME
    }
)
def call_openai(messages: List[dict]) -> str:
    """ 
    - Returns the chat completion output from OpenAI 
    """
    return openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )

@traceable(run_type="chain")
def langsmith_rag(question: str):
    """
    - Calls `retrieve_documents` to fetch documents
    - Calls `generate_response` to generate a response based on the fetched documents
    - Returns the model response
    """
    documents = retrieve_documents(question)
    documents = rerank_documents(question, documents)
    response = generate_response(question, documents)
    
    answer = apply_guardrails(question, response.choices[0].message.content)
    return answer


if __name__ == "__main__":
    question = "အဏုဇီဝဗေဒ ဆိုတာ ဘာလဲ ရှင်းပြပါ"
    answer = langsmith_rag(question)
    print(answer)